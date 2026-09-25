"""Data streams for the teacher--student protein_base training lineage.

The general-protein corpus and the antibody-antigen corpus use different
sample schemas, so a batch is deliberately homogeneous.  The sampler controls
the stream and, for the antibody stream, emits VH-VL/VHH batches in the same
3:1 ratio registered by prod.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from proteinfoundation.datasets.ab_data import (
    AntibodyDesignDataset,
    collate_fn as antibody_collate_fn,
)
from proteinfoundation.datasets.protein_data import (
    ProteinPairDataset,
    collate_fn as general_collate_fn,
)


class ProteinBaseDataset(Dataset):
    """Index-space wrapper for general, VH-VL, and VHH samples."""

    VALID_STREAMS = {"general", "abnb", "joint"}

    def __init__(
        self,
        general_root: str | Path,
        antibody_root: str | Path,
        stream: str = "general",
        max_target_len: int = 450,
        max_partner_len: int = 500,
        max_antibody_len: int = 450,
        max_antigen_len: int = 500,
        mask_strategy: str = "mixed",
        mask_strategy_probs: Sequence[float] = (0.5, 0.3, 0.2),
        general_mask_strategy: str = "none_single_region",
        general_mask_probs: Sequence[float] = (0.2, 0.3, 0.5),
        swap_orientation: bool = True,
    ):
        if stream not in self.VALID_STREAMS:
            raise ValueError(f"stream must be one of {sorted(self.VALID_STREAMS)}")
        self.stream = stream
        self.general = None
        self.antibody = None
        self.nanobody = None

        if stream in {"general", "joint"}:
            self.general = ProteinPairDataset(
                general_root,
                split="train",
                max_target_len=max_target_len,
                max_partner_len=max_partner_len,
                mask_strategy=general_mask_strategy,
                mask_strategy_probs=list(mask_strategy_probs),
                sequence_mask_probs=list(general_mask_probs),
                swap_orientation=swap_orientation,
            )
        if stream in {"abnb", "joint"}:
            self.antibody = AntibodyDesignDataset(
                str(antibody_root),
                split="train",
                max_antibody_len=max_antibody_len,
                max_antigen_len=max_antigen_len,
                mask_strategy=mask_strategy,
                mask_strategy_probs=list(mask_strategy_probs),
                conventional_only=True,
            )
            self.nanobody = AntibodyDesignDataset(
                str(antibody_root),
                split="train",
                max_antibody_len=max_antibody_len,
                max_antigen_len=max_antigen_len,
                mask_strategy=mask_strategy,
                mask_strategy_probs=list(mask_strategy_probs),
                vhh_only=True,
            )

        self.offsets: Dict[str, int] = {}
        cursor = 0
        for name, dataset in (
            ("general", self.general),
            ("antibody", self.antibody),
            ("nanobody", self.nanobody),
        ):
            if dataset is not None:
                self.offsets[name] = cursor
                cursor += len(dataset)
        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> Tuple[str, Dataset, int]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ordered = (
            ("general", self.general),
            ("antibody", self.antibody),
            ("nanobody", self.nanobody),
        )
        for name, dataset in ordered:
            if dataset is None:
                continue
            start = self.offsets[name]
            if start <= index < start + len(dataset):
                return name, dataset, index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> Dict:
        name, dataset, local_index = self._locate(index)
        item = dict(dataset[local_index])
        item["task_type"] = {
            "general": "general_protein",
            "antibody": "antibody",
            "nanobody": "nanobody",
        }[name]
        return item


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate one homogeneous stream and retain its task identity."""
    task_types = {item.get("task_type") for item in batch}
    if len(task_types) != 1:
        raise ValueError(f"mixed task batch is not allowed: {sorted(task_types)}")
    task_type = next(iter(task_types))
    clean_batch = [
        {key: value for key, value in item.items() if key != "task_type"}
        for item in batch
    ]
    if task_type == "general_protein":
        output = general_collate_fn(clean_batch)
    elif task_type in {"antibody", "nanobody"}:
        output = antibody_collate_fn(clean_batch)
    else:
        raise ValueError(f"unknown task type: {task_type}")
    output["task_type"] = task_type
    return output


class ProteinBaseBatchSampler(Sampler[List[int]]):
    """Deterministic stream sampler with an explicit AB/NB ratio.

    ``stream=abnb`` uses the prod slot pattern ``AB, AB, AB, NB`` by default.
    ``stream=general`` emits only the 50k general-protein corpus.  ``joint``
    additionally supports a configurable general:AB:NB slot ratio; its
    default ``1:3:1`` keeps the antibody substream at AB:NB=3:1.
    """

    def __init__(
        self,
        dataset: ProteinBaseDataset,
        batch_size: int,
        steps: int,
        stream: str,
        abnb_ratio: Tuple[int, int] = (3, 1),
        joint_ratio: Tuple[int, int, int] = (1, 3, 1),
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        if batch_size < 1 or steps < 1:
            raise ValueError("batch_size and steps must be positive")
        if stream != dataset.stream:
            raise ValueError(f"sampler stream {stream!r} != dataset stream {dataset.stream!r}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.stream = stream
        self.abnb_ratio = tuple(int(x) for x in abnb_ratio)
        self.joint_ratio = tuple(int(x) for x in joint_ratio)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.rank < 0 or self.world_size < 1 or self.rank >= self.world_size:
            raise ValueError(f"invalid distributed sampler rank/world_size: {rank}/{world_size}")
        if len(self.abnb_ratio) != 2 or min(self.abnb_ratio) <= 0:
            raise ValueError(f"invalid abnb_ratio: {abnb_ratio}")
        if len(self.joint_ratio) != 3 or min(self.joint_ratio) <= 0:
            raise ValueError(f"invalid joint_ratio: {joint_ratio}")

        if stream == "general":
            self.slots = ["general"]
        elif stream == "abnb":
            self.slots = ["antibody"] * self.abnb_ratio[0] + [
                "nanobody"
            ] * self.abnb_ratio[1]
        elif stream == "joint":
            self.slots = (
                ["general"] * self.joint_ratio[0]
                + ["antibody"] * self.joint_ratio[1]
                + ["nanobody"] * self.joint_ratio[2]
            )
        else:
            raise ValueError(stream)

        for task in set(self.slots):
            if task not in dataset.offsets:
                raise ValueError(f"stream {stream} requires unavailable task {task}")

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterable[List[int]]:
        # Every rank keeps the same task slot pattern (so local batches stay
        # homogeneous) but shuffles its own pool independently.  This gives
        # DDP distinct samples without changing the registered AB:NB ratio.
        rng = random.Random(self.seed + 100003 * self.rank)
        pools: Dict[str, List[int]] = {}
        cursors: Dict[str, int] = {}
        for task in set(self.slots):
            start = self.dataset.offsets[task]
            pools[task] = list(range(start, start + len(getattr(self.dataset, task))))
            rng.shuffle(pools[task])
            cursors[task] = 0

        for step in range(self.steps):
            task = self.slots[step % len(self.slots)]
            pool = pools[task]
            cursor = cursors[task]
            indices: List[int] = []
            while len(indices) < self.batch_size:
                if cursor >= len(pool):
                    rng.shuffle(pool)
                    cursor = 0
                take = min(self.batch_size - len(indices), len(pool) - cursor)
                indices.extend(pool[cursor : cursor + take])
                cursor += take
            cursors[task] = cursor
            yield indices


__all__ = [
    "ProteinBaseDataset",
    "ProteinBaseBatchSampler",
    "collate_fn",
]
