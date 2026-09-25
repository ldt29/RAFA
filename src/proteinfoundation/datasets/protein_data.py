"""Runtime loader for the general-protein interacting-pair corpus.

The representation intentionally mirrors ``ab_data.collate_fn``.  Chain type
1 is the randomly selected generated/target chain and chain type 3 is the
fixed interacting partner.  There are no CDRs: ``cdr_mask`` is repurposed as
the runtime *sequence* mask (the randomly hidden span or scattered residues),
while the target-chain ``mask``/``mask_dict.coords`` is the structure mask
that the flow matcher corrupts.  ``native_cdr_mask`` records the full target
chain for diversity/readout code that expects that field.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ProteinPairDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "train",
        max_target_len: int = 450,
        max_partner_len: int = 500,
        mask_strategy: str = "mixed",
        mask_strategy_probs: Optional[List[float]] = None,
        sequence_mask_probs: Optional[List[float]] = None,
        swap_orientation: bool = True,
    ):
        self.data_dir = Path(
            data_dir
            if data_dir is not None
            else os.environ.get("PROTEIN_DATASET", "./data/structure_dataset/protein")
        )
        self.split = split
        self.max_target_len = int(max_target_len)
        self.max_partner_len = int(max_partner_len)
        self.mask_strategy = mask_strategy
        self.mask_strategy_probs = list(mask_strategy_probs or [0.5, 0.3, 0.2])
        if self.mask_strategy not in {
            "full",
            "partial",
            "none",
            "mixed",
            "none_single_region",
        }:
            raise ValueError(
                "mask_strategy must be full, partial, none, mixed, or none_single_region"
            )
        if len(self.mask_strategy_probs) != 3 or not np.isclose(
            sum(self.mask_strategy_probs), 1.0, atol=1e-6
        ):
            raise ValueError("mask_strategy_probs must have three entries summing to one")
        self.sequence_mask_probs = list(sequence_mask_probs or [0.2, 0.3, 0.5])
        if len(self.sequence_mask_probs) != 3 or not np.isclose(
            sum(self.sequence_mask_probs), 1.0, atol=1e-6
        ):
            raise ValueError("sequence_mask_probs must have three entries summing to one")
        self.swap_orientation = bool(swap_orientation)
        processed = self.data_dir / "processed" / f"{split}_processed"
        metainfo_path = processed / "_metainfo"
        if not metainfo_path.exists():
            raise FileNotFoundError(f"processed protein data not found: {metainfo_path}")
        meta = json.loads(metainfo_path.read_text())
        self.data: list[dict] = []
        for filename in meta["file_names"]:
            shard = Path(filename)
            if not shard.exists():
                shard = processed / shard.name
            with shard.open("rb") as handle:
                self.data.extend(pickle.load(handle))
        if not self.data:
            raise RuntimeError(f"empty protein split: {processed}")

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def _partial_mask(native: np.ndarray) -> np.ndarray:
        indices = np.flatnonzero(native)
        if len(indices) == 0:
            return native.copy()
        n_mask = max(1, int(len(indices) * np.random.uniform(0.1, 0.9)))
        out = np.zeros_like(native, dtype=bool)
        if np.random.random() < 0.5:
            start = np.random.randint(0, len(indices) - n_mask + 1)
            out[indices[start : start + n_mask]] = True
        else:
            out[np.random.choice(indices, size=n_mask, replace=False)] = True
        return out

    def _mask(self, native: np.ndarray) -> np.ndarray:
        strategy = self.mask_strategy
        if strategy == "none_single_region":
            choice = int(np.random.choice(3, p=self.sequence_mask_probs))
            if choice == 0:  # no sequence mask
                return np.zeros_like(native, dtype=bool)
            indices = np.flatnonzero(native)
            if len(indices) == 0:
                return native.copy()
            if choice == 1:  # one randomly selected residue
                out = np.zeros_like(native, dtype=bool)
                out[np.random.choice(indices)] = True
                return out
            # One contiguous random region.  Its length is sampled rather
            # than fixed so the model sees both short and long spans.
            n_mask = max(1, int(len(indices) * np.random.uniform(0.1, 0.9)))
            start = np.random.randint(0, len(indices) - n_mask + 1)
            out = np.zeros_like(native, dtype=bool)
            out[indices[start : start + n_mask]] = True
            return out
        if strategy == "mixed":
            choice = int(np.random.choice(3, p=self.mask_strategy_probs))
            strategy = ("full", "partial", "none")[choice]
        if strategy == "full":
            return native.copy()
        if strategy == "none":
            return np.zeros_like(native, dtype=bool)
        return self._partial_mask(native)

    def __getitem__(self, index: int) -> Dict:
        record = self.data[index % len(self.data)]
        first, second = "target", "partner"
        swapped = self.swap_orientation and bool(np.random.randint(0, 2))
        if swapped:
            first, second = second, first

        target_coords = record[f"{first}_coords"][: self.max_target_len]
        target_coord_mask = record[f"{first}_coord_mask"][: self.max_target_len]
        target_seq = record[f"{first}_seq"][: self.max_target_len]
        target_interface = record[f"{first}_interface"][: self.max_target_len]
        partner_coords = record[f"{second}_coords"][: self.max_partner_len]
        partner_coord_mask = record[f"{second}_coord_mask"][: self.max_partner_len]
        partner_seq = record[f"{second}_seq"][: self.max_partner_len]
        partner_interface = record[f"{second}_interface"][: self.max_partner_len]
        native_target = np.ones(len(target_seq), dtype=bool)
        target_mask = self._mask(native_target)

        coords = np.concatenate([target_coords, partner_coords], axis=0)
        coord_mask = np.concatenate([target_coord_mask, partner_coord_mask], axis=0)
        seq = np.concatenate([target_seq, partner_seq], axis=0)
        cdr_mask = np.concatenate([target_mask, np.zeros(len(partner_seq), dtype=bool)])
        native_cdr_mask = np.concatenate([native_target, np.zeros(len(partner_seq), dtype=bool)])
        paratope = np.concatenate([target_interface, np.zeros(len(partner_seq), dtype=bool)])
        epitope = np.concatenate([np.zeros(len(target_seq), dtype=bool), partner_interface])
        chain_type = np.concatenate([
            np.ones(len(target_seq), dtype=np.int64),
            np.full(len(partner_seq), 3, dtype=np.int64),
        ])
        return {
            "sample_id": record["sample_id"],
            "orientation_swapped": swapped,
            "coords": torch.from_numpy(coords),
            "coord_mask": torch.from_numpy(coord_mask),
            "seq": torch.from_numpy(seq),
            "cdr_mask": torch.from_numpy(cdr_mask),
            "native_cdr_mask": torch.from_numpy(native_cdr_mask),
            "paratope_mask": torch.from_numpy(paratope),
            "epitope_mask": torch.from_numpy(epitope),
            "chain_type": torch.from_numpy(chain_type),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    max_len = max(item["seq"].shape[0] for item in batch)
    size = len(batch)
    coords = torch.zeros(size, max_len, 37, 3)
    coord_mask = torch.zeros(size, max_len, 37, dtype=torch.bool)
    seq = torch.zeros(size, max_len, dtype=torch.long)
    full_mask = torch.zeros(size, max_len, dtype=torch.bool)
    target_mask = torch.zeros(size, max_len, dtype=torch.bool)
    cdr_mask = torch.zeros(size, max_len, dtype=torch.bool)
    native_cdr_mask = torch.zeros(size, max_len, dtype=torch.bool)
    paratope_mask = torch.zeros(size, max_len, dtype=torch.bool)
    epitope_mask = torch.zeros(size, max_len, dtype=torch.bool)
    chain_type = torch.zeros(size, max_len, dtype=torch.long)
    sample_ids = []
    orientation_swapped = []
    for i, item in enumerate(batch):
        n = item["seq"].shape[0]
        coords[i, :n] = item["coords"]
        coord_mask[i, :n] = item["coord_mask"]
        seq[i, :n] = item["seq"]
        full_mask[i, :n] = True
        chain_type[i, :n] = item["chain_type"]
        target_mask[i, :n] = item["chain_type"] == 1
        cdr_mask[i, :n] = item["cdr_mask"]
        native_cdr_mask[i, :n] = item["native_cdr_mask"]
        paratope_mask[i, :n] = item["paratope_mask"]
        epitope_mask[i, :n] = item["epitope_mask"]
        sample_ids.append(item["sample_id"])
        orientation_swapped.append(bool(item.get("orientation_swapped", False)))

    chain_breaks = torch.zeros(size, max_len, dtype=torch.bool)
    for i in range(size):
        valid = full_mask[i]
        chain_breaks[i, 1:] = valid[1:] & (chain_type[i, 1:] != chain_type[i, :-1])
    chains = (chain_type - 1).clamp(min=0)
    coords_nm = (coords / 10.0) * coord_mask.unsqueeze(-1)
    # Match the antibody loader's interface-centered coordinates, but use the
    # partner-side interface as the center and fall back to all partner CAs.
    partner = full_mask & (chain_type == 3)
    center_mask = torch.where(epitope_mask.any(dim=1, keepdim=True), epitope_mask, partner)
    ca = coords_nm[:, :, 1, :]
    centroid = (ca * center_mask.unsqueeze(-1)).sum(dim=1) / center_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    coords_nm = (coords_nm - centroid[:, None, None, :]) * coord_mask.unsqueeze(-1)
    return {
        "coords_nm": coords_nm,
        "coords": coords,
        "coord_mask": coord_mask,
        "residue_type": seq,
        "seq": seq,
        "mask": target_mask,
        # Explicit aliases make the general-protein semantics unambiguous to
        # pre-adaptation code while preserving the antibody loader contract.
        "structure_mask": target_mask,
        "sequence_mask": cdr_mask,
        "full_mask": full_mask,
        "cdr_mask": cdr_mask,
        "native_cdr_mask": native_cdr_mask,
        "paratope_mask": paratope_mask,
        "epitope_mask": epitope_mask,
        "chain_type": chain_type,
        "chain_breaks_per_residue": chain_breaks,
        "chains": chains,
        "sample_ids": sample_ids,
        "orientation_swapped": orientation_swapped,
        "mask_dict": {
            "residue_type": target_mask.clone(),
            "coords": (coord_mask.any(dim=-1) & target_mask)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 37, 1)
            .contiguous(),
        },
    }


__all__ = ["ProteinPairDataset", "collate_fn"]
