#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Antibody Design Inference Dataset for La-Proteina.

Reads preprocessed `structure_dataset` data (same format as
`AntibodyDesignDataset`) and produces batches for antigen-conditioned antibody
generation.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset


class AbGenDataset(Dataset):
    """
    Inference dataset for antigen-conditioned antibody generation.

    Reads from the same preprocessed pkl files as AntibodyDesignDataset.
    Returns batches with:
    - Antigen coordinates (fixed, used as conditioning)
    - Antibody framework sequence (CDR positions zeroed)
    - CDR mask, chain_type, etc.
    - Antibody coordinates zeroed (will be noised by FM)
    - nres = antibody length, nsamples = nsamples
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        nsamples: int = 1,
        max_antibody_len: int = 450,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.nsamples = nsamples
        self.max_antibody_len = max_antibody_len

        processed_dir = self.data_dir / "processed" / f"{split}_processed"


        if not processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found: {processed_dir}. "
                "Run AntibodyDesignDataModule to preprocess first."
            )

        metainfo_file = processed_dir / "_metainfo"
        with open(metainfo_file, "r") as f:
            metainfo = json.load(f)

        self.num_entry = metainfo["num_entry"]
        # ``_metainfo`` may have been produced in a different environment.
        # Prefer the recorded path, but relocate only when it is absent and
        # a same-named file exists directly under this verified processed
        # directory.  Do not silently accept a missing or ambiguous part.
        self.file_names = []
        for recorded_name in metainfo["file_names"]:
            recorded_path = Path(recorded_name)
            if recorded_path.is_file():
                resolved_path = recorded_path
            else:
                local_path = processed_dir / recorded_path.name
                if not local_path.is_file():
                    raise FileNotFoundError(
                        f"Processed part is unavailable at both recorded and "
                        f"local paths: {recorded_path} / {local_path}"
                    )
                resolved_path = local_path
            self.file_names.append(str(resolved_path))
        self.file_num_entries = metainfo["file_num_entries"]

        logger.info(
            f"AbGenDataset: {self.num_entry} entries from {processed_dir} "
            f"(nsamples={nsamples})"
        )

        self.cur_file_idx = 0
        self.cur_idx_range = (0, self.file_num_entries[0])
        self._load_part()

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        with open(f, "rb") as fin:
            self.data = pickle.load(fin)

    def _check_load_part(self, idx):
        idx = idx % self.num_entry
        if idx < self.cur_idx_range[0]:
            while idx < self.cur_idx_range[0]:
                end = self.cur_idx_range[0]
                self.cur_file_idx -= 1
                start = end - self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        elif idx >= self.cur_idx_range[1]:
            while idx >= self.cur_idx_range[1]:
                start = self.cur_idx_range[1]
                self.cur_file_idx += 1
                end = start + self.file_num_entries[self.cur_file_idx]
                self.cur_idx_range = (start, end)
            self._load_part()
        return idx - self.cur_idx_range[0]

    def __len__(self):
        return self.num_entry * self.nsamples

    def __getitem__(self, idx: int) -> Dict:
        entry_idx = idx // self.nsamples
        local_idx = self._check_load_part(entry_idx)
        item = self.data[local_idx]

        ab_coords = item["ab_coords"]          # [n_ab, 37, 3]
        ab_mask = item["ab_mask"]              # [n_ab, 37]
        ab_seq = item["ab_seq"]                # [n_ab] 0-19
        h_len = item["h_len"]                  # int
        cdr_mask = item["cdr_mask"]            # [n_ab] bool
        ag_coords = item["ag_coords"]          # [n_ag, 37, 3]
        ag_mask = item["ag_mask"]              # [n_ag, 37]
        ag_seq = item["ag_seq"]                # [n_ag] 0-19
        ag_epitope_flag = item["ag_epitope_flag"]  # [n_ag] bool

        if len(ab_seq) > self.max_antibody_len:
            ab_coords = ab_coords[: self.max_antibody_len]
            ab_mask = ab_mask[: self.max_antibody_len]
            ab_seq = ab_seq[: self.max_antibody_len]
            cdr_mask = cdr_mask[: self.max_antibody_len]
            h_len = min(h_len, self.max_antibody_len)

        n_ab = len(ab_seq)
        n_ag = len(ag_seq)
        l_len = n_ab - h_len

        # chain_type: 1=heavy, 2=light, 3=antigen
        chain_type = np.concatenate([
            np.ones(h_len, dtype=np.int64),
            np.ones(l_len, dtype=np.int64) * 2,
            np.ones(n_ag, dtype=np.int64) * 3,
        ])

        # Concatenate antibody + antigen
        all_coords = np.concatenate([ab_coords, ag_coords], axis=0)   # [n, 37, 3]
        all_coord_mask = np.concatenate([ab_mask, ag_mask], axis=0)   # [n, 37]
        all_seq = np.concatenate([ab_seq, ag_seq], axis=0)            # [n]
        all_cdr = np.concatenate([cdr_mask, np.zeros(n_ag, dtype=np.bool_)])
        all_epitope = np.concatenate([
            np.zeros(n_ab, dtype=np.bool_),
            ag_epitope_flag,
        ])

        # Zero out antibody coordinates — they will be noised by FM
        # Keep ground truth for evaluation
        gt_ab_coords = all_coords[:n_ab].copy()
        gt_ab_seq = ab_seq.copy()
        all_coords[:n_ab] = 0.0
        all_coord_mask[:n_ab] = False

        return {
            "coords": torch.from_numpy(all_coords),
            "coord_mask": torch.from_numpy(all_coord_mask),
            "ab_coord_mask": torch.from_numpy(ab_mask),
            "seq": torch.from_numpy(all_seq),
            "cdr_mask": torch.from_numpy(all_cdr),
            "epitope_mask": torch.from_numpy(all_epitope),
            "chain_type": torch.from_numpy(chain_type),
            "n_ab": n_ab,
            "n_ag": n_ag,
            "pdb_id": item.get("pdb_id", "unknown"),
            # Ground truth for evaluation
            "gt_ab_coords": torch.from_numpy(gt_ab_coords),  # [n_ab, 37, 3]
            "gt_ab_seq": torch.from_numpy(gt_ab_seq),        # [n_ab]
            "gt_cdr_mask": torch.from_numpy(cdr_mask),       # [n_ab] bool
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for AbGenDataset. Each item has a fixed n_ab+n_ag length."""
    max_len = max(item["seq"].shape[0] for item in batch)
    B = len(batch)

    coords = torch.zeros(B, max_len, 37, 3)
    coord_mask = torch.zeros(B, max_len, 37, dtype=torch.bool)
    ab_coord_mask = torch.zeros(B, max_len, 37, dtype=torch.bool)
    seq = torch.zeros(B, max_len, dtype=torch.long)
    full_mask = torch.zeros(B, max_len, dtype=torch.bool)
    ab_mask = torch.zeros(B, max_len, dtype=torch.bool)
    cdr_mask = torch.zeros(B, max_len, dtype=torch.bool)
    epitope_mask = torch.zeros(B, max_len, dtype=torch.bool)
    chain_type = torch.zeros(B, max_len, dtype=torch.long)
    gt_ab_coords = torch.zeros(B, max_len, 37, 3)  # ground-truth ab coords (Angstrom)

    n_ab_list = []
    pdb_ids = []

    for i, item in enumerate(batch):
        n = item["seq"].shape[0]
        n_ab = item["n_ab"]
        n_ab_list.append(n_ab)
        pdb_ids.append(item["pdb_id"])

        coords[i, :n] = item["coords"]
        coord_mask[i, :n] = item["coord_mask"]
        ab_coord_mask[i, :n_ab] = item["ab_coord_mask"][:n_ab]  # antibody coords only
        seq[i, :n] = item["seq"]
        full_mask[i, :n] = True          # all valid residues (ab + ag)
        ab_mask[i, :n_ab] = True         # antibody only (for loss)
        cdr_mask[i, :n] = item["cdr_mask"]
        epitope_mask[i, :n] = item["epitope_mask"]
        chain_type[i, :n] = item["chain_type"]
        if "gt_ab_coords" in item:
            gt_ab_coords[i, :n_ab] = item["gt_ab_coords"]

    # Zero out CDR sequence (unknown during generation)
    gt_seq = seq.clone()  # full sequence preserving true CDR residues
    residue_type = seq.clone()
    residue_type[cdr_mask.bool()] = 0

    # Chain breaks at chain type transitions
    chain_breaks = torch.zeros(B, max_len, dtype=torch.bool)
    for i in range(B):
        for j in range(1, max_len):
            if full_mask[i, j] and chain_type[i, j] != chain_type[i, j - 1]:
                chain_breaks[i, j] = True

    chain_idx = (chain_type - 1).clamp(min=0)
    coords_nm = coords / 10.0

    # Center around epitope CA centroid (same as training)
    use_epitope = epitope_mask.any(dim=1, keepdim=True)
    center_mask = torch.where(use_epitope, epitope_mask, full_mask & (chain_type == 3))
    ag_ca = coords_nm[:, :, 1, :]
    ca_sum = (ag_ca * center_mask.unsqueeze(-1)).sum(dim=1)
    ca_count = center_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
    centroid = (ca_sum / ca_count).unsqueeze(1).unsqueeze(2)
    coords_nm = coords_nm - centroid
    coords_nm = coords_nm * coord_mask.unsqueeze(-1)  # zero out invalid coords

    # Apply same centroid to ground-truth ab coords so ref is in the same frame
    gt_ab_coords_nm = gt_ab_coords / 10.0 - centroid
    gt_ab_coords_nm = gt_ab_coords_nm * ab_coord_mask.unsqueeze(-1)  # zero out invalid coords
    # nres = antibody length (same for all items in batch, or take max)
    nres = max(n_ab_list)

    return {
        "coords_nm": coords_nm,
        "coords": coords,
        "coord_mask": coord_mask,
        "residue_type": residue_type,
        "seq": residue_type,
        "mask": ab_mask,
        "full_mask": full_mask,
        "cdr_mask": cdr_mask,
        "epitope_mask": epitope_mask,
        "chain_type": chain_type,
        "chain_breaks_per_residue": chain_breaks,
        "chains": chain_idx,
        "mask_dict": {
            "residue_type": ab_mask.clone(),
            # [b, n, 37, 1]: FM/AE accesses [..., 0, 0] to get [b, n] ab residue mask.
            "coords": ab_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 37, 1).contiguous(),
        },
        "nres": nres,
        "nsamples": B,
        "pdb_ids": pdb_ids,
        # Ground-truth reference data (for saving ref PDB in same frame as generated PDB)
        "gt_ab_coords_nm": gt_ab_coords_nm,  # [B, max_len, 37, 3] centered, nm
        "gt_seq": gt_seq,                    # [B, max_len] full seq preserving CDR residues
    }


def build_ab_gen_dataloader(
    data_dir: str,
    split: str = "test",
    nsamples: int = 1,
    max_antibody_len: int = 450,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    dataset = AbGenDataset(
        data_dir=data_dir,
        split=split,
        nsamples=nsamples,
        max_antibody_len=max_antibody_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
