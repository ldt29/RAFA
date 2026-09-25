#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Antibody Design Dataset for La-Proteina.

This module provides dataset classes for antibody CDR generation using the
`structure_dataset` antibody-antigen corpus. It supports both conventional
antibodies (VH-VL) and nanobodies (VHH).

Data is preprocessed into atom37 format with sequence indices 0-19.
"""

import itertools
import json
import math
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Sampler

from openfold.np.residue_constants import atom_order, restype_1to3, restypes, RESTYPE_ATOM14_TO_ATOM37

from proteinfoundation.datasets.base_data import BaseLightningDataModule
from proteinfoundation.utils.antibody_utils import VOCAB, AgAbComplex


# Use OpenFold residue index order everywhere in the antibody pipeline.
# OpenFold restypes: ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
_AA_SYMBOLS = list(restypes)
_AA_TO_IDX = {aa: i for i, aa in enumerate(_AA_SYMBOLS)}


def extract_residue_coords_atom37(residue, aa_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract atom37 coordinates from a residue.

    Args:
        residue: Residue object from antibody_utils
        aa_idx: amino acid index 0-19

    Returns:
        coords37: [37, 3] float32
        mask37: [37] bool
    """
    bb_coord = residue.get_backbone_coord_map()
    sc_coord = residue.get_sidechain_coord_map()

    # Build atom14 coords in canonical sidechain atom order for this residue type.
    atom14_coords = np.zeros((14, 3), dtype=np.float32)
    atom14_valid = np.zeros(14, dtype=bool)

    bb_atoms = ['N', 'CA', 'C', 'O']
    for i, atom in enumerate(bb_atoms):
        if atom in bb_coord:
            atom14_coords[i] = bb_coord[atom]
            atom14_valid[i] = True

    sc_atoms = VOCAB.sidechain_map.get(_AA_SYMBOLS[aa_idx], [])
    for j, atom in enumerate(sc_atoms):
        idx14 = 4 + j
        if idx14 < 14 and atom in sc_coord:
            atom14_coords[idx14] = sc_coord[atom]
            atom14_valid[idx14] = True

    # Convert to atom37 using OpenFold's precomputed mapping.
    # RESTYPE_ATOM14_TO_ATOM37[aa_idx] has shape (14,), values are atom37 indices.
    coords37 = np.zeros((37, 3), dtype=np.float32)
    mask37 = np.zeros(37, dtype=bool)

    mapping = RESTYPE_ATOM14_TO_ATOM37[aa_idx]  # shape (14,)
    for i14, i37 in enumerate(mapping):
        if atom14_valid[i14]:
            coords37[i37] = atom14_coords[i14]
            mask37[i37] = True

    return coords37, mask37


def extract_epitope_coords(ag_ab_complex):
    """
    Extract only epitope residues from antigen.
    Returns: epitope_coords [n,37,3], epitope_mask [n,37], epitope_seq [n] (0-19)
    """
    epitope = ag_ab_complex.get_epitope()

    epitope_residues = {}
    for residue, chain_name, idx in epitope:
        if chain_name not in epitope_residues:
            epitope_residues[chain_name] = []
        epitope_residues[chain_name].append((idx, residue))

    for chain in epitope_residues:
        epitope_residues[chain].sort(key=lambda x: x[0])

    coords_list, mask_list, seq_list = [], [], []

    for chain_name in sorted(epitope_residues.keys()):
        for idx, residue in epitope_residues[chain_name]:
            aa_idx = _AA_TO_IDX.get(residue.get_symbol())
            if aa_idx is None:
                continue
            c37, m37 = extract_residue_coords_atom37(residue, aa_idx)
            coords_list.append(c37)
            mask_list.append(m37)
            seq_list.append(aa_idx)

    if len(coords_list) == 0:
        return (np.zeros((0, 37, 3), dtype=np.float32),
                np.zeros((0, 37), dtype=bool),
                np.zeros(0, dtype=np.int64))

    return (np.stack(coords_list).astype(np.float32),
            np.stack(mask_list),
            np.array(seq_list, dtype=np.int64))


def extract_antigen_coords(ag_ab_complex):
    """
    Extract full antigen (all chains) coordinates and sequence.
    Returns: ag_coords [n,37,3], ag_mask [n,37], ag_seq [n] (0-19), epitope_flag [n] bool
    epitope_flag marks which antigen residues are epitope residues.
    """
    antigen = ag_ab_complex.get_antigen()
    epitope_set = set()
    try:
        epitope = ag_ab_complex.get_epitope()
        for residue, chain_name, idx in epitope:
            epitope_set.add((chain_name, idx))
    except Exception:
        pass

    coords_list, mask_list, seq_list, epitope_flag_list = [], [], [], []

    for chain_name in antigen.get_chain_names():
        chain = antigen.get_chain(chain_name)
        if chain is None:
            continue
        for i in range(len(chain)):
            residue = chain.get_residue(i)
            aa_idx = _AA_TO_IDX.get(residue.get_symbol(), _AA_TO_IDX['G'])
            c37, m37 = extract_residue_coords_atom37(residue, aa_idx)
            coords_list.append(c37)
            mask_list.append(m37)
            seq_list.append(aa_idx)
            epitope_flag_list.append((chain_name, i) in epitope_set)

    if len(coords_list) == 0:
        return (np.zeros((0, 37, 3), dtype=np.float32),
                np.zeros((0, 37), dtype=bool),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=bool))

    return (np.stack(coords_list).astype(np.float32),
            np.stack(mask_list),
            np.array(seq_list, dtype=np.int64),
            np.array(epitope_flag_list, dtype=bool))


def extract_ab_coords(ag_ab_complex):
    """
    Extract antibody (heavy + light chain) coordinates and sequence.
    Returns: ab_coords [n,37,3], ab_mask [n,37], ab_seq [n] (0-19), h_len int
    """
    hc = ag_ab_complex.get_heavy_chain()
    lc = ag_ab_complex.get_light_chain()

    coords_list, mask_list, seq_list = [], [], []
    h_len = 0

    if hc:
        for i in range(len(hc)):
            residue = hc.get_residue(i)
            aa_idx = _AA_TO_IDX.get(residue.get_symbol(), _AA_TO_IDX['G'])
            c37, m37 = extract_residue_coords_atom37(residue, aa_idx)
            coords_list.append(c37)
            mask_list.append(m37)
            seq_list.append(aa_idx)
        h_len = len(hc)

    if lc:
        for i in range(len(lc)):
            residue = lc.get_residue(i)
            aa_idx = _AA_TO_IDX.get(residue.get_symbol(), _AA_TO_IDX['G'])
            c37, m37 = extract_residue_coords_atom37(residue, aa_idx)
            coords_list.append(c37)
            mask_list.append(m37)
            seq_list.append(aa_idx)

    if len(coords_list) == 0:
        return (np.zeros((0, 37, 3), dtype=np.float32),
                np.zeros((0, 37), dtype=bool),
                np.zeros(0, dtype=np.int64),
                0)

    return (np.stack(coords_list).astype(np.float32),
            np.stack(mask_list),
            np.array(seq_list, dtype=np.int64),
            h_len)


def get_cdr_mask(ag_ab_complex):
    """Get CDR mask for antibody. Returns: cdr_mask (1 for CDR, 0 for framework)"""
    hc = ag_ab_complex.get_heavy_chain()
    lc = ag_ab_complex.get_light_chain()

    h_len = len(hc) if hc else 0
    l_len = len(lc) if lc else 0

    cdr_mask = np.zeros(h_len + l_len, dtype=np.bool_)

    if hc:
        for cdr in ['H1', 'H2', 'H3']:
            cdr_range = ag_ab_complex.get_cdr_pos(cdr)
            start, end = cdr_range
            if start < h_len and end < h_len:
                cdr_mask[start:end+1] = True

    if lc:
        for cdr in ['L1', 'L2', 'L3']:
            cdr_range = ag_ab_complex.get_cdr_pos(cdr)
            start, end = cdr_range
            adj_start = start + h_len
            adj_end = end + h_len
            if adj_start < len(cdr_mask) and adj_end < len(cdr_mask):
                cdr_mask[adj_start:adj_end+1] = True

    return cdr_mask


def get_paratope_flag(ab_coords: np.ndarray, ab_mask: np.ndarray,
                      ag_coords: np.ndarray, ag_mask: np.ndarray,
                      threshold_ang: float = 10.0) -> np.ndarray:
    """
    Compute paratope mask: antibody residues within threshold_ang of any antigen CA.
    Paratope (antibody-side) corresponds to epitope (antigen-side).
    Vectorized via scipy cdist.
    Returns: paratope_flag [n_ab] bool
    """
    from scipy.spatial.distance import cdist as scipy_cdist
    ab_ca = ab_coords[:, 1, :]          # [n_ab, 3]
    ag_ca = ag_coords[:, 1, :]          # [n_ag, 3]
    ab_ca_valid = ab_mask[:, 1]         # [n_ab] bool
    ag_ca_valid = ag_mask[:, 1]         # [n_ag] bool

    if ab_ca_valid.sum() == 0 or ag_ca_valid.sum() == 0:
        return np.zeros(len(ab_ca), dtype=bool)

    ab_idx = np.where(ab_ca_valid)[0]
    ag_idx = np.where(ag_ca_valid)[0]
    dists = scipy_cdist(ab_ca[ab_idx], ag_ca[ag_idx])   # [n_ab_valid, n_ag_valid]
    close = (dists < threshold_ang).any(axis=1)          # [n_ab_valid]

    paratope_flag = np.zeros(len(ab_ca), dtype=bool)
    paratope_flag[ab_idx] = close
    return paratope_flag


def _trim_antigen_to_epitope(ag_coords, ag_mask, ag_seq, ag_epitope_flag, max_ag_len: int):
    """Trim antigen to max_ag_len residues, centering the window on the epitope span."""
    n_ag = len(ag_seq)
    if n_ag <= max_ag_len:
        return ag_coords, ag_mask, ag_seq, ag_epitope_flag

    epitope_indices = np.where(ag_epitope_flag)[0]
    if len(epitope_indices) == 0:
        start = 0
    else:
        ep_center = int(epitope_indices[0] + epitope_indices[-1]) // 2
        start = ep_center - max_ag_len // 2
        start = max(0, min(start, n_ag - max_ag_len))

    end = start + max_ag_len
    return ag_coords[start:end], ag_mask[start:end], ag_seq[start:end], ag_epitope_flag[start:end]


# ── structure_dataset preprocessing ──────────────────────────────────────────

_DATA_SUBDIRS = ["before_20250630", "after_20250630_novel_ab",
                 "after_20250630_novel_nb", "synthesis_data"]

# split name -> (split_json filename, output subdir, name_filter)
_SPLIT_CONFIG = {
    'train':     ('train.json',     'train_processed',     None),
    'valid':     ('valid.json',     'valid_processed',     None),
    'test':      ('test.json',      'test_processed',      None),
    # antibody-only subset of train (names containing '_ab', i.e. not nanobodies)
    'antibody':  ('train.json',     'antibody_processed',  lambda name: '_ab' in name),
    # synthetic Protenix-cofolded positives (see build_synthesis_dataset.py)
    'synthesis': ('synthesis.json', 'synthesis_processed', None),
    # high-confidence subset of synthesis (best_iptm >= 0.8, ~12k samples)
    'synthesis_hi': ('synthesis_hi.json', 'synthesis_hi_processed', None),
    'train_real': ("train_real.json", "train_real_processed", None),
    # v56: the 2208 train targets that HAVE a real protenix prior in
    # protenix_priors_train_v43 (the other 82%% would fall back to GT and train identity)
    'train_prior': ("train_prior.json", "train_prior_processed", None),
}

# Core splits processed by default / by the auto-rebuild path. `synthesis` is
# opt-in (20k+ samples) so it never rebuilds implicitly with the core cache.
_CORE_SPLITS = ['train', 'valid', 'test', 'antibody']


def _find_sample_dir(structure_dataset_dir: Path, sample_name: str) -> Optional[Path]:
    """Locate a sample folder by name across the three data subdirectories."""
    for sub in _DATA_SUBDIRS:
        p = structure_dataset_dir / sub / sample_name
        if p.exists():
            return p
    return None


def process_single_pdb(args: Tuple) -> Optional[Dict]:
    """Worker function to process a single sample from the new structure_dataset."""
    sample_name, sample_dir, max_antigen_len = args

    try:
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            return None

        meta = json.load(open(meta_path))
        ab_type = meta.get("ab_type", "VH-VL")

        pdb_path = sample_dir / "complex.pdb"
        if not pdb_path.exists():
            return None

        # VHH has no light chain
        light_chain = 'L' if ab_type == "VH-VL" else None

        cplx = AgAbComplex.from_pdb(
            str(pdb_path), 'H', light_chain, ['A'],
            skip_epitope_cal=False
        )

        ab_coords, ab_mask, ab_seq, h_len = extract_ab_coords(cplx)
        ag_coords, ag_mask, ag_seq, ag_epitope_flag = extract_antigen_coords(cplx)
        cdr_mask = get_cdr_mask(cplx)

        if len(ab_seq) == 0 or len(ag_seq) == 0:
            return None

        # Filter: require at least one CDR residue and one epitope residue.
        # Samples missing either cause DDP deadlocks when CDR-epitope cross-attention
        # is used (conditional execution diverges across ranks → all_reduce hangs).
        if not cdr_mask.any():
            print(f"Warning: sample {sample_name} has no CDR residues, skipping")
            return None
        if not ag_epitope_flag.any():
            print(f"Warning: sample {sample_name} has no epitope residues, skipping")
            return None

        ag_coords, ag_mask, ag_seq, ag_epitope_flag = _trim_antigen_to_epitope(
            ag_coords, ag_mask, ag_seq, ag_epitope_flag, max_antigen_len
        )

        # After trimming, re-check epitope is still present in the window.
        if not ag_epitope_flag.any():
            print(f"Warning: sample {sample_name} has no epitope residues after trimming, skipping")
            return None

        # Paratope: antibody residues within 10 Å of any antigen CA (GT-based, for iRMS)
        paratope_flag = get_paratope_flag(ab_coords, ab_mask, ag_coords, ag_mask)

        return {
            'pdb_id': sample_name,
            'ab_type': ab_type,
            'ab_coords': ab_coords,
            'ab_mask': ab_mask,
            'ab_seq': ab_seq,
            'h_len': h_len,
            'cdr_mask': cdr_mask,
            'paratope_flag': paratope_flag,
            'ag_coords': ag_coords,
            'ag_mask': ag_mask,
            'ag_seq': ag_seq,
            'ag_epitope_flag': ag_epitope_flag,
        }

    except Exception:
        return None


def _preprocess_split(structure_dataset_dir: Path, split_json: Path,
                          output_dir: Path, max_antigen_len: int = 500,
                          num_per_file: int = 5000, num_workers: int = 16,
                          name_filter=None):
    """Preprocess a split from the new structure_dataset format.

    Args:
        name_filter: optional callable(sample_name) -> bool to select a subset.
                     E.g. ``lambda n: 'ab' in n`` to keep only antibodies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(split_json, 'r') as f:
        sample_names = json.load(f)

    if name_filter is not None:
        before = len(sample_names)
        sample_names = [n for n in sample_names if name_filter(n)]
        logger.info(f"name_filter kept {len(sample_names)}/{before} entries")

    logger.info(f"Processing {len(sample_names)} entries for {split_json.name} with {num_workers} workers")

    tasks = []
    missing = 0
    for name in sample_names:
        sd = _find_sample_dir(structure_dataset_dir, name)
        if sd is not None:
            tasks.append((name, sd, max_antigen_len))
        else:
            missing += 1

    if missing > 0:
        logger.warning(f"Could not find {missing} sample directories for {split_json.name}")

    logger.info(f"Found {len(tasks)} valid sample directories")

    processed_data = []
    file_idx = 0
    total_processed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_pdb, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc=split_json.name):
            result = future.result()
            if result is not None:
                processed_data.append(result)
                total_processed += 1
                if len(processed_data) >= num_per_file:
                    out = output_dir / f"part_{file_idx}.pkl"
                    with open(out, 'wb') as f:
                        pickle.dump(processed_data, f)
                    file_idx += 1
                    processed_data = []

    if processed_data:
        out = output_dir / f"part_{file_idx}.pkl"
        with open(out, 'wb') as f:
            pickle.dump(processed_data, f)
        file_idx += 1

    metainfo = {
        'num_entry': total_processed,
        'file_names': [str(output_dir / f"part_{i}.pkl") for i in range(file_idx)],
        'file_num_entries': [num_per_file] * (file_idx - 1) + [total_processed % num_per_file or num_per_file],
        'format': 'atom37',
        'max_antigen_len': max_antigen_len,
        'version': 3,  # v3: adds ab_type metadata; older caches infer it from light-chain length.
    }
    with open(output_dir / "_metainfo", 'w') as f:
        json.dump(metainfo, f)

    logger.info(f"Total processed: {total_processed} for {split_json.name}")
    return total_processed


def process_data(structure_dataset_dir: str, max_antigen_len: int = 500,
                 force_reprocess: bool = False, splits: Optional[List[str]] = None):
    """Process structure_dataset if cache doesn't exist or params changed.

    Args:
        splits: which splits to (re)build. Defaults to the core splits
                (train/valid/test/antibody). Pass e.g. ['synthesis'] to build
                only the synthetic-positive cache.
    """
    structure_dataset_dir = Path(structure_dataset_dir)
    processed_base = structure_dataset_dir / "processed"

    if splits is None:
        splits = _CORE_SPLITS

    if processed_base.exists() and not force_reprocess:
        all_exist = all((processed_base / _SPLIT_CONFIG[s][1] / "_metainfo").exists()
                        for s in splits)
        if all_exist:
            probe = splits[0]
            with open(processed_base / _SPLIT_CONFIG[probe][1] / "_metainfo") as f:
                meta = json.load(f)
            if (meta.get('format') == 'atom37'
                    and meta.get('max_antigen_len') == max_antigen_len
                    and meta.get('version', 1) >= 3):
                logger.info("Processed data (atom37) already exists, skipping")
                return

    num_workers = max(1, mp.cpu_count() - 2)
    for split in splits:
        split_fname, out_subdir, name_filter = _SPLIT_CONFIG[split]
        split_file = structure_dataset_dir / "split_json" / split_fname
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            continue
        _preprocess_split(structure_dataset_dir, split_file,
                          processed_base / out_subdir,
                          max_antigen_len=max_antigen_len,
                          num_workers=num_workers,
                          name_filter=name_filter)


class AntibodyDesignDataset(torch.utils.data.Dataset):
    """Dataset for antibody design with antigen conditioning. Uses atom37 format."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cdr_target: str = "all",
        max_antibody_len: int = 450,
        max_antigen_len: int = 500,
        mask_strategy: str = "full",
        mask_strategy_probs: Optional[List[float]] = None,
        cdr_annotation_manifest: Optional[str] = None,
        conventional_only: bool = False,
        vhh_only: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.cdr_target = cdr_target
        self.max_antibody_len = max_antibody_len
        self.max_antigen_len = max_antigen_len
        self.conventional_only = bool(conventional_only)
        self.vhh_only = bool(vhh_only)
        if self.conventional_only and self.vhh_only:
            raise ValueError("conventional_only and vhh_only are mutually exclusive")

        # Mixed masking strategy (only applied during training).
        # mask_strategy: "full" | "mixed"
        # mask_strategy_probs: [p_full, p_partial, p_none], must sum to 1.0
        #   p_full    → mask all CDR residues (de novo generation)
        #   p_partial → randomly mask a contiguous or scattered subset of CDR residues
        #   p_none    → unmask all CDR residues (reconstruction / identity task)
        self.mask_strategy = mask_strategy
        self.cdr_annotation_manifest = cdr_annotation_manifest
        if mask_strategy_probs is None:
            mask_strategy_probs = [0.5, 0.3, 0.2]
        assert abs(sum(mask_strategy_probs) - 1.0) < 1e-6, "mask_strategy_probs must sum to 1"
        self.mask_strategy_probs = mask_strategy_probs

        self.processed_dir = self.data_dir / "processed" / f"{split}_processed"

        if not self.processed_dir.exists():
            logger.info("Processed data not found, running preprocessing...")
            process_data(str(self.data_dir), max_antigen_len=self.max_antigen_len,
                         splits=[split])

        if not self.processed_dir.exists():
            raise FileNotFoundError(f"Processed data not found: {self.processed_dir}")

        metainfo_file = self.processed_dir / "_metainfo"
        if not metainfo_file.exists():
            raise FileNotFoundError(f"Metainfo not found: {metainfo_file}")

        with open(metainfo_file, 'r') as f:
            metainfo = json.load(f)

        if (metainfo.get('format') != 'atom37'
                or metainfo.get('max_antigen_len') != self.max_antigen_len
                or metainfo.get('version', 1) < 2):
            logger.warning("Cache format/version/max_antigen_len mismatch, re-processing...")
            process_data(str(self.data_dir), max_antigen_len=self.max_antigen_len,
                         force_reprocess=True, splits=[split])
            with open(metainfo_file, 'r') as f:
                metainfo = json.load(f)

        self.original_num_entry = int(metainfo['num_entry'])
        self.num_entry = self.original_num_entry
        # Reconstruct part paths from the local processed_dir instead of trusting
        # the absolute paths baked into _metainfo at preprocess time. Caches are
        # rsync'd across boxes with different data_dir roots (e.g. assets/data1 vs
        # assets/data1), so the stored absolutes may not exist here.
        self.file_names = [
            str(self.processed_dir / os.path.basename(p))
            for p in metainfo['file_names']
        ]
        self.file_num_entries = metainfo['file_num_entries']

        logger.info(f"Loaded {self.num_entry} entries (atom37) from {self.processed_dir}")

        self.cur_file_idx = 0
        self.cur_idx_range = (0, self.file_num_entries[0])
        self._load_part()
        self.ab_indices, self.nb_indices = self._build_abnb_indices()
        self._selected_global_indices = None
        if self.conventional_only or self.vhh_only:
            self._selected_global_indices = (
                self.ab_indices if self.conventional_only else self.nb_indices
            )
            self.num_entry = len(self._selected_global_indices)
            label = "Conventional-only" if self.conventional_only else "VHH-only"
            logger.info(
                f"{label} filter selected {self.num_entry}/{self.original_num_entry} entries"
            )

    @staticmethod
    def _infer_ab_type(item: Dict) -> str:
        ab_type = item.get("ab_type")
        if ab_type in ("VH-VL", "VHH"):
            return ab_type
        # Backward-compatible path for v2 caches. VHH has no light chain, so
        # heavy length equals antibody length.
        h_len = int(item.get("h_len", 0))
        ab_seq = item.get("ab_seq")
        n_ab = len(ab_seq) if ab_seq is not None else h_len
        return "VHH" if n_ab == h_len else "VH-VL"

    def _build_abnb_indices(self) -> Tuple[List[int], List[int]]:
        ab_indices: List[int] = []
        nb_indices: List[int] = []
        self.pdb_ids: List[str] = []
        # Geometry-cardinality metadata is cheap to collect during the existing
        # cache scan and lets method-specific samplers exclude inputs on which a
        # rigid alignment is mathematically undefined.  The raw dataset remains
        # byte-for-byte unchanged and default samplers retain every entry.
        self.antigen_residue_counts: List[int] = []
        self.epitope_residue_counts: List[int] = []
        self.cdr_loop_counts: List[int] = []
        global_idx = 0
        for f in self.file_names:
            with open(f, 'rb') as fin:
                part = pickle.load(fin)
            for item in part:
                self.pdb_ids.append(str(item.get("pdb_id", "")))
                self.antigen_residue_counts.append(len(item.get("ag_seq", ())))
                epitope_flag = item.get("ag_epitope_flag")
                self.epitope_residue_counts.append(
                    int(np.asarray(epitope_flag, dtype=np.bool_).sum())
                    if epitope_flag is not None else 0
                )
                # Rigid CDR-frame transports require a minimum number of
                # semantic loops.  Record this public annotation metadata once
                # during the existing cache scan so a candidate can express the
                # eligibility boundary without probing native coordinates in
                # the model path or mutating the cache.
                loop_ids = item.get("cdr_loop_id")
                if loop_ids is None and self.cdr_annotation_manifest:
                    from proteinfoundation.datasets.cdr_annotations import (
                        annotation_arrays,
                        decode_aatype,
                    )

                    _, loop_ids = annotation_arrays(
                        self.cdr_annotation_manifest,
                        str(item.get("pdb_id", "")),
                        decode_aatype(item["ab_seq"]),
                        int(item["h_len"]),
                    )
                if loop_ids is None:
                    loop_count = 0
                else:
                    loop_array = np.asarray(loop_ids, dtype=np.int64)
                    loop_count = int(np.unique(loop_array[loop_array > 0]).size)
                self.cdr_loop_counts.append(loop_count)
                if self._infer_ab_type(item) == "VHH":
                    nb_indices.append(global_idx)
                else:
                    ab_indices.append(global_idx)
                global_idx += 1
        if len(self.antigen_residue_counts) != self.num_entry:
            raise ValueError(
                "geometry-cardinality scan did not cover the complete dataset: "
                f"{len(self.antigen_residue_counts)} != {self.num_entry}"
            )
        logger.info(
            f"Split {self.split}: {len(ab_indices)} VH-VL antibodies, "
            f"{len(nb_indices)} VHH nanobodies"
        )
        return ab_indices, nb_indices

    def _load_part(self):
        f = self.file_names[self.cur_file_idx]
        with open(f, 'rb') as fin:
            self.data = pickle.load(fin)
        self.access_idx = list(range(len(self.data)))

    def _check_load_part(self, idx):
        idx = idx % self.original_num_entry
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
        return self.access_idx[idx - self.cur_idx_range[0]]

    def _apply_mask_strategy(self, cdr_mask: np.ndarray) -> np.ndarray:
        """Apply mixed masking strategy to CDR mask (training only).

        Strategies:
          full    → keep all CDR positions masked (de novo generation)
          partial → randomly unmask a fraction of CDR positions:
                      50% chance: contiguous span (simulates loop reconstruction)
                      50% chance: scattered residues (simulates point mutations)
          none    → unmask all CDR positions (reconstruction task)
        """
        if self.mask_strategy != "mixed":
            return cdr_mask

        choice = np.random.choice(3, p=self.mask_strategy_probs)

        if choice == 0:  # full mask — keep as is
            return cdr_mask

        if choice == 2:  # no mask — all CDR visible
            return np.zeros_like(cdr_mask)

        # partial mask
        cdr_indices = np.where(cdr_mask)[0]
        if len(cdr_indices) == 0:
            return cdr_mask

        # Keep between 10% and 90% of CDR residues masked
        n_keep_masked = max(1, int(len(cdr_indices) * np.random.uniform(0.1, 0.9)))

        new_mask = cdr_mask.copy()
        if np.random.rand() < 0.5:
            # Contiguous span: pick a random window of cdr_indices to keep masked
            max_start = len(cdr_indices) - n_keep_masked
            start = np.random.randint(0, max(1, max_start + 1))
            masked_subset = cdr_indices[start:start + n_keep_masked]
        else:
            # Scattered: random subset without replacement
            masked_subset = np.random.choice(cdr_indices, size=n_keep_masked, replace=False)

        new_mask[:] = False
        new_mask[masked_subset] = True
        return new_mask

    def __len__(self):
        return self.num_entry

    def __getitem__(self, idx: int) -> Dict:
        idx = idx % self.num_entry
        if self._selected_global_indices is not None:
            idx = self._selected_global_indices[idx]
        idx = self._check_load_part(idx)
        item = self.data[idx]

        ab_coords = item['ab_coords']   # [n_ab, 37, 3]
        ab_mask = item['ab_mask']       # [n_ab, 37]
        ab_seq = item['ab_seq']         # [n_ab] 0-19
        h_len = item['h_len']           # int
        ab_type = self._infer_ab_type(item)
        cdr_mask = item['cdr_mask']     # [n_ab]
        cdr_loop_id = np.zeros(len(ab_seq), dtype=np.int64)
        if self.cdr_annotation_manifest:
            from proteinfoundation.datasets.cdr_annotations import (
                annotation_arrays,
                decode_aatype,
            )

            cdr_mask, cdr_loop_id = annotation_arrays(
                self.cdr_annotation_manifest,
                str(item.get('pdb_id', '')),
                decode_aatype(ab_seq),
                int(h_len),
            )
        # Keep the canonical CDR annotation separately from the optional
        # mixed-mask training view. The student boundary must hide native CDR
        # sequence even on partial/no-mask auxiliary examples.
        target_cdr_mask = cdr_mask.copy()
        cdr_mask = self._apply_mask_strategy(cdr_mask)
        # A mixed/partial mask must not retain loop IDs at unmasked positions.
        cdr_loop_id = cdr_loop_id * cdr_mask.astype(np.int64)
        paratope_flag = item.get('paratope_flag', np.zeros(len(ab_seq), dtype=bool))  # [n_ab]
        ag_coords = item['ag_coords']          # [n_ag, 37, 3]
        ag_mask = item['ag_mask']              # [n_ag, 37]
        ag_seq = item['ag_seq']                # [n_ag] 0-19
        ag_epitope_flag = item['ag_epitope_flag']  # [n_ag] bool

        if len(ab_seq) > self.max_antibody_len:
            ab_coords = ab_coords[:self.max_antibody_len]
            ab_mask = ab_mask[:self.max_antibody_len]
            ab_seq = ab_seq[:self.max_antibody_len]
            cdr_mask = cdr_mask[:self.max_antibody_len]
            target_cdr_mask = target_cdr_mask[:self.max_antibody_len]
            cdr_loop_id = cdr_loop_id[:self.max_antibody_len]
            paratope_flag = paratope_flag[:self.max_antibody_len]
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

        all_coords = np.concatenate([ab_coords, ag_coords], axis=0)   # [n, 37, 3]
        all_mask = np.concatenate([ab_mask, ag_mask], axis=0)          # [n, 37]
        all_seq = np.concatenate([ab_seq, ag_seq], axis=0)             # [n]
        all_cdr = np.concatenate([cdr_mask, np.zeros(n_ag, dtype=np.bool_)])
        all_target_cdr = np.concatenate([
            target_cdr_mask,
            np.zeros(n_ag, dtype=np.bool_),
        ])
        all_cdr_loop_id = np.concatenate([
            cdr_loop_id,
            np.zeros(n_ag, dtype=np.int64),
        ])
        # paratope_mask: True for antibody residues within 10 Å of any antigen CA
        all_paratope = np.concatenate([
            paratope_flag,
            np.zeros(n_ag, dtype=np.bool_),
        ])
        # epitope_mask: True for antigen residues that are epitope residues
        all_epitope = np.concatenate([
            np.zeros(n_ab, dtype=np.bool_),
            ag_epitope_flag,
        ])

        return {
            'coords': torch.from_numpy(all_coords),       # [n, 37, 3]
            'coord_mask': torch.from_numpy(all_mask),     # [n, 37]
            'seq': torch.from_numpy(all_seq),             # [n] 0-19
            'cdr_mask': torch.from_numpy(all_cdr),
            'target_cdr_mask': torch.from_numpy(all_target_cdr),
            'cdr_loop_id': torch.from_numpy(all_cdr_loop_id),
            'paratope_mask': torch.from_numpy(all_paratope),
            'epitope_mask': torch.from_numpy(all_epitope),
            'chain_type': torch.from_numpy(chain_type),
            # 0 = conventional VH-VL antibody, 1 = VHH nanobody.
            'ab_type': torch.tensor(1 if ab_type == "VHH" else 0, dtype=torch.long),
            'pdb_id': str(item.get('pdb_id', '')),
        }


class BalancedAbNbBatchSampler(Sampler[List[int]]):
    """Batch sampler that keeps a controlled VH-VL / VHH training mix.

    The sampler is DDP-aware: each rank receives a different local batch stream.
    For small per-rank batch sizes, the requested ratio is matched over a short
    repeating slot pattern rather than forcing every local batch to contain both
    types, which would make ratios such as 3:1 impossible with batch_size=2.
    """

    def __init__(
        self,
        ab_indices: List[int],
        nb_indices: List[int],
        batch_size: int,
        ratio: Tuple[int, int] = (3, 1),
        drop_last: bool = True,
        seed: int = 0,
    ):
        if len(ab_indices) == 0 or len(nb_indices) == 0:
            raise ValueError(
                "BalancedAbNbBatchSampler requires both VH-VL and VHH samples. "
                f"Got {len(ab_indices)} VH-VL and {len(nb_indices)} VHH."
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        ab_weight, nb_weight = int(ratio[0]), int(ratio[1])
        if ab_weight <= 0 or nb_weight <= 0:
            raise ValueError(f"balanced_abnb_ratio must be positive, got {ratio}")

        self.ab_indices = list(ab_indices)
        self.nb_indices = list(nb_indices)
        self.batch_size = int(batch_size)
        self.ratio = (ab_weight, nb_weight)
        self.drop_last = drop_last
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = len(self.ab_indices) + len(self.nb_indices)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    @staticmethod
    def _rank_info() -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __len__(self) -> int:
        rank, world_size = self._rank_info()
        return self._num_global_batches(world_size) // world_size

    def _num_global_batches(self, world_size: int) -> int:
        if self.drop_last:
            global_batches = self.num_samples // self.batch_size
            return (global_batches // world_size) * world_size
        global_batches = math.ceil(self.num_samples / self.batch_size)
        return math.ceil(global_batches / world_size) * world_size

    def _shuffled_cycle(self, indices: List[int], generator: torch.Generator):
        while True:
            order = torch.randperm(len(indices), generator=generator).tolist()
            for idx in order:
                yield indices[idx]

    def __iter__(self):
        rank, world_size = self._rank_info()
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch * 1009)

        ab_iter = self._shuffled_cycle(self.ab_indices, generator)
        nb_iter = self._shuffled_cycle(self.nb_indices, generator)
        type_pattern = ["ab"] * self.ratio[0] + ["nb"] * self.ratio[1]
        pattern = itertools.cycle(type_pattern)

        global_batches = self._num_global_batches(world_size)

        for global_batch_idx in range(global_batches):
            batch = [
                next(ab_iter) if next(pattern) == "ab" else next(nb_iter)
                for _ in range(self.batch_size)
            ]
            if global_batch_idx % world_size == rank:
                yield batch


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate function - pads atom37 data to same length."""
    max_len = max(item['seq'].shape[0] for item in batch)
    B = len(batch)

    coords = torch.zeros(B, max_len, 37, 3)
    coord_mask = torch.zeros(B, max_len, 37, dtype=torch.bool)
    seq = torch.zeros(B, max_len, dtype=torch.long)
    full_mask = torch.zeros(B, max_len, dtype=torch.bool)   # all valid residues (ab + ag)
    ab_mask = torch.zeros(B, max_len, dtype=torch.bool)     # antibody only (for FM loss)
    cdr_mask = torch.zeros(B, max_len, dtype=torch.bool)
    target_cdr_mask = torch.zeros(B, max_len, dtype=torch.bool)
    cdr_loop_id = torch.zeros(B, max_len, dtype=torch.long)
    paratope_mask = torch.zeros(B, max_len, dtype=torch.bool)
    epitope_mask = torch.zeros(B, max_len, dtype=torch.bool)
    chain_type = torch.zeros(B, max_len, dtype=torch.long)
    ab_type = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        n = item['seq'].shape[0]
        coords[i, :n] = item['coords']
        coord_mask[i, :n] = item['coord_mask']
        seq[i, :n] = item['seq']
        full_mask[i, :n] = True
        cdr_mask[i, :n] = item['cdr_mask']
        target_cdr_mask[i, :n] = item.get(
            'target_cdr_mask', item['cdr_mask']
        )
        cdr_loop_id[i, :n] = item.get('cdr_loop_id', torch.zeros(n, dtype=torch.long))
        paratope_mask[i, :n] = item.get('paratope_mask', torch.zeros(n, dtype=torch.bool))
        epitope_mask[i, :n] = item['epitope_mask']
        chain_type[i, :n] = item['chain_type']
        ab_type[i] = item.get('ab_type', torch.tensor(0, dtype=torch.long))

    pdb_ids = [str(item.get('pdb_id', '')) for item in batch]

    ab_mask = full_mask & (chain_type < 3)

    # Chain breaks at chain type transitions
    chain_breaks = torch.zeros(B, max_len, dtype=torch.bool)
    for i in range(B):
        for j in range(1, max_len):
            if full_mask[i, j] and chain_type[i, j] != chain_type[i, j-1]:
                chain_breaks[i, j] = True

    # chain_idx: 0=heavy, 1=light, 2=antigen
    chain_idx = (chain_type - 1).clamp(min=0)

    # Convert coords to nm (model expects nm, PDB is in Angstrom)
    coords_nm = coords / 10.0

    # Center around epitope CA centroid.
    # The epitope is the antibody-antigen interface — centering there puts
    # both CDR loops and binding site near the origin, ideal for conditioning.
    # Fall back to antigen centroid if epitope mask is empty.
    use_epitope = epitope_mask.any(dim=1, keepdim=True)  # [b, 1]
    center_mask = torch.where(use_epitope, epitope_mask, full_mask & (chain_type == 3))
    ca = coords_nm[:, :, 1, :]  # [b, n, 3] CA coords
    ca_sum = (ca * center_mask.unsqueeze(-1)).sum(dim=1)  # [b, 3]
    ca_count = center_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
    centroid = (ca_sum / ca_count).unsqueeze(1).unsqueeze(2)  # [b, 1, 1, 3]
    coords_nm = coords_nm - centroid
    coords_nm = coords_nm * coord_mask.unsqueeze(-1)  # zero out invalid coords

    return {
        'pdb_ids': pdb_ids,
        'coords_nm': coords_nm,                    # [b, n, 37, 3] nm
        'coords': coords,                          # [b, n, 37, 3] Angstrom (for angle features)
        'coord_mask': coord_mask,                  # [b, n, 37]
        'residue_type': seq,                       # [b, n] 0-19
        'seq': seq,
        'mask': ab_mask,                           # [b, n] antibody only — FM generation mask
        'full_mask': full_mask,                    # [b, n] all valid residues (ab + ag) — attention mask
        'cdr_mask': cdr_mask,
        # Canonical CDR positions, independent of mixed training masking.
        'target_cdr_mask': target_cdr_mask,
        'cdr_loop_id': cdr_loop_id,
        'paratope_mask': paratope_mask,
        'epitope_mask': epitope_mask,
        'chain_type': chain_type,
        'ab_type': ab_type,                         # [b], 0=VH-VL, 1=VHH
        'chain_breaks_per_residue': chain_breaks,  # [b, n]
        'chains': chain_idx,                       # [b, n] for ChainIdxSeqFeat
        'mask_dict': {
            'residue_type': ab_mask.clone(),
            # [b, n, 37, 1]: AE/FM accesses [..., 0, 0] to get [b, n] ab residue mask.
            'coords': (coord_mask.any(dim=-1) & ab_mask).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 37, 1).contiguous(),
        },
    }


class AntibodyDesignDataModule(BaseLightningDataModule):
    """DataModule for antibody design training."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        num_workers: int = 16,
        pin_memory: bool = True,
        cdr_target: str = "all",
        max_antibody_len: int = 450,
        max_antigen_len: int = 500,
        train_split: str = "train",
        batch_padding: bool = True,
        mask_strategy: str = "full",
        mask_strategy_probs: Optional[List[float]] = None,
        balanced_abnb: bool = False,
        balanced_abnb_ratio: Optional[List[int]] = None,
        balanced_abnb_seed: int = 42,
    ):
        super().__init__(
            batch_padding=batch_padding,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        self.data_dir = data_dir
        self.cdr_target = cdr_target
        self.max_antibody_len = max_antibody_len
        self.max_antigen_len = max_antigen_len
        self.train_split = train_split
        self.mask_strategy = mask_strategy
        self.mask_strategy_probs = mask_strategy_probs
        self.balanced_abnb = balanced_abnb
        self.balanced_abnb_ratio = balanced_abnb_ratio or [3, 1]
        self.balanced_abnb_seed = balanced_abnb_seed

    def _get_dataset(self, split: str, training: bool = False) -> AntibodyDesignDataset:
        # Mixed masking only during training; val/test always use full mask.
        strategy = self.mask_strategy if training else "full"
        return AntibodyDesignDataset(
            data_dir=self.data_dir,
            split=split,
            cdr_target=self.cdr_target,
            max_antibody_len=self.max_antibody_len,
            max_antigen_len=self.max_antigen_len,
            mask_strategy=strategy,
            mask_strategy_probs=self.mask_strategy_probs,
        )

    def train_dataset(self) -> AntibodyDesignDataset:
        return self._get_dataset(self.train_split, training=True)

    def train_dataloader(self) -> TorchDataLoader:
        if not self.balanced_abnb:
            return super().train_dataloader()

        if self.train_ds is None:
            self.train_ds = self.train_dataset()

        batch_sampler = BalancedAbNbBatchSampler(
            ab_indices=self.train_ds.ab_indices,
            nb_indices=self.train_ds.nb_indices,
            batch_size=self.batch_size,
            ratio=tuple(self.balanced_abnb_ratio),
            drop_last=len(self.train_ds) >= self.batch_size,
            seed=self.balanced_abnb_seed,
        )
        logger.info(
            "Using balanced AB/NB batch sampler with "
            f"VH-VL:VHH ratio {self.balanced_abnb_ratio[0]}:{self.balanced_abnb_ratio[1]}"
        )
        return TorchDataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataset(self) -> AntibodyDesignDataset:
        return self._get_dataset("valid")

    def test_dataset(self) -> AntibodyDesignDataset:
        return self._get_dataset("test")


if __name__ == "__main__":
    import sys
    data_dir = "assets/structure_dataset"

    force = "--reprocess" in sys.argv

    # `--splits a,b,c` selects which splits to build (default: core splits).
    # e.g. `python -m proteinfoundation.datasets.ab_data --splits synthesis`
    splits = None
    if "--splits" in sys.argv:
        splits = sys.argv[sys.argv.index("--splits") + 1].split(",")

    process_data(data_dir, force_reprocess=force, splits=splits)

    probe_split = splits[0] if splits else "valid"
    dataset = AntibodyDesignDataset(data_dir, split=probe_split)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"coords shape: {sample['coords'].shape}")       # [n, 37, 3]
    print(f"coord_mask shape: {sample['coord_mask'].shape}")
    print(f"seq unique: {sample['seq'].unique()}")

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"batch coords_nm shape: {batch['coords_nm'].shape}")
    print(f"batch residue_type range: {batch['residue_type'].min()}-{batch['residue_type'].max()}")
    print("Success!")
