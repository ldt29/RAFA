#!/usr/bin/env python3
"""
evaluation/metrics.py
======================
Unified, repo-portable evaluation interface.

All public functions accept **plain PDB file paths** and **chain ID strings**.
No internal custom classes (AgAbComplex, Protein, …) are exposed.
Drop this file and evaluation/rmsd.py into any repo, install the required
packages, and call the functions directly.

Required packages
-----------------
    pip install biopython numpy scipy DockQ

Optional (for TMscore / LDDT):
    - `TMscore` binary in evaluation/ directory          (for calc_tmscore)
    - `lddt` binary on PATH or in evaluation/ directory  (for calc_lddt)

Public API
----------
    calc_rmsd(model_pdb, native_pdb, chain_ids, aligned=False) -> float
    calc_cdr_rmsd(model_pdb, native_pdb, chain_id, cdr_range) -> float
    calc_tmscore(model_pdb, native_pdb, chain_ids) -> float
    calc_lddt(model_pdb, native_pdb, chain_ids) -> float
    calc_dockq(model_pdb, native_pdb, chain_map) -> dict
    calc_aar(model_pdb, native_pdb, chain_id, residue_range=None) -> float
    calc_all(model_pdb, native_pdb, *, ab_chains, ag_chain,
             cdr_ranges=None, chain_map=None) -> dict
"""

from __future__ import annotations

import os
import re
import sys
import time
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB import is_aa as bpdb_is_aa
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.SeqUtils import seq1

warnings.simplefilter("ignore", BiopythonWarning)

_HERE    = Path(__file__).parent.resolve()
_PARSER  = PDBParser(QUIET=True)
_IO      = PDBIO()
_TMEXEC  = _HERE / "TMscore"
_CACHE   = _HERE / "__cache__"
_CACHE.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load(pdb_path: str | Path) -> Model:
    """Return BioPython Model[0] from a PDB file."""
    return _PARSER.get_structure("s", str(pdb_path))[0]


def _aa_residues(chain) -> list:
    """All amino-acid Residue objects from a BioPython Chain."""
    return [r for r in chain if bpdb_is_aa(r, standard=False) and r.get_id()[0] == " "]


def _chain_seq(chain) -> str:
    return "".join(seq1(r.get_resname()) for r in _aa_residues(chain))


def _chain_ca(chain) -> np.ndarray:
    """(N, 3) array of CA coords for all AA residues."""
    coords = []
    for r in _aa_residues(chain):
        if "CA" in r:
            coords.append(r["CA"].get_vector().get_array())
    return np.array(coords, dtype=float)


def _multi_chain_ca(model: Model, chain_ids: Sequence[str]) -> np.ndarray:
    """Concatenate CA coords across multiple chains."""
    parts = [_chain_ca(model[c]) for c in chain_ids if c in {ch.id for ch in model}]
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 3))


def _save_chains(model: Model, chain_ids: Sequence[str], path: str | Path):
    """Write a subset of chains from a model to a PDB file."""
    new_s = Structure("tmp")
    new_m = Model(0)
    new_s.add(new_m)
    for cid in chain_ids:
        if cid in {ch.id for ch in model}:
            new_m.add(model[cid].copy())
    _IO.set_structure(new_s)
    _IO.save(str(path))


def _kabsch(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kabsch alignment: returns (a_aligned, R, t) so that a_aligned ≈ b."""
    a_mean, b_mean = a.mean(0), b.mean(0)
    ac, bc = a - a_mean, b - b_mean
    C = ac.T @ bc
    V, _, W = np.linalg.svd(C)
    if np.linalg.det(V) * np.linalg.det(W) < 0:
        V[:, -1] *= -1
    R = V @ W
    t = b_mean - a_mean @ R
    return a @ R + t, R, t


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt((diff ** 2).sum() / len(a)))


# ---------------------------------------------------------------------------
# calc_rmsd
# ---------------------------------------------------------------------------

def calc_rmsd(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    chain_ids:  Sequence[str],
    *,
    aligned: bool = False,
) -> float:
    """
    Cα RMSD between model and native over the given chains.

    Parameters
    ----------
    model_pdb, native_pdb : paths to PDB files
    chain_ids : chains to include (e.g. ['H', 'L'])
    aligned : if True, return unaligned RMSD (raw placement);
              if False (default), Kabsch-align first (shape deviation).

    Returns
    -------
    float  RMSD in Ångström
    """
    mod  = _load(model_pdb)
    nat  = _load(native_pdb)
    a    = _multi_chain_ca(mod, chain_ids)
    b    = _multi_chain_ca(nat, chain_ids)
    if len(a) == 0 or len(b) == 0:
        raise ValueError(f"CA coord length zero: model={len(a)} native={len(b)}")
    if len(a) != len(b):
        n = min(len(a), len(b))
        _warn("RMSD", ValueError(
            f"CA coord length mismatch: model={len(a)} native={len(b)}, "
            f"computing on first {n} residues"
        ))
        a, b = a[:n], b[:n]
    if aligned:
        return _rmsd(a, b)
    a_al, _, _ = _kabsch(a, b)
    return _rmsd(a_al, b)


# ---------------------------------------------------------------------------
# calc_cdr_rmsd
# ---------------------------------------------------------------------------

def calc_cdr_rmsd(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    chain_id:   str,
    cdr_range:  Tuple[int, int],
    *,
    global_alignment_chains: Optional[Sequence[str]] = None,
    self_align: bool = False,
) -> float:
    """
    CDR Cα RMSD with three alignment modes (mutually exclusive):

    1. ``global_alignment_chains`` given  →  antibody-aligned CDR RMSD
       Kabsch-superimpose the specified chains (e.g. ['H','L']), then
       measure CDR RMSD.  Reflects CDR conformation relative to framework.

    2. ``self_align=True``  →  CDR self-aligned RMSD
       Kabsch-superimpose on the CDR atoms themselves, then measure RMSD.
       Reflects intrinsic CDR loop conformation quality.

    3. Both None / False  →  antigen-aligned CDR RMSD
       The caller is responsible for pre-aligning PDBs on the antigen
       (as done by _align_and_save).  Raw RMSD = placement in complex.

    Parameters
    ----------
    chain_id   : chain containing the CDR (e.g. 'H')
    cdr_range  : (start, end) 0-indexed inclusive residue positions within chain

    Returns
    -------
    float  CDR RMSD in Ångström
    """
    mod = _load(model_pdb)
    nat = _load(native_pdb)
    s, e = cdr_range

    def _cdr_ca(m: Model) -> np.ndarray:
        chain = m[chain_id]
        aa    = _aa_residues(chain)
        subset = aa[s : e + 1]
        return np.array([r["CA"].get_vector().get_array() for r in subset if "CA" in r])

    mod_cdr = _cdr_ca(mod)
    nat_cdr = _cdr_ca(nat)

    if len(mod_cdr) == 0 or len(nat_cdr) == 0:
        raise ValueError(f"CDR CA length zero: model={len(mod_cdr)} native={len(nat_cdr)}")
    if len(mod_cdr) != len(nat_cdr):
        n = min(len(mod_cdr), len(nat_cdr))
        _warn(f"CDR-RMSD {chain_id}", ValueError(
            f"CDR CA mismatch: model={len(mod_cdr)} native={len(nat_cdr)}, "
            f"computing on first {n} residues"
        ))
        mod_cdr = mod_cdr[:n]
        nat_cdr = nat_cdr[:n]

    if global_alignment_chains:
        # Mode 1: antibody-aligned — Kabsch on H+L (or given chains), then CDR RMSD
        mod_all = _multi_chain_ca(mod, global_alignment_chains)
        nat_all = _multi_chain_ca(nat, global_alignment_chains)
        if len(mod_all) == len(nat_all) and len(mod_all) > 0:
            _, R, t = _kabsch(mod_all, nat_all)
            mod_cdr = mod_cdr @ R + t
            return _rmsd(mod_cdr, nat_cdr)

    if self_align:
        # Mode 2: CDR self-aligned — Kabsch on CDR atoms themselves
        mod_cdr_al, _, _ = _kabsch(mod_cdr, nat_cdr)
        return _rmsd(mod_cdr_al, nat_cdr)

    # Mode 3 (default): antigen-aligned — PDB already antigen-superimposed by caller
    return _rmsd(mod_cdr, nat_cdr)


# ---------------------------------------------------------------------------
# calc_tmscore
# ---------------------------------------------------------------------------

def calc_tmscore(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    chain_ids:  Sequence[str],
) -> float:
    """
    TM-score computed by the TMscore binary (evaluation/TMscore).
    The specified chains are concatenated into a single-chain temporary PDB.

    Returns float in [0, 1].
    """
    if not _TMEXEC.exists():
        raise FileNotFoundError(f"TMscore binary not found at {_TMEXEC}")

    mod = _load(model_pdb)
    nat = _load(native_pdb)

    tag = f"{time.time():.6f}".replace(".", "")
    tmp_mod = str(_CACHE / f"tm_mod_{tag}.pdb")
    tmp_nat = str(_CACHE / f"tm_nat_{tag}.pdb")
    try:
        _save_chains(mod, chain_ids, tmp_mod)
        _save_chains(nat, chain_ids, tmp_nat)
        p    = os.popen(f"{_TMEXEC} {tmp_mod} {tmp_nat}")
        text = p.read(); p.close()
        m    = re.search(r"TM-score\s*=\s*([0-9.]+)", text)
        if not m:
            raise RuntimeError(f"TMscore output not parsed:\n{text[:500]}")
        return float(m.group(1))
    finally:
        for f in (tmp_mod, tmp_nat):
            try: os.remove(f)
            except OSError: pass


# ---------------------------------------------------------------------------
# calc_lddt
# ---------------------------------------------------------------------------

def calc_lddt(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    chain_ids:  Sequence[str],
) -> float:
    """
    Global lDDT score computed by the `lddt` binary (must be on PATH).
    The specified chains are concatenated into single-chain temporary PDBs.

    Returns float in [0, 1].
    """
    mod = _load(model_pdb)
    nat = _load(native_pdb)

    tag = f"{time.time():.6f}".replace(".", "")
    tmp_mod = str(_CACHE / f"lddt_mod_{tag}.pdb")
    tmp_nat = str(_CACHE / f"lddt_nat_{tag}.pdb")
    tmp_log = str(_CACHE / f"lddt_log_{tag}.txt")
    try:
        _save_chains(mod, chain_ids, tmp_mod)
        _save_chains(nat, chain_ids, tmp_nat)
        ret = os.system(f"lddt -x {tmp_mod} {tmp_nat} > {tmp_log} 2>&1")
        if ret != 0:
            raise RuntimeError(f"lddt binary exited with code {ret}")
        with open(tmp_log) as f:
            text = f.read()
        m = re.search(r"Global LDDT score:\s*([0-9.]+)", text)
        if not m:
            raise RuntimeError(f"lDDT output not parsed:\n{text[:500]}")
        return float(m.group(1))
    finally:
        for f in (tmp_mod, tmp_nat, tmp_log):
            try: os.remove(f)
            except OSError: pass


# ---------------------------------------------------------------------------
# _merge_chains_to_pdb  (internal helper for DockQ)
# ---------------------------------------------------------------------------

def _merge_chains_to_pdb(
    pdb_path:     str | Path,
    merge_chains: Sequence[str],
    merged_id:    str,
    ag_chain:     str,
    out_path:     str | Path,
) -> None:
    """
    Write a new PDB with *merge_chains* concatenated into a single chain
    named *merged_id*, followed by *ag_chain* unchanged.

    Residue serial numbers in the merged chain are renumbered 1…N to avoid
    conflicts between H and L.  Heteroatom / HETATM records are dropped.
    """
    model = _load(pdb_path)

    new_s = Structure("merged")
    new_m = Model(0)
    new_s.add(new_m)

    # ── Build merged antibody chain ───────────────────────────────────────────
    merged_chain = Chain(merged_id)
    serial = 1
    for cid in merge_chains:
        if cid not in {c.id for c in model}:
            continue
        for r in _aa_residues(model[cid]):
            new_r = r.copy()
            new_r.id = (" ", serial, " ")
            merged_chain.add(new_r)
            serial += 1
    new_m.add(merged_chain)

    # ── Antigen chain (unchanged) ─────────────────────────────────────────────
    if ag_chain in {c.id for c in model}:
        new_m.add(model[ag_chain].copy())

    _IO.set_structure(new_s)
    _IO.save(str(out_path))


# ---------------------------------------------------------------------------
# calc_dockq
# ---------------------------------------------------------------------------

def calc_dockq(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    ab_chains:  Sequence[str],
    ag_chain:   str,
    merged_ab_id: str = "H",
) -> dict:
    """
    DockQ (pip install DockQ) for the antibody-vs-antigen interface only.

    H and L chains (or just H for VHH) are **merged** into a single chain
    named *merged_ab_id* before scoring, so DockQ sees exactly two chains:
    merged-antibody vs antigen.  This avoids irrelevant H↔L and L↔A scores.

    Parameters
    ----------
    ab_chains    : antibody chain IDs, e.g. ["H", "L"] or ["H"]
    ag_chain     : antigen chain ID, e.g. "A"
    merged_ab_id : chain ID used for the merged antibody in the temp PDB (default "H")

    Returns
    -------
    dict with keys:
        "DockQ"      – DockQ score for the merged-antibody ↔ antigen interface
        "iRMSD"
        "LRMSD"
        "fnat"
        "fnonnat"
        "F1"
        "clashes"
    """
    from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

    tag      = f"{time.time():.6f}".replace(".", "")
    tmp_mod  = str(_CACHE / f"dockq_mod_{tag}.pdb")
    tmp_nat  = str(_CACHE / f"dockq_nat_{tag}.pdb")
    try:
        _merge_chains_to_pdb(model_pdb,  ab_chains, merged_ab_id, ag_chain, tmp_mod)
        _merge_chains_to_pdb(native_pdb, ab_chains, merged_ab_id, ag_chain, tmp_nat)

        model_s  = load_PDB(tmp_mod)
        native_s = load_PDB(tmp_nat)

        # Only two chains now: merged_ab_id and ag_chain
        chain_map = {merged_ab_id: merged_ab_id, ag_chain: ag_chain}
        iface_results, total = run_on_all_native_interfaces(
            model_s, native_s, chain_map=chain_map
        )

        # There is exactly one interface: merged_ab_id ↔ ag_chain
        out: dict = {}
        for iface in iface_results.values():
            for k in ("DockQ", "iRMSD", "LRMSD", "fnat", "fnonnat", "F1", "clashes"):
                if k in iface:
                    out[k] = iface[k]
        return out

    finally:
        for f in (tmp_mod, tmp_nat):
            try: os.remove(f)
            except OSError: pass


# ---------------------------------------------------------------------------
# calc_aar
# ---------------------------------------------------------------------------

def calc_aar(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    chain_id:   str,
    residue_range: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Amino-acid recovery rate (AAR) for a single chain (or a sub-range of it).

    Parameters
    ----------
    chain_id      : chain to compare (e.g. 'H')
    residue_range : (start, end) 0-indexed inclusive; None = whole chain

    Returns
    -------
    float  fraction of residues that match exactly (0–1)
    """
    mod = _load(model_pdb)
    nat = _load(native_pdb)

    mod_aa = _aa_residues(mod[chain_id])
    nat_aa = _aa_residues(nat[chain_id])

    if residue_range is not None:
        s, e = residue_range
        mod_aa = mod_aa[s : e + 1]
        nat_aa = nat_aa[s : e + 1]

    if len(mod_aa) == 0 or len(nat_aa) == 0:
        raise ValueError(f"Residue count zero chain {chain_id}: "
                         f"model={len(mod_aa)} native={len(nat_aa)}")
    if len(mod_aa) != len(nat_aa):
        n = min(len(mod_aa), len(nat_aa))
        _warn(f"AAR {chain_id}", ValueError(
            f"Residue count mismatch chain {chain_id}: "
            f"model={len(mod_aa)} native={len(nat_aa)}, "
            f"computing on first {n} residues"
        ))
        mod_aa = mod_aa[:n]
        nat_aa = nat_aa[:n]

    mod_seq = "".join(seq1(r.get_resname()) for r in mod_aa)
    nat_seq = "".join(seq1(r.get_resname()) for r in nat_aa)
    hits = sum(a == b for a, b in zip(mod_seq, nat_seq))
    return hits / len(nat_seq)


# ---------------------------------------------------------------------------
# calc_all  —  one-stop convenience wrapper
# ---------------------------------------------------------------------------

def calc_all(
    model_pdb:  str | Path,
    native_pdb: str | Path,
    *,
    ab_chains:  Sequence[str],
    ag_chain:   str,
    cdr_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    chain_map:  Optional[Dict[str, str]] = None,
    run_tmscore:  bool = True,
    run_lddt:     bool = True,
    run_dockq:    bool = True,
) -> dict:
    """
    Compute all metrics in one call.

    Parameters
    ----------
    model_pdb, native_pdb : PDB file paths
    ab_chains  : antibody chain IDs, e.g. ['H', 'L'] or ['H']
    ag_chain   : antigen chain ID, e.g. 'A'
    cdr_ranges : {label → (start, end)} 0-indexed inclusive, e.g.
                 {"H1": (26, 33), "H2": (50, 57), "H3": (95, 102),
                  "L1": (24, 34), "L2": (50, 56), "L3": (89, 97)}
                 Parsed from X-masked design.fasta by parse_cdr_ranges().
    chain_map  : for DockQ, {native_chain: model_chain}.
                 Defaults to identity map over ab_chains + [ag_chain].
    run_tmscore / run_lddt / run_dockq : toggle individual metrics

    Returns
    -------
    dict  flat dict of metric_name → float
    """
    results: dict = {}
    all_chains = list(ab_chains) + [ag_chain]

    # ── Pre-compute antigen CA coords for antigen-based alignment ─────────────
    # The input PDBs are already antigen-superimposed by the caller
    # (test_iggm_structure_dataset._align_and_save uses antigen chain A).
    # So "antigen-aligned RMSD" = raw distance after antigen superimposition
    #                           = aligned=True on the pre-aligned files.
    # "antibody-aligned RMSD"  = Kabsch on H+L, then measure deviation.

    # ── RMSD_ab ───────────────────────────────────────────────────────────────
    # Three alignment modes for antibody RMSD:
    #   RMSD_ab_ag  : antigen-aligned  → raw RMSD after antigen superimposition
    #                 measures antibody placement relative to antigen in complex
    #   RMSD_ab_ab  : antibody-aligned → Kabsch on H+L, then antibody RMSD
    #                 measures antibody shape / conformation deviation
    try:
        results["RMSD_ab_ag"] = calc_rmsd(model_pdb, native_pdb, ab_chains, aligned=True)
        results["RMSD_ab_ab"] = calc_rmsd(model_pdb, native_pdb, ab_chains, aligned=False)
    except Exception as exc:
        results["RMSD_ab_ag"] = results["RMSD_ab_ab"] = float("nan")
        _warn("RMSD", exc)

    # ── Per-CDR RMSD ──────────────────────────────────────────────────────────
    # Three alignment modes for each CDR:
    #   RMSD_CDR{X}_ag   : antigen-aligned (PDB pre-aligned on antigen A)
    #                      measures CDR placement in the binding interface
    #   RMSD_CDR{X}_ab   : antibody-aligned (Kabsch on H+L, then CDR RMSD)
    #                      measures CDR position relative to framework
    #   RMSD_CDR{X}_self : CDR self-aligned (Kabsch on CDR atoms themselves)
    #                      measures intrinsic CDR loop conformation quality
    if cdr_ranges:
        for label, rng in cdr_ranges.items():
            chain_id = label.lstrip("CDR-")[0]   # "H" or "L"
            if chain_id not in ab_chains:
                continue
            # antigen-aligned (default: no extra alignment, PDB already ag-superimposed)
            try:
                results[f"RMSD_CDR{label}_ag"] = calc_cdr_rmsd(
                    model_pdb, native_pdb, chain_id, rng,
                    global_alignment_chains=None, self_align=False,
                )
            except Exception as exc:
                results[f"RMSD_CDR{label}_ag"] = float("nan")
                _warn(f"CDR-RMSD {label} ag", exc)
            # antibody-aligned: Kabsch on H+L first
            try:
                results[f"RMSD_CDR{label}_ab"] = calc_cdr_rmsd(
                    model_pdb, native_pdb, chain_id, rng,
                    global_alignment_chains=ab_chains, self_align=False,
                )
            except Exception as exc:
                results[f"RMSD_CDR{label}_ab"] = float("nan")
                _warn(f"CDR-RMSD {label} ab", exc)
            # CDR self-aligned: Kabsch on CDR atoms themselves
            try:
                results[f"RMSD_CDR{label}_self"] = calc_cdr_rmsd(
                    model_pdb, native_pdb, chain_id, rng,
                    global_alignment_chains=None, self_align=True,
                )
            except Exception as exc:
                results[f"RMSD_CDR{label}_self"] = float("nan")
                _warn(f"CDR-RMSD {label} self", exc)

    # ── AAR ──────────────────────────────────────────────────────────────────
    # Reported at four granularities:
    #   AAR_{chain}         per full chain   (e.g. AAR_H, AAR_L)
    #   AAR_CDR{X}          per CDR          (e.g. AAR_CDRH1 … AAR_CDRL3)
    #   AAR_CDRH / AAR_CDRL all H-CDRs / all L-CDRs concatenated
    #   AAR_CDR             all CDRs (H+L) concatenated
    #   AAR_ab              entire antibody (H+L) concatenated
    mod = _load(model_pdb)
    nat = _load(native_pdb)

    for cid in ab_chains:
        # whole chain
        try:
            results[f"AAR_{cid}"] = calc_aar(model_pdb, native_pdb, cid)
        except Exception as exc:
            results[f"AAR_{cid}"] = float("nan")
            _warn(f"AAR {cid}", exc)

        # per-CDR
        if cdr_ranges:
            for label, rng in cdr_ranges.items():
                if label.lstrip("CDR-")[0] != cid:
                    continue
                try:
                    results[f"AAR_CDR{label}"] = calc_aar(model_pdb, native_pdb, cid, rng)
                except Exception as exc:
                    results[f"AAR_CDR{label}"] = float("nan")
                    _warn(f"AAR CDR {label}", exc)

    # aggregate AAR helpers
    def _concat_seqs(model_flag: bool, chain_id: str, ranges: list) -> str:
        m = mod if model_flag else nat
        try:
            aa = _aa_residues(m[chain_id])
        except KeyError:
            return ""
        parts = []
        for s, e in ranges:
            parts.append("".join(seq1(r.get_resname()) for r in aa[s: e + 1]))
        return "".join(parts)

    def _aar_from_seqs(mod_seq: str, nat_seq: str) -> float | None:
        n = min(len(mod_seq), len(nat_seq))
        if n == 0:
            return None
        return sum(a == b for a, b in zip(mod_seq[:n], nat_seq[:n])) / n

    if cdr_ranges:
        for cid, label_prefix, key_suffix in [
            (None,  None, "CDR"),          # all CDRs H+L  → AAR_CDR
            ("H",   "H",  "CDRH"),         # H CDRs only   → AAR_CDRH
            ("L",   "L",  "CDRL"),         # L CDRs only   → AAR_CDRL
        ]:
            target_chains = ab_chains if cid is None else ([cid] if cid in ab_chains else [])
            if not target_chains:
                continue
            mod_seq, nat_seq = "", ""
            for tc in target_chains:
                cdrs_for_chain = [
                    rng for lbl, rng in cdr_ranges.items()
                    if lbl.lstrip("CDR-")[0] == tc
                ]
                mod_seq += _concat_seqs(True,  tc, cdrs_for_chain)
                nat_seq += _concat_seqs(False, tc, cdrs_for_chain)
            v = _aar_from_seqs(mod_seq, nat_seq)
            if v is not None:
                results[f"AAR_{key_suffix}"] = v

        # AAR_ab: entire antibody H+L
        mod_ab, nat_ab = "", ""
        for cid in ab_chains:
            try:
                mod_ab += "".join(seq1(r.get_resname()) for r in _aa_residues(mod[cid]))
                nat_ab += "".join(seq1(r.get_resname()) for r in _aa_residues(nat[cid]))
            except KeyError:
                pass
        v = _aar_from_seqs(mod_ab, nat_ab)
        if v is not None:
            results["AAR_ab"] = v

    # ── TMscore ──────────────────────────────────────────────────────────────
    if run_tmscore:
        try:
            results["TMscore"] = calc_tmscore(model_pdb, native_pdb, ab_chains)
        except Exception as exc:
            results["TMscore"] = float("nan")
            _warn("TMscore", exc)

    # ── lDDT ─────────────────────────────────────────────────────────────────
    if run_lddt:
        try:
            results["LDDT"] = calc_lddt(model_pdb, native_pdb, ab_chains)
        except Exception as exc:
            results["LDDT"] = float("nan")
            _warn("LDDT", exc)

    # ── DockQ: merged antibody (H+L) vs antigen ──────────────────────────────
    # H and L are concatenated into one chain before scoring so DockQ reports
    # only the antibody↔antigen interface, not H↔L or L↔A spurious scores.
    #
    # DockQ quality thresholds (Bbenchmark5 convention):
    #   Incorrect  : DockQ < 0.23
    #   Acceptable : 0.23 ≤ DockQ < 0.49
    #   Medium     : 0.49 ≤ DockQ < 0.80
    #   High       : DockQ ≥ 0.80
    #
    # SR (Success Rate) binary flags are stored per sample (0.0 or 1.0) so
    # their mean across samples equals the population success rate directly.
    if run_dockq:
        try:
            dq = calc_dockq(model_pdb, native_pdb,
                            ab_chains=ab_chains, ag_chain=ag_chain)
            for k, v in dq.items():
                results[f"DockQ_{k}"] = v
            dq_score = dq.get("DockQ", float("nan"))
            if not (isinstance(dq_score, float) and np.isnan(dq_score)):
                results["DockQ_SR"]        = 1.0 if dq_score >= 0.23 else 0.0
                results["DockQ_SR_medium"] = 1.0 if dq_score >= 0.49 else 0.0
                results["DockQ_SR_high"]   = 1.0 if dq_score >= 0.80 else 0.0
        except Exception as exc:
            results["DockQ_DockQ"]     = float("nan")
            results["DockQ_SR"]        = float("nan")
            results["DockQ_SR_medium"] = float("nan")
            results["DockQ_SR_high"]   = float("nan")
            _warn("DockQ", exc)

    return results


# ---------------------------------------------------------------------------
# Utility: parse CDR ranges from X-masked FASTA
# ---------------------------------------------------------------------------

def parse_cdr_ranges(design_fasta: str | Path) -> Dict[str, Tuple[int, int]]:
    """
    Extract CDR (start, end) ranges (0-indexed, inclusive) from an X-masked
    FASTA file produced by the structure_dataset pipeline.

    Each contiguous run of 'X' in H or L is taken as CDR1/CDR2/CDR3 in order.

    Returns
    -------
    dict  e.g. {"H1": (26,33), "H2": (50,57), "H3": (95,102),
                "L1": (24,34), "L2": (50,56), "L3": (89,97)}
    """
    seqs: Dict[str, str] = {}
    cid, parts = None, []
    for line in Path(design_fasta).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cid:
                seqs[cid] = "".join(parts)
            cid = line[1:].split()[0]
            parts = []
        else:
            parts.append(line)
    if cid:
        seqs[cid] = "".join(parts)

    out: Dict[str, Tuple[int, int]] = {}
    for chain_id in ("H", "L"):
        seq = seqs.get(chain_id, "")
        if not seq:
            continue
        blocks, in_x, s = [], False, -1
        for i, ch in enumerate(seq):
            if ch.upper() == "X":
                if not in_x:
                    in_x = True; s = i
            else:
                if in_x:
                    in_x = False; blocks.append((s, i - 1))
        if in_x:
            blocks.append((s, len(seq) - 1))
        for idx, block in enumerate(blocks[:3], start=1):
            out[f"{chain_id}{idx}"] = block
    return out


# ---------------------------------------------------------------------------
# Internal logging
# ---------------------------------------------------------------------------

def _warn(metric: str, exc: Exception):
    print(f"  [WARNING] {metric}: {exc}", file=sys.stderr)