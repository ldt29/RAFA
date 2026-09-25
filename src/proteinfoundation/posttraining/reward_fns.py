"""
Reward functions for GRPO-style RL post-training of the antibody FM model.

Tier 1 — fnat_proxy_reward
    CA-level contact recovery rate (H+L vs Epitope).
    Pure torch, fully batched, no AE decode needed.
    Threshold: 8 Å (0.8 nm), consistent with DockQ fnat convention.

Tier 2 — ca_dockq_proxy_reward
    Full DockQ-like score using CA coordinates only.
    DockQ = (fnat + iRMS_score + LRMS_score) / 3 − λ_clash * clash

    Key simplification (antigen-locked training):
      The FM model locks the antigen CA to ground-truth at every ODE step.
      This means the generated antibody already lives in the GT coordinate frame.
      Consequence: LRMS and iRMS are direct RMSDs — NO Kabsch alignment needed.

      iRMS:  RMSD of CDR (interface) CA atoms vs GT CDR CA atoms
             d_ref = 1.5 Å = 0.15 nm  (DockQ convention)
             iRMS_score = 1 / (1 + (iRMS / d_ref)²)

      LRMS:  RMSD of all antibody CA atoms vs GT antibody CA atoms
             d_ref = 8.5 Å = 0.85 nm  (DockQ convention)
             LRMS_score = 1 / (1 + (LRMS / d_ref)²)

    Clash penalty:
      CA–CA distance < 3.5 Å = 0.35 nm between predicted antibody and true antigen.
      clash = mean_per_residue(relu(0.35 - dist(pred_ab_ca, true_ag_ca)))

Tier 3 — full_dockq_reward (stub)
    Requires AE decode → full-atom structure → external DockQ subprocess.
    Placeholder — implement when ready for Phase 3.

Affinity-related proxy — interface_affinity_proxy_reward
    Ground-truth-free, differentiable interface-geometry terms computed from the
    generated antibody and the fixed antigen/epitope input.  The proxy combines
    contact coverage, a buried-interface proxy, interface compactness, a steric
    gate, and an antibody globularity prior.  It is explicitly a structural
    affinity-related proxy, not a measured binding affinity or an ipTM/pAE value.

Combined/ablation reward — ca_dockq_affinity_proxy_reward
    The prospective RL reward is affinity-primary by default: it uses only the
    native-antibody-free interface-affinity proxy.  A CA-DockQ mixture remains
    available only when an explicit affinity weight below 1.0 is requested for
    a historical/diagnostic ablation; it is never evaluated on the main path.

R_free composite reward
    Affinity-primary proxy plus an optional independent sequence-affinity
    teacher, a train-only-calibrated Protenix fold-confidence surrogate,
    steric/physical terms, and a small sequence-only developability screen.
    DockQ is not evaluated on this path.

All reward functions share the same signature:
    reward_fn(x_1_pred: Dict, batch: Dict) -> Tensor [B]

where x_1_pred = {"bb_ca": [B, N, 3], "local_latents": [B, N, 8]}
and batch contains at minimum: "mask" (ab_mask), "chain_type", "epitope_mask",
"native_cdr_mask" (or legacy "cdr_mask"), and "coords_nm".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Constants (DockQ paper conventions)
# ─────────────────────────────────────────────────────────────────────────────

CONTACT_THRESHOLD_NM = 0.8    # 8 Å  — fnat contact threshold
IRMS_DREF_NM         = 0.15   # 1.5 Å — iRMS denominator (interface RMSD)
LRMS_DREF_NM         = 0.85   # 8.5 Å — LRMS denominator (ligand RMSD)
CLASH_THRESHOLD_NM   = 0.35   # 3.5 Å — CA–CA clash distance

# Differentiable interface-affinity proxy conventions.  These are deliberately
# explicit in the source so a formal run cannot silently inherit a predictor
# confidence score or an unpinned energy implementation.
AFFINITY_CONTACT_CUTOFF_NM = 0.80       # 8 Å interface shell
AFFINITY_CONTACT_SHARPNESS = 12.0       # per nm, soft contact transition
AFFINITY_BURIED_RADIUS_NM = 0.80        # 8 Å shell for buried-interface proxy
AFFINITY_BURIED_TARGET = 12.0            # soft contacting residues at saturation
AFFINITY_COMPACTNESS_SCALE_NM = 1.20    # expected interface patch scale
AFFINITY_CLASH_DISTANCE_NM = 0.35       # 3.5 Å steric gate
AFFINITY_CLASH_SHARPNESS = 12.0         # per nm
AFFINITY_CLASH_TOLERANCE = 0.05         # acceptable soft clash fraction
AFFINITY_CLASH_GATE_STRENGTH = 12.0
AFFINITY_RG_EXPECTED_NM = 1.80           # antibody CA radius of gyration prior
AFFINITY_RG_TOLERANCE_NM = 0.60
# Optional soft geometry penalty.  The default is zero so historical and
# currently queued arms retain their exact reward semantics.  A future
# geometry-corrected arm can penalize out-of-calibration radius without
# multiplying the whole affinity signal by a near-zero prior.
AFFINITY_RADIUS_PENALTY_WEIGHT = 0.0
DEFAULT_AFFINITY_WEIGHTS = {
    "contact_coverage": 0.60,
    "buried_surface_proxy": 0.25,
    "interface_compactness": 0.15,
}

# OpenFold's residue vocabulary is the one used by ``ab_data.py`` and by the
# frozen autoencoder.  The lightweight experimental-affinity teacher predates
# that pipeline and stores features in alphabetical order; R_free remaps
# decoded residues explicitly before invoking it.  Keeping the mapping here
# makes the provenance of the sequence signal auditable instead of relying on
# an implicit vocabulary coincidence.
OPENFOLD_TO_AFFINITY_TEACHER = (0, 14, 11, 2, 1, 13, 3, 5, 6, 7,
                                9, 8, 10, 4, 12, 15, 16, 18, 19, 17)

R_FREE_DEFAULT_WEIGHTS = {
    "affinity": 0.60,
    "fold_confidence": 0.20,
    "physical": 0.15,
    "developability": 0.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_masks(batch: Dict) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Extract ab_mask, epitope_mask, cdr_mask from batch.

    Returns:
        ab_mask:      [B, N] bool — antibody residues (H+L, chain_type 1 or 2)
        epitope_mask: [B, N] bool — antigen epitope residues
        cdr_mask:     [B, N] bool or None — CDR residues (subset of ab_mask)
    """
    ab_mask      = batch["mask"].bool()                  # [B, N]
    epitope_mask = batch.get("epitope_mask")             # [B, N] or None
    cdr_mask     = batch.get("cdr_mask")                 # [B, N] or None

    if epitope_mask is not None:
        epitope_mask = epitope_mask.bool()
    if cdr_mask is not None:
        cdr_mask = cdr_mask.bool()

    return ab_mask, epitope_mask, cdr_mask


def _get_true_ca(batch: Dict) -> Tensor:
    """
    Extract ground-truth CA coordinates (nm) from batch.

    batch["coords_nm"]: [B, N, 37, 3], CA = index 1.
    """
    coords_nm = batch["coords_nm"]
    if coords_nm.dim() == 3:
        coords_nm = coords_nm.unsqueeze(0)
    return coords_nm[:, :, 1, :]  # [B, N, 3]


def _safe_pairwise_distance(lhs: Tensor, rhs: Tensor) -> Tensor:
    """Pairwise Euclidean distance with a finite zero-distance gradient."""
    delta = lhs.unsqueeze(-2) - rhs.unsqueeze(-3)
    return delta.square().sum(dim=-1).add(1e-8).sqrt()


def _get_antigen_mask(batch: Dict) -> Tensor:
    """
    Return antigen (chain_type == 3) boolean mask [B, N].

    Falls back to epitope_mask if chain_type is not available.
    """
    chain_type = batch.get("chain_type")
    if chain_type is not None:
        return (chain_type == 3)  # [B, N] bool
    # Fallback: use epitope_mask as approximate antigen mask
    epi = batch.get("epitope_mask")
    if epi is not None:
        return epi.bool()
    raise KeyError("batch must contain 'chain_type' or 'epitope_mask' for antigen masking")


def kabsch_rmsd(pred: Tensor, true: Tensor) -> Tensor:
    """
    Kabsch-aligned RMSD between two sets of CA coordinates.

    Uses SVD-based optimal rotation alignment (Kabsch algorithm).
    Both inputs are centered before alignment.

    NOTE: In the antibody FM setting the antigen is locked to GT at every ODE
    step, so generated antibody CAs are already in the GT frame.  Kabsch is
    therefore only needed when computing iRMS/LRMS from an *external* (non-locked)
    model.  The helper is kept for generality.

    Args:
        pred: [B, M, 3] predicted coords (nm)
        true: [B, M, 3] ground-truth coords (nm)

    Returns:
        rmsd: [B] in nm
    """
    B, M, _ = pred.shape
    if M == 0:
        return torch.zeros(B, device=pred.device, dtype=pred.dtype)

    # Center
    pred_c = pred - pred.mean(dim=1, keepdim=True)  # [B, M, 3]
    true_c = true - true.mean(dim=1, keepdim=True)  # [B, M, 3]

    # Covariance matrix H = pred_c^T @ true_c
    H = torch.bmm(pred_c.transpose(1, 2), true_c)   # [B, 3, 3]

    # SVD
    try:
        U, S, Vt = torch.linalg.svd(H)               # U:[B,3,3] S:[B,3] Vt:[B,3,3]
    except Exception:
        # Fallback: return unaligned RMSD
        diff = pred_c - true_c
        return diff.pow(2).sum(-1).mean(-1).sqrt()

    # Ensure proper rotation (det correction)
    d = torch.linalg.det(torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2)))  # [B]
    sign_mat = torch.eye(3, device=pred.device, dtype=pred.dtype).unsqueeze(0).expand(B, -1, -1).clone()
    sign_mat[:, 2, 2] = d.sign()

    # Optimal rotation
    R = torch.bmm(Vt.transpose(1, 2), torch.bmm(sign_mat, U.transpose(1, 2)))  # [B, 3, 3]

    # Rotate predicted
    pred_rot = torch.bmm(pred_c, R.transpose(1, 2))  # [B, M, 3]

    # RMSD
    diff = pred_rot - true_c
    rmsd = diff.pow(2).sum(-1).mean(-1).sqrt()       # [B]
    return rmsd


def _direct_rmsd(pred: Tensor, true: Tensor) -> Tensor:
    """
    Direct (no alignment) RMSD between two sets of CA coordinates.

    Valid when both sets are already in the same reference frame, which is
    guaranteed by the antigen-locked ODE in our FM model.

    Args:
        pred: [B, M, 3] predicted coords (nm)
        true: [B, M, 3] ground-truth coords (nm)

    Returns:
        rmsd: [B] in nm
    """
    B, M, _ = pred.shape
    if M == 0:
        return torch.zeros(B, device=pred.device, dtype=pred.dtype)
    diff = pred - true                              # [B, M, 3]
    return diff.pow(2).sum(-1).mean(-1).sqrt()      # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1: fnat proxy
# ─────────────────────────────────────────────────────────────────────────────

def fnat_proxy_reward(x_1_pred: Dict, batch: Dict) -> Tensor:
    """
    CA-level contact recovery rate (fnat proxy).

    Computes the fraction of ground-truth H+L × Epitope contacts
    that are also present in the predicted antibody structure.

    DockQ fnat threshold: 8 Å = 0.8 nm.
    Predicted contacts use predicted Ab CA vs true Ag CA (antigen is fixed).

    Args:
        x_1_pred: {"bb_ca": [B, N, 3], ...}  — predicted clean sample (nm)
        batch:    training batch dict

    Returns:
        fnat: [B] ∈ [0, 1]
    """
    ab_mask, epitope_mask, _ = _get_masks(batch)

    if epitope_mask is None or not epitope_mask.any():
        return torch.zeros(ab_mask.shape[0], device=ab_mask.device)

    ca_true = _get_true_ca(batch)                             # [B, N, 3] nm
    ca_pred_ab = x_1_pred["bb_ca"]                           # [B, N, 3] nm

    # Mask out non-relevant positions (zero them so cdist still works)
    ca_true_ab  = ca_true  * ab_mask.float().unsqueeze(-1)   # [B, N, 3]
    ca_true_epi = ca_true  * epitope_mask.float().unsqueeze(-1)  # [B, N, 3]
    ca_pred_ab_ = ca_pred_ab * ab_mask.float().unsqueeze(-1) # [B, N, 3]

    # gt_dists[b, i, j] = dist(true_ab[b,i], true_epi[b,j])
    gt_dists   = torch.cdist(ca_true_ab, ca_true_epi)         # [B, N, N]
    pred_dists = torch.cdist(ca_pred_ab_, ca_true_epi)        # [B, N, N]

    # Boolean contact masks
    gt_contact   = gt_dists   < CONTACT_THRESHOLD_NM   # [B, N, N]
    pred_contact = pred_dists < CONTACT_THRESHOLD_NM   # [B, N, N]

    # Restrict to valid ab × epi pairs
    pair_valid = (
        ab_mask.unsqueeze(-1) & epitope_mask.unsqueeze(-2)
    )  # [B, N, N]

    gt_contact   = gt_contact   & pair_valid
    pred_contact = pred_contact & pair_valid

    n_gt  = gt_contact.float().sum(dim=(-2, -1)).clamp(min=1.0)   # [B]
    n_hit = (gt_contact & pred_contact).float().sum(dim=(-2, -1)) # [B]

    fnat = n_hit / n_gt  # [B] ∈ [0, 1]
    return fnat


# ─────────────────────────────────────────────────────────────────────────────
# iRMS component
# ─────────────────────────────────────────────────────────────────────────────

def irms_score(x_1_pred: Dict, batch: Dict) -> Tensor:
    """
    Interface RMSD score using CDR CA atoms.

    iRMS = RMSD of CDR (interface) CA atoms.
    Since the antigen is locked to GT at every ODE step, the generated
    antibody lives in the GT reference frame → no alignment needed.

    iRMS_score = 1 / (1 + (iRMS / d_ref)²),  d_ref = 1.5 Å = 0.15 nm

    Args:
        x_1_pred: {"bb_ca": [B, N, 3], ...}
        batch:    training batch dict (needs "cdr_mask")

    Returns:
        score: [B] ∈ (0, 1]  (1.0 = perfect iRMS, 0.5 = iRMS equals d_ref)
    """
    _, _, cdr_mask = _get_masks(batch)
    B = x_1_pred["bb_ca"].shape[0]
    device = x_1_pred["bb_ca"].device

    if cdr_mask is None or not cdr_mask.any():
        return torch.ones(B, device=device)

    ca_true    = _get_true_ca(batch)    # [B, N, 3]
    ca_pred_ab = x_1_pred["bb_ca"]     # [B, N, 3]

    # Gather CDR positions — assume same CDR layout across batch
    cdr_idx = cdr_mask[0].nonzero(as_tuple=False).squeeze(-1)  # [N_cdr]
    if cdr_idx.numel() == 0:
        return torch.ones(B, device=device)

    ca_pred_cdr = ca_pred_ab[:, cdr_idx, :]  # [B, N_cdr, 3]
    ca_true_cdr = ca_true[:, cdr_idx, :]     # [B, N_cdr, 3]

    # Direct RMSD (antigen locked → frame aligned)
    irms = _direct_rmsd(ca_pred_cdr, ca_true_cdr)          # [B] nm
    score = 1.0 / (1.0 + (irms / IRMS_DREF_NM).pow(2))    # [B] ∈ (0, 1]
    return score


# ─────────────────────────────────────────────────────────────────────────────
# LRMS component
# ─────────────────────────────────────────────────────────────────────────────

def lrms_score(x_1_pred: Dict, batch: Dict) -> Tensor:
    """
    Ligand RMSD score using all antibody CA atoms.

    LRMS = RMSD of entire H+L antibody CA atoms vs GT.
    Since the antigen is locked to GT at every ODE step, no alignment needed.

    LRMS_score = 1 / (1 + (LRMS / d_ref)²),  d_ref = 8.5 Å = 0.85 nm

    Args:
        x_1_pred: {"bb_ca": [B, N, 3], ...}
        batch:    training batch dict (needs "mask" for ab_mask)

    Returns:
        score: [B] ∈ (0, 1]  (1.0 = perfect LRMS, 0.5 = LRMS equals d_ref)
    """
    ab_mask, _, _ = _get_masks(batch)
    B = x_1_pred["bb_ca"].shape[0]
    device = x_1_pred["bb_ca"].device

    if not ab_mask.any():
        return torch.ones(B, device=device)

    ca_true    = _get_true_ca(batch)    # [B, N, 3]
    ca_pred_ab = x_1_pred["bb_ca"]     # [B, N, 3]

    # Gather antibody positions — use first sample's mask as layout reference
    ab_idx = ab_mask[0].nonzero(as_tuple=False).squeeze(-1)  # [N_ab]
    if ab_idx.numel() == 0:
        return torch.ones(B, device=device)

    ca_pred_ab_ = ca_pred_ab[:, ab_idx, :]  # [B, N_ab, 3]
    ca_true_ab_ = ca_true[:, ab_idx, :]     # [B, N_ab, 3]

    # Direct RMSD (antigen locked → frame aligned)
    lrms = _direct_rmsd(ca_pred_ab_, ca_true_ab_)          # [B] nm
    score = 1.0 / (1.0 + (lrms / LRMS_DREF_NM).pow(2))   # [B] ∈ (0, 1]
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Clash penalty
# ─────────────────────────────────────────────────────────────────────────────

def clash_penalty(x_1_pred: Dict, batch: Dict) -> Tensor:
    """
    CA-level steric clash penalty between predicted antibody and true antigen.

    Counts soft violations where CA–CA distance < 3.5 Å (0.35 nm).
    The penalty is the sum of relu(threshold − dist) over all ab × ag pairs,
    normalized by the number of antibody residues to be scale-invariant.

    Since the antigen is locked to GT, we compare predicted Ab CAs vs
    true Ag CAs directly.

    Args:
        x_1_pred: {"bb_ca": [B, N, 3], ...}
        batch:    training batch dict

    Returns:
        clash: [B] ≥ 0  (0 = no clashes; add to reward as a *negative* term)
    """
    ab_mask, _, _ = _get_masks(batch)
    B = x_1_pred["bb_ca"].shape[0]
    device = x_1_pred["bb_ca"].device

    try:
        ag_mask = _get_antigen_mask(batch)
    except KeyError:
        return torch.zeros(B, device=device)

    if not ab_mask.any() or not ag_mask.any():
        return torch.zeros(B, device=device)

    ca_true    = _get_true_ca(batch)          # [B, N, 3]
    ca_pred_ab = x_1_pred["bb_ca"]           # [B, N, 3]

    ab_idx = ab_mask[0].nonzero(as_tuple=False).squeeze(-1)  # [N_ab]
    ag_idx = ag_mask[0].nonzero(as_tuple=False).squeeze(-1)  # [N_ag]

    if ab_idx.numel() == 0 or ag_idx.numel() == 0:
        return torch.zeros(B, device=device)

    pred_ab_ca = ca_pred_ab[:, ab_idx, :]   # [B, N_ab, 3]
    true_ag_ca = ca_true[:, ag_idx, :]      # [B, N_ag, 3]

    # Pairwise distances [B, N_ab, N_ag]
    dists = torch.cdist(pred_ab_ca, true_ag_ca)

    # Soft clash: relu(threshold - dist) → positive only when dist < threshold
    raw_clash = F.relu(CLASH_THRESHOLD_NM - dists)   # [B, N_ab, N_ag]

    # Sum over pairs, normalize by N_ab for residue-count invariance
    n_ab = float(ab_idx.numel())
    clash = raw_clash.sum(dim=(-2, -1)) / n_ab        # [B]
    return clash


# ─────────────────────────────────────────────────────────────────────────────
# Reference-guided structural reward (train/validation only)
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_LDDT_RADIUS_NM = 1.50
REFERENCE_LDDT_THRESHOLDS_NM = (0.05, 0.10, 0.20, 0.40)
REFERENCE_LDDT_SIGMA_NM = 0.02
REFERENCE_CONTACT_THRESHOLD_NM = 0.80
REFERENCE_CONTACT_SIGMA_NM = 0.05
REFERENCE_IRMS_DREF_NM = 0.15


def _reference_masks(batch: Dict) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return antibody, native CDR, antigen and valid-residue masks.

    These helpers intentionally consume the native antibody coordinates only
    for a declared reference-guided train/validation reward.  They must not be
    used for the native-antibody-free test-time reward path.
    """
    ab_mask = batch["mask"].bool()
    native_cdr = batch.get("native_cdr_mask", batch.get("cdr_mask"))
    if native_cdr is None:
        native_cdr = ab_mask
    native_cdr = native_cdr.bool() & ab_mask

    chain_type = batch.get("chain_type")
    if chain_type is not None:
        ag_mask = chain_type.eq(3)
    else:
        epitope = batch.get("epitope_mask")
        if epitope is None:
            raise KeyError("reference reward needs chain_type or epitope_mask")
        ag_mask = epitope.bool()

    full_mask = batch.get("full_mask")
    if full_mask is None:
        full_mask = ab_mask | ag_mask
    else:
        full_mask = full_mask.bool()
    return ab_mask, native_cdr, ag_mask, full_mask


def reference_tm_proxy(x_1_pred: Dict, batch: Dict) -> Tensor:
    """Differentiable CA-level TM-score proxy against the train reference.

    The antigen-locked flow keeps generated coordinates in the reference
    frame, so this deliberately uses direct antibody CA distances.  It is a
    smooth TM-like score, not the external TMscore executable.
    """
    pred = x_1_pred["bb_ca"]
    true = _get_true_ca(batch)
    ab_mask, _, _, _ = _reference_masks(batch)

    diff = (pred - true).pow(2).sum(dim=-1).add(1e-8).sqrt()
    length = ab_mask.float().sum(dim=-1).clamp_min(1.0)
    # Standard TM-score d0 in Angstrom, converted to nm and bounded for short
    # antibody chains.  The length is detached because it is a data constant.
    d0_ang = 1.24 * (length.detach() - 15.0).clamp_min(1.0).pow(1.0 / 3.0) - 1.8
    # Keep a 1 Å (=0.10 nm) floor for short synthetic/unit-test chains; the
    # standard antibody lengths use the usual length-dependent value.
    d0_nm = d0_ang.clamp_min(1.0) / 10.0
    per_residue = 1.0 / (1.0 + (diff / d0_nm.unsqueeze(-1)).pow(2))
    denom = ab_mask.float().sum(dim=-1).clamp_min(1.0)
    return (per_residue * ab_mask.float()).sum(dim=-1) / denom


def reference_lddt_proxy(x_1_pred: Dict, batch: Dict) -> Tensor:
    """Differentiable CA-local-distance proxy for lDDT.

    For native antibody CA pairs within 15 Å, the score is the soft fraction
    of pairs whose predicted distance error is below the four lDDT thresholds
    (0.5, 1, 2 and 4 Å).  The hard native neighborhood is a fixed train
    label; only the generated coordinates receive gradients.
    """
    pred = x_1_pred["bb_ca"]
    true = _get_true_ca(batch)
    ab_mask, _, _, _ = _reference_masks(batch)
    pair_mask = ab_mask.unsqueeze(-1) & ab_mask.unsqueeze(-2)
    eye = torch.eye(pred.shape[1], device=pred.device, dtype=torch.bool).unsqueeze(0)
    pair_mask = pair_mask & ~eye

    true_dist = _safe_pairwise_distance(true, true)
    pred_dist = _safe_pairwise_distance(pred, pred)
    pair_mask = pair_mask & (true_dist <= REFERENCE_LDDT_RADIUS_NM)
    delta = (pred_dist - true_dist).abs().unsqueeze(-1)
    thresholds = torch.as_tensor(
        REFERENCE_LDDT_THRESHOLDS_NM,
        device=pred.device,
        dtype=pred.dtype,
    ).view(1, 1, 1, -1)
    soft_hits = torch.sigmoid(
        (thresholds - delta) / REFERENCE_LDDT_SIGMA_NM
    ).mean(dim=-1)
    denom = pair_mask.float().sum(dim=(-1, -2)).clamp_min(1.0)
    return (soft_hits * pair_mask.float()).sum(dim=(-1, -2)) / denom


def reference_dockq_proxy(
    x_1_pred: Dict,
    batch: Dict,
    *,
    return_terms: bool = False,
) -> Tensor | tuple[Tensor, Dict[str, Tensor]]:
    """Differentiable native-reference CA-DockQ proxy.

    This is intentionally named a proxy: it uses soft native-contact recall,
    direct CDR iRMS and whole-antibody LRMS in the antigen-locked coordinate
    frame.  It is not the external full-atom DockQ implementation.
    """
    pred = x_1_pred["bb_ca"]
    true = _get_true_ca(batch)
    ab_mask, native_cdr, ag_mask, _ = _reference_masks(batch)

    pair_mask = ab_mask.unsqueeze(-1) & ag_mask.unsqueeze(-2)
    true_dist = _safe_pairwise_distance(true, true)
    pred_dist = _safe_pairwise_distance(pred, true)
    native_contact = (true_dist <= REFERENCE_CONTACT_THRESHOLD_NM) & pair_mask
    soft_contact = torch.sigmoid(
        (REFERENCE_CONTACT_THRESHOLD_NM - pred_dist) / REFERENCE_CONTACT_SIGMA_NM
    ) * pair_mask.float()

    n_native = native_contact.float().sum(dim=(-1, -2)).clamp_min(1.0)
    fnat = (soft_contact * native_contact.float()).sum(dim=(-1, -2)) / n_native
    n_pred = soft_contact.sum(dim=(-1, -2)).clamp_min(1.0)
    precision = (soft_contact * native_contact.float()).sum(dim=(-1, -2)) / n_pred

    diff_sq = (pred - true).pow(2).sum(dim=-1)
    n_cdr = native_cdr.float().sum(dim=-1).clamp_min(1.0)
    n_ab = ab_mask.float().sum(dim=-1).clamp_min(1.0)
    irms = (diff_sq * native_cdr.float()).sum(dim=-1).div(n_cdr).add(1e-8).sqrt()
    lrms = (diff_sq * ab_mask.float()).sum(dim=-1).div(n_ab).add(1e-8).sqrt()
    irms_score_value = 1.0 / (1.0 + (irms / REFERENCE_IRMS_DREF_NM).pow(2))
    lrms_score_value = 1.0 / (1.0 + (lrms / LRMS_DREF_NM).pow(2))
    dockq = (fnat + irms_score_value + lrms_score_value) / 3.0

    if not return_terms:
        return dockq
    terms = {
        "reference_fnat": fnat,
        "reference_contact_precision": precision,
        "reference_irms_score": irms_score_value,
        "reference_lrms_score": lrms_score_value,
        "reference_dockq_proxy": dockq,
    }
    return dockq, terms


def reference_structure_scores(x_1_pred: Dict, batch: Dict) -> Dict[str, Tensor]:
    """Compute all smooth reference-guided structural scores."""
    dockq, dockq_terms = reference_dockq_proxy(x_1_pred, batch, return_terms=True)
    scores = {
        "reference_lddt": reference_lddt_proxy(x_1_pred, batch),
        "reference_tm": reference_tm_proxy(x_1_pred, batch),
        **dockq_terms,
    }
    # Clash-free is kept as a separate normalized score so the all-mode can
    # address the observed interface-clash regression without double-counting
    # the native contact terms.
    clash = clash_penalty(x_1_pred, batch)
    scores["reference_clash_free"] = 1.0 - (clash / 0.50).clamp(0.0, 1.0)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Full CA-DockQ proxy (fnat + iRMS + LRMS − clash)
# ─────────────────────────────────────────────────────────────────────────────

def ca_dockq_proxy_reward(
    x_1_pred: Dict,
    batch: Dict,
    clash_weight: float = 0.1,
) -> Tensor:
    """
    Full CA-DockQ proxy reward.

    DockQ ≈ (fnat + iRMS_score + LRMS_score) / 3  ∈ [0, 1]

    With clash penalty:
      reward = DockQ − clash_weight * clash_penalty

    Key insight (antigen-locked FM):
      The antigen CA is restored to GT at every ODE step.  All generated
      structures exist in the GT reference frame.  Therefore:
        • iRMS = direct RMSD of CDR CAs vs GT (no Kabsch needed)
        • LRMS = direct RMSD of all Ab CAs vs GT (no Kabsch needed)
        • clash = distance check against fixed true antigen CAs

    Args:
        x_1_pred:     {"bb_ca": [B, N, 3], ...} — predicted clean sample (nm)
        batch:        training batch dict
        clash_weight: λ for clash penalty term (default 0.1)

    Returns:
        reward: [B]  ∈ approximately [0, 1] (can be slightly negative if very
                clashing)
    """
    fnat   = fnat_proxy_reward(x_1_pred, batch)   # [B] ∈ [0, 1]
    irms   = irms_score(x_1_pred, batch)           # [B] ∈ (0, 1]
    lrms   = lrms_score(x_1_pred, batch)           # [B] ∈ (0, 1]
    clash  = clash_penalty(x_1_pred, batch)        # [B] ≥ 0

    dockq  = (fnat + irms + lrms) / 3.0           # [B] ∈ [0, 1]
    reward = dockq - clash_weight * clash          # [B]
    return reward


def ca_dockq_no_clash_reward(x_1_pred: Dict, batch: Dict) -> Tensor:
    """
    CA-DockQ proxy without clash penalty (for ablation / logging).

    DockQ = (fnat + iRMS_score + LRMS_score) / 3

    Returns:
        dockq: [B] ∈ [0, 1]
    """
    fnat = fnat_proxy_reward(x_1_pred, batch)
    irms = irms_score(x_1_pred, batch)
    lrms = lrms_score(x_1_pred, batch)
    return (fnat + irms + lrms) / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3: Full DockQ (stub)
# ─────────────────────────────────────────────────────────────────────────────

def full_dockq_reward(
    x_1_pred: Dict,
    batch: Dict,
    autoencoder,
    tmp_dir: str = "/tmp/dockq_tmp",
) -> Tensor:
    """
    Full DockQ reward via AE decode → full-atom structure → DockQ subprocess.

    This is a Phase-3 reward — slow but matches the final evaluation metric.
    Requires `dockq` to be installed and accessible on PATH.

    Args:
        x_1_pred:    {"bb_ca": [B, N, 3], "local_latents": [B, N, 8]}
        batch:       training batch dict (needs coords_nm, chain_type, etc.)
        autoencoder: frozen AE model for decoding latents → full-atom structure
        tmp_dir:     temporary directory for PDB files

    Returns:
        dockq: [B] ∈ [0, 1]  (DockQ score, 0 if decode/subprocess fails)
    """
    import os, subprocess, tempfile
    from proteinfoundation.utils.coors_utils import nm_to_ang, trans_nm_to_atom37
    from proteinfoundation.utils.pdb_utils import create_full_prot, to_pdb

    B = x_1_pred["bb_ca"].shape[0]
    ab_mask = batch["mask"].bool()  # [B, N]

    # AE decode: latents → full-atom coords + sequence
    with torch.no_grad():
        decoded = autoencoder.decode(
            z_latent    = x_1_pred["local_latents"],  # [B, N, 8]
            ca_coors_nm = x_1_pred["bb_ca"],           # [B, N, 3]
            mask        = ab_mask,                     # [B, N]
        )
    # decoded: {"coors_nm": [B, N, 37, 3], "residue_type": [B, N]}

    os.makedirs(tmp_dir, exist_ok=True)
    scores = []

    for b in range(B):
        try:
            # Build antibody PDB
            ab_len = ab_mask[b].sum().item()
            ab_coors_ang = nm_to_ang(decoded["coors_nm"][b, :ab_len])  # [N_ab, 37, 3]
            ab_seq  = decoded["residue_type"][b, :ab_len]               # [N_ab]

            # Antigen coords from batch
            ag_mask = (batch["chain_type"][b] == 3)
            ag_coors_ang = nm_to_ang(
                batch["coords_nm"][b, ag_mask, :, :]  # [N_ag, 37, 3]
            )
            ag_seq = batch.get("residue_type", batch.get("gt_seq"))[b, ag_mask]

            # Write model PDB (predicted antibody + true antigen)
            model_pdb = os.path.join(tmp_dir, f"model_{b}.pdb")
            # Write reference PDB (true antibody + true antigen)
            ref_pdb = os.path.join(tmp_dir, f"ref_{b}.pdb")

            # TODO: Implement proper PDB writing with chain labels (H, L, A)
            # For now, return 0 as placeholder
            scores.append(0.0)

        except Exception as e:
            scores.append(0.0)

    return torch.tensor(scores, device=x_1_pred["bb_ca"].device, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Reward factory
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# REBUTTAL E4: alternative reward designs (demonstrate FM-GRPO reward-agnostic)
# ---------------------------------------------------------------------------
def fnat_emphasis_reward(x_1_pred, batch, w_fnat: float = 0.6):
    """Reweighted DockQ emphasizing native interface-contact recovery.
    reward = w_fnat*fnat + (1-w_fnat)/2*(iRMS + LRMS). Different objective
    than the (1/3,1/3,1/3) DockQ proxy; tests whether FM-GRPO tracks an
    arbitrary reweighting of the interface signal."""
    fnat = fnat_proxy_reward(x_1_pred, batch)
    irms = irms_score(x_1_pred, batch)
    lrms = lrms_score(x_1_pred, batch)
    wr = (1.0 - w_fnat) / 2.0
    return w_fnat * fnat + wr * irms + wr * lrms


def clash_heavy_reward(x_1_pred, batch, clash_weight: float = 0.5):
    """CA-DockQ proxy with a MUCH stronger clash penalty (default 5x tier-2).
    Directly targets the reviewer concern that DockQ alone tolerates poor
    packing/steric clashes; the reward explicitly pushes clash-free interfaces."""
    fnat = fnat_proxy_reward(x_1_pred, batch)
    irms = irms_score(x_1_pred, batch)
    lrms = lrms_score(x_1_pred, batch)
    clash = clash_penalty(x_1_pred, batch)
    dockq = (fnat + irms + lrms) / 3.0
    return dockq - clash_weight * clash


# ─────────────────────────────────────────────────────────────────────────────
# Affinity-related, native-antibody-free interface proxy
# ─────────────────────────────────────────────────────────────────────────────

def _as_batch_mask(value: Tensor, batch_size: int) -> Tensor:
    """Return a boolean mask with an explicit batch dimension."""
    value = value.bool()
    if value.dim() == 1:
        value = value.unsqueeze(0)
    if value.dim() != 2 or value.shape[0] != batch_size:
        raise ValueError(f"expected [B,N] mask for B={batch_size}, got {tuple(value.shape)}")
    return value


def _as_batch_field(value: Tensor, batch_size: int) -> Tensor:
    """Return a per-residue field with an explicit batch dimension."""
    if value.dim() == 1:
        value = value.unsqueeze(0)
    if value.dim() != 2 or value.shape[0] != batch_size:
        raise ValueError(f"expected [B,N] field for B={batch_size}, got {tuple(value.shape)}")
    return value


def _affinity_masks(batch: Dict, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return designed-antibody, all-antibody and interface-antigen masks.

    The designed-antibody mask is CDR-only and prefers the immutable native CDR
    mask; the stochastic mixed-training mask is only a legacy fallback.  The
    interface-antigen mask prefers the supplied epitope and falls back to all
    antigen residues on a per-example basis.  No native antibody coordinates
    or native contact map is consulted here.
    """
    ab_mask = _as_batch_mask(batch["mask"], batch_size)
    chain_type = batch.get("chain_type")
    if chain_type is None:
        epitope = batch.get("epitope_mask")
        if epitope is None:
            raise KeyError("affinity proxy needs chain_type or epitope_mask")
        epitope = _as_batch_mask(epitope, batch_size)
        ag_mask = epitope.clone()
    else:
        # Preserve integer chain labels; converting them to bool would erase
        # the distinction between antigen chain 3 and antibody chains 1/2.
        chain_type = _as_batch_field(chain_type, batch_size)
        ag_mask = chain_type == 3

    # ``cdr_mask`` is the stochastic training mask under mixed masking: it can
    # be a partial subset or all-zero reconstruction mask.  The prospective
    # affinity reward must target the same native CDR positions on every
    # update, otherwise its objective changes with the masking draw.  Keep the
    # fallback for older cached batches that predate ``native_cdr_mask``.
    cdr = batch.get("native_cdr_mask", batch.get("cdr_mask"))
    if cdr is None:
        designed = ab_mask.clone()
    else:
        designed = ab_mask & _as_batch_mask(cdr, batch_size)
        has_cdr = designed.any(dim=1, keepdim=True)
        designed = torch.where(has_cdr, designed, ab_mask)

    epitope = batch.get("epitope_mask")
    if epitope is not None:
        epitope = _as_batch_mask(epitope, batch_size)
        epi_interface = ag_mask & epitope
        has_epitope = epi_interface.any(dim=1, keepdim=True)
        ag_interface = torch.where(has_epitope, epi_interface, ag_mask)
    else:
        ag_interface = ag_mask
    return designed, ab_mask, ag_interface


def _affinity_terms_one(
    pred_ca: Tensor,
    true_ca: Tensor,
    designed_mask: Tensor,
    ab_mask: Tensor,
    ag_interface_mask: Tensor,
    ag_all_mask: Tensor,
    *,
    weights: Dict[str, float],
    contact_cutoff_nm: float,
    contact_sharpness: float,
    buried_radius_nm: float,
    buried_target: float,
    compactness_scale_nm: float,
    clash_distance_nm: float,
    clash_sharpness: float,
    clash_tolerance: float,
    clash_gate_strength: float,
    rg_expected_nm: float,
    rg_tolerance_nm: float,
    radius_penalty_weight: float,
) -> Dict[str, Tensor]:
    """Compute one-example differentiable interface-affinity proxy terms."""
    zero = pred_ca.sum() * 0.0
    if not bool(designed_mask.any()) or not bool(ag_interface_mask.any()):
        return {
            "affinity_proxy": zero,
            "contact_coverage": zero,
            "buried_surface_proxy": zero,
            "interface_compactness": zero,
            "steric_gate": zero + 1.0,
            "antibody_compactness_prior": zero + 1.0,
            "radius_gyration": zero,
            "radius_prior_penalty": zero,
            "soft_clash_fraction": zero,
        }

    designed = pred_ca[designed_mask]
    antigen = true_ca[ag_interface_mask]
    distances = torch.cdist(designed, antigen)
    nearest_designed = distances.min(dim=1).values
    nearest_antigen = distances.min(dim=0).values

    contact_designed = torch.sigmoid(
        (contact_cutoff_nm - nearest_designed) * contact_sharpness
    )
    contact_antigen = torch.sigmoid(
        (contact_cutoff_nm - nearest_antigen) * contact_sharpness
    )
    contact_coverage = 0.5 * (
        contact_designed.mean() + contact_antigen.mean()
    )

    buried_designed = torch.sigmoid(
        (buried_radius_nm - nearest_designed) * contact_sharpness
    )
    buried_antigen = torch.sigmoid(
        (buried_radius_nm - nearest_antigen) * contact_sharpness
    )
    buried_surface_proxy = torch.tanh(
        (buried_designed.sum() + buried_antigen.sum())
        / (2.0 * max(float(buried_target), 1e-6))
    )

    contact_weight = contact_designed
    weight_sum = contact_weight.sum().clamp_min(1e-8)
    centre = (designed * contact_weight.unsqueeze(-1)).sum(dim=0) / weight_sum
    spread = torch.sqrt(
        (((designed - centre).square().sum(dim=-1)) * contact_weight).sum()
        / weight_sum
        + 1e-8
    )
    interface_compactness = torch.exp(-spread / max(float(compactness_scale_nm), 1e-6))
    # Far-away structures should not receive a compactness reward merely
    # because their antibody coordinates happen to be tight.
    interface_compactness = interface_compactness * torch.tanh(weight_sum / 2.0)

    if bool(ab_mask.any()) and bool(ag_all_mask.any()):
        all_distances = torch.cdist(pred_ca[ab_mask], true_ca[ag_all_mask])
        nearest_ab = all_distances.min(dim=1).values
        nearest_ag = all_distances.min(dim=0).values
        clash_ab = torch.sigmoid(
            (clash_distance_nm - nearest_ab) * clash_sharpness
        ).mean()
        clash_ag = torch.sigmoid(
            (clash_distance_nm - nearest_ag) * clash_sharpness
        ).mean()
        soft_clash_fraction = 0.5 * (clash_ab + clash_ag)
        steric_gate = torch.exp(
            -torch.relu(soft_clash_fraction - clash_tolerance)
            * clash_gate_strength
        )
    else:
        soft_clash_fraction = zero
        steric_gate = zero + 1.0

    if bool(ab_mask.any()):
        antibody = pred_ca[ab_mask]
        antibody_centre = antibody.mean(dim=0)
        radius_gyration = torch.sqrt(
            (antibody - antibody_centre).square().sum(dim=-1).mean() + 1e-8
        )
        antibody_compactness_prior = torch.exp(
            -((radius_gyration - rg_expected_nm).square())
            / (2.0 * max(float(rg_tolerance_nm), 1e-6) ** 2)
        )
    else:
        radius_gyration = zero
        antibody_compactness_prior = zero + 1.0

    radius_prior_penalty = (
        max(float(radius_penalty_weight), 0.0)
        * torch.relu(1.0 - antibody_compactness_prior)
    )

    additive = (
        float(weights["contact_coverage"]) * contact_coverage
        + float(weights["buried_surface_proxy"]) * buried_surface_proxy
        + float(weights["interface_compactness"]) * interface_compactness
    )
    # The historical proxy multiplied by the radius prior.  That makes a
    # train/eval geometry mismatch collapse the entire structural signal and
    # lets a sequence teacher select every winner.  Keep that exact behavior
    # at the zero default for queued/historical arms.  The optional corrected
    # form keeps the contact/buried/compactness signal alive and applies the
    # radius prior as an explicit soft penalty instead.
    if float(radius_penalty_weight) > 0.0:
        affinity_proxy = additive * steric_gate - radius_prior_penalty
    else:
        affinity_proxy = additive * steric_gate * antibody_compactness_prior
    return {
        "affinity_proxy": affinity_proxy,
        "contact_coverage": contact_coverage,
        "buried_surface_proxy": buried_surface_proxy,
        "interface_compactness": interface_compactness,
        "steric_gate": steric_gate,
        "antibody_compactness_prior": antibody_compactness_prior,
        "radius_gyration": radius_gyration,
        "radius_prior_penalty": radius_prior_penalty,
        "soft_clash_fraction": soft_clash_fraction,
    }


def interface_affinity_proxy_reward(
    x_1_pred: Dict,
    batch: Dict,
    *,
    weights: Optional[Dict[str, float]] = None,
    contact_cutoff_nm: float = AFFINITY_CONTACT_CUTOFF_NM,
    contact_sharpness: float = AFFINITY_CONTACT_SHARPNESS,
    buried_radius_nm: float = AFFINITY_BURIED_RADIUS_NM,
    buried_target: float = AFFINITY_BURIED_TARGET,
    compactness_scale_nm: float = AFFINITY_COMPACTNESS_SCALE_NM,
    clash_distance_nm: float = AFFINITY_CLASH_DISTANCE_NM,
    clash_sharpness: float = AFFINITY_CLASH_SHARPNESS,
    clash_tolerance: float = AFFINITY_CLASH_TOLERANCE,
    clash_gate_strength: float = AFFINITY_CLASH_GATE_STRENGTH,
    rg_expected_nm: float = AFFINITY_RG_EXPECTED_NM,
    rg_tolerance_nm: float = AFFINITY_RG_TOLERANCE_NM,
    radius_penalty_weight: float = AFFINITY_RADIUS_PENALTY_WEIGHT,
    return_terms: bool = False,
):
    """Differentiable affinity-related interface proxy, higher is better.

    The only target-side coordinates read are the fixed antigen/epitope input
    coordinates.  Native antibody coordinates, native contacts, DockQ labels,
    and predictor confidence values are not used.  The output is therefore an
    affinity-*related structural proxy*, not a binding-affinity measurement.

    The additive terms are contact coverage, a buried-interface proxy, and
    interface compactness.  A multiplicative steric gate prevents complete
    interpenetration from winning by contact count.  The historical radius
    prior multiplied the whole score; the optional radius_penalty_weight
    instead keeps the structural signal alive while explicitly penalizing
    out-of-calibration geometry.
    """
    pred_ca = x_1_pred["bb_ca"]
    if pred_ca.dim() != 3:
        raise ValueError(f"bb_ca must be [B,N,3], got {tuple(pred_ca.shape)}")
    batch_size = pred_ca.shape[0]
    true_ca = _get_true_ca(batch)
    designed, ab_mask, ag_interface = _affinity_masks(batch, batch_size)
    chain_type = batch.get("chain_type")
    if chain_type is None:
        ag_all = ag_interface
    else:
        ag_all = _as_batch_field(chain_type, batch_size) == 3

    weights = dict(DEFAULT_AFFINITY_WEIGHTS if weights is None else weights)
    expected_weights = set(DEFAULT_AFFINITY_WEIGHTS)
    if set(weights) != expected_weights or any(float(v) < 0 for v in weights.values()):
        raise ValueError(f"affinity weights must be exactly {sorted(expected_weights)} and nonnegative")
    if not math.isclose(sum(float(v) for v in weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("affinity weights must sum to 1")

    per_example = []
    for index in range(batch_size):
        per_example.append(
            _affinity_terms_one(
                pred_ca[index], true_ca[index], designed[index], ab_mask[index],
                ag_interface[index], ag_all[index], weights=weights,
                contact_cutoff_nm=contact_cutoff_nm,
                contact_sharpness=contact_sharpness,
                buried_radius_nm=buried_radius_nm,
                buried_target=buried_target,
                compactness_scale_nm=compactness_scale_nm,
                clash_distance_nm=clash_distance_nm,
                clash_sharpness=clash_sharpness,
                clash_tolerance=clash_tolerance,
                clash_gate_strength=clash_gate_strength,
                rg_expected_nm=rg_expected_nm,
                rg_tolerance_nm=rg_tolerance_nm,
                radius_penalty_weight=radius_penalty_weight,
            )
        )
    terms = {
        name: torch.stack([row[name] for row in per_example])
        for name in per_example[0]
    }
    if return_terms:
        return terms["affinity_proxy"], terms
    return terms["affinity_proxy"]


def ca_dockq_affinity_proxy_reward(
    x_1_pred: Dict,
    batch: Dict,
    *,
    affinity_weight: float = 1.0,
    clash_weight: float = 0.01,
    affinity_kwargs: Optional[Dict] = None,
    return_terms: bool = False,
):
    """Return an affinity-primary reward, optionally mixed with CA-DockQ.

    ``affinity_weight=1.0`` is the only default/prospective setting.  In that
    setting the function does not call any native-antibody-dependent DockQ
    component.  Values below 1.0 are retained solely for explicitly labelled
    historical/diagnostic ablations and must not be used as the main method.
    """
    if not 0.0 <= float(affinity_weight) <= 1.0:
        raise ValueError("affinity_weight must be in [0, 1]")
    affinity_kwargs = dict(affinity_kwargs or {})
    affinity, terms = interface_affinity_proxy_reward(
        x_1_pred, batch, return_terms=True, **affinity_kwargs
    )
    if float(affinity_weight) == 1.0:
        # Do not even evaluate the GT-locked CA-DockQ path on the prospective
        # reward route.  This keeps native antibody coordinates out of RL.
        dockq = torch.zeros_like(affinity)
        reward = affinity
    else:
        dockq = ca_dockq_proxy_reward(x_1_pred, batch, clash_weight=clash_weight)
        reward = (1.0 - float(affinity_weight)) * dockq + float(affinity_weight) * affinity
    if return_terms:
        return reward, {
            **terms,
            "ca_dockq_proxy": dockq,
            "combined_reward": reward,
        }
    return reward


class AffinityAugmentedReward:
    """Callable affinity-primary reward with optional diagnostic summaries.

    ``sequence_teacher`` is an optional frozen teacher fit on independent
    experimental affinity measurements.  When present, its bounded score is
    mixed with the structure-only interface proxy.  The teacher is deliberately
    disabled unless an explicit artifact is supplied; this prevents a missing
    external dataset from silently turning the reward into an uncalibrated
    pseudo-affinity signal.
    """

    def __init__(
        self,
        *,
        affinity_weight: float,
        clash_weight: float,
        affinity_kwargs: Optional[Dict] = None,
        sequence_teacher=None,
        sequence_teacher_weight: float = 0.0,
        sequence_teacher_fusion: str = "linear",
    ):
        self.affinity_weight = float(affinity_weight)
        self.clash_weight = float(clash_weight)
        self.affinity_kwargs = dict(affinity_kwargs or {})
        self.sequence_teacher = sequence_teacher
        self.sequence_teacher_weight = float(sequence_teacher_weight)
        self.sequence_teacher_fusion = str(sequence_teacher_fusion).lower()
        if self.sequence_teacher_fusion not in {"linear", "geometric"}:
            raise ValueError(
                "sequence_teacher_fusion must be 'linear' or 'geometric'"
            )
        if not 0.0 <= self.sequence_teacher_weight <= 1.0:
            raise ValueError("sequence_teacher_weight must be in [0, 1]")
        self.requires_sequence = self.sequence_teacher is not None and self.sequence_teacher_weight > 0.0
        self._term_sums: Dict[str, float] = {}
        self._calls = 0

    def __call__(self, x_1_pred: Dict, batch: Dict) -> Tensor:
        reward, terms = ca_dockq_affinity_proxy_reward(
            x_1_pred,
            batch,
            affinity_weight=self.affinity_weight,
            clash_weight=self.clash_weight,
            affinity_kwargs=self.affinity_kwargs,
            return_terms=True,
        )
        if self.requires_sequence:
            residue_type = x_1_pred.get("residue_type")
            if residue_type is None:
                raise KeyError(
                    "affinity sequence teacher requires autoencoder-decoded residue_type in rollout"
                )
            teacher_score = self.sequence_teacher.score_torch(residue_type, batch).to(reward.dtype)
            weight = self.sequence_teacher_weight
            reward = fuse_affinity_teacher_reward(
                reward,
                teacher_score,
                weight,
                fusion=self.sequence_teacher_fusion,
            )
            terms["sequence_affinity_teacher"] = teacher_score
            terms["sequence_teacher_weight"] = torch.full_like(teacher_score, weight)
            terms["sequence_teacher_fusion_geometric"] = torch.full_like(
                teacher_score, float(self.sequence_teacher_fusion == "geometric")
            )
            terms["combined_reward"] = reward
        for name, value in terms.items():
            self._term_sums[name] = self._term_sums.get(name, 0.0) + float(value.detach().mean().item())
        self._calls += 1
        return reward

    def pop_metrics(self) -> Dict[str, float | int]:
        """Return and clear terms accumulated during the latest trainer step."""
        if self._calls == 0:
            return {}
        divisor = float(self._calls)
        metrics = {
            f"rl/affinity/{name}": value / divisor
            for name, value in sorted(self._term_sums.items())
        }
        metrics["rl/affinity/weight"] = self.affinity_weight
        metrics["rl/affinity/dockq_weight"] = 1.0 - self.affinity_weight
        metrics["rl/affinity/reward_calls"] = self._calls
        metrics["rl/affinity/sequence_teacher_enabled"] = int(self.requires_sequence)
        metrics["rl/affinity/sequence_teacher_fusion"] = self.sequence_teacher_fusion
        self._term_sums = {}
        self._calls = 0
        return metrics


def fuse_affinity_teacher_reward(
    affinity_reward: Tensor,
    teacher_score: Tensor,
    weight: float,
    *,
    fusion: str = "linear",
) -> Tensor:
    """Fuse structural affinity and sequence-teacher scores in ``[0, 1]``.

    The default linear mixture preserves all historical arms.  ``geometric``
    is a deliberately stricter consensus objective: a candidate must score
    well on both the interface proxy and the independent affinity teacher;
    one score cannot fully compensate for a poor score on the other.  Both
    paths remain native-antibody-free and differentiable with respect to the
    structural reward.
    """
    fusion = str(fusion).lower()
    if fusion == "linear":
        return (1.0 - float(weight)) * affinity_reward + float(weight) * teacher_score
    if fusion == "geometric":
        eps = torch.finfo(affinity_reward.dtype).eps
        return torch.exp(
            (1.0 - float(weight)) * torch.log(affinity_reward.clamp_min(eps))
            + float(weight) * torch.log(teacher_score.clamp_min(eps))
        )
    raise ValueError("fusion must be 'linear' or 'geometric'")


# ─────────────────────────────────────────────────────────────────────────────
# R_free: affinity-centered, confidence- and developability-aware reward
# ─────────────────────────────────────────────────────────────────────────────

PROTENIX_FOLD_FEATURE_NAMES = (
    "contact_ab_coverage",
    "contact_ag_coverage",
    "contact_mean_coverage",
    "contact_count_per_ab",
    "nearest_interface_distance_nm",
    "soft_clash_fraction",
    "interface_compactness",
    "antibody_radius_nm",
    "backbone_continuity",
    "backbone_bond_deviation_nm",
    "antibody_length_norm",
    "antigen_length_norm",
)


def _openfold_to_teacher_indices(residue_type: Tensor) -> Tensor:
    """Convert decoded OpenFold indices to the teacher's alphabetical order."""
    mapping = torch.as_tensor(
        OPENFOLD_TO_AFFINITY_TEACHER,
        device=residue_type.device,
        dtype=torch.long,
    )
    return mapping[residue_type.long().clamp(0, len(mapping) - 1)]


def _developability_score(residue_type: Tensor, batch: Dict) -> Dict[str, Tensor]:
    """Return a bounded, sequence-only developability proxy.

    This is intentionally a weak liability screen, not a clinical or
    manufacturability predictor.  It uses only the generated native-CDR
    sequence (Cys/Met/Trp abundance, contiguous N-X-S/T motifs, and an
    excessive hydrophobic fraction).  The bounded score is included as a
    small regularizer so affinity-related geometry cannot win by creating
    obviously difficult sequences.
    """
    if residue_type.dim() == 1:
        residue_type = residue_type.unsqueeze(0)
    batch_size = residue_type.shape[0]
    cdr = batch.get("native_cdr_mask", batch.get("cdr_mask"))
    if cdr is None:
        cdr = batch["mask"].bool()
    cdr = cdr.bool()
    if cdr.dim() == 1:
        cdr = cdr.unsqueeze(0)
    valid = batch.get("full_mask", batch.get("chain_type", cdr).bool()).bool()
    if valid.dim() == 1:
        valid = valid.unsqueeze(0)
    cdr = cdr & valid

    scores = []
    terms = []
    for row in range(batch_size):
        seq = residue_type[row]
        mask = cdr[row]
        n = mask.sum().clamp_min(1).float()
        aa = seq[mask]
        # OpenFold order: A R N D C Q E G H I L K M F P S T W Y V.
        cys = (aa == 4).float().mean() if aa.numel() else seq.sum() * 0.0
        mw = ((aa == 12) | (aa == 17)).float().mean() if aa.numel() else seq.sum() * 0.0
        hydrophobic = ((aa == 0) | (aa == 4) | (aa == 9) | (aa == 10)
                       | (aa == 12) | (aa == 13) | (aa == 17) | (aa == 18))
        hydrophobic_fraction = hydrophobic.float().mean() if aa.numel() else seq.sum() * 0.0

        # Only count motifs inside contiguous CDR positions; this avoids
        # inventing an N-X-S/T liability across a framework gap or CDR gap.
        motif_count = seq.sum() * 0.0
        if aa.numel() >= 3:
            positions = mask.nonzero(as_tuple=False).squeeze(-1)
            contiguous = (positions[1:-1] == positions[:-2] + 1) & (
                positions[2:] == positions[1:-1] + 1
            )
            if contiguous.any():
                triplets = seq[positions[:-2]], seq[positions[1:-1]], seq[positions[2:]]
                motif = (triplets[0] == 2) & ((triplets[2] == 15) | (triplets[2] == 16))
                motif_count = (motif & contiguous).float().sum() / n
        penalty = (
            2.0 * cys
            + 1.0 * mw
            + 1.5 * motif_count
            + torch.relu(hydrophobic_fraction - 0.55)
        )
        score = torch.exp(-penalty).clamp(0.0, 1.0)
        scores.append(score)
        terms.append({
            "cys_fraction": cys,
            "met_trp_fraction": mw,
            "nxs_t_rate": motif_count,
            "hydrophobic_fraction": hydrophobic_fraction,
        })
    return {
        "developability": torch.stack(scores),
        **{name: torch.stack([item[name] for item in terms]) for name in terms[0]},
    }


class ProtenixFoldSurrogate:
    """Frozen geometry-only surrogate calibrated on disjoint Protenix priors.

    The JSON artifact stores a train-only standardized ridge fit.  Inference
    deliberately reconstructs the feature map from the generated antibody
    and fixed antigen CA coordinates, without native antibody coordinates,
    DockQ, or experimental affinity labels.
    """

    def __init__(self, path: str | Path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = tuple(payload.get("feature_names", ()))
        if names != PROTENIX_FOLD_FEATURE_NAMES:
            raise ValueError(
                "unsupported Protenix surrogate feature schema: "
                f"{names!r}"
            )
        self.path = str(path)
        self.source_manifest = payload.get("source_manifest")
        self.feature_mean = tuple(float(x) for x in payload["feature_mean"])
        self.feature_scale = tuple(max(float(x), 1e-8) for x in payload["feature_scale"])
        self.coef = tuple(float(x) for x in payload["coef"])
        self.intercept = float(payload["intercept"])
        self.standardized_feature_clip = float(payload.get("standardized_feature_clip", 5.0))
        if self.standardized_feature_clip <= 0.0:
            raise ValueError("Protenix surrogate standardized_feature_clip must be positive")
        clip = payload.get("clip", [0.0, 1.0])
        self.clip = (float(clip[0]), float(clip[1]))
        if len(self.coef) != len(PROTENIX_FOLD_FEATURE_NAMES):
            raise ValueError("Protenix surrogate coefficient dimension mismatch")

    @staticmethod
    def _chain_segments(pred_ca: Tensor, valid: Tensor, chain_type: Tensor,
                        chain_breaks: Optional[Tensor]) -> list[Tensor]:
        indices = valid.nonzero(as_tuple=False).squeeze(-1).tolist()
        if not indices:
            return []
        segments: list[Tensor] = []
        current: list[int] = []
        previous = None
        for position in indices:
            starts = bool(current) and (
                (chain_breaks is not None and bool(chain_breaks[position]))
                or (previous is not None and int(chain_type[position]) != previous)
            )
            if starts:
                segments.append(pred_ca[torch.as_tensor(current, device=pred_ca.device)])
                current = []
            current.append(position)
            previous = int(chain_type[position])
        if current:
            segments.append(pred_ca[torch.as_tensor(current, device=pred_ca.device)])
        return segments

    def _features_one(self, pred_ca: Tensor, batch: Dict, row: int) -> Tensor:
        chain_type = batch["chain_type"][row].long()
        valid = batch.get("full_mask", chain_type > 0)[row].bool()
        ab_mask = valid & (chain_type < 3)
        ag_mask = valid & (chain_type == 3)
        antibody = pred_ca[row, ab_mask]
        antigen = pred_ca[row, ag_mask]
        zero = pred_ca[row].sum() * 0.0
        if antibody.shape[0] == 0 or antigen.shape[0] == 0:
            return torch.zeros(len(PROTENIX_FOLD_FEATURE_NAMES), device=pred_ca.device, dtype=pred_ca.dtype) + zero
        distances = torch.cdist(antibody, antigen)
        nearest_ab = distances.min(dim=1).values
        nearest_ag = distances.min(dim=0).values
        contact_ab = nearest_ab < CONTACT_THRESHOLD_NM
        contact_ag = nearest_ag < CONTACT_THRESHOLD_NM
        contact_ab_cov = contact_ab.float().mean()
        contact_ag_cov = contact_ag.float().mean()
        contact_mean = 0.5 * (contact_ab_cov + contact_ag_cov)
        contact_count_per_ab = contact_ab_cov
        nearest_distance_nm = torch.minimum(nearest_ab.mean(), nearest_ag.mean())
        soft_clash = 0.5 * (
            (nearest_ab < CLASH_THRESHOLD_NM).float().mean()
            + (nearest_ag < CLASH_THRESHOLD_NM).float().mean()
        )
        interface_points = antibody[contact_ab]
        if interface_points.shape[0]:
            spread = torch.sqrt(((interface_points - interface_points.mean(dim=0)) ** 2).sum(dim=-1).mean() + 1e-8)
            compactness = torch.exp(-spread / AFFINITY_COMPACTNESS_SCALE_NM)
        else:
            compactness = zero
        centre = antibody.mean(dim=0)
        radius_nm = torch.sqrt(((antibody - centre) ** 2).sum(dim=-1).mean() + 1e-8)
        breaks = batch.get("chain_breaks_per_residue")
        breaks_row = breaks[row].bool() if breaks is not None else None
        segments = self._chain_segments(pred_ca[row], valid, chain_type, breaks_row)
        continuity: list[Tensor] = []
        bond_deviation: list[Tensor] = []
        for segment in segments:
            if segment.shape[0] < 2:
                continue
            bond = torch.sqrt(((segment[1:] - segment[:-1]) ** 2).sum(dim=-1) + 1e-8)
            continuity.append(((bond > 0.32) & (bond < 0.45)).float().mean())
            bond_deviation.append((bond - 0.38).abs().mean())
        continuity_value = torch.stack(continuity).mean() if continuity else zero
        deviation_value = torch.stack(bond_deviation).mean() if bond_deviation else zero + 1.0
        values = (
            contact_ab_cov,
            contact_ag_cov,
            contact_mean,
            contact_count_per_ab,
            nearest_distance_nm,
            soft_clash,
            compactness,
            radius_nm,
            continuity_value,
            deviation_value,
            torch.as_tensor(min(int(antibody.shape[0]), 1000) / 1000.0, device=pred_ca.device, dtype=pred_ca.dtype),
            torch.as_tensor(min(int(antigen.shape[0]), 2000) / 2000.0, device=pred_ca.device, dtype=pred_ca.dtype),
        )
        return torch.stack(values)

    def score_torch(self, x_1_pred: Dict, batch: Dict, *, return_features: bool = False):
        pred_ca = x_1_pred["bb_ca"]
        features = torch.stack([
            self._features_one(pred_ca, batch, row)
            for row in range(pred_ca.shape[0])
        ])
        mean = torch.as_tensor(self.feature_mean, device=pred_ca.device, dtype=pred_ca.dtype)
        scale = torch.as_tensor(self.feature_scale, device=pred_ca.device, dtype=pred_ca.dtype)
        coef = torch.as_tensor(self.coef, device=pred_ca.device, dtype=pred_ca.dtype)
        score = (features - mean) / scale
        # Generated antibody geometry can be far outside the narrow prior
        # distribution (especially during early ODE steps).  Clipping the
        # standardized feature coordinates prevents a tiny train variance
        # from turning extrapolation into a saturated confidence reward.
        score = score.clamp(-self.standardized_feature_clip, self.standardized_feature_clip)
        score = score @ coef + self.intercept
        score = score.clamp(self.clip[0], self.clip[1])
        if return_features:
            return score, {name: features[:, i] for i, name in enumerate(PROTENIX_FOLD_FEATURE_NAMES)}
        return score


class RFreeReward:
    """Affinity-centered composite reward with explicit claim boundaries.

    ``R_free`` combines the native-antibody-free interface-affinity proxy and
    optional independent sequence teacher (the affinity term), a frozen
    Protenix-calibrated fold-confidence surrogate, a steric/physical term, and
    a small sequence-only developability regularizer.  The fixed antigen is
    the only target-side structure consumed by the geometry terms; KL/trust
    control remains the trainer's separate reference regularizer.
    """

    def __init__(self, *, fold_surrogate: ProtenixFoldSurrogate,
                 affinity_kwargs: Optional[Dict] = None,
                 sequence_teacher=None, sequence_teacher_weight: float = 0.0,
                 sequence_teacher_fusion: str = "linear",
                 weights: Optional[Dict[str, float]] = None):
        self.fold_surrogate = fold_surrogate
        self.affinity_kwargs = dict(affinity_kwargs or {})
        self.sequence_teacher = sequence_teacher
        self.sequence_teacher_weight = float(sequence_teacher_weight)
        self.sequence_teacher_fusion = str(sequence_teacher_fusion).lower()
        if not 0.0 <= self.sequence_teacher_weight <= 1.0:
            raise ValueError("sequence_teacher_weight must be in [0, 1]")
        if self.sequence_teacher_fusion not in {"linear", "geometric"}:
            raise ValueError("sequence_teacher_fusion must be 'linear' or 'geometric'")
        self.weights = dict(R_FREE_DEFAULT_WEIGHTS if weights is None else weights)
        if set(self.weights) != set(R_FREE_DEFAULT_WEIGHTS) or any(float(v) < 0 for v in self.weights.values()):
            raise ValueError(f"R_free weights must be exactly {sorted(R_FREE_DEFAULT_WEIGHTS)} and nonnegative")
        if not math.isclose(sum(float(v) for v in self.weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("R_free weights must sum to 1")
        self.requires_sequence = True
        self._term_sums: Dict[str, float] = {}
        self._calls = 0

    def score_with_terms(self, x_1_pred: Dict, batch: Dict):
        structural, structural_terms = ca_dockq_affinity_proxy_reward(
            x_1_pred, batch, affinity_weight=1.0,
            affinity_kwargs=self.affinity_kwargs, return_terms=True,
        )
        affinity = structural
        terms: Dict[str, Tensor] = {
            "structural_affinity_proxy": structural,
            **structural_terms,
        }
        teacher_score = torch.zeros_like(structural)
        if self.sequence_teacher is not None and self.sequence_teacher_weight > 0.0:
            residue_type = x_1_pred.get("residue_type")
            if residue_type is None:
                raise KeyError("R_free requires autoencoder-decoded residue_type")
            mapped = _openfold_to_teacher_indices(residue_type)
            teacher_score = self.sequence_teacher.score_torch(mapped, batch).to(structural.dtype)
            affinity = fuse_affinity_teacher_reward(
                structural, teacher_score, self.sequence_teacher_weight,
                fusion=self.sequence_teacher_fusion,
            )
        fold_score, fold_terms = self.fold_surrogate.score_torch(
            x_1_pred, batch, return_features=True
        )
        physical = 0.5 * structural_terms["steric_gate"] + 0.5 * (
            1.0 - structural_terms["soft_clash_fraction"].clamp(0.0, 1.0)
        )
        developability_terms = _developability_score(x_1_pred["residue_type"], batch)
        developability = developability_terms["developability"]
        reward = (
            float(self.weights["affinity"]) * affinity
            + float(self.weights["fold_confidence"]) * fold_score
            + float(self.weights["physical"]) * physical
            + float(self.weights["developability"]) * developability
        )
        terms.update({
            "affinity_component": affinity,
            "sequence_affinity_teacher": teacher_score,
            "fold_confidence": fold_score,
            "physical": physical,
            "developability": developability,
            "r_free_reward": reward,
            **fold_terms,
            **developability_terms,
        })
        return reward, terms

    def __call__(self, x_1_pred: Dict, batch: Dict) -> Tensor:
        reward, terms = self.score_with_terms(x_1_pred, batch)
        for name, value in terms.items():
            if isinstance(value, Tensor):
                self._term_sums[name] = self._term_sums.get(name, 0.0) + float(value.detach().mean().item())
        self._calls += 1
        return reward

    def pop_metrics(self) -> Dict[str, float | int]:
        if self._calls == 0:
            return {}
        divisor = float(self._calls)
        metrics = {f"rl/rfree/{name}": value / divisor for name, value in sorted(self._term_sums.items())}
        for name, value in self.weights.items():
            metrics[f"rl/rfree/weight_{name}"] = float(value)
        metrics["rl/rfree/sequence_teacher_enabled"] = int(self.sequence_teacher is not None and self.sequence_teacher_weight > 0.0)
        metrics["rl/rfree/sequence_teacher_weight"] = self.sequence_teacher_weight
        metrics["rl/rfree/reward_calls"] = self._calls
        self._term_sums = {}
        self._calls = 0
        return metrics


REFERENCE_DEFAULT_MIX_WEIGHTS = {
    "lddt": 0.25,
    "tm": 0.20,
    "dockq": 0.30,
    "contact_precision": 0.15,
    "clash_free": 0.10,
}


class ReferenceAugmentedRFreeReward:
    """R_free augmented with native-reference structural proxies.

    The reference is read from the current batch, so this class is intended
    only for declared train/validation post-training experiments. The test
    evaluator never calls it. ``mode`` supports one-component ablations and
    ``all`` for the final multi-objective variant.
    """

    _MODE_TO_KEYS = {
        "lddt": ("reference_lddt",),
        "tm": ("reference_tm",),
        "dockq": ("reference_dockq_proxy",),
        "all": (
            "reference_lddt",
            "reference_tm",
            "reference_dockq_proxy",
            "reference_contact_precision",
            "reference_clash_free",
        ),
    }

    def __init__(
        self,
        base_reward: RFreeReward,
        *,
        mode: str = "all",
        reference_weight: float = 0.25,
        mix_weights: Optional[Dict[str, float]] = None,
    ):
        self.base_reward = base_reward
        self.mode = str(mode).lower()
        if self.mode not in self._MODE_TO_KEYS:
            raise ValueError(
                f"unknown reference reward mode {mode!r}; "
                "choose lddt, tm, dockq, or all"
            )
        self.reference_weight = float(reference_weight)
        if not 0.0 < self.reference_weight <= 1.0:
            raise ValueError("reference_weight must be in (0, 1]")
        self.mix_weights = dict(
            REFERENCE_DEFAULT_MIX_WEIGHTS if mix_weights is None else mix_weights
        )
        if set(self.mix_weights) != set(REFERENCE_DEFAULT_MIX_WEIGHTS):
            raise ValueError(
                "reference mix weights must contain exactly "
                f"{sorted(REFERENCE_DEFAULT_MIX_WEIGHTS)}"
            )
        if any(float(value) < 0.0 for value in self.mix_weights.values()):
            raise ValueError("reference mix weights must be nonnegative")
        if not math.isclose(
            sum(float(v) for v in self.mix_weights.values()),
            1.0,
            abs_tol=1e-6,
        ):
            raise ValueError("reference mix weights must sum to 1")
        self._term_sums: Dict[str, float] = {}
        self._calls = 0

    def _reference_score(self, scores: Dict[str, Tensor]) -> Tensor:
        keys = self._MODE_TO_KEYS[self.mode]
        if self.mode != "all":
            return scores[keys[0]]
        weights = {
            "reference_lddt": self.mix_weights["lddt"],
            "reference_tm": self.mix_weights["tm"],
            "reference_dockq_proxy": self.mix_weights["dockq"],
            "reference_contact_precision": self.mix_weights["contact_precision"],
            "reference_clash_free": self.mix_weights["clash_free"],
        }
        return sum(float(weights[key]) * scores[key] for key in keys)

    def __call__(self, x_1_pred: Dict, batch: Dict) -> Tensor:
        base = self.base_reward(x_1_pred, batch)
        scores = reference_structure_scores(x_1_pred, batch)
        reference = self._reference_score(scores)
        reward = (1.0 - self.reference_weight) * base + self.reference_weight * reference
        for name, value in scores.items():
            self._term_sums[name] = self._term_sums.get(name, 0.0) + float(
                value.detach().mean().item()
            )
        self._term_sums["reference_mixed_score"] = self._term_sums.get(
            "reference_mixed_score", 0.0
        ) + float(reference.detach().mean().item())
        self._term_sums["augmented_reward"] = self._term_sums.get(
            "augmented_reward", 0.0
        ) + float(reward.detach().mean().item())
        self._calls += 1
        return reward

    def pop_metrics(self) -> Dict[str, float | int]:
        metrics = self.base_reward.pop_metrics()
        if self._calls == 0:
            return metrics
        divisor = float(self._calls)
        metrics.update({
            f"rl/reference/{name}": value / divisor
            for name, value in sorted(self._term_sums.items())
        })
        metrics["rl/reference/mode"] = self.mode
        metrics["rl/reference/weight"] = self.reference_weight
        for name, value in self.mix_weights.items():
            metrics[f"rl/reference/mix_weight_{name}"] = float(value)
        metrics["rl/reference/reward_calls"] = self._calls
        self._term_sums = {}
        self._calls = 0
        return metrics


def get_reward_fn(tier: int | str, **kwargs):
    """
    Returns a reward function for the given tier.

    Tiers:
        1  — fnat_proxy_reward: CA contact recovery only (fastest)
        2  — ca_dockq_proxy_reward: (fnat + iRMS + LRMS) / 3 − λ*clash (recommended)
        2a — ca_dockq_no_clash_reward: CA-DockQ without clash penalty (ablation)
        3  — full_dockq_reward: AE decode → PDB → subprocess DockQ (slowest)
        2_affinity / affinity — affinity-primary interface proxy; explicit
                         weights below 1.0 are diagnostic CA-DockQ mixtures
        r_free / R_free / free — affinity-centered composite using an
                         independently calibrated fold-confidence surrogate,
                         physical steric terms, and a small developability
                         regularizer; no DockQ path is evaluated
        r_free_ref / reference — R_free plus differentiable native-reference
                         structural proxies for train/validation experiments

    Args:
        tier: 1, 2, or 3  (use "2a" string for clash-free variant)
        **kwargs:
            clash_weight (float): weight for clash penalty in tier 2 (default 0.1)
            affinity_weight (float): affinity weight; defaults to 1.0
            affinity_kwargs (dict): frozen affinity-proxy constants/weights
            autoencoder:          required for tier 3
            tmp_dir (str):        temp dir for tier 3 PDB files

    Returns:
        Callable[(x_1_pred, batch) -> Tensor[B]]
    """
    if tier == 1:
        return fnat_proxy_reward

    elif tier == "2a":
        return ca_dockq_no_clash_reward

    elif tier == 2:
        clash_weight = kwargs.get("clash_weight", 0.1)
        return lambda x1, b: ca_dockq_proxy_reward(x1, b, clash_weight=clash_weight)

    elif tier in {"2_affinity", "affinity"}:
        return AffinityAugmentedReward(
            affinity_weight=float(kwargs.get("affinity_weight", 1.0)),
            clash_weight=float(kwargs.get("clash_weight", 0.01)),
            affinity_kwargs=kwargs.get("affinity_kwargs"),
            sequence_teacher=kwargs.get("sequence_teacher"),
            sequence_teacher_weight=float(kwargs.get("sequence_teacher_weight", 0.0)),
            sequence_teacher_fusion=str(kwargs.get("sequence_teacher_fusion", "linear")),
        )

    elif str(tier).lower() in {"r_free", "rfree", "free"}:
        fold_surrogate = kwargs.get("fold_surrogate")
        if fold_surrogate is None:
            fold_path = kwargs.get("fold_surrogate_path")
            if not fold_path:
                raise ValueError("R_free requires fold_surrogate_path")
            fold_surrogate = ProtenixFoldSurrogate(fold_path)
        return RFreeReward(
            fold_surrogate=fold_surrogate,
            affinity_kwargs=kwargs.get("affinity_kwargs"),
            sequence_teacher=kwargs.get("sequence_teacher"),
            sequence_teacher_weight=float(kwargs.get("sequence_teacher_weight", 0.0)),
            sequence_teacher_fusion=str(kwargs.get("sequence_teacher_fusion", "linear")),
            weights=kwargs.get("rfree_weights", kwargs.get("weights")),
        )

    elif str(tier).lower() in {"r_free_ref", "rfree_ref", "reference"}:
        fold_surrogate = kwargs.get("fold_surrogate")
        if fold_surrogate is None:
            fold_path = kwargs.get("fold_surrogate_path")
            if not fold_path:
                raise ValueError("reference R_free requires fold_surrogate_path")
            fold_surrogate = ProtenixFoldSurrogate(fold_path)
        base_reward = RFreeReward(
            fold_surrogate=fold_surrogate,
            affinity_kwargs=kwargs.get("affinity_kwargs"),
            sequence_teacher=kwargs.get("sequence_teacher"),
            sequence_teacher_weight=float(kwargs.get("sequence_teacher_weight", 0.0)),
            sequence_teacher_fusion=str(kwargs.get("sequence_teacher_fusion", "linear")),
            weights=kwargs.get("rfree_weights", kwargs.get("weights")),
        )
        return ReferenceAugmentedRFreeReward(
            base_reward,
            mode=str(kwargs.get("reference_mode", "all")),
            reference_weight=float(kwargs.get("reference_weight", 0.25)),
            mix_weights=kwargs.get("reference_mix_weights"),
        )

    elif tier == 3:
        ae = kwargs.get("autoencoder")
        tmp_dir = kwargs.get("tmp_dir", "/tmp/dockq_tmp")
        return lambda x1, b: full_dockq_reward(x1, b, ae, tmp_dir)

    elif tier == "R1":
        w = kwargs.get("w_fnat", 0.6)
        return lambda x1, b: fnat_emphasis_reward(x1, b, w_fnat=w)

    elif tier == "R2":
        cw = kwargs.get("clash_weight", 0.5)
        return lambda x1, b: clash_heavy_reward(x1, b, clash_weight=cw)

    else:
        raise ValueError(
            f"Unknown reward tier {tier!r}. Choose 1, 2, '2a', '2_affinity', 'r_free', or 3."
        )
