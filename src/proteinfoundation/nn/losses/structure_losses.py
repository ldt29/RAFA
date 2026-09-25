"""
Structure-based auxiliary losses for antibody CA coordinate supervision.

Priority A (NeurIPS innovation plan): Rich CA coordinate supervision.

During FM training the predicted velocity v gives us a clean sample estimate:
    x_1_pred = x_t + (1 - t) * v      [training-only formula]

We exploit this free x_1_pred to add physics-inspired constraints beyond the
vanilla L2 velocity loss, without any extra prediction heads or parameters.

Inspired by dyMEAN (Wu et al.) structure_loss which uses actual N/CA/C/O atoms.
Since our model is CA-only, we implement CA pseudo-angle equivalents:
  dyMEAN bond_loss  (N-CA, CA-C, C-N)  →  A-1: CA-CA virtual bond ≈ 3.8 Å
  dyMEAN angle_loss (phi/psi dihedral)  →  A-3: CA pseudo-bond angle (3 CA)
                                         →  A-4: CA pseudo-dihedral (4 CA)
A-3 / A-4 are rotation-translation invariant and directly penalise local
geometry deviations from ground truth — stronger than fixed-target bond loss.

Functions (Priority A — original):
    backbone_bond_loss        — A-1: CA-CA virtual bond length ≈ 3.8 Å
    interface_distance_loss   — A-2: CDR-Epitope hinge loss on contact pairs
    ca_bond_angle_loss        — A-3: CA pseudo-bond angle (3 consecutive CA)
    ca_dihedral_loss          — A-4: CA pseudo-dihedral angle (4 consecutive CA)

Functions (Priority A-new — docking-aware losses):
    align_rmsd_loss           — L1: Kabsch-aligned antibody RMSD (Self-RMSD / TMscore)
    interface_align_loss      — L2: iRMS-style paratope-aligned RMSD (10 Å threshold)
    soft_fnat_loss            — L3: differentiable fnat via sigmoid contact BCE
    steric_clash_loss         — L4: predicted-ab vs true-ag CA clash penalty (3.5 Å)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# A-1: Backbone CA-CA Bond Length Loss
# ─────────────────────────────────────────────────────────────────────────────

_CA_CA_TARGET_NM = 0.38  # 3.8 Å in nm (standard peptide backbone)


def backbone_bond_loss(
    ca_pred: torch.Tensor,      # [B, N, 3]  predicted clean CA coords (nm)
    ab_mask: torch.Tensor,      # [B, N]     bool, True = antibody position
    chain_type: torch.Tensor,   # [B, N]     int  (1=Heavy, 2=Light, 3=Ag)
) -> torch.Tensor:              # [B]        per-sample loss
    """
    Penalise deviations of consecutive antibody CA-CA distances from 3.8 Å.

    Only consecutive pairs that are:
      (a) both inside the antibody (ab_mask = True), and
      (b) on the same chain (chain_type equal)
    are penalised — this correctly skips the H→L junction and antigen residues.

    Uses smooth_l1_loss (Huber, δ=1) which is less sensitive to outliers at
    the start of training when x_1_pred is still inaccurate.

    Returns per-sample (batch-element) loss shaped [B].
    """
    B = ca_pred.shape[0]

    # Consecutive atom pair vectors
    ca_i = ca_pred[:, :-1, :]   # [B, N-1, 3]
    ca_j = ca_pred[:, 1:, :]    # [B, N-1, 3]
    dist = torch.norm(ca_j - ca_i, dim=-1)  # [B, N-1]

    # Valid pair gate: both antibody positions on the same chain
    valid = (
        ab_mask[:, :-1]                                     # i is antibody
        & ab_mask[:, 1:]                                    # j is antibody
        & (chain_type[:, :-1] == chain_type[:, 1:])        # same chain
    ).float()  # [B, N-1]

    target = torch.full_like(dist, _CA_CA_TARGET_NM)
    per_pair = F.smooth_l1_loss(dist, target, reduction="none")  # [B, N-1]

    # Per-sample mean over valid pairs (clamp avoids div/0 for VHH with 0 L pairs)
    n_valid = valid.sum(dim=-1).clamp(min=1.0)    # [B]
    loss = (per_pair * valid).sum(dim=-1) / n_valid   # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# A-2: CDR-Epitope Interface Distance Loss
# ─────────────────────────────────────────────────────────────────────────────

_CONTACT_THRESHOLD_NM = 0.8   # 8 Å — standard CDR-epitope contact cutoff


def interface_distance_loss(
    ca_pred: torch.Tensor,        # [B, N, 3]  predicted clean CA coords
    ca_true: torch.Tensor,        # [B, N, 3]  true clean CA coords
    cdr_mask: torch.Tensor,       # [B, N]     bool, True = CDR residue
    epitope_mask: torch.Tensor,   # [B, N]     bool, True = epitope residue
    contact_threshold_nm: float = _CONTACT_THRESHOLD_NM,
) -> torch.Tensor:                # [B]        per-sample loss
    """
    Encourage predicted CDR residues to maintain contact with epitope residues
    that are true contacts in the ground-truth structure.

    Strategy:
      - Ground-truth contact pair (i ∈ CDR, j ∈ epitope): dist(ca_true_i, ca_true_j) < threshold
      - For each such pair, apply hinge: relu(dist(ca_pred_i, ca_true_j) - threshold)
      - Non-contact pairs are NOT penalised (avoids over-constraining flexibility)

    Note: epitope positions use ca_true (antigen is locked at GT → ca_pred ≈ ca_true
    there anyway), so we consistently use ca_true for the epitope side.

    Complexity: O(N²) per sample. With N ≈ 300 and B ≈ 2, this is trivial.

    Returns per-sample (batch-element) loss shaped [B].
    """
    # Guard: skip if no CDR or no epitope residues in the batch
    if not cdr_mask.any() or not epitope_mask.any():
        return ca_pred.new_zeros(ca_pred.shape[0])

    # ── Predicted CDR × epitope (GT) distances ────────────────────────────────
    # ca_pred for CDR positions; ca_true for epitope positions
    # Shape: [B, N_cdr_query, N_epi_key] — but we work in full [B, N, N] space
    # and mask out invalid (non-CDR / non-epitope) pairs below.
    pred_dists = torch.cdist(ca_pred, ca_true)          # [B, N, N]

    # ── Ground-truth CDR × epitope distances (defines contact pairs) ──────────
    gt_dists = torch.cdist(ca_true, ca_true)             # [B, N, N]

    # ── Pair mask: i is CDR, j is epitope ─────────────────────────────────────
    pair_mask = (
        cdr_mask.unsqueeze(-1) & epitope_mask.unsqueeze(-2)
    )  # [B, N, N]

    # ── Contact mask: ground-truth contact AND valid pair ─────────────────────
    contact_mask = (gt_dists < contact_threshold_nm) & pair_mask  # [B, N, N]

    # Guard: skip if no contacts found (e.g., very early in training)
    if not contact_mask.any():
        return ca_pred.new_zeros(ca_pred.shape[0])

    # ── Hinge loss for contact pairs ──────────────────────────────────────────
    # Penalise predicted distance exceeding the contact threshold
    hinge = F.relu(pred_dists - contact_threshold_nm)   # [B, N, N]

    n_contacts = contact_mask.float().sum(dim=(-1, -2)).clamp(min=1.0)  # [B]
    loss = (hinge * contact_mask.float()).sum(dim=(-1, -2)) / n_contacts  # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# A-3: CA Pseudo-Bond Angle Loss  (dyMEAN angle_loss equivalent for CA-only)
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-7


def _cos_angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Cosine of the angle at vertex b formed by rays b→a and b→c.
    a, b, c: [..., 3]  Returns: [...]
    """
    u = F.normalize(a - b, dim=-1)
    v = F.normalize(c - b, dim=-1)
    return (u * v).sum(-1).clamp(-1 + _EPS, 1 - _EPS)


def ca_bond_angle_loss(
    ca_pred: torch.Tensor,      # [B, N, 3]  predicted clean CA coords
    ca_true: torch.Tensor,      # [B, N, 3]  ground-truth clean CA coords
    ab_mask: torch.Tensor,      # [B, N]     bool, True = antibody position
    chain_type: torch.Tensor,   # [B, N]     int  (1=H, 2=L, 3=Ag)
) -> torch.Tensor:              # [B]        per-sample loss
    """
    Penalise deviations of CA pseudo-bond angles from their ground-truth values.

    The angle is defined at the middle atom of 3 consecutive CA atoms on the
    same chain.  It captures local backbone bending geometry and correlates
    with secondary structure (helix ≈ 92°, sheet ≈ 120°, loop ≈ variable).

    Key advantage over A-1 (bond length): rotation-translation invariant —
    directly compares local geometry without rigid-body freedom.

    Inspired by dyMEAN's backbone angle_loss which uses N-CA-C atoms; here
    we use 3 consecutive CA atoms as the CA-only equivalent.

    Returns per-sample (batch-element) loss shaped [B].
    """
    # Three consecutive CA atoms: a = i-1, b = i, c = i+1
    a = ca_pred[:, :-2, :]          # [B, N-2, 3]
    b = ca_pred[:, 1:-1, :]         # [B, N-2, 3]
    c = ca_pred[:, 2:, :]           # [B, N-2, 3]

    a_t = ca_true[:, :-2, :]
    b_t = ca_true[:, 1:-1, :]
    c_t = ca_true[:, 2:, :]

    # Valid triplet: all three positions on the same chain and in ab_mask
    valid = (
        ab_mask[:, :-2] & ab_mask[:, 1:-1] & ab_mask[:, 2:]
        & (chain_type[:, :-2] == chain_type[:, 1:-1])
        & (chain_type[:, 1:-1] == chain_type[:, 2:])
    ).float()  # [B, N-2]

    cos_pred = _cos_angle(a, b, c)        # [B, N-2]
    cos_true = _cos_angle(a_t, b_t, c_t)  # [B, N-2]

    per_triplet = F.smooth_l1_loss(cos_pred, cos_true, reduction="none")  # [B, N-2]

    n_valid = valid.sum(-1).clamp(min=1.0)
    loss = (per_triplet * valid).sum(-1) / n_valid  # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# A-4: CA Pseudo-Dihedral Loss  (dyMEAN dihedral_loss equivalent for CA-only)
# ─────────────────────────────────────────────────────────────────────────────


def _cos_dihedral(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    """
    Cosine of the dihedral angle defined by 4 points a-b-c-d.
    Returns the angle between planes (a,b,c) and (b,c,d).
    a, b, c, d: [..., 3]  Returns: [...]
    """
    u1 = b - a          # [..., 3]
    u2 = c - b          # [..., 3]
    u3 = d - c          # [..., 3]
    n1 = F.normalize(torch.linalg.cross(u1, u2), dim=-1)
    n2 = F.normalize(torch.linalg.cross(u2, u3), dim=-1)
    return (n1 * n2).sum(-1).clamp(-1 + _EPS, 1 - _EPS)


def ca_dihedral_loss(
    ca_pred: torch.Tensor,      # [B, N, 3]  predicted clean CA coords
    ca_true: torch.Tensor,      # [B, N, 3]  ground-truth clean CA coords
    ab_mask: torch.Tensor,      # [B, N]     bool, True = antibody position
    chain_type: torch.Tensor,   # [B, N]     int  (1=H, 2=L, 3=Ag)
) -> torch.Tensor:              # [B]        per-sample loss
    """
    Penalise deviations of CA pseudo-dihedral angles from ground-truth values.

    The dihedral is defined by 4 consecutive CA atoms on the same chain.
    It captures backbone torsion information and correlates strongly with
    the Ramachandran phi/psi angles used in full-atom structure quality checks.

    Inspired by dyMEAN's backbone dihedral_loss (cosD from N-CA-C atoms);
    here 4 consecutive CAs give the CA-only equivalent.

    Returns per-sample (batch-element) loss shaped [B].
    """
    a = ca_pred[:, :-3, :]
    b = ca_pred[:, 1:-2, :]
    c = ca_pred[:, 2:-1, :]
    d = ca_pred[:, 3:, :]

    a_t = ca_true[:, :-3, :]
    b_t = ca_true[:, 1:-2, :]
    c_t = ca_true[:, 2:-1, :]
    d_t = ca_true[:, 3:, :]

    # Valid quadruplet: all four on the same chain and in ab_mask
    valid = (
        ab_mask[:, :-3] & ab_mask[:, 1:-2] & ab_mask[:, 2:-1] & ab_mask[:, 3:]
        & (chain_type[:, :-3] == chain_type[:, 1:-2])
        & (chain_type[:, 1:-2] == chain_type[:, 2:-1])
        & (chain_type[:, 2:-1] == chain_type[:, 3:])
    ).float()  # [B, N-3]

    cos_pred = _cos_dihedral(a, b, c, d)          # [B, N-3]
    cos_true = _cos_dihedral(a_t, b_t, c_t, d_t)  # [B, N-3]

    per_quad = F.smooth_l1_loss(cos_pred, cos_true, reduction="none")  # [B, N-3]

    n_valid = valid.sum(-1).clamp(min=1.0)
    loss = (per_quad * valid).sum(-1) / n_valid  # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# L1: Kabsch-Aligned Antibody RMSD Loss  (Self-RMSD / TMscore improvement)
# ─────────────────────────────────────────────────────────────────────────────

def align_rmsd_loss(
    ca_pred: torch.Tensor,   # [B, N, 3]  predicted clean CA coords (nm)
    ca_true: torch.Tensor,   # [B, N, 3]  ground-truth clean CA coords (nm)
    ab_mask: torch.Tensor,   # [B, N]     bool, True = antibody position
) -> torch.Tensor:           # [B]        per-sample mean-squared displacement (nm²)
    """
    Kabsch-align predicted antibody CA to ground-truth, then compute mean
    squared displacement (proxy for RMSD²) over antibody residues.

    Uses existing kabsch_align() from align_utils — no extra code needed.
    Loss = 0 when prediction is a perfect rigid-body match to GT antibody.

    Returns per-sample loss [B].  The t_linear_weight ramp makes this loss
    concentrate supervision on near-clean (high-t) samples where x_1_pred
    is a meaningful coordinate estimate.
    """
    from proteinfoundation.utils.align_utils import kabsch_align

    ab_mask_bool = ab_mask.bool()                                          # [B, N]

    # Align ca_true → ca_pred frame; stop gradient through R so the loss
    # gradient is simply 2*(ca_pred - ca_true_aligned) — no SVD backprop.
    with torch.no_grad():
        ca_true_aligned = kabsch_align(ca_true, ca_pred, mask=ab_mask_bool)  # [B, N, 3]

    diff_sq = ((ca_pred - ca_true_aligned) ** 2).sum(-1)                  # [B, N]
    n_ab = ab_mask_bool.float().sum(-1).clamp(min=1.0)                    # [B]
    loss = (diff_sq * ab_mask_bool.float()).sum(-1) / n_ab                # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# L2: Interface-Aligned RMSD Loss  (iRMS-style, paratope supervision)
# ─────────────────────────────────────────────────────────────────────────────


def interface_align_loss(
    ca_pred: torch.Tensor,          # [B, N, 3]  predicted clean CA coords (nm)
    ca_true: torch.Tensor,          # [B, N, 3]  ground-truth clean CA coords (nm)
    paratope_mask: torch.Tensor,    # [B, N]     bool, True = paratope residue (precomputed)
) -> torch.Tensor:                  # [B]        per-sample loss
    """
    iRMS-style loss: Kabsch-align on antibody interface (paratope) residues,
    then compute mean squared displacement on those residues.

    Paratope mask is precomputed in ab_data.py (antibody residues within 10 Å
    of any antigen CA in the GT structure) — no redundant cdist at train time.

    Loss is 0 when the predicted paratope is a perfect rigid match to GT.

    Returns per-sample loss [B].
    """
    from proteinfoundation.utils.align_utils import kabsch_align

    interface_mask = paratope_mask.bool()   # [B, N]

    if not interface_mask.any():
        return ca_pred.new_zeros(ca_pred.shape[0])

    # Align ca_true → ca_pred frame; stop gradient through R.
    with torch.no_grad():
        ca_true_aligned = kabsch_align(ca_true, ca_pred, mask=interface_mask)  # [B,N,3]
    diff_sq = ((ca_pred - ca_true_aligned) ** 2).sum(-1)                   # [B,N]
    n_iface = interface_mask.float().sum(-1).clamp(min=1.0)                # [B]
    loss = (diff_sq * interface_mask.float()).sum(-1) / n_iface            # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# L3: Soft fnat Loss  (differentiable contact recovery, coordinates level)
# ─────────────────────────────────────────────────────────────────────────────

_FNAT_CONTACT_NM = 0.5    # 5 Å — DockQ fnat uses full-atom 5 Å; CA-level approximation
_FNAT_SIGMA_NM   = 0.05   # sigmoid sharpness (~half-width ≈ sigma)


def soft_fnat_loss(
    ca_pred: torch.Tensor,         # [B, N, 3]  predicted clean CA coords (nm)
    ca_true: torch.Tensor,         # [B, N, 3]  ground-truth clean CA coords (nm)
    ab_mask: torch.Tensor,         # [B, N]     bool, True = antibody
    ag_mask: torch.Tensor,         # [B, N]     bool, True = antigen
    contact_threshold_nm: float = _FNAT_CONTACT_NM,
    sigma_nm: float = _FNAT_SIGMA_NM,
) -> torch.Tensor:                 # [B]        per-sample loss
    """
    Soft (differentiable) fnat loss — coordinate-level, complementary to
    Priority B contact BCE which operates on pair_rep (representation level).

    For each (antibody × antigen) CA pair:
      GT contact  = hard step at contact_threshold_nm (5 Å)
      Pred contact = sigmoid((threshold - dist) / sigma)  (soft, differentiable)

    BCE between pred and GT contact maps, averaged over ab×ag pairs.
    Loss = 0 when predicted antibody places every residue exactly at GT.

    5 Å CA threshold mirrors DockQ's full-atom fnat definition at CA level.

    Returns per-sample loss [B].
    """
    pair_mask = ab_mask.bool().unsqueeze(-1) & ag_mask.bool().unsqueeze(-2)  # [B,N,N]

    if not pair_mask.any():
        return ca_pred.new_zeros(ca_pred.shape[0])

    # Ground-truth hard contact map (does NOT depend on ca_pred → no leakage)
    gt_dists   = torch.cdist(ca_true, ca_true)                       # [B,N,N]
    gt_contact = (gt_dists < contact_threshold_nm).float()           # [B,N,N]

    # Predicted soft contact logit (depends on ca_pred → carries gradient)
    # logit = (threshold - dist) / sigma  →  sigmoid(logit) is the soft contact prob
    # Use binary_cross_entropy_with_logits (autocast-safe) instead of BCE(sigmoid(logit))
    pred_dists  = torch.cdist(ca_pred, ca_true)                      # [B,N,N]
    pred_logits = (contact_threshold_nm - pred_dists) / sigma_nm     # [B,N,N]

    # BCE averaged over valid (ab × ag) pairs
    bce = F.binary_cross_entropy_with_logits(
        pred_logits,
        gt_contact,
        reduction="none",
    )                                                                 # [B,N,N]
    n_pairs = pair_mask.float().sum(dim=(-1, -2)).clamp(min=1.0)    # [B]
    loss = (bce * pair_mask.float()).sum(dim=(-1, -2)) / n_pairs    # [B]
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# L4: Steric Clash Loss  (predicted-ab vs true-ag CA repulsion)
# ─────────────────────────────────────────────────────────────────────────────

_CLASH_THRESHOLD_NM = 0.35  # 3.5 Å — two CA atoms (radius ≈ 1.7 Å each)


def steric_clash_loss(
    ca_pred: torch.Tensor,          # [B, N, 3]  predicted clean CA coords (nm)
    ca_true: torch.Tensor,          # [B, N, 3]  ground-truth clean CA coords (nm)
    ab_mask: torch.Tensor,          # [B, N]     bool, True = antibody
    ag_mask: torch.Tensor,          # [B, N]     bool, True = antigen
    clash_threshold_nm: float = _CLASH_THRESHOLD_NM,
) -> torch.Tensor:                  # [B]        per-sample clash penalty
    """
    Penalise predicted antibody CA atoms that clash (overlap) with ground-truth
    antigen CA atoms.

    Clash = relu(threshold - dist(pred_ab_CA, true_ag_CA))
    Normalised per antibody residue so loss scale is independent of chain length.

    Loss = 0 when no predicted antibody CA is within 3.5 Å of any antigen CA.
    Directly addresses the Clashes metric (6.1 in V2 vs 0.15 in IgGM).

    Returns per-sample loss [B].
    """
    pair_mask = ab_mask.bool().unsqueeze(-1) & ag_mask.bool().unsqueeze(-2)  # [B,N,N]

    if not pair_mask.any():
        return ca_pred.new_zeros(ca_pred.shape[0])

    # Distances between predicted antibody and true antigen CA
    dists = torch.cdist(ca_pred, ca_true)                             # [B,N,N]
    clash = F.relu(clash_threshold_nm - dists)                        # [B,N,N]

    n_ab = ab_mask.bool().float().sum(-1).clamp(min=1.0)             # [B]
    loss = (clash * pair_mask.float()).sum(dim=(-1, -2)) / n_ab      # [B]
    return loss
