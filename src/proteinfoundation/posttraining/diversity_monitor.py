"""
Diversity monitoring for GRPO RL post-training.

Tracks antibody CDR sequence diversity to detect and prevent mode collapse —
a known failure mode of RL-trained generative models (Escalante 2026:
generated structures converge to alpha-helix bundles with extreme A/E enrichment).

Key metrics:
    1. Pairwise sequence identity (PSI) — collapses toward 1.0 on mode collapse
    2. Amino acid entropy — drops toward 0 on collapse
    3. CDR H3 sequence entropy — most sensitive single signal

Usage:
    monitor = DiversityMonitor(entropy_early_stop_bits=1.0, psi_warn_threshold=0.9)
    monitor.update(decoded_seqs, cdr_mask)
    monitor.check_stop()   # raises DiversityCollapseError if triggered

Or use standalone:
    from proteinfoundation.posttraining.diversity_monitor import (
        sequence_entropy_bits, pairwise_sequence_identity
    )
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Stateless helpers
# ─────────────────────────────────────────────────────────────────────────────

# Standard amino acid alphabet (20 canonical)
AA_VOCAB_SIZE = 20


def sequence_entropy_bits(seqs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Per-batch amino acid frequency entropy (Shannon, bits).

    Computes the empirical amino acid distribution over selected positions
    and returns its Shannon entropy in bits.

    Args:
        seqs: [B, N] int tensor of amino acid indices (0–19)
        mask: [B, N] bool — positions to include (e.g., CDR mask).
              If None, all positions are used.

    Returns:
        entropy: [B] in bits.  Healthy value ≈ 2.5 bits (for diverse CDR),
                 collapse < 1.0 bit (all same AA).
    """
    B, N = seqs.shape
    device = seqs.device

    # Build count tensors per sample, only over masked positions
    if mask is not None:
        # Clamp AA indices and zero out positions outside mask by giving
        # them a sentinel index (AA_VOCAB_SIZE) that falls outside [0, 19].
        aa_idx = seqs.clamp(0, AA_VOCAB_SIZE - 1)               # [B, N]
        # Replace non-CDR positions with a sentinel that we won't count
        sentinel = AA_VOCAB_SIZE
        aa_idx = torch.where(mask, aa_idx, torch.full_like(aa_idx, sentinel))
        n_bins = AA_VOCAB_SIZE + 1                               # last bin is sentinel
        n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
    else:
        aa_idx = seqs.clamp(0, AA_VOCAB_SIZE - 1)
        n_bins = AA_VOCAB_SIZE
        n_valid = float(N)

    # One-hot frequency count: [B, n_bins]
    counts = torch.zeros(B, n_bins, device=device, dtype=torch.float32)
    counts.scatter_add_(1, aa_idx, torch.ones_like(aa_idx, dtype=torch.float32))

    # Keep only the real AA bins [0, 19]
    counts = counts[:, :AA_VOCAB_SIZE]                           # [B, 20]

    probs = counts / n_valid                                     # [B, 20]
    # Only include non-zero probs in entropy to avoid 0*log(0) = NaN
    log_probs = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    entropy_nats = -(probs * log_probs).sum(dim=1)               # [B] nats
    entropy_bits = entropy_nats / math.log(2)                    # [B] bits
    return entropy_bits


def pairwise_sequence_identity(seqs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Mean pairwise sequence identity within the batch.

    Args:
        seqs: [B, N] int tensor of amino acid indices
        mask: [B, N] bool — positions to include

    Returns:
        mean_psi: scalar tensor — mean identity across all (i≠j) pairs in [0, 1]
    """
    B, N = seqs.shape
    if B < 2:
        return torch.tensor(0.0, device=seqs.device)

    if mask is not None:
        # Use first sample's mask as reference (assume same CDR layout in batch)
        ref_mask = mask[0]
        seqs_m = seqs[:, ref_mask]   # [B, N_cdr]
    else:
        seqs_m = seqs

    # Identity: fraction of positions where seq[i] == seq[j]
    N_m = seqs_m.shape[1]
    if N_m == 0:
        return torch.tensor(0.0, device=seqs.device)

    # [B, B] pairwise match count
    # seqs_m[:, None, :] == seqs_m[None, :, :] → [B, B, N_m]
    matches = (seqs_m.unsqueeze(1) == seqs_m.unsqueeze(0)).float()  # [B, B, N_m]
    identity = matches.mean(dim=-1)                                   # [B, B]

    # Average over off-diagonal pairs
    off_diag = ~torch.eye(B, dtype=torch.bool, device=seqs.device)
    mean_psi = identity[off_diag].mean()
    return mean_psi


def aa_frequency_divergence(seqs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    KL divergence of current AA distribution from uniform (bits).

    A value close to 0 means uniform / diverse; high value means enrichment
    toward specific amino acids (e.g., extreme A/E bias from RL collapse).

    Returns:
        kl_from_uniform: [B]
    """
    B, N = seqs.shape
    device = seqs.device

    if mask is not None:
        aa_idx = seqs.clamp(0, AA_VOCAB_SIZE - 1)
        sentinel = AA_VOCAB_SIZE
        aa_idx = torch.where(mask, aa_idx, torch.full_like(aa_idx, sentinel))
        n_bins  = AA_VOCAB_SIZE + 1
        n_valid = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    else:
        aa_idx  = seqs.clamp(0, AA_VOCAB_SIZE - 1)
        n_bins  = AA_VOCAB_SIZE
        n_valid = float(N)

    counts = torch.zeros(B, n_bins, device=device, dtype=torch.float32)
    counts.scatter_add_(1, aa_idx, torch.ones_like(aa_idx, dtype=torch.float32))
    counts = counts[:, :AA_VOCAB_SIZE]  # drop sentinel bin

    probs   = counts / n_valid           # [B, 20]
    uniform = torch.full_like(probs, 1.0 / AA_VOCAB_SIZE)
    # Only compute KL where probs > 0
    log_ratio = torch.where(probs > 0, (probs / uniform.clamp(min=1e-10)).log(), torch.zeros_like(probs))
    kl = (probs * log_ratio).sum(dim=1)  # [B] nats
    return kl / math.log(2)              # bits


# ─────────────────────────────────────────────────────────────────────────────
# Stateful monitor
# ─────────────────────────────────────────────────────────────────────────────

class DiversityCollapseError(RuntimeError):
    """Raised when diversity collapse is detected, triggering early stopping."""
    pass


class DiversityMonitor:
    """
    Stateful diversity tracker for GRPO RL training.

    Logs diversity metrics every `log_every` calls to `update()`.
    Raises `DiversityCollapseError` on `check_stop()` if thresholds are exceeded.

    Args:
        entropy_early_stop_bits:  CDR H3 entropy threshold for early stopping (bits).
                                  Healthy ~ 2.5 bits; collapse threshold = 1.0 bits.
        psi_warn_threshold:       Mean pairwise identity threshold for a warning.
                                  Collapse signal at > 0.9.
        log_every:                Log metrics every N update calls.
        window:                   EMA window for smoothing metrics.
    """

    def __init__(
        self,
        entropy_early_stop_bits: float = 1.0,
        psi_warn_threshold: float = 0.9,
        log_every: int = 10,
        window: int = 5,
    ):
        self.entropy_early_stop_bits = entropy_early_stop_bits
        self.psi_warn_threshold = psi_warn_threshold
        self.log_every = log_every
        self.window = window

        self._step = 0
        self._entropy_ema: Optional[float] = None
        self._psi_ema: Optional[float] = None
        self._stop_triggered = False

    def update(
        self,
        decoded_seqs: Tensor,            # [B, N] int, amino acid indices
        cdr_mask: Optional[Tensor] = None,  # [B, N] bool
    ) -> dict:
        """
        Update diversity metrics with a new batch of generated sequences.

        Args:
            decoded_seqs: [B, N] int tensor — full-sequence amino acid indices
            cdr_mask:     [B, N] bool — CDR positions (for entropy / PSI).
                          If None, all positions are used.

        Returns:
            metrics dict with current values.
        """
        self._step += 1
        device = decoded_seqs.device

        entropy = sequence_entropy_bits(decoded_seqs, cdr_mask).mean().item()
        psi = pairwise_sequence_identity(decoded_seqs, cdr_mask).item()
        kl_uniform = aa_frequency_divergence(decoded_seqs, cdr_mask).mean().item()

        # EMA smoothing
        alpha = 2.0 / (self.window + 1)
        if self._entropy_ema is None:
            self._entropy_ema = entropy
            self._psi_ema = psi
        else:
            self._entropy_ema = alpha * entropy + (1 - alpha) * self._entropy_ema
            self._psi_ema     = alpha * psi     + (1 - alpha) * self._psi_ema

        metrics = {
            "diversity/entropy_bits":     entropy,
            "diversity/entropy_ema_bits": self._entropy_ema,
            "diversity/mean_psi":         psi,
            "diversity/mean_psi_ema":     self._psi_ema,
            "diversity/kl_from_uniform":  kl_uniform,
        }

        if self._step % self.log_every == 0:
            logger.info(
                f"[diversity step={self._step}] "
                f"entropy={entropy:.2f} bits (ema={self._entropy_ema:.2f}), "
                f"psi={psi:.3f} (ema={self._psi_ema:.3f}), "
                f"kl_uniform={kl_uniform:.3f} bits"
            )

        # Warnings
        if self._psi_ema is not None and self._psi_ema > self.psi_warn_threshold:
            logger.warning(
                f"[diversity] High sequence identity: mean_psi_ema={self._psi_ema:.3f} "
                f"> threshold={self.psi_warn_threshold:.2f}. Possible mode collapse."
            )

        if self._entropy_ema is not None and self._entropy_ema < self.entropy_early_stop_bits:
            self._stop_triggered = True
            logger.error(
                f"[diversity] Collapse detected: entropy_ema={self._entropy_ema:.2f} bits "
                f"< stop_threshold={self.entropy_early_stop_bits:.2f} bits. "
                "Early stop triggered."
            )

        return metrics

    def check_stop(self) -> None:
        """
        Raise DiversityCollapseError if early-stop threshold has been triggered.

        Call this after each update to stop training on collapse.
        """
        if self._stop_triggered:
            raise DiversityCollapseError(
                f"Diversity collapse: CDR entropy < {self.entropy_early_stop_bits} bits. "
                "Stopping RL training to prevent degenerate outputs."
            )

    def reset(self) -> None:
        """Reset monitor state (e.g., after a curriculum change)."""
        self._step = 0
        self._entropy_ema = None
        self._psi_ema = None
        self._stop_triggered = False