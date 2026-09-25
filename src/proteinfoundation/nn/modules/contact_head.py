"""
ContactHead: lightweight MLP on pair_rep for CDR-Epitope distance prediction.

Analogous to dyMEAN's edge_dist_ffn (ed_loss).

dyMEAN predicts inter-residue distances from pairs of hidden states:
    p_dist = ffn(cat[H_i, H_j]) + ffn(cat[H_j, H_i])   # permutation-invariant

We operate directly on pair_rep[b, i, j] (already encodes pairwise info):
    logit[b, i, j] = MLP(pair_rep[b, i, j])

This is simpler and avoids an extra aggregation step since our pair_rep is
already an explicit pair representation, unlike dyMEAN's node hidden states.

Two prediction targets (both from pair_rep → single MLP, two output dims):
  - contact logit  → binary classification, BCE loss (is CDR_i within 8 Å of Epitope_j?)
  - log-distance   → regression, smooth_l1 loss (predict log(dist) for softness)

Zero-init output projection → safe to finetune from existing checkpoint
(starts as identity = no contact prediction, gracefully degrades to no-op).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ContactHead(nn.Module):
    """
    Predicts CDR-Epitope contact probabilities and distances from pair_rep.

    Applied to the final pair_rep after the transformer trunk.
    Only CDR × Epitope positions are used during loss computation
    (the head outputs [B, N, N] but only CDR×Epitope pairs matter).

    Architecture:
        LayerNorm(pair_repr_dim) → Linear(pair_repr_dim, hidden_dim) → ReLU
        → Linear(hidden_dim, 2)   # [contact_logit, log_dist]

    The output linear is zero-initialized so the head starts as an identity
    (no-op) — safe to load from an existing checkpoint with strict=False.

    Args:
        pair_repr_dim: Dimension of the pair representation (default 256).
        hidden_dim:    Hidden MLP dimension (default 64).
    """

    def __init__(self, pair_repr_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(pair_repr_dim)
        self.fc1  = nn.Linear(pair_repr_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.fc2  = nn.Linear(hidden_dim, 2)   # [contact_logit, log_dist]

        # Zero-init: starts as no-op, safe for checkpoint loading
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pair_rep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pair_rep: [B, N, N, pair_repr_dim]

        Returns:
            contact_logits: [B, N, N]  unnormalised logit for contact classification
            log_dist_pred:  [B, N, N]  predicted log(distance_nm + eps) for regression
        """
        x = self.act(self.fc1(self.norm(pair_rep)))   # [B, N, N, hidden_dim]
        out = self.fc2(x)                              # [B, N, N, 2]
        contact_logits = out[..., 0]                   # [B, N, N]
        log_dist_pred  = out[..., 1]                   # [B, N, N]
        return contact_logits, log_dist_pred
