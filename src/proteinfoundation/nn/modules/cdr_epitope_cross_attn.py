"""
CDR-Epitope Explicit Cross-Attention block for antibody-antigen design.

CDR residues (queries) attend to epitope residues (keys/values).
Keys/values are enriched with a geometric position encoding derived from
antigen CA coordinates — giving the cross-attention novel geometric signal
beyond what the global self-attention already sees.

Architecture (Approach A+):
  - Lightweight: inserted every K self-attention layers (default K=3)
  - AdaLN conditioning on diffusion timestep for queries
  - Fixed LayerNorm for epitope K/V (epitope coords are always ground-truth)
  - EpitopeGeometricEncoder: CA coords + chain_type → MLP → KV enrichment
  - o_proj zero-initialized → starts as identity, supports finetune from checkpoint
  - Vectorized gather/scatter (no Python loop over batch dimension)

Self-conditioning compatibility:
  - Epitope CA coords are always ground-truth (fixed) regardless of noise level
  - Both SC forward passes see identical K/V → no special handling needed

Usage in LocalLatentsTransformer:
  Inserted after every K self-attention layers:
    seqs = cross_attn(seqs, c_pooled, cdr_mask, epitope_mask, ca_coords, chain_type)
  where c_pooled = masked_mean(c, full_mask)  # [B, dim_cond]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized gather / scatter utilities
# ─────────────────────────────────────────────────────────────────────────────


def _gather_masked_vectorized(
    tensor: torch.Tensor,      # [B, N, D]
    mask: torch.BoolTensor,    # [B, N]  True = include
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """
    Fully vectorized gather of masked positions into a dense padded tensor.

    Uses the cumsum trick to assign each True position a unique slot index
    within its row, then scatter_add_ to fill the output — no Python loop
    over the batch dimension.

    Args:
        tensor:  [B, N, D] source tensor
        mask:    [B, N] bool, True = positions to gather
        max_len: optional cap on output length L (clips if needed)

    Returns:
        gathered  [B, L, D] — zero-padded beyond valid positions
        pad_mask  [B, L]    — True where valid (non-pad)
    """
    B, N, D = tensor.shape

    # cumcount[b, n] = 0-based slot index for masked positions (−1 elsewhere)
    cumcount = mask.long().cumsum(dim=1) - 1   # [B, N]
    lengths = mask.sum(dim=1)                  # [B]
    L = int(lengths.max().item()) if max_len is None else max_len
    L = max(L, 1)  # guard against all-False mask

    # Clamp slot indices to [0, L-1]; out-of-range positions are gated by mask
    slot_idx = cumcount.clamp(0, L - 1)       # [B, N]

    # Prepare output buffers
    gathered = tensor.new_zeros(B, L, D)
    pad_mask = mask.new_zeros(B, L)

    # Scatter tensor values into gathered (each slot receives exactly one value)
    src = tensor * mask.unsqueeze(-1)                      # zero non-masked [B, N, D]
    slot_exp = slot_idx.unsqueeze(-1).expand(B, N, D)     # [B, N, D]
    gathered.scatter_add_(dim=1, index=slot_exp, src=src)

    # Build pad_mask (count True positions per slot — each gets exactly 1)
    ones = mask.long()                                     # [B, N]
    pad_long = mask.new_zeros(B, L, dtype=torch.long)
    pad_long.scatter_add_(dim=1, index=slot_idx, src=ones)
    pad_mask = pad_long.bool()                             # [B, L]

    return gathered, pad_mask


def _scatter_masked_vectorized(
    updates: torch.Tensor,       # [B, L, D] — updates for masked positions
    target: torch.Tensor,        # [B, N, D] — original full-sequence tensor
    mask: torch.BoolTensor,      # [B, N]    — True = positions to update
    pad_mask: torch.BoolTensor,  # [B, L]    — True = valid positions in updates
) -> torch.Tensor:
    """
    Scatter updates back into the target tensor at mask=True positions.

    Uses the same cumsum slot assignment as _gather_masked_vectorized.
    Non-masked positions in target are left unchanged.

    Args:
        updates:  [B, L, D] updated representations for CDR / masked positions
        target:   [B, N, D] original sequence tensor (all residues)
        mask:     [B, N] bool, True = positions to overwrite
        pad_mask: [B, L] bool, True = valid (non-pad) slots in updates

    Returns:
        [B, N, D] with mask=True positions replaced by updates
    """
    B, N, D = target.shape
    _, L, _ = updates.shape

    cumcount = mask.long().cumsum(dim=1) - 1  # [B, N]
    slot_idx = cumcount.clamp(0, L - 1)       # [B, N]

    # Gather the update values at each (b, n) masked position
    slot_exp = slot_idx.unsqueeze(-1).expand(B, N, D)     # [B, N, D]
    gathered_updates = updates.gather(dim=1, index=slot_exp)  # [B, N, D]

    # Apply: only overwrite where mask is True; zero-pad slots don't matter
    # because they are gated by mask below
    out = torch.where(
        mask.unsqueeze(-1),   # [B, N, 1]
        gathered_updates,     # updated CDR representations (no residual here)
        target,               # unchanged non-CDR positions
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Epitope Geometric Encoder
# ─────────────────────────────────────────────────────────────────────────────


class EpitopeGeometricEncoder(nn.Module):
    """
    Encodes epitope residue representations enriched with CA-coordinate geometry.

    Takes the running sequence representation of epitope residues plus their
    ground-truth CA coordinates and chain type, and produces a richer
    key/value representation for the cross-attention.

    This provides genuine geometric signal (3D interface shape) that goes
    beyond what the global self-attention — which sees noisy CDR coordinates —
    can already infer about the epitope.

    Args:
        seq_dim:       Input sequence feature dimension (= token_dim, e.g. 768)
        kv_dim:        Output K/V dimension (= token_dim, same)
        n_chain_types: Number of chain type classes (0=pad, 1=H, 2=L, 3=Ag)
        hidden_dim:    Hidden dimension for the coordinate MLP
    """

    def __init__(
        self,
        seq_dim: int = 768,
        kv_dim: int = 768,
        n_chain_types: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.chain_embed = nn.Embedding(n_chain_types, 16)

        # Encode (CA xyz + chain_type embedding) → seq_dim
        coord_in = 3 + 16
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, seq_dim),
        )

        # Fuse sequence features + geometric features → kv_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(seq_dim * 2),
            nn.Linear(seq_dim * 2, kv_dim),
            nn.SiLU(),
            nn.Linear(kv_dim, kv_dim),
        )

    def forward(
        self,
        epi_seqs: torch.Tensor,        # [B, L_epi, seq_dim] gathered epitope seqs
        epi_ca: torch.Tensor,          # [B, L_epi, 3]       ground-truth CA coords
        epi_chain: torch.LongTensor,   # [B, L_epi]          chain type (0–3)
    ) -> torch.Tensor:                 # [B, L_epi, kv_dim]
        chain_emb = self.chain_embed(
            epi_chain.clamp(0, self.chain_embed.num_embeddings - 1)
        )  # [B, L_epi, 16]

        coord_feat = self.coord_mlp(
            torch.cat([epi_ca, chain_emb], dim=-1)
        )  # [B, L_epi, seq_dim]

        fused = self.fusion(
            torch.cat([epi_seqs, coord_feat], dim=-1)
        )  # [B, L_epi, kv_dim]
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# CDR-Epitope Cross-Attention Block
# ─────────────────────────────────────────────────────────────────────────────


class CDREpitopeCrossAttn(nn.Module):
    """
    Explicit CDR→Epitope cross-attention block.

    CDR tokens (queries) attend to epitope tokens (keys/values).
    Only CDR positions in `seqs` are updated; all other positions are
    passed through unchanged.

    Design:
      - Queries:   CDR residue representations from running `seqs`
                   Pre-normalized with AdaLN(timestep conditioning c)
      - Keys/Values: Epitope residue representations + optional geometric
                   encoding from ground-truth CA coordinates
                   Pre-normalized with fixed LayerNorm (epitope is always GT)
      - Output:    o_proj zero-initialized → starts as identity residual
                   Gate = sigmoid(adaln_gate) applied to output
      - Self-cond: Epitope coords always GT → both SC passes see same K/V

    Args:
        token_dim:               Sequence feature dimension (e.g. 768)
        nheads:                  Number of attention heads (e.g. 12)
        dim_cond:                Conditioning (timestep embedding) dimension (e.g. 256)
        dropout:                 Attention dropout probability
        use_geometric_encoder:   Whether to enrich K/V with CA coordinate encoding
    """

    def __init__(
        self,
        token_dim: int = 768,
        nheads: int = 12,
        dim_cond: int = 256,
        dropout: float = 0.0,
        use_geometric_encoder: bool = True,
    ):
        super().__init__()
        assert token_dim % nheads == 0, (
            f"token_dim ({token_dim}) must be divisible by nheads ({nheads})"
        )
        self.token_dim = token_dim
        self.nheads = nheads
        self.head_dim = token_dim // nheads
        self.dropout_p = dropout
        self.use_geometric_encoder = use_geometric_encoder

        # ── AdaLN for CDR queries ─────────────────────────────────────────
        # Produces (shift_q, scale_q, gate) from pooled conditioning c [B, dim_cond]
        # Zero-initialized so cross-attn starts as identity (safe for finetuning)
        self.adaln_q = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim_cond, 3 * token_dim),
        )
        nn.init.zeros_(self.adaln_q[-1].weight)
        nn.init.zeros_(self.adaln_q[-1].bias)

        # elementwise_affine=False: AdaLN supplies its own γ, β
        self.norm_q = nn.LayerNorm(token_dim, elementwise_affine=False)

        # ── Fixed LayerNorm for epitope K/V ──────────────────────────────
        # Epitope is always ground-truth → no time-conditioning needed
        self.norm_kv = nn.LayerNorm(token_dim)

        # ── Q, K, V, O projections ────────────────────────────────────────
        self.q_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.k_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.v_proj = nn.Linear(token_dim, token_dim, bias=False)
        self.o_proj = nn.Linear(token_dim, token_dim, bias=False)
        # Zero-init o_proj: cross-attn starts as identity residual
        nn.init.zeros_(self.o_proj.weight)

        # ── Optional geometric K/V encoder ───────────────────────────────
        if use_geometric_encoder:
            self.epitope_encoder = EpitopeGeometricEncoder(
                seq_dim=token_dim,
                kv_dim=token_dim,
            )
        else:
            self.epitope_encoder = None

    # ─────────────────────────────────────────────────────────────────────
    # Helper: split / merge attention heads
    # ─────────────────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, D] → [B, H, L, Dh]"""
        B, L, _ = x.shape
        return x.view(B, L, self.nheads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, L, Dh] → [B, L, D]"""
        B, H, L, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        seqs: torch.Tensor,              # [B, N, D]  full sequence representation
        c: torch.Tensor,                 # [B, dim_cond]  pooled timestep conditioning
        cdr_mask: torch.BoolTensor,      # [B, N]  True = CDR position (query)
        epitope_mask: torch.BoolTensor,  # [B, N]  True = epitope position (key/value)
        ca_coords: torch.Tensor,         # [B, N, 3]  CA coords (antigen = GT always)
        chain_type: torch.LongTensor,    # [B, N]  chain type (1=H, 2=L, 3=Ag)
    ) -> torch.Tensor:                   # [B, N, D]  seqs with CDR positions updated
        """
        Run CDR-Epitope cross-attention and return updated sequence representations.

        Only CDR positions are modified in the output. Non-CDR, non-epitope
        positions are passed through unchanged.

        If a sample in the batch has no CDR or no epitope residues, it is
        passed through unchanged (guarded by the scatter operation).
        """
        B, N, D = seqs.shape

        # ── 1. Gather CDR and epitope tokens ─────────────────────────────
        cdr_seqs, cdr_pad, = _gather_masked_vectorized(seqs, cdr_mask)
        # cdr_seqs: [B, L_cdr, D],  cdr_pad: [B, L_cdr] (True = valid)

        epi_seqs, epi_pad = _gather_masked_vectorized(seqs, epitope_mask)
        # epi_seqs: [B, L_epi, D],  epi_pad: [B, L_epi]

        L_epi = epi_seqs.shape[1]

        epi_ca, _ = _gather_masked_vectorized(ca_coords, epitope_mask, max_len=L_epi)
        # epi_ca: [B, L_epi, 3]

        # chain_type: gather as int via float trick (avoid _gather on non-float)
        chain_f, _ = _gather_masked_vectorized(
            chain_type.unsqueeze(-1).float(), epitope_mask, max_len=L_epi
        )
        epi_chain = chain_f.squeeze(-1).long().clamp(0, 3)  # [B, L_epi]

        # ── 2. Compute K/V (with optional geometric enrichment) ───────────
        if self.epitope_encoder is not None:
            # Enrich K/V with CA-coordinate geometric encoding
            kv_input = self.epitope_encoder(epi_seqs, epi_ca, epi_chain)
        else:
            kv_input = epi_seqs                                # [B, L_epi, D]

        # ── 3. AdaLN pre-norm on CDR queries ─────────────────────────────
        # adaln_q output: [B, 3 * token_dim]
        adaln_out = self.adaln_q(c)                            # [B, 3D]
        shift_q, scale_q, gate_raw = adaln_out.chunk(3, dim=-1)
        shift_q  = shift_q.unsqueeze(1)    # [B, 1, D]
        scale_q  = scale_q.unsqueeze(1)    # [B, 1, D]
        gate_raw = gate_raw.unsqueeze(1)   # [B, 1, D]

        q_normed = self.norm_q(cdr_seqs)                       # [B, L_cdr, D]
        q_normed = q_normed * (1.0 + scale_q) + shift_q       # AdaLN modulation

        kv_normed = self.norm_kv(kv_input)                     # [B, L_epi, D]

        # ── 4. Project Q, K, V and split heads ───────────────────────────
        Q = self._split_heads(self.q_proj(q_normed))   # [B, H, L_cdr, Dh]
        K = self._split_heads(self.k_proj(kv_normed))  # [B, H, L_epi, Dh]
        V = self._split_heads(self.v_proj(kv_normed))  # [B, H, L_epi, Dh]

        # ── 5. Masked scaled dot-product attention ────────────────────────
        # epi_pad: True = valid epitope → we want to mask OUT padding (False)
        # PyTorch SDPA attn_bias: additive, -inf masks positions to ignore
        # Shape needed: [B, H, L_cdr, L_epi]
        epi_invalid = ~epi_pad                              # [B, L_epi] True = ignore
        attn_bias = torch.zeros(
            B, 1, 1, L_epi, device=seqs.device, dtype=seqs.dtype
        )
        attn_bias = attn_bias.masked_fill(
            epi_invalid.unsqueeze(1).unsqueeze(2), float("-inf")
        )  # [B, 1, 1, L_epi]

        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_bias,
            dropout_p=self.dropout_p if self.training else 0.0,
        )  # [B, H, L_cdr, Dh]

        # ── 6. Merge heads and project output ────────────────────────────
        attn_out = self._merge_heads(attn_out)       # [B, L_cdr, D]
        out = self.o_proj(attn_out)                  # [B, L_cdr, D]

        # ── 7. Gate + residual on CDR tokens only ────────────────────────
        gate = torch.sigmoid(gate_raw)               # [B, 1, D], ≈ 0.5 at init
        cdr_updated = cdr_seqs + gate * out          # [B, L_cdr, D]

        # Zero-out padded positions in the update (safety)
        cdr_updated = cdr_updated * cdr_pad.unsqueeze(-1)

        # ── 8. Scatter CDR updates back into full sequence ────────────────
        seqs_out = _scatter_masked_vectorized(cdr_updated, seqs, cdr_mask, cdr_pad)
        return seqs_out
