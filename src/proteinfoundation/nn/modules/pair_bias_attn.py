# MIT License

# Copyright (c) 2022 MattMcPartlon

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional

import torch
from einops import rearrange
from torch import Tensor, einsum, nn

from proteinfoundation.nn.modules.adaptive_ln_scale import (
    AdaptiveLayerNorm,
    AdaptiveOutputScale,
)


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class PairBiasAttention(nn.Module):
    """
    Scalar Feature masked attention with pair bias and gating.
    Code modified from
    https://github.com/MattMcPartlon/protein-docking/blob/main/protein_learning/network/modules/node_block.py
    """

    def __init__(
        self,
        node_dim: int,
        dim_head: int,
        heads: int,
        bias: bool,
        dim_out: int,
        qkln: bool,
        pair_dim: Optional[int] = None,
        **kawrgs,  # noqa
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.node_dim, self.pair_dim = node_dim, pair_dim
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_qkv = nn.Linear(node_dim, inner_dim * 3, bias=bias)
        self.to_g = nn.Linear(node_dim, inner_dim)
        self.to_out_node = nn.Linear(inner_dim, default(dim_out, node_dim))
        self.node_norm = nn.LayerNorm(node_dim)
        self.q_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        self.k_layer_norm = nn.LayerNorm(inner_dim) if qkln else nn.Identity()
        if exists(pair_dim):
            self.to_bias = nn.Linear(pair_dim, heads, bias=False)
            self.pair_norm = nn.LayerNorm(pair_dim)
        else:
            self.to_bias, self.pair_norm = None, None

    def forward(
        self,
        node_feats: Tensor,
        pair_feats: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        """Multi-head scalar Attention Layer

        :param node_feats: scalar features of shape (b,n,d_s)
        :param pair_feats: pair features of shape (b,n,n,d_e)
        :param mask: boolean tensor of node adjacencies
        :return:
        """
        assert exists(self.to_bias) or not exists(pair_feats)
        node_feats, h = self.node_norm(node_feats), self.heads
        pair_feats = self.pair_norm(pair_feats) if exists(pair_feats) else None
        q, k, v = self.to_qkv(node_feats).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        g = self.to_g(node_feats)
        b = (
            rearrange(self.to_bias(pair_feats), "b ... h -> b h ...")
            if exists(pair_feats)
            else 0
        )
        q, k, v, g = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h), (q, k, v, g)
        )
        attn_feats = self._attn(q, k, v, b, mask)
        attn_feats = rearrange(
            torch.sigmoid(g) * attn_feats, "b h n d -> b n (h d)", h=h
        )
        return self.to_out_node(attn_feats)

    def _attn(self, q, k, v, b, mask: Optional[Tensor]) -> Tensor:
        """Perform attention update"""
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b i j -> b () i j")
            sim = sim.masked_fill(~mask, max_neg_value(sim))
        attn = torch.softmax(sim + b, dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token, dim_cond=dim_cond)

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class CrossAttention(nn.Module):
    """
    Cross-attention: sequence a attends to sequence b (with gating).
    Ported from Proteina-Complexa-dev.

    Q comes from a, K/V come from b.
    Output has same shape as a.
    """

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        dim_head_a: int,
        dim_head_b: int,
        heads: int,
        bias: bool,
        qkln: bool,
        **kwargs,
    ):
        super().__init__()
        inner_dim_a = dim_head_a * heads
        inner_dim_b = dim_head_b * heads
        self.dim_a, self.dim_b = dim_a, dim_b
        self.heads = heads
        self.scale_b = dim_head_b**-0.5

        self.to_q_a = nn.Linear(dim_a, inner_dim_b, bias=bias)
        self.to_g_a = nn.Linear(dim_a, inner_dim_a)
        self.to_v_b = nn.Linear(dim_b, inner_dim_a, bias=bias)
        self.to_k_b = nn.Linear(dim_b, inner_dim_b, bias=bias)
        self.to_out_node_a = nn.Linear(inner_dim_a, dim_a)

        self.node_norm_a = nn.LayerNorm(dim_a)
        self.node_norm_b = nn.LayerNorm(dim_b)
        self.q_layer_norm_a = nn.LayerNorm(inner_dim_b) if qkln else nn.Identity()
        self.k_layer_norm_b = nn.LayerNorm(inner_dim_b) if qkln else nn.Identity()

    def forward(
        self,
        feat_a: Tensor,
        feat_b: Tensor,
        mask_a_b: Optional[Tensor],
    ) -> Tensor:
        """
        Args:
            feat_a: [b, na, dim_a]  — query sequence
            feat_b: [b, nb, dim_b]  — key/value sequence
            mask_a_b: [b, na, nb]   — True = valid pair

        Returns:
            Updated feat_a, shape [b, na, dim_a].
        """
        h = self.heads
        feat_a = self.node_norm_a(feat_a)
        feat_b = self.node_norm_b(feat_b)

        q_a = self.q_layer_norm_a(self.to_q_a(feat_a))
        k_b = self.k_layer_norm_b(self.to_k_b(feat_b))
        v_b = self.to_v_b(feat_b)
        g_a = self.to_g_a(feat_a)

        q_a, k_b, v_b, g_a = map(
            lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=h),
            (q_a, k_b, v_b, g_a),
        )

        # Scaled dot-product attention
        sim = einsum("b h i d, b h j d -> b h i j", q_a, k_b) * self.scale_b
        if mask_a_b is not None:
            sim = sim.masked_fill(
                rearrange(~mask_a_b, "b i j -> b () i j"), max_neg_value(sim)
            )
        attn = torch.softmax(sim, dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v_b)

        # Gating + output projection
        out = rearrange(torch.sigmoid(g_a) * out, "b h n d -> b n (h d)", h=h)
        return self.to_out_node_a(out)


class MultiHeadCrossAttentionADALN_MM(torch.nn.Module):
    """Multi-head cross-attention with AdaLN on the query sequence and
    adaptive output scaling.  Ported from Proteina-Complexa-dev.

    Query sequence (a) attends to key/value sequence (b).
    Only sequence a is updated.
    """

    def __init__(self, dim_token_a, dim_token_b, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head_a = int(dim_token_a // nheads)
        dim_head_b = int(dim_token_b // nheads)
        self.adaln_a = AdaptiveLayerNorm(dim=dim_token_a, dim_cond=dim_cond)
        self.ln_b = nn.LayerNorm(dim_token_b)
        self.mha = CrossAttention(
            dim_a=dim_token_a,
            dim_b=dim_token_b,
            dim_head_a=dim_head_a,
            dim_head_b=dim_head_b,
            heads=nheads,
            bias=True,
            qkln=use_qkln,
        )
        self.scale_output = AdaptiveOutputScale(dim=dim_token_a, dim_cond=dim_cond)

    def forward(self, a, b, cond, mask_a, mask_b):
        """
        Args:
            a:      [b, na, dim_token_a]  query sequence
            b:      [b, nb, dim_token_b]  key/value sequence
            cond:   [b, na, dim_cond]     conditioning variables (for query AdaLN)
            mask_a: [b, na]               binary mask for a
            mask_b: [b, nb]               binary mask for b

        Returns:
            Updated a, shape [b, na, dim_token_a].
        """
        mask_a_b = mask_a[:, :, None] * mask_b[:, None, :]  # [b, na, nb]
        a = self.adaln_a(a, cond, mask_a)
        b = self.ln_b(b) * mask_b[..., None]
        a = self.mha(a, b, mask_a_b)
        a = self.scale_output(a, cond, mask_a)
        return a * mask_a[..., None]
