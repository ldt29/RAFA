from typing import Dict

import torch
from torch.utils.checkpoint import checkpoint

from openfold.np.residue_constants import RESTYPE_ATOM37_MASK
from proteinfoundation.nn.feature_factory import FeatureFactory, get_time_embedding
from proteinfoundation.nn.modules.adaptive_ln_scale import AdaptiveLayerNormIdentical
from proteinfoundation.nn.modules.attn_n_transition import (
    MultiheadAttnAndTransition,
    MultiheadCrossAttnAndTransition,
)
from proteinfoundation.nn.modules.cdr_epitope_cross_attn import CDREpitopeCrossAttn
from proteinfoundation.nn.modules.contact_head import ContactHead
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.modules.pair_rep_initial import PairReprBuilder


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(
        dtype=torch.bool, device=device
    )  # [21, 37]


class PrivilegedInfoEncoder(torch.nn.Module):
    """Training-only encoder for the teacher's privileged clean structure.

    The public student continues to use the ordinary feature path.  When the
    batch asks for ``use_privileged_info`` this small zero-initialized adapter
    injects clean target-chain geometry/latents and sequence context into the
    shared trunk.  Keeping it as part of the same ``nn`` makes the checkpoint a
    single teacher--student lineage while preserving the student inference
    boundary.
    """

    def __init__(self, token_dim: int, pair_repr_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        seq_in_dim = 3 + self.latent_dim + 20 + 1
        self.seq_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(seq_in_dim),
            torch.nn.Linear(seq_in_dim, token_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(token_dim, token_dim, bias=False),
        )
        pair_in_dim = 32 + 1
        self.pair_proj = torch.nn.Sequential(
            torch.nn.LayerNorm(pair_in_dim),
            torch.nn.Linear(pair_in_dim, pair_repr_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(pair_repr_dim, pair_repr_dim, bias=False),
        )
        # Identity at initialization: enabling the architecture is checkpoint
        # safe, while the new adapter can learn through the teacher branch.
        torch.nn.init.zeros_(self.seq_proj[-1].weight)
        torch.nn.init.zeros_(self.pair_proj[-1].weight)

    @staticmethod
    def _one_hot(values: torch.Tensor, num_classes: int) -> torch.Tensor:
        out = torch.zeros(
            *values.shape,
            num_classes,
            device=values.device,
            dtype=torch.float32,
        )
        valid = (values >= 0) & (values < num_classes)
        out.scatter_(-1, values.clamp(0, num_classes - 1).unsqueeze(-1), 1.0)
        return out * valid.unsqueeze(-1).float()

    @staticmethod
    def _rbf(dists: torch.Tensor, dim: int = 32) -> torch.Tensor:
        centers = torch.linspace(
            0.1, 3.0, dim, device=dists.device, dtype=dists.dtype
        )
        return torch.exp(-((dists.unsqueeze(-1) - centers) ** 2) / 0.08)

    def forward(
        self, batch: Dict, full_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_mask = batch.get("mask", full_mask).bool() & full_mask
        x_1 = batch.get("x_1")
        if x_1 is None or "bb_ca" not in x_1 or "local_latents" not in x_1:
            raise ValueError("privileged teacher requires clean x_1 bb_ca/local_latents")

        clean_ca_target = x_1["bb_ca"]
        clean_latents = x_1["local_latents"]
        coords_nm = batch.get("coords_nm")
        if coords_nm is not None:
            clean_ca_full = torch.where(
                target_mask.unsqueeze(-1),
                clean_ca_target,
                coords_nm[:, :, 1, :],
            )
        else:
            clean_ca_full = clean_ca_target

        residue_type = batch.get("residue_type", batch.get("seq"))
        if residue_type is None:
            residue_type = torch.zeros_like(target_mask, dtype=torch.long)
        seq_oh = self._one_hot(residue_type.long(), 20)
        paratope = batch.get(
            "paratope_mask", torch.zeros_like(full_mask)
        ).float().unsqueeze(-1)
        seq_feat = torch.cat(
            [clean_ca_full, clean_latents, seq_oh, paratope], dim=-1
        )
        seq_bias = self.seq_proj(seq_feat) * full_mask.unsqueeze(-1)

        dists = torch.cdist(clean_ca_full, clean_ca_full)
        rbf = self._rbf(dists)
        epitope = batch.get("epitope_mask", torch.zeros_like(full_mask)).float()
        pair_extra = (
            paratope.squeeze(-1)[:, :, None] * epitope[:, None, :]
        ).unsqueeze(-1)
        pair_mask = full_mask[:, :, None] & full_mask[:, None, :]
        pair_bias = self.pair_proj(torch.cat([rbf, pair_extra], dim=-1))
        pair_bias = pair_bias * pair_mask.unsqueeze(-1)
        return seq_bias, pair_bias


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.

    New architecture features (ported from Proteina-Complexa-dev):
      - Arch-1: Differentiable intra-layer recycling (use_dr_seq, use_dr_pair)
      - Arch-2: Per-layer antigen cross-attention (use_antigen_cross_attn)
      - Arch-3: Triangle attention in PairReprUpdate (use_tri_attn, in pair_update.py)
      - Arch-4: Register tokens (num_registers)
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(LocalLatentsTransformer, self).__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_tri_attn = kwargs.get("use_tri_attn", False)
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]
        self.use_privileged_encoder = bool(
            kwargs.get("use_privileged_encoder", False)
        )
        self.privileged_scale = float(kwargs.get("privileged_scale", 1.0))

        # ── Arch-1: Differentiable Intra-layer Recycling ──────────────────
        self.t_emb_dim_diff_rec = 128
        self.use_dr_seq = kwargs.get("use_dr_seq", False)
        self.use_dr_pair = kwargs.get("use_dr_pair", False)

        # ── Arch-4: Register Tokens ───────────────────────────────────────
        num_registers = kwargs.get("num_registers", 0)
        if num_registers is None or num_registers <= 0:
            self.num_registers = 0
            self.registers = None
        else:
            self.num_registers = int(num_registers)
            self.registers = torch.nn.Parameter(
                torch.randn(self.num_registers, self.token_dim) / 20.0
            )

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        if self.use_privileged_encoder:
            self.privileged_encoder = PrivilegedInfoEncoder(
                token_dim=kwargs["token_dim"],
                pair_repr_dim=kwargs["pair_repr_dim"],
                latent_dim=kwargs["latent_dim"],
            )
        else:
            self.privileged_encoder = None

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            use_tri_attn=self.use_tri_attn,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        # ── CDR-Epitope explicit cross-attention (Gap 3 fix) ──────────────
        # Inserted after every `cdr_epitope_cross_attn_period` self-attn layers.
        # Default period=3, nlayers=14 → positions {2, 5, 8, 11} (0-indexed).
        # o_proj is zero-initialized → starts as identity; safe to finetune
        # from an existing checkpoint with strict=False.
        self.use_cdr_epitope_cross_attn = kwargs.get(
            "use_cdr_epitope_cross_attn", False
        )
        self.cdr_epitope_cross_attn_period = kwargs.get(
            "cdr_epitope_cross_attn_period", 3
        )

        if self.use_cdr_epitope_cross_attn:
            cross_attn_positions = set(
                range(
                    self.cdr_epitope_cross_attn_period - 1,
                    self.nlayers,
                    self.cdr_epitope_cross_attn_period,
                )
            )
            self.cross_attn_layers = torch.nn.ModuleDict(
                {
                    str(i): CDREpitopeCrossAttn(
                        token_dim=kwargs["token_dim"],
                        nheads=kwargs["nheads"],
                        dim_cond=kwargs["dim_cond"],
                        dropout=kwargs.get("cross_attn_dropout", 0.0),
                        use_geometric_encoder=kwargs.get(
                            "cross_attn_use_geometric_encoder", True
                        ),
                    )
                    for i in cross_attn_positions
                }
            )
            self._cross_attn_positions = cross_attn_positions
        else:
            self.cross_attn_layers = torch.nn.ModuleDict()
            self._cross_attn_positions = set()

        # ── Arch-2: Per-layer Antigen Cross-Attention ─────────────────────
        # At every layer, antibody tokens (Q) attend to antigen tokens (K/V).
        # This is complementary to the CDR-epitope cross-attn above (which is
        # CDR-only and sparse). New: full antibody sequence, every layer.
        self.use_antigen_cross_attn = kwargs.get("use_antigen_cross_attn", False)
        if self.use_antigen_cross_attn:
            self.antigen_cross_attn_layers = torch.nn.ModuleList(
                [
                    MultiheadCrossAttnAndTransition(
                        dim_token_a=self.token_dim,
                        dim_token_b=self.token_dim,
                        nheads=kwargs["nheads"],
                        dim_cond=kwargs["dim_cond"],
                        residual_mha=True,
                        residual_transition=True,
                        use_qkln=self.use_qkln,
                    )
                    for _ in range(self.nlayers)
                ]
            )

        # ── Arch-1: Differentiable Intra-layer Recycling modules ──────────
        # Intermediate CA prediction head for each layer (except last).
        # Predicts CA coords from current seqs → converts to RBF pair distances
        # → projects back to pair_rep (use_dr_pair) and/or seqs (use_dr_seq).
        # Always created (safe to load existing checkpoints with strict=False)
        # so that gradients and shapes are consistent; controlled by flags at runtime.
        self.linear_int = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.LayerNorm(self.token_dim),
                    torch.nn.Linear(self.token_dim, 3, bias=False),
                )
                for _ in range(self.nlayers - 1)
            ]
        )
        self.linear_int_dr_seq = torch.nn.ModuleList(
            [
                torch.nn.Linear(3, self.token_dim, bias=False)
                for _ in range(self.nlayers - 1)
            ]
        )
        self.adaln_seq_dr = torch.nn.ModuleList(
            [
                AdaptiveLayerNormIdentical(
                    dim=self.token_dim, dim_cond=self.t_emb_dim_diff_rec, mode="single"
                )
                for _ in range(self.nlayers - 1)
            ]
        )
        self.linear_int_dr_pair = torch.nn.ModuleList(
            [
                torch.nn.Linear(39, self.pair_repr_dim, bias=False)
                for _ in range(self.nlayers - 1)
            ]
        )
        self.adaln_pair_dr = torch.nn.ModuleList(
            [
                AdaptiveLayerNormIdentical(
                    dim=self.pair_repr_dim,
                    dim_cond=self.t_emb_dim_diff_rec,
                    mode="pair",
                )
                for _ in range(self.nlayers - 1)
            ]
        )

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

        # ── Priority B: Contact Head ───────────────────────────────────────
        # Lightweight MLP on final pair_rep → CDR-Epitope contact logits +
        # log-distance predictions.  Zero-init output layer → checkpoint-safe.
        # Requires update_pair_repr: True (otherwise pair_rep is constant).
        self.use_contact_head = kwargs.get("use_contact_head", False)
        if self.use_contact_head:
            self.contact_head = ContactHead(
                pair_repr_dim=self.pair_repr_dim,
                hidden_dim=kwargs.get("contact_head_hidden_dim", 64),
            )
        else:
            self.contact_head = None

    # ── Arch-4 helpers ────────────────────────────────────────────────────

    def _extend_w_registers(self, seqs, pair, mask, cond_seq):
        """Prepend register tokens to seq/pair/mask/cond (Arch-4).

        Args:
            seqs:     [b, n, token_dim]
            pair:     [b, n, n, pair_dim]
            mask:     [b, n]
            cond_seq: [b, n, dim_cond]

        Returns: (seqs, pair, mask, cond_seq) all extended by num_registers.
        """
        if self.num_registers == 0:
            return seqs, pair, mask, cond_seq

        b, n, _ = seqs.shape
        dim_pair = pair.shape[-1]
        r = self.num_registers
        dim_cond = cond_seq.shape[-1]

        # Expand registers across batch
        reg_expanded = self.registers[None, :, :].expand(b, -1, -1)  # [b, r, token_dim]
        seqs = torch.cat([reg_expanded, seqs], dim=1)  # [b, r+n, token_dim]

        # Extend mask (registers are always valid)
        true_tensor = torch.ones(b, r, dtype=torch.bool, device=seqs.device)
        mask = torch.cat([true_tensor, mask], dim=1)  # [b, r+n]

        # Extend pair with zeros: [b, n, n, d] → [b, r+n, r+n, d]
        zero_top = torch.zeros(b, r, n, dim_pair, device=seqs.device)
        pair = torch.cat([zero_top, pair], dim=1)  # [b, r+n, n, d]
        zero_left = torch.zeros(b, r + n, r, dim_pair, device=seqs.device)
        pair = torch.cat([zero_left, pair], dim=2)  # [b, r+n, r+n, d]

        # Extend cond with zeros
        zero_cond = torch.zeros(b, r, dim_cond, device=seqs.device)
        cond_seq = torch.cat([zero_cond, cond_seq], dim=1)  # [b, r+n, dim_cond]

        return seqs, pair, mask, cond_seq

    def _undo_registers(self, seqs, pair, mask):
        """Remove register tokens from seq/pair/mask (Arch-4)."""
        if self.num_registers == 0:
            return seqs, pair, mask
        r = self.num_registers
        return seqs[:, r:, :], pair[:, r:, r:, :], mask[:, r:]

    # ── Arch-1 helper ─────────────────────────────────────────────────────

    def _get_recycling_seq_n_pair(
        self,
        input: dict,
        bb_ca_int: torch.Tensor,
        layer_num: int,
        t_bb_ca: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> tuple:
        """Differentiable intra-layer recycling (Arch-1).

        Converts intermediate CA coordinate prediction into recycled sequence
        and pair representations via RBF distance encoding.

        Args:
            input:      batch dict (used for x_t and full_mask)
            bb_ca_int:  intermediate CA velocity prediction, shape [b, n, 3]
            layer_num:  index of current transformer layer
            t_bb_ca:    diffusion time for bb_ca modality, shape [b]
            loss_mask:  antibody-only mask [b, n] (full sequence length incl. antigen)

        Returns:
            rec_seq:  [b, n, token_dim]
            rec_pair: [b, n, n, pair_dim]
        """

        def _rbf_pair_dists(pair_dists_nm):
            """RBF encoding of pairwise distances using 39 Gaussian kernels."""
            centers = torch.linspace(0.1, 5, 39, device=pair_dists_nm.device)
            centers = centers[None, None, None, :]  # [1, 1, 1, 39]
            pair_dists = pair_dists_nm[..., None]  # [b, n, n, 1]
            return torch.exp(-((pair_dists - centers) ** 2) / 0.1)  # [b, n, n, 39]

        # If output parameterization is velocity "v", convert to clean sample x_1_pred
        if self.output_param["bb_ca"] == "v":
            t_brc = t_bb_ca[..., None, None]  # [b, 1, 1]
            bb_ca_int = input["x_t"]["bb_ca"] + bb_ca_int * (1.0 - t_brc)

        # Time embedding for conditioning the AdaLN
        t_emb_cond = get_time_embedding(t_bb_ca, self.t_emb_dim_diff_rec)  # [b, 128]

        # Pairwise distances — use loss_mask (antibody) for pair distances
        # (antigen is ground-truth, antibody is noisy; recycle over antibody)
        pair_mask = loss_mask[:, :, None] * loss_mask[:, None, :]  # [b, n, n]
        pair_dist = torch.norm(
            bb_ca_int[:, :, None, :] - bb_ca_int[:, None, :, :], dim=-1
        ) * pair_mask  # [b, n, n]
        rbf_pair_dist = _rbf_pair_dists(pair_dist) * pair_mask[..., None]  # [b, n, n, 39]

        # Sequence recycling: bb_ca_int [b, n, 3] → [b, n, token_dim]
        linear_out_seq = self.linear_int_dr_seq[layer_num](bb_ca_int)
        rec_seq = self.adaln_seq_dr[layer_num](linear_out_seq, t_emb_cond, loss_mask)

        # Pair recycling: rbf [b, n, n, 39] → [b, n, n, pair_dim]
        linear_out_pair = self.linear_int_dr_pair[layer_num](rbf_pair_dist)
        rec_pair = self.adaln_pair_dr[layer_num](linear_out_pair, t_emb_cond, pair_mask)

        return rec_seq, rec_pair

    # @torch.compile
    def forward(self, input: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean — antibody-only (loss mask)
        full_mask = input.get("full_mask", mask)  # all valid residues including antigen

        # Temporarily use full_mask for feature extraction so antigen is visible
        input["mask"] = full_mask

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, full_mask), full_mask)  # [b, n, dim_cond]

        # Initial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * full_mask[..., None]  # [b, n, token_dim]

        pair_rep = self.pair_repr_builder(input)  # [b, n, n, pair_dim]

        if input.get("use_privileged_info", False):
            if self.privileged_encoder is None:
                raise ValueError(
                    "Batch requested privileged teacher input, but the NN "
                    "was built without use_privileged_encoder."
                )
            priv_seq, priv_pair = self.privileged_encoder(input, full_mask)
            seqs = seqs + self.privileged_scale * priv_seq
            pair_rep = pair_rep + self.privileged_scale * priv_pair

        # Restore loss mask
        input["mask"] = mask

        # ── Arch-4: prepend register tokens ──────────────────────────────
        seqs, pair_rep, full_mask_ext, c_ext = self._extend_w_registers(
            seqs, pair_rep, full_mask, c
        )
        # Also extend loss mask for recycling (pad with False for registers)
        if self.num_registers > 0:
            b = mask.shape[0]
            r = self.num_registers
            false_pad = torch.zeros(b, r, dtype=torch.bool, device=mask.device)
            mask_ext = torch.cat([false_pad, mask], dim=1)
        else:
            mask_ext = mask
            full_mask_ext = full_mask
            c_ext = c

        # ── Pre-compute CDR-Epitope cross-attn inputs (outside loop) ──────
        if self.use_cdr_epitope_cross_attn:
            cdr_mask_xa     = input.get("cdr_mask")
            epitope_mask_xa = input.get("epitope_mask")
            ca_coords_xa    = input["x_t"]["bb_ca"]
            chain_type_xa   = input.get("chain_type")

            # Pooled conditioning: [b, n, dim_cond] → [b, dim_cond]
            c_pooled = (c * full_mask.unsqueeze(-1)).sum(dim=1) / (
                full_mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
            )  # [b, dim_cond]

            _can_use_xa = (
                cdr_mask_xa is not None
                and epitope_mask_xa is not None
                and chain_type_xa is not None
                and cdr_mask_xa.any()
                and epitope_mask_xa.any()
            )
        else:
            _can_use_xa = False

        # ── Arch-2: Pre-compute antigen representation (outside loop) ─────
        # Antigen tokens: chain_type == 3 (antigen) or use full_mask minus mask
        if self.use_antigen_cross_attn:
            chain_type = input.get("chain_type")  # [b, n] int or None
            if chain_type is not None:
                # Antigen mask: chain_type == 3 (antigen heavy/light = 1/2, antigen = 3)
                ag_mask_orig = (chain_type == 3) & full_mask  # [b, n]
            else:
                # Fallback: use full_mask minus loss mask
                ag_mask_orig = full_mask & (~mask)  # [b, n]

            # Extend ag_mask with False for registers if needed
            if self.num_registers > 0:
                b_sz = mask.shape[0]
                false_pad_ag = torch.zeros(
                    b_sz, self.num_registers, dtype=torch.bool, device=mask.device
                )
                ag_mask = torch.cat([false_pad_ag, ag_mask_orig], dim=1)  # [b, r+n]
            else:
                ag_mask = ag_mask_orig

            _can_use_ag_xa = ag_mask.any()
        else:
            _can_use_ag_xa = False
            ag_mask = None

        # Run trunk — attend over all residues (including antigen)
        for i in range(self.nlayers):
            # ── Arch-2: per-layer antigen cross-attention (BEFORE self-attn) ──
            if _can_use_ag_xa:
                # antibody+register mask (exclude antigen from Q)
                ab_mask_ext = full_mask_ext & (~ag_mask)  # [b, r+n]
                # Use c_ext conditioning; ag tokens act as K/V
                seqs = self.antigen_cross_attn_layers[i](
                    seqs, seqs, c_ext, ab_mask_ext, ag_mask
                )

            seqs = self.transformer_layers[i](
                seqs, pair_rep, c_ext, full_mask_ext
            )  # [b, r+n, token_dim]

            # ── CDR-Epitope cross-attention ────────────────────────────────
            if _can_use_xa and i in self._cross_attn_positions:
                # CDR-Epitope cross-attn operates on original (non-register-extended) indices.
                # We need to handle the register offset: slice off register prefix.
                if self.num_registers > 0:
                    r = self.num_registers
                    seqs_for_xa = seqs[:, r:, :]  # [b, n, token_dim]
                    seqs_for_xa = self.cross_attn_layers[str(i)](
                        seqs=seqs_for_xa,
                        c=c_pooled,
                        cdr_mask=cdr_mask_xa,
                        epitope_mask=epitope_mask_xa,
                        ca_coords=ca_coords_xa,
                        chain_type=chain_type_xa,
                    )
                    seqs = torch.cat([seqs[:, :r, :], seqs_for_xa], dim=1)
                else:
                    seqs = self.cross_attn_layers[str(i)](
                        seqs=seqs,
                        c=c_pooled,
                        cdr_mask=cdr_mask_xa,
                        epitope_mask=epitope_mask_xa,
                        ca_coords=ca_coords_xa,
                        chain_type=chain_type_xa,
                    )

            # ── Arch-1: Differentiable intra-layer recycling ───────────────
            if i < self.nlayers - 1:
                # Predict intermediate CA coordinates from current token reps
                # Operate on non-register tokens for recycling (slice off registers)
                if self.num_registers > 0:
                    r = self.num_registers
                    seqs_for_dr = seqs[:, r:, :]  # [b, n, token_dim]
                else:
                    seqs_for_dr = seqs

                bb_ca_int = self.linear_int[i](seqs_for_dr) * mask[..., None]  # [b, n, 3]

                rec_seq, rec_pair = checkpoint(
                    self._get_recycling_seq_n_pair,
                    *(input, bb_ca_int, i, input["t"]["bb_ca"], mask),
                    use_reentrant=False,
                )

                # Merge recycled features back (with optional zero coefficients)
                f_dr_seq = 1.0 if self.use_dr_seq else 0.0
                f_dr_pair = 1.0 if self.use_dr_pair else 0.0

                if self.num_registers > 0:
                    r = self.num_registers
                    # rec_seq is [b, n, token_dim] — pad with zeros for registers
                    zero_reg = torch.zeros(
                        rec_seq.shape[0], r, rec_seq.shape[-1], device=rec_seq.device
                    )
                    rec_seq_ext = torch.cat([zero_reg, rec_seq], dim=1)
                    seqs = seqs + rec_seq_ext * f_dr_seq

                    # rec_pair is [b, n, n, pair_dim] — pad with zeros for registers
                    b_sz, n_orig, _, d_pair = rec_pair.shape
                    zero_top = torch.zeros(b_sz, r, n_orig, d_pair, device=rec_pair.device)
                    rec_pair_ext = torch.cat([zero_top, rec_pair], dim=1)
                    zero_left = torch.zeros(b_sz, r + n_orig, r, d_pair, device=rec_pair.device)
                    rec_pair_ext = torch.cat([zero_left, rec_pair_ext], dim=2)
                    pair_rep = pair_rep + rec_pair_ext * f_dr_pair
                else:
                    seqs = seqs + rec_seq * f_dr_seq
                    pair_rep = pair_rep + rec_pair * f_dr_pair

            # ── Pair representation update ─────────────────────────────────
            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, full_mask_ext
                        )  # [b, r+n, r+n, pair_dim]

        # ── Arch-4: remove register tokens ───────────────────────────────
        seqs, pair_rep, full_mask_ext = self._undo_registers(seqs, pair_rep, full_mask_ext)

        # Get outputs — mask to antibody only for loss
        local_latents_out = (
            self.local_latents_linear(seqs) * mask[..., None]
        )  # [b, n, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n, 3]

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {
            self.output_param["local_latents"]: local_latents_out
        }

        # ── Priority B: Contact Head ──────────────────────────────────────
        # pair_rep here is [b, n, n, pair_repr_dim] (registers already removed)
        if self.contact_head is not None:
            contact_logits, log_dist_pred = self.contact_head(pair_rep)
            nn_out["contact_logits"] = contact_logits   # [b, n, n]
            nn_out["log_dist_pred"]  = log_dist_pred    # [b, n, n]

        return nn_out
