"""
GRPO-style RL post-training for the antibody latent flow matching model.

Algorithm (Escalante 2026, adapted for FM):
    1. Given an epitope-conditioned batch, generate K antibody samples using
       the current model (no_grad, fast ODE with nsteps_sample steps).
    2. Compute reward r_k for each sample (fnat proxy by default).
    3. Compute an in-group advantage.  ``GRPOTrainer`` uses the historical
       mean baseline; ``GRAFTTrainer`` replaces it with an alpha-quantile
       baseline aligned with best-of-N deployment.
    4. Clip advantages to [-advantage_clip, +advantage_clip] (PPO-lite stability).
    5. Re-noise each accepted sample (fresh x_0 ~ N(0,I), t ~ Uniform(0,1)),
       forward through the model WITH gradients, compute per-sample FM loss.
    6. Gradient step:  L_GRPO = 1/K * Σ_k  A_k * fm_loss_k.  Since FM loss
       is a positive squared error, gradient descent pulls positive-reward
       samples closer and pushes negative-reward samples away.

Key design choices:
    - GRPO needs no reference model; GRAFT can opt into a frozen FM-loss
      reference snapshot for its trust-region penalty.
    - Antigen coordinates are always locked (same as training/inference).
    - RL sampling uses nsteps_sample=50 (fast); gradient FM loss uses any t.
    - pair_update_layers optionally frozen to save memory (recommended).

Usage:
    trainer = GRPOTrainer(proteina_model, cfg, reward_fn=fnat_proxy_reward)
    trainer.train(dataloader)

or step-by-step:
    for batch in dataloader:
        metrics = trainer.step(batch)
        trainer.log(metrics)
"""

from __future__ import annotations

import math
import os
import copy
import json
from functools import partial
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import Adam, SGD

from proteinfoundation.posttraining.reward_fns import fnat_proxy_reward, get_reward_fn
from proteinfoundation.posttraining.diversity_monitor import DiversityMonitor, DiversityCollapseError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_nn_only_lineage(model, path: str) -> Dict:
    """Load and audit an nn-only post-training initialization artifact.

    The shared protein-base artifact contains the student trunk plus ten
    teacher-only ``privileged_encoder.*`` keys.  The antibody Proteina model
    intentionally instantiates only the student path, so those ten keys are
    the sole permitted unexpected keys.  Failing closed here prevents an
    unrelated architecture mismatch from becoming an undocumented lineage.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("nn_state"), dict):
        raise ValueError(f"nn-only lineage has no dict nn_state: {path}")

    contract = payload.get("shared_weight_contract")
    artifact_type = payload.get("artifact_type")
    if artifact_type == "protein_base_teacher_student_nn_only_weight":
        if contract != "prod_pretrain_state_dict_nn_prefix":
            raise ValueError(
                "shared protein-base artifact has an unexpected weight contract: "
                f"{contract!r} ({path})"
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"shared protein-base artifact has no state_dict: {path}")
        expected = {f"nn.{key}": value for key, value in payload["nn_state"].items()}
        if set(expected) != set(state_dict):
            raise ValueError(
                "shared protein-base nn_state/state_dict key mismatch: "
                f"nn_state={len(expected)} state_dict={len(state_dict)}"
            )

    bad_tensors = [
        key for key, value in payload["nn_state"].items()
        if not isinstance(value, Tensor) or not torch.isfinite(value).all().item()
    ]
    if bad_tensors:
        raise ValueError(f"nn-only lineage contains non-finite/non-tensor values: {bad_tensors[:5]}")

    missing, unexpected = model.nn.load_state_dict(payload["nn_state"], strict=False)
    allowed_unexpected = [key for key in unexpected if key.startswith("privileged_encoder.")]
    disallowed_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    if missing or disallowed_unexpected:
        raise ValueError(
            "nn-only lineage is incompatible with the RAFA student model: "
            f"missing={missing[:8]} unexpected={disallowed_unexpected[:8]}"
        )
    provenance = {
        "path": os.path.abspath(path),
        "artifact_type": artifact_type,
        "shared_weight_contract": contract,
        "lineage": payload.get("lineage"),
        "global_step": payload.get("global_step"),
        "loaded_nn_keys": len(payload["nn_state"]),
        "ignored_teacher_only_keys": allowed_unexpected,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    logger.info(
        "[GRPO] Loaded nn-only lineage %s (keys=%d, ignored teacher-only=%d)",
        provenance["path"], provenance["loaded_nn_keys"],
        len(allowed_unexpected),
    )
    return provenance

def _freeze_pair_update(model) -> None:
    """Freeze pair_update_layers parameters (most memory-expensive, O(N²·d))."""
    n_frozen = 0
    for name, param in model.nn.named_parameters():
        if "pair_update_layers" in name:
            param.requires_grad_(False)
            n_frozen += param.numel()
    logger.info(f"[GRPO] Frozen pair_update_layers: {n_frozen / 1e6:.2f}M params")


def _configure_trainable_scope(model, scope: str) -> None:
    """Restrict RL updates to an explicit NN parameter scope.

    The default ``all`` preserves the historical/formal trainer.  The
    opt-in ``output_heads`` scope is a structural drift diagnostic: the
    transformer trunk remains frozen while only the clean-sample heads and
    optional contact head receive GRAFT gradients.
    """
    scope = str(scope).lower()
    if scope == "all":
        return
    if scope != "output_heads":
        raise ValueError("trainable_scope must be 'all' or 'output_heads'")
    prefixes = ("local_latents_linear.", "ca_linear.", "contact_head.")
    n_trainable = 0
    n_frozen = 0
    for name, parameter in model.nn.named_parameters():
        keep = name.startswith(prefixes)
        parameter.requires_grad_(keep)
        if keep:
            n_trainable += parameter.numel()
        else:
            n_frozen += parameter.numel()
    if n_trainable == 0:
        raise ValueError("trainable_scope='output_heads' found no output-head parameters")
    logger.info(
        f"[GRPO] Trainable scope=output_heads: "
        f"trainable={n_trainable / 1e6:.2f}M frozen={n_frozen / 1e6:.2f}M"
    )


def _restore_antigen(x_t: Dict[str, Tensor], batch: Dict) -> None:
    """
    In-place: restore antigen CA to ground-truth and zero antigen latents.
    Mirrors the lock-antigen logic in training_step / predict_for_sampling.
    """
    chain_type = batch.get("chain_type")
    coords_nm  = batch.get("coords_nm")
    if chain_type is None or coords_nm is None:
        return

    if chain_type.dim() == 1:
        chain_type = chain_type.unsqueeze(0)
    if coords_nm.dim() == 3:
        coords_nm = coords_nm.unsqueeze(0)

    ag_mask = (chain_type == 3)  # [B, N]
    if not ag_mask.any():
        return

    ag_ca = coords_nm[:, :, 1, :]  # [B, N, 3] CA in nm
    x_t["bb_ca"] = torch.where(ag_mask.unsqueeze(-1), ag_ca, x_t["bb_ca"])
    if "local_latents" in x_t:
        x_t["local_latents"] = torch.where(
            ag_mask.unsqueeze(-1),
            torch.zeros_like(x_t["local_latents"]),
            x_t["local_latents"],
        )


def _renoise_sample(
    x_1: Dict[str, Tensor],
    ab_mask: Tensor,
    device: torch.device,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Re-noise a clean sample x_1 with fresh noise and uniform t.

    ``RDNFlowMatcher.compute_fm_loss`` uses a ``1 / (1 - t)^2`` clean-sample
    weighting.  The historical RL path samples the full ``[0, 1)`` interval;
    explicit bounds make a numerically safer truncated-time diagnostic
    possible without silently changing that historical default.

    Returns (x_0, x_t, t_dict) where:
        x_0:   [B, N, *]  Gaussian noise
        x_t:   [B, N, *]  interpolated noisy sample = (1-t)*x_0 + t*x_1
        t_dict: {"bb_ca": [B], "local_latents": [B]}  — same t for both modalities
    """
    B = next(iter(x_1.values())).shape[0]
    # Single t per sample (same for both modalities, simplest RL variant).
    t_scalar = t_min + (t_max - t_min) * torch.rand(B, device=device)

    x_0, x_t, t_dict = {}, {}, {}
    # Rollouts may carry non-differentiable annotations (for example the
    # autoencoder-decoded ``residue_type`` used by the sequence teacher).
    # Only the two continuous FM modalities participate in re-noising.
    for dm, x1 in x_1.items():
        if dm not in {"bb_ca", "local_latents"}:
            continue
        noise = torch.randn_like(x1)                         # [B, N, *]
        t_view = t_scalar.view(B, *([1] * (x1.dim() - 1)))  # [B, 1, ...]
        noised = (1 - t_view) * noise + t_view * x1          # [B, N, *]

        # Antibody only: antigen positions should be GT (handled by _restore_antigen)
        x_0[dm]    = noise
        x_t[dm]    = noised
        t_dict[dm] = t_scalar

    return x_0, x_t, t_dict


def _compute_per_sample_fm_loss(
    model,
    batch: Dict,
    x_1: Dict[str, Tensor],
    x_0: Dict[str, Tensor],
    x_t: Dict[str, Tensor],
    t_dict: Dict[str, Tensor],
    nn_override=None,
) -> Tensor:
    """
    Compute per-sample FM loss (bb_ca + local_latents) for a single generated x_1.

    Creates a shallow copy of batch with x_1/x_0/x_t/t injected so that
    compute_fm_loss can be called verbatim.  Using a copy (not in-place mutation)
    avoids the gradient-checkpoint recompute bug where batch is already restored
    by the time the recompute pass runs.

    Returns:
        per_sample_loss: [B]  (sum of bb_ca and local_latents losses)
    """
    # Restore antigen positions in x_t (model must see true antigen at every call)
    _restore_antigen(x_t, batch)

    # Build a shallow copy with the RL-specific keys injected.
    # Shallow copy is sufficient: tensors are not mutated, only the dict keys differ.
    batch_rl = {k: v for k, v in batch.items() if k not in ("x_1", "x_0", "x_t", "t", "x_sc")}
    batch_rl["x_1"] = x_1
    batch_rl["x_0"] = x_0
    batch_rl["x_t"] = x_t
    batch_rl["t"]   = t_dict
    # No self-conditioning during RL for simplicity

    # ``nn_override`` is used only for the optional GRAFT reference-model
    # trust-region check.  With no override this is exactly the historical
    # model.call_nn(n_recycle=0) path.
    nn_out = (
        model.call_nn(batch_rl, n_recycle=0)
        if nn_override is None
        else nn_override(batch_rl)
    )
    fm_losses = model.fm.compute_fm_loss(batch_rl, nn_out)
    # fm_losses: {"bb_ca": [B], "local_latents": [B]}

    per_sample = sum(fm_losses[dm] for dm in fm_losses)  # [B]
    return per_sample


# ─────────────────────────────────────────────────────────────────────────────
# Main GRPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GRPOTrainer:
    """
    GRPO-style RL post-training trainer for the antibody FM model.

    Args:
        model:              Proteina lightning module (FM + AE).
        rl_cfg:             OmegaConf DictConfig from configs/posttraining_rl.yaml
                            (posttraining sub-config).
        reward_fn:          Callable[(x_1_pred, batch) -> Tensor[B]].
                            Defaults to fnat_proxy_reward (Tier 1).
        diversity_monitor:  Optional DiversityMonitor instance.

    Config keys (rl_cfg):
        K:                    int   — samples per batch (default 8)
        nsteps_sample:        int   — ODE steps for RL sampling (default 50)
        lr:                   float — learning rate (default 1e-6 in the
                                      affinity-primary GRAFT config)
        max_iters:            int   — max gradient updates (default 5000)
        freeze_pair_update:   bool  — freeze pair_update_layers (default True)
        advantage_clip:       float — clamp advantage to ±this (default 3.0)
        advantage_normalize:  bool  — normalize each target's rollout-group
                                      advantage before clipping (default False)
        advantage_objective:  str   — ``signed_fm`` preserves the historical
                                      signed surrogate; ``exp_weighted_fm``
                                      uses nonnegative, per-target exponential
                                      weights as a stability diagnostic;
                                      ``positive_fm`` reinforces only samples
                                      above the GRAFT baseline
        advantage_temperature: float — temperature for ``exp_weighted_fm``
        entropy_penalty_weight: float — λ for entropy penalty (default 0.01)
        diversity_penalty_weight: float — λ for decoded CDR PSI penalty in the
                                      rollout reward (default 0; diagnostic-only
                                      monitor remains enabled separately)
        rollout_sc_scale_noise: float — optional override for the rollout SDE
                                       noise scale; ``None`` keeps the model
                                       inference configuration unchanged
        rollout_latent_spread_scale: float — opt-in multiplicative spread of
                                             candidate local latents around
                                             their pool mean on native CDRs;
                                             ``1`` preserves the sampler
        rollout_pool_multiplier: int — opt-in oversampling factor for a
                                      diversity-selected rollout pool; ``1``
                                      preserves the historical K-sample path
        rollout_pool_chunk_size: int — number of oversampled candidates to
                                      simulate together; ``1`` preserves the
                                      historical per-candidate path
        diversity_backtrack: bool — opt-in rollback/controller that rejects an
                                   update when the next rollout group crosses
                                   the PSI gate
        diversity_backtrack_max_retries: int — consecutive controller retries
                                              before stopping
        diversity_backtrack_lr_factor: float — multiplicative LR reduction on
                                               each rejected update
        diversity_backtrack_min_psi_delta: float — minimum increase in the
                                                   post-update PSI over the
                                                   pre-update rollout before
                                                   rejecting an update
        diversity_backtrack_use_instant_psi: bool — use instantaneous PSI for
                                                   rollback decisions; the
                                                   formal monitor still keeps
                                                   its EMA gate
        log_interval:         int   — log every N steps (default 10)
        ckpt_interval:        int   — save checkpoint every N steps (default 500)
        ckpt_dir:             str   — directory for checkpoints
        metrics_path:         str   — run-local JSONL metrics path
        decode_for_diversity: bool — decode candidates for CDR diversity checks
        rollout_initial_noise_design: str — ``iid`` preserves the historical
                                            sampler; ``antithetic`` pairs
                                            initial reference noises as z/-z
                                            for oversampled rollout pools
        rollout_step_noise_design: str — ``iid`` preserves independent SDE
                                         increments; ``antithetic`` pairs the
                                         Brownian increments across rollout
                                         slots for an oversampled pool
        rollout_sampling_mode: str — ``default`` preserves inference config;
                                      diagnostics may select an explicit flow
                                      integration mode
    """

    def __init__(
        self,
        model,                          # Proteina
        rl_cfg: DictConfig,
        reward_fn: Optional[Callable] = None,
        diversity_monitor: Optional[DiversityMonitor] = None,
        checkpoint_provenance: Optional[Dict] = None,
    ):
        self.model = model
        self.cfg = rl_cfg
        self.checkpoint_provenance = checkpoint_provenance or {}

        # ── Hyper-parameters ──────────────────────────────────────────────────
        self.K                      = int(rl_cfg.get("K", 8))
        self.nsteps_sample          = int(rl_cfg.get("nsteps_sample", 50))
        self.lr                     = float(rl_cfg.get("lr", 5e-6))
        self.optimizer_name         = str(rl_cfg.get("optimizer", "adam")).lower()
        self.optimizer_momentum     = float(rl_cfg.get("optimizer_momentum", 0.0))
        self.max_iters              = int(rl_cfg.get("max_iters", 5000))
        self.freeze_pair_update     = bool(rl_cfg.get("freeze_pair_update", True))
        self.advantage_clip         = float(rl_cfg.get("advantage_clip", 3.0))
        # Reward magnitudes are deliberately bounded on the affinity-primary
        # path.  Keep the historical raw advantage as the default, but expose
        # a reproducible scale for a stronger-update sensitivity arm instead
        # of silently changing the prospective configuration.
        self.advantage_scale        = float(rl_cfg.get("advantage_scale", 1.0))
        if self.advantage_scale <= 0.0:
            raise ValueError("advantage_scale must be positive")
        # Optional per-rollout normalization keeps the quantile-centred
        # estimator's update magnitude comparable across targets whose
        # affinity proxy has different dynamic ranges.  It is disabled by
        # default so historical/prospective arms remain byte-for-byte
        # configuration compatible; new experiments opt in explicitly.
        self.advantage_normalize   = bool(rl_cfg.get("advantage_normalize", False))
        self.advantage_objective   = str(
            rl_cfg.get("advantage_objective", "signed_fm")
        ).lower()
        self.advantage_temperature = float(
            rl_cfg.get("advantage_temperature", 0.1)
        )
        if self.advantage_objective not in {
            "signed_fm", "exp_weighted_fm", "positive_fm"
        }:
            raise ValueError(
                "advantage_objective must be 'signed_fm', 'exp_weighted_fm', "
                "or 'positive_fm'"
            )
        if self.advantage_temperature <= 0.0:
            raise ValueError("advantage_temperature must be positive")
        self.entropy_penalty_weight = float(rl_cfg.get("entropy_penalty_weight", 0.01))
        self.diversity_penalty_weight = float(rl_cfg.get("diversity_penalty_weight", 0.0))
        self.trainable_scope = str(rl_cfg.get("trainable_scope", "all")).lower()
        self.rollout_ema_decay = float(rl_cfg.get("rollout_ema_decay", 0.0))
        if self.rollout_ema_decay < 0.0 or self.rollout_ema_decay >= 1.0:
            raise ValueError("rollout_ema_decay must satisfy 0 <= decay < 1")
        self.psi_early_stop = bool(rl_cfg.get("psi_early_stop", False))
        # The formal historical arm keeps the original full uniform interval.
        # Prospective stability diagnostics can opt into a truncated interval
        # to avoid the RDN clean-sample weight singularity near t=1.
        self.renoise_t_min = float(rl_cfg.get("renoise_t_min", 0.0))
        self.renoise_t_max = float(rl_cfg.get("renoise_t_max", 1.0))
        if not 0.0 <= self.renoise_t_min < self.renoise_t_max <= 1.0:
            raise ValueError(
                "renoise_t_min/max must satisfy 0 <= min < max <= 1"
            )
        if self.diversity_penalty_weight < 0.0:
            raise ValueError("diversity_penalty_weight must be non-negative")
        self.log_interval           = int(rl_cfg.get("log_interval", 10))
        self.ckpt_interval          = int(rl_cfg.get("ckpt_interval", 500))
        self.ckpt_dir               = str(rl_cfg.get("ckpt_dir", "./store/rl_ckpts"))
        self.metrics_path           = str(
            rl_cfg.get("metrics_path", os.path.join(self.ckpt_dir, "metrics.jsonl"))
        )
        self.decode_for_diversity   = bool(rl_cfg.get("decode_for_diversity", False))
        self.rollout_pool_multiplier = int(
            rl_cfg.get("rollout_pool_multiplier", 1)
        )
        if self.rollout_pool_multiplier < 1:
            raise ValueError("rollout_pool_multiplier must be >= 1")
        if self.rollout_pool_multiplier > 1 and not self.decode_for_diversity:
            raise ValueError(
                "rollout_pool_multiplier > 1 requires decode_for_diversity=true"
            )
        self.rollout_pool_chunk_size = int(
            rl_cfg.get("rollout_pool_chunk_size", 1)
        )
        if self.rollout_pool_chunk_size < 1:
            raise ValueError("rollout_pool_chunk_size must be >= 1")
        self._last_rollout_pool_size = self.K * self.rollout_pool_multiplier
        self._last_rollout_selection_min_distance = 0.0
        self.diversity_backtrack = bool(rl_cfg.get("diversity_backtrack", False))
        self.diversity_backtrack_max_retries = int(
            rl_cfg.get("diversity_backtrack_max_retries", 3)
        )
        self.diversity_backtrack_lr_factor = float(
            rl_cfg.get("diversity_backtrack_lr_factor", 0.25)
        )
        self.diversity_backtrack_min_psi_delta = float(
            rl_cfg.get("diversity_backtrack_min_psi_delta", 0.0)
        )
        self.diversity_backtrack_use_instant_psi = bool(
            rl_cfg.get("diversity_backtrack_use_instant_psi", True)
        )
        if self.diversity_backtrack_max_retries < 0:
            raise ValueError("diversity_backtrack_max_retries must be non-negative")
        if not 0.0 < self.diversity_backtrack_lr_factor < 1.0:
            raise ValueError("diversity_backtrack_lr_factor must be in (0, 1)")
        if self.diversity_backtrack_min_psi_delta < 0.0:
            raise ValueError("diversity_backtrack_min_psi_delta must be non-negative")
        if self.diversity_backtrack and not self.decode_for_diversity:
            raise ValueError(
                "diversity_backtrack requires decode_for_diversity=true"
            )

        # ── Reward ────────────────────────────────────────────────────────────
        if reward_fn is None:
            tier_raw = rl_cfg.get("reward_tier", 1)
            # tier can be int (1, 2, 3) or string ("2a")
            tier = tier_raw if isinstance(tier_raw, str) else int(tier_raw)
            clash_weight = float(rl_cfg.get("clash_weight", 0.1))
            self.reward_fn = get_reward_fn(tier, clash_weight=clash_weight)
            logger.info(f"[GRPO] Using reward Tier {tier!r} (clash_weight={clash_weight})")
        else:
            self.reward_fn = reward_fn

        # ── Diversity monitor ─────────────────────────────────────────────────
        if diversity_monitor is None:
            self.monitor = DiversityMonitor(
                entropy_early_stop_bits=float(rl_cfg.get("entropy_early_stop_bits", 1.0)),
                psi_warn_threshold=float(rl_cfg.get("psi_warn_threshold", 0.9)),
                log_every=self.log_interval,
            )
        else:
            self.monitor = diversity_monitor

        # ── Architecture freeze ───────────────────────────────────────────────
        if self.freeze_pair_update:
            _freeze_pair_update(model)
        _configure_trainable_scope(model, self.trainable_scope)

        self.rollout_ema_nn = None
        if self.rollout_ema_decay > 0.0:
            # Rollouts can use a slow shadow of the trainable NN while the
            # gradient model continues to learn. Decay=0 preserves the
            # historical on-policy path and avoids a second model copy.
            self.rollout_ema_nn = copy.deepcopy(model.nn).to(model.device).eval()
            for parameter in self.rollout_ema_nn.parameters():
                parameter.requires_grad_(False)
            logger.info(
                f"[GRPO] Rollout EMA enabled: decay={self.rollout_ema_decay:.6g}"
            )

        # ── Optimizer (trainable params only) ─────────────────────────────────
        trainable = [p for p in model.nn.parameters() if p.requires_grad]
        logger.info(
            f"[GRPO] Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M"
        )
        if self.optimizer_name == "adam":
            self.optimizer = Adam(trainable, lr=self.lr)
        elif self.optimizer_name == "sgd":
            if self.optimizer_momentum < 0.0 or self.optimizer_momentum >= 1.0:
                raise ValueError("optimizer_momentum must satisfy 0 <= momentum < 1")
            self.optimizer = SGD(
                trainable,
                lr=self.lr,
                momentum=self.optimizer_momentum,
            )
        else:
            raise ValueError(
                f"unsupported optimizer={self.optimizer_name!r}; use 'adam' or 'sgd'"
            )
        self._optimizer_base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        self._backtrack_lr_scale = 1.0

        # ── sampling_model_args: copied from inf_cfg.model if available ────────
        # These control the ODE schedule / step params for full_simulation.
        if hasattr(model, "inf_cfg") and model.inf_cfg is not None:
            self.sampling_model_args = copy.deepcopy(dict(model.inf_cfg.model))
        else:
            # Fallback: minimal config matching inference_ab_design.yaml defaults
            self.sampling_model_args = _default_sampling_model_args()

        # Keep the historical sampling configuration by default, but make a
        # rollout-diversity diagnostic explicit and reproducible.  The
        # underlying RDN SDE reads ``sc_scale_noise`` from each modality's
        # simulation-step parameters.  Changing it here affects generation
        # only; the FM re-noising path has its own independently logged time
        # window.
        rollout_noise = rl_cfg.get("rollout_sc_scale_noise", None)
        self.rollout_sc_scale_noise = None if rollout_noise is None else float(rollout_noise)
        if self.rollout_sc_scale_noise is not None:
            if self.rollout_sc_scale_noise < 0.0:
                raise ValueError("rollout_sc_scale_noise must be non-negative")
            for mode_cfg in self.sampling_model_args.values():
                step_params = mode_cfg.get("simulation_step_params")
                if step_params is not None and "sc_scale_noise" in step_params:
                    step_params["sc_scale_noise"] = self.rollout_sc_scale_noise
            logger.info(
                f"[GRPO] Rollout SDE noise override: "
                f"sc_scale_noise={self.rollout_sc_scale_noise:.4g}"
            )
        self.rollout_latent_spread_scale = float(
            rl_cfg.get("rollout_latent_spread_scale", 1.0)
        )
        if self.rollout_latent_spread_scale < 1.0:
            raise ValueError("rollout_latent_spread_scale must be >= 1")
        if self.rollout_latent_spread_scale != 1.0:
            logger.info(
                "[GRPO] Rollout candidate latent spread: "
                f"scale={self.rollout_latent_spread_scale:.6g}"
            )
        self.rollout_sampling_mode = str(
            rl_cfg.get("rollout_sampling_mode", "default")
        ).lower()
        valid_sampling_modes = {"default", "vf", "sc", "vf_ss", "vf_ss_sc_sn"}
        if self.rollout_sampling_mode not in valid_sampling_modes:
            raise ValueError(
                "rollout_sampling_mode must be one of "
                f"{sorted(valid_sampling_modes)}"
            )
        if self.rollout_sampling_mode != "default":
            for mode_cfg in self.sampling_model_args.values():
                step_params = mode_cfg.get("simulation_step_params")
                if step_params is not None:
                    step_params["sampling_mode"] = self.rollout_sampling_mode
            logger.info(
                f"[GRPO] Rollout sampling-mode override: {self.rollout_sampling_mode}"
            )
        self.rollout_initial_noise_design = str(
            rl_cfg.get("rollout_initial_noise_design", "iid")
        ).lower()
        if self.rollout_initial_noise_design not in {"iid", "antithetic"}:
            raise ValueError(
                "rollout_initial_noise_design must be 'iid' or 'antithetic'"
            )
        if self.rollout_initial_noise_design != "iid":
            logger.info(
                "[GRPO] Rollout initial-noise design: "
                f"{self.rollout_initial_noise_design}"
            )
        self.rollout_step_noise_design = str(
            rl_cfg.get("rollout_step_noise_design", "iid")
        ).lower()
        if self.rollout_step_noise_design not in {"iid", "antithetic"}:
            raise ValueError(
                "rollout_step_noise_design must be 'iid' or 'antithetic'"
            )
        if self.rollout_step_noise_design != "iid":
            logger.info(
                "[GRPO] Rollout step-noise design: "
                f"{self.rollout_step_noise_design}"
            )

        self._step = 0
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(self.metrics_path)), exist_ok=True)

    def _fm_update_weights(self, advantages: Tensor) -> Tensor:
        """Convert group advantages into weights for the FM surrogate.

        The historical signed objective deliberately pushes negative-reward
        samples toward larger FM error.  The opt-in exponential objective is
        a bounded-direction diagnostic: every sample remains a nonnegative
        regression target, while per-target weights are normalized to mean one
        so the update scale is comparable to the signed path.

        ``positive_fm`` keeps only positive GRAFT advantages, normalizes their
        mean to one per target, and falls back to uniform weights for a tied or
        all-zero group. It removes the anti-reward direction without changing
        rollout generation or reward computation.
        """
        if self.advantage_objective == "signed_fm":
            return advantages
        if self.advantage_objective == "positive_fm":
            positive = advantages.clamp_min(0.0)
            positive_mean = positive.mean(dim=0, keepdim=True)
            normalized = positive / positive_mean.clamp_min(1e-6)
            return torch.where(
                positive_mean > 1e-6,
                normalized,
                torch.ones_like(positive),
            )
        scaled = (advantages / self.advantage_temperature).clamp(-20.0, 20.0)
        weights = torch.exp(scaled)
        return weights / weights.mean(dim=0, keepdim=True).clamp_min(1e-6)

    def _snapshot_nn_state_cpu(self) -> Dict[str, Tensor]:
        """Copy the trainable model state for an opt-in diversity rollback."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.model.nn.state_dict().items()
        }

    def _restore_nn_state_cpu(self, state: Dict[str, Tensor]) -> None:
        """Restore a CPU snapshot and reset optimizer history after rollback."""
        self.model.nn.load_state_dict(state, strict=True)
        if self.rollout_ema_nn is not None:
            # A rejected update must not survive in the rollout shadow.  The
            # next diversity probe should be generated from the same confirmed
            # state as the gradient model, otherwise rollback is incomplete.
            self.rollout_ema_nn.load_state_dict(self.model.nn.state_dict(), strict=True)
        # Adam moments encode the rejected direction; retaining them would
        # reintroduce the same update after a rollback.  Resetting state is
        # deliberately limited to the opt-in controller path.
        self.optimizer.state.clear()

    def _set_backtrack_lr_scale(self, scale: float) -> None:
        self._backtrack_lr_scale = float(scale)
        for group, base_lr in zip(self.optimizer.param_groups, self._optimizer_base_lrs):
            group["lr"] = base_lr * self._backtrack_lr_scale

    @torch.no_grad()
    def _update_rollout_ema(self) -> None:
        if self.rollout_ema_nn is None:
            return
        decay = self.rollout_ema_decay
        current_parameters = dict(self.model.nn.named_parameters())
        for name, shadow in self.rollout_ema_nn.named_parameters():
            shadow.mul_(decay).add_(current_parameters[name], alpha=1.0 - decay)
        current_buffers = dict(self.model.nn.named_buffers())
        for name, shadow in self.rollout_ema_nn.named_buffers():
            shadow.copy_(current_buffers[name])

    # ─────────────────────────────────────────────────────────────────────────
    # Core GRPO step
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _make_rollout_initial_noise(
        self,
        batch_copy: Dict,
        n: int,
        sample_count: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, Tensor]]:
        """Build an opt-in antithetic initial-noise block for a rollout chunk.

        The default ``iid`` path returns ``None`` and leaves
        ``full_simulation`` byte-compatible with its historical sampler.  The
        antithetic path samples the same Gaussian reference distribution once,
        then pairs each candidate's initial state with its negation.  It does
        not change marginal variance or the SDE schedule; it only makes the
        initial pool cover opposite latent directions instead of relying on
        independent draws that can all enter the same decoded mode.
        """
        if self.rollout_initial_noise_design == "iid" or sample_count <= 1:
            return None
        if self.rollout_initial_noise_design != "antithetic":
            raise RuntimeError(
                "unsupported rollout_initial_noise_design: "
                f"{self.rollout_initial_noise_design}"
            )

        mask = batch_copy.get("mask")
        noise = self.model.fm.sample_noise(
            n=n,
            shape=(sample_count * batch_size,),
            device=device,
            mask=mask,
        )
        paired_noise = {}
        half = (sample_count + 1) // 2
        for data_mode, value in noise.items():
            # ``full_simulation`` consumes sample-major layout:
            # [sample, batch, residue, feature].
            grouped = value.reshape(sample_count, batch_size, *value.shape[1:])
            positive = grouped[:half]
            paired = torch.cat((positive, -positive), dim=0)[:sample_count]
            paired_noise[data_mode] = paired.reshape(
                sample_count * batch_size, *value.shape[1:]
            )
        return paired_noise

    @torch.no_grad()
    def _make_rollout_step_noise(
        self,
        batch_copy: Dict,
        n: int,
        sample_count: int,
        batch_size: int,
        device: torch.device,
    ) -> Optional[Dict[str, Tensor]]:
        """Build paired Brownian paths for an oversampled rollout chunk.

        Initial-noise pairing alone does not control the independent SDE
        increments injected by ``sc`` at every Euler step. This opt-in path
        uses the same marginal Gaussian law but pairs each rollout slot with
        the negative increment at every step. ``iid`` returns ``None`` so the
        historical simulator remains unchanged.
        """
        if self.rollout_step_noise_design == "iid" or sample_count <= 1:
            return None
        if self.rollout_step_noise_design != "antithetic":
            raise RuntimeError(
                "unsupported rollout_step_noise_design: "
                f"{self.rollout_step_noise_design}"
            )

        mask = batch_copy.get("mask")
        noise = self.model.fm.sample_noise(
            n=n,
            shape=(self.nsteps_sample, sample_count * batch_size),
            device=device,
            mask=mask,
        )
        paired_noise = {}
        half = (sample_count + 1) // 2
        for data_mode, value in noise.items():
            # [step, sample, batch, residue, feature]
            grouped = value.reshape(
                self.nsteps_sample,
                sample_count,
                batch_size,
                *value.shape[2:],
            )
            positive = grouped[:, :half]
            paired = torch.cat((positive, -positive), dim=1)[:, :sample_count]
            paired_noise[data_mode] = paired.reshape(
                self.nsteps_sample,
                sample_count * batch_size,
                *value.shape[2:],
            )
        return paired_noise

    @torch.no_grad()
    def _generate_samples(self, batch: Dict) -> list[Dict[str, Tensor]]:
        """
        Generate K antibody structures from the current model.

        Uses full_simulation (Euler ODE, nsteps_sample steps) with antigen locked.
        All K samples share the same epitope condition (same batch).

        Returns:
            list of K dicts, each {"bb_ca": [B, N, 3], "local_latents": [B, N, 8]}
        """
        device = self.model.device
        B      = batch["mask"].shape[0]
        N      = batch["coords_nm"].shape[1]

        predict_fn = partial(
            self.model.predict_for_sampling,
            mode="full",
            n_recycle=0,
            nn_override=self.rollout_ema_nn,
        )

        pool_size = self.K * self.rollout_pool_multiplier
        samples = []
        chunk_size = min(self.rollout_pool_chunk_size, pool_size)
        for start in range(0, pool_size, chunk_size):
            sample_count = min(chunk_size, pool_size - start)
            # full_simulation squeezes B=1 batches — handle that
            batch_copy = {
                k: v.clone() if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            # Ensure mask and full_mask are present
            if "full_mask" not in batch_copy and "chain_type" in batch_copy:
                ct = batch_copy["chain_type"]
                batch_copy["full_mask"] = (ct > 0)
                batch_copy["mask"]      = (ct > 0) & (ct < 3)

            # The simulation API treats its leading dimension as the number
            # of independent samples.  Repeat each conditioning batch in
            # sample-major order when a diagnostic chunk contains >1 rollout;
            # the later split restores the historical list-of-[B,...] layout.
            if sample_count > 1:
                for key, value in list(batch_copy.items()):
                    if (
                        isinstance(value, Tensor)
                        and value.dim() > 0
                        and value.size(0) == B
                    ):
                        batch_copy[key] = value.repeat(
                            (sample_count,) + (1,) * (value.dim() - 1)
                        )

            initial_noise = self._make_rollout_initial_noise(
                batch_copy=batch_copy,
                n=N,
                sample_count=sample_count,
                batch_size=B,
                device=device,
            )
            step_noise = self._make_rollout_step_noise(
                batch_copy=batch_copy,
                n=N,
                sample_count=sample_count,
                batch_size=B,
                device=device,
            )

            x_gen, extra = self.model.fm.full_simulation(
                batch               = batch_copy,
                predict_for_sampling= predict_fn,
                nsteps              = self.nsteps_sample,
                nsamples            = B * sample_count,
                n                   = N,
                self_cond           = True,
                sampling_model_args = self.sampling_model_args,
                device              = device,
                save_trajectory_every = 0,
                guidance_w          = 1.0,
                ag_ratio            = 0.0,
                initial_noise       = initial_noise,
                step_noise          = step_noise,
            )
            # x_gen: {"bb_ca": [B*sample_count, N, 3], ...}.  Detach from any
            # accidental grad tape and restore one dict per rollout slot.
            for slot in range(sample_count):
                lo = slot * B
                hi = (slot + 1) * B
                samples.append({k: v[lo:hi].detach() for k, v in x_gen.items()})

        self._apply_rollout_latent_spread(samples, batch)
        self._last_rollout_pool_size = pool_size
        if self.rollout_pool_multiplier == 1:
            self._last_rollout_selection_min_distance = 0.0
            return samples
        return self._select_diverse_rollout_samples(samples, batch)

    @torch.no_grad()
    def _apply_rollout_latent_spread(
        self,
        samples: list[Dict[str, Tensor]],
        batch: Dict,
    ) -> None:
        """Spread candidate local latents around the pool mean on native CDRs.

        The stochastic initial and step-noise designs can be compressed by the
        learned flow before the decoder sees them. This opt-in proposal policy
        amplifies the between-candidate latent deviations only, preserving the
        pool mean and all non-CDR residues. It is applied before sequence
        decoding/selection and is disabled at the default scale of one.
        """
        scale = self.rollout_latent_spread_scale
        if scale == 1.0 or not samples or "local_latents" not in samples[0]:
            return
        latent_pool = torch.stack(
            [sample["local_latents"] for sample in samples], dim=0
        )
        center = latent_pool.mean(dim=0, keepdim=True)
        spread = center + scale * (latent_pool - center)

        ab_mask = batch["mask"].bool()
        cdr_mask = batch.get("native_cdr_mask", batch.get("cdr_mask"))
        if cdr_mask is None:
            cdr_mask = ab_mask
        if cdr_mask.dim() == 1:
            cdr_mask = cdr_mask.unsqueeze(0)
        cdr_mask = (ab_mask & cdr_mask.bool()).unsqueeze(0).unsqueeze(-1)
        spread = torch.where(cdr_mask, spread, latent_pool)
        for index, sample in enumerate(samples):
            sample["local_latents"] = spread[index].detach()

    @torch.no_grad()
    def _select_diverse_rollout_samples(
        self,
        pool: list[Dict[str, Tensor]],
        batch: Dict,
    ) -> list[Dict[str, Tensor]]:
        """Select K candidates by greedy native-CDR sequence dissimilarity.

        The pool is intentionally oversampled from the same current policy,
        then reduced without using rewards.  For each antigen-conditioned
        batch element, the first candidate is retained and each subsequent
        candidate maximizes its minimum Hamming distance to the already
        selected CDR sequences.  This creates an explicit diversity pressure
        before GRAFT computes its group-relative rewards; the default pool
        multiplier is one, so historical/formal arms are unchanged.
        """
        pool_size = len(pool)
        if pool_size < self.K:
            raise RuntimeError(
                f"rollout pool has {pool_size} samples but K={self.K}"
            )
        autoencoder = getattr(self.model, "autoencoder", None)
        if autoencoder is None:
            raise RuntimeError(
                "diversity-selected rollout pool requires the model autoencoder"
            )

        ab_mask = batch["mask"].bool()
        batch_size, n_res = ab_mask.shape
        cdr_mask = batch.get("native_cdr_mask", batch.get("cdr_mask"))
        if cdr_mask is None:
            cdr_mask = ab_mask
        if cdr_mask.dim() == 1:
            cdr_mask = cdr_mask.unsqueeze(0)
        cdr_mask = cdr_mask.bool()

        z = torch.cat([sample["local_latents"] for sample in pool], dim=0)
        ca = torch.cat([sample["bb_ca"] for sample in pool], dim=0)
        decoded = autoencoder.decode(
            z_latent=z,
            ca_coors_nm=ca,
            mask=ab_mask.repeat(pool_size, 1),
        )
        seqs = decoded["residue_type"].long().reshape(pool_size, batch_size, n_res)

        selected_by_batch: list[list[int]] = []
        selected_distances: list[float] = []
        for b in range(batch_size):
            mask_b = cdr_mask[b]
            if not bool(mask_b.any()):
                mask_b = ab_mask[b]
            if not bool(mask_b.any()):
                mask_b = torch.ones(n_res, dtype=torch.bool, device=seqs.device)

            selected = [0]
            available = torch.ones(pool_size, dtype=torch.bool, device=seqs.device)
            available[0] = False
            while len(selected) < self.K:
                candidate_ids = torch.nonzero(available, as_tuple=False).flatten()
                selected_ids = torch.tensor(
                    selected, dtype=torch.long, device=seqs.device
                )
                candidate_seqs = seqs[candidate_ids, b][:, mask_b]
                selected_seqs = seqs[selected_ids, b][:, mask_b]
                identity = (
                    candidate_seqs[:, None, :] == selected_seqs[None, :, :]
                ).float().mean(dim=-1)
                min_distance = 1.0 - identity.max(dim=1).values
                best_offset = int(min_distance.argmax().item())
                best_id = int(candidate_ids[best_offset].item())
                selected.append(best_id)
                available[best_id] = False
                selected_distances.append(float(min_distance[best_offset].item()))
            selected_by_batch.append(selected)

        # Batch elements may choose different pool members.  Reassemble each
        # selected rollout slot so reward/advantage tensors remain [K, B].
        selected_samples: list[Dict[str, Tensor]] = []
        for slot in range(self.K):
            assembled: Dict[str, Tensor] = {}
            for key in pool[0]:
                pieces = []
                for b in range(batch_size):
                    value = pool[selected_by_batch[b][slot]][key]
                    if not isinstance(value, Tensor):
                        raise TypeError(
                            f"rollout field {key!r} is not a Tensor; cannot select pool"
                        )
                    pieces.append(value[b : b + 1])
                assembled[key] = torch.cat(pieces, dim=0)
            selected_samples.append(assembled)

        self._last_rollout_selection_min_distance = (
            sum(selected_distances) / len(selected_distances)
            if selected_distances
            else 0.0
        )
        return selected_samples

    @torch.no_grad()
    def _decode_sequence_teacher_samples(
        self,
        samples: list[Dict[str, Tensor]],
        batch: Dict,
    ) -> None:
        """Attach frozen-AE decoded residue types for sequence-aware rewards."""
        if not getattr(self.reward_fn, "requires_sequence", False):
            return
        autoencoder = getattr(self.model, "autoencoder", None)
        if autoencoder is None:
            raise RuntimeError("sequence-aware reward requires the model autoencoder")
        z = torch.cat([sample["local_latents"] for sample in samples], dim=0)
        ca = torch.cat([sample["bb_ca"] for sample in samples], dim=0)
        ab_mask = batch["mask"].bool()
        decoded = autoencoder.decode(
            z_latent=z,
            ca_coors_nm=ca,
            mask=ab_mask.repeat(len(samples), 1),
        )
        residue_type = decoded["residue_type"].long()
        batch_size = ab_mask.shape[0]
        for index, sample in enumerate(samples):
            sample["residue_type"] = residue_type[
                index * batch_size : (index + 1) * batch_size
            ]

    def _compute_advantages(
        self,
        samples: list[Dict[str, Tensor]],
        batch: Dict,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute per-sample rewards and GRPO advantages.

        Returns:
            rewards:    [K, B]
            advantages: [K, B]  (centered, clipped)
        """
        # A sequence-affinity teacher is frozen and black-box, so it is safe
        # to decode the K rollout candidates under no_grad before scoring.  We
        # decode the concatenated batch once rather than invoking the
        # autoencoder separately for every candidate.
        self._decode_sequence_teacher_samples(samples, batch)
        with torch.no_grad():
            rewards_list = [
                self.reward_fn(x1, batch).float()  # [B]
                for x1 in samples
            ]
        rewards    = torch.stack(rewards_list, dim=0)                    # [K, B]
        # GRPO: center only, NO std normalization (Escalante 2026)
        advantages = rewards - rewards.mean(dim=0, keepdim=True)         # [K, B]
        advantages = advantages.clamp(-self.advantage_clip, self.advantage_clip)
        return rewards, advantages

    def _grpo_gradient_step(
        self,
        samples: list[Dict[str, Tensor]],
        advantages: Tensor,
        batch: Dict,
    ) -> float:
        """
        Compute GRPO loss and perform one gradient update.

        L_GRPO = 1/K * Σ_k  A_k * fm_loss_k

        Args:
            samples:    list of K dicts (generated x_1 samples)
            advantages: [K, B] centered & clipped advantages
            batch:      training batch (conditioning information)

        Returns:
            grpo_loss_scalar: Python float (for logging)
        """
        device = self.model.device
        ab_mask = batch["mask"].bool()  # [B, N]

        self.optimizer.zero_grad()
        total_loss_scalar = 0.0
        update_weights = self._fm_update_weights(advantages)

        for k, (x_1_k, a_k) in enumerate(zip(samples, update_weights)):
            # Re-noise the generated sample (decouple sampling from gradient)
            x_0, x_t, t_dict = _renoise_sample(
                x_1_k, ab_mask, device,
                t_min=self.renoise_t_min,
                t_max=self.renoise_t_max,
            )

            # Prepare x_1 as clean target (only antibody positions matter)
            x_1_train = {
                dm: v.clone()
                for dm, v in x_1_k.items()
                if dm in {"bb_ca", "local_latents"}
            }

            # Forward with grad
            per_sample_loss = _compute_per_sample_fm_loss(
                self.model, batch, x_1_train, x_0, x_t, t_dict
            )  # [B]

            # Signed FM surrogate: positive-advantage samples are pulled
            # toward lower FM error; negative-advantage samples are pushed
            # toward higher error.  This sign matches the positive squared
            # FM loss and the policy-gradient direction.
            # Divide by K here so the gradient contribution is 1/K per sample —
            # this is equivalent to accumulating gradients and calling backward
            # once on total_loss/K, but only keeps ONE computation graph alive
            # at a time (vs K graphs simultaneously → K× pair-rep memory).
            weighted = (a_k * per_sample_loss).mean() / self.K
            weighted.backward()                        # free graph immediately
            total_loss_scalar += weighted.item()

        grpo_loss = total_loss_scalar  # plain Python float, no graph

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.nn.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self._update_rollout_ema()

        return grpo_loss

    # ─────────────────────────────────────────────────────────────────────────
    # Single step (public API)
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, batch: Dict) -> Dict:
        """
        Run one GRPO step: generate → reward → advantage → gradient update.

        Args:
            batch: antibody design batch (must include chain_type, epitope_mask,
                   cdr_mask, coords_nm, mask, full_mask).

        Returns:
            metrics: dict with reward stats, loss, diversity info.
        """
        self._step += 1
        self.model.nn.train()
        self.model.autoencoder.eval()  # AE always frozen

        # 1. Generate K samples
        samples = self._generate_samples(batch)

        # 2. Compute rewards and advantages
        rewards, advantages = self._compute_advantages(samples, batch)
        reward_metrics = {}
        pop_metrics = getattr(self.reward_fn, "pop_metrics", None)
        if callable(pop_metrics):
            # AffinityAugmentedReward accumulates interpretable component
            # terms across the K rollouts.  Preserve them in the per-step
            # record without coupling the trainer to a specific reward class.
            reward_metrics = pop_metrics()

        # 3. Gradient update
        grpo_loss = self._grpo_gradient_step(samples, advantages, batch)

        # 4. Diversity monitoring (best sample in batch for proxy)
        # Use the sample with highest mean reward for diversity logging
        best_k = rewards.mean(dim=1).argmax().item()
        best_x1 = samples[best_k]

        metrics = {
            "rl/grpo_loss":       grpo_loss,
            "rl/reward_mean":     rewards.mean().item(),
            "rl/reward_max":      rewards.max().item(),
            "rl/reward_min":      rewards.min().item(),
            "rl/reward_std":      rewards.std().item(),
            "rl/advantage_mean":  advantages.mean().item(),
            "rl/advantage_objective": self.advantage_objective,
            "rl/advantage_temperature": self.advantage_temperature,
            "rl/rollout_sc_scale_noise": self.rollout_sc_scale_noise,
            "rl/rollout_latent_spread_scale": self.rollout_latent_spread_scale,
            "rl/rollout_sampling_mode": self.rollout_sampling_mode,
            "rl/rollout_initial_noise_design": self.rollout_initial_noise_design,
            "rl/rollout_step_noise_design": self.rollout_step_noise_design,
            "rl/rollout_ema_decay": self.rollout_ema_decay,
            "rl/rollout_pool_multiplier": self.rollout_pool_multiplier,
            "rl/rollout_pool_size": self._last_rollout_pool_size,
            "rl/rollout_pool_chunk_size": self.rollout_pool_chunk_size,
            "rl/rollout_selection_min_distance": self._last_rollout_selection_min_distance,
            "rl/diversity_backtrack": int(self.diversity_backtrack),
            "rl/diversity_backtrack_min_psi_delta": self.diversity_backtrack_min_psi_delta,
            "rl/optimizer_lr_scale": self._backtrack_lr_scale,
            "rl/step":            self._step,
        }
        metrics.update(reward_metrics)
        if self.decode_for_diversity:
            pre_diversity_metrics = self._diversity_metrics(samples, batch)
            metrics.update(pre_diversity_metrics)
            metrics.update({
                f"diversity/pre_update_{key[len('diversity/'):]}"
                if key.startswith("diversity/") else key: value
                for key, value in pre_diversity_metrics.items()
            })
            post_diversity_metrics = self._post_update_diversity_metrics(batch)
            metrics.update(post_diversity_metrics)
            if (
                "diversity/mean_psi" in pre_diversity_metrics
                and "diversity/post_update_mean_psi" in post_diversity_metrics
            ):
                metrics["diversity/post_update_psi_delta"] = (
                    float(post_diversity_metrics["diversity/post_update_mean_psi"])
                    - float(pre_diversity_metrics["diversity/mean_psi"])
                )
        metrics = self._augment_metrics(metrics)

        # Optional diversity metrics are decoded above when
        # ``decode_for_diversity`` is enabled; the default prospective config
        # enables them and the monitor is checked in the training loop.

        if self._step % self.log_interval == 0:
            graft_suffix = ""
            if "rl/graft_alpha" in metrics:
                graft_suffix = (
                    f"  alpha={metrics['rl/graft_alpha']:.3f}"
                    f"  trust={metrics['rl/graft_trust_penalty']:.4g}"
                )
            logger.info(
                f"[GRPO step={self._step}] "
                f"loss={grpo_loss:.4f}  "
                f"reward={rewards.mean().item():.4f} ± {rewards.std().item():.4f}  "
                f"adv_clip={advantages.abs().max().item():.2f}"
                f"{graft_suffix}"
            )

        # Keep a machine-readable per-update record alongside checkpoints so
        # learning curves and affinity subterms can be recomputed without
        # scraping loguru output.
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        # 5. Checkpoint
        if self._step % self.ckpt_interval == 0:
            self._save_checkpoint()

        return metrics

    def _augment_metrics(self, metrics: Dict) -> Dict:
        """Hook for estimator-specific metrics before logging/checkpointing."""
        return metrics

    @torch.no_grad()
    def _diversity_metrics(self, samples: list[Dict[str, Tensor]], batch: Dict) -> Dict:
        """Decode the rollout group and update the CDR diversity monitor.

        Formal RL uses batch size one, so the monitor compares the K candidates
        for the same antigen.  For larger diagnostic batches we update one
        group at a time and average the instantaneous metrics.
        """
        if not samples or "local_latents" not in samples[0]:
            return {}
        autoencoder = getattr(self.model, "autoencoder", None)
        if autoencoder is None:
            return {}
        z = torch.cat([sample["local_latents"] for sample in samples], dim=0)
        ca = torch.cat([sample["bb_ca"] for sample in samples], dim=0)
        ab_mask = batch["mask"].bool()
        mask = ab_mask.repeat(len(samples), 1)
        decoded = autoencoder.decode(z_latent=z, ca_coors_nm=ca, mask=mask)
        seqs = decoded["residue_type"].long()
        # ``cdr_mask`` is the *training mask* and is intentionally empty for
        # the mixed-mask ``none``/reconstruction branch.  Diversity must be
        # measured on the immutable native CDR positions, otherwise those
        # valid batches appear as entropy=0 / PSI=0 and can falsely trigger a
        # collapse gate.  Keep the fallback for older cached batches.
        cdr_mask = batch.get("native_cdr_mask", batch.get("cdr_mask"))
        if cdr_mask is not None:
            if cdr_mask.dim() == 1:
                cdr_mask = cdr_mask.unsqueeze(0)
            cdr_mask = cdr_mask.bool()

        # Concatenation is sample-major ([sample, batch]); gather each
        # antigen-conditioned group before computing PSI/entropy.
        batch_size = ab_mask.shape[0]
        group_metrics = []
        for b in range(batch_size):
            indices = torch.arange(
                b, len(samples) * batch_size, batch_size, device=seqs.device
            )
            cdr_b = None if cdr_mask is None else cdr_mask[b].unsqueeze(0).expand(len(indices), -1)
            group_metrics.append(self.monitor.update(seqs[indices], cdr_b))
        if len(group_metrics) == 1:
            return group_metrics[0]
        keys = group_metrics[0].keys()
        return {key: sum(float(item[key]) for item in group_metrics) / len(group_metrics) for key in keys}

    @torch.no_grad()
    def _post_update_diversity_metrics(self, batch: Dict) -> Dict:
        """Probe diversity after an update without mutating the main monitor.

        The historical controller inspected the rollout generated *before* the
        optimizer step and then attributed a high PSI to the preceding update.
        That is not causally valid: a zero-update base control can produce the
        same high-PSI rollout.  When the opt-in backtracking controller is
        enabled, generate a second rollout after the update and evaluate it
        with a cloned monitor.  The main monitor remains the pre-update
        training trace, while the namespaced metrics expose the actual
        post-update probe used for rollback decisions.
        """
        if not self.diversity_backtrack or not self.decode_for_diversity:
            return {}

        main_monitor = self.monitor
        pool_size = self._last_rollout_pool_size
        selection_distance = self._last_rollout_selection_min_distance
        try:
            self.monitor = copy.deepcopy(main_monitor)
            post_samples = self._generate_samples(batch)
            post_metrics = self._diversity_metrics(post_samples, batch)
        finally:
            self.monitor = main_monitor
            self._last_rollout_pool_size = pool_size
            self._last_rollout_selection_min_distance = selection_distance

        return {
            f"diversity/post_update_{key[len('diversity/'):]}" if key.startswith("diversity/") else key: value
            for key, value in post_metrics.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, dataloader, max_iters: Optional[int] = None) -> None:
        """
        Run the full GRPO training loop.

        Args:
            dataloader: iterable yielding antibody design batches (same format
                        as the normal training DataLoader).
            max_iters:  override for self.max_iters.
        """
        max_iters = max_iters or self.max_iters
        logger.info(
            f"[GRPO] Starting RL post-training: "
            f"K={self.K}, nsteps_sample={self.nsteps_sample}, "
            f"lr={self.lr}, max_iters={max_iters}"
        )

        device = self.model.device
        self.model.to(device)

        confirmed_state = (
            self._snapshot_nn_state_cpu() if self.diversity_backtrack else None
        )
        backtrack_retries = 0
        step = 0
        try:
            while step < max_iters:
                for batch in dataloader:
                    if step >= max_iters:
                        break

                    # Move batch to device
                    batch = _batch_to_device(batch, device)

                    # ``step`` records both the pre-update rollout and, when
                    # the controller is enabled, a separate post-update probe.
                    # The latter is the only signal used to reject an update;
                    # a high pre-update PSI can be an intrinsic property of
                    # the current policy, as shown by the LR=0 control.
                    monitor_before_step = (
                        copy.deepcopy(self.monitor.__dict__)
                        if self.diversity_backtrack
                        else None
                    )

                    metrics = self.step(batch)
                    step += 1

                    if self.diversity_backtrack:
                        psi_suffix = (
                            "mean_psi"
                            if self.diversity_backtrack_use_instant_psi
                            else "mean_psi_ema"
                        )
                        pre_psi_key = f"diversity/pre_update_{psi_suffix}"
                        post_psi_key = f"diversity/post_update_{psi_suffix}"
                        pre_psi_value = metrics.get(pre_psi_key)
                        psi_value = metrics.get(post_psi_key)
                        psi_crossed = (
                            psi_value is not None
                            and float(psi_value) > float(self.monitor.psi_warn_threshold)
                            and (
                                pre_psi_value is None
                                or float(psi_value) - float(pre_psi_value)
                                > self.diversity_backtrack_min_psi_delta
                            )
                        )
                        if psi_crossed:
                            if backtrack_retries < self.diversity_backtrack_max_retries:
                                assert confirmed_state is not None
                                assert monitor_before_step is not None
                                self._restore_nn_state_cpu(confirmed_state)
                                self.monitor.__dict__.clear()
                                self.monitor.__dict__.update(monitor_before_step)
                                backtrack_retries += 1
                                self._set_backtrack_lr_scale(
                                    self._backtrack_lr_scale
                                    * self.diversity_backtrack_lr_factor
                                )
                                logger.warning(
                                    f"[GRPO] diversity backtrack rejected update "
                                    f"step={self._step} {post_psi_key}={float(psi_value):.6f} "
                                    f"pre={float(pre_psi_value) if pre_psi_value is not None else float('nan'):.6f}; "
                                    f"retry={backtrack_retries}/"
                                    f"{self.diversity_backtrack_max_retries} "
                                    f"lr_scale={self._backtrack_lr_scale:.6g}"
                                )
                                continue
                            assert confirmed_state is not None
                            assert monitor_before_step is not None
                            self._restore_nn_state_cpu(confirmed_state)
                            self.monitor.__dict__.clear()
                            self.monitor.__dict__.update(monitor_before_step)
                            logger.error(
                                f"[GRPO] diversity backtrack exhausted at "
                                f"{post_psi_key}={float(psi_value):.6f}; stopping RL training."
                            )
                            self._save_checkpoint(tag="diversity_backtrack_stop")
                            return

                        # The post-update probe passed, so the state now in the
                        # model is the newest confirmed rollback point.  This
                        # must be snapshotted after the optimizer step; using
                        # the pre-update candidate would discard a valid update
                        # on the next rejection.
                        confirmed_state = self._snapshot_nn_state_cpu()
                        backtrack_retries = 0

                    # Diversity early stop check
                    try:
                        self.monitor.check_stop()
                    except DiversityCollapseError as e:
                        logger.error(f"[GRPO] {e}")
                        self._save_checkpoint(tag="collapse_stop")
                        return
                    if (
                        self.psi_early_stop
                        and metrics.get("diversity/mean_psi_ema") is not None
                        and float(metrics["diversity/mean_psi_ema"])
                        > float(self.monitor.psi_warn_threshold)
                    ):
                        psi = float(metrics["diversity/mean_psi_ema"])
                        threshold = float(self.monitor.psi_warn_threshold)
                        logger.error(
                            f"[GRPO] PSI diversity gate crossed: "
                            f"mean_psi_ema={psi:.6f} > threshold={threshold:.6f}; "
                            "stopping RL training."
                        )
                        self._save_checkpoint(tag="psi_stop")
                        return

        except KeyboardInterrupt:
            logger.info("[GRPO] Training interrupted by user.")
            self._save_checkpoint(tag="interrupt")
            return

        logger.info(f"[GRPO] Training completed after {step} steps.")
        self._save_checkpoint(tag="final")

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str = "") -> None:
        """Save model nn state dict as a GRPO checkpoint."""
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.ckpt_dir, f"grpo_step{self._step:06d}{suffix}.pt")
        if OmegaConf.is_config(self.cfg):
            rl_cfg = OmegaConf.to_container(self.cfg, resolve=True)
        else:
            rl_cfg = dict(self.cfg)
        torch.save(
            {
                "step":       self._step,
                "nn_state":   self.model.nn.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "trainer":    self.__class__.__name__,
                "rl_cfg":     rl_cfg,
                "reward_fn":  type(self.reward_fn).__name__,
                "checkpoint_provenance": self.checkpoint_provenance,
            },
            path,
        )
        logger.info(f"[GRPO] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume from a GRPO checkpoint."""
        ckpt = torch.load(path, map_location=self.model.device)
        self.model.nn.load_state_dict(ckpt["nn_state"], strict=False)
        if self.rollout_ema_nn is not None:
            self.rollout_ema_nn.load_state_dict(self.model.nn.state_dict(), strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
        logger.info(f"[GRPO] Resumed from {path} (step={self._step})")


class GRAFTTrainer(GRPOTrainer):
    """Budget-aligned affinity reward alignment for flow matching.

    GRAFT keeps the rollout, re-noising, and FM surrogate of the small GRPO
    trainer, but changes the credit assignment: the baseline is the group's
    ``alpha`` quantile rather than its mean.  With ``alpha=0.5`` this reduces
    to the median/standard group-relative estimator; with ``alpha`` close to
    ``1 - 1/K`` it focuses updates on candidates that can survive a
    best-of-K deployment readout.  Rewards are computed without native
    antibody coordinates on the affinity-primary path.

    This class is deliberately a thin variant so that comparisons differ only
    in the estimator and remain compatible with the existing checkpoint and
    data-loader plumbing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(self.cfg.get("alpha", 0.875))
        if not 0.0 <= self.alpha < 1.0:
            raise ValueError(f"GRAFT alpha must satisfy 0 <= alpha < 1, got {self.alpha}")
        self.trust_region_weight = float(self.cfg.get("trust_region_weight", 0.0))
        if self.trust_region_weight < 0.0:
            raise ValueError("GRAFT trust_region_weight must be non-negative")
        # The reference is an EMA-free frozen snapshot of the common starting
        # point.  It is opt-in because duplicating a 162M-parameter network is
        # unnecessary for the alpha-only ablation and expensive on CPU.
        self.reference_nn = None
        if self.trust_region_weight > 0.0:
            self.reference_nn = copy.deepcopy(self.model.nn).to(self.model.device).eval()
            for param in self.reference_nn.parameters():
                param.requires_grad_(False)
        self._last_baseline: Optional[Tensor] = None
        self._last_rewards: Optional[Tensor] = None
        self._last_trust_penalty = 0.0
        logger.info(
            f"[GRAFT] quantile baseline alpha={self.alpha:.3f} "
            f"(best-of-K target K={self.K}); "
            f"trust_region_weight={self.trust_region_weight:g}"
        )

    def _grpo_gradient_step(
        self,
        samples: list[Dict[str, Tensor]],
        advantages: Tensor,
        batch: Dict,
    ) -> float:
        """GRAFT surrogate with an optional FM-loss reference trust region."""
        if self.trust_region_weight <= 0.0:
            self._last_trust_penalty = 0.0
            return super()._grpo_gradient_step(samples, advantages, batch)

        device = self.model.device
        self.optimizer.zero_grad()
        total_loss_scalar = 0.0
        trust_penalty_scalar = 0.0
        update_weights = self._fm_update_weights(advantages)
        for x_1_k, a_k in zip(samples, update_weights):
            x_0, x_t, t_dict = _renoise_sample(
                x_1_k, batch["mask"].bool(), device,
                t_min=self.renoise_t_min,
                t_max=self.renoise_t_max,
            )
            x_1_train = {
                dm: v.clone()
                for dm, v in x_1_k.items()
                if dm in {"bb_ca", "local_latents"}
            }

            # Both losses see the identical (x_0, x_t, t) tuple.  The frozen
            # reference is evaluated without a graph; only the current model
            # loss contributes gradients to the update.
            with torch.no_grad():
                ref_loss = _compute_per_sample_fm_loss(
                    self.model, batch, x_1_train, x_0, x_t, t_dict,
                    nn_override=self.reference_nn,
                )
            cur_loss = _compute_per_sample_fm_loss(
                self.model, batch, x_1_train, x_0, x_t, t_dict,
            )
            drift_penalty = self.trust_region_weight * (cur_loss - ref_loss).square()
            weighted = (a_k * cur_loss + drift_penalty).mean() / self.K
            weighted.backward()
            total_loss_scalar += weighted.item()
            trust_penalty_scalar += drift_penalty.mean().item() / self.K

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.nn.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self._update_rollout_ema()
        self._last_trust_penalty = trust_penalty_scalar
        return total_loss_scalar

    def _compute_advantages(
        self,
        samples: list[Dict[str, Tensor]],
        batch: Dict,
    ) -> tuple[Tensor, Tensor]:
        """Return affinity rewards and quantile-centered, clipped advantages."""
        self._decode_sequence_teacher_samples(samples, batch)
        with torch.no_grad():
            rewards_list = [
                self.reward_fn(x1, batch).float()
                for x1 in samples
            ]
            rewards = torch.stack(rewards_list, dim=0)  # [K, B]
            self._last_diversity_penalty = torch.zeros((), device=rewards.device)
            if self.diversity_penalty_weight > 0.0 and all(
                "residue_type" in sample for sample in samples
            ):
                # Penalize within-rollout CDR identity without introducing a
                # native-sequence target.  This is a small auxiliary term on
                # top of the affinity-primary reward; the frozen sequence
                # teacher and structural affinity proxy remain the main score.
                cdr_mask = batch.get("native_cdr_mask", batch.get("cdr_mask"))
                if cdr_mask is not None:
                    cdr_mask = cdr_mask.bool()
                    if cdr_mask.dim() == 1:
                        cdr_mask = cdr_mask.unsqueeze(0)
                    seq = torch.stack(
                        [sample["residue_type"].long() for sample in samples], dim=0
                    )  # [K, B, N]
                    K, B, _ = seq.shape
                    penalty = torch.zeros((K, B), device=rewards.device)
                    for b in range(B):
                        selected = seq[:, b, cdr_mask[b]]
                        if selected.shape[-1] == 0 or K < 2:
                            continue
                        identity = (
                            selected[:, None, :] == selected[None, :, :]
                        ).float().mean(dim=-1)
                        identity.fill_diagonal_(0.0)
                        penalty[:, b] = identity.sum(dim=1) / float(K - 1)
                    rewards = rewards - self.diversity_penalty_weight * penalty
                    self._last_diversity_penalty = penalty.mean().detach()
            # ``torch.quantile`` handles K=1 and preserves a per-example
            # baseline, which is important when a batch contains different
            # antibody/antigen lengths or masking layouts.
            baseline = torch.quantile(
                rewards, q=self.alpha, dim=0, keepdim=True,
                interpolation="linear",
            )
            advantages = rewards - baseline
            if self.advantage_normalize:
                # Normalize within each target's rollout group, not across
                # the batch.  This preserves target-local ranking while
                # preventing high-variance affinity proxies from dominating
                # the FM surrogate.  The small floor keeps K=1 and tied
                # rewards finite; the subsequent clip remains the safety
                # boundary shared by all GRAFT arms.
                mean = advantages.mean(dim=0, keepdim=True)
                std = advantages.std(dim=0, keepdim=True, unbiased=False)
                advantages = (advantages - mean) / std.clamp_min(1e-6)
            advantages = (self.advantage_scale * advantages).clamp(
                -self.advantage_clip, self.advantage_clip
            )

        self._last_baseline = baseline.detach()
        self._last_rewards = rewards.detach()
        return rewards, advantages

    def _augment_metrics(self, metrics: Dict) -> Dict:
        if hasattr(self, "_last_diversity_penalty"):
            metrics["rl/diversity_penalty_weight"] = self.diversity_penalty_weight
            metrics["rl/diversity_penalty_mean"] = float(self._last_diversity_penalty.item())
        if self._last_baseline is not None:
            metrics["rl/graft_alpha"] = self.alpha
            metrics["rl/graft_advantage_scale"] = self.advantage_scale
            metrics["rl/graft_advantage_normalize"] = int(self.advantage_normalize)
            metrics["rl/graft_baseline"] = self._last_baseline.mean().item()
            metrics["rl/graft_top_tail_fraction"] = float(
                (self._last_rewards >= self._last_baseline).float().mean().item()
            )
            metrics["rl/graft_trust_penalty"] = self._last_trust_penalty
        return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _batch_to_device(batch: Dict, device: torch.device) -> Dict:
    """Recursively move all tensors in a batch dict to the given device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _batch_to_device(v, device)
        else:
            out[k] = v
    return out


def _default_sampling_model_args() -> Dict:
    """
    Fallback sampling_model_args matching inference_ab_design.yaml defaults.
    Used when inf_cfg is not set on the model.
    """
    step_params = {
        "sampling_mode": "sc",
        "sc_scale_noise": 0.1,
        "sc_scale_score": 1.0,
        "t_lim_ode": 0.98,
        "t_lim_ode_below": 0.02,
        "center_every_step": False,
    }
    return {
        "bb_ca": {
            "schedule": {"mode": "log", "p": 2.0},
            "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
            "simulation_step_params": step_params,
        },
        "local_latents": {
            "schedule": {"mode": "power", "p": 2.0},
            "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
            "simulation_step_params": dict(step_params),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point for GRPO RL post-training.

    Usage:
        python -m proteinfoundation.posttraining.grpo_trainer \
            --config_name posttraining_rl

    The config is loaded from configs/posttraining_rl.yaml via Hydra.
    The FM model checkpoint is loaded from cfg.ckpt_path / cfg.ckpt_name.
    The AE checkpoint from cfg.autoencoder_ckpt_path.
    """
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../../configs", config_name="posttraining_rl", version_base=None)
    def _run(cfg):
        import torch
        import random
        import numpy as np
        from omegaconf import OmegaConf
        from proteinfoundation.proteina import Proteina
        from proteinfoundation.datasets.ab_data import AntibodyDesignDataset, collate_fn
        from torch.utils.data import DataLoader

        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

        seed = int(cfg.get("seed", 5))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"[GRPO] Seed={seed}")

        # ── Load model ────────────────────────────────────────────────────────
        ckpt_path = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"base checkpoint does not exist: {ckpt_path}")
        model = Proteina.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            strict=False,
            autoencoder_ckpt_path=cfg.autoencoder_ckpt_path,
        )
        checkpoint_provenance = {
            "base_checkpoint": os.path.abspath(ckpt_path),
            "base_checkpoint_mutated": False,
        }
        # Optional nn-only lineage initialization is used by the isolated
        # general-protein -> antibody branch.  It is deliberately applied
        # after loading the immutable base checkpoint and before constructing
        # GRAFT's frozen reference snapshot; existing arms omit this option.
        init_nn_state_path = cfg.posttraining.get("init_nn_state", None)
        if init_nn_state_path:
            checkpoint_provenance["init_nn_state"] = _load_nn_only_lineage(
                model, str(init_nn_state_path)
            )
        else:
            checkpoint_provenance["init_nn_state"] = None
        model.ab_design_mode = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Load inf_cfg for sampling_model_args
        inf_cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../configs/inference_ab_design.yaml",
        )
        if os.path.exists(inf_cfg_path):
            inf_cfg = OmegaConf.load(inf_cfg_path)
            model.inf_cfg = inf_cfg.generation

        # ── Dataset ───────────────────────────────────────────────────────────
        ds_cfg   = cfg.get("dataset", {})
        data_dir = ds_cfg.get(
            "data_dir", os.environ.get("DATASET", "./data/structure_dataset")
        )
        split    = ds_cfg.get("split", "valid")  # Phase 1: valid split

        # Keep the training mask policy explicit. Validation/test use the
        # dataset's full-mask path; RL training follows the frozen mixed policy.
        mask_strategy = ds_cfg.get("mask_strategy", "mixed")
        mask_strategy_probs = ds_cfg.get("mask_strategy_probs", [0.5, 0.3, 0.2])
        dataset = AntibodyDesignDataset(
            data_dir=data_dir,
            split=split,
            mask_strategy=mask_strategy if split == "train" else "full",
            mask_strategy_probs=list(mask_strategy_probs),
            max_antibody_len=int(ds_cfg.get("max_antibody_len", 450)),
            max_antigen_len=int(ds_cfg.get("max_antigen_len", 500)),
            conventional_only=bool(ds_cfg.get("conventional_only", False)) if split == "train" else False,
            vhh_only=bool(ds_cfg.get("vhh_only", False)) if split == "train" else False,
        )
        loader  = DataLoader(
            dataset,
            batch_size = ds_cfg.get("batch_size", 4),
            shuffle    = True,
            num_workers= ds_cfg.get("num_workers", 4),
            collate_fn = collate_fn,
            pin_memory = True,
        )

        # ── Reward function ───────────────────────────────────────────────────
        reward_tier_raw = cfg.posttraining.get("reward_tier", 1)
        reward_tier = reward_tier_raw if isinstance(reward_tier_raw, str) else int(reward_tier_raw)
        clash_weight = float(cfg.posttraining.get("clash_weight", 0.1))
        affinity_weight = float(cfg.posttraining.get("affinity_weight", 1.0))
        affinity_kwargs = cfg.posttraining.get("affinity_kwargs", None)
        sequence_teacher = None
        sequence_teacher_path = cfg.posttraining.get("affinity_teacher_path", None)
        sequence_teacher_weight = float(cfg.posttraining.get("affinity_teacher_weight", 0.0))
        sequence_teacher_fusion = str(
            cfg.posttraining.get("affinity_teacher_fusion", "linear")
        )
        if sequence_teacher_weight > 0.0:
            if not sequence_teacher_path:
                raise ValueError(
                    "affinity_teacher_weight > 0 requires affinity_teacher_path"
                )
            from proteinfoundation.posttraining.affinity_teacher import AffinitySequenceTeacher

            sequence_teacher = AffinitySequenceTeacher.load(str(sequence_teacher_path))
            logger.info(
                f"[GRPO] Loaded frozen sequence-affinity teacher from {sequence_teacher_path} "
                f"(weight={sequence_teacher_weight:.3f})"
            )
        fold_surrogate_path = cfg.posttraining.get("fold_surrogate_path", None)
        rfree_weights = cfg.posttraining.get("rfree_weights", None)
        if str(reward_tier).lower() in {"r_free", "rfree", "free"} and not fold_surrogate_path:
            raise ValueError("R_free reward requires posttraining.fold_surrogate_path")
        if fold_surrogate_path:
            logger.info(f"[GRPO] Loading frozen fold-confidence surrogate from {fold_surrogate_path}")
        reward_fn   = get_reward_fn(
            reward_tier,
            autoencoder  = model.autoencoder,
            clash_weight = clash_weight,
            affinity_weight = affinity_weight,
            affinity_kwargs = affinity_kwargs,
            sequence_teacher = sequence_teacher,
            sequence_teacher_weight = sequence_teacher_weight,
            sequence_teacher_fusion = sequence_teacher_fusion,
            fold_surrogate_path = str(fold_surrogate_path) if fold_surrogate_path else None,
            rfree_weights = rfree_weights,
        )

        # ── Diversity monitor ─────────────────────────────────────────────────
        monitor = DiversityMonitor(
            entropy_early_stop_bits = float(cfg.posttraining.get("entropy_early_stop_bits", 1.0)),
            psi_warn_threshold      = float(cfg.posttraining.get("psi_warn_threshold", 0.9)),
            log_every               = int(cfg.posttraining.get("log_interval", 10)),
        )

        # ── Trainer ───────────────────────────────────────────────────────────
        # Keep the legacy GRPO entry point available for controlled ablations;
        # the prospective config selects the budget-aligned GRAFT estimator.
        algorithm = str(cfg.posttraining.get("algorithm", "grpo")).lower()
        trainer_cls = GRAFTTrainer if algorithm == "graft" else GRPOTrainer
        trainer = trainer_cls(
            model             = model,
            rl_cfg            = cfg.posttraining,
            reward_fn         = reward_fn,
            diversity_monitor = monitor,
            checkpoint_provenance = checkpoint_provenance,
        )

        trainer.train(loader)

    _run()


if __name__ == "__main__":
    main()
