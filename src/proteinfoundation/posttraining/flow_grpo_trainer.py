"""
Flow-GRPO post-training for the antibody latent flow matching model.

FAITHFUL matched-compute baseline for the FM-GRPO comparison (Reviewer h1aJ).

Flow-GRPO (Liu et al., NeurIPS 2025) aligns flow models via online GRPO on the
*stochastic* SDE trajectory, using per-step Gaussian transition log-probabilities
and a PPO-style clipped importance-sampling surrogate:

    1. Roll out K SDE trajectories from the current policy (same base model, same
       reward, same K, same nsteps as FM-GRPO). Record, for every stochastic step,
       the Gaussian transition (mu_t, sigma_t), the sampled x_{t+1}, and the
       per-step log-prob under the behaviour policy (old_logprob).
    2. Score the final sample with the SAME reward_fn as FM-GRPO; compute the
       std-normalized group advantage A_k = (r_k - mean_k) / (std_k + eps).
    3. Re-run the recorded trajectory WITH gradients (a subset of timesteps for
       memory / matched wall-clock), recompute new_logprob_t, form the ratio
       exp(new - old), and take the PPO-clipped policy-gradient step
           L = - mean_{k,t} min(ratio * A, clip(ratio, 1±eps) * A).

This differs from FM-GRPO deliberately and faithfully to the published method:
Flow-GRPO needs the SDE transition log-probabilities (per-step likelihood
bookkeeping) and a gradient pass over trajectory steps; FM-GRPO instead uses a
single re-noised flow-matching-loss surrogate with no trajectory likelihood.

The rollout reproduces the exact `sc`-mode step math in
`rdn_flow_matcher.simulation_step` (drift v via get_clean_pred_n_guided_vector +
vf_to_score/score_to_vf), so the behaviour policy is identical to the shipped
sampler; we only additionally record the Gaussian transition statistics.

We reuse GRPOTrainer's scaffolding (antigen locking, pair-freeze, checkpoint
format, dataset/model loading) verbatim so the two methods share everything
except the update rule.

Usage:
    python -m proteinfoundation.posttraining.flow_grpo_trainer \
        --config_name rb_flow_grpo
"""

from __future__ import annotations

import os
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Adam

from proteinfoundation.posttraining.reward_fns import get_reward_fn
from proteinfoundation.posttraining.diversity_monitor import (
    DiversityMonitor,
    DiversityCollapseError,
)
from proteinfoundation.posttraining.grpo_trainer import (
    _freeze_pair_update,
    _restore_antigen,
    _batch_to_device,
    _default_sampling_model_args,
)
from proteinfoundation.flow_matching.rdn_flow_matcher import vf_to_score, score_to_vf
from proteinfoundation.flow_matching.product_space_flow_matcher import (
    get_schedule,
    get_gt,
)


# ─────────────────────────────────────────────────────────────────────────────
# SDE transition helpers (mirror rdn_flow_matcher.simulation_step `sc` branch)
# ─────────────────────────────────────────────────────────────────────────────

def _sde_transition(
    x_t: Tensor,
    v: Tensor,
    t_scalar: float,
    dt: float,
    gt: float,
    step_params: Dict,
):
    """
    Reproduce the `sc` (SDE with noise scaling) branch of
    rdn_flow_matcher.simulation_step and return the Gaussian transition
    (drift-mean, std) plus a flag for whether this step is stochastic.

    `sc` branch:
        if t > t_lim_ode:  low-temp ODE (deterministic, no noise)
            delta_x = score_to_vf(x_t, vf_to_score(x_t,v,t)*1.5, t) * dt
        else:              SDE
            score   = vf_to_score(x_t, v, t)
            delta_x = (v + gt*score)*dt + sqrt(2*gt*sc_scale_noise*dt) * eps

    Returns (mean, std_scalar, stochastic: bool). `mean = x_t + drift*dt`.
    std is a scalar (isotropic Gaussian); std=0 for deterministic steps.
    """
    sc_scale_noise = step_params["sc_scale_noise"]
    sc_scale_score = step_params["sc_scale_score"]
    t_lim_ode = step_params["t_lim_ode"]
    sc_scale_score_def = 1.5

    t_vec = torch.full(
        (x_t.shape[0],), float(t_scalar), device=x_t.device, dtype=x_t.dtype
    )

    if t_scalar > t_lim_ode:
        # Deterministic low-temp ODE step near t->1 (no policy randomness).
        score = vf_to_score(x_t, v, t_vec)
        scaled_score = score * sc_scale_score_def
        v_scaled = score_to_vf(x_t, scaled_score, t_vec)
        mean = x_t + v_scaled * dt
        return mean, 0.0, False

    # Stochastic SDE step.
    score = vf_to_score(x_t, v, t_vec)
    drift = (v + gt * score) * dt
    mean = x_t + drift
    std = float((2.0 * gt * sc_scale_noise * dt) ** 0.5)
    return mean, std, True


def _gaussian_logprob_sum(
    x_next: Tensor, mean: Tensor, std: float, res_mask: Tensor
) -> Tensor:
    """
    Per-sample summed log N(x_next; mean, std^2 I) over masked residues/coords.

    Args:
        x_next, mean: [B, N, d]
        std: scalar Gaussian std (isotropic)
        res_mask: [B, N] bool — antibody residues that carry policy randomness
    Returns:
        [B] summed log-prob (0 where std==0 / deterministic step).
    """
    if std <= 0.0:
        return torch.zeros(x_next.shape[0], device=x_next.device, dtype=x_next.dtype)
    var = std * std
    # elementwise gaussian log density
    lp = -0.5 * ((x_next - mean) ** 2) / var - 0.5 * torch.log(
        torch.tensor(2.0 * torch.pi * var, device=x_next.device, dtype=x_next.dtype)
    )
    lp = lp.sum(dim=-1)  # [B, N]  sum over coord dims
    lp = lp * res_mask.to(lp.dtype)  # zero antigen / padding
    return lp.sum(dim=-1)  # [B]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class FlowGRPOTrainer:
    """
    Faithful Flow-GRPO trainer. Shares scaffolding with GRPOTrainer; replaces the
    update rule with SDE-trajectory + clipped importance-sampling policy gradient.
    """

    def __init__(
        self,
        model,
        rl_cfg: DictConfig,
        reward_fn: Optional[Callable] = None,
        diversity_monitor: Optional[DiversityMonitor] = None,
    ):
        self.model = model
        self.cfg = rl_cfg

        self.K = int(rl_cfg.get("K", 8))
        self.nsteps_sample = int(rl_cfg.get("nsteps_sample", 50))
        self.lr = float(rl_cfg.get("lr", 5e-6))
        self.max_iters = int(rl_cfg.get("max_iters", 150))
        self.freeze_pair_update = bool(rl_cfg.get("freeze_pair_update", True))
        self.log_interval = int(rl_cfg.get("log_interval", 10))
        self.ckpt_interval = int(rl_cfg.get("ckpt_interval", 50))
        self.ckpt_dir = str(rl_cfg.get("ckpt_dir", "./store/flow_grpo"))

        # Flow-GRPO specific
        self.clip_eps = float(rl_cfg.get("clip_eps", 0.2))
        self.adv_eps = float(rl_cfg.get("adv_eps", 1e-4))
        self.timestep_subsample = int(rl_cfg.get("timestep_subsample", 10))
        self.grad_accum_microbatch = int(rl_cfg.get("grad_accum_microbatch", 1))

        # Reward (identical to FM-GRPO)
        if reward_fn is None:
            tier_raw = rl_cfg.get("reward_tier", 2)
            tier = tier_raw if isinstance(tier_raw, str) else int(tier_raw)
            clash_weight = float(rl_cfg.get("clash_weight", 0.1))
            self.reward_fn = get_reward_fn(tier, clash_weight=clash_weight)
            logger.info(f"[FlowGRPO] reward Tier {tier!r} (clash_weight={clash_weight})")
        else:
            self.reward_fn = reward_fn

        if diversity_monitor is None:
            self.monitor = DiversityMonitor(
                entropy_early_stop_bits=float(rl_cfg.get("entropy_early_stop_bits", 1.0)),
                psi_warn_threshold=float(rl_cfg.get("psi_warn_threshold", 0.9)),
                log_every=self.log_interval,
            )
        else:
            self.monitor = diversity_monitor

        if self.freeze_pair_update:
            _freeze_pair_update(model)

        trainable = [p for p in model.nn.parameters() if p.requires_grad]
        logger.info(
            f"[FlowGRPO] Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M"
        )
        self.optimizer = Adam(trainable, lr=self.lr)

        if hasattr(model, "inf_cfg") and model.inf_cfg is not None:
            self.sampling_model_args = dict(model.inf_cfg.model)
        else:
            self.sampling_model_args = _default_sampling_model_args()

        self.data_modes = list(model.fm.data_modes)
        self._step = 0
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _predict_fn(self):
        return partial(self.model.predict_for_sampling, mode="full", n_recycle=0)

    def _build_masks(self, batch: Dict):
        """Antibody residue mask [B,N] (policy randomness) and full mask."""
        ct = batch.get("chain_type")
        if ct.dim() == 1:
            ct = ct.unsqueeze(0)
        full_mask = ct > 0
        ab_mask = (ct > 0) & (ct < 3)  # antibody residues only (exclude antigen==3)
        return ab_mask, full_mask

    def _schedules(self, nsteps: int, device):
        ts = {
            dm: get_schedule(
                mode=self.sampling_model_args[dm]["schedule"]["mode"],
                nsteps=int(nsteps),
                p1=self.sampling_model_args[dm]["schedule"]["p"],
            )
            for dm in self.data_modes
        }
        gt = {
            dm: get_gt(
                t=ts[dm][:-1],
                mode=self.sampling_model_args[dm]["gt"]["mode"],
                param=self.sampling_model_args[dm]["gt"]["p"],
                clamp_val=self.sampling_model_args[dm]["gt"]["clamp_val"],
            )
            for dm in self.data_modes
        }
        return ts, gt

    def _one_step_drift(self, batch, x, t_scalars, step_params, record: bool):
        """
        Compute the nn drift for one integration step and return, per data mode,
        (mean, std, stochastic, v). `record=False` runs under no_grad; `record=True`
        keeps the graph for the gradient pass. x is a dict[dm]->tensor.
        """
        device = self.model.device
        B = next(iter(x.values())).shape[0]
        t = {
            dm: t_scalars[dm] * torch.ones(B, device=device) for dm in self.data_modes
        }
        batch["x_t"] = x
        batch["t"] = t
        batch["mask"] = batch["_ab_full_mask"]
        nn_out = self.model.fm.get_clean_pred_n_guided_vector(
            batch=batch,
            predict_for_sampling=self._predict_fn(),
            guidance_w=1.0,
            ag_ratio=0.0,
        )
        out = {}
        for dm in self.data_modes:
            v = nn_out[dm]["v"]
            mean, std, stoch = _sde_transition(
                x[dm], v, float(t_scalars[dm]), self._dt[dm], self._gt_step[dm],
                step_params[dm],
            )
            out[dm] = (mean, std, stoch, v)
        return out

    # ── rollout with trajectory recording ────────────────────────────────────

    @torch.no_grad()
    def _rollout(self, batch: Dict) -> List[Dict]:
        """
        Run K SDE rollouts; record per-step Gaussian transitions and old log-probs.
        Returns a list of K trajectory dicts:
            { "x_1": {dm: [B,N,d]},            final clean sample
              "steps": [ {t, x_t{dm}, x_next{dm}, mean{dm}, std{dm}, stoch{dm}} ] }
        """
        device = self.model.device
        ab_mask, full_mask = batch["_ab_mask"], batch["_ab_full_mask"]
        B = full_mask.shape[0]
        N = batch["coords_nm"].shape[1] if batch["coords_nm"].dim() == 4 else batch["coords_nm"].shape[0]

        ts, gt = self._schedules(self.nsteps_sample, device)
        step_params = {
            dm: self.sampling_model_args[dm]["simulation_step_params"]
            for dm in self.data_modes
        }

        trajectories = []
        for _ in range(self.K):
            x = self.model.fm.sample_noise(
                N, shape=(B,), device=device, mask=full_mask
            )
            _restore_antigen(x, batch)
            steps = []
            for si in range(self.nsteps_sample):
                t_scalars = {dm: float(ts[dm][si]) for dm in self.data_modes}
                self._dt = {dm: float(ts[dm][si + 1] - ts[dm][si]) for dm in self.data_modes}
                self._gt_step = {dm: float(gt[dm][si]) for dm in self.data_modes}
                drift = self._one_step_drift(batch, x, t_scalars, step_params, record=False)

                rec = {"t": t_scalars, "x_t": {}, "x_next": {}, "mean": {},
                       "std": {}, "stoch": {}}
                x_new = {}
                for dm in self.data_modes:
                    mean, std, stoch, _v = drift[dm]
                    if stoch:
                        eps = torch.randn_like(mean)
                        xn = mean + std * eps
                    else:
                        xn = mean
                    # Offload recorded trajectory to CPU: only the live gradient
                    # graph should occupy GPU memory during _policy_step.
                    rec["x_t"][dm] = x[dm].detach().to("cpu")
                    rec["mean"][dm] = mean.detach().to("cpu")
                    rec["std"][dm] = std
                    rec["stoch"][dm] = stoch
                    x_new[dm] = xn
                _restore_antigen(x_new, batch)
                for dm in self.data_modes:
                    rec["x_next"][dm] = x_new[dm].detach().to("cpu")
                steps.append(rec)
                x = x_new
            trajectories.append({"x_1": {dm: x[dm].detach().clone() for dm in self.data_modes},
                                 "steps": steps})
        return trajectories

    def _compute_advantages(self, trajectories: List[Dict], batch: Dict):
        rewards = []
        for traj in trajectories:
            r = self.reward_fn(traj["x_1"], batch)  # [B]
            rewards.append(r)
        rewards = torch.stack(rewards, dim=0)  # [K, B]
        mean = rewards.mean(dim=0, keepdim=True)
        std = rewards.std(dim=0, keepdim=True)
        adv = (rewards - mean) / (std + self.adv_eps)  # [K,B] std-normalized (Flow-GRPO)
        return rewards, adv

    # ── clipped policy-gradient update over recorded trajectory ──────────────

    def _policy_step(self, trajectories, advantages, batch) -> Dict:
        device = self.model.device
        ab_mask = batch["_ab_mask"]
        step_params = {
            dm: self.sampling_model_args[dm]["simulation_step_params"]
            for dm in self.data_modes
        }
        ts, gt = self._schedules(self.nsteps_sample, device)

        # Which steps to differentiate (stochastic only), subsampled for memory.
        n = self.nsteps_sample
        stoch_idx = [i for i in range(n) if any(trajectories[0]["steps"][i]["stoch"][dm]
                                                for dm in self.data_modes)]
        if self.timestep_subsample > 0 and len(stoch_idx) > self.timestep_subsample:
            perm = torch.randperm(len(stoch_idx))[: self.timestep_subsample]
            sel = sorted(stoch_idx[p] for p in perm.tolist())
        else:
            sel = stoch_idx

        self.optimizer.zero_grad()
        total_loss = 0.0
        total_ratio = 0.0
        total_clipfrac = 0.0
        count = 0

        for k, traj in enumerate(trajectories):
            a_k = advantages[k]  # [B]
            for si in sel:
                rec = traj["steps"][si]
                self._dt = {dm: float(ts[dm][si + 1] - ts[dm][si]) for dm in self.data_modes}
                self._gt_step = {dm: float(gt[dm][si]) for dm in self.data_modes}
                t_scalars = rec["t"]
                x_in = {dm: rec["x_t"][dm].to(device) for dm in self.data_modes}

                drift = self._one_step_drift(batch, x_in, t_scalars, step_params, record=True)

                new_lp = torch.zeros(a_k.shape[0], device=device)
                old_lp = torch.zeros(a_k.shape[0], device=device)
                for dm in self.data_modes:
                    if not rec["stoch"][dm]:
                        continue
                    mean_new, std_dm, _stoch, _v = drift[dm]
                    x_next = rec["x_next"][dm].to(device)
                    new_lp = new_lp + _gaussian_logprob_sum(x_next, mean_new, std_dm, ab_mask)
                    old_mean = rec["mean"][dm].to(device)
                    with torch.no_grad():
                        old_lp = old_lp + _gaussian_logprob_sum(x_next, old_mean, std_dm, ab_mask)

                ratio = torch.exp(new_lp - old_lp)  # [B]
                unclipped = ratio * a_k
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a_k
                loss = -torch.min(unclipped, clipped).mean()
                (loss / (len(sel) * self.K)).backward()

                total_loss += loss.item()
                total_ratio += ratio.mean().item()
                total_clipfrac += ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                count += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.nn.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        return {
            "loss": total_loss / max(count, 1),
            "ratio": total_ratio / max(count, 1),
            "clipfrac": total_clipfrac / max(count, 1),
            "n_grad_steps": count,
        }

    # ── public step / train ──────────────────────────────────────────────────

    def step(self, batch: Dict) -> Dict:
        self._step += 1
        self.model.nn.train()
        self.model.autoencoder.eval()

        # Precompute masks once and stash on batch (used by rollout + update).
        ab_mask, full_mask = self._build_masks(batch)
        batch["_ab_mask"] = ab_mask
        batch["_ab_full_mask"] = full_mask

        trajectories = self._rollout(batch)
        rewards, advantages = self._compute_advantages(trajectories, batch)
        upd = self._policy_step(trajectories, advantages, batch)

        metrics = {
            "rl/loss": upd["loss"],
            "rl/reward_mean": rewards.mean().item(),
            "rl/reward_std": rewards.std().item(),
            "rl/ratio": upd["ratio"],
            "rl/clipfrac": upd["clipfrac"],
            "rl/step": self._step,
        }
        if self._step % self.log_interval == 0:
            logger.info(
                f"[FlowGRPO step={self._step}] loss={upd['loss']:.4f} "
                f"reward={rewards.mean().item():.4f} ± {rewards.std().item():.4f} "
                f"ratio={upd['ratio']:.3f} clipfrac={upd['clipfrac']:.3f} "
                f"grad_steps={upd['n_grad_steps']}"
            )
        if self._step % self.ckpt_interval == 0:
            self._save_checkpoint()
        return metrics

    def train(self, dataloader, max_iters: Optional[int] = None) -> None:
        max_iters = max_iters or self.max_iters
        logger.info(
            f"[FlowGRPO] start: K={self.K}, nsteps={self.nsteps_sample}, "
            f"lr={self.lr}, clip_eps={self.clip_eps}, subsample={self.timestep_subsample}, "
            f"max_iters={max_iters}"
        )
        device = self.model.device
        self.model.to(device)
        step = 0
        try:
            while step < max_iters:
                for batch in dataloader:
                    if step >= max_iters:
                        break
                    batch = _batch_to_device(batch, device)
                    self.step(batch)
                    step += 1
                    try:
                        self.monitor.check_stop()
                    except DiversityCollapseError as e:
                        logger.error(f"[FlowGRPO] {e}")
                        self._save_checkpoint(tag="collapse_stop")
                        return
        except KeyboardInterrupt:
            logger.info("[FlowGRPO] interrupted.")
            self._save_checkpoint(tag="interrupt")
            return
        logger.info(f"[FlowGRPO] done after {step} steps.")
        self._save_checkpoint(tag="final")

    def _save_checkpoint(self, tag: str = "") -> None:
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.ckpt_dir, f"flow_grpo_step{self._step:06d}{suffix}.pt")
        torch.save(
            {"step": self._step, "nn_state": self.model.nn.state_dict(),
             "optimizer": self.optimizer.state_dict()},
            path,
        )
        logger.info(f"[FlowGRPO] checkpoint -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import hydra
    from omegaconf import OmegaConf

    @hydra.main(config_path="../../configs", config_name="rb_flow_grpo", version_base=None)
    def _run(cfg):
        from proteinfoundation.proteina import Proteina
        from proteinfoundation.datasets.ab_data import AntibodyDesignDataset, collate_fn
        from torch.utils.data import DataLoader

        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

        if "seed" in cfg:
            torch.manual_seed(int(cfg.seed))

        ckpt_path = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
        model = Proteina.load_from_checkpoint(
            ckpt_path, map_location="cpu", strict=False,
            autoencoder_ckpt_path=cfg.autoencoder_ckpt_path,
        )
        model.ab_design_mode = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        inf_cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../configs/inference_ab_design.yaml",
        )
        if os.path.exists(inf_cfg_path):
            model.inf_cfg = OmegaConf.load(inf_cfg_path).generation

        ds_cfg = cfg.get("dataset", {})
        dataset = AntibodyDesignDataset(
            data_dir=ds_cfg.get(
                "data_dir", os.environ.get("DATASET", "./data/structure_dataset")
            ),
            split=ds_cfg.get("split", "valid"),
        )
        loader = DataLoader(
            dataset, batch_size=ds_cfg.get("batch_size", 4), shuffle=True,
            num_workers=ds_cfg.get("num_workers", 4), collate_fn=collate_fn,
            pin_memory=True,
        )

        tier_raw = cfg.posttraining.get("reward_tier", 2)
        tier = tier_raw if isinstance(tier_raw, str) else int(tier_raw)
        reward_fn = get_reward_fn(
            tier, autoencoder=model.autoencoder,
            clash_weight=float(cfg.posttraining.get("clash_weight", 0.1)),
        )
        monitor = DiversityMonitor(
            entropy_early_stop_bits=float(cfg.posttraining.get("entropy_early_stop_bits", 1.0)),
            psi_warn_threshold=float(cfg.posttraining.get("psi_warn_threshold", 0.9)),
            log_every=int(cfg.posttraining.get("log_interval", 10)),
        )
        trainer = FlowGRPOTrainer(
            model=model, rl_cfg=cfg.posttraining, reward_fn=reward_fn,
            diversity_monitor=monitor,
        )
        trainer.train(loader)

    _run()


if __name__ == "__main__":
    main()
