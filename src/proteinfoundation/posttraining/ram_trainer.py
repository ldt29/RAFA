"""Historical reward-gradient adjoint control (legacy identifier: RAMTrainer).

This implementation was originally labelled Reinforce Adjoint Matching (RAM), but a later
source-level audit found that attribution to be wrong. The RAM paper derives a REINFORCE-based
consistency objective and explicitly does not require reward gradients or backward adjoint sweeps.
The code below instead differentiates the reward at the terminal sample and transports that vector
approximately. It is therefore an adjoint-style experimental control, not a RAM reproduction.
The class/config identifiers are retained only so historical checkpoints remain loadable.

The reward every RL arm in this paper uses is tier 2:
    reward = (fnat_proxy + iRMS_score + LRMS_score)/3 - clash_weight * clash_penalty
built entirely from differentiable torch ops (cdist, direct RMSD against the GT-locked antigen
frame, soft distance thresholds). A gradient check confirms d(reward)/d(x_1) flows with nonzero
entries. Only tier 3 -- decode to PDB and shell out to the DockQ binary -- is a true black box, and
that is not the reward used for the reported runs. This fact makes the control executable; it does
not make it RAM.

What this control does differently
----------------------------------
Flow-GRPO and TempFlow-GRPO are policy-gradient methods: they estimate reward improvement from
score-function estimators over sampled trajectories (log-prob ratios, group advantages, PPO clip).
DiffusionNFT avoids gradients entirely and does negative-aware supervised regression. This control
keeps the dynamics deterministic, computes the exact terminal reward gradient, approximately
transports that vector to sampled times, and matches the model's velocity field to the resulting
reward-improving direction. It is the only arm here that uses first-order reward information.

Implementation (memory-tractable adjoint on our model):
  * Deterministic ODE rollout, gradients detached step to step, storing the trajectory.
  * At the terminal sample compute a = d(reward)/d(x_1) -- the adjoint seed. This is the exact
    reward gradient, not an estimate.
  * Approximate transport rather than a continuous adjoint. A full continuous adjoint would require
    vector--Jacobian products through every step, which does not fit our 171M model at K rollouts.
    This control instead transports the terminal gradient along the linear-interpolant flow to each
    sampled time and uses it as the matching target. `ram_adjoint_decay` controls transport damping.
  * Adjoint-matching loss at randomly sampled times:
        L = mean_t || v_theta(x_t, t) - (v_detached + eta * a_t) ||^2
    so the velocity field is pulled toward the reward-improving correction of its own output.
    `ram_eta` is the step size on the reward direction.
  * The adjoint is normalized per sample (`ram_grad_clip`) because DockQ-proxy gradients are
    badly scaled across targets of different size.

Matched to the other arms in base checkpoint, reward, K, rollout steps, learning rate, split and
freeze policy, so only the update rule differs.

Honest scope statement: this is not evidence about RAM. It is retained as a separately labelled
reward-gradient adjoint control run at the same resource-adjusted budget.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig

from proteinfoundation.posttraining.flow_grpo_trainer import (
    FlowGRPOTrainer,
    _restore_antigen,
)

logger = logging.getLogger(__name__)


class RAMTrainer(FlowGRPOTrainer):
    """Reward-gradient adjoint control; name retained for checkpoint compatibility."""

    def __init__(
        self,
        model,
        rl_cfg: DictConfig,
        reward_fn: Optional[Callable] = None,
        diversity_monitor=None,
    ):
        super().__init__(model, rl_cfg, reward_fn=reward_fn, diversity_monitor=diversity_monitor)
        self.ram_eta = float(rl_cfg.get("ram_eta", 1.0))
        self.adjoint_decay = float(rl_cfg.get("ram_adjoint_decay", 1.0))
        self.grad_clip = float(rl_cfg.get("ram_grad_clip", 1.0))
        self.n_time_samples = int(rl_cfg.get("ram_n_time_samples", 4))
        self.normalize_adjoint = bool(rl_cfg.get("ram_normalize_adjoint", True))
        self.ckpt_dir = str(rl_cfg.get("ckpt_dir", "./store/ram"))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        logger.info(
            "[RAM] eta=%.3f adjoint_decay=%.3f grad_clip=%.3f n_time_samples=%d",
            self.ram_eta, self.adjoint_decay, self.grad_clip, self.n_time_samples,
        )

    # ── adjoint seed: exact d(reward)/d(x_1) ─────────────────────────────────

    def _adjoint_seed(self, x1: Dict, batch: Dict):
        """Return {dm: d(reward)/d(x1[dm])}, per-sample normalized, plus the reward."""
        leaves = {dm: x1[dm].detach().clone().requires_grad_(True) for dm in self.data_modes}
        with torch.enable_grad():
            r = self.reward_fn(leaves, batch)          # [B]
            grads = torch.autograd.grad(
                r.sum(), [leaves[dm] for dm in self.data_modes],
                allow_unused=True, retain_graph=False,
            )
        adj = {}
        raw_norm = 0.0
        for dm, g in zip(self.data_modes, grads):
            if g is None:
                adj[dm] = torch.zeros_like(leaves[dm])
                continue
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            # Per-sample RESCALING to a target norm, not a cap.
            #
            # This matters and is disclosed: the tier-2 DockQ-proxy gradient has a raw per-sample
            # norm around 1e-3 on this model. A cap of the form min(clip/||g||, 1) never binds at
            # that scale, so the adjoint -- and therefore the entire control update -- stays
            # numerically negligible and the method would appear to do nothing for a reason that
            # is purely a scaling artifact of our reward, not a property of the control. We therefore
            # normalize the adjoint to a fixed target norm (`ram_grad_clip`), which preserves the
            # reward-gradient DIRECTION used by this control while making the step size
            # comparable to the other arms' effective updates.
            flat = g.reshape(g.shape[0], -1)
            nrm = flat.norm(dim=1).clamp(min=1e-12)
            raw_norm += float(nrm.mean())
            if self.normalize_adjoint:
                scale = self.grad_clip / nrm
            else:
                scale = (self.grad_clip / nrm).clamp(max=1.0)
            adj[dm] = g * scale.view(-1, *([1] * (g.dim() - 1)))
        return adj, r.detach(), raw_norm

    # ── adjoint-matching update ──────────────────────────────────────────────

    def _ram_update(self, trajectories: List[Dict], batch: Dict) -> Dict:
        device = self.model.device
        ab_mask = batch["_ab_mask"]
        full_mask = batch["_ab_full_mask"]
        m = ab_mask.float()
        denom = m.sum(dim=1).clamp(min=1.0)

        self.optimizer.zero_grad()
        total_loss, total_r, total_adj, total_raw, count = 0.0, 0.0, 0.0, 0.0, 0

        for traj in trajectories:
            x1 = {dm: traj["x_1"][dm].to(device) for dm in self.data_modes}
            adj, r, raw_norm = self._adjoint_seed(x1, batch)
            total_r += float(r.mean())
            total_adj += float(
                sum(a.reshape(a.shape[0], -1).norm(dim=1).mean() for a in adj.values())
            )
            total_raw += float(raw_norm)

            B = x1[self.data_modes[0]].shape[0]
            for _ in range(self.n_time_samples):
                t = torch.rand(B, device=device).clamp(0.02, 0.98)
                x0 = self.model.fm.sample_noise(
                    x1[self.data_modes[0]].shape[1], shape=(B,), device=device, mask=full_mask
                )
                xt = {}
                for dm in self.data_modes:
                    tb = t.view(-1, *([1] * (x1[dm].dim() - 1)))
                    xt[dm] = (1.0 - tb) * x0[dm] + tb * x1[dm]
                _restore_antigen(xt, batch)

                nn_batch = dict(batch)
                nn_batch["x_t"] = xt
                nn_batch["t"] = {dm: t for dm in self.data_modes}
                nn_batch["mask"] = full_mask
                nn_out = self.model.fm.get_clean_pred_n_guided_vector(
                    batch=nn_batch, predict_for_sampling=self._predict_fn(),
                    guidance_w=1.0, ag_ratio=0.0,
                )

                loss = torch.zeros((), device=device)
                for dm in self.data_modes:
                    v = nn_out[dm]["v"]
                    # transport the terminal adjoint to time t (damped), then form the
                    # reward-improving target for the velocity field
                    tb = t.view(-1, *([1] * (adj[dm].dim() - 1)))
                    a_t = adj[dm] * (self.adjoint_decay * tb + (1.0 - self.adjoint_decay))
                    target = (v.detach() + self.ram_eta * a_t)
                    err = (v - target) ** 2
                    while err.dim() > 2:
                        err = err.mean(dim=-1)
                    loss = loss + ((err * m).sum(dim=1) / denom).mean()

                (loss / (self.n_time_samples * self.K)).backward()
                total_loss += float(loss)
                count += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.nn.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        n_traj = max(len(trajectories), 1)
        return {"loss": total_loss / max(count, 1),
                "reward": total_r / n_traj,
                "adj_norm": total_adj / n_traj,
                "raw_adj_norm": total_raw / n_traj,
                "n_grad_steps": count}

    def step(self, batch: Dict) -> Dict:
        self._step += 1
        self.model.nn.train()
        self.model.autoencoder.eval()

        ab_mask, full_mask = self._build_masks(batch)
        batch["_ab_mask"] = ab_mask
        batch["_ab_full_mask"] = full_mask

        trajectories = self._rollout(batch)
        upd = self._ram_update(trajectories, batch)

        metrics = {"rl/loss": upd["loss"], "rl/reward_mean": upd["reward"],
                   "rl/adj_norm": upd["adj_norm"],
                   "rl/raw_adj_norm": upd["raw_adj_norm"], "rl/step": self._step}
        if self._step % self.log_interval == 0:
            logger.info(
                f"[RAM step={self._step}] loss={upd['loss']:.6f} "
                f"reward={upd['reward']:.4f} adj_norm={upd['adj_norm']:.4f} "
                f"raw_adj_norm={upd['raw_adj_norm']:.3e}"
            )
        if self._step % self.ckpt_interval == 0:
            self._save_checkpoint()
        return metrics

    def _save_checkpoint(self, tag: str = "") -> None:
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.ckpt_dir, f"ram_step{self._step:06d}{suffix}.pt")
        torch.save({"step": self._step, "nn_state": self.model.nn.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)
        logger.info(f"[RAM] checkpoint -> {path}")


def main():
    import hydra
    from omegaconf import OmegaConf

    from proteinfoundation.posttraining.diversity_monitor import DiversityMonitor
    from proteinfoundation.posttraining.reward_fns import get_reward_fn

    @hydra.main(config_path="../../configs", config_name="rb_ram", version_base=None)
    def _run(cfg):
        from proteinfoundation.proteina import Proteina
        from proteinfoundation.datasets.ab_data import AntibodyDesignDataset, collate_fn
        from torch.utils.data import DataLoader

        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
        if "seed" in cfg:
            torch.manual_seed(int(cfg.seed))

        model = Proteina.load_from_checkpoint(
            os.path.join(cfg.ckpt_path, cfg.ckpt_name), map_location="cpu", strict=False,
            autoencoder_ckpt_path=cfg.autoencoder_ckpt_path,
        )
        model.ab_design_mode = True
        model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        inf_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../../configs/inference_ab_design.yaml")
        if os.path.exists(inf_cfg_path):
            model.inf_cfg = OmegaConf.load(inf_cfg_path).generation

        ds_cfg = cfg.get("dataset", {})
        dataset = AntibodyDesignDataset(
            data_dir=ds_cfg.get(
                "data_dir", os.environ.get("DATASET", "./data/structure_dataset")
            ),
            split=ds_cfg.get("split", "valid"),
        )
        loader = DataLoader(dataset, batch_size=ds_cfg.get("batch_size", 1), shuffle=True,
                            num_workers=ds_cfg.get("num_workers", 4), collate_fn=collate_fn,
                            pin_memory=True)

        tier_raw = cfg.posttraining.get("reward_tier", 2)
        tier = tier_raw if isinstance(tier_raw, str) else int(tier_raw)
        reward_fn = get_reward_fn(tier, autoencoder=model.autoencoder,
                                  clash_weight=float(cfg.posttraining.get("clash_weight", 0.1)))
        monitor = DiversityMonitor(
            entropy_early_stop_bits=float(cfg.posttraining.get("entropy_early_stop_bits", 1.0)),
            psi_warn_threshold=float(cfg.posttraining.get("psi_warn_threshold", 0.9)),
            log_every=int(cfg.posttraining.get("log_interval", 10)),
        )
        RAMTrainer(model=model, rl_cfg=cfg.posttraining, reward_fn=reward_fn,
                   diversity_monitor=monitor).train(loader)

    _run()


if __name__ == "__main__":
    main()
