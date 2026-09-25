"""TempFlow-GRPO baseline for LaFMA (Reviewer h1aJ, round 2).

Reviewer h1aJ asked for empirical positioning against Flow-GRPO, DiffusionNFT, RAM and
TempFlow-GRPO. This implements TempFlow-GRPO ("When Timing Matters for GRPO in Flow Models").

What TempFlow-GRPO changes relative to Flow-GRPO
------------------------------------------------
Flow-GRPO assigns the SAME terminal-reward advantage to every timestep of a trajectory, which is
a uniform credit assignment: an early step that set up the global pose and a late step that only
polished side-chain latents receive identical learning signal. TempFlow-GRPO's argument is that
timing matters -- the reward is far more sensitive to some regions of the trajectory than others --
so credit should be assigned time-dependently.

Two mechanisms from the paper are implemented here, both switchable:

1. **Time-weighted credit assignment** (`temp_mode: "weight"`). Each step's contribution to the
   policy gradient is scaled by w(t), concentrating credit where the trajectory is actually
   decided. We use the paper's motivating shape -- higher weight at high noise / early
   generation, where the coarse pose is committed -- via
       w(t) = (1 - t)^alpha  normalized to mean 1 over the sampled steps,
   with `temp_alpha` controlling the sharpness. alpha = 0 recovers Flow-GRPO exactly, which makes
   this a strict generalization and gives a clean ablation axis.

2. **Single-step / branch-time credit** (`temp_mode: "branch"`). Rather than spreading the update
   over all stochastic steps, each rollout group is attributed to ONE sampled branching time,
   drawn per update from the time-weighted distribution above. This is the sparser, lower-variance
   form of the same idea.

Everything else -- SDE rollout, per-step Gaussian log-probability ratio, PPO clipping,
group-relative std-normalized advantages, reward, K, learning rate, freeze policy -- is inherited
unchanged from FlowGRPOTrainer, so the ONLY difference between the arms is the credit assignment.

Honest scope statement for the rebuttal: this is our good-faith reimplementation of the
TempFlow-GRPO *mechanism* on our architecture and reward, not the authors' released code, run at
our resource-adjusted budget.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List, Optional

import torch
from omegaconf import DictConfig

from proteinfoundation.posttraining.flow_grpo_trainer import (
    FlowGRPOTrainer,
    _gaussian_logprob_sum,
)

logger = logging.getLogger(__name__)


class TempFlowGRPOTrainer(FlowGRPOTrainer):
    """Flow-GRPO with time-dependent credit assignment."""

    def __init__(
        self,
        model,
        rl_cfg: DictConfig,
        reward_fn: Optional[Callable] = None,
        diversity_monitor=None,
    ):
        super().__init__(model, rl_cfg, reward_fn=reward_fn, diversity_monitor=diversity_monitor)
        self.temp_mode = str(rl_cfg.get("temp_mode", "weight"))
        self.temp_alpha = float(rl_cfg.get("temp_alpha", 1.0))
        self.ckpt_dir = str(rl_cfg.get("ckpt_dir", "./store/tempflow_grpo"))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        logger.info(
            "[TempFlowGRPO] temp_mode=%s temp_alpha=%.2f (alpha=0 reduces to Flow-GRPO)",
            self.temp_mode, self.temp_alpha,
        )

    def _time_weights(self, trajectories: List[Dict], sel: List[int], device) -> torch.Tensor:
        """w(t) over the selected steps, normalized to mean 1 so the effective LR is unchanged."""
        dm0 = self.data_modes[0]
        ts = torch.tensor(
            [float(trajectories[0]["steps"][i]["t"][dm0]) for i in sel],
            device=device, dtype=torch.float32,
        )
        w = (1.0 - ts).clamp(min=1e-6) ** self.temp_alpha
        w = w / w.mean().clamp(min=1e-8)
        return w

    def _policy_step(self, trajectories, advantages, batch) -> Dict:
        """Flow-GRPO's clipped update with time-dependent credit assignment."""
        device = self.model.device
        ab_mask = batch["_ab_mask"]
        step_params = {
            dm: self.sampling_model_args[dm]["simulation_step_params"]
            for dm in self.data_modes
        }
        ts, gt = self._schedules(self.nsteps_sample, device)

        n = self.nsteps_sample
        stoch_idx = [i for i in range(n)
                     if any(trajectories[0]["steps"][i]["stoch"][dm] for dm in self.data_modes)]
        if self.timestep_subsample > 0 and len(stoch_idx) > self.timestep_subsample:
            perm = torch.randperm(len(stoch_idx))[: self.timestep_subsample]
            sel = sorted(stoch_idx[p] for p in perm.tolist())
        else:
            sel = stoch_idx
        if not sel:
            return {"loss": 0.0, "ratio": 1.0, "clipfrac": 0.0, "n_grad_steps": 0}

        w = self._time_weights(trajectories, sel, device)

        if self.temp_mode == "branch":
            # attribute this update to ONE branching time drawn from the time-weighted law
            j = int(torch.multinomial(w / w.sum(), num_samples=1).item())
            sel = [sel[j]]
            w = torch.ones(1, device=device)

        self.optimizer.zero_grad()
        total_loss = total_ratio = total_clipfrac = 0.0
        count = 0

        for k, traj in enumerate(trajectories):
            a_k = advantages[k]
            for wi, si in enumerate(sel):
                rec = traj["steps"][si]
                self._dt = {dm: float(ts[dm][si + 1] - ts[dm][si]) for dm in self.data_modes}
                self._gt_step = {dm: float(gt[dm][si]) for dm in self.data_modes}
                x_in = {dm: rec["x_t"][dm].to(device) for dm in self.data_modes}

                drift = self._one_step_drift(batch, x_in, rec["t"], step_params, record=True)

                new_lp = torch.zeros(a_k.shape[0], device=device)
                old_lp = torch.zeros(a_k.shape[0], device=device)
                for dm in self.data_modes:
                    if not rec["stoch"][dm]:
                        continue
                    mean_new, std_dm, _s, _v = drift[dm]
                    x_next = rec["x_next"][dm].to(device)
                    new_lp = new_lp + _gaussian_logprob_sum(x_next, mean_new, std_dm, ab_mask)
                    old_mean = rec["mean"][dm].to(device)
                    with torch.no_grad():
                        old_lp = old_lp + _gaussian_logprob_sum(x_next, old_mean, std_dm, ab_mask)

                ratio = torch.exp(new_lp - old_lp)
                unclipped = ratio * a_k
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * a_k
                # ── the TempFlow-GRPO difference: time-dependent credit ──
                loss = -(w[wi] * torch.min(unclipped, clipped)).mean()
                (loss / (len(sel) * self.K)).backward()

                total_loss += loss.item()
                total_ratio += ratio.mean().item()
                total_clipfrac += ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
                count += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.nn.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        return {"loss": total_loss / max(count, 1), "ratio": total_ratio / max(count, 1),
                "clipfrac": total_clipfrac / max(count, 1), "n_grad_steps": count}

    def _save_checkpoint(self, tag: str = "") -> None:
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.ckpt_dir, f"tempflow_grpo_step{self._step:06d}{suffix}.pt")
        torch.save({"step": self._step, "nn_state": self.model.nn.state_dict(),
                    "optimizer": self.optimizer.state_dict()}, path)
        logger.info(f"[TempFlowGRPO] checkpoint -> {path}")


def main():
    import hydra
    from omegaconf import OmegaConf

    from proteinfoundation.posttraining.diversity_monitor import DiversityMonitor
    from proteinfoundation.posttraining.reward_fns import get_reward_fn

    @hydra.main(config_path="../../configs", config_name="rb_tempflow_grpo", version_base=None)
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
        TempFlowGRPOTrainer(model=model, rl_cfg=cfg.posttraining, reward_fn=reward_fn,
                            diversity_monitor=monitor).train(loader)

    _run()


if __name__ == "__main__":
    main()
