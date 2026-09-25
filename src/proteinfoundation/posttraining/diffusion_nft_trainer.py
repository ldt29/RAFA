"""DiffusionNFT-style reward alignment baseline for LaFMA (Reviewer h1aJ, round 2).

Reviewer h1aJ asked for empirical positioning against recent flow/diffusion RL methods, noting
that only Flow-GRPO was implemented in round 1 and that DiffusionNFT / RAM / TempFlow-GRPO
remained conceptual.

This adds DiffusionNFT (Zheng et al., "DiffusionNFT: Online Diffusion Reinforcement with Forward
Process"). We reuse the SAME rollout machinery and the SAME reward as FlowGRPOTrainer and
GRPOTrainer, so the ONLY thing that differs between the three arms is the update rule. That is
what makes the comparison interpretable.

The DiffusionNFT idea, and how it is realized here
--------------------------------------------------
DiffusionNFT does *not* use policy-gradient / likelihood ratios at all. It performs Negative-aware
Fine-Tuning on the FORWARD process: within a group of rollouts it splits samples into a positive
and a negative set by reward, then fits the flow-matching objective so that the model moves toward
the positive set and away from the negative set. Concretely, for a group of K completed samples
with rewards r_k, DiffusionNFT forms reward-derived weights and trains an implicit-policy
flow-matching regression on re-noised samples, i.e. it is supervised-style training on generated
data, not an importance-weighted trajectory update.

Implementation notes (kept faithful to the paper's mechanism, adapted to our model):
  * Group = the K rollouts for one batch (same K as the other two arms).
  * Positive/negative split by within-group reward rank, with a configurable quantile
    (`nft_pos_quantile`, default 0.5 -> top half positive, bottom half negative).
  * The update is a flow-matching regression at randomly sampled times on RE-NOISED completed
    samples, with weight +w_pos on the positive set and -w_neg on the negative set. This is the
    "negative-aware" part: negative samples enter with a negative coefficient, so their velocity
    targets are actively pushed away rather than merely down-weighted.
  * Negative-branch loss is clamped (`nft_neg_clamp`) so a single bad rollout cannot produce an
    unbounded ascent direction -- without this the objective is unbounded below, which is a
    practical divergence mode rather than a property of the method.
  * No SDE conversion, no per-step Gaussian log-probs, no PPO clip. Deterministic ODE rollouts
    are sufficient, which is one of DiffusionNFT's selling points over Flow-GRPO.

Honest scope statement for the rebuttal: this is our good-faith reimplementation of the DiffusionNFT
*mechanism* on our architecture and reward, not the authors' released code, and it is run at our
resource-adjusted budget. It is evidence about how the update rule behaves in our setting; it is
not a claim to have reproduced their published numbers.
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


class DiffusionNFTTrainer(FlowGRPOTrainer):
    """Negative-aware forward-process fine-tuning.

    Inherits rollout / masks / reward / checkpointing from FlowGRPOTrainer so that the arms are
    matched everywhere except the update rule.
    """

    def __init__(
        self,
        model,
        rl_cfg: DictConfig,
        reward_fn: Optional[Callable] = None,
        diversity_monitor=None,
    ):
        super().__init__(model, rl_cfg, reward_fn=reward_fn, diversity_monitor=diversity_monitor)
        self.pos_quantile = float(rl_cfg.get("nft_pos_quantile", 0.5))
        self.w_pos = float(rl_cfg.get("nft_w_pos", 1.0))
        self.w_neg = float(rl_cfg.get("nft_w_neg", 1.0))
        self.neg_clamp = float(rl_cfg.get("nft_neg_clamp", 1.0))
        self.n_time_samples = int(rl_cfg.get("nft_n_time_samples", 4))
        self.ckpt_dir = str(rl_cfg.get("ckpt_dir", "./store/diffusion_nft"))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        logger.info(
            "[DiffusionNFT] pos_quantile=%.2f w_pos=%.2f w_neg=%.2f neg_clamp=%.2f "
            "n_time_samples=%d",
            self.pos_quantile, self.w_pos, self.w_neg, self.neg_clamp, self.n_time_samples,
        )

    # ── forward-process negative-aware update ────────────────────────────────

    def _nft_update(self, trajectories: List[Dict], rewards: torch.Tensor, batch: Dict) -> Dict:
        """Flow-matching regression on re-noised completed samples, signed by reward rank.

        rewards: [K, B]
        """
        device = self.model.device
        ab_mask = batch["_ab_mask"]          # [B, N] antibody-only
        full_mask = batch["_ab_full_mask"]

        K = rewards.shape[0]
        # within-group (per batch element) reward rank -> positive / negative membership
        # rank 0 = worst. sign[k, b] = +1 for positive set, -1 for negative set.
        order = rewards.argsort(dim=0)                      # [K,B]
        rank = torch.zeros_like(rewards)
        ar = torch.arange(K, device=rewards.device).float().unsqueeze(1)
        rank.scatter_(0, order, ar.expand_as(rewards))
        thresh = (K - 1) * self.pos_quantile
        sign = torch.where(rank > thresh,
                           torch.ones_like(rank),
                           -torch.ones_like(rank))          # [K,B]
        # If a group is degenerate (all equal reward) there is no signal: skip it.
        degenerate = (rewards.max(dim=0).values - rewards.min(dim=0).values).abs() < 1e-8  # [B]

        self.optimizer.zero_grad()
        total_loss, total_pos, total_neg, count = 0.0, 0.0, 0.0, 0

        for k, traj in enumerate(trajectories):
            x1 = {dm: traj["x_1"][dm].to(device) for dm in self.data_modes}
            s_k = sign[k]                                    # [B]
            for _ in range(self.n_time_samples):
                # sample a time and re-noise the completed sample: the FORWARD process
                B = s_k.shape[0]
                t = torch.rand(B, device=device).clamp(0.02, 0.98)
                x0 = self.model.fm.sample_noise(
                    x1[self.data_modes[0]].shape[1], shape=(B,), device=device, mask=full_mask
                )
                xt, target = {}, {}
                for dm in self.data_modes:
                    tb = t.view(-1, *([1] * (x1[dm].dim() - 1)))
                    xt[dm] = (1.0 - tb) * x0[dm] + tb * x1[dm]
                    # linear-interpolant velocity target toward the completed sample
                    target[dm] = x1[dm] - x0[dm]
                _restore_antigen(xt, batch)

                nn_batch = dict(batch)
                nn_batch["x_t"] = xt
                nn_batch["t"] = {dm: t for dm in self.data_modes}
                nn_batch["mask"] = full_mask
                # Same prediction path the sampler and the other two RL arms use, so the
                # velocity convention is identical across arms.
                nn_out = self.model.fm.get_clean_pred_n_guided_vector(
                    batch=nn_batch,
                    predict_for_sampling=self._predict_fn(),
                    guidance_w=1.0,
                    ag_ratio=0.0,
                )

                # per-sample masked flow-matching MSE
                per_sample = torch.zeros(B, device=device)
                m = ab_mask.float()
                denom = m.sum(dim=1).clamp(min=1.0)
                for dm in self.data_modes:
                    v = nn_out[dm]["v"]
                    err = (v - target[dm]) ** 2
                    while err.dim() > 2:
                        err = err.mean(dim=-1)
                    per_sample = per_sample + (err * m).sum(dim=1) / denom

                # negative-aware signed objective:
                #   positive set -> minimize FM loss (pull toward high-reward samples)
                #   negative set -> maximize it, but clamped so it cannot run away
                pos_term = per_sample.clamp(min=0.0)
                neg_term = (-per_sample).clamp(min=-self.neg_clamp)
                signed = torch.where(s_k > 0, self.w_pos * pos_term, self.w_neg * neg_term)
                signed = torch.where(degenerate, torch.zeros_like(signed), signed)
                loss = signed.mean()
                (loss / (self.n_time_samples * K)).backward()

                total_loss += loss.item()
                total_pos += per_sample[s_k > 0].mean().item() if (s_k > 0).any() else 0.0
                total_neg += per_sample[s_k <= 0].mean().item() if (s_k <= 0).any() else 0.0
                count += 1

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.nn.parameters() if p.requires_grad], max_norm=1.0
        )
        self.optimizer.step()
        return {
            "loss": total_loss / max(count, 1),
            "fm_pos": total_pos / max(count, 1),
            "fm_neg": total_neg / max(count, 1),
            "n_grad_steps": count,
        }

    # ── public step ──────────────────────────────────────────────────────────

    def step(self, batch: Dict) -> Dict:
        self._step += 1
        self.model.nn.train()
        self.model.autoencoder.eval()

        ab_mask, full_mask = self._build_masks(batch)
        batch["_ab_mask"] = ab_mask
        batch["_ab_full_mask"] = full_mask

        trajectories = self._rollout(batch)
        rewards, _adv = self._compute_advantages(trajectories, batch)
        upd = self._nft_update(trajectories, rewards, batch)

        metrics = {
            "rl/loss": upd["loss"],
            "rl/reward_mean": rewards.mean().item(),
            "rl/reward_std": rewards.std().item(),
            "rl/fm_pos": upd["fm_pos"],
            "rl/fm_neg": upd["fm_neg"],
            "rl/step": self._step,
        }
        if self._step % self.log_interval == 0:
            logger.info(
                f"[DiffusionNFT step={self._step}] loss={upd['loss']:.4f} "
                f"reward={rewards.mean().item():.4f} ± {rewards.std().item():.4f} "
                f"fm_pos={upd['fm_pos']:.4f} fm_neg={upd['fm_neg']:.4f}"
            )
        if self._step % self.ckpt_interval == 0:
            self._save_checkpoint()
        return metrics


    def _save_checkpoint(self, tag: str = "") -> None:
        """Override so DiffusionNFT checkpoints are not named flow_grpo_*."""
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.ckpt_dir, f"diffusion_nft_step{self._step:06d}{suffix}.pt")
        torch.save(
            {"step": self._step, "nn_state": self.model.nn.state_dict(),
             "optimizer": self.optimizer.state_dict()},
            path,
        )
        logger.info(f"[DiffusionNFT] checkpoint -> {path}")


def main():
    """Entry point mirroring flow_grpo_trainer.main() exactly, so that the only difference
    between the two baselines is the trainer class (i.e. the update rule)."""
    import hydra
    from omegaconf import OmegaConf

    from proteinfoundation.posttraining.diversity_monitor import DiversityMonitor
    from proteinfoundation.posttraining.reward_fns import get_reward_fn

    @hydra.main(config_path="../../configs", config_name="rb_diffusion_nft", version_base=None)
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
        trainer = DiffusionNFTTrainer(
            model=model, rl_cfg=cfg.posttraining, reward_fn=reward_fn,
            diversity_monitor=monitor,
        )
        trainer.train(loader)

    _run()


if __name__ == "__main__":
    main()
