"""RAFA Reward-Gradient Transport (RGT) post-training.

RGT is the RAFA-specific reward-gradient transport control derived from the
historical RAM-labelled experiment.  It deliberately keeps the old
``RAMTrainer`` available as a frozen compatibility baseline, while fixing two
issues in that control:

* the update uses the states from the same rollout that produced the terminal
  reward instead of drawing a second, unrelated ``x_0``/``t`` pair;
* the reward gradient is normalized jointly across the product-space modes,
  preserving the relative ``bb_ca``/``local_latents`` direction.

This is not a reimplementation of the Reinforce Adjoint Matching paper.  The
method uses a differentiable terminal reward gradient and an approximate time
transport, hence the separate RGT name and provenance boundary.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig

from proteinfoundation.posttraining.flow_grpo_trainer import _restore_antigen
from proteinfoundation.posttraining.ram_trainer import RAMTrainer


logger = logging.getLogger(__name__)


class RGTTrainer(RAMTrainer):
    """Reward-Gradient Transport trainer used by the RAFA-RGT method."""

    def __init__(
        self,
        model,
        rl_cfg: DictConfig,
        reward_fn=None,
        diversity_monitor=None,
    ):
        # Reuse the matched rollout, optimizer, checkpoint and mask scaffolding
        # from the historical control.  RGT overrides the update path below.
        super().__init__(
            model=model,
            rl_cfg=rl_cfg,
            reward_fn=reward_fn,
            diversity_monitor=diversity_monitor,
        )
        self.rgt_eta = float(rl_cfg.get("rgt_eta", 1.0))
        if self.rgt_eta <= 0.0:
            raise ValueError("rgt_eta must be positive")
        self.rgt_adjoint_decay = float(rl_cfg.get("rgt_adjoint_decay", 1.0))
        self.rgt_time_power = max(float(rl_cfg.get("rgt_time_power", 1.0)), 0.0)
        self.rgt_transport_mode = str(
            rl_cfg.get("rgt_transport_mode", "time_weighted")
        )
        if self.rgt_transport_mode not in {
            "time_weighted",
            "constant_mean",
            "constant_unit",
        }:
            raise ValueError(
                "rgt_transport_mode must be one of time_weighted, "
                "constant_mean, constant_unit"
            )
        self.rgt_grad_norm = float(rl_cfg.get("rgt_grad_norm", 0.5))
        self.rgt_model_grad_clip = float(rl_cfg.get("rgt_model_grad_clip", 1.0))
        self.rgt_n_time_samples = max(int(rl_cfg.get("rgt_n_time_samples", 4)), 1)
        self.rgt_normalize_gradient = bool(
            rl_cfg.get("rgt_normalize_gradient", True)
        )
        # Reviewer/goal negative control: keep the rollout states and reward
        # values fixed, but assign each terminal reward gradient to a
        # different rollout before the transport update.  This is only used
        # by an explicitly labelled control run; formal arms keep it false.
        self.shuffle_reward_gradient = bool(
            rl_cfg.get("shuffle_reward_gradient", False)
        )
        # Mechanism control: replace the true terminal reward direction by a
        # same-shape, same-norm random direction.  This is deliberately kept
        # separate from the shuffled-rollout control so the two tests answer
        # different questions.
        self.randomize_reward_gradient = bool(
            rl_cfg.get("randomize_reward_gradient", False)
        )
        logger.info(
            "[RGT] eta=%.3f adjoint_decay=%.3f time_power=%.3f "
            "transport_mode=%s target_grad_norm=%.3f n_time_samples=%d "
            "shuffle=%s random=%s",
            self.rgt_eta,
            self.rgt_adjoint_decay,
            self.rgt_time_power,
            self.rgt_transport_mode,
            self.rgt_grad_norm,
            self.rgt_n_time_samples,
            self.shuffle_reward_gradient,
            self.randomize_reward_gradient,
        )

    def _random_adjoint(self, adjoint: Dict, batch: Dict) -> Dict:
        """Build a random antibody-only direction with the RGT target norm."""

        ab_mask = batch["_ab_mask"].to(dtype=next(iter(adjoint.values())).dtype)
        random_values = {}
        joint_sq = None
        for dm, value in adjoint.items():
            noise = torch.randn_like(value)
            mask = ab_mask
            while mask.dim() < noise.dim():
                mask = mask.unsqueeze(-1)
            noise = noise * mask
            random_values[dm] = noise
            term = noise.reshape(noise.shape[0], -1).square().sum(dim=1)
            joint_sq = term if joint_sq is None else joint_sq + term
        joint_norm = joint_sq.sqrt().clamp(min=1e-12)
        scale = self.rgt_grad_norm / joint_norm
        return {
            dm: value * scale.view(-1, *([1] * (value.dim() - 1)))
            for dm, value in random_values.items()
        }

    def _adjoint_seed(self, x1: Dict, batch: Dict):
        """Compute a jointly normalized terminal reward gradient.

        The historical control normalized each data mode independently.  That
        can change the direction in the product space and gives the two modes
        equal norm even when one carries substantially less reward signal.
        RGT masks the antibody residues and uses one joint per-sample norm.
        """

        leaves = {
            dm: x1[dm].detach().clone().requires_grad_(True)
            for dm in self.data_modes
        }
        with torch.enable_grad():
            reward = self.reward_fn(leaves, batch)
            grads = torch.autograd.grad(
                reward.sum(),
                [leaves[dm] for dm in self.data_modes],
                allow_unused=True,
                retain_graph=False,
            )

        ab_mask = batch["_ab_mask"].to(dtype=next(iter(leaves.values())).dtype)
        joint_sq = torch.zeros(
            reward.shape[0], device=reward.device, dtype=reward.dtype
        )
        clean_grads = {}
        raw_mode_norm = 0.0
        for dm, grad in zip(self.data_modes, grads):
            if grad is None:
                clean_grads[dm] = torch.zeros_like(leaves[dm])
                continue
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            mask = ab_mask
            while mask.dim() < grad.dim():
                mask = mask.unsqueeze(-1)
            grad = grad * mask
            clean_grads[dm] = grad
            joint_sq = joint_sq + grad.reshape(grad.shape[0], -1).square().sum(dim=1)
            raw_mode_norm += float(
                grad.reshape(grad.shape[0], -1).norm(dim=1).mean().detach().cpu()
            )

        joint_norm = joint_sq.sqrt().clamp(min=1e-12)
        if self.rgt_normalize_gradient:
            scale = self.rgt_grad_norm / joint_norm
        else:
            scale = (self.rgt_grad_norm / joint_norm).clamp(max=1.0)
        adjoint = {
            dm: grad * scale.view(-1, *([1] * (grad.dim() - 1)))
            for dm, grad in clean_grads.items()
        }
        return adjoint, reward.detach(), float(joint_norm.mean().detach().cpu())

    def _time_indices(self, trajectory: Dict) -> List[int]:
        """Select deterministic, rollout-aligned states across the trajectory."""

        n_steps = len(trajectory["steps"])
        n_samples = min(self.rgt_n_time_samples, n_steps)
        if n_samples <= 1:
            # The terminal-adjoint-only control must use the final recorded
            # state, not the initial noise state.  The last ledger entry is
            # the pre-terminal integration state and is the closest
            # rollout-aligned point available without creating a new sample.
            return [max(n_steps - 1, 0)]
        points = torch.linspace(0, n_steps - 1, n_samples).round().long().tolist()
        return sorted(set(int(point) for point in points))

    def _transport_factor(self, t: float) -> float:
        t = min(max(float(t), 0.0), 1.0)
        return (1.0 - self.rgt_adjoint_decay) + self.rgt_adjoint_decay * (
            t**self.rgt_time_power
        )

    @staticmethod
    def _trajectory_mean_time(trajectory: Dict, selected: List[int], dm: str) -> float:
        """Return the amplitude-matching mean t_x for one trajectory/mode."""

        values = []
        for index in selected:
            value = trajectory["steps"][index]["t"][dm]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            values.append(float(value))
        return sum(values) / max(len(values), 1)

    def _rgt_update(self, trajectories: List[Dict], batch: Dict) -> Dict:
        device = self.model.device
        ab_mask = batch["_ab_mask"].float()
        denom = ab_mask.sum(dim=1).clamp(min=1.0)

        self.optimizer.zero_grad()
        total_loss = 0.0
        total_reward = 0.0
        total_adj_norm = 0.0
        total_raw_norm = 0.0
        count = 0
        schedules, gt_schedule = self._schedules(self.nsteps_sample, device)
        step_params = {
            dm: self.sampling_model_args[dm]["simulation_step_params"]
            for dm in self.data_modes
        }

        prepared = []
        for trajectory in trajectories:
            x1 = {dm: trajectory["x_1"][dm].to(device) for dm in self.data_modes}
            adjoint, reward, raw_norm = self._adjoint_seed(x1, batch)
            if self.randomize_reward_gradient:
                adjoint = self._random_adjoint(adjoint, batch)
            total_reward += float(reward.mean().cpu())
            joint_adj_sq = torch.zeros(reward.shape[0], device=device)
            for grad in adjoint.values():
                joint_adj_sq = joint_adj_sq + grad.reshape(grad.shape[0], -1).square().sum(dim=1)
            total_adj_norm += float(joint_adj_sq.sqrt().mean().detach().cpu())
            total_raw_norm += raw_norm
            prepared.append((trajectory, adjoint, reward))

        if self.shuffle_reward_gradient and len(prepared) > 1:
            permutation = torch.randperm(len(prepared)).tolist()
            # Make the negative control a genuine cross-rollout assignment,
            # even in the vanishingly unlikely identity-permutation case.
            if all(index == value for index, value in enumerate(permutation)):
                permutation = permutation[1:] + permutation[:1]
        else:
            permutation = list(range(len(prepared)))

        for trajectory_index, (trajectory, _own_adjoint, reward) in enumerate(prepared):
            adjoint = prepared[permutation[trajectory_index]][1]
            selected = self._time_indices(trajectory)
            mean_time_factors = {
                dm: self._trajectory_mean_time(trajectory, selected, dm)
                for dm in self.data_modes
            }
            for step_index in selected:
                record = trajectory["steps"][step_index]
                x_t = {
                    dm: record["x_t"][dm].to(device).detach().clone()
                    for dm in self.data_modes
                }
                _restore_antigen(x_t, batch)
                t_scalars = record["t"]
                self._dt = {
                    dm: float(
                        schedules[dm][step_index + 1] - schedules[dm][step_index]
                    )
                    for dm in self.data_modes
                }
                # The RGT update only needs the velocity output.  Set the
                # integration metadata for the shared drift helper as well.
                self._gt_step = {
                    dm: float(gt_schedule[dm][step_index])
                    for dm in self.data_modes
                }
                drift = self._one_step_drift(
                    batch, x_t, t_scalars, step_params, record=True
                )

                loss = torch.zeros((), device=device)
                for dm in self.data_modes:
                    velocity = drift[dm][3]
                    if self.rgt_transport_mode == "time_weighted":
                        factor = self._transport_factor(t_scalars[dm])
                    elif self.rgt_transport_mode == "constant_mean":
                        factor = mean_time_factors[dm]
                    else:
                        factor = 1.0
                    transported = adjoint[dm] * (self.rgt_eta * factor)
                    # Directional matching is numerically equivalent to the
                    # detached-target squared loss, but removes the large
                    # common v.detach() term from the scalar and makes the
                    # update's intended direction explicit.
                    directional = velocity * transported.detach()
                    while directional.dim() > 2:
                        directional = directional.mean(dim=-1)
                    loss = loss - ((directional * ab_mask).sum(dim=1) / denom).mean()

                normalizer = max(len(selected) * len(trajectories), 1)
                (loss / normalizer).backward()
                total_loss += float(loss.detach().cpu())
                count += 1

        trainable = [p for p in self.model.nn.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=self.rgt_model_grad_clip)
        self.optimizer.step()
        n_trajectories = max(len(trajectories), 1)
        return {
            "loss": total_loss / max(count, 1),
            "reward": total_reward / n_trajectories,
            "adj_norm": total_adj_norm / n_trajectories,
            "raw_adj_norm": total_raw_norm / n_trajectories,
            "n_grad_steps": count,
            "n_trajectories": len(trajectories),
            "transport_mode": self.rgt_transport_mode,
            "shuffle_reward_gradient": int(self.shuffle_reward_gradient),
            "randomize_reward_gradient": int(self.randomize_reward_gradient),
        }

    def step(self, batch: Dict) -> Dict:
        self._step += 1
        self.model.nn.train()
        self.model.autoencoder.eval()

        ab_mask, full_mask = self._build_masks(batch)
        batch["_ab_mask"] = ab_mask
        batch["_ab_full_mask"] = full_mask
        trajectories = self._rollout(batch)
        update = self._rgt_update(trajectories, batch)

        metrics = {
            "rl/loss": update["loss"],
            "rl/reward_mean": update["reward"],
            "rl/rgt_adj_norm": update["adj_norm"],
            "rl/rgt_raw_adj_norm": update["raw_adj_norm"],
            "rl/rgt_step": self._step,
            "rl/rgt_shuffle_reward_gradient": update["shuffle_reward_gradient"],
            "rl/rgt_randomize_reward_gradient": update["randomize_reward_gradient"],
            "rl/rgt_n_grad_steps": update["n_grad_steps"],
            "rl/rgt_n_trajectories": update["n_trajectories"],
            "rl/rgt_transport_mode": update["transport_mode"],
        }
        pop_metrics = getattr(self.reward_fn, "pop_metrics", None)
        if callable(pop_metrics):
            metrics.update(pop_metrics())
        if self._step % self.log_interval == 0:
            logger.info(
                "[RGT step=%d] loss=%.6f reward=%.4f adj_norm=%.4f raw_adj_norm=%.3e",
                self._step,
                update["loss"],
                update["reward"],
                update["adj_norm"],
                update["raw_adj_norm"],
            )
        if self._step % self.ckpt_interval == 0:
            self._save_checkpoint()
        return metrics

    def _save_checkpoint(self, tag: str = "") -> None:
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self.ckpt_dir, f"rgt_step{self._step:06d}{suffix}.pt")
        torch.save(
            {
                "step": self._step,
                "method": "rgt",
                "nn_state": self.model.nn.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("[RGT] checkpoint -> %s", path)


class GraftRGTTrainer(RGTTrainer):
    """RAFA-RGT with the matched GRAFT latent-spread rollout policy.

    The base :class:`RGTTrainer` records the exact stochastic trajectory used
    by its reward-gradient update.  This hybrid keeps that trajectory record,
    but applies GRAFT's opt-in CDR-only latent-spread transform to the whole
    recorded rollout pool before reward scoring.  Applying the same transform
    to ``x_t``, ``x_next`` and ``x_1`` keeps the reward and transport states
    matched; it does not create a second, unrelated update trajectory.

    The default RGT arm never instantiates this class, so its historical
    protocol and checkpoint lineage remain unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_latent_spread_scale = float(
            self.cfg.get("rollout_latent_spread_scale", 1.0)
        )
        if self.rollout_latent_spread_scale < 1.0:
            raise ValueError("rollout_latent_spread_scale must be >= 1")
        self.rollout_pool_multiplier = int(self.cfg.get("rollout_pool_multiplier", 1))
        if self.rollout_pool_multiplier < 1:
            raise ValueError("rollout_pool_multiplier must be >= 1")
        self.decode_for_diversity = bool(self.cfg.get("decode_for_diversity", False))
        self._last_rollout_pool_size = self.K * self.rollout_pool_multiplier
        self._last_rollout_selection_min_distance = 0.0
        self._last_rollout_trajectories: List[Dict] = []

    @staticmethod
    def _cdr_mask(batch: Dict, ab_mask: torch.Tensor) -> torch.Tensor:
        cdr_mask = batch.get("native_cdr_mask", batch.get("cdr_mask"))
        if cdr_mask is None:
            cdr_mask = ab_mask
        if cdr_mask.dim() == 1:
            cdr_mask = cdr_mask.unsqueeze(0)
        return ab_mask.bool() & cdr_mask.bool()

    @staticmethod
    def _spread_values(values: List[torch.Tensor], mask: torch.Tensor, scale: float):
        stacked = torch.stack(values, dim=0)
        center = stacked.mean(dim=0, keepdim=True)
        spread = center + scale * (stacked - center)
        # ``stacked`` is [pool, batch, residue, feature].  Align the
        # batch/residue mask to those middle dimensions, then broadcast over
        # the feature axis.
        while mask.dim() < spread.dim() - 1:
            mask = mask.unsqueeze(0)
        if mask.dim() == spread.dim() - 1:
            mask = mask.unsqueeze(-1)
        return [value.detach() for value in torch.where(mask, spread, stacked)]

    @torch.no_grad()
    def _apply_latent_spread(self, trajectories: List[Dict], batch: Dict) -> None:
        if self.rollout_latent_spread_scale == 1.0 or not trajectories:
            return
        if "local_latents" not in trajectories[0]["x_1"]:
            return
        ab_mask = batch["_ab_mask"].bool()
        cdr_mask = self._cdr_mask(batch, ab_mask)
        scale = self.rollout_latent_spread_scale

        terminal = [trajectory["x_1"]["local_latents"] for trajectory in trajectories]
        for trajectory, value in zip(
            trajectories, self._spread_values(terminal, cdr_mask.to(terminal[0].device), scale)
        ):
            trajectory["x_1"]["local_latents"] = value

        # The rollout ledger is kept on CPU.  Transform every recorded local
        # latent state with the same pool center so the transport update sees
        # the candidate policy that produced the terminal reward.
        for step_index in range(len(trajectories[0]["steps"])):
            for field in ("x_t", "x_next", "mean"):
                values = [trajectory["steps"][step_index][field]["local_latents"]
                          for trajectory in trajectories]
                spread = self._spread_values(values, cdr_mask.to(values[0].device), scale)
                for trajectory, value in zip(trajectories, spread):
                    trajectory["steps"][step_index][field]["local_latents"] = value

    @torch.no_grad()
    def _select_diverse_trajectories(self, pool: List[Dict], batch: Dict) -> List[Dict]:
        """Greedily select K trajectories using decoded CDR Hamming distance."""
        if len(pool) <= self.K:
            return pool
        autoencoder = getattr(self.model, "autoencoder", None)
        if autoencoder is None:
            raise RuntimeError("hybrid rollout selection requires the autoencoder")
        ab_mask = batch["_ab_mask"].bool()
        batch_size, n_res = ab_mask.shape
        cdr_mask = self._cdr_mask(batch, ab_mask)
        z = torch.cat([item["x_1"]["local_latents"] for item in pool], dim=0)
        ca = torch.cat([item["x_1"]["bb_ca"] for item in pool], dim=0)
        decoded = autoencoder.decode(
            z_latent=z,
            ca_coors_nm=ca,
            mask=ab_mask.repeat(len(pool), 1),
        )
        seqs = decoded["residue_type"].long().reshape(len(pool), batch_size, n_res)

        selections: List[List[int]] = []
        distances: List[float] = []
        for b in range(batch_size):
            mask_b = cdr_mask[b]
            if not bool(mask_b.any()):
                mask_b = ab_mask[b]
            selected = [0]
            available = torch.ones(len(pool), dtype=torch.bool, device=seqs.device)
            available[0] = False
            while len(selected) < self.K:
                candidates = torch.nonzero(available, as_tuple=False).flatten()
                chosen = torch.tensor(selected, dtype=torch.long, device=seqs.device)
                identity = (
                    seqs[candidates, b][:, None, mask_b]
                    == seqs[chosen, b][None, :, mask_b]
                ).float().mean(dim=-1)
                min_distance = 1.0 - identity.max(dim=1).values
                offset = int(min_distance.argmax().item())
                index = int(candidates[offset].item())
                selected.append(index)
                available[index] = False
                distances.append(float(min_distance[offset].item()))
            selections.append(selected)

        def assemble(slot: int) -> Dict:
            pieces = [pool[selections[b][slot]] for b in range(batch_size)]
            out = {"x_1": {}, "steps": []}
            for dm in pool[0]["x_1"]:
                out["x_1"][dm] = torch.cat(
                    [piece["x_1"][dm][b:b + 1] for b, piece in enumerate(pieces)], dim=0
                )
            for step_index in range(len(pool[0]["steps"])):
                reference = pieces[0]["steps"][step_index]
                record = {
                    "t": dict(reference["t"]),
                    "std": dict(reference["std"]),
                    "stoch": dict(reference["stoch"]),
                    "x_t": {},
                    "x_next": {},
                    "mean": {},
                }
                for field in ("x_t", "x_next", "mean"):
                    for dm in reference[field]:
                        record[field][dm] = torch.cat(
                            [piece["steps"][step_index][field][dm][b:b + 1]
                             for b, piece in enumerate(pieces)], dim=0
                        )
                out["steps"].append(record)
            return out

        self._last_rollout_selection_min_distance = (
            sum(distances) / len(distances) if distances else 0.0
        )
        return [assemble(slot) for slot in range(self.K)]

    @torch.no_grad()
    def _rollout(self, batch: Dict) -> List[Dict]:
        original_k = self.K
        self.K = original_k * self.rollout_pool_multiplier
        try:
            trajectories = super()._rollout(batch)
        finally:
            self.K = original_k
        self._apply_latent_spread(trajectories, batch)
        self._last_rollout_pool_size = len(trajectories)
        selected = self._select_diverse_trajectories(trajectories, batch)
        self._last_rollout_trajectories = selected
        return selected

    @torch.no_grad()
    def _diversity_metrics(self, trajectories: List[Dict], batch: Dict) -> Dict:
        if not self.decode_for_diversity or not trajectories:
            return {}
        autoencoder = getattr(self.model, "autoencoder", None)
        if autoencoder is None:
            return {}
        ab_mask = batch["_ab_mask"].bool()
        cdr_mask = self._cdr_mask(batch, ab_mask)
        z = torch.cat([item["x_1"]["local_latents"] for item in trajectories], dim=0)
        ca = torch.cat([item["x_1"]["bb_ca"] for item in trajectories], dim=0)
        seqs = autoencoder.decode(
            z_latent=z,
            ca_coors_nm=ca,
            mask=ab_mask.repeat(len(trajectories), 1),
        )["residue_type"].long()
        batch_size = ab_mask.shape[0]
        metrics = []
        for b in range(batch_size):
            indices = torch.arange(b, len(trajectories) * batch_size, batch_size, device=seqs.device)
            metrics.append(self.monitor.update(seqs[indices], cdr_mask[b].unsqueeze(0).expand(len(indices), -1)))
        if len(metrics) == 1:
            return metrics[0]
        return {key: sum(float(item[key]) for item in metrics) / len(metrics) for key in metrics[0]}

    def step(self, batch: Dict) -> Dict:
        metrics = super().step(batch)
        diversity = self._diversity_metrics(self._last_rollout_trajectories, batch)
        metrics.update(diversity)
        metrics.update({
            "rl/graft_rgt": 1,
            "rl/rollout_latent_spread_scale": self.rollout_latent_spread_scale,
            "rl/rollout_pool_multiplier": self.rollout_pool_multiplier,
            "rl/rollout_pool_size": self._last_rollout_pool_size,
            "rl/rollout_selection_min_distance": self._last_rollout_selection_min_distance,
        })
        return metrics
