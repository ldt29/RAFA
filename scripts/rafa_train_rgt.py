#!/usr/bin/env python3
"""Train one RAFA-RGT release model from the released shared base.

The historical estimator modules have Hydra entry points tied to an old
reward/checkpoint contract.  This runner is the explicit bridge for the
current experiment: it loads the immutable antibody envelope, overlays the
shared NN-only base, decodes each rollout for the R_free sequence/developability
terms, and uses identical data, K, ODE steps, learning rate, and update count
for Flow-GRPO, DiffusionNFT, TempFlow-GRPO, and the disclosed RAM control.

The runner also exposes a matched ``graft`` arm for a direct GRAFT-only
comparison.  It intentionally writes a compact final checkpoint and
per-update metrics but does not modify either base checkpoint or any input
dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proteinfoundation.datasets.ab_data import AntibodyDesignDataset, collate_fn
from proteinfoundation.posttraining.diversity_monitor import DiversityCollapseError, DiversityMonitor
from proteinfoundation.posttraining.diffusion_nft_trainer import DiffusionNFTTrainer
from proteinfoundation.posttraining.flow_grpo_trainer import (
    FlowGRPOTrainer,
    _batch_to_device,
)
from proteinfoundation.posttraining.grpo_trainer import GRAFTTrainer
from proteinfoundation.posttraining.grpo_trainer import _load_nn_only_lineage
from proteinfoundation.posttraining.ram_trainer import RAMTrainer
from proteinfoundation.posttraining.reward_fns import get_reward_fn
from proteinfoundation.posttraining.rgt_trainer import GraftRGTTrainer, RGTTrainer
from proteinfoundation.posttraining.tempflow_grpo_trainer import TempFlowGRPOTrainer
from proteinfoundation.proteina import Proteina


METHODS = {
    "graft": GRAFTTrainer,
    "flow_grpo": FlowGRPOTrainer,
    "diffusion_nft": DiffusionNFTTrainer,
    "tempflow_grpo": TempFlowGRPOTrainer,
    "ram": RAMTrainer,
    "rgt": RGTTrainer,
    "graft_rgt": GraftRGTTrainer,
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


class DecodedRFreeReward:
    """Decode latent rollouts before evaluating the R_free callable.

    Flow-based competitor trainers expose only ``bb_ca`` and ``local_latents``
    at the end of a rollout.  R_free also needs the decoded residue sequence
    for its independent teacher and developability term.  The RAM control must
    retain the graph from coordinates/latents to the structural reward, while
    the other three controls only need detached scalar rewards.
    """

    def __init__(self, model, reward_fn, keep_graph: bool):
        self.model = model
        self.reward_fn = reward_fn
        self.keep_graph = keep_graph

    def __call__(self, sample: dict, batch: dict) -> torch.Tensor:
        mask = batch["mask"].bool()
        if self.keep_graph:
            decoded = self.model.autoencoder.decode(
                z_latent=sample["local_latents"],
                ca_coors_nm=sample["bb_ca"],
                mask=mask,
            )
            scored = dict(sample)
            scored["residue_type"] = decoded["residue_type"].long()
            return self.reward_fn(scored, batch)
        with torch.no_grad():
            decoded = self.model.autoencoder.decode(
                z_latent=sample["local_latents"],
                ca_coors_nm=sample["bb_ca"],
                mask=mask,
            )
            scored = dict(sample)
            scored["residue_type"] = decoded["residue_type"].long()
            return self.reward_fn(scored, batch)

    def pop_metrics(self):
        """Expose component metrics from the wrapped reward to the trainer."""
        pop_metrics = getattr(self.reward_fn, "pop_metrics", None)
        return pop_metrics() if callable(pop_metrics) else {}


def finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{path}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"non-finite metric at {path}: {value}")


def build_cfg(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "K": args.K,
        "nsteps_sample": args.nsteps,
        "lr": args.lr,
        "max_iters": args.max_iters,
        "freeze_pair_update": True,
        "log_interval": args.log_interval,
        # Disable the legacy trainer checkpoints; the runner writes one
        # provenance-complete final artifact after the loop.
        "ckpt_interval": 1_000_000,
        "ckpt_dir": str(run_dir),
        "clip_eps": 0.2,
        "adv_eps": 1e-4,
        "timestep_subsample": 10,
        "grad_accum_microbatch": 1,
        "reward_tier": "r_free_ref" if args.reference_reward != "none" else "r_free",
        "reference_reward_mode": args.reference_reward,
        "reference_reward_weight": float(args.reference_reward_weight),
        "affinity_teacher_weight": 0.25,
        "affinity_teacher_fusion": "linear",
        "affinity_kwargs": {
            "weights": {
                "contact_coverage": 0.60,
                "buried_surface_proxy": 0.25,
                "interface_compactness": 0.15,
            }
        },
        "rfree_weights": {
            "affinity": 0.60,
            "fold_confidence": 0.20,
            "physical": 0.15,
            "developability": 0.05,
        },
        # Method-specific defaults are frozen here so all four runs have a
        # complete, comparable protocol rather than hidden module defaults.
        "nft_pos_quantile": 0.5,
        "nft_w_pos": 1.0,
        "nft_w_neg": 1.0,
        "nft_neg_clamp": 1.0,
        "nft_n_time_samples": 4,
        "temp_mode": "weight",
        "temp_alpha": 1.0,
        "ram_eta": 1.0,
        "ram_adjoint_decay": 1.0,
        "ram_grad_clip": 1.0,
        "ram_n_time_samples": 4,
        "ram_normalize_adjoint": True,
        "rgt_model_grad_clip": 1.0,
        "rgt_n_time_samples": int(args.rgt_n_time_samples),
        "rgt_normalize_gradient": True,
        "shuffle_reward_gradient": bool(args.shuffle_reward_gradient),
        "randomize_reward_gradient": bool(args.randomize_reward_gradient),
        "rgt_adjoint_decay": float(args.rgt_adjoint_decay),
        "rgt_time_power": float(args.rgt_time_power),
        "rgt_transport_mode": str(args.rgt_transport_mode),
        "rgt_eta": float(args.rgt_eta),
        "rgt_grad_norm": float(args.rgt_grad_norm),
        "rollout_latent_spread_scale": (
            float(args.rollout_latent_spread_scale)
            if args.rollout_latent_spread_scale is not None
            else (2.0 if args.method == "graft_rgt" else 1.0)
        ),
        "rollout_pool_multiplier": (
            int(args.rollout_pool_multiplier)
            if args.rollout_pool_multiplier is not None else 1
        ),
        "decode_for_diversity": bool(
            args.decode_for_diversity or args.method == "graft_rgt"
        ),
        # Keep the matched GRAFT-only arm explicit rather than relying on
        # module defaults.  Its rollout policy can be set to the same
        # latent-spread scale as GRAFT+RGT at launch time.
        "alpha": 0.875,
        "advantage_scale": 1.0,
        "advantage_normalize": False,
        "advantage_objective": "signed_fm",
        "advantage_temperature": 0.1,
        "advantage_clip": 3.0,
        "trust_region_weight": 0.0,
    }
    return cfg


def _load_staged_nn_state(model: Any, path: Path) -> dict[str, Any]:
    """Overlay a prior staged runner artifact after loading shared base."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("nn_state"), dict):
        raise ValueError(f"staged init has no dict nn_state: {path}")
    state = payload["nn_state"]
    bad_tensors = [
        key for key, value in state.items()
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all().item()
    ]
    if bad_tensors:
        raise ValueError(f"staged init contains non-finite/non-tensor values: {bad_tensors[:5]}")
    missing, unexpected = model.nn.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(
            f"staged init is incompatible with the RAFA student model: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return {
        "path": str(path.resolve()),
        "artifact_type": payload.get("artifact_type"),
        "method": payload.get("method"),
        "step": payload.get("step"),
        "trainer": payload.get("trainer"),
    }


def make_model(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    model = Proteina.load_from_checkpoint(
        str(args.base_checkpoint),
        map_location="cpu",
        strict=False,
        autoencoder_ckpt_path=str(args.autoencoder),
    )
    lineage = _load_nn_only_lineage(model, str(args.shared_state))
    model.ab_design_mode = True
    inf_cfg_path = ROOT / "configs/inference_ab_design.yaml"
    if inf_cfg_path.exists():
        model.inf_cfg = OmegaConf.load(inf_cfg_path).generation
    provenance = {
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_mutated": False,
        "base_checkpoint_role": (
            "immutable Lightning/envelope loader; the student NN is replaced "
            "by the explicit shared NN-only initialization below"
        ),
        "init_nn_state": lineage,
    }
    if args.init_rl_state is not None:
        provenance["staged_init"] = _load_staged_nn_state(model, args.init_rl_state)
    return model, provenance


def make_reward(args: argparse.Namespace, model: Any, keep_graph: bool):
    from proteinfoundation.posttraining.affinity_teacher import AffinitySequenceTeacher

    teacher = AffinitySequenceTeacher.load(str(args.affinity_teacher))
    reward_tier = "r_free_ref" if args.reference_reward != "none" else "r_free"
    reward = get_reward_fn(
        reward_tier,
        autoencoder=model.autoencoder,
        affinity_kwargs={
            "weights": {
                "contact_coverage": 0.60,
                "buried_surface_proxy": 0.25,
                "interface_compactness": 0.15,
            }
        },
        sequence_teacher=teacher,
        sequence_teacher_weight=0.25,
        sequence_teacher_fusion="linear",
        fold_surrogate_path=str(args.fold_surrogate),
        rfree_weights={
            "affinity": 0.60,
            "fold_confidence": 0.20,
            "physical": 0.15,
            "developability": 0.05,
        },
        reference_mode=args.reference_reward,
        reference_weight=float(args.reference_reward_weight),
    )
    return DecodedRFreeReward(model, reward, keep_graph=keep_graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--shared-state", type=Path, required=True)
    parser.add_argument("--autoencoder", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--affinity-teacher", type=Path, required=True)
    parser.add_argument("--fold-surrogate", type=Path, required=True)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--nsteps", type=int, default=50)
    parser.add_argument("--max-iters", type=int, default=150)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--rollout-latent-spread-scale", type=float, default=None)
    parser.add_argument("--rollout-pool-multiplier", type=int, default=None)
    parser.add_argument("--decode-for-diversity", action="store_true")
    parser.add_argument(
        "--reference-reward",
        choices=("none", "lddt", "tm", "dockq", "all"),
        default="none",
        help=(
            "Train/validation-only native-reference structural reward. "
            "Use one component for ablations or all for the combined arm."
        ),
    )
    parser.add_argument(
        "--reference-reward-weight",
        type=float,
        default=0.25,
        help="Mixture weight replacing the native-free R_free component.",
    )
    parser.add_argument(
        "--shuffle-reward-gradient",
        action="store_true",
        help="Negative control: assign each terminal reward gradient to another rollout.",
    )
    parser.add_argument(
        "--randomize-reward-gradient",
        action="store_true",
        help="Mechanism control: replace the reward-gradient direction with a same-norm random direction.",
    )
    parser.add_argument(
        "--rgt-adjoint-decay",
        type=float,
        default=1.0,
        help="RGT time-transport decay; 0 gives a constant terminal-adjoint factor across time.",
    )
    parser.add_argument(
        "--rgt-transport-mode",
        choices=("time_weighted", "constant_mean", "constant_unit"),
        default="time_weighted",
        help=(
            "Transport schedule. constant_mean uses the per-trajectory mean "
            "recorded t_x so its average amplitude matches t_x*g."
        ),
    )
    parser.add_argument("--rgt-time-power", type=float, default=1.0)
    parser.add_argument("--rgt-eta", type=float, default=1.0)
    parser.add_argument("--rgt-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--rgt-n-time-samples",
        type=int,
        default=4,
        help=(
            "Number of recorded rollout times used by RGT.  Setting this to "
            "1 selects the final recorded pre-terminal state and gives the "
            "matched terminal-adjoint-only control."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("formal", "extended", "reference_pilot", "graft_warmup", "rgt_transport", "negative_control", "mechanism_control"),
        default="formal",
        help="Use a declared extended or non-formal staged protocol when requested.",
    )
    parser.add_argument(
        "--init-rl-state",
        type=Path,
        default=None,
        help="Prior staged runner final checkpoint whose nn_state seeds this run.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run exactly one update for wiring/shape validation; never a formal arm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    for path in (
        args.base_checkpoint,
        args.shared_state,
        args.autoencoder,
        args.dataset,
        args.affinity_teacher,
        args.fold_surrogate,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.stage == "formal":
        if args.shuffle_reward_gradient:
            raise ValueError("formal arms cannot enable --shuffle-reward-gradient")
        if args.init_rl_state is not None:
            raise ValueError("--init-rl-state is only valid for --stage rgt_transport")
        if args.K != 8 or args.nsteps != 50 or (not args.preflight and args.max_iters != 150):
            raise ValueError("formal matched competitor protocol is K=8, nsteps=50, max_iters=150")
    elif args.stage == "extended":
        if args.method != "rgt":
            raise ValueError("extended protocol is currently restricted to method=rgt")
        if args.shuffle_reward_gradient:
            raise ValueError("extended runs cannot enable --shuffle-reward-gradient")
        if args.init_rl_state is not None:
            raise ValueError("--init-rl-state is only valid for --stage rgt_transport")
        if args.K != 8 or args.nsteps != 50 or args.max_iters <= 150:
            raise ValueError("extended RGT protocol requires K=8, nsteps=50, max_iters>150")
    elif args.stage == "reference_pilot":
        if args.method != "rgt":
            raise ValueError("reference_pilot is currently restricted to method=rgt")
        if args.shuffle_reward_gradient or args.init_rl_state is not None:
            raise ValueError("reference_pilot cannot use shuffled gradients or staged init")
        if args.K != 8 or args.nsteps != 50 or args.max_iters != 50:
            raise ValueError("reference_pilot protocol is method=rgt, K=8, nsteps=50, max_iters=50")
    elif args.stage == "graft_warmup":
        if args.method != "graft_rgt" or args.init_rl_state is not None:
            raise ValueError("graft_warmup requires method=graft_rgt and no staged init")
        if args.K != 8 or args.nsteps != 50 or args.max_iters != 50:
            raise ValueError("graft_warmup protocol is method=graft_rgt, K=8, nsteps=50, max_iters=50")
    elif args.stage == "rgt_transport":
        if args.method != "rgt" or args.init_rl_state is None:
            raise ValueError("rgt_transport requires method=rgt and --init-rl-state")
        if args.K != 8 or args.nsteps != 50 or args.max_iters != 100:
            raise ValueError("rgt_transport protocol is method=rgt, K=8, nsteps=50, max_iters=100")
    elif args.stage == "negative_control":
        if args.method not in {"rgt", "graft_rgt"}:
            raise ValueError("negative_control requires method=rgt or graft_rgt")
        if (not args.shuffle_reward_gradient and not args.randomize_reward_gradient) or args.init_rl_state is not None:
            raise ValueError(
                "negative_control requires a shuffled or random reward gradient and no staged init"
            )
        if args.shuffle_reward_gradient and args.randomize_reward_gradient:
            raise ValueError("choose either shuffled or random reward gradient, not both")
        if args.K != 8 or args.nsteps != 50 or args.max_iters != 50:
            raise ValueError(
                "negative_control protocol is K=8, nsteps=50, max_iters=50"
            )
    elif args.stage == "mechanism_control":
        if args.method != "rgt" or args.init_rl_state is not None:
            raise ValueError("mechanism_control requires method=rgt and no staged init")
        if args.K != 8 or args.nsteps != 50 or args.max_iters != 150:
            raise ValueError("mechanism_control protocol is RGT, K=8, nsteps=50, max_iters=150")
        if args.shuffle_reward_gradient and args.randomize_reward_gradient:
            raise ValueError("choose either shuffled or random reward gradient, not both")
    if args.init_rl_state is not None:
        args.init_rl_state = args.init_rl_state.resolve()
        if not args.init_rl_state.exists():
            raise FileNotFoundError(args.init_rl_state)
    if not 0.0 <= args.rgt_adjoint_decay <= 1.0:
        raise ValueError("--rgt-adjoint-decay must be in [0, 1]")
    if args.rgt_time_power < 0.0 or args.rgt_eta <= 0.0 or args.rgt_grad_norm <= 0.0:
        raise ValueError("RGT time power must be nonnegative and eta/grad norm must be positive")
    if args.rgt_n_time_samples < 1:
        raise ValueError("--rgt-n-time-samples must be positive")
    if args.preflight:
        args.max_iters = 1
    args.run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, provenance = make_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    keep_graph = args.method in {"ram", "rgt", "graft_rgt"}
    reward_fn = make_reward(args, model, keep_graph=keep_graph)
    monitor = DiversityMonitor(
        entropy_early_stop_bits=1.0,
        psi_warn_threshold=0.9,
        log_every=args.log_interval,
    )
    cfg = build_cfg(args, args.run_dir)
    trainer_cls = METHODS[args.method]
    trainer = trainer_cls(model=model, rl_cfg=cfg, reward_fn=reward_fn, diversity_monitor=monitor)

    dataset = AntibodyDesignDataset(
        data_dir=str(args.dataset),
        split="train",
        mask_strategy="mixed",
        mask_strategy_probs=[0.5, 0.3, 0.2],
        max_antibody_len=450,
        max_antigen_len=500,
        conventional_only=True,
        vhh_only=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    metrics_path = args.run_dir / "metrics.jsonl"
    config_payload = {
        "artifact_type": "shared_base_rfree_competitor_protocol",
        "method": args.method,
        "stage": args.stage,
        "seed": args.seed,
        "run_dir": str(args.run_dir),
        "config": cfg,
        "dataset": {
            "data_dir": str(args.dataset.resolve()),
            "split": "train",
            "mask_strategy": "mixed",
            "mask_strategy_probs": [0.5, 0.3, 0.2],
            "conventional_only": True,
            "vhh_only": False,
        },
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_role": provenance["base_checkpoint_role"],
        "shared_init": provenance["init_nn_state"],
        "student_initialization": "shared_init.nn_state",
        "staged_init": provenance.get("staged_init"),
        "autoencoder": str(args.autoencoder.resolve()),
        "affinity_teacher": str(args.affinity_teacher.resolve()),
        "fold_surrogate": str(args.fold_surrogate.resolve()),
        "claim_boundary": (
            "Reference-guided structural post-training: native antibody coordinates are used "
            "only on the train split for the differentiable reward; validation may be used "
            "for selection diagnostics, while final test selection remains native-free. "
            "No binding-affinity or general capability claim."
            if args.reference_reward != "none"
            else "Matched estimator control only; no binding-affinity or capability claim."
        ),
    }
    (args.run_dir / "protocol.json").write_text(
        json.dumps(jsonable(config_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    completed = 0
    status = "completed"
    try:
        with metrics_path.open("w", encoding="utf-8") as handle:
            while completed < args.max_iters:
                for batch in loader:
                    if completed >= args.max_iters:
                        break
                    metrics = trainer.step(_batch_to_device(batch, device))
                    finite_tree(metrics, "metrics")
                    handle.write(json.dumps(jsonable(metrics), sort_keys=True) + "\n")
                    handle.flush()
                    # RGT keeps exact rollout trajectories on CPU, but its
                    # per-time reward-gradient pass can leave large cached
                    # CUDA blocks when consecutive targets have different
                    # lengths.  Releasing the allocator cache here preserves
                    # the matched protocol while preventing a length-driven
                    # reserved-memory staircase during long formal runs.
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    completed += 1
                    if completed % args.log_interval == 0:
                        print(json.dumps({"method": args.method, "step": completed, "metrics": jsonable(metrics)}, sort_keys=True), flush=True)
                    try:
                        monitor.check_stop()
                    except DiversityCollapseError:
                        status = "stopped_diversity_collapse"
                        break
                if status != "completed":
                    break
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        if completed and status == "completed":
            final_path = args.run_dir / f"{args.method}_step{completed:06d}_final.pt"
            final_payload = {
                "artifact_type": "shared_base_rfree_competitor_nn_state",
                "step": completed,
                "method": args.method,
                "nn_state": {key: value.detach().cpu() for key, value in model.nn.state_dict().items()},
                "trainer": trainer.__class__.__name__,
                "rl_cfg": jsonable(cfg),
                "checkpoint_provenance": provenance,
                "status": status,
            }
            if args.stage in {"formal", "extended"}:
                final_payload["optimizer"] = trainer.optimizer.state_dict()
            else:
                # Staged artifacts are consumed only as nn_state initialization
                # and should not carry a redundant multi-GB Adam state.
                final_payload["checkpoint_storage"] = "nn_state_only_staged"
            torch.save(final_payload, final_path)
            config_payload["final_checkpoint"] = str(final_path)
        config_payload["status"] = status
        config_payload["completed_updates"] = completed
        (args.run_dir / "protocol.json").write_text(
            json.dumps(jsonable(config_payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if status != "completed":
        raise RuntimeError(f"competitor run ended with status={status} after {completed} updates")
    print(json.dumps({"method": args.method, "run_dir": str(args.run_dir), "updates": completed, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
