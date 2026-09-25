"""GT-free affinity-proxy evaluation for a frozen base or RL checkpoint.

Validation/test masking is deliberately hard-coded to ``full`` here.  This
entry point reports the affinity-related proxy and its interpretable terms;
native antibody coordinates and DockQ are not read by the scoring path.

Example (on an allowed GPU):
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 python -m \
      proteinfoundation.posttraining.evaluate_affinity \
      --rl-state artifacts/graft_gpu_k8_pilot10/grpo_step000010_final.pt \
      --limit 32 --K 8 --nsteps 50 --out artifacts/eval/graft_valid.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from proteinfoundation.datasets.ab_data import AntibodyDesignDataset, collate_fn
from proteinfoundation.posttraining.grpo_trainer import GRPOTrainer, _batch_to_device
from proteinfoundation.posttraining.reward_fns import (
    ca_dockq_affinity_proxy_reward,
    ca_dockq_proxy_reward,
    fuse_affinity_teacher_reward,
    get_reward_fn,
    reference_structure_scores,
)
from proteinfoundation.posttraining.affinity_teacher import AffinitySequenceTeacher
from proteinfoundation.proteina import Proteina


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ckpt",
        default=os.environ.get("BASE_CHECKPOINT", "./checkpoints/design_base.ckpt"),
    )
    parser.add_argument(
        "--autoencoder",
        default=os.environ.get("AUTOENCODER", "./checkpoints/ae.ckpt"),
    )
    parser.add_argument(
        "--init-nn-state",
        default=None,
        help="Optional nn-only lineage state loaded before --rl-state; keeps the base/AE files immutable.",
    )
    parser.add_argument("--rl-state", default=None, help="Optional GRPO/GRAFT .pt state dict")
    parser.add_argument(
        "--data-dir", default=os.environ.get("DATASET", "./data/structure_dataset")
    )
    parser.add_argument("--split", default="valid", choices=("valid", "test"))
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument(
        "--condition-policy",
        choices=("native_epitope", "all_antigen"),
        default="native_epitope",
        help=(
            "Conditioning mask policy. 'all_antigen' hides the native epitope "
            "and paratope labels, exposes every antigen residue, and recenters "
            "on the antigen centroid. The preprocessed crop remains fixed."
        ),
    )
    parser.add_argument(
        "--indices-file",
        default=None,
        help="Optional JSON list of original processed-split indices; preserves exact novel-stratum pairing.",
    )
    parser.add_argument(
        "--conventional-only",
        action="store_true",
        help="Evaluate only VH-VL entries (chain_type contains a light-chain label); preserve dataset indices.",
    )
    parser.add_argument(
        "--vhh-only",
        action="store_true",
        help="Evaluate only VHH entries (no light-chain label); preserve dataset indices.",
    )
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--nsteps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--reward-tier",
        choices=("affinity", "r_free"),
        default="affinity",
        help="Selection reward. R_free adds fold-confidence, physical, and developability terms.",
    )
    parser.add_argument(
        "--fold-surrogate",
        default=None,
        help="Train-only Protenix fold-confidence surrogate JSON (required for --reward-tier r_free).",
    )
    parser.add_argument("--rfree-affinity-weight", type=float, default=0.60)
    parser.add_argument("--rfree-fold-weight", type=float, default=0.20)
    parser.add_argument("--rfree-physical-weight", type=float, default=0.15)
    parser.add_argument("--rfree-developability-weight", type=float, default=0.05)
    parser.add_argument("--clash-gate-strength", type=float, default=None)
    parser.add_argument("--clash-tolerance", type=float, default=None)
    parser.add_argument(
        "--affinity-teacher",
        default=None,
        help="Optional frozen sequence-affinity teacher .npz; adds an independent measured-affinity signal.",
    )
    parser.add_argument("--affinity-teacher-weight", type=float, default=0.25)
    parser.add_argument(
        "--affinity-teacher-fusion",
        choices=("linear", "geometric"),
        default="linear",
        help="Fusion of structural proxy and independent sequence teacher.",
    )
    parser.add_argument("--affinity-contact-weight", type=float, default=0.45)
    parser.add_argument("--affinity-buried-weight", type=float, default=0.30)
    parser.add_argument("--affinity-compactness-weight", type=float, default=0.25)
    parser.add_argument("--affinity-rg-expected", type=float, default=None,
                        help="Optional antibody radius-of-gyration prior in nm.")
    parser.add_argument("--affinity-rg-tolerance", type=float, default=None,
                        help="Optional radius-of-gyration prior tolerance in nm.")
    parser.add_argument(
        "--affinity-radius-penalty-weight",
        type=float,
        default=None,
        help="Optional soft penalty weight for radius-prior mismatch.",
    )
    parser.add_argument(
        "--diagnostic-dockq",
        action="store_true",
        help="Also report native-pose CA-DockQ as a held-out diagnostic; never used for selection.",
    )
    parser.add_argument(
        "--diagnostic-reference",
        action="store_true",
        help=(
            "Report differentiable native-reference structural proxies for "
            "validation diagnostics only; never used for selection."
        ),
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.conventional_only and args.vhh_only:
        raise ValueError("--conventional-only and --vhh-only are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = Proteina.load_from_checkpoint(
        args.base_ckpt,
        map_location="cpu",
        strict=False,
        autoencoder_ckpt_path=args.autoencoder,
    )
    model.ab_design_mode = True
    # Resolve the sampling contract from this checkout rather than relying on
    # a node-specific historical mount alias.  This keeps paired readouts
    # reproducible across machines.
    repo_root = Path(__file__).resolve().parents[2]
    model.inf_cfg = OmegaConf.load(
        repo_root / "configs/inference_ab_design.yaml"
    ).generation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    state_step = None
    init_nn_state = None
    if args.init_nn_state:
        init_nn_state = torch.load(args.init_nn_state, map_location=device, weights_only=False)
        if "nn_state" not in init_nn_state:
            raise ValueError(f"--init-nn-state has no nn_state: {args.init_nn_state}")
        model.nn.load_state_dict(init_nn_state["nn_state"], strict=False)
    if args.rl_state:
        state = torch.load(args.rl_state, map_location=device, weights_only=False)
        if "nn_state" not in state:
            raise ValueError(f"--rl-state has no nn_state: {args.rl_state}")
        model.nn.load_state_dict(state["nn_state"], strict=False)
        state_step = state.get("step")
    model.nn.eval()
    model.autoencoder.eval()

    dataset = AntibodyDesignDataset(
        data_dir=args.data_dir,
        split=args.split,
        mask_strategy="full",  # formal evaluation boundary
        max_antibody_len=450,
        max_antigen_len=500,
        # Keep original processed-split indices for paired artifacts; apply
        # the VH-VL/VHH filter below without shrinking the dataset index map.
        conventional_only=False,
        vhh_only=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    cfg = OmegaConf.create({
        "K": args.K,
        "nsteps_sample": args.nsteps,
        "freeze_pair_update": True,
        "lr": 0.0,
        "ckpt_dir": os.path.dirname(args.out) or ".",
        "metrics_path": os.path.join(os.path.dirname(args.out) or ".", ".eval_metrics.jsonl"),
    })
    # The trainer is used only for its antigen-locked rollout implementation;
    # scoring below calls the affinity proxy directly so terms stay per sample.
    sampler = GRPOTrainer(model=model, rl_cfg=cfg, reward_fn=lambda x, b: ca_dockq_affinity_proxy_reward(x, b))
    sequence_teacher = None
    if args.affinity_teacher:
        if not 0.0 <= args.affinity_teacher_weight <= 1.0:
            raise ValueError("--affinity-teacher-weight must be in [0, 1]")
        sequence_teacher = AffinitySequenceTeacher.load(args.affinity_teacher)
    affinity_weights = {
        "contact_coverage": float(args.affinity_contact_weight),
        "buried_surface_proxy": float(args.affinity_buried_weight),
        "interface_compactness": float(args.affinity_compactness_weight),
    }
    if any(value < 0.0 for value in affinity_weights.values()) or not np.isclose(
        sum(affinity_weights.values()), 1.0, atol=1e-6
    ):
        raise ValueError(
            "affinity weights must be nonnegative and sum to one; got "
            f"{affinity_weights}"
        )
    affinity_kwargs = {"weights": affinity_weights}
    if args.clash_gate_strength is not None:
        affinity_kwargs["clash_gate_strength"] = args.clash_gate_strength
    if args.clash_tolerance is not None:
        affinity_kwargs["clash_tolerance"] = args.clash_tolerance
    if args.affinity_rg_expected is not None:
        affinity_kwargs["rg_expected_nm"] = args.affinity_rg_expected
    if args.affinity_rg_tolerance is not None:
        affinity_kwargs["rg_tolerance_nm"] = args.affinity_rg_tolerance
    if args.affinity_radius_penalty_weight is not None:
        affinity_kwargs["radius_penalty_weight"] = args.affinity_radius_penalty_weight

    if args.reward_tier == "r_free":
        if not args.fold_surrogate:
            raise ValueError("--fold-surrogate is required for --reward-tier r_free")
        rfree_weights = {
            "affinity": float(args.rfree_affinity_weight),
            "fold_confidence": float(args.rfree_fold_weight),
            "physical": float(args.rfree_physical_weight),
            "developability": float(args.rfree_developability_weight),
        }
        if any(value < 0.0 for value in rfree_weights.values()) or not np.isclose(
            sum(rfree_weights.values()), 1.0, atol=1e-6
        ):
            raise ValueError(f"R_free weights must be nonnegative and sum to one; got {rfree_weights}")
        selection_reward_fn = get_reward_fn(
            "r_free",
            fold_surrogate_path=args.fold_surrogate,
            affinity_kwargs=affinity_kwargs,
            sequence_teacher=sequence_teacher,
            sequence_teacher_weight=args.affinity_teacher_weight if sequence_teacher is not None else 0.0,
            sequence_teacher_fusion=args.affinity_teacher_fusion,
            rfree_weights=rfree_weights,
        )
    else:
        selection_reward_fn = None
    if selection_reward_fn is not None:
        # _decode_sequence_teacher_samples is also the shared frozen-AE path
        # for R_free's developability term, even when the optional affinity
        # teacher weight is zero.
        sampler.reward_fn = selection_reward_fn

    def apply_condition_policy(batch: Dict[str, torch.Tensor]) -> None:
        """Remove interface labels for the deployable all-antigen diagnostic."""
        if args.condition_policy == "native_epitope":
            return
        full_mask = batch["full_mask"].bool()
        chain_type = batch["chain_type"]
        antigen_mask = full_mask & (chain_type == 3)
        # The model never receives the native interface labels in this mode.
        batch["epitope_mask"] = antigen_mask
        batch["paratope_mask"] = torch.zeros_like(batch["paratope_mask"])
        # collate_fn centers on the native epitope.  Recenter using only the
        # exposed antigen coordinates so the diagnostic does not retain an
        # interface-centering oracle.
        ca = batch["coords_nm"][:, :, 1, :]
        count = antigen_mask.sum(dim=1, keepdim=True).clamp_min(1).to(ca.dtype)
        centroid = (ca * antigen_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / count.unsqueeze(-1)
        batch["coords_nm"] = (
            batch["coords_nm"] - centroid.unsqueeze(2)
        ) * batch["coord_mask"].unsqueeze(-1)

    rows = []
    selected = 0
    requested_indices = None
    if args.indices_file:
        with open(args.indices_file, "r", encoding="utf-8") as handle:
            requested_indices = {int(value) for value in json.load(handle)}
        if not requested_indices:
            raise ValueError("--indices-file must contain at least one dataset index")
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if requested_indices is not None and index not in requested_indices:
                continue
            # The processed split contains both VH-VL and VHH examples.  The
            # conventional-antibody phase must not silently mix the two; the
            # filter is applied before any GPU rollout and keeps the original
            # dataset index so paired artifacts can be joined exactly.
            if args.conventional_only:
                chain_type = batch.get("chain_type")
                if chain_type is None or not bool((chain_type == 2).any()):
                    continue
            if args.vhh_only:
                chain_type = batch.get("chain_type")
                if chain_type is None or bool((chain_type == 2).any()):
                    continue
            if requested_indices is None and selected >= args.limit:
                break
            batch = _batch_to_device(batch, device)
            apply_condition_policy(batch)
            samples = sampler._generate_samples(batch)
            if selection_reward_fn is not None:
                sampler._decode_sequence_teacher_samples(samples, batch)
            elif sequence_teacher is not None:
                z = torch.cat([sample["local_latents"] for sample in samples], dim=0)
                ca = torch.cat([sample["bb_ca"] for sample in samples], dim=0)
                decoded = model.autoencoder.decode(
                    z_latent=z,
                    ca_coors_nm=ca,
                    mask=batch["mask"].bool().repeat(len(samples), 1),
                )
                residue_type = decoded["residue_type"].long()
                batch_size = batch["mask"].shape[0]
                for sample_index, sample in enumerate(samples):
                    sample["residue_type"] = residue_type[
                        sample_index * batch_size : (sample_index + 1) * batch_size
                    ]
            sample_rows = []
            for sample in samples:
                teacher_score = None
                if selection_reward_fn is not None:
                    reward, terms = selection_reward_fn.score_with_terms(sample, batch)
                else:
                    reward, terms = ca_dockq_affinity_proxy_reward(
                        sample,
                        batch,
                        affinity_weight=1.0,
                        affinity_kwargs=affinity_kwargs,
                        return_terms=True,
                    )
                    if sequence_teacher is not None:
                        teacher_score = sequence_teacher.score_torch(sample["residue_type"], batch).to(reward.dtype)
                        reward = fuse_affinity_teacher_reward(
                            reward,
                            teacher_score,
                            args.affinity_teacher_weight,
                            fusion=args.affinity_teacher_fusion,
                        )
                if selection_reward_fn is not None:
                    teacher_score = terms.get("sequence_affinity_teacher")
                row = {
                    "reward": float(reward.mean().item()),
                    "contact_coverage": float(terms["contact_coverage"].mean().item()),
                    "buried_surface_proxy": float(terms["buried_surface_proxy"].mean().item()),
                    "interface_compactness": float(terms["interface_compactness"].mean().item()),
                    "steric_gate": float(terms["steric_gate"].mean().item()),
                    "antibody_compactness_prior": float(terms["antibody_compactness_prior"].mean().item()),
                    "radius_gyration": float(terms["radius_gyration"].mean().item()),
                    "radius_prior_penalty": float(terms["radius_prior_penalty"].mean().item()),
                }
                if selection_reward_fn is not None:
                    for name in (
                        "affinity_component", "fold_confidence", "physical",
                        "developability", "soft_clash_fraction",
                        "backbone_continuity", "backbone_bond_deviation_nm",
                        "cys_fraction", "met_trp_fraction", "nxs_t_rate",
                        "hydrophobic_fraction",
                    ):
                        if name in terms:
                            row[name] = float(terms[name].mean().item())
                if teacher_score is not None:
                    row["sequence_affinity_teacher"] = float(teacher_score.mean().item())
                if args.diagnostic_dockq:
                    row["diagnostic_dockq"] = float(ca_dockq_proxy_reward(sample, batch).mean().item())
                if args.diagnostic_reference:
                    reference_terms = reference_structure_scores(sample, batch)
                    row.update({
                        f"diagnostic_{name}": float(value.mean().item())
                        for name, value in reference_terms.items()
                    })
                sample_rows.append(row)
            values = [row["reward"] for row in sample_rows]
            rows.append({
                "index": index,
                "mask_strategy": "full",
                "K": args.K,
                "nsteps": args.nsteps,
                "seed": args.seed,
                "one_candidate": values[0],
                "best_of_K": max(values),
                "samples": sample_rows,
            })
            selected += 1
            # Each target can have a very different padded sequence length;
            # release cached blocks between targets so a long tail (e.g. the
            # 800-residue test complexes) cannot turn a complete evaluation
            # into an allocator/OOM failure.
            del samples, sample_rows, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if requested_indices is not None:
        observed = {int(row["index"]) for row in rows}
        missing = sorted(requested_indices - observed)
        if missing:
            raise RuntimeError(
                f"indices-file coverage failure: requested {len(requested_indices)}, "
                f"observed {len(observed)}, missing {missing[:10]}"
            )
    flat = [sample["reward"] for row in rows for sample in row["samples"]]
    result: Dict = {
        "split": args.split,
        "mask_strategy": "full",
        "K": args.K,
        "nsteps": args.nsteps,
        "seed": args.seed,
        "condition_policy": args.condition_policy,
        "limit": len(rows),
        "indices_file": args.indices_file,
        "requested_indices": sorted(requested_indices) if requested_indices is not None else None,
        "conventional_only": bool(args.conventional_only),
        "vhh_only": bool(args.vhh_only),
        "rl_state": args.rl_state,
        "init_nn_state": args.init_nn_state,
        "base_ckpt": args.base_ckpt,
        "autoencoder": args.autoencoder,
        "state_step": state_step,
        "reward_tier": args.reward_tier,
        "diagnostic_reference": bool(args.diagnostic_reference),
        "selection_metric": (
            "R_free"
            if selection_reward_fn is not None
            else (
                f"affinity_proxy_plus_sequence_teacher_{args.affinity_teacher_fusion}"
                if sequence_teacher is not None
                else "affinity_proxy"
            )
        ),
        "affinity_kwargs": affinity_kwargs,
        "affinity_teacher": args.affinity_teacher,
        "affinity_teacher_weight": args.affinity_teacher_weight if sequence_teacher is not None else 0.0,
        "affinity_teacher_fusion": args.affinity_teacher_fusion if sequence_teacher is not None else "none",
        "fold_surrogate": args.fold_surrogate if selection_reward_fn is not None else None,
        "rfree_weights": (
            selection_reward_fn.weights if selection_reward_fn is not None else None
        ),
        "diagnostic_dockq": args.diagnostic_dockq,
        "one_candidate_mean": sum(row["one_candidate"] for row in rows) / max(len(rows), 1),
        "best_of_K_mean": sum(row["best_of_K"] for row in rows) / max(len(rows), 1),
        "sample_reward_mean": sum(flat) / max(len(flat), 1),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({k: result[k] for k in (
        "split", "mask_strategy", "K", "nsteps", "seed", "limit", "reward_tier", "selection_metric",
        "one_candidate_mean", "best_of_K_mean", "sample_reward_mean",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
