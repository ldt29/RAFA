#!/usr/bin/env python3
"""
AE inference on the structure_dataset test split.

Evaluates autoencoder reconstruction quality using ab_metrics (calc_all).
Saves per-sample PDBs, metrics.json, a summary CSV and JSON.

Usage:
    python proteinfoundation/partial_autoencoder/inference.py \
        --config_name inference_ae_ab_design
"""

import os
import sys
import json
import csv
import tempfile
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

root = os.path.abspath(".")
sys.path.insert(0, root)

import hydra
import lightning as L
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb


def _rewrite_pdb_chain_ids(pdb_path: str, chain_map: Dict[str, str]) -> None:
    """Rewrite chain IDs in a PDB file in-place."""
    out_lines = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM", "TER")) and len(line) > 21:
                old_chain = line[21]
                if old_chain in chain_map:
                    line = line[:21] + chain_map[old_chain] + line[22:]
            out_lines.append(line)
    with open(pdb_path, "w") as f:
        f.writelines(out_lines)


def parse_args_and_cfg() -> Tuple[Dict, Dict, str]:
    parser = argparse.ArgumentParser(description="AE inference")
    parser.add_argument("--config_name", type=str, default="inference_ae_ab_design")
    parser.add_argument("--config_number", type=int, default=-1)
    args = parser.parse_args()

    config_path = "../../configs"
    with hydra.initialize(config_path, version_base=hydra.__version__):
        config_name = args.config_name if args.config_number == -1 else f"inf_{args.config_number}"
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")
    return args, cfg, config_name


def setup(cfg, config_name):
    assert torch.cuda.is_available(), "CUDA not available"
    root_path = f"./inference/{config_name}"
    os.makedirs(root_path, exist_ok=True)
    L.seed_everything(cfg.seed)
    return root_path


def load_ae_dataloader(cfg):
    """Create dataloader from structure_dataset test split."""
    from proteinfoundation.datasets.ab_data import AntibodyDesignDataModule

    dm = AntibodyDesignDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.get("bs", 4),
        num_workers=4,
        max_antibody_len=cfg.dataset.get("max_antibody_len", 450),
        max_antigen_len=cfg.dataset.get("max_antigen_len", 500),
        train_split=cfg.dataset.get("split", "test"),
    )
    dm.setup("test")
    dl = dm.test_dataloader()
    logger.info(f"Test dataloader: {len(dl)} batches")
    return dl, dm


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    return x


def evaluate_ae_reconstruction(root_path, ckpt_file, cfg):
    """
    Run AE predict on the test split, save reconstructed PDBs, and evaluate
    with ab_metrics.
    """
    from proteinfoundation.metrics.ab_metrics import calc_all, parse_cdr_ranges
    from proteinfoundation.datasets.ab_data import AntibodyDesignDataset, collate_fn

    data_dir = cfg.dataset.data_dir
    split = cfg.dataset.get("split", "test")

    # Load AE
    model = AutoEncoder.load_from_checkpoint(ckpt_file, strict=False)
    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = AntibodyDesignDataset(
        data_dir=data_dir,
        split=split,
        max_antibody_len=cfg.dataset.get("max_antibody_len", 450),
        max_antigen_len=cfg.dataset.get("max_antigen_len", 500),
    )
    dataloader = DataLoader(dataset, batch_size=cfg.get("bs", 4),
                            shuffle=False, collate_fn=collate_fn)

    run_tmscore = cfg.get("metrics", {}).get("tmscore", True)
    run_lddt = cfg.get("metrics", {}).get("lddt", True)
    run_dockq = cfg.get("metrics", {}).get("dockq", False)

    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = _to_device(batch, device)

            # AE forward: encode → decode
            mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]
            batch["mask"] = mask
            ca_coors_nm = batch["coords_nm"][..., 1, :] * mask[..., None]

            # Trim to antibody-only (same as training_step)
            chain_type = batch.get("chain_type", None)
            if chain_type is not None:
                ab_only = (chain_type > 0) & (chain_type < 3)
                max_ab_len = ab_only.sum(dim=1).max().item()
                orig_n = mask.shape[1]
                if max_ab_len > 0 and max_ab_len < orig_n:
                    trim_keys = ["coords_nm", "coords", "coord_mask", "residue_type", "seq",
                                 "chain_breaks_per_residue", "chains", "chain_type"]
                    for k in trim_keys:
                        if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].shape[1] == orig_n:
                            batch[k] = batch[k][:, :max_ab_len]
                    for k in ["residue_type", "coords"]:
                        if k in batch["mask_dict"] and batch["mask_dict"][k].shape[1] == orig_n:
                            batch["mask_dict"][k] = batch["mask_dict"][k][:, :max_ab_len]
                    mask = mask[:, :max_ab_len]
                    batch["mask"] = mask
                    ca_coors_nm = ca_coors_nm[:, :max_ab_len]

            output_enc = model.encoder(batch)
            input_decoder = {
                "z_latent": output_enc["z_latent"],
                "ca_coors_nm": ca_coors_nm,
                "residue_mask": mask,
                "mask": mask,
            }
            output_dec = model.decoder(input_decoder)

            bs = mask.shape[0]
            for b in range(bs):
                n_ab = int(mask[b].sum().item())
                if n_ab == 0:
                    continue

                # True coordinates (Angstrom)
                true_nm = batch["coords_nm"][b, :n_ab]  # [n_ab, 37, 3]
                true_ang = (true_nm * 10.0).cpu().numpy()
                true_seq = batch["residue_type"][b, :n_ab].cpu().numpy()
                true_atom_mask = batch["coord_mask"][b, :n_ab].cpu().numpy()

                # Predicted coordinates (Angstrom)
                pred_nm = output_dec["coors_nm"][b, :n_ab]  # [n_ab, 37, 3]
                pred_ang = (pred_nm * 10.0).cpu().numpy()
                pred_seq = output_dec["aatype_max"][b, :n_ab].cpu().numpy()
                pred_atom_mask = output_dec["atom_mask"][b, :n_ab].cpu().numpy()

                # Get sample info
                sample_idx = batch_idx * cfg.get("bs", 4) + b
                if sample_idx >= len(dataset):
                    break
                item = dataset.data[sample_idx % len(dataset.data)]
                pdb_id = item.get("pdb_id", f"sample_{sample_idx}")
                h_len = item["h_len"]
                ab_type = "VH-VL" if (n_ab - h_len) > 0 else "VHH"
                ab_chains = ["H", "L"] if ab_type == "VH-VL" else ["H"]

                # Build chain_index: 0=heavy, 1=light (if exists)
                chain_index = np.zeros(n_ab, dtype=np.int64)
                if ab_type == "VH-VL" and n_ab > h_len:
                    chain_index[h_len:] = 1

                # Create sample directory
                safe_name = pdb_id.replace("/", "_")
                sample_dir = os.path.join(root_path, safe_name)
                os.makedirs(sample_dir, exist_ok=True)

                # Save true PDB (H/L chains)
                true_path = os.path.join(sample_dir, f"{safe_name}_true.pdb")
                write_prot_to_pdb(
                    prot_pos=true_ang * true_atom_mask[..., None],
                    aatype=true_seq * mask[b, :n_ab].cpu().numpy().astype(np.int64),
                    file_path=true_path,
                    chain_index=chain_index,
                    overwrite=True, no_indexing=True,
                )
                _rewrite_pdb_chain_ids(true_path, {"A": "H", "B": "L"})

                # Save reconstructed PDB (H/L chains)
                pred_path = os.path.join(sample_dir, f"{safe_name}_pred.pdb")
                write_prot_to_pdb(
                    prot_pos=pred_ang * pred_atom_mask[..., None],
                    aatype=pred_seq * mask[b, :n_ab].cpu().numpy().astype(np.int64),
                    file_path=pred_path,
                    chain_index=chain_index,
                    overwrite=True, no_indexing=True,
                )
                _rewrite_pdb_chain_ids(pred_path, {"A": "H", "B": "L"})

                # Parse CDR ranges from design.fasta
                cdr_ranges = None
                for sub in ["before_20250630", "after_20250630_novel_ab", "after_20250630_novel_nb"]:
                    fasta_path = Path(data_dir) / sub / pdb_id / "design.fasta"
                    if fasta_path.exists():
                        cdr_ranges = parse_cdr_ranges(fasta_path)
                        break

                # Compute metrics (no antigen chain — AE only reconstructs antibody)
                try:
                    metrics = calc_all(
                        pred_path, true_path,
                        ab_chains=ab_chains,
                        ag_chain="L" if ab_type == "VH-VL" else "H",  # dummy: not used for ab-only eval
                        cdr_ranges=cdr_ranges,
                        run_tmscore=run_tmscore,
                        run_lddt=run_lddt,
                        run_dockq=False,  # no antigen, DockQ not applicable
                    )
                    # Remove DockQ metrics (not applicable for ab-only)
                    metrics = {k: v for k, v in metrics.items() if not k.startswith("DockQ")}
                    metrics['pdb_id'] = pdb_id
                    metrics['ab_type'] = ab_type

                    with open(os.path.join(sample_dir, 'metrics.json'), 'w') as f:
                        json.dump(metrics, f, indent=2)

                    all_metrics.append(metrics)
                    logger.info(
                        f"[{sample_idx+1}/{len(dataset)}] {pdb_id}: "
                        f"RMSD_ab_ag={metrics.get('RMSD_ab_ag', float('nan')):.3f} "
                        f"RMSD_ab_ab={metrics.get('RMSD_ab_ab', float('nan')):.3f} "
                        f"AAR_ab={metrics.get('AAR_ab', float('nan')):.3f}"
                    )
                except Exception as e:
                    logger.warning(f"Metrics failed for {pdb_id}: {e}")

    # Summary
    if not all_metrics:
        logger.warning("No metrics computed")
        return

    csv_path = os.path.join(root_path, 'metrics_summary.csv')
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    fieldnames = ['pdb_id', 'ab_type'] + sorted(k for k in all_keys if k not in ('pdb_id', 'ab_type'))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    summary = {}
    numeric_keys = [k for k in all_keys
                    if k not in ('pdb_id', 'ab_type')
                    and all(isinstance(m.get(k), (int, float)) for m in all_metrics)]
    for k in sorted(numeric_keys):
        vals = [m[k] for m in all_metrics if m.get(k) is not None and not np.isnan(m[k])]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    logger.info("=" * 60)
    logger.info("AE Reconstruction Metrics Summary")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k:30s}: {v['mean']:.4f} ± {v['std']:.4f}")
    logger.info("=" * 60)

    summary_json_path = os.path.join(root_path, 'metrics_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"CSV: {csv_path}")
    logger.info(f"JSON: {summary_json_path}")


def main():
    load_dotenv()
    args, cfg, config_name = parse_args_and_cfg()
    root_path = setup(cfg, config_name)

    ckpt_file = cfg.get("ckpt_file", None)
    if ckpt_file is None:
        logger.error("No ckpt_file specified in config")
        return
    if not os.path.exists(ckpt_file):
        logger.error(f"Checkpoint not found: {ckpt_file}")
        return

    logger.info(f"Evaluating AE: {ckpt_file}")
    evaluate_ae_reconstruction(root_path, ckpt_file, cfg)


if __name__ == "__main__":
    import argparse
    main()
