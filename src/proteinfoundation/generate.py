import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import numpy as np

root = Path(".").resolve()
source_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source_root))  # Adds the release src directory
sys.path.insert(0, str(root))  # Also supports local auxiliary imports
# isort: split

import argparse
import csv
import json

import hydra
import lightning as L
import torch
from dotenv import load_dotenv
from loguru import logger
from torch.utils.data import DataLoader

from proteinfoundation.datasets.gen_dataset import GenDataset
from proteinfoundation.proteina import Proteina
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
    """
    Parses command line arguments and loads the corresponding config file.

    Returns:
        Command line arguments (dict)
        Config file (dict)
        config_name (string)
    """
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_base",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job id for this config to determine which split to use.",
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Name of the data path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Optional absolute output directory for generated structures.",
    )
    args, hydra_overrides = parser.parse_known_args()
    unknown_options = [item for item in hydra_overrides if item.startswith("-")]
    if unknown_options:
        parser.error(
            "unrecognized option(s): " + ", ".join(unknown_options)
        )
    if args.data_path is not None:
        os.environ["DATA_PATH"] = args.data_path
    if args.output_dir is not None:
        os.environ["PODA_OUTPUT_DIR"] = args.output_dir
    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    if args.config_subdir is None:
        config_path = "../../configs"
    else:
        config_path = f"../../configs/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name, overrides=hydra_overrides)
        logger.info(f"Inference config {cfg}")

    return args, cfg, config_name


def setup(
    cfg: Dict, create_root: bool = True, config_name: str = ".", job_id: int = 0
) -> str:
    """
    Checks if metrics being computed are compatible, sets the right seed, and creates the root directory
    where the run will store things.

    Returns:
        Path of the root directory (string)
    """
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout

    assert (
        not (
            cfg.generation.metric.compute_designability
            or cfg.generation.metric.compute_novelty_pdb
            or cfg.generation.metric.compute_novelty_afdb
        )
        or not cfg.generation.metric.compute_fid
    ), "Designability/Novelty cannot be computed together with FID"

    # Set root path for this inference run.  The release wrapper supplies an
    # explicit directory so independent runs can coexist without editing Hydra
    # configs; the historical config-name layout remains the fallback.
    configured_output = os.environ.get("PODA_OUTPUT_DIR")
    if configured_output:
        root_path = str(Path(configured_output).expanduser().resolve())
    elif "motif_task_name" in cfg.generation.dataset:
        root_path = (
            f"./inference/{config_name}_{cfg.generation.dataset.motif_task_name}"
        )
    else:
        root_path = f"./inference/{config_name}"
    if create_root:
        os.makedirs(root_path, exist_ok=True)
    else:
        if not os.path.exists(root_path):
            raise ValueError("Results path %s does not exist" % root_path)

    # Set seed
    cfg.seed = cfg.seed + job_id  # Different seeds for different splits ids
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    return root_path


def check_cfg_validity(cfg_data: Dict, cfg_sample_args: Dict) -> None:
    """
    Checks if guidance arguments (CFG and AG) are valid.
    """
    # Logging CFG
    if cfg_sample_args.guidance_w != 1.0:
        logger.info(
            f"Guidance is turned on with guidance weight {cfg_sample_args.guidance_w} and autoguidance ratio {cfg_sample_args.ag_ratio}."
        )
        assert (
            cfg_sample_args.ag_ratio >= 0.0 and cfg_sample_args.ag_ratio <= 1.0
        ), f"Autoguidance ratio should be between 0 and 1, but now is {cfg_sample_args.ag_ratio}."
        assert (cfg_sample_args.ag_ratio == 0.0) or (
            cfg_sample_args.ag_ckpt_path is not None
        ), f"Autoguidance checkpoint path should be provided"
    else:
        logger.info(f"Guidance is turned off.")

    # Logging conditional generation
    if cfg_sample_args.fold_cond:
        logger.info("Conditional generation is turned on.")
        assert (
            cfg_data.empirical_distribution_cfg.len_cath_code_path is not None
        ), "Empirical (len, cath_code) distribution file should be provided when using conditional generation."
    else:
        logger.info("Conditional generation is turned off.")
        assert (
            cfg_data.empirical_distribution_cfg.len_cath_code_path is None
        ), "Empirical (len, cath_code) distribution file shouldn't be provided when using unconditional generation."


def load_ag_ckpt(cfg: Dict) -> Union[None, torch.nn.Module]:
    """
    Loads the neural network for the "bad" checkpoint in autoguidance, if requested.

    Returns:
        A nn module, if autogudance enabled.
    """
    nn_ag = None
    if cfg.ag_ratio > 0 and cfg.guidance_w != 1.0:
        logger.info(
            f"Using autoguidance with guidance weight {cfg.guidance_w} and autoguidance ratio {cfg.ag_ratio} based on the checkpoint {cfg.ag_ckpt_path}"
        )
        ckpt_ag_file = cfg.ag_ckpt_path
        assert os.path.exists(ckpt_ag_file), f"Not a valid checkpoint {ckpt_ag_file}"
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file, strict=False)

        # OPTIMIZATION: Remove encoder from autoguidance model autoencoder during generation (only decoder needed)
        if model_ag.autoencoder is not None:
            logger.info(
                "Removing autoencoder encoder from autoguidance model during generation to save memory"
            )
            del model_ag.autoencoder.encoder
            model_ag.autoencoder.encoder = None

        nn_ag = model_ag.nn
    return nn_ag


def load_ckpt_n_configure_inference(cfg: Dict) -> Proteina:
    """
    Loads the model, potentially the autoguidance checkpoint as well, if requested.

    Returns:
        Model (Proteina)
    """
    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"

    model = Proteina.load_from_checkpoint(ckpt_file, strict=False, autoencoder_ckpt_path=cfg.get("autoencoder_ckpt_path", None))

    # Set inference variables and potentially load autoguidance
    nn_ag = load_ag_ckpt(cfg.generation.args)

    model.configure_inference(cfg.generation, nn_ag=nn_ag)

    return model


def split_by_job(cfg: Dict, job_id: int, njobs: int) -> Dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataset.nsamples
    nsamples_per_split = (nsamples - 1) // njobs + 1
    if nsamples_per_split * job_id >= nsamples:
        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
        exit(0)
    else:
        cfg.dataset.nsamples = min(
            nsamples_per_split, nsamples - nsamples_per_split * job_id
        )
    return cfg


def binder_split_by_job(cfg: Dict, job_id: int, njobs: int) -> Dict:
    """
    Since generation may be split across multiple jobs, this function determines how many samples are produced per job.
    Then, it sets the right value in the config dict, and returns the updated config.

    Returns:
        Config updated with the correct number of samples to generate.
    """
    nsamples = cfg.dataset.nlens_cfg.random_lens[2]
    nsamples_per_split = (nsamples - 1) // njobs + 1
    if nsamples_per_split * job_id >= nsamples:
        logger.info(f"Job id {job_id} get 0 samples. Finishing job...")
        exit(0)
    else:
        cfg.dataset.nlens_cfg.random_lens[2] = min(
            nsamples_per_split, nsamples - nsamples_per_split * job_id
        )
    return cfg


def _to_device(x, device):
    """Recursively move tensors to device, leaving non-tensors unchanged."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_device(v, device) for v in x)
    return x


def save_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.tensor]]],
    job_id: int = 0,
    chain_indexes: np.ndarray = None,
    cath_codes: List[List[List[str]]] = None,
    ref_data: List[Dict] = None,
    pdb_names: List[str] = None,
) -> None:
    """
    Saves generated samples and reference structures with unified format.
    
    Both generated and reference structures are saved with the same chain format
    (H, L, A chains for antibody-antigen complexes) and epitope center alignment.

    Args:
        root_path: root directory where samples will be stored (within subdirectories)
        predictions: List of lists of tuples. Each tuple represents a sample
            (coors [n, 37, 3], aatype [n]) or (coors [n, 37, 3], aatype [n], chain_index [n])
        job_id: job number, used to store files
        chain_indexes: chain indexes for each sample, used to store files
        cath_codes: conditional sampling metadata
        ref_data: List of reference data dicts with 'coords', 'residue_type', and 'chain_index' keys
        pdb_names: Optional list of PDB names to use as directory names instead of job/length/id scheme
    """
    predictions = [sample for sublist in predictions for sample in sublist]
    # List[tuple] where each tuple is (coors [n, 37, 3], aatype [n]) or
    # (coors [n, 37, 3], aatype [n], chain_index [n]) for ab_design complex

    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions):
        if len(pred) == 3:
            coors_atom37, residue_type, chain_idx = pred
            chain_index = chain_idx.detach().cpu().numpy()
        else:
            coors_atom37, residue_type = pred
            chain_index = chain_indexes[j].numpy() if chain_indexes else None

        n = coors_atom37.shape[-3]

        # Create directory where everything related to this sample will be stored
        suffix = ""
        if pdb_names is not None and j < len(pdb_names):
            pdb_name = os.path.splitext(os.path.basename(pdb_names[j]))[0]
            dir_name = f"{pdb_name}_job_{job_id}_id_{samples_per_length[n]}{suffix}"
        else:
            dir_name = f"job_{job_id}_n_{n}_id_{samples_per_length[n]}{suffix}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(
            root_path, dir_name
        )
        os.makedirs(sample_root_path, exist_ok=False)

        # Save generated structure as pdb with unified format (H, L, A chains)
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            chain_index=chain_index,
            overwrite=True,
            no_indexing=True,
        )
        # write_prot_to_pdb uses A/B/C for chain_index 0/1/2; remap to H/L/A for antibody design
        if chain_index is not None:
            _rewrite_pdb_chain_ids(pdb_path, {"A": "H", "B": "L", "C": "A"})
        logger.info(f"✓ Saved generated PDB: {pdb_path}")
        
        # Save reference structure as pdb with unified format (if available)
        if ref_data is not None and j < len(ref_data):
            ref_info = ref_data[j]
            if 'coords' in ref_info and 'residue_type' in ref_info:
                ref_coords = ref_info['coords']
                ref_residue_type = ref_info['residue_type']
                # Use the same chain_index format as generated structure for consistency
                ref_chain_index = ref_info.get('chain_index', chain_index)
                
                # Convert to numpy if needed
                if isinstance(ref_coords, torch.Tensor):
                    ref_coords = ref_coords.float().detach().cpu().numpy()
                if isinstance(ref_residue_type, torch.Tensor):
                    ref_residue_type = ref_residue_type.detach().cpu().numpy()
                if isinstance(ref_chain_index, torch.Tensor):
                    ref_chain_index = ref_chain_index.detach().cpu().numpy()
                
                # Handle batch dimension if present
                if ref_coords.ndim == 4:  # [batch, n, 37, 3]
                    ref_coords = ref_coords[0]
                if ref_residue_type.ndim == 2:  # [batch, n]
                    ref_residue_type = ref_residue_type[0]
                if ref_chain_index is not None and ref_chain_index.ndim == 2:  # [batch, n]
                    ref_chain_index = ref_chain_index[0]
                
                # Ensure residue_type is integer type for proper indexing
                if ref_residue_type.dtype != np.int64 and ref_residue_type.dtype != np.int32:
                    ref_residue_type = ref_residue_type.astype(np.int64)
                
                # Save ref PDB with unified format (H, L, A chains) and _ref suffix
                ref_fname = dir_name + "_ref.pdb"
                ref_pdb_path = os.path.join(sample_root_path, ref_fname)
                try:
                    write_prot_to_pdb(
                        prot_pos=ref_coords,
                        aatype=ref_residue_type,
                        file_path=ref_pdb_path,
                        chain_index=ref_chain_index,  # Use same chain format as generated
                        overwrite=True,
                        no_indexing=True,
                    )
                    if ref_chain_index is not None:
                        _rewrite_pdb_chain_ids(ref_pdb_path, {"A": "H", "B": "L", "C": "A"})
                    logger.info(f"✓ Saved reference PDB: {ref_pdb_path}")
                except Exception as e:
                    logger.warning(f"Failed to save reference PDB: {e}")


def evaluate_ab_design(root_path, dataset, job_id=0, cfg=None):
    """
    Evaluate all generated antibody samples in root_path using ab_metrics.calc_all.

    Uses calc_all() for: RMSD, per-CDR RMSD, AAR (per-chain / per-CDR / aggregate),
    TMscore, LDDT, DockQ.
    Uses parse_cdr_ranges() to extract CDR boundaries from the design.fasta.

    Saves per-sample metrics.json, a summary CSV, and prints mean/std statistics.
    """
    from proteinfoundation.metrics.ab_metrics import calc_all, parse_cdr_ranges

    # Determine which metrics to compute
    run_tmscore = True
    run_lddt = True
    run_dockq_flag = False
    if cfg is not None:
        run_tmscore = cfg.get("ab_eval_tmscore", True)
        run_lddt = cfg.get("ab_eval_lddt", True)
        run_dockq_flag = cfg.get("ab_eval_dockq", False)

    # Build a mapping from safe_pdb_id -> (pdb_id, item) for directory name matching.
    # dataset.data items use keys: pdb_id, h_len, ab_seq (=ab_coords shape[0]), ab_coords, ag_coords, etc.
    # NOTE: ab_seq in the raw data is an int array [n_ab], NOT a string — use ab_coords.shape[0] for n_ab.
    pdb_to_item = {}
    for item in dataset.data:
        pdb_id = item.get('pdb_id')
        if pdb_id:
            safe_pid = pdb_id.replace('/', '_')
            pdb_to_item[safe_pid] = (pdb_id, item)

    # Find sample directories that belong to this job_id
    sample_dirs = []
    if os.path.exists(root_path):
        for d in sorted(os.listdir(root_path)):
            full = os.path.join(root_path, d)
            if not os.path.isdir(full):
                continue
            # Must contain the job_id marker produced by save_predictions:
            #   "{safe_pdb_id}_job_{job_id}_id_{sample_idx}"
            if f"_job_{job_id}_" in d:
                sample_dirs.append(d)

    if not sample_dirs:
        logger.warning(f"No sample directories found for job_{job_id} in {root_path}")
        return

    all_metrics = []
    processed_count = 0

    for dir_name in sample_dirs:
        sample_dir = os.path.join(root_path, dir_name)
        gen_pdb = os.path.join(sample_dir, dir_name + ".pdb")
        ref_pdb = os.path.join(sample_dir, dir_name + "_ref.pdb")

        if not os.path.exists(gen_pdb):
            logger.warning(f"Missing generated PDB: {gen_pdb}, skipping")
            continue

        if not os.path.exists(ref_pdb):
            logger.warning(f"Missing reference PDB: {ref_pdb}, skipping")
            continue

        # Recover pdb_id: directory name is "{safe_pdb_id}_job_{job_id}_id_{n}"
        # Strip the trailing "_job_{job_id}_id_*" suffix.
        suffix_marker = f"_job_{job_id}_"
        safe_pid = dir_name[:dir_name.index(suffix_marker)]

        if safe_pid not in pdb_to_item:
            logger.warning(f"pdb_id '{safe_pid}' not found in dataset for dir {dir_name}, skipping")
            continue
        if 'nb' in safe_pid:
            logger.warning(f"Skipping nanobody {safe_pid} for now since metrics may be unreliable")
            continue

        pdb_id, item = pdb_to_item[safe_pid]

        # Derive n_ab and h_len from the raw item arrays (ab_seq is an int array, not a string).
        h_len = int(item.get('h_len', 0))
        ab_coords = item.get('ab_coords')  # [n_ab, 37, 3]
        n_ab = ab_coords.shape[0] if ab_coords is not None else 0

        if n_ab == 0:
            logger.warning(f"Empty antibody for {pdb_id}, skipping")
            continue

        l_len = n_ab - h_len
        ab_type = "VH-VL" if l_len > 0 else "VHH"
        ab_chains = ["H", "L"] if ab_type == "VH-VL" else ["H"]

        # Parse CDR ranges from design.fasta (if available).
        # The fasta lives under data_dir/{sub}/{pdb_id}/design.fasta.
        cdr_ranges = None
        data_dir = None
        if cfg is not None:
            # cfg may be the top-level OmegaConf or a plain dict
            try:
                data_dir = cfg.generation.dataset.data_dir
            except Exception:
                try:
                    data_dir = cfg.get("generation", {}).get("dataset", {}).get("data_dir")
                except Exception:
                    pass
        if data_dir:
            for sub in ["before_20250630", "after_20250630_novel_ab", "after_20250630_novel_nb"]:
                fasta_path = Path(data_dir) / sub / pdb_id / "design.fasta"
                if fasta_path.exists():
                    try:
                        cdr_ranges = parse_cdr_ranges(fasta_path)
                    except Exception as e:
                        logger.warning(f"Failed to parse CDR ranges from {fasta_path}: {e}")
                    break

        try:
            metrics = calc_all(
                gen_pdb, ref_pdb,
                ab_chains=ab_chains,
                ag_chain="A",
                cdr_ranges=cdr_ranges,
                run_tmscore=run_tmscore,
                run_lddt=run_lddt,
                run_dockq=run_dockq_flag,
            )
            metrics['pdb_id'] = pdb_id
            metrics['sample_dir'] = dir_name
            metrics['ab_type'] = ab_type

            # Save per-sample metrics
            with open(os.path.join(sample_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)

            all_metrics.append(metrics)
            processed_count += 1

            # Log key metrics
            nan = float('nan')
            logger.info(
                f"[{processed_count}/{len(sample_dirs)}] {pdb_id}: "
                f"AAR_ab={metrics.get('AAR_ab', nan):.3f} "
                f"AAR_CDRH3={metrics.get('AAR_CDRH3', nan):.3f} "
                f"RMSD_ab_ag={metrics.get('RMSD_ab_ag', nan):.3f} "
                f"RMSD_ab_ab={metrics.get('RMSD_ab_ab', nan):.3f} "
                f"TM={metrics.get('TMscore', nan):.3f}"
            )
        except Exception as e:
            logger.warning(f"Failed to compute metrics for {dir_name}: {e}")
            import traceback
            traceback.print_exc()

    if not all_metrics:
        logger.warning("All metrics calculations failed or no matches found.")
        return

    # Summary CSV
    csv_path = os.path.join(root_path, 'metrics_summary.csv')
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())
    fieldnames = ['pdb_id', 'sample_dir', 'ab_type'] + sorted(
        k for k in all_keys if k not in ('pdb_id', 'sample_dir', 'ab_type'))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in all_metrics:
            writer.writerow(m)

    # Aggregate summary: mean/std over numeric fields
    summary = {}
    numeric_keys = [k for k in all_keys
                    if k not in ('pdb_id', 'sample_dir', 'ab_type')
                    and all(isinstance(m.get(k), (int, float)) for m in all_metrics)]
    for k in sorted(numeric_keys):
        vals = [m[k] for m in all_metrics if m.get(k) is not None and not np.isnan(m[k])]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "min": float(np.min(vals)), "max": float(np.max(vals))}

    logger.info("=" * 60)
    logger.info("Metrics Summary")
    logger.info("=" * 60)
    for k, v in summary.items():
        logger.info(f"  {k:30s}: {v['mean']:.4f} ± {v['std']:.4f}")
    logger.info("=" * 60)

    summary_json_path = os.path.join(root_path, 'metrics_summary.json')
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Per-sample CSV: {csv_path}")
    logger.info(f"Summary JSON:   {summary_json_path}")


def _parse_complex_pdb(pdb_path: str):
    """
    Parse a complex PDB file (H+L+Ag chains) into atom37 arrays.

    Our pipeline writes PDBs with chain_index 0->A (heavy), 1->B (light), 2->C (antigen).

    Returns:
        coords:      [n, 37, 3] float32
        seq:         [n] int64 (0-19)
        chain_index: [n] int64 (0=heavy, 1=light, 2=antigen)
    """
    from openfold.np.residue_constants import atom_order, restype_3to1, restypes

    # Build reverse mapping: 1-letter -> index
    res1_to_idx = {r: i for i, r in enumerate(restypes)}

    # Parse PDB
    residues = {}  # (chain_letter, resnum) -> {'atoms': {atom_idx: (x,y,z)}, 'resname': str}
    chain_order = []  # track chain appearance order

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            aname = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])

            key = (chain, resnum)
            if key not in residues:
                residues[key] = {'atoms': {}, 'resname': resname}
                chain_order.append(key)

            atom_idx = atom_order.get(aname, -1)
            if atom_idx >= 0:
                residues[key]['atoms'][atom_idx] = (x, y, z)

    # Sort by appearance order
    n = len(chain_order)
    coords = np.zeros((n, 37, 3), dtype=np.float32)
    seq = np.zeros(n, dtype=np.int64)
    chain_index = np.zeros(n, dtype=np.int64)

    # Map chain letters to indices: A->0, B->1, C->2
    chain_letter_to_idx = {}
    idx_counter = 0
    for key in chain_order:
        ch = key[0]
        if ch not in chain_letter_to_idx:
            chain_letter_to_idx[ch] = idx_counter
            idx_counter += 1

    for i, key in enumerate(chain_order):
        info = residues[key]
        for atom_idx, xyz in info['atoms'].items():
            coords[i, atom_idx] = xyz

        # Convert 3-letter to 1-letter to index
        resname3 = info['resname']
        res1 = restype_3to1.get(resname3, 'X')
        seq[i] = res1_to_idx.get(res1, 0)

        chain_index[i] = chain_letter_to_idx[key[0]]

    return coords, seq, chain_index


def save_motif_predictions(
    root_path: str,
    predictions: List[List[Tuple[torch.tensor]]],
    job_id: int = 0,
    motif_pdb_name: str = None,
) -> None:
    predictions = [sample for sublist in predictions for sample in sublist]
    print([(p[0].shape, p[1].shape) for p in predictions])
    samples_per_length = defaultdict(int)
    for j, pred in enumerate(predictions):
        coors_atom37, residue_type = pred  # [n, 37, 3] and [n]
        n = coors_atom37.shape[-3]
        dir_name = f"job_{job_id}_id_{j}_motif_{motif_pdb_name}"
        samples_per_length[n] += 1
        sample_root_path = os.path.join(root_path, dir_name)
        os.makedirs(sample_root_path, exist_ok=False)
        fname = dir_name + ".pdb"
        pdb_path = os.path.join(sample_root_path, fname)
        write_prot_to_pdb(
            prot_pos=coors_atom37.float().detach().cpu().numpy(),
            aatype=residue_type.detach().cpu().numpy(),
            file_path=pdb_path,
            overwrite=True,
            no_indexing=True,
        )


def main():
    load_dotenv()

    # Parse arguments, load appropriate config, and set up root path
    args, cfg, config_name = parse_args_and_cfg()

    # ── Multi-GPU distributed inference ───────────────────────────────────────
    # Launched via:  torchrun --nproc_per_node=N proteinfoundation/generate.py ...
    # Falls back to single-GPU silently when not launched with torchrun.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist    = world_size > 1

    if is_dist:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        if local_rank == 0:
            logger.info(f"[dist] Distributed inference: {world_size} GPUs")

    # Combine CLI --job_id (for cluster gen_njobs) with local GPU rank so each
    # (job, rank) pair writes to a unique directory suffix.
    effective_job_id = args.job_id * world_size + local_rank
    # ──────────────────────────────────────────────────────────────────────────

    motif_cond = cfg.generation.args.get("motif_cond", False)
    target_cond = cfg.generation.args.get("target_cond", False)
    cfg.generation.args.get("multi_cond", False)
    cfg.generation.args.get("fold_cond", False)
    njobs = cfg.get("gen_njobs", 1)
    root_path = setup(
        cfg, create_root=(local_rank == 0), config_name=config_name, job_id=args.job_id
    )
    # Non-rank-0 processes wait for rank 0 to create the directory
    if is_dist:
        torch.distributed.barrier()
    if local_rank != 0:
        os.makedirs(root_path, exist_ok=True)

    # Exit if results from analysis already exist (assumes samples already there)
    csv_filename = f"results_{config_name}_{effective_job_id}.csv"
    csv_path = os.path.join(root_path, "..", csv_filename)
    if os.path.exists(csv_path):
        logger.info(f"Results already exist at {csv_path}. Exiting generate.py.")
        if is_dist:
            torch.distributed.destroy_process_group()
        sys.exit(0)

    cfg_gen = cfg.generation
    if not cfg.get("ab_design_mode", False):
        check_cfg_validity(cfg_gen.dataset, cfg_gen.args)

    # Load model — each rank loads its own copy onto its own GPU
    model = load_ckpt_n_configure_inference(cfg)

    # Create generation dataset
    cfg_gen = split_by_job(cfg_gen, args.job_id, njobs)

    # Motif-specific dataset creation
    if cfg.get("ab_design_mode", False):
        from proteinfoundation.datasets.ab_gen_dataset import AbGenDataset, collate_fn as ab_collate_fn
        dataset = AbGenDataset(
            data_dir=cfg.generation.dataset.data_dir,
            split=cfg.generation.dataset.get("split", "test"),
            nsamples=cfg.generation.dataset.get("nsamples", 1),
            max_antibody_len=cfg.generation.dataset.get("max_antibody_len", 450),
        )
        ab_batch_size = cfg.generation.dataset.get("batch_size", 1)

        if is_dist:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=ab_batch_size,
                sampler=sampler,
                collate_fn=ab_collate_fn,
            )
            if local_rank == 0:
                logger.info(
                    f"[dist] Dataset split across {world_size} GPUs: "
                    f"~{len(dataset) // world_size} structures/GPU, "
                    f"batch_size={ab_batch_size}"
                )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=ab_batch_size,
                shuffle=False,
                collate_fn=ab_collate_fn,
            )
            if ab_batch_size > 1:
                logger.info(f"[ab_design] Parallel generation: batch_size={ab_batch_size}")

    elif motif_cond or ("motif_task_name" in cfg.generation.dataset):
        motif_csv_path = os.path.join(
            root_path,
            f"{cfg_gen.dataset.get('motif_task_name', 'motif')}_{args.job_id}_motif_info.csv",
        )
        """
        Motif Configuration Examples:

        The motif dataset supports two modes for specifying which atoms to include:

        1. **Atom-level specification** (precise control):
           motif_dict_cfg:
             my_motif:
               motif_pdb_path: "path/to/motif.pdb"
               motif_atom_spec: "A64: [O, CG]; A65: [N, CA]; A66: [CB, CD]"
               # atom_selection_mode is ignored when motif_atom_spec is provided

        2. **Residue/range-based specification** (automatic atom selection):
           motif_dict_cfg:
             my_motif:
               motif_pdb_path: "path/to/motif.pdb"
               contig_string: "A1-7/A28-79"
               atom_selection_mode: "tip_atoms"  # NEW: Choose atom selection mode

           Available atom_selection_mode options:
           - "ca_only": Only CA atoms (default, fastest)
           - "all": All available atoms (most complete motif)
           - "backbone": Backbone atoms only (N, CA, C, O)
           - "sidechain": Sidechain atoms only
           - "tip_atoms": Tip atoms of sidechains (e.g., OH for Ser, NH2 for Arg)
           - "random": Random subset of available atoms

        If atom_selection_mode is not specified, defaults to "ca_only" for backward compatibility.
        """
        dataset = GenDataset(motif_csv_path=motif_csv_path, **cfg_gen.dataset)
    else:
        dataset = GenDataset(**cfg_gen.dataset)
    if not cfg.get("ab_design_mode", False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Move model to the correct GPU for this rank
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    model.eval()

    # Run generation loop — collect predictions and reference data in one pass
    all_predictions = []
    ref_data = []
    pdb_names = []
    chain_indexes = None

    # Only rank 0 shows the progress bar to avoid interleaved output
    show_pbar = (local_rank == 0)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, disable=not show_pbar)):
            batch = _to_device(batch, device)

            # Collect reference data for ab_design_mode before prediction
            if cfg.get("ab_design_mode", False):
                B_local = batch["chain_type"].shape[0]
                for b in range(B_local):
                    ct        = batch["chain_type"][b]           # [N]
                    chains_s  = batch["chains"][b]               # [N]
                    coords_nm_s = batch["coords_nm"][b]          # [N, 37, 3] nm
                    gt_ab_nm  = batch["gt_ab_coords_nm"][b]      # [N_ab_padded, 37, 3] nm
                    gt_seq_s  = batch["gt_seq"][b]               # [N]

                    ag_mask_b = (ct == 3)
                    ab_mask_b = (ct > 0) & (ct < 3)
                    n_ab      = int(ab_mask_b.sum().item())

                    ag_coords   = coords_nm_s[ag_mask_b] * 10.0  # nm → Å [N_ag, 37, 3]
                    ag_seq_r    = gt_seq_s[ag_mask_b]
                    ag_chain_r  = chains_s[ag_mask_b]

                    gt_ab_coords = gt_ab_nm[:n_ab] * 10.0        # nm → Å [N_ab, 37, 3]
                    gt_ab_seq_r  = gt_seq_s[ab_mask_b]
                    ab_chain_r   = chains_s[ab_mask_b]

                    ref_data.append({
                        'coords':       torch.cat([gt_ab_coords, ag_coords], dim=0),
                        'residue_type': torch.cat([gt_ab_seq_r,  ag_seq_r],  dim=0),
                        'chain_index':  torch.cat([ab_chain_r,   ag_chain_r], dim=0),
                    })
                pdb_names.extend(batch["pdb_ids"])

            pred = model.predict_step(batch, batch_idx)
            all_predictions.append(pred)

    # Save predictions — each rank writes its own files tagged with effective_job_id
    if motif_cond or ("motif_task_name" in cfg.generation.dataset):
        save_motif_predictions(
            root_path,
            all_predictions,
            job_id=effective_job_id,
            motif_pdb_name=cfg_gen.dataset.get("motif_task_name", None),
        )
        import shutil
        motif_csv = f"./{cfg_gen.dataset.get('motif_task_name', '')}_motif_info.csv"
        if os.path.exists(motif_csv):
            shutil.copy(motif_csv, root_path)
    else:
        save_predictions(
            root_path,
            all_predictions,
            job_id=effective_job_id,
            chain_indexes=chain_indexes,
            cath_codes=getattr(dataset, "cath_codes", None),
            ref_data=ref_data if ref_data else None,
            pdb_names=pdb_names if pdb_names else None,
        )

    # Wait for all ranks to finish writing before evaluation
    if is_dist:
        torch.distributed.barrier()

    # Evaluate antibody design metrics — each rank evaluates its own predictions
    if cfg.get("ab_design_mode", False) and cfg.get("ab_eval_metrics", True):
        logger.info(f"[rank {local_rank}] Computing antibody design metrics...")
        evaluate_ab_design(root_path, dataset, job_id=effective_job_id, cfg=cfg)

    # Clean up distributed process group
    if is_dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
