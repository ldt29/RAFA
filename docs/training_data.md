# Training data and preprocessing

This document describes the data contract needed to reproduce the RAFA
training path and the preprocessing used by the runtime loaders in this
release. The release includes code, inference checkpoints, and the public
143-case test split. The autoencoder is frozen in this release; its optimizer
and callback history are intentionally not included. It does **not**
redistribute the full training corpora or upstream PDB, SAbDAb, and AF3
`release_data` archives. Those inputs must be obtained and redistributed
according to their own terms.

## Code map

The training and data path is split into the following components:

| Component | Role |
| --- | --- |
| `scripts/rafa_train_rgt.sh` | Reproducible RGT-150 launcher; checks inputs, extracts the shared NN state, and starts the formal loop. |
| `scripts/rafa_train_rgt.py` | Explicit RGT runner: loads the immutable model envelope, overlays the shared base, runs rollouts/rewards/updates, and writes a trainer artifact. |
| `scripts/rafa_extract_nn_state.py` | Extracts the NN-only shared-base state from `checkpoints/design_base.ckpt`. |
| `scripts/rafa_package_rgt.sh` and `scripts/rafa_package_checkpoint.py` | Package a trainer artifact into a model checkpoint for inference. |
| `src/proteinfoundation/datasets/ab_data.py` | SAbDAb/PDB-style antibody–antigen preprocessing, atom37 conversion, CDR/epitope/paratope masks, and the antibody train loader. |
| `src/proteinfoundation/datasets/ab_gen_dataset.py` | Test/inference loader for an atom37 antibody–antigen cache. |
| `src/proteinfoundation/datasets/protein_data.py` | Runtime loader for the general interacting-protein corpus. |
| `src/proteinfoundation/datasets/protein_base_data.py` | Combines general protein, VH–VL, and VHH streams and controls their batch ratio. |
| `src/proteinfoundation/generate.py` | Antigen-conditioned generation and output writing. |

The runtime loaders expect a local `structure_dataset` root. All paths in the
release are relative or supplied by the user at launch; no training machine
mount is required.

## Expected directory contract

For a full training run, set `DATASET` to a directory with this layout:

```text
structure_dataset/
├── before_20250630/
├── after_20250630_novel_ab/
├── after_20250630_novel_nb/
├── split_json/
│   ├── train.json
│   ├── valid.json
│   └── test.json
└── processed/
    ├── train_processed/_metainfo + part_*.pkl
    ├── valid_processed/_metainfo + part_*.pkl
    ├── test_processed/_metainfo + part_*.pkl
    └── antibody_processed/_metainfo + part_*.pkl
```

The release's `data/143-case` contains only the test-side subset and its
`test_processed` cache. It is sufficient to run inference and metric checks,
but intentionally cannot start a full training run because it has no train or
validation shards.

## RGT training command

After preparing the full antibody–antigen cache:

```bash
cd /path/to/release
DATASET=/path/to/structure_dataset \
PYTHON_BIN=/path/to/python \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/rafa_train_rgt.sh
```

The launcher defaults to seed 5, 150 updates, `K=8` rollouts, 50 ODE steps,
and the release calibration artifacts under `assets/`. Use a separate
`RUN_DIR` for each run. A wiring-only GPU preflight is available with
`MAX_ITERS=150 PREFLIGHT=true`; it forces one update while retaining the formal
K/ODE settings. The RGT reward uses the training split and the shipped
calibration artifacts; native test structures are not used to select a model.

`rafa_train_rgt.sh` is the post-training launcher for the released shared
base. Its formal RGT-150 path reads the antibody `train` cache with
`conventional_only=True` (VH–VL); it does not rebuild the general-protein
pre-adaptation stage and it does not consume the VHH stream in this formal
runner. The general-protein corpus and `ProteinBaseDataset` below document the
data contract needed to reproduce the earlier shared-base pre-adaptation or to
extend the training code, but that pre-adaptation launcher is not part of this
minimal public bundle.

For the antibody stream, `ProteinBaseDataset` creates separate VH–VL and VHH
datasets from the same train cache. The registered `abnb` batch pattern is
three VH–VL slots followed by one VHH slot. The formal RGT release starts from
the immutable shared-base model and does not overwrite either base checkpoint
or the input data.

## Antibody–antigen processing

The antibody corpus is derived from PDB/SAbDAb-style antibody–antigen
complexes. The upstream sample preparation is represented by the standardized
sample folders in the public test subset and follows these steps:

1. Split complexes containing multiple antigen chains into one sample per
   antigen target.
2. Write standardized chain records: `H` for heavy/VHH, `L` for light in
   VH–VL complexes, and `A` for the selected antigen. Generate the complex,
   antibody-only, antigen-only, and design FASTAs.
3. Detect and split merged heavy/light records when needed, then reconcile
   PDB and FASTA sequences. The PDB sequence is the structural ground truth;
   small sequence discrepancies such as standard residue-name substitutions
   are corrected, while unresolved large/middle mismatches are rejected.
4. Number antibody variable regions with IMGT-compatible CDR annotations.
   Replace H/L CDR residues with `X` in `design.fasta`; antigen residues are
   never masked. VHH samples mask H only.
5. Assign train/validation/test membership from the release-date split. The
   current public test set is the post-2025-06-30 novel split with 93 VH–VL
   and 50 VHH cases. `metadata.json` retains PDB identifier, release date,
   resolution, standardized chain IDs, source label, and QC/contact
   statistics.
6. Convert each valid complex to an atom37 cache. For every residue the cache
   stores coordinates, atom-validity masks, and sequence indices. It also
   stores the antibody CDR mask, antibody paratope flag, antigen epitope flag,
   and heavy-chain length. Samples without a usable CDR or epitope are
   excluded because the conditional cross-attention path would otherwise have
   inconsistent execution across distributed workers.

The release implementation of the cache conversion is
`src/proteinfoundation/datasets/ab_data.py`. To preprocess a complete
`structure_dataset`, importing `process_data` or constructing
`AntibodyDesignDataset` will create `processed/{train,valid,test,antibody}_processed`.
The conversion uses `max_antigen_len=500` by default and centers a truncation
window on the epitope when an antigen is longer than that limit. Paratope
residues are precomputed as antibody residues within 10 Å of an antigen CA;
epitope flags come from the complex's contact definition.

At training time, the antibody loader uses a maximum antibody length of 450
and applies the mixed CDR masking schedule `full/partial/none = 0.5/0.3/0.2`.
The partial case masks a random contiguous or scattered CDR subset; validation
and test use the full CDR mask. Chain types are `1=heavy`, `2=light`, and
`3=antigen`. Batches are centered around the epitope CA centroid (or the
antigen centroid when no epitope center is available).

## General-protein processing

The general-protein pre-adaptation corpus is a separate interacting-pair
dataset extracted from AF3 `release_data` `prot_prot` rows. It is not bundled
in this public release because the upstream archive is not redistributed here.
The source extraction contract is:

- require at least two protein chains;
- keep structures at resolution ≤4.5 Å;
- require each selected chain to have at least 30 residues;
- require the selected pair to have at most 950 residues in total; and
- require at least three Cα contacts within 8 Å.

AF3 label/subchain IDs are resolved against mmCIF author-chain names before
extraction. Each retained pair is standardized to a target chain (`A`) and a
fixed partner (`B`), with source indices, chain provenance, filtering
statistics, and hashes recorded in the upstream manifest. The expanded
training corpus uses a deterministic 50,000-pair cap; it is training-only and
is not a held-out benchmark.

The processed general-protein cache follows the same local-shard convention:

```text
protein/
├── dataset_manifest.json
└── processed/train_processed/
    ├── _metainfo
    └── part_0.pkl
```

Each record contains atom37 coordinates/masks, residue indices, target/partner
interface flags, and the two-chain provenance. The runtime
`ProteinPairDataset` randomly swaps which chain is generated, truncates the
target/partner to 450/500 residues, and uses chain types `1=target` and
`3=partner`. There are no CDRs in this stream. The default sequence-mask
schedule is `none_single_region` with probabilities `[0.2, 0.3, 0.5]` for no
mask, one-residue mask, and a random contiguous region. The structure/target
mask schedule is the shared full/partial/none schedule `[0.5, 0.3, 0.2]`.

`ProteinBaseDataset` exposes three stream modes:

```python
from proteinfoundation.datasets.protein_base_data import ProteinBaseDataset

dataset = ProteinBaseDataset(
    general_root="/path/to/structure_dataset/protein",
    antibody_root="/path/to/structure_dataset",
    stream="joint",       # "general", "abnb", or "joint"
)
```

The default joint slot ratio is general:VH–VL:VHH = `1:3:1`; `abnb` uses
VH–VL:VHH = `3:1`. Batches are deliberately homogeneous by task type so the
general and antibody schemas cannot be mixed silently.

## Public-data and privacy note

An audit of this release checked text metadata, source paths, scripts, and
checkpoint string tables. The package contains no detected personal names,
email addresses, credentials, private keys, internal host paths, or internal
IP addresses. PDB identifiers, structural coordinates, release dates, chain
IDs, and dataset statistics are scientific provenance and are retained for
reproducibility. Third-party PDB/SAbDAb terms still apply to the public
143-case structures, so downstream redistribution should preserve the
relevant attribution and license notices.
