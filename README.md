# RAFA

This directory is an anonymous code-and-weight package for the RAFA
antibody/nanobody design model. It contains the frozen autoencoder, the
shared-base checkpoint, the RGT checkpoint, the model source, and the two
calibration artifacts needed by the native-free R_free training path.

## Checkpoints

Download the anonymous checkpoint archive from:

[`https://drive.google.com/file/d/190WmzYz5AKsrWao5aZWeK3KAKUWH1dnv/view`](https://drive.google.com/file/d/190WmzYz5AKsrWao5aZWeK3KAKUWH1dnv/view)

Place these files in `checkpoints/`:

```text
ae.ckpt
design_base.ckpt
design_rl.ckpt
```

## Layout

```text
checkpoints/
  ae.ckpt             frozen partial autoencoder
  design_base.ckpt    RAFA design model
  design_rl.ckpt      RAFA(RL) design model
src/
  proteinfoundation/  RAFA model, flow matching, datasets, and RGT trainer
  openfold/           local structural utilities
configs/              inference and RGT training defaults
scripts/              release-only launch, validation, and packaging scripts
assets/               sequence-teacher and fold-surrogate calibration files
docs/                 training-data contract and preprocessing description
data/                 public 143-case test split (93 VH--VL + 50 VHH)
```

`design_base.ckpt` and `design_rl.ckpt` contain only model parameters and
configuration. The autoencoder is stored once in `ae.ckpt`; the launch script
passes its package-local path explicitly.

## Environment

Use a CUDA-enabled PyTorch environment with matching PyG wheels. The package
requirements are intentionally unpinned for anonymous release; select wheels
compatible with the installed PyTorch and CUDA runtime.

```bash
python -m pip install -r requirements.txt
python -m pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

Verify the environment before a run:

```bash
PYTHON_BIN=/path/to/python
PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c \
  "import torch, lightning, torch_geometric, torch_scatter; \
   assert torch.cuda.is_available()"
PYTHONNOUSERSITE=1 "$PYTHON_BIN" scripts/rafa_validate_release.py
```

Inference requires a CUDA-capable GPU. RGT training is substantially more
expensive than inference; set `CUDA_VISIBLE_DEVICES` externally and use a
separate `RUN_DIR` for each training run.

The full training/inference dataset is the processed `structure_dataset` tree
used by the paper. Set `DATASET` to its root; it must contain the antibody
split metadata and processed structure data expected by
`AntibodyDesignDataset`. For a self-contained test run, use the bundled
`data/143-case` directory instead. The data contract, preprocessing steps,
and the distinction between the public test data and the full training
corpora are documented in [`docs/training_data.md`](docs/training_data.md).

Before sharing the bundle, run:

```bash
/path/to/python scripts/rafa_validate_release.py
```

The validator checks the three checkpoint schemas, the shared NN key contract,
and Python syntax without changing any release file.

## Inference

The default for the release `generate.py` entry point is the released RGT-150
model, seed 10, 400 ODE steps. This is
the documented seed-10 protocol whose VH--VL DockQ is about 0.17:

```bash
DATASET=/path/to/structure_dataset \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_infer.sh
```

The generated complex PDB files are written to
`inference/design_rl_seed10/`. To run the shared base instead:

```bash
DATASET=/path/to/structure_dataset \
MODEL=design_base SEED=10 \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_infer.sh
```

Useful overrides are `SPLIT`, `NSTEPS`, `NSAMPLES`, `BATCH_SIZE`, and
`OUTPUT_DIR`. A non-empty output directory is rejected so that runs are not
silently overwritten. A newly packaged checkpoint can be used with
`CHECKPOINT=/absolute/path/model.ckpt`.

## RGT training and packaging

The release includes the formal RGT-150 loop. It starts from the shared-base
checkpoint, uses K=8 rollouts and 50 ODE steps, and writes a trainer-facing
NN-only artifact. The R_free reward uses only the shipped calibration files
and the training split; native test metrics are not used by the selector.

```bash
DATASET=/path/to/structure_dataset \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_train_rgt.sh
```

The default output is `runs/rgt_seed5_steps150/`. For a smoke test, use a
separate run directory and one update:

```bash
DATASET=/path/to/structure_dataset MAX_ITERS=1 \
RUN_DIR="$PWD/runs/rgt_smoke" \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_train_rgt.sh
```

For a wiring-only GPU preflight that forces exactly one update while keeping
the formal K/ODE settings, set `PREFLIGHT=true` and keep `MAX_ITERS=150`:

```bash
DATASET=/path/to/structure_dataset MAX_ITERS=150 PREFLIGHT=true \
RUN_DIR="$PWD/runs/rgt_preflight" \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_train_rgt.sh
```

Package a completed trainer artifact into an inference checkpoint:

```bash
RUN_DIR="$PWD/runs/rgt_seed5_steps150" \
OUTPUT="$PWD/checkpoints/design_rl_retrained.ckpt" \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_package_rgt.sh
```

Then run it with:

```bash
DATASET=/path/to/structure_dataset \
CHECKPOINT="$PWD/checkpoints/design_rl_retrained.ckpt" \
PYTHON_BIN=/path/to/python \
bash scripts/rafa_infer.sh
```

Training is GPU-intensive. The release scripts do not select or terminate
processes on shared machines; set `CUDA_VISIBLE_DEVICES` externally.

## Reproducibility notes

- Training outputs are written under `runs/`, and newly packaged checkpoints
  use a new filename by default.
- Dataset paths and CUDA/PyTorch builds are environment-specific and are
  intentionally supplied by the user at launch time.
