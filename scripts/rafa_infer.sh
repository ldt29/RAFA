#!/usr/bin/env bash
set -euo pipefail

# Minimal antibody/nanobody complex design inference entry point.
# Usage:
#   DATASET=/path/to/structure_dataset MODEL=design_rl \
#   ./scripts/rafa_infer.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-}"
MODEL="${MODEL:-design_rl}"
CHECKPOINT="${CHECKPOINT:-}"
SPLIT="${SPLIT:-test}"
SEED="${SEED:-10}"
NSTEPS="${NSTEPS:-400}"
NSAMPLES="${NSAMPLES:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
if [[ -n "$CHECKPOINT" ]]; then
  CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
  CKPT_DIR="$(dirname "$CHECKPOINT")"
  CKPT_NAME="$(basename "$CHECKPOINT")"
  MODEL_LABEL="custom"
else
  CKPT_DIR="$ROOT/checkpoints"
  CKPT_NAME="${MODEL}.ckpt"
  MODEL_LABEL="$MODEL"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/inference/${MODEL_LABEL}_seed${SEED}}"

if [[ -z "$DATASET" ]]; then
  echo "Set DATASET to the structure_dataset root." >&2
  exit 2
fi
if [[ -z "$CHECKPOINT" && "$MODEL" != "design_base" && "$MODEL" != "design_rl" ]]; then
  echo "MODEL must be design_base or design_rl, got: $MODEL" >&2
  exit 2
fi
[[ -d "$DATASET" ]] || { echo "Dataset does not exist: $DATASET" >&2; exit 2; }
[[ -s "$CKPT_DIR/$CKPT_NAME" ]] || {
  echo "Checkpoint is missing: $CKPT_DIR/$CKPT_NAME" >&2
  exit 2
}
[[ -s "$ROOT/checkpoints/ae.ckpt" ]] || {
  echo "Checkpoint is missing: $ROOT/checkpoints/ae.ckpt" >&2
  exit 2
}
if [[ -d "$OUTPUT_DIR" ]] && find "$OUTPUT_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "Output directory is non-empty; choose a new OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

exec "$PYTHON_BIN" "$ROOT/src/proteinfoundation/generate.py" \
  --config_name inference_ab_design \
  --data_path "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  ckpt_path="$CKPT_DIR" \
  ckpt_name="$CKPT_NAME" \
  autoencoder_ckpt_path="$ROOT/checkpoints/ae.ckpt" \
  seed="$SEED" \
  ab_design_mode=true \
  ab_eval_metrics=false \
  generation.args.nsteps="$NSTEPS" \
  generation.dataset.data_dir="$DATASET" \
  generation.dataset.split="$SPLIT" \
  generation.dataset.nsamples="$NSAMPLES" \
  generation.dataset.batch_size="$BATCH_SIZE"
