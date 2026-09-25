#!/usr/bin/env bash
set -euo pipefail

# Reproduce the released RGT-150 training loop from design_base.ckpt.
# R_free training uses the two small calibration files shipped in assets/.
# Usage:
#   DATASET=/path/to/structure_dataset ./scripts/rafa_train_rgt.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-}"
SEED="${SEED:-5}"
MAX_ITERS="${MAX_ITERS:-150}"
K="${K:-8}"
NSTEPS="${NSTEPS:-50}"
LR="${LR:-5e-7}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/rgt_seed${SEED}_steps${MAX_ITERS}}"
PREFLIGHT="${PREFLIGHT:-false}"

if [[ -z "$DATASET" ]]; then
  echo "Set DATASET to the structure_dataset root." >&2
  exit 2
fi
[[ -d "$DATASET" ]] || { echo "Dataset does not exist: $DATASET" >&2; exit 2; }
[[ -s "$ROOT/checkpoints/design_base.ckpt" ]] || { echo "Missing design_base.ckpt" >&2; exit 2; }
[[ -s "$ROOT/checkpoints/ae.ckpt" ]] || { echo "Missing ae.ckpt" >&2; exit 2; }
[[ -s "$ROOT/assets/affinity_sequence_teacher.npz" ]] || { echo "Missing affinity teacher" >&2; exit 2; }
[[ -s "$ROOT/assets/protenix_fold_surrogate.json" ]] || { echo "Missing fold surrogate" >&2; exit 2; }
if [[ -e "$RUN_DIR/rgt_step$(printf '%06d' "$MAX_ITERS")_final.pt" ]]; then
  echo "Run already has a final checkpoint; choose another RUN_DIR: $RUN_DIR" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

SHARED_STATE="$RUN_DIR/shared_init.pt"
if [[ ! -s "$SHARED_STATE" ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/rafa_extract_nn_state.py" \
    --checkpoint "$ROOT/checkpoints/design_base.ckpt" \
    --out "$SHARED_STATE"
fi

PREFLIGHT_ARGS=()
if [[ "$PREFLIGHT" == "true" ]]; then
  PREFLIGHT_ARGS+=(--preflight)
fi

exec "$PYTHON_BIN" "$ROOT/scripts/rafa_train_rgt.py" \
  --method rgt \
  --stage formal \
  --run-dir "$RUN_DIR" \
  --base-checkpoint "$ROOT/checkpoints/design_base.ckpt" \
  --shared-state "$SHARED_STATE" \
  --autoencoder "$ROOT/checkpoints/ae.ckpt" \
  --dataset "$DATASET" \
  --affinity-teacher "$ROOT/assets/affinity_sequence_teacher.npz" \
  --fold-surrogate "$ROOT/assets/protenix_fold_surrogate.json" \
  --K "$K" \
  --nsteps "$NSTEPS" \
  --max-iters "$MAX_ITERS" \
  --lr "$LR" \
  --seed "$SEED" \
  "${PREFLIGHT_ARGS[@]}"
