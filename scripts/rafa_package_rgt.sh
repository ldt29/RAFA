#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/rgt_seed5_steps150}"
RAW="${RAW:-$RUN_DIR/rgt_step000150_final.pt}"
OUTPUT="${OUTPUT:-$ROOT/checkpoints/design_rl_retrained.ckpt}"

[[ -s "$RAW" ]] || { echo "Missing RGT trainer artifact: $RAW" >&2; exit 2; }
[[ -s "$ROOT/checkpoints/design_base.ckpt" ]] || { echo "Missing design_base.ckpt" >&2; exit 2; }
[[ -s "$ROOT/checkpoints/ae.ckpt" ]] || { echo "Missing ae.ckpt" >&2; exit 2; }

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec "$PYTHON_BIN" "$ROOT/scripts/rafa_package_checkpoint.py" \
  --raw "$RAW" \
  --base "$ROOT/checkpoints/design_base.ckpt" \
  --ae "$ROOT/checkpoints/ae.ckpt" \
  --out "$OUTPUT"
