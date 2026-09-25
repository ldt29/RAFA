"""Fit and serialize the frozen sequence-affinity teacher.

Example:
  python -m proteinfoundation.posttraining.fit_affinity_teacher \
    --data-dir artifacts/affinity_calibration/external/curated/binding_affinity_curated \
    --output artifacts/affinity_calibration/affinity_sequence_teacher.npz \
    --summary artifacts/affinity_calibration/affinity_sequence_teacher.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proteinfoundation.posttraining.affinity_teacher import fit_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-cdr-len", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    teacher, summary = fit_teacher(
        args.data_dir,
        max_cdr_len=args.max_cdr_len,
        alpha=args.alpha,
        seed=args.seed,
    )
    teacher.save(args.output)
    summary = dict(summary)
    summary["data_dir"] = str(args.data_dir)
    summary["artifact"] = str(args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
