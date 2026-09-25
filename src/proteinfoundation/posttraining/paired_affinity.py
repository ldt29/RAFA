"""Create a paired bootstrap summary from two evaluator JSON artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    base_rows = {int(row["index"]): row for row in base["rows"]}
    cand_rows = {int(row["index"]): row for row in candidate["rows"]}
    indices = sorted(set(base_rows) & set(cand_rows))
    if not indices or len(indices) != len(base_rows) or len(indices) != len(cand_rows):
        raise ValueError("base/candidate rows are not an exact paired cohort")
    rng = np.random.default_rng(args.seed)
    output = {
        "artifact_type": "paired_bootstrap_summary",
        "base": str(args.base),
        "candidate": str(args.candidate),
        "paired_by": "dataset index",
        "n": len(indices),
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "selection_metric": candidate.get("selection_metric"),
        "affinity_teacher": candidate.get("affinity_teacher"),
        "affinity_teacher_weight": candidate.get("affinity_teacher_weight", 0.0),
        "claim_boundary": "Screening unless the preregistered full-mask cohort and structural gates pass.",
    }
    for metric in ("one_candidate", "best_of_K"):
        base_values = np.asarray([float(base_rows[index][metric]) for index in indices])
        cand_values = np.asarray([float(cand_rows[index][metric]) for index in indices])
        delta = cand_values - base_values
        draws = rng.integers(0, len(delta), size=(args.bootstrap, len(delta)))
        bootstrap = delta[draws].mean(axis=1)
        positive = int(np.count_nonzero(delta > 0.0))
        negative = int(np.count_nonzero(delta < 0.0))
        zero = int(np.count_nonzero(delta == 0.0))
        nonzero = positive + negative
        # Two-sided exact sign test, ignoring exact ties.  This is a compact
        # diagnostic for directional consistency and is not a replacement for
        # the paired bootstrap interval.
        if nonzero:
            k = min(positive, negative)
            tail = sum(math.comb(nonzero, i) for i in range(k + 1)) / (2.0 ** nonzero)
            sign_p = min(1.0, 2.0 * tail)
        else:
            sign_p = 1.0
        output[metric] = {
            "base_mean": float(base_values.mean()),
            "candidate_mean": float(cand_values.mean()),
            "delta_mean": float(delta.mean()),
            "bootstrap95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "signs": {
                "positive": positive,
                "negative": negative,
                "zero": zero,
                "exact_two_sided_p": float(sign_p),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
