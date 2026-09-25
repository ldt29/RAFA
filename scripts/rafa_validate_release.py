#!/usr/bin/env python3
"""Fail-closed structural validation for the anonymous RAFA bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import DictConfig, ListConfig
import torch


def internal_path_strings(value, path: str = "root"):
    """Return serialized strings that expose a machine-local path."""
    found = []
    if isinstance(value, (dict, DictConfig)):
        for key, item in value.items():
            found.extend(internal_path_strings(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple, ListConfig)):
        for index, item in enumerate(value):
            found.extend(internal_path_strings(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        lowered = value.lower()
        if any(token in lowered for token in ("/mnt/", "/home/", "/root/")):
            found.append((path, value))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    required = [
        root / "README.md",
        root / "configs/inference_ab_design.yaml",
        root / "configs/posttraining_rl.yaml",
        root / "checkpoints/ae.ckpt",
        root / "checkpoints/design_base.ckpt",
        root / "checkpoints/design_rl.ckpt",
        root / "scripts/rafa_infer.sh",
        root / "scripts/rafa_train_rgt.sh",
        root / "src/proteinfoundation/proteina.py",
        root / "src/openfold/np/residue_constants.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing release files:\n" + "\n".join(missing))

    ae = torch.load(root / "checkpoints/ae.ckpt", map_location="meta", weights_only=False)
    base = torch.load(root / "checkpoints/design_base.ckpt", map_location="meta", weights_only=False)
    rgt = torch.load(root / "checkpoints/design_rl.ckpt", map_location="meta", weights_only=False)

    leaked = []
    for label, payload in (("ae", ae), ("design_base", base), ("design_rl", rgt)):
        leaked.extend((label, path, value) for path, value in internal_path_strings(payload))
    if leaked:
        preview = "\n".join(f"{label}: {path}: {value}" for label, path, value in leaked[:8])
        raise SystemExit("checkpoint metadata contains machine-local paths:\n" + preview)

    if "hyper_parameters" not in ae or "cfg_ae" not in ae["hyper_parameters"]:
        raise SystemExit("ae.ckpt does not contain cfg_ae hyperparameters")
    for label, payload in (("design_base", base), ("design_rl", rgt)):
        if "hyper_parameters" not in payload or "cfg_exp" not in payload["hyper_parameters"]:
            raise SystemExit(f"{label}.ckpt does not contain cfg_exp hyperparameters")
        state = payload.get("state_dict")
        if not isinstance(state, dict) or not state:
            raise SystemExit(f"{label}.ckpt has no state_dict")
        if not all(str(key).startswith("nn.") for key in state):
            raise SystemExit(f"{label}.ckpt contains non-NN state_dict keys")

    base_keys = set(base["state_dict"])
    rgt_keys = set(rgt["state_dict"])
    missing = base_keys - rgt_keys
    allowed = {key for key in missing if key.startswith("nn.privileged_encoder.")}
    if missing - allowed or rgt_keys - base_keys:
        raise SystemExit(
            "design_rl NN contract mismatch: "
            f"missing={sorted(missing - allowed)[:8]} "
            f"unexpected={sorted(rgt_keys - base_keys)[:8]}"
        )

    py_files = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    for path in py_files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    report = {
        "root": str(root),
        "checkpoint_sizes": {
            name: (root / "checkpoints" / name).stat().st_size
            for name in ("ae.ckpt", "design_base.ckpt", "design_rl.ckpt")
        },
        "design_base_state_dict_keys": len(base_keys),
        "design_rl_state_dict_keys": len(rgt_keys),
        "allowed_rgt_missing_keys": sorted(allowed),
        "python_files_compiled": len(py_files),
        "status": "ok",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
