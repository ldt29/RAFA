#!/usr/bin/env python3
"""Package an RGT NN-only artifact as a release inference checkpoint.

The frozen autoencoder is deliberately not duplicated into the packaged
checkpoint.  The base envelope supplies the serialized model configuration;
the release inference command supplies ``checkpoints/ae.ckpt`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
import torch


def scrub_public_metadata(value: Any, key: str = "") -> Any:
    """Remove machine-local paths from serialized checkpoint metadata.

    Tensor values are not passed through this function.  Only the config is
    sanitized, so packaging cannot accidentally
    publish the build host's absolute paths.
    """
    if isinstance(value, DictConfig):
        return OmegaConf.create({
            item_key: scrub_public_metadata(item_value, str(item_key))
            for item_key, item_value in value.items()
        })
    if isinstance(value, ListConfig):
        return OmegaConf.create([scrub_public_metadata(item, key) for item in value])
    if isinstance(value, dict):
        return {
            item_key: scrub_public_metadata(item_value, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [scrub_public_metadata(item, key) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_public_metadata(item, key) for item in value)
    if not isinstance(value, str):
        return value

    lowered = key.lower()
    if lowered == "store_dir":
        return "./tmp"
    if lowered == "data_dir":
        return "external/structure_dataset"
    if "autoencoder" in lowered and value.endswith(".ckpt"):
        return "checkpoints/ae.ckpt"
    if value.startswith(("/mnt/", "/home/", "/root/")):
        return "external/" + Path(value).name
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="RGT trainer artifact")
    parser.add_argument("--base", type=Path, required=True, help="design_base.ckpt")
    parser.add_argument("--ae", type=Path, required=True, help="ae.ckpt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = torch.load(args.raw, map_location="cpu", weights_only=False)
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("nn_state"), dict):
        raise ValueError("--raw must contain a dict nn_state")
    if not isinstance(base, dict) or not isinstance(base.get("state_dict"), dict):
        raise ValueError("--base must contain a state_dict")
    if "hyper_parameters" not in base:
        raise ValueError("--base must contain hyper_parameters")

    state_dict = {
        key if key.startswith("nn.") else f"nn.{key}": value
        for key, value in raw["nn_state"].items()
    }
    base_keys = set(base["state_dict"])
    state_keys = set(state_dict)
    allowed_missing = {
        key
        for key in base_keys - state_keys
        if key.startswith("nn.privileged_encoder.")
    }
    missing = sorted(base_keys - state_keys)
    unexpected = sorted(state_keys - base_keys)
    disallowed_missing = sorted(set(missing) - allowed_missing)
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "release NN contract mismatch: "
            f"missing={disallowed_missing[:8]} unexpected={unexpected[:8]}"
        )

    hyper_parameters = scrub_public_metadata(base["hyper_parameters"])
    hyper_parameters["autoencoder_ckpt_path"] = "checkpoints/ae.ckpt"
    payload = {
        "state_dict": {
            key: value.detach().cpu() for key, value in state_dict.items()
        },
        "hyper_parameters": hyper_parameters,
        # Required by Lightning's checkpoint migration loader.
        "pytorch-lightning_version": base.get("pytorch-lightning_version", "2.5.0"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp)
    tmp.replace(args.out)
    print(json.dumps({"checkpoint": args.out.name, "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
