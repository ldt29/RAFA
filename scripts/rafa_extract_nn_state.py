#!/usr/bin/env python3
"""Extract the trainable ``nn_state`` contract from a release envelope."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint must contain state_dict")
    nn_state = {
        key.removeprefix("nn."): value.detach().cpu()
        for key, value in state_dict.items()
        if key.startswith("nn.")
    }
    if not nn_state:
        raise ValueError("checkpoint contains no nn.* parameters")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "nn_state": nn_state,
        },
        args.out,
    )
    print(f"wrote {args.out} ({len(nn_state)} tensors)")


if __name__ == "__main__":
    main()
