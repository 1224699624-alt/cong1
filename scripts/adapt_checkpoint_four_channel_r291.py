#!/usr/bin/env python3
"""Expand a mature one-channel nnU-Net checkpoint to four channels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-sha256")
    args = p.parse_args()
    actual = sha256(args.input)
    if args.expected_sha256 and actual != args.expected_sha256:
        raise RuntimeError({"expected": args.expected_sha256, "actual": actual})
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint["network_weights"]
    source = {k: v.clone() for k, v in state.items()}
    changed = []
    for key, value in list(state.items()):
        if key.endswith("stages.0.0.convs.0.conv.weight") or key.endswith(
                "stages.0.0.convs.0.all_modules.0.weight"):
            if tuple(value.shape) != (32, 1, 3, 3):
                raise RuntimeError((key, tuple(value.shape)))
            expanded = torch.zeros((32, 4, 3, 3), dtype=value.dtype)
            expanded[:, 0] = value[:, 0]
            state[key] = expanded
            changed.append(key)
    if len(changed) != 4:
        raise RuntimeError({"expected_changed_aliases": 4, "actual": changed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    restored = torch.load(args.output, map_location="cpu", weights_only=False)["network_weights"]
    for key, old in source.items():
        new = restored[key]
        if key in changed:
            if not torch.equal(new[:, 0], old[:, 0]) or int(torch.count_nonzero(new[:, 1:])):
                raise RuntimeError(f"Invalid expansion: {key}")
        elif not torch.equal(new, old):
            raise RuntimeError(f"Unexpected change: {key}")
    print(json.dumps({
        "input_sha256": actual, "output_sha256": sha256(args.output),
        "changed": sorted(changed), "xray_channel_exact": True,
        "auxiliary_weight_nonzero": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
