#!/usr/bin/env python3
"""Expand only a mature binary nnU-Net checkpoint's input convolution to two channels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    input_hash = sha256(args.input)
    if args.expected_sha256 and input_hash != args.expected_sha256:
        raise RuntimeError({"expected": args.expected_sha256, "actual": input_hash})
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state = checkpoint["network_weights"]
    source = {key: value.clone() for key, value in state.items()}
    changed: list[str] = []
    for key, value in list(state.items()):
        if key.endswith("stages.0.0.convs.0.conv.weight") or key.endswith(
                "stages.0.0.convs.0.all_modules.0.weight"):
            if tuple(value.shape) != (32, 1, 3, 3):
                raise RuntimeError((key, tuple(value.shape)))
            expanded = torch.zeros((32, 2, 3, 3), dtype=value.dtype)
            expanded[:, 0] = value[:, 0]
            state[key] = expanded
            changed.append(key)
    if len(changed) != 4:
        raise RuntimeError({"expected_changed_aliases": 4, "actual": changed})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    restored = torch.load(args.output, map_location="cpu", weights_only=False)["network_weights"]
    if set(restored) != set(source):
        raise RuntimeError("Network key set changed")
    for key, old in source.items():
        new = restored[key]
        if key in changed:
            if not torch.equal(new[:, 0], old[:, 0]) or int(torch.count_nonzero(new[:, 1])) != 0:
                raise RuntimeError(f"Invalid expanded input tensor: {key}")
        elif not torch.equal(new, old):
            raise RuntimeError(f"Unexpected tensor change: {key}")
    heads = [key for key in restored if "seg_layers" in key]
    if not heads:
        raise RuntimeError("Segmentation heads missing")
    print(json.dumps({
        "input_sha256": input_hash,
        "output_sha256": sha256(args.output),
        "changed": sorted(changed),
        "network_key_count": len(restored),
        "unchanged_tensor_count": len(restored) - len(changed),
        "xray_channel_exact": True,
        "prior_channel_nonzero": 0,
        "segmentation_head_tensor_count": len(heads),
    }, indent=2))


if __name__ == "__main__":
    main()
