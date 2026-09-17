#!/usr/bin/env python3
"""Create an isolated Dataset293 plan with the mature R275 patch geometry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    src = source["configurations"]["2d"]
    ref = reference["configurations"]["2d"]
    if src["architecture"] != ref["architecture"]:
        raise RuntimeError("Dataset293 and R275 architectures are not identical")
    for key in ("spacing", "normalization_schemes", "use_mask_for_norm"):
        if src[key] != ref[key]:
            raise RuntimeError({key: {"dataset293": src[key], "r275": ref[key]}})
    before = list(src["patch_size"])
    src["patch_size"] = list(ref["patch_size"])
    src["batch_size"] = int(ref["batch_size"])
    # nnU-Net derives the results-folder identifier from this field, not from
    # the JSON filename. Keep both aligned so launchers can deterministically
    # locate checkpoints produced with this isolated plan.
    source["plans_name"] = args.output.stem
    if src["patch_size"] != [512, 512]:
        raise RuntimeError(src["patch_size"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(source, indent=2), encoding="utf-8")
    manifest = {
        "experiment": "R296", "dataset": source["dataset_name"],
        "source_plan": str(args.input), "reference_plan": str(args.reference),
        "patch_size_before": before, "patch_size_after": src["patch_size"],
        "batch_size": src["batch_size"], "architecture_exact_match": True,
        "normalization_exact_match": True, "plans_name": source["plans_name"],
        "clean_test_used": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
