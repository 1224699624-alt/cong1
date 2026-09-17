#!/usr/bin/env python3
"""Create GT-free two-channel original-val inputs for R279 inference."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from train_r256_scale_invariant_pair_prior import find_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path,
                        default=Path("outputs/priors/r279_frozen_instance_seam/val"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = " ".join(map(str, (args.dataset_root, args.prior_root, args.output))).lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R279 inference inputs are restricted to Epiphysis original-val")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    labels = sorted((args.dataset_root / "val_labels").glob("*.png"))
    if len(labels) != 96:
        raise RuntimeError(len(labels))
    for label_path in labels:
        stem = label_path.stem
        xray = Image.open(find_image(args.dataset_root / "val", stem)).convert("L")
        prior_path = args.prior_root / f"{stem}.png"
        if not prior_path.exists():
            raise RuntimeError(f"Missing frozen validation prior: {prior_path}")
        prior = Image.open(prior_path).convert("L")
        if prior.size != xray.size:
            raise RuntimeError(f"Shape mismatch for {stem}")
        xray.save(args.output / f"val_{stem}_0000.png")
        prior.save(args.output / f"val_{stem}_0001.png")
    print({"cases": 96, "channels": 2, "validation_gt_read_for_input": False,
           "clean_test_used": False})


if __name__ == "__main__":
    main()
