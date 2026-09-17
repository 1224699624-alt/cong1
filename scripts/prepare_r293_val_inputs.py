#!/usr/bin/env python3
"""Create GT-free, X-ray-only original-val inputs for R293."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joined = f"{args.dataset_root} {args.output}".lower()
    if (args.dataset_root.name != "TSRS_RSNA-Epiphysis" or
            "articular" in joined or "clean-test" in joined or "test-v2" in joined):
        raise RuntimeError("R293 is restricted to Epiphysis original-val")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    label_paths = sorted((args.dataset_root / "val_labels").glob("*.png"))
    if len(label_paths) != 96:
        raise RuntimeError(len(label_paths))
    for label_path in label_paths:
        xray = Image.open(find_image(args.dataset_root / "val", label_path.stem)).convert("L")
        xray.save(args.output / f"val_{label_path.stem}_0000.png")
    print({"cases": len(label_paths), "channels": 1,
           "labels_used_for_input": False, "clean_test_used": False})


if __name__ == "__main__":
    main()
