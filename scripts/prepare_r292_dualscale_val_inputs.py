#!/usr/bin/env python3
"""Create GT-free dual-scale original-val inputs."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from prepare_r292_dualscale_dataset import global_context
from train_r256_scale_invariant_pair_prior import find_image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--global-long-side", type=int, default=512)
    args = p.parse_args()
    joined = f"{args.dataset_root} {args.output}".lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R292 is restricted to Epiphysis original-val")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    labels = sorted((args.dataset_root / "val_labels").glob("*.png"))
    if len(labels) != 96:
        raise RuntimeError(len(labels))
    for label_path in labels:
        stem = label_path.stem
        xray = Image.open(find_image(args.dataset_root / "val", stem)).convert("L")
        xray.save(args.output / f"val_{stem}_0000.png")
        global_context(xray, args.global_long_side).save(
            args.output / f"val_{stem}_0001.png")
    print({"cases": 96, "channels": 2, "labels_used_for_input": False,
           "clean_test_used": False})


if __name__ == "__main__":
    main()
