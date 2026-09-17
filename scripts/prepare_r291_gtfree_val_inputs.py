#!/usr/bin/env python3
"""Create four-channel original-val inputs whose auxiliary channels are zero."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from audit_r291_image_driven_interface_targets import find_image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    joined = f"{args.dataset_root} {args.output}".lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R291-A inference is restricted to Epiphysis original-val")
    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    labels = sorted((args.dataset_root / "val_labels").glob("*.png"))
    if len(labels) != 96:
        raise RuntimeError(len(labels))
    for label_path in labels:
        stem = label_path.stem
        xray = Image.open(find_image(args.dataset_root, "val", stem)).convert("L")
        zero = Image.new("L", xray.size, 0)
        xray.save(args.output / f"val_{stem}_0000.png")
        for channel in (1, 2, 3):
            zero.save(args.output / f"val_{stem}_{channel:04d}.png")
    print({"cases": 96, "channels": 4, "non_xray_channels": "all exact zero",
           "validation_gt_read_for_input": False, "clean_test_used": False})


if __name__ == "__main__":
    main()
