#!/usr/bin/env python3
"""Convert an isolated nnU-Net PNG prediction folder to binary masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--expected-count", type=int, required=True)
    p.add_argument("--strip-prefix", default="")
    p.add_argument("--foreground-label", type=int, default=None,
                   help="If set, only this label is foreground; otherwise all labels > 0 are foreground.")
    args = p.parse_args()
    source, output = Path(args.pred_dir), Path(args.mask_dir)
    output.mkdir(parents=True, exist_ok=True)
    converted = 0
    for path in sorted(source.glob("*.png")):
        stem = path.stem[:-5] if path.stem.endswith("_0000") else path.stem
        if args.strip_prefix and stem.startswith(args.strip_prefix):
            stem = stem[len(args.strip_prefix):]
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        foreground = arr == args.foreground_label if args.foreground_label is not None else arr > 0
        Image.fromarray(foreground.astype(np.uint8) * 255).save(output / f"{stem}.png")
        converted += 1
    print(f"converted {converted} predictions to {output}")
    if converted != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count}, got {converted}")


if __name__ == "__main__":
    main()
