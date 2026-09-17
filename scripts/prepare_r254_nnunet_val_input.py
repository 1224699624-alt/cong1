#!/usr/bin/env python3
"""Prepare original-val images for isolated R254 baseline/ours inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/raw/TSRS_RSNA-Epiphysis/val")
    p.add_argument("--output", default="outputs/nnunet/r254_hapdsp_pcr/imagesVal")
    args = p.parse_args()
    source, output = Path(args.image_dir), Path(args.output)
    if "TSRS_RSNA-Epiphysis" not in source.as_posix() or "Articular" in source.as_posix():
        raise RuntimeError("R254 only accepts TSRS_RSNA-Epiphysis")
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*"):
        if old.is_file():
            old.unlink()
    count = 0
    for image in sorted(source.iterdir()):
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        Image.open(image).convert("L").save(output / f"{image.stem}_0000.png")
        count += 1
    print(f"prepared {count} original-val images in {output}")
    if count != 96:
        raise SystemExit(f"Expected 96 original-val images, got {count}")


if __name__ == "__main__":
    main()
