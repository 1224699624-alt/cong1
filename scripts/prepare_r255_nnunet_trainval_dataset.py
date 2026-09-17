#!/usr/bin/env python3
"""Prepare Dataset202 from Epiphysis original train/val only for R255."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


# Match the original R202 preparation order. Some validation cases retain
# zero-byte placeholder PNG/JPEG/BMP files alongside the valid JPG source.
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_image(root: Path, split: str, stem: str) -> Path:
    for extension in EXTENSIONS:
        path = root / split / f"{stem}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No image for {split}/{stem}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--nnunet-root", type=Path, default=Path("outputs/nnunet/r202"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "Articular" in str(args.dataset_root):
        raise RuntimeError("R255 permits TSRS_RSNA-Epiphysis only")
    dataset = args.nnunet_root / "nnUNet_raw" / "Dataset202_TSRS_RSNAEpiphysis2D"
    if dataset.exists() and args.overwrite:
        resolved, allowed = dataset.resolve(), (args.nnunet_root / "nnUNet_raw").resolve()
        if not str(resolved).startswith(str(allowed)):
            raise RuntimeError(f"Refusing to remove {resolved}")
        shutil.rmtree(dataset)
    images, labels = dataset / "imagesTr", dataset / "labelsTr"
    train_keys, val_keys = [], []
    for split, keys in (("train", train_keys), ("val", val_keys)):
        for label_path in sorted((args.dataset_root / f"{split}_labels").glob("*.png")):
            key = f"{split}_{label_path.stem}"
            image = Image.open(find_image(args.dataset_root, split, label_path.stem)).convert("L")
            label = np.asarray(Image.open(label_path))
            if label.ndim == 3:
                label = label[..., 0]
            images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
            image.save(images / f"{key}_0000.png")
            Image.fromarray((label > 0).astype(np.uint8), mode="L").save(labels / f"{key}.png")
            keys.append(key)
    if (len(train_keys), len(val_keys)) != (875, 96):
        raise RuntimeError(f"Unexpected original split sizes: {len(train_keys)}, {len(val_keys)}")
    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "xray"},
        "labels": {"background": 0, "epiphysis": 1},
        "numTraining": len(train_keys) + len(val_keys),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }, indent=2), encoding="utf-8")
    split_path = args.nnunet_root / "splits_final_r202.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps([{"train": train_keys, "val": val_keys}], indent=2), encoding="utf-8")
    print(json.dumps({"scope": "Epiphysis original train/val only", "train": len(train_keys), "val": len(val_keys), "dataset": str(dataset)}, indent=2))


if __name__ == "__main__":
    main()
