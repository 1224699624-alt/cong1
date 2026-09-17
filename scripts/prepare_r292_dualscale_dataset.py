#!/usr/bin/env python3
"""Build an isolated two-scale X-ray nnU-Net dataset.

Channel 0 is the original-resolution grayscale image. Channel 1 is a smooth
global-context image made by downscaling the whole hand to a fixed long-side
size and resizing it back to native geometry. No labels are read to construct
either input channel.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from train_r256_scale_invariant_pair_prior import find_image


def global_context(image: Image.Image, long_side: int) -> Image.Image:
    w, h = image.size
    scale = min(1.0, float(long_side) / max(w, h))
    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--nnunet-root", type=Path,
                   default=Path("outputs/nnunet/r292_dualscale_input/data"))
    p.add_argument("--global-long-side", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    joined = " ".join(str(v) for v in vars(args).values()).lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R292 is restricted to Epiphysis train/original-val")

    dataset = args.nnunet_root / "nnUNet_raw" / "Dataset292_TSRS_RSNAEpiphysisDualScale2D"
    if dataset.exists():
        if not args.overwrite:
            raise RuntimeError(f"Dataset exists: {dataset}")
        shutil.rmtree(dataset)
    images, labels = dataset / "imagesTr", dataset / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    train_keys, val_keys = [], []
    rows = []
    for split, keys in (("train", train_keys), ("val", val_keys)):
        for label_path in sorted((args.dataset_root / f"{split}_labels").glob("*.png")):
            stem = label_path.stem
            key = f"{split}_{stem}"
            xray = Image.open(find_image(args.dataset_root / split, stem)).convert("L")
            ids = np.asarray(Image.open(label_path))
            if ids.ndim != 2:
                raise RuntimeError(f"Expected indexed label: {label_path}")
            if xray.size != (ids.shape[1], ids.shape[0]):
                xray = xray.resize((ids.shape[1], ids.shape[0]), Image.Resampling.BILINEAR)
            context = global_context(xray, args.global_long_side)
            xray.save(images / f"{key}_0000.png")
            context.save(images / f"{key}_0001.png")
            Image.fromarray((ids > 0).astype(np.uint8)).save(labels / f"{key}.png")
            keys.append(key)
            rows.append({"case": key, "width": xray.width, "height": xray.height,
                         "global_long_side": args.global_long_side})
    if (len(train_keys), len(val_keys)) != (875, 96):
        raise RuntimeError({"train": len(train_keys), "val": len(val_keys)})
    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "xray_high_resolution", "1": "xray_global_context_512"},
        "labels": {"background": 0, "epiphysis": 1},
        "numTraining": 971, "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }, indent=2), encoding="utf-8")
    args.nnunet_root.mkdir(parents=True, exist_ok=True)
    (args.nnunet_root / "splits_final_r292.json").write_text(
        json.dumps([{"train": train_keys, "val": val_keys}], indent=2), encoding="utf-8")
    (args.nnunet_root / "r292_dataset_manifest.json").write_text(json.dumps({
        "dataset": str(dataset), "train": 875, "original_val": 96,
        "global_long_side": args.global_long_side,
        "input_channels": "native high-resolution X-ray + downsampled whole-hand context",
        "labels_used_for_input": False, "clean_test_used": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(dataset), "train": 875, "original_val": 96,
                      "global_long_side": args.global_long_side}, indent=2))


if __name__ == "__main__":
    main()
