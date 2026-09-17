#!/usr/bin/env python3
"""Build the isolated one-channel R293 nnU-Net development dataset.

Only TSRS_RSNA-Epiphysis train and original-val are included. Instance labels
are reduced to the binary task label; local interface supervision is derived
online from the augmented binary target so it remains perfectly aligned.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from train_r256_scale_invariant_pair_prior import find_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--nnunet-root", type=Path,
                        default=Path("outputs/nnunet/r293_dualbranch/data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    joined = f"{args.dataset_root} {args.nnunet_root}".lower()
    if (args.dataset_root.name != "TSRS_RSNA-Epiphysis" or
            "articular" in joined or "clean-test" in joined or "test-v2" in joined):
        raise RuntimeError("R293 is restricted to Epiphysis train/original-val")

    name = "Dataset293_TSRS_RSNAEpiphysisDualBranch2D"
    dataset = args.nnunet_root / "nnUNet_raw" / name
    if dataset.exists():
        if not args.overwrite:
            raise RuntimeError(f"Dataset exists: {dataset}")
        shutil.rmtree(dataset)
    images, labels = dataset / "imagesTr", dataset / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    train_keys: list[str] = []
    val_keys: list[str] = []
    rows: list[dict[str, object]] = []
    for split, keys in (("train", train_keys), ("val", val_keys)):
        for label_path in sorted((args.dataset_root / f"{split}_labels").glob("*.png")):
            stem = label_path.stem
            key = f"{split}_{stem}"
            ids = np.asarray(Image.open(label_path))
            if ids.ndim != 2:
                raise RuntimeError(f"Expected indexed 2-D label: {label_path}")
            xray = Image.open(find_image(args.dataset_root / split, stem)).convert("L")
            if xray.size != (ids.shape[1], ids.shape[0]):
                xray = xray.resize((ids.shape[1], ids.shape[0]), Image.Resampling.BILINEAR)
            xray.save(images / f"{key}_0000.png")
            Image.fromarray((ids > 0).astype(np.uint8)).save(labels / f"{key}.png")
            keys.append(key)
            rows.append({"case": key, "width": xray.width, "height": xray.height,
                         "instances": int(np.count_nonzero(np.unique(ids) > 0))})
    if (len(train_keys), len(val_keys)) != (875, 96):
        raise RuntimeError({"train": len(train_keys), "val": len(val_keys)})

    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "xray_native_resolution"},
        "labels": {"background": 0, "epiphysis": 1},
        "numTraining": len(rows), "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }, indent=2), encoding="utf-8")
    args.nnunet_root.mkdir(parents=True, exist_ok=True)
    (args.nnunet_root / "splits_final_r293.json").write_text(json.dumps([
        {"train": train_keys, "val": val_keys}
    ], indent=2), encoding="utf-8")
    manifest = {
        "dataset": str(dataset), "dataset_name": name,
        "train": len(train_keys), "original_val": len(val_keys),
        "input": "native-resolution X-ray only",
        "binary_target_from_instance_ids": True,
        "online_interface_target_after_augmentation": True,
        "source_dataset_unchanged": True, "clean_test_used": False,
    }
    (args.nnunet_root / "r293_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
