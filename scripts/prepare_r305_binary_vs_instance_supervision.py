#!/usr/bin/env python3
"""Prepare an isolated paired nnU-Net dataset differing only in supervision.

Dataset305 uses binary background/bone labels. Dataset306 uses the same images
and marks instance-derived background seams as class 2. The raw dataset is not
modified and clean-test-v2 is never read.
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
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--seam-root", type=Path, default=Path("outputs/targets/r271_instance_seam_full"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nnunet/r305_binary_vs_instance/data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    joined = " ".join(map(str, vars(args).values())).lower()
    if args.raw_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R305 is restricted to original TSRS_RSNA-Epiphysis train/original-val")

    raw_base = args.output / "nnUNet_raw"
    binary = raw_base / "Dataset305_TSRSBinarySupervision2D"
    instance = raw_base / "Dataset306_TSRSInstanceSeamSupervision2D"
    if args.overwrite:
        shutil.rmtree(binary, ignore_errors=True)
        shutil.rmtree(instance, ignore_errors=True)
    for root in (binary, instance):
        (root / "imagesTr").mkdir(parents=True, exist_ok=True)
        (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    train_keys: list[str] = []
    val_keys: list[str] = []
    seam_pixels = 0
    for split, keys in (("train", train_keys), ("val", val_keys)):
        label_dir = args.raw_root / f"{split}_labels"
        for label_path in sorted(label_dir.glob("*.png")):
            stem = label_path.stem
            key = f"{split}_{stem}"
            image_path = find_image(args.raw_root / split, stem)
            image = Image.open(image_path).convert("L")
            ids = np.asarray(Image.open(label_path)).astype(np.int32)
            seam_path = args.seam_root / split / "seam" / f"{stem}.png"
            if not seam_path.exists():
                raise FileNotFoundError(seam_path)
            seam = np.asarray(Image.open(seam_path).convert("L")) > 0
            if ids.shape != seam.shape or image.size != (ids.shape[1], ids.shape[0]):
                raise RuntimeError(f"shape mismatch for {split}/{stem}")
            binary_target = (ids > 0).astype(np.uint8)
            instance_target = binary_target.copy()
            instance_target[(ids == 0) & seam] = 2
            seam_pixels += int((instance_target == 2).sum())
            for root, target in ((binary, binary_target), (instance, instance_target)):
                image.save(root / "imagesTr" / f"{key}_0000.png")
                Image.fromarray(target, mode="L").save(root / "labelsTr" / f"{key}.png")
            keys.append(key)

    if (len(train_keys), len(val_keys)) != (875, 96):
        raise RuntimeError((len(train_keys), len(val_keys)))
    split_payload = [{"train": train_keys, "val": val_keys}]
    common = {"channel_names": {"0": "xray"}, "numTraining": 971,
              "file_ending": ".png", "overwrite_image_reader_writer": "NaturalImage2DIO"}
    (binary / "dataset.json").write_text(json.dumps({**common, "labels": {"background": 0, "bone": 1}}, indent=2))
    (instance / "dataset.json").write_text(json.dumps({**common, "labels": {"background": 0, "bone": 1, "instance_seam": 2}}, indent=2))
    for root in (binary, instance):
        (root / "splits_final.json").write_text(json.dumps(split_payload, indent=2))

    val_input = args.output.parent / "imagesVal"
    val_input.mkdir(parents=True, exist_ok=True)
    for key in val_keys:
        stem = key.removeprefix("val_")
        Image.open(find_image(args.raw_root / "val", stem)).convert("L").save(val_input / f"{key}_0000.png")
    manifest = {"experiment": "R305", "train": 875, "val": 96, "seam_pixels": seam_pixels,
                "only_variable": "binary labels versus instance-ID-derived seam labels",
                "source_unchanged": True, "clean_test_used": False}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
