#!/usr/bin/env python3
"""Prepare a strict R260 binary-vs-instance-auxiliary supervision pair.

Both datasets use the same original X-ray and frozen R259 prior channels. The
instance-auxiliary target stores instance-derived background seam pixels as
value 2, while the dataset remains a two-class background/bone task. The R306
trainer maps value 2 back to background for the ordinary nnU-Net loss and uses
it only for the auxiliary seam penalty.
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
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r259_frozen_relation"))
    parser.add_argument("--seam-root", type=Path, default=Path("outputs/targets/r271_instance_seam_full"))
    parser.add_argument("--output", type=Path, default=Path("outputs/nnunet/r306_r260_instance_aux_pair/data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    joined = " ".join(map(str, vars(args).values())).lower()
    if args.raw_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R306 is restricted to original TSRS_RSNA-Epiphysis train/original-val")

    raw_base = args.output / "nnUNet_raw"
    binary = raw_base / "Dataset307_TSRSR260BinaryPrior2D"
    instance = raw_base / "Dataset308_TSRSR260InstanceAuxPrior2D"
    if args.overwrite:
        shutil.rmtree(binary, ignore_errors=True)
        shutil.rmtree(instance, ignore_errors=True)
    for root in (binary, instance):
        (root / "imagesTr").mkdir(parents=True, exist_ok=True)
        (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    train_keys: list[str] = []
    val_keys: list[str] = []
    seam_pixels = 0
    prior_nonzero = 0
    for split, keys in (("train", train_keys), ("val", val_keys)):
        for label_path in sorted((args.raw_root / f"{split}_labels").glob("*.png")):
            stem = label_path.stem
            key = f"{split}_{stem}"
            image = Image.open(find_image(args.raw_root / split, stem)).convert("L")
            ids = np.asarray(Image.open(label_path)).astype(np.int32)
            if ids.ndim == 3:
                ids = ids[..., 0]
            prior_npy = args.prior_root / split / f"{stem}.npy"
            prior_png = args.prior_root / split / f"{stem}.png"
            seam_path = args.seam_root / split / "seam" / f"{stem}.png"
            if not prior_npy.exists() or not prior_png.exists() or not seam_path.exists():
                raise FileNotFoundError(f"missing prior/seam for {split}/{stem}")
            prior_values = np.load(prior_npy)
            prior = Image.open(prior_png).convert("L")
            seam = np.asarray(Image.open(seam_path).convert("L")) > 0
            if ids.shape != seam.shape or prior_values.shape != ids.shape or image.size != prior.size:
                raise RuntimeError(f"shape mismatch for {split}/{stem}")
            if not np.isfinite(prior_values).all() or float(prior_values.max()) <= 0:
                raise RuntimeError(f"invalid prior for {split}/{stem}")
            prior_nonzero += 1

            binary_target = (ids > 0).astype(np.uint8)
            instance_target = binary_target.copy()
            instance_target[(ids == 0) & seam] = 2
            seam_pixels += int((instance_target == 2).sum())
            if not set(np.unique(instance_target)).issubset({0, 1, 2}):
                raise RuntimeError(f"unexpected target values for {split}/{stem}")

            for root, target in ((binary, binary_target), (instance, instance_target)):
                image.save(root / "imagesTr" / f"{key}_0000.png")
                prior.save(root / "imagesTr" / f"{key}_0001.png")
                Image.fromarray(target, mode="L").save(root / "labelsTr" / f"{key}.png")
            keys.append(key)

    if (len(train_keys), len(val_keys), prior_nonzero) != (875, 96, 971):
        raise RuntimeError((len(train_keys), len(val_keys), prior_nonzero))

    common = {
        "channel_names": {"0": "xray", "1": "frozen_relation_prior"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": 971,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (binary / "dataset.json").write_text(json.dumps(common, indent=2))
    (instance / "dataset.json").write_text(json.dumps(common, indent=2))
    split_payload = [{"train": train_keys, "val": val_keys}]
    for root in (binary, instance):
        (root / "splits_final.json").write_text(json.dumps(split_payload, indent=2))

    val_input = args.output.parent / "imagesValPrior"
    shutil.rmtree(val_input, ignore_errors=True)
    val_input.mkdir(parents=True, exist_ok=True)
    for key in val_keys:
        stem = key.removeprefix("val_")
        Image.open(find_image(args.raw_root / "val", stem)).convert("L").save(val_input / f"{key}_0000.png")
        Image.open(args.prior_root / "val" / f"{stem}.png").convert("L").save(val_input / f"{key}_0001.png")

    manifest = {
        "experiment": "R306",
        "train": 875,
        "val": 96,
        "seam_pixels": seam_pixels,
        "prior_maps": prior_nonzero,
        "shared_output": "binary background/bone",
        "only_variable": "instance-ID-derived seam auxiliary penalty",
        "source_unchanged": True,
        "clean_test_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
