#!/usr/bin/env python3
"""Build the isolated two-channel R279 dataset (X-ray + frozen instance-seam prior)."""
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
    parser.add_argument("--prior-root", type=Path,
                        default=Path("outputs/priors/r279_frozen_instance_seam"))
    parser.add_argument("--nnunet-root", type=Path,
                        default=Path("outputs/nnunet/r279_instance_prior/data"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    joined = " ".join(map(str, (args.dataset_root, args.prior_root, args.nnunet_root))).lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R279 is restricted to Epiphysis train/original-val")

    dataset = (args.nnunet_root / "nnUNet_raw" /
               "Dataset279_TSRS_RSNAEpiphysisInstancePrior2D")
    if dataset.exists():
        if not args.overwrite:
            raise RuntimeError(f"Dataset already exists: {dataset}; pass --overwrite")
        shutil.rmtree(dataset)

    images_tr = dataset / "imagesTr"
    labels_tr = dataset / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)
    train_keys: list[str] = []
    val_keys: list[str] = []
    prior_stats: list[dict[str, float | str]] = []

    for split, keys in (("train", train_keys), ("val", val_keys)):
        for label_path in sorted((args.dataset_root / f"{split}_labels").glob("*.png")):
            stem = label_path.stem
            key = f"{split}_{stem}"
            xray = Image.open(find_image(args.dataset_root / split, stem)).convert("L")
            prior_path = args.prior_root / split / f"{stem}.png"
            if not prior_path.exists():
                raise RuntimeError(f"Missing frozen prior: {prior_path}")
            prior = Image.open(prior_path).convert("L")
            prior_array = np.asarray(prior, dtype=np.uint8)
            if prior.size != xray.size or not np.isfinite(prior_array).all():
                raise RuntimeError(f"Invalid prior: {prior_path}")

            instance_label = np.asarray(Image.open(label_path))
            if instance_label.ndim == 3:
                instance_label = instance_label[..., 0]
            binary_label = (instance_label > 0).astype(np.uint8)

            xray.save(images_tr / f"{key}_0000.png")
            prior.save(images_tr / f"{key}_0001.png")
            Image.fromarray(binary_label).save(labels_tr / f"{key}.png")
            keys.append(key)
            prior_stats.append({
                "case": key,
                "mean": float(prior_array.mean() / 255.0),
                "max": float(prior_array.max() / 255.0),
                "nonzero_fraction": float((prior_array > 0).mean()),
            })

    if (len(train_keys), len(val_keys)) != (875, 96):
        raise RuntimeError({"train": len(train_keys), "val": len(val_keys)})
    if not any(float(row["max"]) > 0 for row in prior_stats):
        raise RuntimeError("All frozen priors are empty")

    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "xray", "1": "frozen_instance_seam_prior"},
        "labels": {"background": 0, "epiphysis": 1},
        "numTraining": 971,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }, indent=2), encoding="utf-8")
    split_path = args.nnunet_root / "splits_final_r279.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps([{"train": train_keys, "val": val_keys}], indent=2),
                          encoding="utf-8")
    (args.nnunet_root / "r279_dataset_manifest.json").write_text(json.dumps({
        "dataset": str(dataset),
        "train": len(train_keys),
        "original_val": len(val_keys),
        "clean_test_used": False,
        "prior_source": str(args.prior_root),
        "prior_stats": prior_stats,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(dataset), "train": 875, "original_val": 96,
                      "clean_test_used": False}))


if __name__ == "__main__":
    main()
