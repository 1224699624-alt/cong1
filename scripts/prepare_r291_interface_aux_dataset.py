#!/usr/bin/env python3
"""Build an isolated nnU-Net dataset with training-only R291 interface targets.

Channels 1--3 are alignment carriers for loss supervision. They are explicitly
zeroed before every network forward pass and are zero at inference, so the
deployed model only uses the X-ray in channel 0.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from audit_r291_image_driven_interface_targets import build_targets, find_image, resize_case


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        size, Image.Resampling.NEAREST), dtype=np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--nnunet-root", type=Path,
                   default=Path("outputs/nnunet/r291a_interface_hinge/data"))
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--max-gap-px", type=float, default=8.0)
    p.add_argument("--gap-halfwidth", type=int, default=1)
    p.add_argument("--interface-radius", type=int, default=7)
    p.add_argument("--support-depth", type=float, default=3.0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    joined = " ".join(str(v) for v in vars(args).values()).lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R291-A is restricted to Epiphysis train/original-val")

    dataset = args.nnunet_root / "nnUNet_raw" / "Dataset291_TSRS_RSNAEpiphysisInterfaceAux2D"
    if dataset.exists():
        if not args.overwrite:
            raise RuntimeError(f"Dataset exists: {dataset}")
        shutil.rmtree(dataset)
    images = dataset / "imagesTr"
    labels = dataset / "labelsTr"
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
                raise RuntimeError(f"Expected indexed label: {label_path} {ids.shape}")
            xray = Image.open(find_image(args.dataset_root, split, stem)).convert("L")
            if xray.size != (ids.shape[1], ids.shape[0]):
                xray = xray.resize((ids.shape[1], ids.shape[0]), Image.Resampling.BILINEAR)
            image_small, ids_small = resize_case(np.asarray(xray), ids.astype(np.int32), args.target_size)
            target = build_targets(ids_small, args.max_gap_px, args.gap_halfwidth,
                                   args.interface_radius, args.support_depth)
            native_size = xray.size
            gap = resize_mask(np.asarray(target["reliable_gap"], bool), native_size)
            support = resize_mask(np.asarray(target["support"], bool), native_size)
            ambiguous = np.asarray(target["ignore"], bool) & ~np.asarray(target["foreground"], bool)
            ignore = resize_mask(ambiguous, native_size)

            binary = (ids > 0).astype(np.uint8)
            encoded = binary.copy()
            encoded[(ignore > 0) & (binary == 0)] = 2

            xray.save(images / f"{key}_0000.png")
            Image.fromarray(gap).save(images / f"{key}_0001.png")
            Image.fromarray(support).save(images / f"{key}_0002.png")
            Image.fromarray(ignore).save(images / f"{key}_0003.png")
            Image.fromarray(encoded).save(labels / f"{key}.png")
            keys.append(key)
            rows.append({
                "case": key,
                "gap_fraction": float((gap > 0).mean()),
                "support_fraction": float((support > 0).mean()),
                "ignore_fraction": float((ignore > 0).mean()),
            })

    if (len(train_keys), len(val_keys)) != (875, 96):
        raise RuntimeError({"train": len(train_keys), "val": len(val_keys)})
    if any(float(r["gap_fraction"]) == 0 or float(r["support_fraction"]) == 0 for r in rows):
        raise RuntimeError("Every R291-A case must contain gap and support supervision")

    (dataset / "dataset.json").write_text(json.dumps({
        "channel_names": {
            "0": "xray",
            "1": "training_only_reliable_gap",
            "2": "training_only_bone_support",
            "3": "training_only_interface_ignore",
        },
        "labels": {"background": 0, "epiphysis": 1, "ignore": 2},
        "numTraining": len(train_keys) + len(val_keys),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }, indent=2), encoding="utf-8")
    split_path = args.nnunet_root / "splits_final_r291.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps([{"train": train_keys, "val": val_keys}], indent=2),
                          encoding="utf-8")
    manifest = {
        "dataset": str(dataset), "train": len(train_keys), "original_val": len(val_keys),
        "clean_test_used": False, "age_or_sex_used": False, "fixed_pair_list_used": False,
        "network_input_at_train_and_inference": "xray only; auxiliary channels zeroed",
        "parameters": {
            "target_size": args.target_size, "max_gap_px": args.max_gap_px,
            "gap_halfwidth": args.gap_halfwidth, "interface_radius": args.interface_radius,
            "support_depth": args.support_depth,
        },
        "mean_fractions": {k: float(np.mean([float(r[k]) for r in rows]))
                           for k in ("gap_fraction", "support_fraction", "ignore_fraction")},
    }
    (args.nnunet_root / "r291_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
