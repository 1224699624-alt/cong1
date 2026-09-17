#!/usr/bin/env python3
"""Prepare the clean-panel subset for a standard one-channel nnU-Net."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from train_r256_scale_invariant_pair_prior import find_image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--source-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/nnunet/r310_clean_plain_nnunet/data"))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    joined = " ".join(map(str, vars(a).values())).lower()
    if a.raw_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R310 is restricted to TSRS_RSNA-Epiphysis train/original-val")

    payload = json.loads(a.selection.read_text(encoding="utf-8"))
    keep = {s: sorted({str(x) for x in payload["keep"][s]}) for s in ("train", "val")}
    if (len(keep["train"]), len(keep["val"])) != (814, 94):
        raise RuntimeError({s: len(v) for s, v in keep.items()})

    dataset = a.output / "nnUNet_raw" / "Dataset310_TSRSPlainCleanPanels2D"
    if a.overwrite:
        shutil.rmtree(dataset, ignore_errors=True)
        for name in ("imagesVal", "cleanValGT"):
            shutil.rmtree(a.output.parent / name, ignore_errors=True)
    (dataset / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset / "labelsTr").mkdir(parents=True, exist_ok=True)

    keys = {"train": [], "val": []}
    for split in ("train", "val"):
        for stem in keep[split]:
            label_path = a.raw_root / f"{split}_labels" / f"{stem}.png"
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            image = Image.open(find_image(a.raw_root / split, stem)).convert("L")
            ids = np.asarray(Image.open(label_path))
            if ids.ndim == 3:
                ids = ids[..., 0]
            key = f"{split}_{stem}"
            image.save(dataset / "imagesTr" / f"{key}_0000.png")
            Image.fromarray((ids > 0).astype(np.uint8)).save(dataset / "labelsTr" / f"{key}.png")
            keys[split].append(key)

    dataset_json = {
        "channel_names": {"0": "xray"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": 908,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    (dataset / "splits_final.json").write_text(
        json.dumps([{"train": keys["train"], "val": keys["val"]}], indent=2), encoding="utf-8"
    )

    plan = json.loads(a.source_plan.read_text(encoding="utf-8"))
    plan["dataset_name"] = dataset.name
    plan["foreground_intensity_properties_per_channel"] = {
        "0": plan["foreground_intensity_properties_per_channel"]["0"]
    }
    cfg = plan["configurations"]["2d"]
    cfg["batch_size"] = 1
    cfg["normalization_schemes"] = [cfg["normalization_schemes"][0]]
    cfg["use_mask_for_norm"] = [cfg["use_mask_for_norm"][0]]
    plan_dir = a.output / "nnUNet_preprocessed" / dataset.name
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "nnUNetPlans.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    (plan_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    val_input = a.output.parent / "imagesVal"
    val_gt = a.output.parent / "cleanValGT"
    val_input.mkdir(parents=True, exist_ok=True)
    val_gt.mkdir(parents=True, exist_ok=True)
    for stem in keep["val"]:
        Image.open(find_image(a.raw_root / "val", stem)).convert("L").save(val_input / f"val_{stem}_0000.png")
        shutil.copy2(a.raw_root / "val_labels" / f"{stem}.png", val_gt / f"{stem}.png")

    manifest = {
        "experiment": "R310",
        "model": "standard one-channel nnU-Net 2.8.1",
        "train": 814,
        "val": 94,
        "prior_channel": False,
        "prior_loss": False,
        "instance_auxiliary": False,
        "source_unchanged": True,
        "clean_test_used": False,
    }
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
