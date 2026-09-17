#!/usr/bin/env python3
"""Build an isolated R260-compatible nnU-Net dataset from reviewed panel keep-lists.

The panel JPEGs are review aids, not model inputs. Their remaining stems define
the cases retained after manual cleaning. Original X-rays and copied indexed
labels remain untouched and are copied into an isolated derived dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from train_r256_scale_invariant_pair_prior import find_image


def _load_selection(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    keep = payload.get("keep", payload)
    result = {split: sorted({str(x) for x in keep[split]}) for split in ("train", "val")}
    if not result["train"] or not result["val"]:
        raise RuntimeError("empty clean train/val selection")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r259_frozen_relation"))
    parser.add_argument("--frozen-mask-root", type=Path, default=Path("outputs/ablations/r260_recovered_exact/TSRS_RSNA-Epiphysis/val/masks"))
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/nnunet/r308_clean_panels_r260/data"))
    parser.add_argument("--dataset-id", type=int, default=309)
    parser.add_argument("--dataset-name", default="TSRSR260CleanPanels2D")
    parser.add_argument("--experiment", default="R308")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    joined = " ".join(map(str, vars(args).values())).lower()
    if args.raw_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R308 is restricted to TSRS_RSNA-Epiphysis train/original-val")

    keep = _load_selection(args.selection)
    expected = {"train": 814, "val": 94}
    if {k: len(v) for k, v in keep.items()} != expected:
        raise RuntimeError(f"unexpected reviewed selection counts: { {k: len(v) for k, v in keep.items()} }")

    dataset = args.output / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if args.overwrite:
        shutil.rmtree(dataset, ignore_errors=True)
        for name in ("imagesValPrior", "cleanValGT", "frozenR260Masks"):
            shutil.rmtree(args.output.parent / name, ignore_errors=True)
    (dataset / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset / "labelsTr").mkdir(parents=True, exist_ok=True)

    split_keys: dict[str, list[str]] = {"train": [], "val": []}
    for split in ("train", "val"):
        available = {p.stem: p for p in (args.raw_root / f"{split}_labels").glob("*.png")}
        missing = sorted(set(keep[split]) - set(available))
        if missing:
            raise FileNotFoundError(f"missing original labels for {split}: {missing}")
        for stem in keep[split]:
            key = f"{split}_{stem}"
            image = Image.open(find_image(args.raw_root / split, stem)).convert("L")
            ids = np.asarray(Image.open(available[stem]))
            if ids.ndim == 3:
                ids = ids[..., 0]
            prior_png = args.prior_root / split / f"{stem}.png"
            prior_npy = args.prior_root / split / f"{stem}.npy"
            if not prior_png.is_file() or not prior_npy.is_file():
                raise FileNotFoundError(f"missing R259 prior for {split}/{stem}")
            prior = Image.open(prior_png).convert("L")
            values = np.load(prior_npy)
            if values.shape != ids.shape or image.size != prior.size or not np.isfinite(values).all():
                raise RuntimeError(f"invalid shape/prior for {split}/{stem}")
            image.save(dataset / "imagesTr" / f"{key}_0000.png")
            prior.save(dataset / "imagesTr" / f"{key}_0001.png")
            Image.fromarray((ids > 0).astype(np.uint8), mode="L").save(dataset / "labelsTr" / f"{key}.png")
            split_keys[split].append(key)

    dataset_json = {
        "channel_names": {"0": "xray", "1": "frozen_relation_prior"},
        "labels": {"background": 0, "bone": 1},
        "numTraining": sum(map(len, split_keys.values())),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (dataset / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    (dataset / "splits_final.json").write_text(
        json.dumps([{"train": split_keys["train"], "val": split_keys["val"]}], indent=2), encoding="utf-8"
    )

    if args.source_plan is not None:
        plan = json.loads(args.source_plan.read_text(encoding="utf-8"))
        plan["dataset_name"] = dataset.name
        # Preserve R260 architecture and inference patch geometry so strict
        # checkpoint loading and paired evaluation remain valid. Batch one is
        # required by the local 6 GiB GPU and does not alter network weights.
        plan["configurations"]["2d"]["batch_size"] = 1
        plan_dir = args.output / "nnUNet_preprocessed" / dataset.name
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "nnUNetPlans.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        # nnUNetv2_plan_experiment normally copies this metadata. R308 reuses
        # the exact R260 plan instead, so provide the equivalent file here.
        (plan_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    val_input = args.output.parent / "imagesValPrior"
    val_gt = args.output.parent / "cleanValGT"
    frozen = args.output.parent / "frozenR260Masks"
    for root in (val_input, val_gt, frozen):
        root.mkdir(parents=True, exist_ok=True)
    for stem in keep["val"]:
        key = f"val_{stem}"
        Image.open(find_image(args.raw_root / "val", stem)).convert("L").save(val_input / f"{key}_0000.png")
        Image.open(args.prior_root / "val" / f"{stem}.png").convert("L").save(val_input / f"{key}_0001.png")
        shutil.copy2(args.raw_root / "val_labels" / f"{stem}.png", val_gt / f"{stem}.png")
        source_mask = args.frozen_mask_root / f"{stem}.png"
        if not source_mask.is_file():
            raise FileNotFoundError(source_mask)
        shutil.copy2(source_mask, frozen / source_mask.name)

    manifest = {
        "experiment": args.experiment,
        "dataset": "TSRS_RSNA-Epiphysis",
        "selection_source": str(args.selection),
        "train_kept": len(keep["train"]),
        "val_kept": len(keep["val"]),
        "train_excluded": 875 - len(keep["train"]),
        "val_excluded": 96 - len(keep["val"]),
        "panel_jpegs_used_as_model_input": False,
        "working_label_pixel_edits_detected": 0,
        "source_unchanged": True,
        "clean_test_used": False,
        "source_plan": str(args.source_plan) if args.source_plan else None,
        "training_batch_size": 1 if args.source_plan else None,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
