#!/usr/bin/env python3
"""Prepare an isolated nnU-Net v2 dataset for R202.

The conversion uses original TSRS_RSNA-Epiphysis train/val images and labels for
training/model selection, and clean-test-v2 image IDs for inference only.
Labels are converted to binary foreground masks because the paper claim is about
union epiphysis segmentation and R201 evaluates binary masks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare R202 nnU-Net v2 dataset.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--clean-label-root", default="data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels")
    parser.add_argument("--nnunet-root", default="outputs/nnunet/r202")
    parser.add_argument("--dataset-id", type=int, default=202)
    parser.add_argument("--dataset-name", default="TSRS_RSNAEpiphysis2D")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_image_path(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for {dataset}/{split}/{stem}")


def case_id(split: str, stem: str) -> str:
    return f"{split}_{stem}"


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("L").save(dst)


def write_binary_label(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(Image.open(src))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.uint8)
    Image.fromarray(mask, mode="L").save(dst)


def prepare_training_split(
    raw_root: Path,
    dataset: str,
    split: str,
    images_tr: Path,
    labels_tr: Path,
) -> list[str]:
    label_dir = raw_root / dataset / f"{split}_labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    cases: list[str] = []
    for label_path in sorted(label_dir.glob("*.png")):
        stem = label_path.stem
        cid = case_id(split, stem)
        src_image = find_image_path(raw_root, dataset, split, stem)
        copy_image(src_image, images_tr / f"{cid}_0000.png")
        write_binary_label(label_path, labels_tr / f"{cid}.png")
        cases.append(cid)
    return cases


def prepare_clean_test(
    raw_root: Path,
    dataset: str,
    clean_label_root: Path,
    images_ts: Path,
) -> list[str]:
    if not clean_label_root.exists():
        raise FileNotFoundError(f"Clean-test-v2 label directory not found: {clean_label_root}")
    cases: list[str] = []
    for label_path in sorted(clean_label_root.glob("*.png")):
        stem = label_path.stem
        src_image = find_image_path(raw_root, dataset, "test", stem)
        copy_image(src_image, images_ts / f"{stem}_0000.png")
        cases.append(stem)
    return cases


def main() -> None:
    args = parse_args()
    nnunet_root = Path(args.nnunet_root)
    raw_dataset = nnunet_root / "nnUNet_raw" / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    if raw_dataset.exists() and args.overwrite:
        resolved = raw_dataset.resolve()
        allowed_root = (nnunet_root / "nnUNet_raw").resolve()
        if not str(resolved).startswith(str(allowed_root)):
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(raw_dataset)
    raw_dataset.mkdir(parents=True, exist_ok=True)

    images_tr = raw_dataset / "imagesTr"
    labels_tr = raw_dataset / "labelsTr"
    images_ts = raw_dataset / "imagesTs"
    train_cases = prepare_training_split(Path(args.raw_root), args.dataset, "train", images_tr, labels_tr)
    val_cases = prepare_training_split(Path(args.raw_root), args.dataset, "val", images_tr, labels_tr)
    test_cases = prepare_clean_test(Path(args.raw_root), args.dataset, Path(args.clean_label_root), images_ts)

    dataset_json = {
        "channel_names": {"0": "xray"},
        "labels": {"background": 0, "epiphysis": 1},
        "numTraining": len(train_cases) + len(val_cases),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (raw_dataset / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")

    split = [{"train": train_cases, "val": val_cases}]
    split_path = nnunet_root / "splits_final_r202.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    summary = {
        "run_id": "R202",
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "raw_dataset": str(raw_dataset),
        "nnunet_root": str(nnunet_root),
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "clean_test_cases": len(test_cases),
        "split_file_to_copy_after_preprocess": str(split_path),
        "leakage_rule": "clean-test-v2 labels are used only to choose imagesTs case IDs; they are not copied into nnUNet training labels.",
        "expected_after_preprocess_split_path": str(
            nnunet_root
            / "nnUNet_preprocessed"
            / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
            / "splits_final.json"
        ),
    }
    summary_path = nnunet_root / "r202_nnunet_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
