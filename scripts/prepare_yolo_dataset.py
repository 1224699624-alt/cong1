#!/usr/bin/env python3
"""Convert instance masks into a YOLO detection dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO boxes from hand bone instance masks.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/raw.")
    parser.add_argument("--raw-root", default="data/raw", help="Root containing raw TSRS_RSNA datasets.")
    parser.add_argument("--out-root", default="data/yolo", help="Output root for YOLO formatted data.")
    parser.add_argument("--config-dir", default="configs", help="Directory for generated data yaml.")
    parser.add_argument("--padding-ratio", type=float, default=0.08, help="Box expansion ratio on each side.")
    parser.add_argument("--min-area", type=int, default=12, help="Ignore mask instances smaller than this area.")
    parser.add_argument("--copy-images", action="store_true", default=True, help="Copy images into YOLO folders.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labels and copied images.")
    return parser.parse_args()


def find_image_files(split_dir: Path) -> list[Path]:
    files: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        files.extend(split_dir.glob(f"*{extension}"))
    return sorted(files)


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
        max(0.0, min(float(width - 1), x2)),
        max(0.0, min(float(height - 1), y2)),
    )


def mask_to_yolo_boxes(mask_path: Path, padding_ratio: float, min_area: int) -> list[str]:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]

    height, width = mask.shape[:2]
    label_lines: list[str] = []
    for instance_id in sorted(int(value) for value in np.unique(mask) if int(value) > 0):
        ys, xs = np.where(mask == instance_id)
        if xs.size < min_area:
            continue

        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        box_width = max(1.0, x2 - x1 + 1.0)
        box_height = max(1.0, y2 - y1 + 1.0)
        pad_x = box_width * padding_ratio
        pad_y = box_height * padding_ratio
        x1, y1, x2, y2 = clip_box(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, width, height)

        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        norm_width = max(1.0, x2 - x1 + 1.0) / width
        norm_height = max(1.0, y2 - y1 + 1.0) / height
        label_lines.append(f"0 {center_x:.8f} {center_y:.8f} {norm_width:.8f} {norm_height:.8f}")
    return label_lines


def copy_image(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare_split(raw_dataset: Path, out_dataset: Path, split: str, args: argparse.Namespace) -> tuple[int, int]:
    image_dir = raw_dataset / split
    mask_dir = raw_dataset / f"{split}_labels"
    out_image_dir = out_dataset / "images" / split
    out_label_dir = out_dataset / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = find_image_files(image_dir)
    missing_masks = 0
    total_boxes = 0
    for image_path in tqdm(image_files, desc=f"{raw_dataset.name}/{split}"):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            missing_masks += 1
            continue

        if args.copy_images:
            copy_image(image_path, out_image_dir / image_path.name, args.overwrite)

        label_lines = mask_to_yolo_boxes(mask_path, args.padding_ratio, args.min_area)
        total_boxes += len(label_lines)
        label_path = out_label_dir / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            continue
        label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    if missing_masks:
        print(f"Warning: {missing_masks} images in {image_dir} have no matching mask.")
    return len(image_files), total_boxes


def write_data_yaml(dataset_name: str, out_dataset: Path, config_dir: Path) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir / f"{dataset_name}.yaml"
    payload = {
        "path": str(out_dataset.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "bone_region"},
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    raw_dataset = Path(args.raw_root) / args.dataset
    out_dataset = Path(args.out_root) / args.dataset

    if not raw_dataset.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_dataset}")

    summary = {}
    for split in SPLITS:
        image_count, box_count = prepare_split(raw_dataset, out_dataset, split, args)
        summary[split] = {"images": image_count, "boxes": box_count}

    yaml_path = write_data_yaml(args.dataset, out_dataset, Path(args.config_dir))
    print(f"Wrote YOLO yaml: {yaml_path}")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
