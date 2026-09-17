#!/usr/bin/env python3
"""Prepare a tiny COCO instance dataset for the R190 MaskDINO smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--output-root", default="outputs/analysis/r190_maskdino_tiny_train_smoke/coco")
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--val-count", type=int, default=4)
    parser.add_argument("--max-size", type=int, default=384)
    return parser.parse_args()


def find_image(raw_root: Path, dataset: str, split: str, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        path = raw_root / dataset / split / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found for {split}/{stem}")


def resize_pair(image: Image.Image, label: Image.Image, max_size: int) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    scale = min(1.0, float(max_size) / float(max(width, height)))
    if scale >= 1.0:
        return image, label
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.BILINEAR), label.resize(new_size, Image.NEAREST)


def encode_binary_mask(binary: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def convert_split(raw_root: Path, dataset: str, split: str, count: int, max_size: int, out_root: Path) -> dict:
    label_paths = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))[:count]
    image_dir = out_root / split
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    ann_id = 1
    for image_id, label_path in enumerate(label_paths, start=1):
        image_path = find_image(raw_root, dataset, split, label_path.stem)
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        image, label = resize_pair(image, label, max_size)
        out_name = f"{label_path.stem}.jpg"
        image.save(image_dir / out_name, quality=95)
        label_arr = np.asarray(label)
        height, width = label_arr.shape[:2]
        images.append({
            "id": image_id,
            "file_name": out_name,
            "width": width,
            "height": height,
        })
        for value in sorted(int(v) for v in np.unique(label_arr) if int(v) > 0):
            binary = label_arr == value
            area = int(binary.sum())
            if area < 4:
                continue
            ys, xs = np.where(binary)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": area,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "segmentation": encode_binary_mask(binary),
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "epiphysis", "supercategory": "bone"}],
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, count in (("train", args.train_count), ("val", args.val_count)):
        payload = convert_split(raw_root, args.dataset, split, count, args.max_size, out_root)
        ann_path = out_root / f"{split}.json"
        ann_path.write_text(json.dumps(payload), encoding="utf-8")
        summary[split] = {
            "images": len(payload["images"]),
            "annotations": len(payload["annotations"]),
            "json": str(ann_path),
            "image_root": str(out_root / split),
        }
    summary_path = out_root.parent / "r190_coco_smoke_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

