#!/usr/bin/env python3
"""Validate a YOLO detection dataset generated from mask labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO image/label pairs and normalized box labels.")
    parser.add_argument("--data", required=True, help="YOLO data yaml path.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_label_file(label_path: Path) -> tuple[int, list[str]]:
    errors: list[str] = []
    box_count = 0
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path.name}:{line_number}: expected 5 fields, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path.name}:{line_number}: failed to parse numeric values")
            continue
        if class_id != 0:
            errors.append(f"{label_path.name}:{line_number}: class id must be 0, got {class_id}")
        if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{label_path.name}:{line_number}: normalized xywh values must be in [0, 1]")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{label_path.name}:{line_number}: width and height must be positive")
        box_count += 1
    return box_count, errors


def validate_split(base_path: Path, split_name: str, split_value: str) -> dict:
    image_dir = base_path / split_value
    label_dir = base_path / split_value.replace("images", "labels", 1)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    images = image_files(image_dir)
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    errors: list[str] = []
    box_count = 0
    empty_labels = 0
    unreadable_images = 0
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            unreadable_images += 1
            errors.append(f"{image_path.name}: unreadable image: {exc}")

    for label_path in labels:
        if label_path.read_text(encoding="utf-8").strip() == "":
            empty_labels += 1
        current_box_count, current_errors = validate_label_file(label_path)
        box_count += current_box_count
        errors.extend(current_errors)

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)
    errors.extend(f"{name}: missing label file" for name in missing_labels[:20])
    errors.extend(f"{name}: orphan label file" for name in orphan_labels[:20])

    return {
        "split": split_name,
        "images": len(images),
        "labels": len(labels),
        "boxes": box_count,
        "empty_labels": empty_labels,
        "unreadable_images": unreadable_images,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "errors": errors[:100],
        "ok": not errors and len(images) == len(labels),
    }


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base_path = Path(data["path"])
    report = {
        "data": str(data_yaml),
        "path": str(base_path),
        "names": data.get("names", {}),
        "splits": [],
    }
    for split_name in ("train", "val", "test"):
        if split_name in data:
            report["splits"].append(validate_split(base_path, split_name, data[split_name]))

    report["ok"] = all(split["ok"] for split in report["splits"])
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
