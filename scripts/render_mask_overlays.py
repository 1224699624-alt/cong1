#!/usr/bin/env python3
"""Render binary mask overlays on source images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render overlays for predicted binary masks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--yolo-weights", default=None, help="Optional YOLO weights to render detection boxes.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--box-padding-ratio", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def find_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        files.extend(image_dir.glob(f"*{extension}"))
    return sorted(files)


def read_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_binary_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def expand_boxes(boxes: np.ndarray, image_shape: tuple[int, int], padding_ratio: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)

    height, width = image_shape
    expanded = boxes.astype(np.float32).copy()
    box_widths = expanded[:, 2] - expanded[:, 0] + 1.0
    box_heights = expanded[:, 3] - expanded[:, 1] + 1.0
    expanded[:, 0] -= box_widths * padding_ratio
    expanded[:, 2] += box_widths * padding_ratio
    expanded[:, 1] -= box_heights * padding_ratio
    expanded[:, 3] += box_heights * padding_ratio
    expanded[:, [0, 2]] = np.clip(expanded[:, [0, 2]], 0, width - 1)
    expanded[:, [1, 3]] = np.clip(expanded[:, [1, 3]], 0, height - 1)
    return expanded


def main() -> None:
    args = parse_args()
    image_dir = Path(args.raw_root) / args.dataset / args.split
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(args.yolo_weights) if args.yolo_weights else None

    for image_path in tqdm(find_image_files(image_dir), desc=f"overlay/{args.dataset}/{args.split}"):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue
        image_rgb = read_rgb(image_path)
        mask = read_binary_mask(mask_path)
        if mask.shape != image_rgb.shape[:2]:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(image_rgb.shape[:2][::-1], Image.NEAREST)) > 0
        overlay = image_rgb.copy()
        overlay[mask] = ((1.0 - args.alpha) * overlay[mask] + args.alpha * np.array([255, 40, 40])).astype(np.uint8)
        if yolo is not None:
            result = yolo.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False,
                device=args.device,
            )[0]
            boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
            boxes = expand_boxes(boxes.astype(np.float32).reshape(-1, 4), image_rgb.shape[:2], args.box_padding_ratio)
            for x1, y1, x2, y2 in boxes.astype(int):
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 220, 60), 2)
        Image.fromarray(overlay).save(output_dir / f"{image_path.stem}.jpg")


if __name__ == "__main__":
    main()
