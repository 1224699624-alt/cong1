#!/usr/bin/env python3
"""Visualize YOLO detection boxes on hand bone X-ray images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference and save detection-box visualizations.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/raw.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--weights", required=True, help="YOLO weights path, e.g. runs/.../weights/best.pt.")
    parser.add_argument("--out-dir", default=None, help="Output directory for visualization images.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument("--font-scale", type=float, default=0.55)
    parser.add_argument("--save-json", action="store_true", help="Save box coordinates and scores as JSON.")
    return parser.parse_args()


def find_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_image_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    names: dict[int, str],
    line_width: int,
    font_scale: float,
) -> np.ndarray:
    canvas = image.copy()
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
        color = (40, 220, 40)
        label = f"{names.get(int(class_id), str(int(class_id)))} {float(score):.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_w, text_h = text_size
        text_y = max(0, y1 - text_h - baseline - 3)
        cv2.rectangle(canvas, (x1, text_y), (x1 + text_w + 6, text_y + text_h + baseline + 4), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, text_y + text_h + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = parse_args()
    image_dir = Path(args.raw_root) / args.dataset / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split not found: {image_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / "yolo_boxes" / args.dataset / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    records = []

    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    for image_path in tqdm(find_images(image_dir), desc=f"yolo-vis/{args.dataset}/{args.split}"):
        image = read_image_bgr(image_path)
        result = model.predict(source=str(image_path), **predict_kwargs)[0]

        if result.boxes is None or len(result.boxes) == 0:
            boxes = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            class_ids = np.empty((0,), dtype=np.int64)
        else:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int64)

        visualization = draw_boxes(
            image=image,
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            names=names,
            line_width=args.line_width,
            font_scale=args.font_scale,
        )
        Image.fromarray(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)).save(out_dir / f"{image_path.stem}.jpg")

        records.append(
            {
                "image": image_path.name,
                "num_boxes": int(len(boxes)),
                "boxes_xyxy": boxes.round(2).tolist(),
                "scores": [round(float(score), 4) for score in scores.tolist()],
                "classes": [int(class_id) for class_id in class_ids.tolist()],
            }
        )

    if args.save_json:
        (out_dir / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved YOLO box visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
