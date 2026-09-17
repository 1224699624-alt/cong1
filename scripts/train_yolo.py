#!/usr/bin/env python3
"""Train a YOLO detector for SAM box prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on generated hand bone boxes.")
    parser.add_argument("--data", required=True, help="YOLO data yaml path.")
    parser.add_argument("--model", default="yolov8l.pt", help="YOLO base model or checkpoint.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0, or cpu.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="hand_bone_yolo")
    parser.add_argument("--project", default="outputs/yolo")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"YOLO data yaml not found: {data_path}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "patience": args.patience,
        "resume": args.resume,
        "task": "detect",
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
