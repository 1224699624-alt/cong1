#!/usr/bin/env python3
"""Evaluate predicted binary masks against instance PNG labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLO-SAM predicted masks.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", default=None, help="Metrics JSON path.")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def boundary(mask: np.ndarray, kernel: int) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(pil_mask.filter(ImageFilter.MaxFilter(kernel))) > 0
    eroded = np.asarray(pil_mask.filter(ImageFilter.MinFilter(kernel))) > 0
    return np.logical_xor(dilated, eroded)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 1.0


def compute_metrics(pred: np.ndarray, gt: np.ndarray, boundary_kernel: int) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    dice = safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    iou = safe_divide(tp, tp + fp + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    boundary_iou = 0.0
    if boundary_kernel > 0:
        pred_boundary = boundary(pred, boundary_kernel)
        gt_boundary = boundary(gt, boundary_kernel)
        boundary_intersection = float(np.logical_and(pred_boundary, gt_boundary).sum())
        boundary_union = float(np.logical_or(pred_boundary, gt_boundary).sum())
        boundary_iou = safe_divide(boundary_intersection, boundary_union)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "boundary_iou": boundary_iou,
    }


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    metric_names = [key for key in records[0] if key != "image"]
    return {
        metric: float(np.mean([record[metric] for record in records]))
        for metric in metric_names
    }


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    pred_dir = Path(args.pred_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth label dir not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")

    records: list[dict[str, float]] = []
    missing = []
    for gt_path in tqdm(sorted(gt_dir.glob("*.png")), desc=f"eval/{args.dataset}/{args.split}"):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            missing.append(gt_path.name)
            continue

        gt = read_binary_mask(gt_path)
        pred = read_binary_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0

        metrics = compute_metrics(pred, gt, args.boundary_kernel)
        metrics["image"] = gt_path.name
        records.append(metrics)

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "num_evaluated": len(records),
        "num_missing": len(missing),
        "missing": missing[:50],
        "mean": mean_metrics(records),
        "per_image": records,
    }

    output = Path(args.output) if args.output else pred_dir.parent / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2))
    print(f"Saved metrics: {output}")


if __name__ == "__main__":
    main()
