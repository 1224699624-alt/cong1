#!/usr/bin/env python3
"""Analyze per-image layout/component failure modes for clean-test-v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze layout failure modes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def components(mask: np.ndarray) -> list[dict[str, float]]:
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out: list[dict[str, float]] = []
    height, width = mask.shape
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        x = float(stats[idx, cv2.CC_STAT_LEFT])
        y = float(stats[idx, cv2.CC_STAT_TOP])
        w = float(stats[idx, cv2.CC_STAT_WIDTH])
        h = float(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        out.append({
            "area": area,
            "area_frac": area / float(height * width),
            "cx_norm": float(cx) / float(width),
            "cy_norm": float(cy) / float(height),
            "w_norm": w / float(width),
            "h_norm": h / float(height),
            "aspect": w / max(h, 1.0),
        })
    return sorted(out, key=lambda item: (item["cy_norm"], item["cx_norm"]))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    pred_dir = Path(args.pred_dir)
    records = []
    for gt_path in sorted(gt_dir.glob("*.png")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = read_mask(gt_path)
        pred = read_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        metrics = compute_metrics(pred, gt, args.boundary_kernel)
        gt_components = components(gt)
        pred_components = components(pred)
        rec = {
            "image": gt_path.name,
            **metrics,
            "gt_component_count": len(gt_components),
            "pred_component_count": len(pred_components),
            "component_count_error": abs(len(pred_components) - len(gt_components)),
            "component_delta": len(pred_components) - len(gt_components),
            "gt_area_frac": float(gt.mean()),
            "pred_area_frac": float(pred.mean()),
            "area_frac_delta": float(pred.mean() - gt.mean()),
            "precision_minus_recall": float(metrics["precision"] - metrics["recall"]),
            "gt_components": gt_components,
            "pred_components": pred_components,
        }
        records.append(rec)

    low_dice = sorted(records, key=lambda r: r["dice"])[:20]
    precision_limited = sorted(records, key=lambda r: r["precision_minus_recall"])[:20]
    recall_limited = sorted(records, key=lambda r: r["precision_minus_recall"], reverse=True)[:20]
    bridge_like = [
        r for r in records
        if r["pred_component_count"] < r["gt_component_count"] and r["pred_area_frac"] >= 0.9 * r["gt_area_frac"]
    ]
    missing_component_like = [
        r for r in records
        if r["pred_component_count"] < r["gt_component_count"] and r["recall"] < 0.92
    ]
    overmask_like = [r for r in records if r["precision"] < 0.88 and r["recall"] > 0.92]

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "pred_dir": str(pred_dir),
        "num_records": len(records),
        "mean": {k: float(np.mean([r[k] for r in records])) for k in [
            "dice", "iou", "precision", "recall", "boundary_iou",
            "gt_component_count", "pred_component_count", "component_count_error",
            "area_frac_delta", "precision_minus_recall",
        ]},
        "component_count_error": summarize([float(r["component_count_error"]) for r in records]),
        "low_dice": low_dice,
        "precision_limited": precision_limited,
        "recall_limited": recall_limited,
        "bridge_like": bridge_like[:30],
        "missing_component_like": missing_component_like[:30],
        "overmask_like": overmask_like[:30],
        "counts": {
            "bridge_like": len(bridge_like),
            "missing_component_like": len(missing_component_like),
            "overmask_like": len(overmask_like),
            "dice_below_090": sum(1 for r in records if r["dice"] < 0.90),
            "dice_below_085": sum(1 for r in records if r["dice"] < 0.85),
            "boundary_below_020": sum(1 for r in records if r["boundary_iou"] < 0.20),
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "num_records": len(records),
        "mean": summary["mean"],
        "counts": summary["counts"],
        "low_dice": [r["image"] for r in low_dice[:10]],
    }, indent=2))


if __name__ == "__main__":
    main()
