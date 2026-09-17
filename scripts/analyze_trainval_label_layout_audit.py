#!/usr/bin/env python3
"""Audit train/val label-layout outliers without touching the dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit train/val label layout outliers.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--anchor-val-pred-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--min-component-area", type=int, default=4)
    parser.add_argument("--outlier-threshold", type=float, default=6.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def robust_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {"median": 0.0, "mad": 1.0, "mean": 0.0, "std": 1.0}
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    std = float(arr.std())
    return {
        "median": median,
        "mad": max(mad, 1e-6),
        "mean": float(arr.mean()),
        "std": max(std, 1e-6),
    }


def component_features(mask: np.ndarray, min_area: int) -> dict[str, float]:
    height, width = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = float(stats[idx, cv2.CC_STAT_LEFT])
        y = float(stats[idx, cv2.CC_STAT_TOP])
        w = float(stats[idx, cv2.CC_STAT_WIDTH])
        h = float(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        comps.append({
            "area": float(area),
            "area_frac": float(area) / float(height * width),
            "cx": float(cx) / float(width),
            "cy": float(cy) / float(height),
            "w": w / float(width),
            "h": h / float(height),
            "aspect": w / max(h, 1.0),
            "thickness": min(w, h) / float(max(height, width)),
        })
    areas = np.asarray([c["area_frac"] for c in comps], dtype=np.float32)
    cxs = np.asarray([c["cx"] for c in comps], dtype=np.float32)
    cys = np.asarray([c["cy"] for c in comps], dtype=np.float32)
    aspects = np.asarray([c["aspect"] for c in comps], dtype=np.float32)
    thickness = np.asarray([c["thickness"] for c in comps], dtype=np.float32)
    count = len(comps)
    if count == 0:
        return {
            "component_count": 0.0,
            "fg_frac": float(mask.mean()),
            "area_median": 0.0,
            "area_iqr": 0.0,
            "area_max": 0.0,
            "cx_span": 0.0,
            "cy_span": 0.0,
            "cy_median": 0.0,
            "aspect_median": 0.0,
            "thickness_median": 0.0,
            "layout_width": 0.0,
            "layout_height": 0.0,
        }
    return {
        "component_count": float(count),
        "fg_frac": float(mask.mean()),
        "area_median": float(np.median(areas)),
        "area_iqr": float(np.percentile(areas, 75) - np.percentile(areas, 25)),
        "area_max": float(areas.max()),
        "cx_span": float(cxs.max() - cxs.min()),
        "cy_span": float(cys.max() - cys.min()),
        "cy_median": float(np.median(cys)),
        "aspect_median": float(np.median(aspects)),
        "thickness_median": float(np.median(thickness)),
        "layout_width": float((cxs + 0.0).max() - (cxs + 0.0).min()),
        "layout_height": float((cys + 0.0).max() - (cys + 0.0).min()),
    }


def collect_split(raw_root: Path, dataset: str, split: str, min_area: int) -> list[dict[str, object]]:
    label_dir = raw_root / dataset / f"{split}_labels"
    records: list[dict[str, object]] = []
    for path in sorted(label_dir.glob("*.png")):
        mask = read_mask(path)
        features = component_features(mask, min_area)
        records.append({"image": path.name, "split": split, "features": features})
    return records


def score_records(records: list[dict[str, object]], train_stats: dict[str, dict[str, float]]) -> None:
    feature_names = sorted(train_stats)
    for rec in records:
        features = rec["features"]  # type: ignore[assignment]
        z_by_feature = {}
        abs_z = []
        for name in feature_names:
            value = float(features[name])
            stats = train_stats[name]
            z = 0.6745 * (value - stats["median"]) / stats["mad"]
            z_by_feature[name] = float(z)
            abs_z.append(abs(float(z)))
        top = sorted(z_by_feature.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        rec["layout_outlier_score"] = float(np.mean(sorted(abs_z, reverse=True)[:3])) if abs_z else 0.0
        rec["top_outlier_features"] = [{"feature": k, "robust_z": v} for k, v in top]


def add_anchor_val_metrics(
    val_records: list[dict[str, object]],
    raw_root: Path,
    dataset: str,
    pred_dir: Path,
    boundary_kernel: int,
) -> None:
    label_dir = raw_root / dataset / "val_labels"
    for rec in val_records:
        name = str(rec["image"])
        pred_path = pred_dir / name
        if not pred_path.exists():
            rec["anchor_missing"] = True
            continue
        gt = read_mask(label_dir / name)
        pred = read_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        metrics = compute_metrics(pred, gt, boundary_kernel)
        pred_features = component_features(pred, min_area=4)
        gt_features = rec["features"]  # type: ignore[assignment]
        metrics["pred_component_count"] = float(pred_features["component_count"])
        metrics["gt_component_count"] = float(gt_features["component_count"])
        metrics["component_count_error"] = abs(metrics["pred_component_count"] - metrics["gt_component_count"])
        metrics["false_bridge_flag"] = float(
            metrics["pred_component_count"] < metrics["gt_component_count"]
            and float(pred.mean()) >= 0.90 * float(gt.mean())
        )
        rec["anchor_metrics"] = metrics


def summarize_split(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {"count": 0}
    feature_names = sorted(records[0]["features"])  # type: ignore[index]
    features = {name: [float(rec["features"][name]) for rec in records] for name in feature_names}  # type: ignore[index]
    return {
        "count": len(records),
        "features": {
            name: {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for name, values in features.items()
        },
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    train_records = collect_split(raw_root, args.dataset, "train", args.min_component_area)
    val_records = collect_split(raw_root, args.dataset, "val", args.min_component_area)
    if not train_records:
        raise ValueError("no train labels found")
    feature_names = sorted(train_records[0]["features"])  # type: ignore[index]
    train_stats = {
        name: robust_stats([float(rec["features"][name]) for rec in train_records])  # type: ignore[index]
        for name in feature_names
    }
    score_records(train_records, train_stats)
    score_records(val_records, train_stats)
    if args.anchor_val_pred_dir:
        add_anchor_val_metrics(
            val_records,
            raw_root,
            args.dataset,
            Path(args.anchor_val_pred_dir),
            args.boundary_kernel,
        )

    train_suspects = sorted(train_records, key=lambda rec: float(rec["layout_outlier_score"]), reverse=True)[: args.top_k]
    val_layout_suspects = sorted(val_records, key=lambda rec: float(rec["layout_outlier_score"]), reverse=True)[: args.top_k]
    val_anchor_hard = sorted(
        [rec for rec in val_records if "anchor_metrics" in rec],
        key=lambda rec: float(rec["anchor_metrics"]["dice"]),  # type: ignore[index]
    )[: args.top_k]
    manifest_excludes = {
        "train": {
            str(rec["image"]).rsplit(".", 1)[0]: {
                "reason": "train_label_layout_outlier",
                "layout_outlier_score": float(rec["layout_outlier_score"]),
                "top_features": rec["top_outlier_features"],
            }
            for rec in train_records
            if float(rec["layout_outlier_score"]) >= args.outlier_threshold
        },
        "val": {},
        "test": {},
    }
    output = {
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "min_component_area": args.min_component_area,
        "outlier_threshold": args.outlier_threshold,
        "train_summary": summarize_split(train_records),
        "val_summary": summarize_split(val_records),
        "train_stats": train_stats,
        "train_suspects": train_suspects,
        "val_layout_suspects": val_layout_suspects,
        "val_anchor_hard": val_anchor_hard,
        "manifest_candidate": {
            "source_dataset": args.dataset,
            "target_dataset": f"{args.dataset}_layout_audit_filtered_v1",
            "excludes": manifest_excludes,
            "notes": "Draft only: generated from train layout outliers without clean-test-v2 tuning.",
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(output["manifest_candidate"], indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out_path),
        "num_train": len(train_records),
        "num_val": len(val_records),
        "manifest_train_excludes": len(manifest_excludes["train"]),
        "top_train_suspects": [
            {
                "image": rec["image"],
                "score": rec["layout_outlier_score"],
                "top": rec["top_outlier_features"][:3],
            }
            for rec in train_suspects[:10]
        ],
        "top_val_hard": [
            {
                "image": rec["image"],
                "dice": rec.get("anchor_metrics", {}).get("dice"),
                "score": rec["layout_outlier_score"],
            }
            for rec in val_anchor_hard[:10]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
