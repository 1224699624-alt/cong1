#!/usr/bin/env python3
"""Apply anatomy-layout watershed cuts to bridge-like mask components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair bridge-like components with watershed cuts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-mask-dir", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--min-component-area", type=int, default=24)
    parser.add_argument("--large-area-percentile", type=float, default=65.0)
    parser.add_argument("--peak-thresholds", default="0.35,0.45,0.55")
    parser.add_argument("--cut-dilations", default="0,1")
    parser.add_argument("--max-remove-fracs", default="0.0015,0.003,0.005")
    parser.add_argument("--fixed-peak-threshold", type=float, default=None)
    parser.add_argument("--fixed-cut-dilation", type=int, default=None)
    parser.add_argument("--fixed-max-remove-frac", type=float, default=None)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0)


def add_structure_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def component_areas(mask: np.ndarray) -> list[int]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0]


def watershed_cut_component(component: np.ndarray, peak_threshold: float, cut_dilation: int, min_area: int) -> np.ndarray:
    if int(component.sum()) < max(min_area * 2, 8):
        return component
    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 1.0:
        return component
    peaks = (dist >= float(peak_threshold) * float(dist.max())).astype(np.uint8)
    peaks = cv2.erode(peaks, np.ones((3, 3), np.uint8), iterations=1)
    n, markers = cv2.connectedComponents(peaks)
    if n <= 2:
        return component

    markers = markers.astype(np.int32)
    markers[~component] = -1
    height, width = component.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[component] = 255
    cv2.watershed(canvas, markers)
    cut = (markers == -1) & component
    if cut_dilation > 0:
        cut = cv2.dilate(cut.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(cut_dilation)) > 0
        cut &= component

    repaired = component & ~cut
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(repaired.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(component, dtype=bool)
    kept_count = 0
    for idx in range(1, n2):
        if int(stats2[idx, cv2.CC_STAT_AREA]) >= min_area:
            kept |= labels2 == idx
            kept_count += 1
    if kept_count < 2:
        return component
    return kept


def repair_mask(mask: np.ndarray, peak_threshold: float, cut_dilation: int, max_remove_frac: float, min_area: int, large_area_percentile: float) -> np.ndarray:
    areas = component_areas(mask)
    if not areas:
        return mask.copy()
    large_cutoff = max(float(min_area * 2), float(np.percentile(np.asarray(areas, dtype=np.float32), large_area_percentile)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    repaired = np.zeros_like(mask, dtype=bool)
    removed_total = 0
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        component = labels == idx
        if area < large_cutoff:
            repaired |= component
            continue
        cut_component = watershed_cut_component(component, peak_threshold, cut_dilation, min_area)
        removed = int(component.sum() - cut_component.sum())
        if removed <= int(max_remove_frac * max(1, int(mask.sum()))):
            repaired |= cut_component
            removed_total += removed
        else:
            repaired |= component
    return repaired


def evaluate_config(gt_paths: list[Path], pred_dir: Path, args: argparse.Namespace, peak: float, dilation: int, max_remove: float, write_dir: Path | None) -> dict[str, object]:
    records: list[dict[str, float]] = []
    for gt_path in gt_paths:
        gt = read_mask(gt_path)
        pred = read_mask(pred_dir / gt_path.name)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        repaired = repair_mask(pred, peak, dilation, max_remove, args.min_component_area, args.large_area_percentile)
        if write_dir is not None:
            write_mask(write_dir / gt_path.name, repaired)
        rec = compute_metrics(repaired, gt, args.boundary_kernel)
        rec = add_structure_metrics(rec, repaired, gt)
        rec["image"] = gt_path.name
        records.append(rec)
    return {
        "peak_threshold": peak,
        "cut_dilation": dilation,
        "max_remove_frac": max_remove,
        "mean": mean_metrics(records),
        "per_image": records,
    }


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    pred_dir = Path(args.pred_dir)
    gt_paths = sorted(gt_dir.glob("*.png"))
    if args.grid_search:
        peaks = [float(x) for x in args.peak_thresholds.split(",") if x.strip()]
        dilations = [int(x) for x in args.cut_dilations.split(",") if x.strip()]
        max_removes = [float(x) for x in args.max_remove_fracs.split(",") if x.strip()]
        grid = []
        for peak in peaks:
            for dilation in dilations:
                for max_remove in max_removes:
                    grid.append(evaluate_config(gt_paths, pred_dir, args, peak, dilation, max_remove, None))
        best = max(grid, key=lambda item: float(item["mean"]["dice"]))
        summary = {"dataset": args.dataset, "split": args.split, "grid": grid, "best": best}
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"best": {k: best[k] for k in ["peak_threshold", "cut_dilation", "max_remove_frac"]}, "mean": best["mean"]}, indent=2))
        return

    if args.fixed_peak_threshold is None or args.fixed_cut_dilation is None or args.fixed_max_remove_frac is None:
        raise ValueError("Fixed mode requires --fixed-peak-threshold, --fixed-cut-dilation, and --fixed-max-remove-frac.")
    summary = evaluate_config(
        gt_paths,
        pred_dir,
        args,
        float(args.fixed_peak_threshold),
        int(args.fixed_cut_dilation),
        float(args.fixed_max_remove_frac),
        Path(args.output_mask_dir),
    )
    summary["dataset"] = args.dataset
    summary["split"] = args.split
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
