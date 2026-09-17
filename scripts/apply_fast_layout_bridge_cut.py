#!/usr/bin/env python3
"""Fast geometry-only bridge cuts for epiphysis layout repair.

This is a deliberately small diagnostic: it cuts only thin low-distance
pixels in large connected components that contain multiple distance peaks.
Parameters should be selected on a proxy split, then applied once to the
target split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply fast layout bridge cuts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-mask-dir", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--peak-fracs", default="0.45,0.55,0.65")
    parser.add_argument("--valley-fracs", default="0.16,0.22,0.28")
    parser.add_argument("--min-areas", default="18,32")
    parser.add_argument("--max-remove-fracs", default="0.001,0.002,0.004")
    parser.add_argument("--fixed-peak-frac", type=float, default=None)
    parser.add_argument("--fixed-valley-frac", type=float, default=None)
    parser.add_argument("--fixed-min-area", type=int, default=None)
    parser.add_argument("--fixed-max-remove-frac", type=float, default=None)
    parser.add_argument("--large-area-percentile", type=float, default=60.0)
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
    record["gt_component_count"] = float(component_count(gt))
    record["component_count_error"] = float(abs(pred_count - int(record["gt_component_count"])))
    record["false_bridge_flag"] = float(pred_count < int(record["gt_component_count"]) and pred.sum() >= gt.sum() * 0.90)
    return record


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def area_cutoff(mask: np.ndarray, percentile: float, min_area: int) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0]
    if not areas:
        return float(min_area * 2)
    return max(float(min_area * 2), float(np.percentile(np.asarray(areas, dtype=np.float32), percentile)))


def local_maxima(dist: np.ndarray, component: np.ndarray, peak_frac: float, min_area: int) -> np.ndarray:
    if float(dist.max()) <= 0:
        return np.zeros_like(component, dtype=np.int32)
    peaks = (dist >= float(peak_frac) * float(dist.max())) & component
    peaks = cv2.erode(peaks.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
    markers = np.zeros_like(labels, dtype=np.int32)
    kept = 0
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= max(1, min_area // 4):
            kept += 1
            markers[labels == idx] = kept
    return markers


def repair_component(component: np.ndarray, peak_frac: float, valley_frac: float, min_area: int) -> np.ndarray:
    if int(component.sum()) < min_area * 2:
        return component
    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    markers = local_maxima(dist, component, peak_frac, min_area)
    if int(markers.max()) < 2:
        return component
    low_dist = (dist <= float(valley_frac) * float(dist.max())) & component
    if not low_dist.any():
        return component
    # Keep only thin low-distance pixels that behave like internal separators:
    # they must be surrounded by foreground on more than one side.
    neighbor_count = cv2.filter2D(component.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
    cut = low_dist & (neighbor_count >= 4)
    if not cut.any():
        return component

    repaired = component & ~cut
    n, labels, stats, _ = cv2.connectedComponentsWithStats(repaired.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(component, dtype=bool)
    kept_count = 0
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            kept |= labels == idx
            kept_count += 1
    if kept_count < 2:
        return component
    return kept


def repair_mask(mask: np.ndarray, peak_frac: float, valley_frac: float, min_area: int, max_remove_frac: float, large_area_percentile: float) -> np.ndarray:
    cutoff = area_cutoff(mask, large_area_percentile, min_area)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    repaired = np.zeros_like(mask, dtype=bool)
    max_remove = int(float(max_remove_frac) * max(1, int(mask.sum())))
    for idx in range(1, n):
        component = labels == idx
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < cutoff:
            repaired |= component
            continue
        candidate = repair_component(component, peak_frac, valley_frac, min_area)
        removed = int(component.sum() - candidate.sum())
        if 0 <= removed <= max_remove:
            repaired |= candidate
        else:
            repaired |= component
    return repaired


def evaluate_config(
    gt_paths: list[Path],
    pred_dir: Path,
    args: argparse.Namespace,
    peak_frac: float,
    valley_frac: float,
    min_area: int,
    max_remove_frac: float,
    write_dir: Path | None,
) -> dict[str, object]:
    records: list[dict[str, float]] = []
    changed = 0
    for gt_path in gt_paths:
        gt = read_mask(gt_path)
        pred = read_mask(pred_dir / gt_path.name)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        repaired = repair_mask(pred, peak_frac, valley_frac, min_area, max_remove_frac, args.large_area_percentile)
        changed += int(not np.array_equal(pred, repaired))
        if write_dir is not None:
            write_mask(write_dir / gt_path.name, repaired)
        rec = compute_metrics(repaired, gt, args.boundary_kernel)
        rec = add_structure_metrics(rec, repaired, gt)
        rec["image"] = gt_path.name
        records.append(rec)
    return {
        "peak_frac": peak_frac,
        "valley_frac": valley_frac,
        "min_area": min_area,
        "max_remove_frac": max_remove_frac,
        "changed_images": changed,
        "mean": mean_metrics(records),
        "per_image": records,
    }


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    pred_dir = Path(args.pred_dir)
    gt_paths = sorted(gt_dir.glob("*.png"))
    if args.grid_search:
        grid = []
        for peak in parse_float_list(args.peak_fracs):
            for valley in parse_float_list(args.valley_fracs):
                for min_area in parse_int_list(args.min_areas):
                    for max_remove in parse_float_list(args.max_remove_fracs):
                        grid.append(evaluate_config(gt_paths, pred_dir, args, peak, valley, min_area, max_remove, None))
        best = max(grid, key=lambda item: float(item["mean"]["dice"]))
        summary = {"dataset": args.dataset, "split": args.split, "grid": grid, "best": best}
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "best": {k: best[k] for k in ["peak_frac", "valley_frac", "min_area", "max_remove_frac", "changed_images"]},
            "mean": best["mean"],
        }, indent=2))
        return

    required = [args.fixed_peak_frac, args.fixed_valley_frac, args.fixed_min_area, args.fixed_max_remove_frac]
    if any(value is None for value in required):
        raise ValueError("Fixed mode requires all fixed parameters.")
    result = evaluate_config(
        gt_paths,
        pred_dir,
        args,
        float(args.fixed_peak_frac),
        float(args.fixed_valley_frac),
        int(args.fixed_min_area),
        float(args.fixed_max_remove_frac),
        Path(args.output_mask_dir),
    )
    result["dataset"] = args.dataset
    result["split"] = args.split
    out = Path(args.metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
