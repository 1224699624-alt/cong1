#!/usr/bin/env python3
"""Image-evidence layout scorer for conservative bridge-like mask edits.

The script precomputes candidate bridge pixels from anchor-mask geometry, then
scores those pixels with local image evidence. Grid search is cheap because the
expensive component and distance-transform work is done once per image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply image-evidence layout scorer.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-mask-dir", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--score-thresholds", default="0.55,0.65,0.75")
    parser.add_argument("--max-remove-fracs", default="0.001,0.002,0.004")
    parser.add_argument("--peak-frac", type=float, default=0.45)
    parser.add_argument("--valley-frac", type=float, default=0.40)
    parser.add_argument("--min-area", type=int, default=18)
    parser.add_argument("--large-area-percentile", type=float, default=55.0)
    parser.add_argument("--fixed-score-threshold", type=float, default=None)
    parser.add_argument("--fixed-max-remove-frac", type=float, default=None)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def find_image_path(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found for {stem}: {root}/{dataset}/{split}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image.astype(np.float32) / 255.0


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


def normalize_inside(image: np.ndarray, component: np.ndarray) -> np.ndarray:
    values = image[component]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo + 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def has_multiple_peaks(component: np.ndarray, peak_frac: float, min_area: int) -> bool:
    dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 1.0:
        return False
    peaks = (dist >= float(peak_frac) * float(dist.max())) & component
    n, _, stats, _ = cv2.connectedComponentsWithStats(peaks.astype(np.uint8), connectivity=8)
    kept = sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) >= max(1, min_area // 4))
    return kept >= 2


def precompute_candidates(mask: np.ndarray, image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    score = np.zeros(mask.shape, dtype=np.float32)
    candidate = np.zeros(mask.shape, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0]
    if not areas:
        return candidate, score
    cutoff = max(float(args.min_area * 2), float(np.percentile(np.asarray(areas, dtype=np.float32), args.large_area_percentile)))
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.2)
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < cutoff:
            continue
        component = labels == idx
        if not has_multiple_peaks(component, args.peak_frac, args.min_area):
            continue
        dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        if float(dist.max()) <= 1.0:
            continue
        dist_norm = dist / float(dist.max())
        low_dist = (dist_norm <= float(args.valley_frac)) & component
        neighbor_count = cv2.filter2D(component.astype(np.uint8), -1, np.ones((3, 3), np.uint8))
        cand = low_dist & (neighbor_count >= 4)
        if not cand.any():
            continue
        local_img = normalize_inside(smooth, component)
        dark_score = 1.0 - local_img
        dist_score = 1.0 - np.clip(dist_norm / max(args.valley_frac, 1e-6), 0.0, 1.0)
        combined = (0.65 * dark_score + 0.35 * dist_score).astype(np.float32)
        candidate |= cand
        score[cand] = np.maximum(score[cand], combined[cand])
    return candidate, score


def collect_items(args: argparse.Namespace) -> list[dict[str, object]]:
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    pred_dir = Path(args.pred_dir)
    items: list[dict[str, object]] = []
    gt_paths = sorted(gt_dir.glob("*.png"))
    if args.limit > 0:
        gt_paths = gt_paths[: args.limit]
    for gt_path in tqdm(gt_paths, desc=f"collect/{args.dataset}/{args.split}"):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt = read_mask(gt_path)
        pred = read_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        image = read_gray(find_image_path(Path(args.raw_root), args.dataset, args.split, gt_path.stem))
        if image.shape != pred.shape:
            image = cv2.resize(image, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)
        candidate, score = precompute_candidates(pred, image, args)
        items.append({"name": gt_path.name, "gt": gt, "pred": pred, "candidate": candidate, "score": score})
    return items


def apply_rule(item: dict[str, object], score_threshold: float, max_remove_frac: float) -> np.ndarray:
    pred = item["pred"]  # type: ignore[assignment]
    candidate = item["candidate"]  # type: ignore[assignment]
    score = item["score"]  # type: ignore[assignment]
    eligible = candidate & (score >= float(score_threshold))
    if not eligible.any():
        return pred.copy()
    max_remove = int(float(max_remove_frac) * max(1, int(pred.sum())))
    if max_remove <= 0:
        return pred.copy()
    ys, xs = np.where(eligible)
    order = np.argsort(score[ys, xs])[::-1]
    keep = order[: min(max_remove, len(order))]
    out = pred.copy()
    out[ys[keep], xs[keep]] = False
    return out


def evaluate_items(
    items: list[dict[str, object]],
    score_threshold: float,
    max_remove_frac: float,
    boundary_kernel: int,
    out_dir: Path | None = None,
) -> dict[str, object]:
    records: list[dict[str, float]] = []
    changed = 0
    removed_pixels = 0
    for item in items:
        pred = item["pred"]  # type: ignore[assignment]
        gt = item["gt"]  # type: ignore[assignment]
        out = apply_rule(item, score_threshold, max_remove_frac)
        changed += int(not np.array_equal(pred, out))
        removed_pixels += int(pred.sum() - out.sum())
        if out_dir is not None:
            write_mask(out_dir / str(item["name"]), out)
        rec = compute_metrics(out, gt, boundary_kernel=boundary_kernel)
        rec = add_structure_metrics(rec, out, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    return {
        "score_threshold": float(score_threshold),
        "max_remove_frac": float(max_remove_frac),
        "changed_images": int(changed),
        "removed_pixels": int(removed_pixels),
        "mean": mean_metrics(records),
        "per_image": records,
    }


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    items = collect_items(args)
    if args.grid_search:
        grid = []
        for threshold in parse_float_list(args.score_thresholds):
            for max_remove in parse_float_list(args.max_remove_fracs):
                grid.append(evaluate_items(items, threshold, max_remove, args.boundary_kernel))
        best = max(grid, key=lambda item: float(item["mean"]["dice"]))
        summary = {"dataset": args.dataset, "split": args.split, "pred_dir": args.pred_dir, "grid": grid, "best": best}
        out = Path(args.metrics_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({
            "best": {k: best[k] for k in ["score_threshold", "max_remove_frac", "changed_images", "removed_pixels"]},
            "mean": best["mean"],
        }, indent=2))
        return
    if args.fixed_score_threshold is None or args.fixed_max_remove_frac is None:
        raise ValueError("fixed mode requires --fixed-score-threshold and --fixed-max-remove-frac")
    result = evaluate_items(
        items,
        float(args.fixed_score_threshold),
        float(args.fixed_max_remove_frac),
        args.boundary_kernel,
        Path(args.output_mask_dir),
    )
    result["dataset"] = args.dataset
    result["split"] = args.split
    result["pred_dir"] = args.pred_dir
    out = Path(args.metrics_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2))


if __name__ == "__main__":
    main()
