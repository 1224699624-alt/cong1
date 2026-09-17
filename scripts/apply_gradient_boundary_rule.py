#!/usr/bin/env python3
"""Fast image-gradient boundary rule around an anchor mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a cheap gradient-guided boundary edit rule.")
    parser.add_argument("--tune-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--tune-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--apply-anchor-exp", default=None)
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--radii", default="1,2,3,4")
    parser.add_argument("--low-grad-thresholds", default="0.15,0.25,0.35")
    parser.add_argument("--high-grad-thresholds", default="0.55,0.70,0.85")
    parser.add_argument("--max-remove-fracs", default="0,0.001,0.0025,0.005")
    parser.add_argument("--max-add-fracs", default="0,0.001,0.0025")
    parser.add_argument("--top-full-metrics", type=int, default=12)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--output-exp", default="r085_gradient_boundary_rule")
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def find_image(raw_root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = raw_root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found: {raw_root}/{dataset}/{split}/{stem}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def find_anchor(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None = None) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"anchor not found: exp={exp} dataset={dataset} split={split} name={name}")


def resize_like(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if mask.shape == gt.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def gradient01(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    scale = max(1.0, float(np.percentile(mag, 95)))
    return np.clip(mag / scale, 0.0, 1.0)


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
    return {k: float(np.mean([float(r[k]) for r in records])) for k in keys}


def limit_candidates(candidates: np.ndarray, score: np.ndarray, max_frac: float, total_pixels: int) -> np.ndarray:
    if max_frac <= 0:
        return np.zeros_like(candidates, dtype=bool)
    max_count = int(round(max_frac * float(total_pixels)))
    idx = np.flatnonzero(candidates.ravel())
    if max_count <= 0 or idx.size == 0:
        return np.zeros_like(candidates, dtype=bool)
    if idx.size <= max_count:
        return candidates
    keep = idx[np.argsort(score.ravel()[idx])[-max_count:]]
    out = np.zeros(candidates.size, dtype=bool)
    out[keep] = True
    return out.reshape(candidates.shape)


def apply_rule(item: dict[str, object], cfg: dict[str, float]) -> np.ndarray:
    anchor = item["anchor"]  # type: ignore[assignment]
    grad = item["grad"]  # type: ignore[assignment]
    radius = int(cfg["radius"])
    low = float(cfg["low_grad"])
    high = float(cfg["high_grad"])
    max_remove = float(cfg["max_remove_frac"])
    max_add = float(cfg["max_add_frac"])

    dist_in = ndimage.distance_transform_edt(anchor)
    dist_out = ndimage.distance_transform_edt(~anchor)
    inner = anchor & (dist_in <= radius)
    outer = (~anchor) & (dist_out <= radius)

    remove_candidates = inner & (grad <= low)
    add_candidates = outer & (grad >= high)
    remove = limit_candidates(remove_candidates, 1.0 - grad, max_remove, anchor.size)
    add = limit_candidates(add_candidates, grad, max_add, anchor.size)

    pred = anchor.copy()
    pred[remove] = False
    pred[add] = True
    return pred


def collect(raw_root: Path, dataset: str, split: str, roots: list[Path], anchor_exp: str, source_dataset: str | None) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    label_dir = raw_root / dataset / f"{split}_labels"
    for label_path in sorted(label_dir.glob("*.png")):
        gt = read_mask(label_path)
        gray = read_gray(find_image(raw_root, dataset, split, label_path.stem))
        anchor = resize_like(read_mask(find_anchor(roots, anchor_exp, dataset, split, label_path.name, source_dataset)), gt)
        items.append({
            "name": label_path.name,
            "gt": gt,
            "anchor": anchor,
            "grad": gradient01(gray),
        })
    if not items:
        raise FileNotFoundError(label_dir)
    return items


def evaluate_items(
    items: list[dict[str, object]],
    cfg: dict[str, float],
    boundary_kernel: int,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    for item in items:
        pred = apply_rule(item, cfg)
        gt = item["gt"]  # type: ignore[assignment]
        if write_dir is not None:
            write_mask(write_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel)
        add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    return mean_metrics(records), records


def evaluate_items_fast(items: list[dict[str, object]], cfg: dict[str, float]) -> dict[str, float]:
    dice_values = []
    precision_values = []
    recall_values = []
    changed_values = []
    for item in items:
        pred = apply_rule(item, cfg)
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        tp = float(np.logical_and(pred, gt).sum())
        fp = float(np.logical_and(pred, ~gt).sum())
        fn = float(np.logical_and(~pred, gt).sum())
        dice_values.append((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6))
        precision_values.append((tp + 1e-6) / (tp + fp + 1e-6))
        recall_values.append((tp + 1e-6) / (tp + fn + 1e-6))
        changed_values.append(float(np.mean(pred != anchor)))
    return {
        "dice": float(np.mean(dice_values)),
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "changed_frac": float(np.mean(changed_values)),
    }


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    tune_items = collect(Path(args.tune_raw_root), args.tune_dataset, args.tune_split, roots, args.anchor_exp, None)
    fast_grid = []
    for radius in parse_ints(args.radii):
        for low_grad in parse_floats(args.low_grad_thresholds):
            for high_grad in parse_floats(args.high_grad_thresholds):
                for max_remove in parse_floats(args.max_remove_fracs):
                    for max_add in parse_floats(args.max_add_fracs):
                        cfg = {
                            "radius": float(radius),
                            "low_grad": float(low_grad),
                            "high_grad": float(high_grad),
                            "max_remove_frac": float(max_remove),
                            "max_add_frac": float(max_add),
                        }
                        mean = evaluate_items_fast(tune_items, cfg)
                        fast_grid.append({"cfg": cfg, "fast_mean": mean})
    fast_grid = sorted(fast_grid, key=lambda row: row["fast_mean"]["dice"], reverse=True)
    full_grid = []
    for row in fast_grid[: max(1, args.top_full_metrics)]:
        mean, _ = evaluate_items(tune_items, row["cfg"], args.boundary_kernel)
        full_grid.append({**row, "mean": mean})
    best = max(full_grid, key=lambda row: row["mean"]["dice"])

    apply_anchor = args.apply_anchor_exp or args.anchor_exp
    apply_items = collect(
        Path(args.apply_raw_root),
        args.apply_dataset,
        args.apply_split,
        roots,
        apply_anchor,
        args.source_apply_dataset,
    )
    write_dir = Path(args.pred_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    apply_mean, apply_records = evaluate_items(apply_items, best["cfg"], args.boundary_kernel, write_dir)

    output = {
        "tune_dataset": args.tune_dataset,
        "apply_dataset": args.apply_dataset,
        "anchor_exp": args.anchor_exp,
        "apply_anchor_exp": apply_anchor,
        "best": best,
        "apply_mean": apply_mean,
        "apply_per_image": apply_records,
        "fast_grid": fast_grid,
        "full_grid": full_grid,
        "output_exp": args.output_exp,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "apply_mean": apply_mean, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
