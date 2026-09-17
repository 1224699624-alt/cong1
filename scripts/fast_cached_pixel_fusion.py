#!/usr/bin/env python3
"""Fast cached pixel-level fusion for a small set of binary mask candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached pixel mask fusion table search.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--close-kernels", default="1,3")
    parser.add_argument("--open-kernels", default="1")
    parser.add_argument("--min-components", default="0,10,25,50")
    parser.add_argument("--boundary-kernel", type=int, default=1)
    return parser.parse_args()


def parse_numbers(text: str, cast=float) -> list:
    return [cast(item) for item in text.split(",") if item.strip()]


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def label_names(raw_root: Path, dataset: str, split: str) -> list[str]:
    label_dir = raw_root / dataset / f"{split}_labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")
    return [path.name for path in sorted(label_dir.glob("*.png"))]


def find_mask_path(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None = None) -> Path | None:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for current_dataset in datasets:
            path = root / exp / current_dataset / split / "masks" / name
            if path.exists():
                return path
    return None


def resize_like(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    if mask.shape == target.shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(target.shape[::-1], Image.NEAREST)) > 0


def stack_code(masks: list[np.ndarray]) -> np.ndarray:
    code = np.zeros(masks[0].shape, dtype=np.uint32)
    for idx, mask in enumerate(masks):
        code |= mask.astype(np.uint32) << idx
    return code


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    labels, num = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, num + 1):
        if int((labels == idx).sum()) >= min_area:
            keep |= labels == idx
    return keep


def morph(mask: np.ndarray, open_kernel: int, close_kernel: int, min_component: int) -> np.ndarray:
    out = mask.astype(bool)
    if open_kernel > 1:
        out = ndimage.binary_opening(out, structure=np.ones((open_kernel, open_kernel), dtype=bool))
    if close_kernel > 1:
        out = ndimage.binary_closing(out, structure=np.ones((close_kernel, close_kernel), dtype=bool))
    return remove_small_components(out, min_component)


def fast_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    return {
        "dice": float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0,
        "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0,
        "boundary_iou": 0.0,
    }


def metric_fn(pred: np.ndarray, gt: np.ndarray, boundary_kernel: int) -> dict[str, float]:
    return fast_metrics(pred, gt) if boundary_kernel <= 1 else compute_metrics(pred, gt, boundary_kernel)


def load_codes(
    roots: list[Path],
    raw_root: Path,
    dataset: str,
    split: str,
    candidates: list[str],
    source_dataset: str | None,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for name in tqdm(label_names(raw_root, dataset, split), desc=f"cache/{dataset}/{split}"):
        gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
        masks = []
        for exp in candidates:
            path = find_mask_path(roots, exp, dataset, split, name, source_dataset)
            if path is None:
                masks = []
                break
            masks.append(resize_like(read_mask(path), gt))
        if masks:
            items.append({"name": name, "gt": gt, "code": stack_code(masks)})
    if not items:
        raise RuntimeError(f"No complete cached items for {dataset}/{split}.")
    return items


def train_table(items: list[dict[str, object]], n_patterns: int, alpha: float) -> np.ndarray:
    fg_counts = np.zeros(n_patterns, dtype=np.float64)
    total_counts = np.zeros(n_patterns, dtype=np.float64)
    for item in items:
        code = item["code"]
        gt = item["gt"]
        fg_counts += np.bincount(code[gt].ravel(), minlength=n_patterns)
        total_counts += np.bincount(code.ravel(), minlength=n_patterns)
    return (fg_counts + alpha) / (total_counts + 2.0 * alpha)


def evaluate_items(items: list[dict[str, object]], probabilities: np.ndarray, threshold: float, open_kernel: int, close_kernel: int, min_component: int, boundary_kernel: int) -> tuple[dict[str, float], list[dict[str, float]], list[np.ndarray]]:
    records = []
    preds = []
    for item in items:
        pred = morph(probabilities[item["code"]] >= threshold, open_kernel, close_kernel, min_component)
        metrics = metric_fn(pred, item["gt"], boundary_kernel)
        metrics["image"] = item["name"]
        records.append(metrics)
        preds.append(pred)
    keys = [key for key in records[0] if key != "image"]
    mean = {key: float(np.mean([record[key] for record in records])) for key in keys}
    return mean, records, preds


def main() -> None:
    args = parse_args()
    roots = [Path(root) for root in args.ablations_roots]
    train_items = load_codes(roots, Path(args.train_raw_root), args.train_dataset, args.train_split, args.candidates, None)
    apply_items = load_codes(roots, Path(args.apply_raw_root), args.apply_dataset, args.apply_split, args.candidates, args.source_apply_dataset)
    probabilities = train_table(train_items, 1 << len(args.candidates), args.alpha)

    best = None
    tried = []
    for threshold in parse_numbers(args.thresholds, float):
        for open_kernel in parse_numbers(args.open_kernels, int):
            for close_kernel in parse_numbers(args.close_kernels, int):
                for min_component in parse_numbers(args.min_components, int):
                    mean, _, _ = evaluate_items(train_items, probabilities, threshold, open_kernel, close_kernel, min_component, args.boundary_kernel)
                    item = {
                        "threshold": float(threshold),
                        "open_kernel": int(open_kernel),
                        "close_kernel": int(close_kernel),
                        "min_component": int(min_component),
                        "mean": mean,
                    }
                    tried.append(item)
                    if best is None or mean["dice"] > best["mean"]["dice"]:
                        best = item
    assert best is not None

    apply_mean, apply_records, preds = evaluate_items(
        apply_items,
        probabilities,
        best["threshold"],
        best["open_kernel"],
        best["close_kernel"],
        best["min_component"],
        args.boundary_kernel,
    )
    out_dir = Path(args.ablations_roots[0]) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    for item, pred in zip(apply_items, preds, strict=True):
        write_mask(out_dir / str(item["name"]), pred)

    model = {
        "train_dataset": args.train_dataset,
        "train_split": args.train_split,
        "apply_dataset": args.apply_dataset,
        "apply_split": args.apply_split,
        "candidates": args.candidates,
        "alpha": args.alpha,
        "complete_train_images": len(train_items),
        "complete_apply_images": len(apply_items),
        "probabilities": [float(value) for value in probabilities],
        "best": best,
        "tried": tried,
        "output_exp": args.output_exp,
    }
    metrics = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "output_exp": args.output_exp,
        "num_evaluated": len(apply_records),
        "mean": apply_mean,
        "per_image": apply_records,
        "model_json": args.model_json,
    }
    Path(args.model_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_json).write_text(json.dumps(model, indent=2), encoding="utf-8")
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"best_val": best, "apply_mean": apply_mean, "written_masks": len(preds)}, indent=2))
    print(f"Saved masks: {out_dir}")
    print(f"Saved model: {args.model_json}")
    print(f"Saved metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
