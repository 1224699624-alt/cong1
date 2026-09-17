#!/usr/bin/env python3
"""Calibrate pixel-level candidate-mask fusion from validation labels."""

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
    parser = argparse.ArgumentParser(description="Train/apply a bit-pattern pixel mask fusion table.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="val", help="Split(s) for fitting, comma-separated among train,val,test.")
    parser.add_argument("--tune-split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--apply-split", default="test", choices=["train", "val", "test"])
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
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def parse_numbers(text: str, cast=float) -> list:
    return [cast(item) for item in text.split(",") if item.strip()]


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def label_names(raw_root: Path, dataset: str, split: str) -> list[str]:
    label_dir = raw_root / dataset / f"{split}_labels"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")
    return [path.name for path in sorted(label_dir.glob("*.png"))]


def find_mask_path(
    roots: list[Path],
    exp: str,
    dataset: str,
    split: str,
    name: str,
    source_dataset: str | None = None,
) -> Path | None:
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
    for idx in range(1, num):
        if int((labels == idx).sum()) >= min_area:
            keep |= labels == idx
    return keep


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
    if boundary_kernel <= 1:
        return fast_metrics(pred, gt)
    return compute_metrics(pred, gt, boundary_kernel)


def morph(mask: np.ndarray, open_kernel: int, close_kernel: int, min_component: int) -> np.ndarray:
    out = mask.astype(bool)
    if open_kernel > 1:
        out = ndimage.binary_opening(out, structure=np.ones((open_kernel, open_kernel), dtype=bool))
    if close_kernel > 1:
        out = ndimage.binary_closing(out, structure=np.ones((close_kernel, close_kernel), dtype=bool))
    return remove_small_components(out.astype(bool), min_component)


def train_table(
    roots: list[Path],
    raw_root: Path,
    dataset: str,
    split: str,
    candidates: list[str],
    alpha: float,
) -> tuple[np.ndarray, list[str]]:
    split_names = [item.strip() for item in split.split(",") if item.strip()]
    if not split_names:
        raise ValueError("At least one train split is required.")
    n_patterns = 1 << len(candidates)
    fg_counts = np.zeros(n_patterns, dtype=np.float64)
    total_counts = np.zeros(n_patterns, dtype=np.float64)
    complete_names = []
    for current_split in split_names:
        names = label_names(raw_root, dataset, current_split)
        for name in tqdm(names, desc=f"fit/{dataset}/{current_split}"):
            gt = read_mask(raw_root / dataset / f"{current_split}_labels" / name)
            masks = []
            missing = False
            for exp in candidates:
                path = find_mask_path(roots, exp, dataset, current_split, name)
                if path is None:
                    missing = True
                    break
                masks.append(resize_like(read_mask(path), gt))
            if missing:
                continue
            code = stack_code(masks)
            fg_counts += np.bincount(code[gt].ravel(), minlength=n_patterns)
            total_counts += np.bincount(code.ravel(), minlength=n_patterns)
            complete_names.append(f"{current_split}/{name}")
    if not complete_names:
        raise RuntimeError("No complete training images found.")
    probabilities = (fg_counts + alpha) / (total_counts + 2.0 * alpha)
    return probabilities, complete_names


def predict_mask(code: np.ndarray, probabilities: np.ndarray, threshold: float) -> np.ndarray:
    return probabilities[code] >= threshold


def evaluate_config(
    roots: list[Path],
    raw_root: Path,
    dataset: str,
    split: str,
    source_dataset: str | None,
    candidates: list[str],
    probabilities: np.ndarray,
    threshold: float,
    open_kernel: int,
    close_kernel: int,
    min_component: int,
    boundary_kernel: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for name in label_names(raw_root, dataset, split):
        gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
        masks = []
        missing = False
        for exp in candidates:
            path = find_mask_path(roots, exp, dataset, split, name, source_dataset)
            if path is None:
                missing = True
                break
            masks.append(resize_like(read_mask(path), gt))
        if missing:
            continue
        pred = morph(predict_mask(stack_code(masks), probabilities, threshold), open_kernel, close_kernel, min_component)
        item = metric_fn(pred, gt, boundary_kernel)
        item["image"] = name
        records.append(item)
    if not records:
        raise RuntimeError(f"No complete evaluation images found for {dataset}/{split}.")
    keys = [key for key in records[0] if key != "image"]
    mean = {key: float(np.mean([record[key] for record in records])) for key in keys}
    return mean, records


def apply_masks(
    roots: list[Path],
    raw_root: Path,
    dataset: str,
    split: str,
    source_dataset: str | None,
    candidates: list[str],
    probabilities: np.ndarray,
    threshold: float,
    open_kernel: int,
    close_kernel: int,
    min_component: int,
    output_dir: Path,
) -> int:
    count = 0
    for name in tqdm(label_names(raw_root, dataset, split), desc=f"apply/{dataset}/{split}"):
        gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
        masks = []
        missing = False
        for exp in candidates:
            path = find_mask_path(roots, exp, dataset, split, name, source_dataset)
            if path is None:
                missing = True
                break
            masks.append(resize_like(read_mask(path), gt))
        if missing:
            continue
        pred = morph(predict_mask(stack_code(masks), probabilities, threshold), open_kernel, close_kernel, min_component)
        write_mask(output_dir / name, pred)
        count += 1
    return count


def main() -> None:
    args = parse_args()
    tune_split = args.tune_split or args.train_split
    roots = [Path(root) for root in args.ablations_roots]
    probabilities, complete_train_names = train_table(
        roots,
        Path(args.train_raw_root),
        args.train_dataset,
        args.train_split,
        args.candidates,
        args.alpha,
    )

    thresholds = parse_numbers(args.thresholds, float)
    open_kernels = parse_numbers(args.open_kernels, int)
    close_kernels = parse_numbers(args.close_kernels, int)
    min_components = parse_numbers(args.min_components, int)
    best = None
    tried = []
    for threshold in thresholds:
        for open_kernel in open_kernels:
            for close_kernel in close_kernels:
                for min_component in min_components:
                    mean, _ = evaluate_config(
                        roots,
                        Path(args.train_raw_root),
                        args.train_dataset,
                        tune_split,
                        None,
                        args.candidates,
                        probabilities,
                        threshold,
                        open_kernel,
                        close_kernel,
                        min_component,
                        args.boundary_kernel,
                    )
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

    output_dir = Path(args.ablations_roots[0]) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    written = apply_masks(
        roots,
        Path(args.apply_raw_root),
        args.apply_dataset,
        args.apply_split,
        args.source_apply_dataset,
        args.candidates,
        probabilities,
        best["threshold"],
        best["open_kernel"],
        best["close_kernel"],
        best["min_component"],
        output_dir,
    )
    apply_mean, apply_records = evaluate_config(
        roots,
        Path(args.apply_raw_root),
        args.apply_dataset,
        args.apply_split,
        args.source_apply_dataset,
        args.candidates,
        probabilities,
        best["threshold"],
        best["open_kernel"],
        best["close_kernel"],
        best["min_component"],
        args.boundary_kernel,
    )

    model = {
        "train_dataset": args.train_dataset,
        "train_split": args.train_split,
        "tune_split": tune_split,
        "apply_dataset": args.apply_dataset,
        "apply_split": args.apply_split,
        "candidates": args.candidates,
        "alpha": args.alpha,
        "complete_train_images": len(complete_train_names),
        "probabilities": [float(value) for value in probabilities],
        "best": best,
        "tried": tried,
        "output_exp": args.output_exp,
        "written_masks": written,
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
    print(json.dumps({"best_val": best, "apply_mean": apply_mean, "written_masks": written}, indent=2))
    print(f"Saved masks: {output_dir}")
    print(f"Saved model: {args.model_json}")
    print(f"Saved metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
