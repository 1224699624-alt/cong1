#!/usr/bin/env python3
"""Analyze ensemble/oracle upper bounds for binary mask candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics, mean_metrics, read_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze mask candidate upper bounds.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-dice", type=float, default=0.9317660066557425)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--fast-no-boundary", action="store_true")
    return parser.parse_args()


def find_mask_path(
    roots: list[Path],
    exp: str,
    dataset: str,
    split: str,
    name: str,
    source_dataset: str | None,
) -> Path | None:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    return None


def resize_like(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if mask.shape == gt.shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0


def summarize_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    numeric_keys = [
        key
        for key, value in records[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    return {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in numeric_keys
    }


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


def metric_fn(pred: np.ndarray, gt: np.ndarray, boundary_kernel: int, fast_no_boundary: bool) -> dict[str, float]:
    if fast_no_boundary:
        return fast_metrics(pred, gt)
    return compute_metrics(pred, gt, boundary_kernel)


def evaluate_single(
    exp: str,
    masks: dict[str, list[np.ndarray]],
    gts: list[np.ndarray],
    names: list[str],
    boundary_kernel: int,
    fast_no_boundary: bool,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for name, pred, gt in zip(names, masks[exp], gts, strict=True):
        metrics = metric_fn(pred, gt, boundary_kernel, fast_no_boundary)
        metrics["image"] = name
        records.append(metrics)
    return summarize_records(records), records


def evaluate_generated(
    label: str,
    preds: list[np.ndarray],
    gts: list[np.ndarray],
    names: list[str],
    boundary_kernel: int,
    fast_no_boundary: bool,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for name, pred, gt in zip(names, preds, gts, strict=True):
        metrics = metric_fn(pred, gt, boundary_kernel, fast_no_boundary)
        metrics["image"] = name
        records.append(metrics)
    return summarize_records(records), records


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    roots = [Path(p) for p in args.ablations_roots]
    gt_dir = raw_root / args.dataset / f"{args.split}_labels"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth label dir not found: {gt_dir}")

    names = [p.name for p in sorted(gt_dir.glob("*.png"))]
    gts = [read_binary_mask(gt_dir / name) for name in names]

    masks: dict[str, list[np.ndarray]] = {}
    missing: dict[str, list[str]] = {}
    for exp in args.experiments:
        exp_masks = []
        exp_missing = []
        for name, gt in zip(names, gts, strict=True):
            path = find_mask_path(roots, exp, args.dataset, args.split, name, args.source_dataset)
            if path is None:
                exp_missing.append(name)
                continue
            exp_masks.append(resize_like(read_binary_mask(path), gt))
        if exp_missing:
            missing[exp] = exp_missing[:50]
            continue
        masks[exp] = exp_masks

    if not masks:
        raise RuntimeError("No complete candidate experiments found.")

    single = {}
    per_image_records = {}
    for exp in tqdm(sorted(masks), desc="single candidates"):
        mean, records = evaluate_single(exp, masks, gts, names, args.boundary_kernel, args.fast_no_boundary)
        single[exp] = mean
        per_image_records[exp] = records

    stack_by_image = [
        np.stack([masks[exp][idx] for exp in sorted(masks)], axis=0)
        for idx in range(len(names))
    ]
    candidate_names = sorted(masks)

    generated: dict[str, dict[str, float]] = {}
    generated_records: dict[str, list[dict[str, float]]] = {}

    union_preds = [stack.any(axis=0) for stack in stack_by_image]
    intersection_preds = [stack.all(axis=0) for stack in stack_by_image]
    generated["union_all"], generated_records["union_all"] = evaluate_generated(
        "union_all", union_preds, gts, names, args.boundary_kernel
        , args.fast_no_boundary
    )
    generated["intersection_all"], generated_records["intersection_all"] = evaluate_generated(
        "intersection_all", intersection_preds, gts, names, args.boundary_kernel, args.fast_no_boundary
    )

    n = len(candidate_names)
    for threshold in range(1, n + 1):
        vote_preds = [stack.sum(axis=0) >= threshold for stack in stack_by_image]
        label = f"vote_ge_{threshold}"
        generated[label], generated_records[label] = evaluate_generated(
            label, vote_preds, gts, names, args.boundary_kernel, args.fast_no_boundary
        )

    per_image_best_preds = []
    per_image_best_records = []
    per_image_best_counts: Counter[str] = Counter()
    for idx, name in enumerate(names):
        best_exp = max(candidate_names, key=lambda exp: per_image_records[exp][idx]["dice"])
        per_image_best_counts[best_exp] += 1
        pred = masks[best_exp][idx]
        per_image_best_preds.append(pred)
        metrics = metric_fn(pred, gts[idx], args.boundary_kernel, args.fast_no_boundary)
        metrics["image"] = name
        metrics["selected_exp"] = best_exp
        per_image_best_records.append(metrics)
    generated["per_image_oracle"] = summarize_records(per_image_best_records)
    generated_records["per_image_oracle"] = per_image_best_records

    pixel_oracle_preds = []
    for stack, gt in zip(stack_by_image, gts, strict=True):
        can_fg = stack.any(axis=0)
        can_bg = (~stack).any(axis=0)
        pred = np.where(gt, can_fg, ~can_bg)
        pixel_oracle_preds.append(pred.astype(bool))
    generated["pixel_oracle"], generated_records["pixel_oracle"] = evaluate_generated(
        "pixel_oracle", pixel_oracle_preds, gts, names, args.boundary_kernel, args.fast_no_boundary
    )

    all_systems = {**single, **generated}
    best_label = max(all_systems, key=lambda key: all_systems[key].get("dice", 0.0))
    output = {
        "dataset": args.dataset,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "num_images": len(names),
        "target_dice": args.target_dice,
        "candidate_experiments": candidate_names,
        "missing": missing,
        "single": single,
        "generated": generated,
        "per_image_oracle_selection_counts": dict(per_image_best_counts),
        "best_label": best_label,
        "best_mean": all_systems[best_label],
        "target_reached_by_best": all_systems[best_label].get("dice", 0.0) >= args.target_dice,
        "target_margin": all_systems[best_label].get("dice", 0.0) - args.target_dice,
        "per_image": {
            "per_image_oracle": per_image_best_records,
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_label": output["best_label"],
        "best_mean": output["best_mean"],
        "target_dice": output["target_dice"],
        "target_reached_by_best": output["target_reached_by_best"],
        "target_margin": output["target_margin"],
    }, indent=2))
    print(f"Saved oracle analysis: {out_path}")


if __name__ == "__main__":
    main()
