#!/usr/bin/env python3
"""Rank potentially low-quality labels from mask agreement vs GT mismatch."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze suspicious labels via multi-model agreement and GT mismatch.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--pairwise-dice-min", type=float, default=0.93)
    parser.add_argument("--mean-dice-max", type=float, default=0.87)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    den = float(a.sum() + b.sum())
    return 1.0 if den <= 0 else (2.0 * inter) / den


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return 1.0 if union <= 0 else inter / union


def boundary(mask: np.ndarray, kernel: int) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(pil_mask.filter(ImageFilter.MaxFilter(kernel))) > 0
    eroded = np.asarray(pil_mask.filter(ImageFilter.MinFilter(kernel))) > 0
    return np.logical_xor(dilated, eroded)


def boundary_iou(a: np.ndarray, b: np.ndarray, kernel: int) -> float:
    ba = boundary(a, kernel)
    bb = boundary(b, kernel)
    inter = float(np.logical_and(ba, bb).sum())
    union = float(np.logical_or(ba, bb).sum())
    return 1.0 if union <= 0 else inter / union


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    ablations_root = Path(args.ablations_root)
    gt_dir = raw_root / args.dataset / f"{args.split}_labels"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth label dir not found: {gt_dir}")

    experiment_dirs = {
        experiment: ablations_root / experiment / args.dataset / args.split / "masks"
        for experiment in args.experiments
    }
    for experiment, mask_dir in experiment_dirs.items():
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask dir not found for {experiment}: {mask_dir}")

    records: list[dict[str, object]] = []
    for gt_path in sorted(gt_dir.glob("*.png")):
        gt = read_binary_mask(gt_path)
        preds = {experiment: read_binary_mask(mask_dir / gt_path.name) for experiment, mask_dir in experiment_dirs.items()}

        method_dice = {experiment: dice(pred, gt) for experiment, pred in preds.items()}
        method_biou = {experiment: boundary_iou(pred, gt, args.boundary_kernel) for experiment, pred in preds.items()}

        pairwise_dice_values: list[float] = []
        pairwise_iou_values: list[float] = []
        for exp_a, exp_b in combinations(args.experiments, 2):
            pairwise_dice_values.append(dice(preds[exp_a], preds[exp_b]))
            pairwise_iou_values.append(iou(preds[exp_a], preds[exp_b]))

        consensus_prob = sum(pred.astype(np.float32) for pred in preds.values()) / float(len(preds))
        consensus = consensus_prob >= 0.6
        record = {
            "image": gt_path.name,
            "mean_dice_to_gt": float(np.mean(list(method_dice.values()))),
            "std_dice_to_gt": float(np.std(list(method_dice.values()))),
            "mean_biou_to_gt": float(np.mean(list(method_biou.values()))),
            "mean_pairwise_dice": float(np.mean(pairwise_dice_values)),
            "mean_pairwise_iou": float(np.mean(pairwise_iou_values)),
            "consensus_dice_to_gt": dice(consensus, gt),
            "consensus_biou_to_gt": boundary_iou(consensus, gt, args.boundary_kernel),
            "method_dice": method_dice,
            "method_biou": method_biou,
        }
        record["suspicion_score"] = (
            0.45 * float(record["mean_pairwise_dice"])
            + 0.20 * float(record["mean_pairwise_iou"])
            - 0.45 * float(record["mean_dice_to_gt"])
            - 0.10 * float(record["mean_biou_to_gt"])
            - 0.05 * float(record["std_dice_to_gt"])
        )
        records.append(record)

    records_by_suspicion = sorted(records, key=lambda item: float(item["suspicion_score"]), reverse=True)
    records_by_lowdice = sorted(records, key=lambda item: float(item["mean_dice_to_gt"]))
    high_consensus_low_gt = [
        item
        for item in records_by_suspicion
        if float(item["mean_pairwise_dice"]) >= args.pairwise_dice_min
        and float(item["mean_dice_to_gt"]) <= args.mean_dice_max
    ]

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "experiments": args.experiments,
        "pairwise_dice_min": args.pairwise_dice_min,
        "mean_dice_max": args.mean_dice_max,
        "top_suspects": records_by_suspicion[: args.top_k],
        "lowest_mean_dice": records_by_lowdice[: args.top_k],
        "high_consensus_low_gt": high_consensus_low_gt[: args.top_k],
    }

    output = Path(args.output) if args.output else (
        Path("outputs/analysis") / f"suspect_labels_{args.dataset}_{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "num_top_suspects": len(report["top_suspects"]),
        "num_high_consensus_low_gt": len(report["high_consensus_low_gt"]),
        "first_high_consensus_low_gt": [item["image"] for item in report["high_consensus_low_gt"][:10]],
    }, indent=2))


if __name__ == "__main__":
    main()
