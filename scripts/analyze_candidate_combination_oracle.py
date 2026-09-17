#!/usr/bin/env python3
"""Analyze complementarity among clean-test candidate masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate-mask combination oracles.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "EXP"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def resize_like(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(target_shape[::-1], Image.NEAREST)) > 0


def find_mask(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None) -> Path:
    ds_list = [dataset]
    if source_dataset and source_dataset not in ds_list:
        ds_list.append(source_dataset)
    for root in roots:
        for ds in ds_list:
            p = root / exp / ds / split / "masks" / name
            if p.exists():
                return p
    raise FileNotFoundError(f"mask not found: {exp} {dataset}/{split}/{name}")


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


def add_metric(records: dict[str, list[dict[str, float]]], key: str, pred: np.ndarray, gt: np.ndarray, image: str, kernel: int) -> None:
    rec = compute_metrics(pred, gt, kernel)
    rec["image"] = image
    records.setdefault(key, []).append(rec)


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    candidates = [(name, exp) for name, exp in args.candidate]
    records: dict[str, list[dict[str, float]]] = {}
    oracle_choices: list[dict[str, object]] = []
    pixel_oracle_records: list[dict[str, float]] = []
    disagreement_stats = []

    for gt_path in tqdm(sorted(gt_dir.glob("*.png")), desc="combo/oracle"):
        gt = read_mask(gt_path)
        masks = []
        for _, exp in candidates:
            masks.append(resize_like(read_mask(find_mask(roots, exp, args.dataset, args.split, gt_path.name, args.source_dataset)), gt.shape))
        stack = np.stack(masks, axis=0)
        for (name, _), mask in zip(candidates, masks):
            add_metric(records, name, mask, gt, gt_path.name, args.boundary_kernel)

        union = stack.any(axis=0)
        inter = stack.all(axis=0)
        vote2 = stack.sum(axis=0) >= max(1, int(np.ceil(len(candidates) / 2)))
        vote3 = stack.sum(axis=0) >= min(len(candidates), 3)
        add_metric(records, "union_all", union, gt, gt_path.name, args.boundary_kernel)
        add_metric(records, "intersection_all", inter, gt, gt_path.name, args.boundary_kernel)
        add_metric(records, "vote_majority", vote2, gt, gt_path.name, args.boundary_kernel)
        add_metric(records, "vote_ge_3", vote3, gt, gt_path.name, args.boundary_kernel)

        per_candidate = [(compute_metrics(mask, gt, args.boundary_kernel)["dice"], name, mask) for (name, _), mask in zip(candidates, masks)]
        best_dice, best_name, best_mask = max(per_candidate, key=lambda x: x[0])
        add_metric(records, "per_image_oracle", best_mask, gt, gt_path.name, args.boundary_kernel)
        oracle_choices.append({"image": gt_path.name, "best": best_name, "dice": float(best_dice)})

        # GT-leaking pixel oracle chooses any candidate that matches GT per pixel.
        anchor = masks[0]
        pixel = anchor.copy()
        fg_target = gt
        bg_target = ~gt
        if fg_target.any():
            pixel[fg_target] = stack[:, fg_target].any(axis=0)
        if bg_target.any():
            pixel[bg_target] = stack[:, bg_target].all(axis=0)
        rec = compute_metrics(pixel, gt, args.boundary_kernel)
        rec["image"] = gt_path.name
        pixel_oracle_records.append(rec)

        disagreement = stack.max(axis=0) ^ stack.min(axis=0)
        anchor_error = anchor ^ gt
        disagreement_stats.append(
            {
                "image": gt_path.name,
                "disagreement_frac": float(disagreement.mean()),
                "anchor_error_frac": float(anchor_error.mean()),
                "error_capture": float((disagreement & anchor_error).sum() / max(1, anchor_error.sum())),
            }
        )

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "candidates": [{"name": n, "exp": e} for n, e in candidates],
        "mean": {key: mean_metrics(value) for key, value in records.items()},
        "pixel_oracle": mean_metrics(pixel_oracle_records),
        "oracle_choices": oracle_choices,
        "disagreement": {
            "mean_disagreement_frac": float(np.mean([x["disagreement_frac"] for x in disagreement_stats])),
            "mean_anchor_error_frac": float(np.mean([x["anchor_error_frac"] for x in disagreement_stats])),
            "mean_error_capture": float(np.mean([x["error_capture"] for x in disagreement_stats])),
            "per_image": disagreement_stats,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v["dice"] for k, v in summary["mean"].items()}, indent=2))
    print("pixel_oracle_dice", summary["pixel_oracle"]["dice"])


if __name__ == "__main__":
    main()
