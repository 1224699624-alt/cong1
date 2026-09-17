#!/usr/bin/env python3
"""Boundary-band oracle for testing whether local contour correction can close a Dice gap."""

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
    parser = argparse.ArgumentParser(description="Analyze GT-leaking boundary-band correction upper bound.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--radii", default="1,2,3,4,6,8,10,12,16")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--target-dice", type=float, default=0.9317660066557425)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"))
    return mask > 0


def resize_like(mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    if mask.shape == target.shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(target.shape[::-1], Image.NEAREST)) > 0


def find_mask(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"mask not found: exp={exp} dataset={dataset} split={split} name={name}")


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in records[0] if key != "image"]
    return {key: float(np.mean([float(record[key]) for record in records])) for key in keys}


def component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask)
    return int(n)


def add_layout_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def evaluate_preds(names: list[str], preds: list[np.ndarray], gts: list[np.ndarray], boundary_kernel: int) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for name, pred, gt in zip(names, preds, gts, strict=True):
        record = compute_metrics(pred, gt, boundary_kernel)
        add_layout_metrics(record, pred, gt)
        record["image"] = name
        records.append(record)
    return mean_records(records), records


def main() -> None:
    args = parse_args()
    roots = [Path(root) for root in args.ablations_roots]
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")

    names = [path.name for path in sorted(gt_dir.glob("*.png"))]
    gts = [read_mask(gt_dir / name) for name in names]
    anchors = [
        resize_like(read_mask(find_mask(roots, args.anchor_exp, args.dataset, args.split, name, args.source_dataset)), gt)
        for name, gt in tqdm(list(zip(names, gts, strict=True)), desc="load")
    ]

    anchor_mean, anchor_records = evaluate_preds(names, anchors, gts, args.boundary_kernel)
    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    by_radius: dict[str, dict[str, object]] = {}
    best_label = "anchor"
    best_mean = anchor_mean

    for radius in tqdm(radii, desc="radii"):
        preds = []
        for anchor, gt in zip(anchors, gts, strict=True):
            band = boundary_band(anchor, radius)
            pred = anchor.copy()
            pred[band] = gt[band]
            preds.append(pred)
        mean, records = evaluate_preds(names, preds, gts, args.boundary_kernel)
        label = f"boundary_band_oracle_r{radius}"
        by_radius[label] = {"radius": radius, "mean": mean, "per_image": records}
        if mean["dice"] > best_mean["dice"]:
            best_label = label
            best_mean = mean

    output = {
        "dataset": args.dataset,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "target_dice": args.target_dice,
        "num_images": len(names),
        "anchor": {"mean": anchor_mean, "per_image": anchor_records},
        "by_radius": by_radius,
        "best_label": best_label,
        "best_mean": best_mean,
        "target_reached_by_best": float(best_mean.get("dice", 0.0)) >= args.target_dice,
        "target_margin": float(best_mean.get("dice", 0.0)) - args.target_dice,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "anchor_dice": anchor_mean.get("dice"),
        "best_label": best_label,
        "best_mean": best_mean,
        "target_reached_by_best": output["target_reached_by_best"],
        "target_margin": output["target_margin"],
    }, indent=2))
    print(f"Saved boundary-band oracle: {out_path}")


if __name__ == "__main__":
    main()
