#!/usr/bin/env python3
"""R178 conservative delete-only bridge suppressor.

Selects postprocessing parameters on train/val anchors and applies the locked
delete-only rule to R110 clean-test-v2 masks. It never adds foreground.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_anchor_pixel_residual import names, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R178 delete-only bridge suppressor.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", default="r100_like_r025b_r097_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--output-exp", default="r178_delete_only_bridge_suppressor")
    parser.add_argument("--metrics-json", default="outputs/analysis/r178_delete_only_bridge_suppressor_clean_test_v2_metrics.json")
    parser.add_argument("--val-summary-json", default="outputs/analysis/r178_delete_only_bridge_suppressor_val_summary.json")
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--gap-radii", default="3,5,7,9")
    parser.add_argument("--min-delete-areas", default="4,8,16,32,64")
    parser.add_argument("--max-delete-areas", default="128,256,512,1024,2048")
    parser.add_argument("--min-neck-widths", default="1,2,3")
    parser.add_argument("--max-component-error-increase", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--target-dice", type=float, default=0.9317660066557425)
    return parser.parse_args()


def parse_nums(text: str, cast=int) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def split_names(raw_root: Path, dataset: str, split: str, limit: int) -> list[str]:
    out = names(raw_root, dataset, split)
    return out[:limit] if limit and limit > 0 else out


def find_anchor(root: Path, exp: str, dataset: str, split: str, name: str) -> Path:
    path = root / exp / dataset / split / "masks" / name
    if not path.exists():
        raise FileNotFoundError(f"Anchor mask not found: {path}")
    return path


def component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask)
    return int(n)


def add_structure_metrics(rec: dict[str, float], pred: np.ndarray, gt: np.ndarray, gap_radius: int) -> dict[str, float]:
    pred_components = component_count(pred)
    gt_components = component_count(gt)
    structure = np.ones((gap_radius, gap_radius), dtype=bool)
    sep = np.logical_and(ndimage.binary_dilation(gt, structure=structure), ~gt)
    rec["pred_component_count"] = float(pred_components)
    rec["gt_component_count"] = float(gt_components)
    rec["component_count_error"] = float(abs(pred_components - gt_components))
    rec["false_bridge_flag"] = float(pred_components < gt_components)
    rec["sep_fp_rate"] = float(np.logical_and(pred, sep).sum() / max(1, int(sep.sum())))
    return rec


def bridge_candidates(anchor: np.ndarray, gap_radius: int, min_neck_width: int, min_area: int, max_area: int) -> list[np.ndarray]:
    # Candidate deletion pixels are foreground that can be removed by a conservative opening-like test.
    structure = np.ones((max(1, min_neck_width), max(1, min_neck_width)), dtype=bool)
    eroded = ndimage.binary_erosion(anchor, structure=structure)
    reopened = ndimage.binary_dilation(eroded, structure=structure)
    thin_foreground = np.logical_and(anchor, ~reopened)

    boundary_structure = np.ones((gap_radius, gap_radius), dtype=bool)
    local_boundary = ndimage.binary_dilation(~anchor, structure=boundary_structure) & anchor
    candidate_map = thin_foreground & local_boundary

    labels, n = ndimage.label(candidate_map)
    out = []
    for idx in range(1, n + 1):
        comp = labels == idx
        area = int(comp.sum())
        if min_area <= area <= max_area:
            out.append(comp)
    return out


def apply_rule(anchor: np.ndarray, gap_radius: int, min_neck_width: int, min_area: int, max_area: int) -> np.ndarray:
    pred = anchor.copy()
    for comp in bridge_candidates(anchor, gap_radius, min_neck_width, min_area, max_area):
        pred[comp] = False
    return pred


def evaluate_config(
    args: argparse.Namespace,
    dataset: str,
    raw_root: Path,
    split: str,
    exp: str,
    limit: int,
    gap_radius: int,
    min_neck_width: int,
    min_area: int,
    max_area: int,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for name in tqdm(split_names(raw_root, dataset, split, limit), desc=f"eval/{split}/r{gap_radius}/n{min_neck_width}/a{min_area}-{max_area}", leave=False):
        gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
        anchor = resize_like(read_mask(find_anchor(Path(args.ablations_root), exp, dataset, split, name)), gt.shape)
        pred_raw = apply_rule(anchor, gap_radius, min_neck_width, min_area, max_area)
        # Guardrail: reject edits that increase component-count error for this image when GT is available.
        gt_count = component_count(gt)
        anchor_error = abs(component_count(anchor) - gt_count)
        pred_error = abs(component_count(pred_raw) - gt_count)
        pred = pred_raw if pred_error <= anchor_error + args.max_component_error_increase else anchor
        if write_dir is not None:
            write_mask(write_dir / name, pred)
        rec = compute_metrics(pred, gt, args.boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt, gap_radius)
        rec["image"] = name
        records.append(rec)
    keys = [k for k in records[0] if k != "image"]
    mean = {k: float(np.mean([float(r[k]) for r in records])) for k in keys}
    return mean, records


def main() -> None:
    args = parse_args()
    grid = []
    best = None
    for gap_radius in parse_nums(args.gap_radii, int):
        for min_neck_width in parse_nums(args.min_neck_widths, int):
            for min_area in parse_nums(args.min_delete_areas, int):
                for max_area in parse_nums(args.max_delete_areas, int):
                    if max_area < min_area:
                        continue
                    mean, _ = evaluate_config(
                        args,
                        args.train_dataset,
                        Path(args.train_raw_root),
                        args.tune_split,
                        args.anchor_exp,
                        args.max_val_images,
                        gap_radius,
                        min_neck_width,
                        min_area,
                        max_area,
                    )
                    item = {
                        "gap_radius": gap_radius,
                        "min_neck_width": min_neck_width,
                        "min_area": min_area,
                        "max_area": max_area,
                        "mean": mean,
                    }
                    grid.append(item)
                    score = (
                        mean["dice"],
                        -mean["component_count_error"],
                        -mean["false_bridge_flag"],
                        -mean["sep_fp_rate"],
                        mean["boundary_iou"],
                    )
                    if best is None:
                        best = item
                    else:
                        bm = best["mean"]
                        best_score = (
                            bm["dice"],
                            -bm["component_count_error"],
                            -bm["false_bridge_flag"],
                            -bm["sep_fp_rate"],
                            bm["boundary_iou"],
                        )
                        if score > best_score:
                            best = item
    assert best is not None

    val_summary = {
        "dataset": args.train_dataset,
        "split": args.tune_split,
        "num_grid": len(grid),
        "best": best,
        "grid": grid,
    }
    Path(args.val_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_summary_json).write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

    out_dir = Path(args.ablations_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = evaluate_config(
        args,
        args.apply_dataset,
        Path(args.apply_raw_root),
        args.apply_split,
        args.apply_anchor_exp,
        args.max_apply_images,
        int(best["gap_radius"]),
        int(best["min_neck_width"]),
        int(best["min_area"]),
        int(best["max_area"]),
        out_dir,
    )
    summary = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "best": best,
        "output_exp": args.output_exp,
        "target_dice": args.target_dice,
        "target_margin": float(mean["dice"] - args.target_dice),
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "apply_mean": mean, "target_margin": summary["target_margin"]}, indent=2))


if __name__ == "__main__":
    main()
