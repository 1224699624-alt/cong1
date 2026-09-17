#!/usr/bin/env python3
"""R185 component-accepted layout bridge cuts.

This is a train/val-selected postprocess for a strong anchor mask. It differs
from earlier bridge cutters by requiring non-zero validation edits and by
rejecting per-image edits that hurt component-count error when labels are
available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from apply_fast_layout_bridge_cut import repair_mask
from evaluate_masks import compute_metrics


TARGET = 0.9317660066557425
R110_CLEAN_DICE = 0.9177231563529792
R110_CLEAN_BOUNDARY_IOU = 0.25189601044085763


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R185 component-accepted layout cut.")
    parser.add_argument("--tune-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--tune-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--output-exp", default="r185_component_accepted_layout_cut")
    parser.add_argument("--val-summary-json", default="outputs/analysis/r185_component_accepted_layout_cut_val_summary.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r185_component_accepted_layout_cut_clean_test_v2_metrics.json")
    parser.add_argument("--result-json", default="outputs/analysis/r185_component_accepted_layout_cut_result_summary.json")
    parser.add_argument("--peak-fracs", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--valley-fracs", default="0.12,0.16,0.22,0.28")
    parser.add_argument("--min-areas", default="8,16,24,32")
    parser.add_argument("--max-remove-fracs", default="0.0005,0.001,0.002,0.004")
    parser.add_argument("--large-area-percentiles", default="40,50,60")
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--min-val-edited-frac", type=float, default=1e-6)
    parser.add_argument("--val-dice-drop-tol", type=float, default=0.0005)
    parser.add_argument("--val-boundary-drop-tol", type=float, default=0.0)
    parser.add_argument("--val-component-tol", type=float, default=0.0)
    parser.add_argument("--allow-clean-test-on-gate-fail", action="store_true")
    return parser.parse_args()


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0)


def separation_band(gt: np.ndarray, radius: int = 7) -> np.ndarray:
    kernel = np.ones((radius, radius), np.uint8)
    dilated = cv2.dilate(gt.astype(np.uint8), kernel, iterations=1) > 0
    return dilated & ~gt


def add_structure_metrics(rec: dict[str, float], pred: np.ndarray, gt: np.ndarray, edited: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    sep = separation_band(gt)
    rec["pred_component_count"] = float(pred_count)
    rec["gt_component_count"] = float(gt_count)
    rec["component_count_error"] = float(abs(pred_count - gt_count))
    rec["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    rec["sep_fp_rate"] = float(np.logical_and(pred, sep).sum() / max(1, int(sep.sum())))
    rec["edited_frac"] = float(edited.sum() / max(1, edited.size))
    return rec


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {k: float(np.mean([float(r[k]) for r in records])) for k in keys}


def load_items(raw_root: Path, dataset: str, split: str, anchor_root: Path, limit: int) -> list[dict[str, object]]:
    label_dir = raw_root / dataset / f"{split}_labels"
    items = []
    paths = sorted(label_dir.glob("*.png"))
    if limit and limit > 0:
        paths = paths[:limit]
    for gt_path in paths:
        anchor_path = anchor_root / gt_path.name
        if not anchor_path.exists():
            raise FileNotFoundError(f"anchor mask missing: {anchor_path}")
        gt = read_mask(gt_path)
        anchor = resize_like(read_mask(anchor_path), gt.shape)
        items.append({"name": gt_path.name, "gt": gt, "anchor": anchor})
    if not items:
        raise FileNotFoundError(label_dir)
    return items


def anchor_root(ablations_root: Path, exp: str, dataset: str, split: str) -> Path:
    root = ablations_root / exp / dataset / split / "masks"
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def accept_with_gt(anchor: np.ndarray, candidate: np.ndarray, gt: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    anchor_rec = compute_metrics(anchor, gt, args.boundary_kernel)
    cand_rec = compute_metrics(candidate, gt, args.boundary_kernel)
    anchor_comp = abs(component_count(anchor) - component_count(gt))
    cand_comp = abs(component_count(candidate) - component_count(gt))
    if cand_comp > anchor_comp + args.val_component_tol:
        return anchor
    if cand_rec["dice"] < anchor_rec["dice"] - args.val_dice_drop_tol:
        return anchor
    if cand_rec["boundary_iou"] < anchor_rec["boundary_iou"] - args.val_boundary_drop_tol:
        return anchor
    return candidate


def apply_config_to_items(
    items: list[dict[str, object]],
    cfg: dict[str, float],
    args: argparse.Namespace,
    use_gt_acceptance: bool,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for item in items:
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        candidate = repair_mask(
            anchor,
            float(cfg["peak_frac"]),
            float(cfg["valley_frac"]),
            int(cfg["min_area"]),
            float(cfg["max_remove_frac"]),
            float(cfg["large_area_percentile"]),
        )
        pred = accept_with_gt(anchor, candidate, gt, args) if use_gt_acceptance else candidate
        edited = pred != anchor
        if write_dir is not None:
            write_mask(write_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, args.boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt, edited)
        rec["image"] = str(item["name"])
        records.append(rec)
    return mean_records(records), records


def grid_configs(args: argparse.Namespace) -> list[dict[str, float]]:
    configs = []
    for peak_frac in parse_floats(args.peak_fracs):
        for valley_frac in parse_floats(args.valley_fracs):
            for min_area in parse_ints(args.min_areas):
                for max_remove_frac in parse_floats(args.max_remove_fracs):
                    for large_area_percentile in parse_floats(args.large_area_percentiles):
                        configs.append(
                            {
                                "peak_frac": float(peak_frac),
                                "valley_frac": float(valley_frac),
                                "min_area": float(min_area),
                                "max_remove_frac": float(max_remove_frac),
                                "large_area_percentile": float(large_area_percentile),
                            }
                        )
    return configs


def gate_pass(mean: dict[str, float], anchor_mean: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        mean["edited_frac"] >= args.min_val_edited_frac
        and mean["dice"] >= anchor_mean["dice"] - args.val_dice_drop_tol
        and mean["boundary_iou"] >= anchor_mean["boundary_iou"] - args.val_boundary_drop_tol
        and mean["component_count_error"] <= anchor_mean["component_count_error"] + args.val_component_tol
        and mean["false_bridge_flag"] <= anchor_mean["false_bridge_flag"] + 1e-9
    )


def main() -> None:
    args = parse_args()
    ab_root = Path(args.ablations_root)
    tune_items = load_items(
        Path(args.tune_raw_root),
        args.tune_dataset,
        args.tune_split,
        anchor_root(ab_root, args.anchor_exp, args.tune_dataset, args.tune_split),
        args.max_val_images,
    )

    zero_cfg = {
        "peak_frac": 1.0,
        "valley_frac": 0.0,
        "min_area": 999999.0,
        "max_remove_frac": 0.0,
        "large_area_percentile": 100.0,
    }
    anchor_mean, anchor_records = apply_config_to_items(tune_items, zero_cfg, args, use_gt_acceptance=False)

    grid = []
    best = None
    for cfg in grid_configs(args):
        mean, _ = apply_config_to_items(tune_items, cfg, args, use_gt_acceptance=True)
        passed = gate_pass(mean, anchor_mean, args)
        item = {"cfg": cfg, "mean": mean, "passes_gate": passed}
        grid.append(item)
        if not passed:
            continue
        score = (
            mean["dice"] - anchor_mean["dice"],
            mean["boundary_iou"] - anchor_mean["boundary_iou"],
            -(mean["component_count_error"] - anchor_mean["component_count_error"]),
            -(mean["false_bridge_flag"] - anchor_mean["false_bridge_flag"]),
            mean["edited_frac"],
        )
        if best is None or score > best["score"]:
            best = item | {"score": score}

    val_summary = {
        "run_id": "R185",
        "anchor_exp": args.anchor_exp,
        "apply_anchor_exp": args.apply_anchor_exp,
        "anchor_mean": anchor_mean,
        "anchor_per_image": anchor_records,
        "gate_pass": best is not None,
        "gate_rules": {
            "min_val_edited_frac": args.min_val_edited_frac,
            "val_dice_drop_tol": args.val_dice_drop_tol,
            "val_boundary_drop_tol": args.val_boundary_drop_tol,
            "val_component_tol": args.val_component_tol,
        },
        "best": best,
        "grid": grid,
    }
    Path(args.val_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_summary_json).write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

    result = {
        "run_id": "R185",
        "status": "val_gate_failed",
        "val_summary_path": args.val_summary_json,
        "metrics_path": args.metrics_json,
        "target_dice": TARGET,
        "r110_clean_dice": R110_CLEAN_DICE,
        "r110_clean_boundary_iou": R110_CLEAN_BOUNDARY_IOU,
    }
    if best is None and not args.allow_clean_test_on_gate_fail:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if best is None:
        raise RuntimeError("gate failed but clean-test application was forced")

    apply_items = load_items(
        Path(args.apply_raw_root),
        args.apply_dataset,
        args.apply_split,
        anchor_root(ab_root, args.apply_anchor_exp, args.apply_dataset, args.apply_split),
        args.max_apply_images,
    )
    out_dir = ab_root / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = apply_config_to_items(apply_items, best["cfg"], args, use_gt_acceptance=False, write_dir=out_dir)
    metrics = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "best": best,
        "output_exp": args.output_exp,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dice = float(mean.get("dice", 0.0))
    result.update(
        {
            "status": "target_met" if dice >= TARGET else ("new_best_below_target" if dice > R110_CLEAN_DICE else "below_best"),
            "mean": mean,
            "target_margin": dice - TARGET,
            "dice_delta_vs_r110": dice - R110_CLEAN_DICE,
            "boundary_iou_delta_vs_r110": float(mean.get("boundary_iou", 0.0)) - R110_CLEAN_BOUNDARY_IOU,
            "val_best": best,
        }
    )
    Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
