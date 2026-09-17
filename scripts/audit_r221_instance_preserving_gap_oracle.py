#!/usr/bin/env python3
"""R221 train/val-only instance-preserving gap oracle audit.

This is an upper-bound diagnostic for the "background seam connectivity"
route. It uses GT on original train/val only to ask whether suppressing R110
foreground pixels that fall inside true bone-gap background can improve gap,
boundary, and component diagnostics without sacrificing overlap/recall.

It writes CSV/JSON diagnostics only. It must not be used on clean-test-v2 for
threshold search or model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from run_r201_unified_eval import GtCache, boundary as r201_boundary, build_gt_cache, compute_metrics, read_binary_mask
from run_r209_component_preserving_feasibility_audit import analyze_component_matches, read_instance
from train_anchor_pixel_residual import names, resize_like


METRIC_KEYS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_iou",
    "boundary_f1",
    "surface_dice_2px",
    "surface_dice_5px",
    "hd95_px",
    "assd_px",
    "gap_region_fp_rate",
    "component_merge_rate",
    "component_count_mae",
    "component_delta_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R221 instance-preserving gap oracle on original train/val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gap-kernels", default="5,9,13")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument(
        "--metric-mode",
        default="fast",
        choices=["fast", "full"],
        help="fast skips Surface Dice/HD95/ASSD for feasibility screening; full uses R201 compute_metrics.",
    )
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--min-adjacent-instances", type=int, default=2)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-fracs", default="0.003,0.006,0.012")
    parser.add_argument("--recall-tol", type=float, default=1e-9)
    parser.add_argument("--dice-tol", type=float, default=5e-5)
    parser.add_argument("--iou-tol", type=float, default=5e-5)
    parser.add_argument("--min-boundary-gain", type=float, default=1e-4)
    parser.add_argument("--min-gap-gain", type=float, default=1e-4)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r221_instance_preserving_gap_oracle_val.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r221_instance_preserving_gap_oracle_val.json"))
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list[Any]:
    return [cast(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, split: str, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / split / "masks" / name


def instance_gap(instance: np.ndarray, kernel: int) -> np.ndarray:
    gt = instance > 0
    structure = np.ones((kernel, kernel), dtype=bool)
    return np.logical_and(ndimage.binary_dilation(gt, structure=structure), ~gt)


def adjacent_instance_count(cut_component: np.ndarray, instance: np.ndarray, ring_radius: int = 2) -> int:
    if not cut_component.any():
        return 0
    structure = np.ones((2 * ring_radius + 1, 2 * ring_radius + 1), dtype=bool)
    ring = np.logical_and(ndimage.binary_dilation(cut_component, structure=structure), ~cut_component)
    ids = np.unique(instance[ring])
    return int(sum(int(value) > 0 for value in ids))


def build_cut_variants(
    anchor: np.ndarray,
    instance: np.ndarray,
    kernel: int,
    min_adjacent_instances: int,
    min_cut_area: int,
    max_cut_fracs: list[float],
) -> dict[str, np.ndarray]:
    gt = instance > 0
    gap = instance_gap(instance, kernel)
    base_cut = np.logical_and(anchor, gap)
    variants: dict[str, np.ndarray] = {f"gap_all_k{kernel}": base_cut}

    labels, n_labels = ndimage.label(base_cut, structure=np.ones((3, 3), dtype=np.uint8))
    between_instances = np.zeros_like(base_cut, dtype=bool)
    for component_id in range(1, n_labels + 1):
        comp = labels == component_id
        if int(comp.sum()) < min_cut_area:
            continue
        if adjacent_instance_count(comp, instance) >= min_adjacent_instances:
            between_instances |= comp
    variants[f"gap_between_instances_k{kernel}"] = between_instances

    anchor_area = max(1, int(anchor.sum()))
    for max_frac in max_cut_fracs:
        max_pixels = max(min_cut_area, int(round(anchor_area * max_frac)))
        guarded = np.zeros_like(base_cut, dtype=bool)
        rows: list[tuple[int, int, np.ndarray]] = []
        labels2, n_labels2 = ndimage.label(between_instances, structure=np.ones((3, 3), dtype=np.uint8))
        for component_id in range(1, n_labels2 + 1):
            comp = labels2 == component_id
            area = int(comp.sum())
            if area >= min_cut_area:
                rows.append((adjacent_instance_count(comp, instance), area, comp))
        rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        used = 0
        for _adj, area, comp in rows:
            if used + area > max_pixels:
                continue
            guarded |= comp
            used += area
        variants[f"gap_between_instances_k{kernel}_maxfrac{max_frac:g}"] = guarded

    # Since gap excludes GT foreground, these cuts should not remove true GT.
    # Keep the assertion local and explicit; if labels are malformed, fail early.
    for name, cut in variants.items():
        if np.logical_and(cut, gt).any():
            raise AssertionError(f"{name} intersects GT foreground; gap oracle is not instance-preserving")
    return variants


def metric_delta(new: dict[str, float | None], old: dict[str, float | None], key: str) -> float | None:
    if new.get(key) is None or old.get(key) is None:
        return None
    return float(new[key]) - float(old[key])


def safe_divide(numer: float, denom: float, default: float = 1.0) -> float:
    return float(numer / denom) if denom > 0 else default


def fast_metrics(pred: np.ndarray, gt: np.ndarray, gt_cache: GtCache, boundary_kernel: int) -> dict[str, float | None]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    pred_boundary = r201_boundary(pred, boundary_kernel)
    gt_boundary = gt_cache.boundary
    b_tp = float(np.logical_and(pred_boundary, gt_boundary).sum())
    b_fp = float(np.logical_and(pred_boundary, ~gt_boundary).sum())
    b_fn = float(np.logical_and(gt_boundary, ~pred_boundary).sum())
    b_union = float(np.logical_or(pred_boundary, gt_boundary).sum())

    _, pred_n = ndimage.label(pred, structure=np.ones((3, 3), dtype=np.uint8))
    gap_pixels = float(gt_cache.gap_region.sum())
    gap_fp = float(np.logical_and(pred, gt_cache.gap_region).sum())
    gt_n = gt_cache.component_count

    return {
        "dice": safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": safe_divide(tp, tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "boundary_iou": safe_divide(b_tp, b_union, default=0.0),
        "boundary_f1": safe_divide(2.0 * b_tp, 2.0 * b_tp + b_fp + b_fn, default=0.0),
        "surface_dice_2px": None,
        "surface_dice_5px": None,
        "hd95_px": None,
        "assd_px": None,
        "gap_region_fp_rate": safe_divide(gap_fp, gap_pixels, default=0.0),
        "gap_region_precision": 1.0 - safe_divide(gap_fp, gap_pixels, default=0.0),
        "component_merge_rate": float(int(pred_n) < int(gt_n)),
        "component_count_mae": float(abs(int(pred_n) - int(gt_n))),
        "component_delta_mean": float(int(pred_n) - int(gt_n)),
    }


def eval_metrics(pred: np.ndarray, gt: np.ndarray, gt_cache: GtCache, args: argparse.Namespace) -> dict[str, float | None]:
    if args.metric_mode == "full":
        return compute_metrics(
            pred,
            gt,
            args.boundary_kernel,
            9,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
    return fast_metrics(pred, gt, gt_cache, args.boundary_kernel)


def build_fast_gt_cache(gt_dir: Path, boundary_kernel: int, gap_kernel: int) -> dict[str, GtCache]:
    cache: dict[str, GtCache] = {}
    gap_structure = np.ones((gap_kernel, gap_kernel), dtype=bool)
    for gt_path in sorted(gt_dir.glob("*.png")):
        gt = read_binary_mask(gt_path)
        gt_boundary = r201_boundary(gt, boundary_kernel)
        gt_gap = np.logical_and(ndimage.binary_dilation(gt, structure=gap_structure), ~gt)
        _, gt_n = ndimage.label(gt, structure=np.ones((3, 3), dtype=np.uint8))
        cache[gt_path.name] = GtCache(
            image=gt_path.name,
            mask=gt,
            boundary=gt_boundary,
            surface=np.zeros_like(gt, dtype=bool),
            surface_distance=None,
            gap_region=gt_gap,
            component_count=int(gt_n),
        )
    return cache


def component_match_metrics(mask: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> dict[str, float]:
    match = analyze_component_matches(mask, instance, min_overlap_frac)
    return {
        "pred_component_count_instance_match": float(match["pred_component_count"]),
        "gt_instance_count": float(match["gt_instance_count"]),
        "merged_pred_components": float(match["merged_pred_components"]),
        "gt_instances_covered": float(match["gt_instances_covered"]),
        "gt_instances_multi_covered": float(match["gt_instances_multi_covered"]),
    }


def evaluate_one_variant(
    name: str,
    cut: np.ndarray,
    anchor: np.ndarray,
    gt: np.ndarray,
    instance: np.ndarray,
    gt_cache: GtCache,
    args: argparse.Namespace,
    anchor_metrics: dict[str, float | None],
    anchor_match: dict[str, float],
) -> dict[str, Any]:
    pred = np.logical_and(anchor, ~cut)
    metrics = eval_metrics(pred, gt, gt_cache, args)
    match = component_match_metrics(pred, instance, args.min_overlap_frac)
    row: dict[str, Any] = {
        "variant": name,
        "cut_pixels": float(cut.sum()),
        "cut_frac_anchor": float(cut.sum() / max(1, anchor.sum())),
        "cut_gt_fg_pixels": float(np.logical_and(cut, gt).sum()),
        **{f"anchor_{k}": v for k, v in anchor_metrics.items()},
        **{f"r221_{k}": v for k, v in metrics.items()},
        **{f"anchor_{k}": v for k, v in anchor_match.items()},
        **{f"r221_{k}": v for k, v in match.items()},
    }
    for key in METRIC_KEYS:
        row[f"delta_{key}"] = metric_delta(metrics, anchor_metrics, key)
    for key in anchor_match:
        row[f"delta_{key}"] = float(match[key]) - float(anchor_match[key])
    row["passes_instance_preserving_guard"] = float(
        float(row["cut_gt_fg_pixels"]) == 0.0
        and float(row.get("delta_recall") or 0.0) >= -args.recall_tol
        and float(row.get("delta_dice") or 0.0) >= -args.dice_tol
        and float(row.get("delta_iou") or 0.0) >= -args.iou_tol
        and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
    )
    row["has_target_gain"] = float(
        float(row.get("delta_boundary_iou") or 0.0) >= args.min_boundary_gain
        or float(row.get("delta_boundary_f1") or 0.0) >= args.min_boundary_gain
        or float(row.get("delta_gap_region_fp_rate") or 0.0) <= -args.min_gap_gain
        or float(row.get("delta_merged_pred_components") or 0.0) < 0.0
    )
    row["promotable_oracle_case"] = float(row["passes_instance_preserving_guard"] > 0.5 and row["has_target_gain"] > 0.5)
    return row


def mean_or_none(values: list[Any]) -> float | None:
    finite = []
    for value in values:
        if value in (None, "", "None", "nan"):
            continue
        value = float(value)
        if np.isfinite(value):
            finite.append(value)
    return float(np.mean(finite)) if finite else None


def summarize_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_cols = [
        "cut_pixels",
        "cut_frac_anchor",
        "promotable_oracle_case",
        "passes_instance_preserving_guard",
        "has_target_gain",
        "delta_dice",
        "delta_iou",
        "delta_precision",
        "delta_recall",
        "delta_boundary_iou",
        "delta_boundary_f1",
        "delta_surface_dice_2px",
        "delta_hd95_px",
        "delta_assd_px",
        "delta_gap_region_fp_rate",
        "delta_component_merge_rate",
        "delta_component_count_mae",
        "delta_merged_pred_components",
        "r221_dice",
        "r221_iou",
        "r221_recall",
        "r221_boundary_iou",
        "r221_gap_region_fp_rate",
        "r221_component_count_mae",
        "r221_merged_pred_components",
    ]
    out = {key: mean_or_none([row.get(key) for row in rows]) for key in metric_cols}
    out["num_images"] = len(rows)
    out["num_promotable_oracle_cases"] = int(sum(float(row.get("promotable_oracle_case") or 0.0) > 0.5 for row in rows))
    out["num_guard_pass"] = int(sum(float(row.get("passes_instance_preserving_guard") or 0.0) > 0.5 for row in rows))
    out["num_target_gain"] = int(sum(float(row.get("has_target_gain") or 0.0) > 0.5 for row in rows))
    return out


def summarize(rows: list[dict[str, Any]], missing: list[str], args: argparse.Namespace) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    summaries = {key: summarize_variant(value) for key, value in sorted(by_variant.items())}

    feasible = []
    for variant, summary in summaries.items():
        if (
            float(summary.get("passes_instance_preserving_guard") or 0.0) >= 0.95
            and float(summary.get("delta_recall") or 0.0) >= -args.recall_tol
            and float(summary.get("delta_dice") or 0.0) >= -args.dice_tol
            and float(summary.get("delta_iou") or 0.0) >= -args.iou_tol
            and float(summary.get("delta_component_count_mae") or 0.0) <= 0.0
            and (
                float(summary.get("delta_boundary_iou") or 0.0) >= args.min_boundary_gain
                or float(summary.get("delta_gap_region_fp_rate") or 0.0) <= -args.min_gap_gain
                or float(summary.get("delta_merged_pred_components") or 0.0) < 0.0
            )
        ):
            feasible.append(variant)

    return {
        "run_id": "R221-instance-preserving-gap-oracle",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "train_val_oracle_upper_bound",
        "purpose": "test whether true gap/background suppression can improve seam/boundary diagnostics without overlap or component harm",
        "num_variant_rows": len(rows),
        "num_images_evaluated": len({row["image"] for row in rows}),
        "num_missing_anchor_masks": len(missing),
        "missing_anchor_masks": missing,
        "variant_summaries": summaries,
        "feasible_variants": feasible,
        "decision": "go_train_gt_free_seam_background_scorer" if feasible else "no_go_oracle_not_sufficient",
        "warning": "Oracle diagnostic uses GT-derived gap/background maps on original train/val only; not deployable and not R201 clean-test-v2 evidence.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split = args.split
    gt_dir = args.raw_root / args.dataset / f"{split}_labels"
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if args.metric_mode == "full":
        gt_cache = build_gt_cache(gt_dir, args.boundary_kernel, 9)
    else:
        gt_cache = build_fast_gt_cache(gt_dir, args.boundary_kernel, 9)
    case_names = names(args.raw_root, args.dataset, split)[: args.limit or None]
    gap_kernels = parse_nums(args.gap_kernels, int)
    max_cut_fracs = parse_nums(args.max_cut_fracs, float)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in tqdm(case_names, desc=f"r221/gap-oracle/{split}"):
        mask_path = anchor_path(args, split, name)
        if not mask_path.exists():
            missing.append(name)
            continue
        instance = read_instance(gt_dir / name)
        gt = instance > 0
        anchor = resize_like(read_binary_mask(mask_path), gt.shape)
        anchor_metrics = eval_metrics(anchor, gt, gt_cache[name], args)
        anchor_match = component_match_metrics(anchor, instance, args.min_overlap_frac)
        for kernel in gap_kernels:
            variants = build_cut_variants(
                anchor=anchor,
                instance=instance,
                kernel=kernel,
                min_adjacent_instances=args.min_adjacent_instances,
                min_cut_area=args.min_cut_area,
                max_cut_fracs=max_cut_fracs,
            )
            for variant_name, cut in variants.items():
                row = evaluate_one_variant(variant_name, cut, anchor, gt, instance, gt_cache[name], args, anchor_metrics, anchor_match)
                row["image"] = name
                row["split"] = split
                rows.append(row)

    write_csv(args.output_csv, rows)
    report = summarize(rows, missing, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "feasible_variants": report["feasible_variants"], "outputs": [str(args.output_csv), str(args.output_json)]}, indent=2))


if __name__ == "__main__":
    main()
