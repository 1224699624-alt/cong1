#!/usr/bin/env python3
"""R232 seam-probability ridge mask editor.

Original-val mask-level development experiment. R232 keeps the R230 safety
gate but replaces seed-pair corridor candidates with candidates generated
directly from high-probability seam ridges in the R228/R229 pixel scorer map.

This script writes isolated val masks only. Do not use clean-test-v2 here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, is_hard_risk, is_quick_useful, quick_metrics
from audit_r228_oracle_pixel_seam_localizer import as_float, fit_model, predict_prob, sample_training
from run_r229_seam_prob_mask_editor import (
    anchor_path,
    component_labels,
    gate_candidate,
    load_case,
    metric_delta,
    parse_float_list,
    safe_divide,
    source_names,
    write_csv,
)
from train_anchor_pixel_residual import write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R232 seam-probability ridge mask editor on original val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r232_seam_prob_ridge_mask_editor")
    parser.add_argument("--split", default="val")
    parser.add_argument("--r224-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--source-candidate-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--limit-source", type=int, default=32)
    parser.add_argument("--limit-train-rows", type=int, default=300)
    parser.add_argument("--samples-per-image", type=int, default=1024)
    parser.add_argument("--positive-oversample", type=int, default=512)
    parser.add_argument("--max-components-per-image", type=int, default=2)
    parser.add_argument("--ridge-prob-thresholds", default="0.55,0.65,0.75,0.85")
    parser.add_argument("--ridge-action-fracs", default="0.20,0.30,0.45,0.60")
    parser.add_argument("--ridge-sigma", type=float, default=0.75)
    parser.add_argument("--ridge-max-dist", type=float, default=5.0)
    parser.add_argument("--ridge-dilate-radii", default="0,1")
    parser.add_argument("--seam-prob-threshold", type=float, default=0.4)
    parser.add_argument("--seam-prob-p90-threshold", type=float, default=0.8)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--min-ridge-area", type=int, default=3)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.006)
    parser.add_argument("--max-ridges-per-component", type=int, default=12)
    parser.add_argument("--max-candidates-per-image", type=int, default=80)
    parser.add_argument("--max-cuts-per-image", type=int, default=1)
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--boundary-tol", type=float, default=0.0)
    parser.add_argument("--allow-component-mae-worsen", action="store_true")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r232_seam_prob_ridge_pilot32_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r232_seam_prob_ridge_pilot32_per_image.csv"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r232_seam_prob_ridge_pilot32_candidates.csv"))
    parser.add_argument("--seed", type=int, default=202607232)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(records),
        "num_images": len({str(row.get("image")) for row in records}),
        "num_edited": int(sum(as_float(row, "edited") > 0.5 for row in records)),
        "mean_anchor_dice": mean_value(records, "anchor_dice"),
        "mean_r232_dice": mean_value(records, "r232_dice"),
        "mean_delta_dice": mean_value(records, "delta_dice"),
        "mean_delta_iou": mean_value(records, "delta_iou"),
        "mean_delta_recall": mean_value(records, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(records, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(records, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(records, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(records, "delta_component_count_mae"),
        "mean_cut_pixels": mean_value(records, "cut_pixels"),
    }


def slenderness(mask: np.ndarray) -> float:
    area = int(mask.sum())
    if area <= 0:
        return 0.0
    ys, xs = np.where(mask)
    height = int(ys.max() - ys.min() + 1)
    width = int(xs.max() - xs.min() + 1)
    return float(max(height, width) / max(1, min(height, width)))


def candidate_cut_from_ridge(
    prob: np.ndarray,
    anchor: np.ndarray,
    ridge: np.ndarray,
    action_frac: float,
    dilate_radius: int,
    min_cut_area: int,
    max_pixels: int,
) -> np.ndarray:
    raw = ridge.astype(bool)
    if dilate_radius > 0:
        raw = ndimage.binary_dilation(raw, structure=np.ones((2 * dilate_radius + 1, 2 * dilate_radius + 1), dtype=bool))
    raw &= anchor
    n = int(raw.sum())
    if n < min_cut_area:
        return np.zeros_like(anchor, dtype=bool)
    keep_n = max(min_cut_area, int(round(n * action_frac)))
    keep_n = min(n, max_pixels)
    ys, xs = np.where(raw)
    vals = prob[ys, xs]
    order = np.argsort(vals)[::-1][:keep_n]
    cut = np.zeros_like(anchor, dtype=bool)
    cut[ys[order], xs[order]] = True
    if int(cut.sum()) > max_pixels:
        cut = np.zeros_like(anchor, dtype=bool)
        order = order[:max_pixels]
        cut[ys[order], xs[order]] = True
    return cut


def ridge_regions_for_component(args: argparse.Namespace, prob: np.ndarray, component: np.ndarray) -> list[tuple[float, np.ndarray]]:
    dist_in = ndimage.distance_transform_edt(component)
    eligible = component & (dist_in <= args.ridge_max_dist)
    if not eligible.any():
        return []
    prob_s = ndimage.gaussian_filter(prob.astype(np.float32), sigma=args.ridge_sigma) if args.ridge_sigma > 0 else prob.astype(np.float32)
    regions: list[tuple[float, np.ndarray]] = []
    for threshold in parse_float_list(args.ridge_prob_thresholds):
        high = eligible & (prob_s >= threshold)
        labels, n_labels = ndimage.label(high, structure=np.ones((3, 3), dtype=np.uint8))
        for label_id in range(1, n_labels + 1):
            ridge = labels == label_id
            area = int(ridge.sum())
            if area < args.min_ridge_area:
                continue
            score = float(np.percentile(prob_s[ridge], 90)) + 0.02 * slenderness(ridge)
            regions.append((score, ridge))
    regions.sort(key=lambda item: item[0], reverse=True)
    return regions[: args.max_ridges_per_component]


def candidate_rows_for_case(
    args: argparse.Namespace,
    instance: np.ndarray,
    gt: np.ndarray,
    anchor: np.ndarray,
    prob: np.ndarray,
    name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
    anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
    comp_labels, n_components = component_labels(anchor)
    component_order = [(int((comp_labels == comp_id).sum()), comp_id) for comp_id in range(1, n_components + 1)]
    component_order.sort(reverse=True)
    max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
    seen: set[bytes] = set()
    for _area, comp_id in component_order[: args.max_components_per_image]:
        component = comp_labels == comp_id
        comp_area = int(component.sum())
        if comp_area < args.min_component_area or comp_area > args.max_component_area:
            continue
        ridges = ridge_regions_for_component(args, prob, component)
        for ridge_score, ridge in ridges:
            for action_frac in parse_float_list(args.ridge_action_fracs):
                for dilate_radius in parse_int_list(args.ridge_dilate_radii):
                    cut = candidate_cut_from_ridge(prob, anchor, ridge, action_frac, dilate_radius, args.min_cut_area, max_pixels)
                    cut_area = int(cut.sum())
                    if cut_area < args.min_cut_area or cut_area > max_pixels:
                        continue
                    key = np.packbits(cut.ravel()).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    trial = anchor & ~cut
                    trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                    row: dict[str, Any] = {
                        "image": name,
                        "candidate_family": "prob_ridge",
                        "pred_component_id": float(comp_id),
                        "ridge_score": float(ridge_score),
                        "ridge_area": float(ridge.sum()),
                        "ridge_slenderness": slenderness(ridge),
                        "cut_area": float(cut_area),
                        "action_frac": float(action_frac),
                        "dilate_radius": float(dilate_radius),
                        "seam_prob_mean": float(np.mean(prob[cut])),
                        "seam_prob_p90": float(np.percentile(prob[cut], 90)),
                        "cut_gt_fg_frac": safe_divide(float(np.logical_and(cut, instance > 0).sum()), float(cut_area)),
                        "cut_gt_gap_frac": safe_divide(float(np.logical_and(cut, instance == 0).sum()), float(cut_area)),
                        "_cut": cut,
                    }
                    for metric, value in trial_metrics.items():
                        row[f"candidate_{metric}"] = value
                        row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                    row["quick_useful"] = float(is_quick_useful(row))
                    row["hard_risk"] = float(is_hard_risk(row))
                    row["safe_gap_positive"] = float(row["cut_gt_fg_frac"] <= 0.25 and row["cut_gt_gap_frac"] >= 0.75)
                    row["overerosion_proxy"] = float(row["cut_gt_fg_frac"] > 0.5)
                    accepted, reason = gate_candidate(row, args)
                    row["accepted"] = float(accepted)
                    row["reject_reason"] = reason
                    rows.append(row)
                    if len(rows) >= args.max_candidates_per_image:
                        return rows
    return rows


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x, y, train_info = sample_training(args)
    model = fit_model(args, x, y)
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.split / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_mask in output_dir.glob("*.png"):
        old_mask.unlink()
    per_image: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for name in source_names(args):
        if not anchor_path(args, name).exists():
            continue
        image, instance, gt, anchor = load_case(args, name)
        prob = predict_prob(model, image, anchor)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        current = anchor.copy()
        accepted_cuts = []
        rows = candidate_rows_for_case(args, instance, gt, anchor, prob, name)
        rows.sort(key=lambda row: (-as_float(row, "accepted"), -as_float(row, "seam_prob_mean"), -as_float(row, "delta_boundary_iou")))
        for row in rows:
            cut = row.pop("_cut")
            if as_float(row, "accepted") <= 0.5:
                candidate_rows.append(row)
                continue
            if len(accepted_cuts) >= args.max_cuts_per_image:
                row["accepted"] = 0.0
                row["reject_reason"] = "max_cuts_reached"
                candidate_rows.append(row)
                continue
            cut = cut & current
            if int(cut.sum()) < args.min_cut_area:
                row["accepted"] = 0.0
                row["reject_reason"] = "cut_empty_after_previous"
                candidate_rows.append(row)
                continue
            current = current & ~cut
            accepted_cuts.append(cut)
            candidate_rows.append(row)
        pred_metrics = quick_metrics(current, gt, gt_cache, args.boundary_kernel)
        write_mask(output_dir / name, current)
        record: dict[str, Any] = {
            "image": name,
            "edited": float(bool(accepted_cuts)),
            "num_accepted_cuts": float(len(accepted_cuts)),
            "cut_pixels": float(sum(int(cut.sum()) for cut in accepted_cuts)),
        }
        for metric, value in anchor_metrics.items():
            record[f"anchor_{metric}"] = value
        for metric, value in pred_metrics.items():
            record[f"r232_{metric}"] = value
            record[f"delta_{metric}"] = metric_delta(pred_metrics, anchor_metrics, metric)
        per_image.append(record)
    return {
        "run_id": "R232-seam-prob-ridge-mask-editor",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "clean_test_v2_used": False,
        "writes_masks": True,
        "evidence_level": "original_val_mask_level_development",
        "train_info": {**train_info, "num_samples": int(len(y)), "positive_sample_rate": float(np.mean(y))},
        "output_mask_dir": str(output_dir),
        "num_images": len(per_image),
        "summary": summarize_records(per_image),
        "per_image": per_image,
        "num_candidate_rows": len(candidate_rows),
        "num_accepted_candidates": int(sum(as_float(row, "accepted") > 0.5 for row in candidate_rows)),
        "warning": "Original-val development only; not clean-test-v2 or R201 final evidence.",
    }, candidate_rows


def main() -> None:
    args = parse_args()
    report, candidates = run(args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.per_image_csv, report["per_image"])
    write_csv(args.candidate_csv, candidates)
    print(json.dumps({"summary_json": str(args.summary_json), "per_image_csv": str(args.per_image_csv), "candidate_csv": str(args.candidate_csv), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
