#!/usr/bin/env python3
"""R233 merged seam-probability mask editor.

Original-val mask-level development experiment. R233 combines the R230
seed-pair corridor candidates and R232 probability-ridge candidates under the
same strict safety gate. It writes isolated val masks only and must not be used
on clean-test-v2 for tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, quick_metrics
from audit_r228_oracle_pixel_seam_localizer import as_float, fit_model, predict_prob, sample_training
from run_r229_seam_prob_mask_editor import (
    anchor_path,
    candidate_rows_for_case as corridor_candidate_rows_for_case,
    gate_candidate,
    load_case,
    metric_delta,
    write_csv,
)
from run_r232_seam_prob_ridge_mask_editor import candidate_rows_for_case as ridge_candidate_rows_for_case
from train_anchor_pixel_residual import names, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R233 merged seam-probability mask editor on original val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r233_merged_seam_prob_mask_editor")
    parser.add_argument("--split", default="val")
    parser.add_argument("--r224-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--source-candidate-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--source-mode", choices=["source-csv", "all-anchors"], default="source-csv")
    parser.add_argument("--limit-source", type=int, default=32)
    parser.add_argument("--limit-train-rows", type=int, default=300)
    parser.add_argument("--samples-per-image", type=int, default=1024)
    parser.add_argument("--positive-oversample", type=int, default=512)

    parser.add_argument("--max-components-per-image", type=int, default=1)
    parser.add_argument("--max-bg-components", type=int, default=3)
    parser.add_argument("--max-pairs-per-component", type=int, default=2)
    parser.add_argument("--corridor-radii", default="1,2")
    parser.add_argument("--action-fracs", default="0.25,0.35,0.50")

    parser.add_argument("--ridge-prob-thresholds", default="0.75,0.85")
    parser.add_argument("--ridge-action-fracs", default="0.20,0.30")
    parser.add_argument("--ridge-sigma", type=float, default=0.75)
    parser.add_argument("--ridge-max-dist", type=float, default=5.0)
    parser.add_argument("--ridge-dilate-radii", default="0")
    parser.add_argument("--min-ridge-area", type=int, default=3)
    parser.add_argument("--max-ridges-per-component", type=int, default=4)
    parser.add_argument("--max-candidates-per-image", type=int, default=96)

    parser.add_argument("--seam-prob-threshold", type=float, default=0.4)
    parser.add_argument("--seam-prob-p90-threshold", type=float, default=0.8)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.006)
    parser.add_argument("--max-cuts-per-image", type=int, default=1)
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--boundary-tol", type=float, default=0.0)
    parser.add_argument("--allow-component-mae-worsen", action="store_true")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r233_merged_seam_prob_pilot32_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r233_merged_seam_prob_pilot32_per_image.csv"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r233_merged_seam_prob_pilot32_candidates.csv"))
    parser.add_argument("--seed", type=int, default=202607233)
    return parser.parse_args()


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
        "mean_r233_dice": mean_value(records, "r233_dice"),
        "mean_delta_dice": mean_value(records, "delta_dice"),
        "mean_delta_iou": mean_value(records, "delta_iou"),
        "mean_delta_recall": mean_value(records, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(records, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(records, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(records, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(records, "delta_component_count_mae"),
        "mean_cut_pixels": mean_value(records, "cut_pixels"),
    }


def application_names(args: argparse.Namespace) -> list[str]:
    if args.source_mode == "all-anchors":
        anchor_dir = args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks"
        ordered = sorted(path.name for path in anchor_dir.glob("*.png"))
        if not ordered:
            ordered = names(args.raw_root, args.dataset, args.split)
        return ordered[: args.limit_source or None]
    if args.source_candidate_csv.exists():
        ordered: list[str] = []
        seen: set[str] = set()
        with args.source_candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("image") or "")
                if name and name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered[: args.limit_source or None]
    return names(args.raw_root, args.dataset, args.split)[: args.limit_source or None]


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for row in rows:
        cut = row.get("_cut")
        if cut is None:
            out.append(row)
            continue
        key = np.packbits(cut.ravel()).tobytes()
        if key in seen:
            row = dict(row)
            row.pop("_cut", None)
            row["accepted"] = 0.0
            row["reject_reason"] = "duplicate_cut"
            out.append(row)
            continue
        seen.add(key)
        out.append(row)
    return out


def candidate_rows_for_case(args: argparse.Namespace, image: np.ndarray, instance: np.ndarray, gt: np.ndarray, anchor: np.ndarray, prob: np.ndarray, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in corridor_candidate_rows_for_case(args, image, instance, gt, anchor, prob, name):
        row["candidate_family"] = "corridor"
        rows.append(row)
    for row in ridge_candidate_rows_for_case(args, instance, gt, anchor, prob, name):
        row["candidate_family"] = "prob_ridge"
        accepted, reason = gate_candidate(row, args)
        row["accepted"] = float(accepted)
        row["reject_reason"] = reason
        rows.append(row)
    return dedupe_rows(rows)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x, y, train_info = sample_training(args)
    model = fit_model(args, x, y)
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.split / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_mask in output_dir.glob("*.png"):
        old_mask.unlink()
    per_image: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for name in application_names(args):
        if not anchor_path(args, name).exists():
            continue
        image, instance, gt, anchor = load_case(args, name)
        prob = predict_prob(model, image, anchor)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        current = anchor.copy()
        accepted_cuts = []
        rows = candidate_rows_for_case(args, image, instance, gt, anchor, prob, name)
        rows.sort(
            key=lambda row: (
                -as_float(row, "accepted"),
                -as_float(row, "delta_boundary_iou"),
                as_float(row, "delta_gap_region_fp_rate"),
                -as_float(row, "seam_prob_mean"),
            )
        )
        for row in rows:
            cut = row.pop("_cut", None)
            if cut is None:
                candidate_rows.append(row)
                continue
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
            record[f"r233_{metric}"] = value
            record[f"delta_{metric}"] = metric_delta(pred_metrics, anchor_metrics, metric)
        per_image.append(record)
    return {
        "run_id": "R233-merged-seam-prob-mask-editor",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "clean_test_v2_used": False,
        "writes_masks": True,
        "evidence_level": "original_val_mask_level_development",
        "candidate_families": ["corridor", "prob_ridge"],
        "source_mode": args.source_mode,
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
