#!/usr/bin/env python3
"""R251 dense reverse-background selector on original validation candidates.

The scorer is trained from original-train inverted bone GT inside GT-free
R244-style candidate corridors. Original-val inference uses image and anchor
only. GT is read after candidate generation solely for labels and audit metrics.
No masks are written and clean-test-v2 is forbidden.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import sklearn
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, quick_metrics
from audit_r237_component_merge_targeted_candidates import local_bg_connectivity
from audit_r244_gtfree_merge_targeted_candidates import (
    component_labels,
    component_match_metrics,
    cut_features,
    is_risk,
    is_safe_useful,
    metric_delta,
    parse_float_list,
    safe_divide,
)
from run_r209_component_preserving_feasibility_audit import read_instance
from run_r210_f1_neck_candidate_gate import candidate_components
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like
from train_r248_r244_context_crop_scorer import as_float, write_csv
from train_r251_reverse_background_channel import (
    anchor_path,
    collect_training,
    fit_model,
    predict_background_prob,
    select_case_names,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R251 reverse-background selector on original val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit-train", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-scan-images", type=int, default=160)
    parser.add_argument("--dist-percentiles", default="6,8,10,12")
    parser.add_argument("--min-cut-area", type=int, default=3)
    parser.add_argument("--max-cut-frac", type=float, default=0.008)
    parser.add_argument("--train-max-candidates-per-percentile", type=int, default=32)
    parser.add_argument("--max-candidates-generated", type=int, default=64)
    parser.add_argument("--max-components-per-image", type=int, default=5)
    parser.add_argument("--hard-negative-ring", type=int, default=5)
    parser.add_argument("--min-unique-positive", type=int, default=8)
    parser.add_argument("--min-unique-corridor-negative", type=int, default=8)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--positive-samples-per-image", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2-regularization", type=float, default=0.03)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--mean-thresholds", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95,0.98")
    parser.add_argument("--p10-thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--max-actions-per-image", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=202607251)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r251_reverse_background_candidate_selector_fullval.json"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r251_reverse_background_candidate_selector_fullval_candidates.csv"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r251_reverse_background_candidate_selector_fullval_grid.csv"))
    parser.add_argument("--progress-json", type=Path, default=Path("outputs/analysis/r251_reverse_background_candidate_selector_fullval_progress.json"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/r251_reverse_background_candidate_selector_fullval.pkl"))
    return parser.parse_args()


def validate_scope(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset != "TSRS_RSNA-Epiphysis":
        raise ValueError("R251 is locked to TSRS_RSNA-Epiphysis.")
    if args.train_split != "train" or args.split != "val":
        raise ValueError("R251 requires original train -> original val.")
    if args.max_actions_per_image != 1:
        raise ValueError("R251 single-cut audit requires --max-actions-per-image 1.")
    raw_root = args.raw_root.resolve()
    dataset_root = (raw_root / args.dataset).resolve()
    if raw_root.name != "raw" or "clean_test" in str(dataset_root).lower() or "articular-surface" in str(dataset_root).lower():
        raise ValueError("R251 requires the original data/raw Epiphysis root.")
    for split in ("train", "val"):
        if not (dataset_root / split).is_dir() or not (dataset_root / f"{split}_labels").is_dir():
            raise FileNotFoundError(f"Missing original {split} image/label directories.")
    return {"dataset_root": str(dataset_root), "train_split": "train", "eval_split": "val", "clean_test_v2_used": False}


def train_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=args.dataset,
        raw_root=args.raw_root,
        ablations_root=args.ablations_root,
        anchor_exp=args.anchor_exp,
        train_split="train",
        eval_split="train",
        limit_train=args.limit_train,
        limit_eval=args.limit_train,
        max_scan_images=args.max_scan_images,
        dist_percentiles=args.dist_percentiles,
        min_cut_area=args.min_cut_area,
        max_cut_frac=args.max_cut_frac,
        max_candidates_generated=args.train_max_candidates_per_percentile,
        hard_negative_ring=args.hard_negative_ring,
        min_unique_positive=args.min_unique_positive,
        min_unique_corridor_negative=args.min_unique_corridor_negative,
        samples_per_image=args.samples_per_image,
        positive_samples_per_image=args.positive_samples_per_image,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
        seed=args.seed,
    )


def load_val_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / "val_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, "val", name))
    anchor = resize_like(read_mask(anchor_path(args, "val", name)), gt.shape)
    return image, instance, gt, anchor


def build_candidates(args: argparse.Namespace, model: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    processed_images: list[str] = []
    zero_candidate_images: list[str] = []
    case_names = names(args.raw_root, args.dataset, "val")[: args.limit or None]
    for index, name in enumerate(tqdm(case_names, desc="r251/reverse-bg-candidates/val"), start=1):
        if not anchor_path(args, "val", name).exists():
            skipped.append({"image": name, "reason": "missing_anchor"})
            continue
        try:
            image, instance, gt, anchor = load_val_case(args, name)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"image": name, "reason": type(exc).__name__, "message": str(exc)})
            continue
        processed_images.append(name)
        rows_before = len(rows)
        prob = predict_background_prob(model, image, anchor)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        anchor_match = component_match_metrics(anchor, instance, args.min_overlap_frac)
        comp_labels, _ = component_labels(anchor)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        seen: set[bytes] = set()
        for dist_percentile in parse_float_list(args.dist_percentiles):
            cuts = candidate_components(anchor, image, dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
            for cut_rank, raw_cut in enumerate(cuts, start=1):
                cut = raw_cut.astype(bool) & anchor
                area = int(cut.sum())
                if area < args.min_cut_area or area > max_pixels:
                    continue
                key_bytes = np.packbits(cut.ravel()).tobytes()
                if key_bytes in seen:
                    continue
                seen.add(key_bytes)
                comp_ids = np.unique(comp_labels[cut])
                comp_ids = comp_ids[comp_ids > 0]
                if comp_ids.size == 0:
                    continue
                comp_id = int(comp_ids[np.argmax([(comp_labels[cut] == cid).sum() for cid in comp_ids])])
                component = comp_labels == comp_id
                trial = anchor & ~cut
                trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                trial_match = component_match_metrics(trial, instance, args.min_overlap_frac)
                cut_prob = prob[cut]
                row: dict[str, Any] = {
                    "run_id": "R251-reverse-background-candidate-selector",
                    "split": "val",
                    "image": name,
                    "dist_percentile": float(dist_percentile),
                    "candidate_rank": float(cut_rank),
                    "pred_component_id": float(comp_id),
                    "r251_bg_prob_mean": float(np.mean(cut_prob)),
                    "r251_bg_prob_p10": float(np.percentile(cut_prob, 10)),
                    "r251_bg_prob_p50": float(np.percentile(cut_prob, 50)),
                    "r251_bg_prob_p90": float(np.percentile(cut_prob, 90)),
                    "r251_bg_prob_min": float(np.min(cut_prob)),
                    "r251_bg_prob_ge_05_frac": float(np.mean(cut_prob >= 0.5)),
                    "cut_gt_fg_frac": safe_divide(float((cut & gt).sum()), float(area)),
                    "cut_gt_gap_frac": safe_divide(float((cut & gt_cache.gap_region).sum()), float(area)),
                }
                row.update(cut_features(image, anchor, component, cut))
                row.update(local_bg_connectivity(anchor, cut, 24))
                for metric, value in trial_metrics.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                for metric, value in trial_match.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = float(value) - float(anchor_match[metric])
                row["safe_useful_label"] = float(is_safe_useful(row, args))
                row["risk_label"] = float(is_risk(row, args))
                rows.append(row)
        if len(rows) == rows_before:
            zero_candidate_images.append(name)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.candidate_csv, rows)
            args.progress_json.parent.mkdir(parents=True, exist_ok=True)
            args.progress_json.write_text(json.dumps({"processed_images": index, "total_images": len(case_names), "num_rows": len(rows), "num_images": len({str(row["image"]) for row in rows})}, indent=2), encoding="utf-8")
    coverage = {
        "expected_images": len(case_names),
        "expected_image_names": case_names,
        "processed_images": len(processed_images),
        "processed_image_names": processed_images,
        "zero_candidate_images": zero_candidate_images,
        "skipped": skipped,
        "complete": len(processed_images) == len(case_names) and not skipped,
        "is_fullval": args.limit == 0,
    }
    return rows, coverage


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [as_float(row, key, default=np.nan) for row in rows]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected": len(rows),
        "selected_images": len({str(row["image"]) for row in rows}),
        "safe_useful": int(sum(as_float(row, "safe_useful_label") > 0.5 for row in rows)),
        "risk": int(sum(as_float(row, "risk_label") > 0.5 for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
    }


def select_rows(rows: list[dict[str, Any]], mean_thr: float, p10_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if as_float(row, "r251_bg_prob_mean") >= mean_thr and as_float(row, "r251_bg_prob_p10") >= p10_thr:
            grouped.setdefault(str(row["image"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        image_rows.sort(key=lambda row: (-as_float(row, "r251_bg_prob_p10"), -as_float(row, "r251_bg_prob_mean"), as_float(row, "cut_area")))
        selected.extend(image_rows[:max_per_image])
    return selected


def binary_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"roc_auc": None, "average_precision": None}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def calibration_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mean_scores = np.asarray([as_float(row, "r251_bg_prob_mean") for row in rows], dtype=np.float64)
    p10_scores = np.asarray([as_float(row, "r251_bg_prob_p10") for row in rows], dtype=np.float64)
    safe_labels = np.asarray([as_float(row, "safe_useful_label") > 0.5 for row in rows], dtype=np.int32)
    risk_labels = np.asarray([as_float(row, "risk_label") > 0.5 for row in rows], dtype=np.int32)
    quantile_levels = np.asarray([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])

    def describe(values: np.ndarray) -> dict[str, Any]:
        if values.size == 0:
            return {"min": None, "max": None, "quantiles": {}}
        quantiles = np.quantile(values, quantile_levels)
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "quantiles": {f"q{int(level * 100):02d}": float(value) for level, value in zip(quantile_levels, quantiles)},
        }

    return {
        "num_rows": len(rows),
        "mean_score": describe(mean_scores),
        "p10_score": describe(p10_scores),
        "safe_useful": {"positive_rows": int(safe_labels.sum()), **binary_metric(safe_labels, mean_scores)},
        "risk": {"positive_rows": int(risk_labels.sum()), "score_definition": "1 - r251_bg_prob_mean", **binary_metric(risk_labels, 1.0 - mean_scores)},
    }


def reachable_thresholds(configured: str, rows: list[dict[str, Any]], key: str) -> list[float]:
    values = np.asarray([as_float(row, key) for row in rows], dtype=np.float64)
    thresholds = set(parse_float_list(configured))
    if values.size:
        thresholds.update(float(value) for value in np.quantile(values, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]))
    return sorted(thresholds)


def evaluate_grid(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    allow_promotion: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    grid: list[dict[str, Any]] = []
    mean_thresholds = reachable_thresholds(args.mean_thresholds, rows, "r251_bg_prob_mean")
    p10_thresholds = reachable_thresholds(args.p10_thresholds, rows, "r251_bg_prob_p10")
    for mean_thr in mean_thresholds:
        for p10_thr in p10_thresholds:
            selected = select_rows(rows, mean_thr, p10_thr, args.max_actions_per_image)
            item = {"mean_threshold": mean_thr, "p10_threshold": p10_thr, **summarize(selected)}
            item["meets_metric_gate"] = bool(
                item["selected_images"] >= 10
                and item["safe_useful"] >= 8
                and item["risk"] <= 2
                and item["mean_delta_recall"] is not None
                and float(item["mean_delta_recall"]) >= 0.0
                and item["mean_cut_gt_fg_frac"] is not None
                and float(item["mean_cut_gt_fg_frac"]) <= 0.05
                and item["mean_delta_boundary_iou"] is not None
                and float(item["mean_delta_boundary_iou"]) > 0.0
                and item["mean_delta_boundary_f1"] is not None
                and float(item["mean_delta_boundary_f1"]) > 0.0
                and item["mean_delta_gap_region_fp_rate"] is not None
                and float(item["mean_delta_gap_region_fp_rate"]) < 0.0
                and item["mean_delta_component_count_mae"] is not None
                and float(item["mean_delta_component_count_mae"]) <= 0.0
            )
            item["passes_gate"] = bool(allow_promotion and item["meets_metric_gate"])
            grid.append(item)
    return grid, {"mean_thresholds": mean_thresholds, "p10_thresholds": p10_thresholds}


def choose_best(grid: list[dict[str, Any]]) -> dict[str, Any] | None:
    nonempty = [row for row in grid if int(row.get("selected") or 0) > 0]
    if not nonempty:
        return None
    passing = [row for row in nonempty if row.get("passes_gate")]
    pool = passing if passing else nonempty
    def finite_or(value: Any, default: float) -> float:
        if value is None:
            return default
        number = float(value)
        return number if np.isfinite(number) else default

    return sorted(
        pool,
        key=lambda row: (
            bool(row.get("passes_gate")),
            int(row["safe_useful"]) - int(row["risk"]),
            int(row["safe_useful"]),
            -int(row["risk"]),
            -finite_or(row.get("mean_cut_gt_fg_frac"), 999.0),
            finite_or(row.get("mean_delta_boundary_iou"), -999.0),
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    scope = validate_scope(args)
    train_args = train_namespace(args)
    train_names, train_selection_audit, cache = select_case_names(train_args)
    if len(train_names) != args.limit_train:
        raise RuntimeError(f"Needed {args.limit_train} usable train corridor cases, found {len(train_names)}.")
    x, y, train_info = collect_training(train_args, train_names, cache)
    model = fit_model(train_args, x, y)
    candidates, coverage = build_candidates(args, model)
    calibration = calibration_diagnostics(candidates)
    allow_promotion = bool(coverage["is_fullval"] and coverage["complete"])
    grid, threshold_grid = evaluate_grid(candidates, args, allow_promotion)
    best = choose_best(grid)
    write_csv(args.candidate_csv, candidates)
    write_csv(args.grid_csv, grid)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_model = args.model_path.with_name(f"{args.model_path.stem}_{stamp}{args.model_path.suffix}")
    timestamped_model.parent.mkdir(parents=True, exist_ok=True)
    with timestamped_model.open("wb") as handle:
        pickle.dump(model, handle)
    shutil.copyfile(timestamped_model, args.model_path)
    report = {
        "run_id": "R251-reverse-background-candidate-selector",
        "dataset": args.dataset,
        "split": "val",
        "scope_validation": scope,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "inference_inputs": ["image", "anchor"],
        "gt_allowed_at_inference": False,
        "train_info": train_info,
        "train_selection_audit": train_selection_audit,
        "num_candidate_rows": len(candidates),
        "num_candidate_images": len({str(row["image"]) for row in candidates}),
        "num_safe_useful": int(sum(as_float(row, "safe_useful_label") > 0.5 for row in candidates)),
        "num_risk": int(sum(as_float(row, "risk_label") > 0.5 for row in candidates)),
        "num_passing_configs": int(sum(bool(row.get("passes_gate")) for row in grid)),
        "best": best,
        "coverage": coverage,
        "calibration": calibration,
        "threshold_grid": threshold_grid,
        "reproducibility": {
            "seed": args.seed,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "feature_source": "audit_r228_oracle_pixel_seam_localizer.feature_stack",
            "train_case_names": train_names,
            "dist_percentiles": args.dist_percentiles,
            "train_max_candidates_per_percentile": args.train_max_candidates_per_percentile,
            "eval_max_candidates_per_percentile": args.max_candidates_generated,
        },
        "decision": "passing_config_found_review_before_mask_writing" if best and best.get("passes_gate") else "no_go_dense_reverse_background_selector_not_clean_enough",
        "promotion_gate": "Only a passing full original-val config may proceed to isolated masks and R201 audit; clean-test-v2 remains locked.",
    }
    timestamped_json = args.output_json.with_name(f"{args.output_json.stem}_{stamp}{args.output_json.suffix}")
    report["timestamped_output_json"] = str(timestamped_json)
    payload = json.dumps(report, indent=2)
    timestamped_json.parent.mkdir(parents=True, exist_ok=True)
    timestamped_json.write_text(payload, encoding="utf-8")
    args.output_json.write_text(payload, encoding="utf-8")
    args.progress_json.write_text(json.dumps({"status": "completed", "processed_images": coverage["processed_images"], "total_images": coverage["expected_images"], "num_rows": len(candidates), "coverage_complete": coverage["complete"]}, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "candidate_csv": str(args.candidate_csv), "grid_csv": str(args.grid_csv), "num_passing_configs": report["num_passing_configs"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
