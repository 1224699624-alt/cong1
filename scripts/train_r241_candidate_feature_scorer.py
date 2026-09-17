#!/usr/bin/env python3
"""R241-F0 candidate-feature scorer for seam/background cuts.

This grouped-CV audit trains a lightweight scorer on existing original-val
candidate rows. Features are GT-free candidate geometry/probability values;
labels use GT only for original-val supervision/audit. It writes no masks and
does not touch clean-test-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


FEATURE_KEYS = [
    "action_frac",
    "bg_a_area",
    "bg_b_area",
    "bg_channel_proxy",
    "bg_component_count_delta",
    "bg_components_after",
    "bg_components_before",
    "bg_pair_distance",
    "bridge_score_mean",
    "bridge_score_p90",
    "component_area",
    "corridor_radius",
    "cut_adjacent_bg_components_before",
    "cut_anchor_area_frac",
    "cut_area",
    "cut_bbox_h",
    "cut_bbox_w",
    "cut_component_bg_frac_after",
    "cut_component_border_contacts_after",
    "cut_fill",
    "cut_max_dist_in",
    "cut_mean_dist_in",
    "cut_mean_grad",
    "cut_mean_image",
    "cut_p90_grad",
    "cut_slenderness",
    "cut_std_image",
    "pred_component_id",
    "ring_bg_frac",
    "ring_fg_frac",
    "seam_prob_mean",
    "seam_prob_p90",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate R241 candidate-feature scorer with grouped CV.")
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r239_gtfree_bg_channel_light_fullval_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r241_candidate_feature_scorer_groupedcv.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r241_candidate_feature_scorer_groupedcv_rows.csv"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r241_candidate_feature_scorer_threshold_grid.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--max-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202607241)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def safe_label(row: dict[str, Any]) -> int:
    return int(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= 0.0
        and as_float(row, "delta_boundary_iou") >= 0.0
        and as_float(row, "delta_boundary_f1") >= 0.0
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
        and as_float(row, "cut_gt_fg_frac") <= 0.25
        and as_float(row, "cut_gt_gap_frac") >= 0.75
    )


def risk_label(row: dict[str, Any]) -> int:
    return int(
        as_float(row, "cut_gt_fg_frac") > 0.5
        or as_float(row, "delta_recall") < 0.0
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_boundary_f1") < 0.0
        or as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
    )


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[as_float(row, key) for key in FEATURE_KEYS] for row in rows], dtype=np.float32)


def make_model(args: argparse.Namespace):
    if not SKLEARN_AVAILABLE:
        return None
    if args.model == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed))
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.035,
        max_leaf_nodes=15,
        l2_regularization=0.05,
        random_state=args.seed,
    )


def fallback_prob(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    pos = x_train[y_train > 0.5]
    neg = x_train[y_train <= 0.5]
    center = np.nanmean(x_train, axis=0)
    scale = np.nanstd(x_train, axis=0) + 1e-6
    x_test = np.where(np.isfinite(x_test), x_test, center)
    z_test = (x_test - center) / scale
    z_pos = (np.nanmean(pos, axis=0) - center) / scale if len(pos) else np.zeros((x_train.shape[1],), dtype=np.float32)
    z_neg = (np.nanmean(neg, axis=0) - center) / scale if len(neg) else np.zeros((x_train.shape[1],), dtype=np.float32)
    d_pos = np.sum((z_test - z_pos) ** 2, axis=1)
    d_neg = np.sum((z_test - z_neg) ** 2, axis=1)
    logits = np.clip(0.5 * (d_neg - d_pos), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def grouped_cv(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[np.ndarray, list[dict[str, Any]]]:
    x = matrix(rows)
    y = np.asarray([safe_label(row) for row in rows], dtype=np.int32)
    groups = np.asarray([str(row.get("image")) for row in rows])
    unique_groups = np.unique(groups)
    n_splits = max(2, min(args.folds, len(unique_groups)))
    prob = np.zeros((len(rows),), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        if len(set(y[train_idx].tolist())) < 2:
            fold_prob = np.full((len(test_idx),), float(np.mean(y[train_idx])) if len(train_idx) else 0.0)
            model_name = "constant"
        elif SKLEARN_AVAILABLE:
            model = make_model(args)
            model.fit(x[train_idx], y[train_idx])
            fold_prob = model.predict_proba(x[test_idx])[:, 1]
            model_name = args.model
        else:
            fold_prob = fallback_prob(x[train_idx], y[train_idx], x[test_idx])
            model_name = "prototype_fallback"
        prob[test_idx] = fold_prob
        fold_rows.append(
            {
                "fold": fold,
                "model": model_name,
                "num_train": int(len(train_idx)),
                "num_test": int(len(test_idx)),
                "train_safe": int(y[train_idx].sum()),
                "test_safe": int(y[test_idx].sum()),
                "test_risk": int(sum(risk_label(rows[i]) for i in test_idx)),
            }
        )
    return prob, fold_rows


def parse_thresholds(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({str(row.get("image")) for row in rows}),
        "num_safe": int(sum(safe_label(row) for row in rows)),
        "num_risk": int(sum(risk_label(row) for row in rows)),
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


def selected_at_threshold(rows: list[dict[str, Any]], prob: np.ndarray, threshold: float, max_per_image: int) -> list[dict[str, Any]]:
    scored = [{**row, "r241_safe_prob": float(p)} for row, p in zip(rows, prob.tolist()) if float(p) >= threshold]
    scored.sort(
        key=lambda row: (
            str(row.get("image")),
            -as_float(row, "r241_safe_prob"),
            -as_float(row, "bg_channel_proxy"),
            -as_float(row, "cut_component_bg_frac_after"),
            as_float(row, "cut_mean_dist_in"),
            as_float(row, "cut_area"),
        )
    )
    out: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in scored:
        image = str(row.get("image"))
        count = counts.get(image, 0)
        if count >= max_per_image:
            continue
        counts[image] = count + 1
        out.append(row)
    return out


def threshold_grid(rows: list[dict[str, Any]], prob: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for threshold in parse_thresholds(args.thresholds):
        selected = selected_at_threshold(rows, prob, threshold, args.max_per_image)
        row = {"threshold": threshold, **summarize(selected)}
        grid.append(row)
    return grid


def choose_best(grid: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not grid:
        return None
    viable = [
        row
        for row in grid
        if int(row["num_rows"]) > 0
        and int(row["num_risk"]) <= max(1, int(row["num_safe"]) // 4)
        and (row["mean_delta_recall"] is not None and float(row["mean_delta_recall"]) >= 0.0)
        and (row["mean_delta_boundary_iou"] is not None and float(row["mean_delta_boundary_iou"]) > 0.0)
        and (row["mean_delta_gap_region_fp_rate"] is not None and float(row["mean_delta_gap_region_fp_rate"]) < 0.0)
    ]
    pool = viable if viable else [row for row in grid if int(row["num_rows"]) > 0]
    if not pool:
        return sorted(grid, key=lambda row: float(row["threshold"]), reverse=True)[0]
    return sorted(
        pool,
        key=lambda row: (
            int(row["num_risk"]) <= max(1, int(row["num_safe"]) // 4),
            -int(row["num_risk"]),
            int(row["num_safe"]),
            float(row["mean_delta_boundary_iou"] or -999.0),
            -float(row["mean_delta_gap_region_fp_rate"] or 999.0),
        ),
        reverse=True,
    )[0]


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
    rows = read_rows(args.candidate_csv)
    prob, folds = grouped_cv(rows, args)
    y = np.asarray([safe_label(row) for row in rows], dtype=np.int32)
    risk = np.asarray([risk_label(row) for row in rows], dtype=np.int32)
    auc = None
    ap = None
    if len(set(y.tolist())) > 1:
        try:
            auc = float(roc_auc_score(y, prob)) if SKLEARN_AVAILABLE else None
            ap = float(average_precision_score(y, prob)) if SKLEARN_AVAILABLE else None
        except Exception:
            auc = None
            ap = None
    scored_rows = []
    for row, p in zip(rows, prob.tolist()):
        scored_rows.append({**row, "r241_safe_prob": float(p), "r241_label_safe": safe_label(row), "r241_label_risk": risk_label(row)})
    grid = threshold_grid(rows, prob, args)
    best = choose_best(grid)
    report = {
        "run_id": "R241-F0-candidate-feature-scorer",
        "candidate_csv": str(args.candidate_csv),
        "clean_test_v2_used": False,
        "writes_masks": False,
        "selection_uses_gt": False,
        "labels_use_gt_original_val_only": True,
        "model": args.model if SKLEARN_AVAILABLE else "prototype_fallback",
        "feature_keys": FEATURE_KEYS,
        "num_rows": len(rows),
        "num_images": len({str(row.get("image")) for row in rows}),
        "num_safe": int(y.sum()),
        "num_risk": int(risk.sum()),
        "safe_rate": float(np.mean(y)) if len(y) else 0.0,
        "risk_rate": float(np.mean(risk)) if len(risk) else 0.0,
        "grouped_cv": folds,
        "cv_auc": auc,
        "cv_average_precision": ap,
        "threshold_grid": grid,
        "best_grid_row": best,
        "decision": "go_mask_level_editor_next" if best and int(best.get("num_safe") or 0) >= 10 and int(best.get("num_risk") or 0) <= max(1, int(best.get("num_safe") or 0) // 4) else "no_go_feature_scorer_not_selective_enough",
        "warning": "Original-val candidate-level audit only; do not use clean-test-v2 for scorer selection.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, scored_rows)
    write_csv(args.grid_csv, grid)
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "grid_csv": str(args.grid_csv), "decision": report["decision"], "cv_auc": auc, "cv_average_precision": ap, "best_grid_row": best}, indent=2))


if __name__ == "__main__":
    main()
