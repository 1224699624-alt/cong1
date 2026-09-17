#!/usr/bin/env python3
"""R238-F0 grouped-CV gate for R237 background-connectivity candidates.

This script audits whether R237's oracle-generated seam/background candidates
can be ranked using GT-free candidate features. It uses R237 CSV labels only on
original train/val for supervision/evaluation. It writes no masks and must not
be used on clean-test-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


DROP_PREFIXES = ("anchor_", "candidate_", "delta_")
DROP_EXACT = {
    "run_id",
    "split",
    "image",
    "gt_a",
    "gt_b",
    "cut_gt_fg_pixels",
    "cut_gt_fg_frac",
    "cut_gt_gap_frac",
    "merge_targeted_promotable",
    "safe_nonmerge_gain",
    "overerosion_proxy",
    "reject_reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate R238-F0 GT-free gate on R237 candidates.")
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r237_component_merge_targeted_bounded_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r238_r237_candidate_gate_groupedcv.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r238_r237_candidate_gate_groupedcv_rows.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--risk-max", default="0,2,5,10,20,40")
    parser.add_argument("--min-prob-grid", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--target", choices=["safe_nonmerge_gain", "strict_safe_gain"], default="safe_nonmerge_gain")
    parser.add_argument("--seed", type=int, default=202607238)
    return parser.parse_args()


def as_float(value: Any, default: float = math.nan) -> float:
    if value in (None, "", "None", "nan"):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def label_row(row: dict[str, Any], target: str) -> tuple[int, int]:
    safe = int(as_float(row.get("safe_nonmerge_gain"), 0.0) > 0.5)
    risk = int(as_float(row.get("overerosion_proxy"), 0.0) > 0.5)
    if target == "strict_safe_gain":
        safe = int(
            safe
            and as_float(row.get("delta_recall"), 0.0) >= 0.0
            and as_float(row.get("delta_component_count_mae"), 0.0) <= 0.0
            and (
                as_float(row.get("delta_boundary_iou"), 0.0) > 0.0
                or as_float(row.get("delta_gap_region_fp_rate"), 0.0) < 0.0
            )
        )
    return safe, risk


def numeric_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for key in sorted({k for row in rows for k in row}):
        if key in DROP_EXACT or key.startswith(DROP_PREFIXES):
            continue
        values = [as_float(row.get(key)) for row in rows]
        finite = [v for v in values if np.isfinite(v)]
        if len(finite) >= max(8, len(rows) // 20) and len(set(round(v, 8) for v in finite)) > 1:
            names.append(key)
    return names


def build_matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    x = np.zeros((len(rows), len(features)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, key in enumerate(features):
            value = as_float(row.get(key), 0.0)
            x[i, j] = value if np.isfinite(value) else 0.0
    return x


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def fit_predict_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> dict[str, np.ndarray | dict[str, Any]]:
    if not SKLEARN_AVAILABLE:
        # Deterministic fallback: score by background-channel and cut geometry.
        raw = x.mean(axis=1) if x.shape[1] else np.zeros((x.shape[0],), dtype=np.float32)
        raw = (raw - raw.min()) / max(1e-8, float(raw.max() - raw.min()))
        return {"prob": raw.astype(np.float64), "models": {"fallback": "feature_mean_no_sklearn"}}

    unique_groups = np.unique(groups)
    n_splits = max(2, min(folds, len(unique_groups)))
    prob = np.zeros((len(y),), dtype=np.float64)
    model_info: dict[str, Any] = {"folds": []}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
        if len(set(y[train_idx].tolist())) < 2:
            fold_prob = np.full((len(test_idx),), float(np.mean(y[train_idx])) if train_idx.size else 0.0)
            model_name = "constant"
        else:
            if int(y[train_idx].sum()) >= 20 and int((1 - y[train_idx]).sum()) >= 20:
                model = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, random_state=seed + fold)
                model_name = "HistGradientBoostingClassifier"
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed + fold))
                model_name = "LogisticRegression"
            model.fit(x[train_idx], y[train_idx])
            fold_prob = model.predict_proba(x[test_idx])[:, 1]
        prob[test_idx] = fold_prob
        model_info["folds"].append(
            {
                "fold": fold,
                "model": model_name,
                "num_train": int(train_idx.size),
                "num_test": int(test_idx.size),
                "train_positive": int(y[train_idx].sum()),
                "test_positive": int(y[test_idx].sum()),
            }
        )
    return {"prob": prob, "models": model_info}


def summarize_selected(rows: list[dict[str, Any]], selected: np.ndarray) -> dict[str, Any]:
    chosen = [row for row, keep in zip(rows, selected.tolist()) if keep]

    def mean(key: str) -> float | None:
        vals = [as_float(row.get(key)) for row in chosen]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    return {
        "num_selected": int(selected.sum()),
        "num_images": len({row.get("image") for row in chosen}),
        "num_safe_nonmerge_gain": int(sum(as_float(row.get("safe_nonmerge_gain"), 0.0) > 0.5 for row in chosen)),
        "num_overerosion_proxy": int(sum(as_float(row.get("overerosion_proxy"), 0.0) > 0.5 for row in chosen)),
        "mean_delta_dice": mean("delta_dice"),
        "mean_delta_iou": mean("delta_iou"),
        "mean_delta_recall": mean("delta_recall"),
        "mean_delta_boundary_iou": mean("delta_boundary_iou"),
        "mean_delta_boundary_f1": mean("delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean("delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean("delta_component_count_mae"),
        "mean_delta_merged_pred_components": mean("delta_merged_pred_components"),
        "mean_background_connected_by_cut": mean("background_connected_by_cut"),
        "mean_cut_gt_fg_frac": mean("cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean("cut_gt_gap_frac"),
    }


def threshold_sweep(rows: list[dict[str, Any]], prob: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    risk_limits = parse_int_list(args.risk_max)
    prob_grid = parse_float_list(args.min_prob_grid)
    out: list[dict[str, Any]] = []
    for thr in prob_grid:
        base = prob >= thr
        for risk_max in risk_limits:
            selected = base.copy()
            summary = summarize_selected(rows, selected)
            if summary["num_overerosion_proxy"] > risk_max:
                continue
            row = {"prob_threshold": thr, "risk_max": risk_max, **summary}
            out.append(row)
    out.sort(
        key=lambda row: (
            int(row["num_overerosion_proxy"]),
            -int(row["num_safe_nonmerge_gain"]),
            -(row.get("mean_delta_boundary_iou") or -999.0),
            row.get("mean_delta_gap_region_fp_rate") or 999.0,
            row.get("mean_delta_component_count_mae") or 999.0,
        )
    )
    return out


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
    if not rows:
        raise SystemExit(f"No rows found: {args.candidate_csv}")
    features = numeric_feature_names(rows)
    x = build_matrix(rows, features)
    y = np.asarray([label_row(row, args.target)[0] for row in rows], dtype=np.int64)
    risk = np.asarray([label_row(row, args.target)[1] for row in rows], dtype=np.int64)
    groups = np.asarray([str(row.get("image") or i) for i, row in enumerate(rows)])
    pred = fit_predict_cv(x, y, groups, args.folds, args.seed)
    prob = np.asarray(pred["prob"], dtype=np.float64)
    try:
        auc = float(roc_auc_score(y, prob)) if SKLEARN_AVAILABLE and len(set(y.tolist())) > 1 else None
    except Exception:
        auc = None

    eval_rows: list[dict[str, Any]] = []
    for row, p, label, risky in zip(rows, prob.tolist(), y.tolist(), risk.tolist()):
        eval_rows.append(
            {
                "image": row.get("image"),
                "prob_safe": p,
                "label_safe": int(label),
                "label_risk": int(risky),
                "safe_nonmerge_gain": row.get("safe_nonmerge_gain"),
                "overerosion_proxy": row.get("overerosion_proxy"),
                "delta_dice": row.get("delta_dice"),
                "delta_iou": row.get("delta_iou"),
                "delta_recall": row.get("delta_recall"),
                "delta_boundary_iou": row.get("delta_boundary_iou"),
                "delta_boundary_f1": row.get("delta_boundary_f1"),
                "delta_gap_region_fp_rate": row.get("delta_gap_region_fp_rate"),
                "delta_component_count_mae": row.get("delta_component_count_mae"),
                "background_connected_by_cut": row.get("background_connected_by_cut"),
            }
        )
    sweep = threshold_sweep(rows, prob, args)
    best = sweep[0] if sweep else {}
    report = {
        "run_id": "R238-F0-r237-candidate-gate-grouped-cv",
        "candidate_csv": str(args.candidate_csv),
        "clean_test_v2_used": False,
        "writes_masks": False,
        "target": args.target,
        "num_rows": len(rows),
        "num_images": len(set(groups.tolist())),
        "num_features": len(features),
        "features": features,
        "num_positive": int(y.sum()),
        "num_risk": int(risk.sum()),
        "positive_rate": float(np.mean(y)),
        "risk_rate": float(np.mean(risk)),
        "cv_auc": auc,
        "model_info": pred["models"],
        "threshold_sweep": sweep,
        "best_low_risk_row": best,
        "decision": "go_r238_mask_editor" if best and int(best.get("num_selected") or 0) >= 20 and int(best.get("num_overerosion_proxy") or 0) <= 2 and (best.get("mean_delta_boundary_iou") or 0.0) > 0.0 else "no_go_gate_not_selective",
        "warning": "R237 candidates are oracle-generated; this audit only tests GT-free ranking of those candidates on original val.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, eval_rows)
    print(json.dumps({"decision": report["decision"], "cv_auc": auc, "best_low_risk_row": best, "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
