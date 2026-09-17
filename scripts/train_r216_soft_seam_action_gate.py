#!/usr/bin/env python3
"""Train/evaluate an R216 action-level useful/risk gate.

Consumes R216 soft seam-action candidate diagnostics and tests whether
inference-time features can select useful partial seam actions while rejecting
over-erosion risk. Writes JSON/CSV only; no masks and no clean-test-v2 use.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_KEYS = [
    "candidate_rank",
    "action_frac",
    "r216_crop_area",
    "r216_cut_area",
    "r216_cut_frac_crop",
    "r216_bg_components_before",
    "r216_bg_components_after",
    "r216_bg_component_count_delta",
    "r216_cut_adjacent_bg_components_before",
    "r216_cut_adjacent_bg_frac_before",
    "r216_cut_component_bg_area_after",
    "r216_cut_component_bg_frac_crop_after",
    "r216_cut_component_border_contacts_after",
    "r216_bg_channel_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate R216 soft seam-action gate.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--classifier", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--useful-target", choices=["quick", "safe_channel"], default="quick")
    parser.add_argument("--risk-target", choices=["hard", "overerosion_or_hard"], default="hard")
    parser.add_argument("--grouped-cv-folds", type=int, default=5)
    parser.add_argument("--useful-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--risk-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--max-actions-per-image", type=int, default=1)
    parser.add_argument(
        "--extra-feature-prefix",
        action="append",
        default=[],
        help="Append numeric CSV columns whose names start with this prefix. Default keeps the original R216 feature set.",
    )
    parser.add_argument("--seed", type=int, default=202607216)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def is_quick_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= -8e-4
        and as_float(row, "delta_boundary_iou") > 0.0
        and as_float(row, "delta_boundary_f1") > 0.0
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
    )


def is_safe_channel_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "r216_cut_gt_fg_frac") <= 0.5
        and as_float(row, "r216_cut_gt_gap_frac") >= 0.5
        and is_quick_useful(row)
    )


def is_useful(row: dict[str, Any], target: str) -> bool:
    if target == "quick":
        return is_quick_useful(row)
    if target == "safe_channel":
        return is_safe_channel_useful(row)
    raise ValueError(target)


def is_hard_risk(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
        or as_float(row, "delta_recall") < -8e-4
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_boundary_f1") < 0.0
    )


def is_risk(row: dict[str, Any], target: str) -> bool:
    if target == "hard":
        return is_hard_risk(row)
    if target == "overerosion_or_hard":
        return bool(is_hard_risk(row) or as_float(row, "r216_cut_gt_fg_frac") > 0.5)
    raise ValueError(target)


def feature_keys(rows: list[dict[str, Any]], prefixes: list[str] | None = None) -> list[str]:
    keys = list(FEATURE_KEYS)
    prefixes = prefixes or []
    for prefix in prefixes:
        extra = sorted({key for row in rows for key in row if key.startswith(prefix)})
        for key in extra:
            if key not in keys:
                keys.append(key)
    return keys


def matrix(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    return np.asarray([[as_float(row, key) for key in keys] for row in rows], dtype=np.float32)


def labels(rows: list[dict[str, Any]], kind: str, args: argparse.Namespace) -> np.ndarray:
    if kind == "useful":
        return np.asarray([int(is_useful(row, args.useful_target)) for row in rows], dtype=np.int32)
    if kind == "risk":
        return np.asarray([int(is_risk(row, args.risk_target)) for row in rows], dtype=np.int32)
    raise ValueError(kind)


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("clf", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed, l2_regularization=0.02)),
        ]
    )


def fit_model(rows: list[dict[str, Any]], kind: str, args: argparse.Namespace, seed: int, keys: list[str]) -> Pipeline:
    y = labels(rows, kind, args)
    if len(set(y.tolist())) < 2:
        raise RuntimeError(f"{kind} labels contain only one class")
    model = make_model(args.classifier, seed)
    model.fit(matrix(rows, keys), y)
    return model


def add_probs(rows: list[dict[str, Any]], useful_model: Pipeline, risk_model: Pipeline, args: argparse.Namespace, keys: list[str]) -> list[dict[str, Any]]:
    x = matrix(rows, keys)
    useful_probs = useful_model.predict_proba(x)[:, 1]
    risk_probs = risk_model.predict_proba(x)[:, 1]
    out = []
    for row, useful_prob, risk_prob in zip(rows, useful_probs, risk_probs):
        out.append(
            {
                **row,
                "r216_useful_label": float(is_useful(row, args.useful_target)),
                "r216_risk_label": float(is_risk(row, args.risk_target)),
                "r216_useful_prob": float(useful_prob),
                "r216_risk_prob": float(risk_prob),
            }
        )
    return out


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    return grouped


def selected_rows(rows: list[dict[str, Any]], useful_thr: float, risk_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for image_rows in group_rows(rows).values():
        candidates = [
            row
            for row in image_rows
            if as_float(row, "r216_useful_prob") >= useful_thr and as_float(row, "r216_risk_prob") <= risk_thr
        ]
        candidates.sort(
            key=lambda row: (
                -as_float(row, "r216_useful_prob"),
                as_float(row, "r216_risk_prob"),
                as_float(row, "candidate_rank"),
                as_float(row, "action_frac"),
            )
        )
        selected.extend(candidates[:max_per_image])
    return selected


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else None


def evaluate_grid(rows: list[dict[str, Any]], useful_thresholds: list[float], risk_thresholds: list[float], max_per_image: int) -> list[dict[str, Any]]:
    out = []
    for useful_thr in useful_thresholds:
        for risk_thr in risk_thresholds:
            selected = selected_rows(rows, useful_thr, risk_thr, max_per_image)
            useful = int(sum(as_float(row, "r216_useful_label") > 0.5 for row in selected))
            risk = int(sum(as_float(row, "r216_risk_label") > 0.5 for row in selected))
            out.append(
                {
                    "useful_threshold": useful_thr,
                    "risk_threshold": risk_thr,
                    "accepted": len(selected),
                    "accepted_images": len({row["image"] for row in selected}),
                    "useful": useful,
                    "risk": risk,
                    "precision_useful": float(useful / max(1, len(selected))),
                    "risk_rate": float(risk / max(1, len(selected))),
                    "mean_action_frac": mean(selected, "action_frac"),
                    "mean_delta_dice": mean(selected, "delta_dice"),
                    "mean_delta_iou": mean(selected, "delta_iou"),
                    "mean_delta_recall": mean(selected, "delta_recall"),
                    "mean_delta_boundary_iou": mean(selected, "delta_boundary_iou"),
                    "mean_delta_boundary_f1": mean(selected, "delta_boundary_f1"),
                    "mean_delta_gap_region_fp_rate": mean(selected, "delta_gap_region_fp_rate"),
                }
            )
    return out


def grouped_cv(rows: list[dict[str, Any]], args: argparse.Namespace, keys: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = np.asarray(sorted({row["image"] for row in rows}))
    folds = min(args.grouped_cv_folds, len(images))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=args.seed)
    probed: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(images), start=1):
        train_images = set(images[train_idx].tolist())
        val_images = set(images[val_idx].tolist())
        train_rows = [row for row in rows if row["image"] in train_images]
        val_rows = [row for row in rows if row["image"] in val_images]
        useful_model = fit_model(train_rows, "useful", args, args.seed + fold_idx, keys)
        risk_model = fit_model(train_rows, "risk", args, args.seed + 100 + fold_idx, keys)
        probed.extend(add_probs(val_rows, useful_model, risk_model, args, keys))
    grid = evaluate_grid(probed, parse_float_list(args.useful_thresholds), parse_float_list(args.risk_thresholds), args.max_actions_per_image)
    frontier = {}
    for max_risk in [0, 1, 2, 5, 10, 20]:
        subset = [row for row in grid if int(row["risk"]) <= max_risk]
        subset = sorted(subset, key=lambda row: (row["useful"], row["precision_useful"], row["accepted"], row["mean_delta_boundary_iou"] or -999), reverse=True)
        frontier[f"risk_le_{max_risk}"] = subset[:20]
    best = sorted(grid, key=lambda row: (row["risk"] <= 2, row["useful"], -row["risk"], row["mean_delta_boundary_iou"] or -999), reverse=True)[0]
    return grid, {"folds": folds, "best": best, "frontier": frontier}


def main() -> None:
    args = parse_args()
    rows = read_csv(args.candidate_csv)
    keys = feature_keys(rows, args.extra_feature_prefix)
    grid, cv_summary = grouped_cv(rows, args, keys)
    report = {
        "run_id": "R216-soft-seam-action-gate",
        "candidate_csv": str(args.candidate_csv),
        "evidence_level": "candidate_action_grouped_cv_diagnostic",
        "clean_test_v2_used": False,
        "writes_masks": False,
        "feature_keys": keys,
        "extra_feature_prefix": args.extra_feature_prefix,
        "classifier": args.classifier,
        "useful_target": args.useful_target,
        "risk_target": args.risk_target,
        "num_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_useful": int(sum(is_useful(row, args.useful_target) for row in rows)),
        "num_risk": int(sum(is_risk(row, args.risk_target) for row in rows)),
        "grouped_cv": cv_summary,
        "warning": "candidate-action grouped-CV diagnostic only; not mask-level R201 evidence",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, grid)
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "best": cv_summary["best"]}, indent=2))


if __name__ == "__main__":
    main()
