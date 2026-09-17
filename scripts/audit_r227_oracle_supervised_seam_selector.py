#!/usr/bin/env python3
"""R227 oracle-supervised selector for GT-free seam candidates.

R224 showed that true pairwise inter-instance seams are highly useful, but its
candidate generator uses GT instances. R226 generated deployable background-
connectivity candidates, but its self-CV selector accepted nothing. R227 tests a
small bridge: train useful/risk scorers on R224 oracle candidates using only
deployable morphology/image features, then score R226 GT-free candidates.

This is still train/val diagnostic evidence only. It does not write masks and
must not be used on clean-test-v2 for threshold search or model selection.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_KEYS = [
    "action_frac",
    "anchor_boundary_f1",
    "anchor_boundary_iou",
    "anchor_component_count_mae",
    "anchor_component_delta_mean",
    "anchor_dice",
    "anchor_gap_region_fp_rate",
    "anchor_iou",
    "anchor_precision",
    "anchor_recall",
    "bg_channel_proxy",
    "bg_component_count_delta",
    "bg_components_after",
    "bg_components_before",
    "cut_adjacent_bg_components_before",
    "cut_area",
    "cut_bbox_h",
    "cut_bbox_w",
    "cut_component_bg_frac_after",
    "cut_component_border_contacts_after",
    "cut_fill",
    "cut_mean_grad",
    "cut_mean_image",
    "cut_p90_grad",
    "cut_slenderness",
    "cut_std_image",
    "local_anchor_frac",
    "local_bg_frac",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R227 oracle-supervised selector for R226 GT-free candidates.")
    parser.add_argument("--train-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--target-csv", type=Path, default=Path("outputs/analysis/r226_bg_connectivity_seam_r224pilot32_candidates.csv"))
    parser.add_argument("--calibration-csv", type=Path, default=Path(""))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r227_oracle_supervised_seam_selector_summary.json"))
    parser.add_argument("--scored-csv", type=Path, default=Path("outputs/analysis/r227_oracle_supervised_seam_selector_scored.csv"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r227_oracle_supervised_seam_selector_grid.csv"))
    parser.add_argument("--classifier", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--useful-thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--risk-thresholds", default="0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--max-proposals-per-image", type=int, default=1)
    parser.add_argument("--exclude-overerosion-proxy", action="store_true")
    parser.add_argument("--min-train-gap-frac", type=float, default=0.75)
    parser.add_argument("--max-train-fg-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=202607227)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if np.isfinite(out) else default


def label_value(row: dict[str, Any], suffix: str) -> float:
    for prefix in ("r224", "r226", "r225", "r223"):
        key = f"{prefix}_{suffix}"
        if key in row:
            return as_float(row, key)
    return 0.0


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    # R226 uses shorter feature names; map them to the R224/R216 convention.
    aliases = {
        "cut_component_bg_frac_after": "cut_component_bg_frac_after",
        "local_anchor_frac": "local_anchor_frac",
        "local_bg_frac": "local_bg_frac",
    }
    for dst, src in aliases.items():
        if dst not in out and src in row:
            out[dst] = row[src]
    if "local_anchor_frac" not in out:
        out["local_anchor_frac"] = row.get("component_area", 0.0)
    if "local_bg_frac" not in out:
        out["local_bg_frac"] = row.get("ring_bg_frac", 0.0)
    if "cut_component_bg_area_after" in row and "cut_component_bg_frac_after" not in out:
        out["cut_component_bg_frac_after"] = row.get("cut_component_bg_frac_after", 0.0)
    return out


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[as_float(row, key) for key in FEATURE_KEYS] for row in rows], dtype=np.float32)


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
            ("clf", HistGradientBoostingClassifier(max_iter=260, learning_rate=0.035, random_state=seed, l2_regularization=0.04)),
        ]
    )


def fit_model(rows: list[dict[str, Any]], suffix: str, args: argparse.Namespace, seed_offset: int) -> Pipeline:
    y = np.asarray([int(label_value(row, suffix) > 0.5) for row in rows], dtype=np.int32)
    if len(set(y.tolist())) < 2:
        raise ValueError(f"cannot fit {suffix}: only one label class")
    model = make_model(args.classifier, args.seed + seed_offset)
    model.fit(matrix(rows), y)
    return model


def filtered_train_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    normalized = [normalize_row(row) for row in rows]
    # Keep all hard-risk negatives, but focus positives on high-purity R224 seam
    # candidates so the selector learns "gap-like seam" rather than generic cut.
    out: list[dict[str, Any]] = []
    for row in normalized:
        hard = label_value(row, "hard_risk") > 0.5
        useful = label_value(row, "quick_useful") > 0.5
        safe_gap = as_float(row, "cut_gt_gap_frac") >= args.min_train_gap_frac and as_float(row, "cut_gt_fg_frac") <= args.max_train_fg_frac
        if hard or (useful and safe_gap):
            out.append(row)
    return out


def calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in [normalize_row(r) for r in rows]:
        hard = label_value(row, "hard_risk") > 0.5
        over = label_value(row, "overerosion_proxy") > 0.5
        useful = label_value(row, "quick_useful") > 0.5
        safe = label_value(row, "safe_gap_positive") > 0.5
        if hard or over or (useful and safe):
            out.append(row)
    return out


def score_target(train_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    useful_model = fit_model(train_rows, "quick_useful", args, 1)
    risk_model = fit_model(train_rows, "hard_risk", args, 101)
    normalized = [normalize_row(row) for row in target_rows]
    x = matrix(normalized)
    useful_prob = useful_model.predict_proba(x)[:, 1]
    risk_prob = risk_model.predict_proba(x)[:, 1]
    scored: list[dict[str, Any]] = []
    for row, up, rp in zip(normalized, useful_prob, risk_prob):
        scored.append({**row, "r227_useful_prob": float(up), "r227_risk_prob": float(rp)})
    return scored


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({str(row.get("image")) for row in rows}),
        "num_quick_useful": int(sum(label_value(row, "quick_useful") > 0.5 for row in rows)),
        "num_hard_risk": int(sum(label_value(row, "hard_risk") > 0.5 for row in rows)),
        "num_safe_gap_positive": int(sum(label_value(row, "safe_gap_positive") > 0.5 for row in rows)),
        "num_overerosion_proxy": int(sum(label_value(row, "overerosion_proxy") > 0.5 for row in rows)),
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


def selected_rows(rows: list[dict[str, Any]], useful_thr: float, risk_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("image")), []).append(row)
    selected: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        candidates = [row for row in image_rows if as_float(row, "r227_useful_prob") >= useful_thr and as_float(row, "r227_risk_prob") <= risk_thr]
        candidates.sort(key=lambda row: (-as_float(row, "r227_useful_prob"), as_float(row, "r227_risk_prob"), -as_float(row, "delta_boundary_iou")))
        selected.extend(candidates[:max_per_image])
    return selected


def evaluate_grid(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid_rows = rows
    if args.exclude_overerosion_proxy:
        grid_rows = [row for row in rows if label_value(row, "overerosion_proxy") <= 0.5]
    grid: list[dict[str, Any]] = []
    for useful_thr in parse_float_list(args.useful_thresholds):
        for risk_thr in parse_float_list(args.risk_thresholds):
            selected = selected_rows(grid_rows, useful_thr, risk_thr, args.max_proposals_per_image)
            summary = summarize_group(selected)
            grid.append(
                {
                    "useful_threshold": useful_thr,
                    "risk_threshold": risk_thr,
                    "accepted": summary["num_rows"],
                    "accepted_images": summary["num_images"],
                    **summary,
                }
            )
    return grid


def choose_best_grid(grid: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not grid:
        return None
    viable = [
        row
        for row in grid
        if int(row.get("accepted") or 0) > 0
        and int(row.get("num_hard_risk") or 0) == 0
        and (row.get("mean_delta_boundary_iou") is not None and float(row["mean_delta_boundary_iou"]) > 0.0)
        and (row.get("mean_delta_gap_region_fp_rate") is not None and float(row["mean_delta_gap_region_fp_rate"]) < 0.0)
        and (row.get("mean_delta_dice") is not None and float(row["mean_delta_dice"]) >= -5e-4)
    ]
    pool = viable if viable else grid
    return sorted(
        pool,
        key=lambda row: (
            int(row.get("accepted") or 0) > 0,
            -int(row.get("num_hard_risk") or 0),
            float(row.get("mean_delta_boundary_iou") or -999.0),
            -float(row.get("mean_delta_gap_region_fp_rate") or 999.0),
            float(row.get("mean_delta_dice") or -999.0),
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
    raw_train = read_rows(args.train_csv)
    raw_target = read_rows(args.target_csv)
    train_rows = filtered_train_rows(raw_train, args)
    raw_calibration: list[dict[str, Any]] = []
    if str(args.calibration_csv) and args.calibration_csv != Path(".") and args.calibration_csv.exists():
        raw_calibration = read_rows(args.calibration_csv)
        train_rows = train_rows + calibration_rows(raw_calibration)
    scored = score_target(train_rows, raw_target, args)
    grid = evaluate_grid(scored, args)
    best = choose_best_grid(grid)
    write_csv(args.scored_csv, scored)
    write_csv(args.grid_csv, grid)
    selected = selected_rows(scored, float(best["useful_threshold"]), float(best["risk_threshold"]), args.max_proposals_per_image) if best else []
    report = {
        "run_id": "R227-oracle-supervised-seam-selector-diagnostic",
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "train_val_cross_candidate_domain_diagnostic",
        "train_csv": str(args.train_csv),
        "target_csv": str(args.target_csv),
        "calibration_csv": str(args.calibration_csv) if raw_calibration else "",
        "feature_keys": FEATURE_KEYS,
        "classifier": args.classifier,
        "exclude_overerosion_proxy": args.exclude_overerosion_proxy,
        "num_raw_train_rows": len(raw_train),
        "num_raw_calibration_rows": len(raw_calibration),
        "num_filtered_train_rows": len(train_rows),
        "train_summary": summarize_group(train_rows),
        "target_summary": summarize_group(scored),
        "best_grid": best,
        "best_selected_summary": summarize_group(selected),
        "promotion_gate": {
            "passed": bool(best and int(best.get("accepted") or 0) > 0 and int(best.get("num_hard_risk") or 0) == 0 and (best.get("mean_delta_boundary_iou") or 0.0) > 0.0 and (best.get("mean_delta_gap_region_fp_rate") or 0.0) < 0.0),
            "requirements": "nonzero accepted rows, zero hard-risk preferred, positive Boundary IoU/F1 delta, negative gap FP delta, and no material Dice/IoU/Recall regression",
        },
        "warning": "R224 labels train the selector, but target candidates are R226 GT-free; this is original-val diagnostic evidence only, not R201 final evidence.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "scored_csv": str(args.scored_csv), "grid_csv": str(args.grid_csv), "best_grid": best}, indent=2))


if __name__ == "__main__":
    main()
