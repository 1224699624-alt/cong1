"""R245-F0 search for deployable GT-free filters over R244 candidates.

R244 changed the candidate generator toward thin neck/background-connectivity
cuts, but its first GT-free clean subset was still risk-heavy. This script
searches stricter GT-free selection rules and audits them with original-val GT
labels already stored in the candidate table.

It never writes masks and never uses clean-test-v2. GT-derived columns are used
only after selection to report safety and metric deltas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GT_DERIVED_PREFIXES = (
    "candidate_",
    "delta_",
)
GT_DERIVED_COLUMNS = {
    "cut_gt_fg_frac",
    "cut_gt_gap_frac",
    "safe_useful_label",
    "risk_label",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search R245 GT-free filters over R244 candidate rows.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/analysis/r245_gtfree_merge_filter_search.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/analysis/r245_gtfree_merge_filter_search.csv"),
    )
    parser.add_argument("--min-selected", type=int, default=5)
    parser.add_argument("--min-images", type=int, default=5)
    parser.add_argument("--top-k-per-image", default="1,2,999999")
    return parser.parse_args()


def as_float_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def metric_mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or len(df) == 0:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def audit_subset(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) == 0:
        return {
            "num_rows": 0,
            "num_images": 0,
            "num_safe_useful": 0,
            "num_risk": 0,
            "safe_rate": None,
            "risk_rate": None,
        }
    safe = as_float_series(df, "safe_useful_label") > 0.5
    risk = as_float_series(df, "risk_label") > 0.5
    return {
        "num_rows": int(len(df)),
        "num_images": int(df["image"].nunique()) if "image" in df.columns else 0,
        "num_safe_useful": int(safe.sum()),
        "num_risk": int(risk.sum()),
        "safe_rate": float(safe.mean()),
        "risk_rate": float(risk.mean()),
        "mean_delta_dice": metric_mean(df, "delta_dice"),
        "mean_delta_iou": metric_mean(df, "delta_iou"),
        "mean_delta_recall": metric_mean(df, "delta_recall"),
        "mean_delta_boundary_iou": metric_mean(df, "delta_boundary_iou"),
        "mean_delta_boundary_f1": metric_mean(df, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": metric_mean(df, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": metric_mean(df, "delta_component_count_mae"),
        "mean_delta_merged_pred_components": metric_mean(df, "delta_merged_pred_components"),
        "mean_cut_gt_fg_frac": metric_mean(df, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": metric_mean(df, "cut_gt_gap_frac"),
        "mean_cut_area": metric_mean(df, "cut_area"),
        "mean_cut_mean_dist_in": metric_mean(df, "cut_mean_dist_in"),
        "mean_cut_bg_channel_frac_after": metric_mean(df, "cut_bg_channel_frac_after"),
        "mean_cut_bg_channel_border_contacts_after": metric_mean(df, "cut_bg_channel_border_contacts_after"),
    }


def add_gtfree_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    channel = as_float_series(out, "cut_bg_channel_frac_after")
    contacts = as_float_series(out, "cut_bg_channel_border_contacts_after")
    area = as_float_series(out, "cut_area")
    fill = as_float_series(out, "cut_fill")
    mean_dist = as_float_series(out, "cut_mean_dist_in")
    ring_bg = as_float_series(out, "ring_bg_frac")
    local_bg_delta = as_float_series(out, "local_bg_component_delta")
    # Purely deployable heuristic: favor cuts that create a clear local background
    # channel while staying tiny, shallow, and background-facing.
    out["_r245_gtfree_score"] = (
        0.5 * channel
        + 0.20 * contacts
        + 0.75 * ring_bg
        + 0.30 * local_bg_delta.clip(lower=0.0, upper=3.0)
        - 0.06 * area
        - 0.75 * fill
        - 0.35 * mean_dist
    )
    return out


def top_k_by_image(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k >= 999999 or "image" not in df.columns:
        return df
    return (
        df.sort_values(["image", "_r245_gtfree_score"], ascending=[True, False])
        .groupby("image", as_index=False, group_keys=False)
        .head(k)
    )


def filter_rows(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    mask &= as_float_series(df, "background_connected_by_cut") >= 0.5
    mask &= as_float_series(df, "cut_bg_channel_border_contacts_after") >= cfg["min_contacts"]
    mask &= as_float_series(df, "cut_bg_channel_frac_after") >= cfg["min_channel_frac"]
    if cfg["max_channel_frac"] is not None:
        mask &= as_float_series(df, "cut_bg_channel_frac_after") <= cfg["max_channel_frac"]
    mask &= as_float_series(df, "cut_mean_dist_in") <= cfg["max_mean_dist"]
    mask &= as_float_series(df, "cut_max_dist_in") <= cfg["max_dist"]
    mask &= as_float_series(df, "cut_area") >= cfg["min_area"]
    mask &= as_float_series(df, "cut_area") <= cfg["max_area"]
    mask &= as_float_series(df, "cut_fill") <= cfg["max_fill"]
    mask &= as_float_series(df, "ring_bg_frac") >= cfg["min_ring_bg"]
    mask &= as_float_series(df, "ring_fg_frac") <= cfg["max_ring_fg"]
    mask &= as_float_series(df, "cut_slenderness") >= cfg["min_cut_slenderness"]
    mask &= as_float_series(df, "cut_mean_image") >= cfg["min_cut_mean_image"]
    if cfg["max_cut_p90_grad"] is not None:
        mask &= as_float_series(df, "cut_p90_grad") <= cfg["max_cut_p90_grad"]
    if cfg["max_rank"] is not None:
        mask &= as_float_series(df, "candidate_rank") <= cfg["max_rank"]
    selected = df[mask]
    return top_k_by_image(selected, int(cfg["top_k_per_image"]))


def config_grid(top_k_values: list[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for min_contacts in [2, 3, 4]:
        for min_channel_frac in [0.10, 0.20]:
            for max_channel_frac in [None, 0.50, 0.65]:
                for max_mean_dist in [1.55, 1.75]:
                    for max_dist in [2.0, 3.0]:
                        for min_area in [8]:
                            for max_area in [16, 24, 32]:
                                if min_area > max_area:
                                    continue
                                for max_fill in [0.50, 1.00]:
                                    for min_ring_bg in [0.35, 0.45]:
                                        for max_ring_fg in [0.60, 0.75]:
                                            for min_cut_slenderness in [1.0, 2.0]:
                                                for min_cut_mean_image in [0.0, 0.55]:
                                                    for max_cut_p90_grad in [None, 0.12]:
                                                        for max_rank in [None]:
                                                            for top_k in top_k_values:
                                                                configs.append(
                                                                    {
                                                                        "min_contacts": min_contacts,
                                                                        "min_channel_frac": min_channel_frac,
                                                                        "max_channel_frac": max_channel_frac,
                                                                        "max_mean_dist": max_mean_dist,
                                                                        "max_dist": max_dist,
                                                                        "min_area": min_area,
                                                                        "max_area": max_area,
                                                                        "max_fill": max_fill,
                                                                        "min_ring_bg": min_ring_bg,
                                                                        "max_ring_fg": max_ring_fg,
                                                                        "min_cut_slenderness": min_cut_slenderness,
                                                                        "min_cut_mean_image": min_cut_mean_image,
                                                                        "max_cut_p90_grad": max_cut_p90_grad,
                                                                        "max_rank": max_rank,
                                                                        "top_k_per_image": top_k,
                                                                    }
                                                                )
    return configs


def config_passes(audit: dict[str, Any], args: argparse.Namespace) -> bool:
    mean_fg = audit.get("mean_cut_gt_fg_frac")
    if mean_fg is None:
        mean_fg = 1.0
    return bool(
        audit["num_rows"] >= args.min_selected
        and audit["num_images"] >= args.min_images
        and audit["num_safe_useful"] > audit["num_risk"]
        and (audit.get("mean_delta_recall") or 0.0) >= 0.0
        and (audit.get("mean_delta_boundary_iou") or 0.0) > 0.0
        and (audit.get("mean_delta_gap_region_fp_rate") or 0.0) < 0.0
        and float(mean_fg) <= 0.25
    )


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    safe_margin = float(row["num_safe_useful"]) - float(row["num_risk"])
    recall = float(row.get("mean_delta_recall") or 0.0)
    boundary = float(row.get("mean_delta_boundary_iou") or 0.0)
    gap_gain = -float(row.get("mean_delta_gap_region_fp_rate") or 0.0)
    fg = -float(row.get("mean_cut_gt_fg_frac") or 1.0)
    coverage = float(row["num_images"])
    return (safe_margin, recall, boundary, gap_gain, fg, coverage)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = add_gtfree_score(df)
    top_k_values = [int(value) for value in str(args.top_k_per_image).split(",") if value.strip()]

    rows: list[dict[str, Any]] = []
    for cfg in config_grid(top_k_values):
        selected = filter_rows(df, cfg)
        audit = audit_subset(selected)
        if audit["num_rows"] < args.min_selected or audit["num_images"] < args.min_images:
            continue
        row = {**cfg, **audit}
        row["passes_gate"] = bool(config_passes(audit, args))
        rows.append(row)

    rows.sort(key=rank_key, reverse=True)
    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    passing = [row for row in rows if row["passes_gate"]]
    best = passing[0] if passing else (rows[0] if rows else None)
    report = {
        "run_id": "R245-F0-gtfree-merge-filter-search",
        "input_csv": str(args.input_csv),
        "clean_test_v2_used": False,
        "writes_masks": False,
        "selection_features_are_gtfree": True,
        "gt_columns_used_only_for_audit": True,
        "gt_derived_columns": sorted([*GT_DERIVED_COLUMNS]),
        "num_input_rows": int(len(df)),
        "num_input_images": int(df["image"].nunique()) if "image" in df.columns else 0,
        "num_configs_scored": int(len(rows)),
        "num_passing_configs": int(len(passing)),
        "best": best,
        "decision": "go_next_original_val_mask_editor" if passing else "no_go_filter_not_clean_enough",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
