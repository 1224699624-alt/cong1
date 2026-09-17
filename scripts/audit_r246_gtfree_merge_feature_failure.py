"""R246-F0 audit of why R245 GT-free filters fail on R244 candidates.

This is an analysis-only script. It reads an R244 candidate CSV and an optional
R245 search JSON, then compares GT-free feature distributions for:

- safe useful rows
- risk rows
- R245 best-rule selected rows
- oracle-safe best per image

GT labels are used only to define audit groups and outcome summaries. The goal
is to decide whether the next step should change candidate generation, ranking,
or both.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FEATURES = [
    "dist_percentile",
    "candidate_rank",
    "background_connected_by_cut",
    "cut_area",
    "cut_anchor_area_frac",
    "cut_bbox_h",
    "cut_bbox_w",
    "cut_fill",
    "cut_slenderness",
    "cut_mean_dist_in",
    "cut_max_dist_in",
    "ring_bg_frac",
    "ring_fg_frac",
    "cut_bg_channel_area_after",
    "cut_bg_channel_border_contacts_after",
    "cut_bg_channel_frac_after",
    "local_bg_component_delta",
    "local_bg_components_before",
    "local_bg_components_after",
    "component_area",
    "component_bbox_h",
    "component_bbox_w",
    "component_fill",
    "component_slenderness",
    "cut_mean_image",
    "cut_std_image",
    "cut_mean_grad",
    "cut_p90_grad",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R245/R244 GT-free feature failure modes.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--r245-json", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/analysis/r246_gtfree_merge_feature_failure.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/analysis/r246_gtfree_merge_feature_failure_feature_table.csv"),
    )
    return parser.parse_args()


def sf(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def summarize(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) == 0:
        return {"num_rows": 0, "num_images": 0}
    safe = sf(df, "safe_useful_label") > 0.5
    risk = sf(df, "risk_label") > 0.5
    return {
        "num_rows": int(len(df)),
        "num_images": int(df["image"].nunique()) if "image" in df.columns else 0,
        "num_safe_useful": int(safe.sum()),
        "num_risk": int(risk.sum()),
        "mean_delta_dice": mean(df, "delta_dice"),
        "mean_delta_iou": mean(df, "delta_iou"),
        "mean_delta_recall": mean(df, "delta_recall"),
        "mean_delta_boundary_iou": mean(df, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean(df, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean(df, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean(df, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean(df, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean(df, "cut_gt_gap_frac"),
    }


def mean(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or len(df) == 0:
        return None
    values = pd.to_numeric(df[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def feature_rows(groups: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        for group_name, df in groups.items():
            if feature not in df.columns or len(df) == 0:
                continue
            values = pd.to_numeric(df[feature], errors="coerce").dropna()
            if len(values) == 0:
                continue
            rows.append(
                {
                    "feature": feature,
                    "group": group_name,
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "p10": float(values.quantile(0.10)),
                    "p50": float(values.quantile(0.50)),
                    "p90": float(values.quantile(0.90)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    return rows


def oracle_safe_best(df: pd.DataFrame) -> pd.DataFrame:
    safe = df[(sf(df, "safe_useful_label") > 0.5) & (sf(df, "risk_label") <= 0.5)].copy()
    if len(safe) == 0 or "image" not in safe.columns:
        return safe
    safe["_sort_component"] = sf(safe, "delta_component_count_mae")
    safe["_sort_boundary"] = -sf(safe, "delta_boundary_iou")
    safe["_sort_gap"] = sf(safe, "delta_gap_region_fp_rate")
    return (
        safe.sort_values(["image", "_sort_component", "_sort_boundary", "_sort_gap"])
        .groupby("image", group_keys=False)
        .head(1)
        .drop(columns=["_sort_component", "_sort_boundary", "_sort_gap"])
    )


def selected_by_r245_best(df: pd.DataFrame, r245_json: Path | None) -> pd.DataFrame:
    if r245_json is None or not r245_json.exists():
        return df.iloc[0:0].copy()
    best = json.loads(r245_json.read_text(encoding="utf-8")).get("best") or {}
    if not best:
        return df.iloc[0:0].copy()
    mask = pd.Series(True, index=df.index)
    mask &= sf(df, "background_connected_by_cut") >= 0.5
    mask &= sf(df, "cut_bg_channel_border_contacts_after") >= float(best["min_contacts"])
    mask &= sf(df, "cut_bg_channel_frac_after") >= float(best["min_channel_frac"])
    if best.get("max_channel_frac") is not None and not pd.isna(best.get("max_channel_frac")):
        mask &= sf(df, "cut_bg_channel_frac_after") <= float(best["max_channel_frac"])
    mask &= sf(df, "cut_mean_dist_in") <= float(best["max_mean_dist"])
    mask &= sf(df, "cut_max_dist_in") <= float(best["max_dist"])
    mask &= sf(df, "cut_area") >= float(best["min_area"])
    mask &= sf(df, "cut_area") <= float(best["max_area"])
    mask &= sf(df, "cut_fill") <= float(best["max_fill"])
    mask &= sf(df, "ring_bg_frac") >= float(best["min_ring_bg"])
    mask &= sf(df, "ring_fg_frac") <= float(best["max_ring_fg"])
    mask &= sf(df, "cut_slenderness") >= float(best["min_cut_slenderness"])
    mask &= sf(df, "cut_mean_image") >= float(best["min_cut_mean_image"])
    if best.get("max_cut_p90_grad") is not None and not pd.isna(best.get("max_cut_p90_grad")):
        mask &= sf(df, "cut_p90_grad") <= float(best["max_cut_p90_grad"])
    selected = df[mask].copy()
    if int(best.get("top_k_per_image", 999999)) < 999999 and len(selected):
        selected["_score"] = (
            0.5 * sf(selected, "cut_bg_channel_frac_after")
            + 0.20 * sf(selected, "cut_bg_channel_border_contacts_after")
            + 0.75 * sf(selected, "ring_bg_frac")
            + 0.30 * sf(selected, "local_bg_component_delta").clip(lower=0.0, upper=3.0)
            - 0.06 * sf(selected, "cut_area")
            - 0.75 * sf(selected, "cut_fill")
            - 0.35 * sf(selected, "cut_mean_dist_in")
        )
        selected = (
            selected.sort_values(["image", "_score"], ascending=[True, False])
            .groupby("image", group_keys=False)
            .head(int(best["top_k_per_image"]))
            .drop(columns=["_score"])
        )
    return selected


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.candidate_csv)
    groups = {
        "all": df,
        "safe_useful": df[sf(df, "safe_useful_label") > 0.5],
        "risk": df[sf(df, "risk_label") > 0.5],
        "oracle_safe_best": oracle_safe_best(df),
        "r245_best_selected": selected_by_r245_best(df, args.r245_json),
    }
    feature_table = pd.DataFrame(feature_rows(groups))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.output_csv, index=False)

    report = {
        "run_id": "R246-F0-gtfree-merge-feature-failure-audit",
        "candidate_csv": str(args.candidate_csv),
        "r245_json": str(args.r245_json) if args.r245_json else None,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "gt_columns_used_only_for_audit": True,
        "groups": {name: summarize(group) for name, group in groups.items()},
        "interpretation": (
            "If r245_best_selected is much worse than oracle_safe_best while both share "
            "overlapping GT-free feature ranges, the next route should revise candidate "
            "generation/component preselection rather than only tightening scalar thresholds."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
