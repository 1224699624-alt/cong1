#!/usr/bin/env python3
"""Audit R240 GT-free geometry gates on an existing candidate CSV.

This is an original-val development audit. It does not write masks and does
not use clean-test-v2. Selection uses only GT-free candidate columns; GT-derived
columns are used afterward to measure safety and potential benefit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


GT_DERIVED_PREFIXES = ("candidate_", "anchor_", "delta_")
GT_DERIVED_EXACT = {"cut_gt_fg_frac", "cut_gt_gap_frac", "overerosion_proxy", "accepted", "reject_reason"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R240 geometry gates from candidate CSV.")
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r239_gtfree_bg_channel_light_fullval_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r240_geometry_gate_from_r239_candidates_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r240_geometry_gate_from_r239_candidates_selected.csv"))
    parser.add_argument("--min-bg-channel-frac", type=float, default=0.20)
    parser.add_argument("--min-ring-bg-frac", type=float, default=0.20)
    parser.add_argument("--max-ring-bg-frac", type=float, default=0.60)
    parser.add_argument("--max-cut-mean-dist-in", type=float, default=1.75)
    parser.add_argument("--max-cut-max-dist-in", type=float, default=3.0)
    parser.add_argument("--max-cut-area", type=float, default=24.0)
    parser.add_argument("--max-bridge-score-p90", type=float, default=3.0)
    parser.add_argument("--max-per-image", type=int, default=1)
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


def is_selected(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        as_float(row, "bg_channel_proxy") > 0.5
        and as_float(row, "cut_component_bg_frac_after") >= args.min_bg_channel_frac
        and as_float(row, "ring_bg_frac") >= args.min_ring_bg_frac
        and as_float(row, "ring_bg_frac") <= args.max_ring_bg_frac
        and as_float(row, "cut_mean_dist_in") <= args.max_cut_mean_dist_in
        and as_float(row, "cut_max_dist_in") <= args.max_cut_max_dist_in
        and as_float(row, "cut_area") <= args.max_cut_area
        and as_float(row, "bridge_score_p90") <= args.max_bridge_score_p90
    )


def is_safe_gain(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= 0.0
        and as_float(row, "delta_boundary_iou") >= 0.0
        and as_float(row, "delta_boundary_f1") >= 0.0
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
        and as_float(row, "cut_gt_fg_frac") <= 0.25
    )


def is_risk(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "cut_gt_fg_frac") > 0.5
        or as_float(row, "delta_recall") < 0.0
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
    )


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({str(row.get("image")) for row in rows}),
        "num_safe_gain": int(sum(is_safe_gain(row) for row in rows)),
        "num_risk": int(sum(is_risk(row) for row in rows)),
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
    selected = [row for row in rows if is_selected(row, args)]
    selected.sort(
        key=lambda row: (
            str(row.get("image")),
            -as_float(row, "cut_component_bg_frac_after"),
            -as_float(row, "ring_bg_frac"),
            as_float(row, "cut_mean_dist_in"),
            -as_float(row, "seam_prob_p90"),
            -as_float(row, "seam_prob_mean"),
            as_float(row, "cut_area"),
        )
    )
    per_image: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for row in selected:
        image = str(row.get("image"))
        count = counts.get(image, 0)
        if count >= args.max_per_image:
            continue
        counts[image] = count + 1
        out = dict(row)
        out["r240_gate_safe_gain"] = float(is_safe_gain(row))
        out["r240_gate_risk"] = float(is_risk(row))
        per_image.append(out)

    report = {
        "run_id": "R240-geometry-gate-from-candidates",
        "candidate_csv": str(args.candidate_csv),
        "clean_test_v2_used": False,
        "writes_masks": False,
        "selection_uses_gt": False,
        "audit_uses_gt": True,
        "selected_candidate_summary": summarize(per_image),
        "all_candidate_summary": summarize(rows),
        "gate": {
            "min_bg_channel_frac": args.min_bg_channel_frac,
            "min_ring_bg_frac": args.min_ring_bg_frac,
            "max_ring_bg_frac": args.max_ring_bg_frac,
            "max_cut_mean_dist_in": args.max_cut_mean_dist_in,
            "max_cut_max_dist_in": args.max_cut_max_dist_in,
            "max_cut_area": args.max_cut_area,
            "max_bridge_score_p90": args.max_bridge_score_p90,
            "max_per_image": args.max_per_image,
        },
        "decision": "no_go_geometry_gate_too_risky" if summarize(per_image)["num_risk"] > summarize(per_image)["num_safe_gain"] else "inspect_visuals_or_mask_level_next",
        "warning": "Candidate rows come from original-val R239; this is gate audit only, not clean-test-v2 evidence.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, per_image)
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "decision": report["decision"], "selected": report["selected_candidate_summary"]}, indent=2))


if __name__ == "__main__":
    main()
