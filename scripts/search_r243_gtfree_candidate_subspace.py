#!/usr/bin/env python3
"""R243-F0 search for cleaner GT-free candidate subspaces.

R242 showed the crop scorer cannot safely select from the full R239 candidate
distribution. This diagnostic asks a simpler question: is there any deployable
feature subspace inside the same candidates that is already clean enough to
justify a mask-level editor?

Inputs are original-val candidate rows with GT-derived audit labels/metrics.
Selection uses only GT-free columns. GT-derived columns are used only after
selection to audit safety/effect. Writes JSON/CSV only; no masks.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search R243 GT-free candidate subspaces over R242/R239 rows.")
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r242_candidate_crop_scorer_fullval_probed_rows.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r243_gtfree_candidate_subspace_search.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r243_gtfree_candidate_subspace_search.csv"))
    parser.add_argument("--max-per-image", type=int, default=1)
    parser.add_argument("--min-selected", type=int, default=5)
    parser.add_argument("--max-risk-rate", type=float, default=0.25)
    parser.add_argument("--min-safe-rate", type=float, default=0.50)
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


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def is_safe(row: dict[str, Any]) -> bool:
    return as_float(row, "r242_safe_useful_label") > 0.5


def is_risk(row: dict[str, Any]) -> bool:
    return as_float(row, "r242_risk_label") > 0.5


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else None


def grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["image"]), []).append(row)
    return out


def gtfree_score(row: dict[str, Any]) -> tuple[float, ...]:
    # Ranking intentionally avoids GT-derived labels/metrics.
    return (
        as_float(row, "bg_channel_proxy"),
        as_float(row, "cut_component_bg_frac_after"),
        as_float(row, "cut_component_border_contacts_after"),
        as_float(row, "ring_bg_frac"),
        as_float(row, "seam_prob_p90"),
        -as_float(row, "cut_mean_dist_in"),
        -as_float(row, "cut_area"),
    )


def select_rows(rows: list[dict[str, Any]], cfg: dict[str, Any], max_per_image: int) -> list[dict[str, Any]]:
    pool = []
    for row in rows:
        if cfg["family"] != "any" and str(row.get("candidate_family")) != cfg["family"]:
            continue
        if as_float(row, "bg_channel_proxy") < cfg["min_bg_channel_proxy"]:
            continue
        if as_float(row, "cut_component_bg_frac_after") < cfg["min_cut_bg_frac_after"]:
            continue
        if as_float(row, "cut_component_border_contacts_after") < cfg["min_border_contacts"]:
            continue
        if as_float(row, "ring_bg_frac") < cfg["min_ring_bg_frac"]:
            continue
        if as_float(row, "ring_fg_frac") > cfg["max_ring_fg_frac"]:
            continue
        if as_float(row, "cut_mean_dist_in") > cfg["max_mean_dist_in"]:
            continue
        if as_float(row, "cut_max_dist_in") > cfg["max_max_dist_in"]:
            continue
        if as_float(row, "cut_area") > cfg["max_cut_area"]:
            continue
        if as_float(row, "cut_fill") > cfg["max_cut_fill"]:
            continue
        if as_float(row, "cut_slenderness") < cfg["min_slenderness"]:
            continue
        if as_float(row, "seam_prob_p90") < cfg["min_seam_prob_p90"]:
            continue
        if as_float(row, "action_frac") > cfg["max_action_frac"]:
            continue
        pool.append(row)
    selected = []
    for image_rows in grouped(pool).values():
        image_rows.sort(key=gtfree_score, reverse=True)
        selected.extend(image_rows[:max_per_image])
    return selected


def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
    safe = sum(is_safe(row) for row in selected)
    risk = sum(is_risk(row) for row in selected)
    return {
        "selected": len(selected),
        "selected_images": len({row["image"] for row in selected}),
        "safe": safe,
        "risk": risk,
        "safe_rate": float(safe / max(1, len(selected))),
        "risk_rate": float(risk / max(1, len(selected))),
        "mean_delta_dice": mean(selected, "delta_dice"),
        "mean_delta_iou": mean(selected, "delta_iou"),
        "mean_delta_recall": mean(selected, "delta_recall"),
        "mean_delta_boundary_iou": mean(selected, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean(selected, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean(selected, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean(selected, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean(selected, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean(selected, "cut_gt_gap_frac"),
        "mean_cut_area": mean(selected, "cut_area"),
    }


def candidate_configs() -> list[dict[str, Any]]:
    configs = []
    for values in product(
        ["bg_seed"],
        [1.0],
        [0.30, 0.45, 0.60],
        [4.0],
        [0.20, 0.35],
        [0.80],
        [1.25, 1.75],
        [2.0, 3.0],
        [4.0, 8.0],
        [0.05, 0.10],
        [2.0, 4.0],
        [0.0, 0.3],
        [0.15, 0.25],
    ):
        (
            family,
            min_bg_channel_proxy,
            min_cut_bg_frac_after,
            min_border_contacts,
            min_ring_bg_frac,
            max_ring_fg_frac,
            max_mean_dist_in,
            max_max_dist_in,
            max_cut_area,
            max_cut_fill,
            min_slenderness,
            min_seam_prob_p90,
            max_action_frac,
        ) = values
        if min_ring_bg_frac > 1.0 - max_ring_fg_frac + 1e-9:
            continue
        configs.append(
            {
                "family": family,
                "min_bg_channel_proxy": min_bg_channel_proxy,
                "min_cut_bg_frac_after": min_cut_bg_frac_after,
                "min_border_contacts": min_border_contacts,
                "min_ring_bg_frac": min_ring_bg_frac,
                "max_ring_fg_frac": max_ring_fg_frac,
                "max_mean_dist_in": max_mean_dist_in,
                "max_max_dist_in": max_max_dist_in,
                "max_cut_area": max_cut_area,
                "max_cut_fill": max_cut_fill,
                "min_slenderness": min_slenderness,
                "min_seam_prob_p90": min_seam_prob_p90,
                "max_action_frac": max_action_frac,
            }
        )
    return configs


def main() -> None:
    args = parse_args()
    rows = read_csv(args.candidate_csv)
    out = []
    for idx, cfg in enumerate(candidate_configs(), start=1):
        selected = select_rows(rows, cfg, args.max_per_image)
        if len(selected) < args.min_selected:
            continue
        summary = summarize(selected)
        decision = (
            summary["safe_rate"] >= args.min_safe_rate
            and summary["risk_rate"] <= args.max_risk_rate
            and (summary["mean_delta_recall"] is not None and summary["mean_delta_recall"] >= -1e-8)
            and (summary["mean_delta_boundary_iou"] is not None and summary["mean_delta_boundary_iou"] > 0.0)
            and (summary["mean_delta_gap_region_fp_rate"] is not None and summary["mean_delta_gap_region_fp_rate"] < 0.0)
            and (summary["mean_cut_gt_fg_frac"] is not None and summary["mean_cut_gt_fg_frac"] <= 0.25)
        )
        out.append({"config_id": idx, "decision_pass": float(decision), **cfg, **summary})
    out.sort(
        key=lambda row: (
            row["decision_pass"],
            -row["risk_rate"],
            row["safe_rate"],
            row["safe"],
            row["mean_delta_boundary_iou"] or -999.0,
            -(row["mean_delta_gap_region_fp_rate"] or 999.0),
        ),
        reverse=True,
    )
    report = {
        "run_id": "R243-F0-gtfree-candidate-subspace-search",
        "candidate_csv": str(args.candidate_csv),
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "original_val_candidate_subspace_diagnostic",
        "num_input_rows": len(rows),
        "num_input_images": len({row["image"] for row in rows}),
        "num_safe_labels": int(sum(is_safe(row) for row in rows)),
        "num_risk_labels": int(sum(is_risk(row) for row in rows)),
        "num_configs_evaluated_after_min_selected": len(out),
        "num_pass": int(sum(float(row["decision_pass"]) > 0.5 for row in out)),
        "best": out[0] if out else None,
        "warning": "Selection uses GT-free columns, but labels/metrics are GT-derived original-val audit only. No masks are written.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, out)
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "num_pass": report["num_pass"], "best": report["best"]}, indent=2))


if __name__ == "__main__":
    main()
