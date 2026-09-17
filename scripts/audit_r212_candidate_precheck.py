#!/usr/bin/env python3
"""Candidate-level precheck for R212 acceptance-gate runs.

This is a lightweight risk audit over an already materialized candidate CSV.
It does not produce masks and is not a substitute for full R201 evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


DEFAULT_METRICS = [
    "delta_dice",
    "delta_boundary_iou",
    "delta_boundary_f1",
    "delta_surface_dice_2px",
    "delta_surface_dice_5px",
    "delta_assd_px",
    "delta_gap_region_fp_rate",
    "delta_recall",
    "label_positive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit likely accepted R212 candidates from a candidate CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--threshold-grid", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def parse_thresholds(text: str, fallback: str) -> list[float]:
    source = text.strip() or fallback
    return [float(item) for item in source.split(",") if item.strip()]


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def predict_probs(rows: list[dict[str, Any]], payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    args = payload.get("args", {}) if isinstance(payload, dict) else {}
    feature_keys = payload.get("feature_keys", []) if isinstance(payload, dict) else []
    if not feature_keys:
        raise ValueError("checkpoint payload does not contain feature_keys")
    for row in rows:
        x = np.asarray([[as_float(row, key) for key in feature_keys]], dtype=np.float32)
        row["candidate_prob"] = float(model.predict_proba(x)[0, 1])
    return rows, args, feature_keys


def passes_filters(row: dict[str, Any], args: dict[str, Any]) -> bool:
    if as_float(row, "cut_frac_component", 1.0) > float(args.get("max_cut_frac_component") or 1.0):
        return False
    max_rank = int(args.get("max_accept_candidate_rank") or 0)
    if max_rank > 0 and as_float(row, "candidate_rank") > max_rank:
        return False
    if as_float(row, "gradient_ratio") < float(args.get("min_accept_gradient_ratio") or 0.0):
        return False
    if as_float(row, "dist_ratio") < float(args.get("min_accept_dist_ratio") or 0.0):
        return False
    if as_float(row, "slenderness") < float(args.get("min_accept_slenderness") or 0.0):
        return False
    if as_float(row, "fill") > float(args.get("max_accept_fill") or 1.0):
        return False
    return True


def group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    return grouped


def summarize_threshold(threshold: float, grouped: dict[str, list[dict[str, Any]]], args: dict[str, Any]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    for image, image_rows in grouped.items():
        ranked = sorted(image_rows, key=lambda row: as_float(row, "candidate_prob"), reverse=True)
        for row in ranked:
            if as_float(row, "candidate_prob") < threshold:
                continue
            if not passes_filters(row, args):
                continue
            accepted.append({"image": image, **row})
            break

    negative = [row for row in accepted if as_float(row, "label_positive") <= 0.5]
    risk = [
        row
        for row in accepted
        if as_float(row, "delta_dice") < 0.0
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_recall") < 0.0
    ]
    return {
        "threshold": threshold,
        "accepted_images": len(accepted),
        "label_negative_accepted": len(negative),
        "risk_accepted": len(risk),
        "accepted": accepted,
    }


def compact_row(threshold: float, row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "threshold": threshold,
        "image": row.get("image"),
        "candidate_prob": as_float(row, "candidate_prob"),
        "candidate_rank": as_float(row, "candidate_rank"),
        "cut_frac_component": as_float(row, "cut_frac_component"),
    }
    for metric in DEFAULT_METRICS:
        out[metric] = as_float(row, metric)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "image",
        "candidate_prob",
        "candidate_rank",
        "cut_frac_component",
        *DEFAULT_METRICS,
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cli = parse_args()
    payload = joblib.load(cli.checkpoint)
    rows, checkpoint_args, feature_keys = predict_probs(read_candidates(cli.candidate_csv), payload)
    thresholds = parse_thresholds(cli.threshold_grid, str(checkpoint_args.get("threshold_grid") or "0.50,0.60,0.70,0.80"))
    grouped = group_by_image(rows)
    summaries = [summarize_threshold(threshold, grouped, checkpoint_args) for threshold in thresholds]
    flat_rows = [compact_row(item["threshold"], row) for item in summaries for row in item["accepted"]]
    report = {
        "run_id": "R212-candidate-precheck",
        "candidate_csv": str(cli.candidate_csv),
        "checkpoint": str(cli.checkpoint),
        "warning": "candidate-level risk audit only; not a substitute for full R201 mask evaluation",
        "num_candidate_rows": len(rows),
        "num_images_with_candidates": len(grouped),
        "feature_keys": feature_keys,
        "checkpoint_filter_args": {
            key: checkpoint_args.get(key)
            for key in [
                "threshold_grid",
                "max_cut_frac_component",
                "max_accept_candidate_rank",
                "min_accept_gradient_ratio",
                "min_accept_dist_ratio",
                "min_accept_slenderness",
                "max_accept_fill",
                "max_pred_component_increase",
                "max_step_component_increase",
            ]
        },
        "thresholds": [
            {
                key: value
                for key, value in item.items()
                if key != "accepted"
            }
            for item in summaries
        ],
        "risk_images_by_threshold": {
            f"{item['threshold']:.2f}": [
                row["image"]
                for row in item["accepted"]
                if as_float(row, "label_positive") <= 0.5
                or as_float(row, "delta_boundary_iou") < 0.0
                or as_float(row, "delta_dice") < 0.0
            ]
            for item in summaries
        },
    }
    cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    cli.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(cli.output_csv, flat_rows)
    print(json.dumps({"output_json": str(cli.output_json), "output_csv": str(cli.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
