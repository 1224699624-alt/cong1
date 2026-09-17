#!/usr/bin/env python3
"""Search candidate-level R213 policy directions on validation candidates.

This is a diagnostic tool. It uses validation candidate labels/metric deltas
to understand which inference-filter families might improve useful-candidate
recall while limiting risk. It does not produce masks and is not clean-test-v2
evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np


DEFAULT_THRESHOLDS = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"
DEFAULT_MAX_CUT_FRACS = "0.006,0.008,0.010,0.012,0.016,0.020"
DEFAULT_MAX_RANKS = "2,3,4,5"
DEFAULT_MIN_GRADIENTS = "1.4,1.6,1.8,2.0"
DEFAULT_MIN_DISTS = "0.09,0.10,0.11,0.12"
DEFAULT_MIN_SLENDERNESS = "1.0,2.0,3.0,4.0"
DEFAULT_MAX_FILLS = "0.80,0.90,1.00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search R213 candidate-policy directions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--max-cut-fracs", default=DEFAULT_MAX_CUT_FRACS)
    parser.add_argument("--max-ranks", default=DEFAULT_MAX_RANKS)
    parser.add_argument("--min-gradients", default=DEFAULT_MIN_GRADIENTS)
    parser.add_argument("--min-dists", default=DEFAULT_MIN_DISTS)
    parser.add_argument("--min-slenderness", default=DEFAULT_MIN_SLENDERNESS)
    parser.add_argument("--max-fills", default=DEFAULT_MAX_FILLS)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def predict_probs(rows: list[dict[str, Any]], checkpoint: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = joblib.load(checkpoint)
    model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    feature_keys = payload.get("feature_keys", []) if isinstance(payload, dict) else []
    if not feature_keys:
        raise ValueError("checkpoint payload does not contain feature_keys")
    for row in rows:
        x = np.asarray([[as_float(row, key) for key in feature_keys]], dtype=np.float32)
        row["candidate_prob"] = float(model.predict_proba(x)[0, 1])
        row["strict_useful"] = float(is_strict_useful(row))
        row["risk"] = float(is_risk(row))
        row["positive"] = float(as_float(row, "label_positive") > 0.5)
    return rows, feature_keys


def is_strict_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= -8e-4
        and as_float(row, "delta_boundary_iou") >= 1e-4
        and as_float(row, "delta_boundary_f1") >= 1e-4
        and as_float(row, "delta_surface_dice_2px") >= 1e-4
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
    )


def is_risk(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
        or as_float(row, "delta_recall") < -8e-4
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_boundary_f1") < 0.0
        or as_float(row, "delta_surface_dice_2px") < 0.0
    )


def passes_policy(row: dict[str, Any], policy: dict[str, float | int]) -> bool:
    return bool(
        as_float(row, "candidate_prob") >= float(policy["threshold"])
        and as_float(row, "cut_frac_component", 1.0) <= float(policy["max_cut_frac"])
        and as_float(row, "candidate_rank") <= float(policy["max_rank"])
        and as_float(row, "gradient_ratio") >= float(policy["min_gradient"])
        and as_float(row, "dist_ratio") >= float(policy["min_dist"])
        and as_float(row, "slenderness") >= float(policy["min_slenderness"])
        and as_float(row, "fill") <= float(policy["max_fill"])
    )


def group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)
    return grouped


def evaluate_policy(grouped: dict[str, list[dict[str, Any]]], policy: dict[str, float | int]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    for image, image_rows in grouped.items():
        for row in sorted(image_rows, key=lambda item: as_float(item, "candidate_prob"), reverse=True):
            if passes_policy(row, policy):
                accepted.append(row)
                break
    n = len(accepted)
    useful = [row for row in accepted if as_float(row, "strict_useful") > 0.5]
    risk = [row for row in accepted if as_float(row, "risk") > 0.5]
    positive = [row for row in accepted if as_float(row, "positive") > 0.5]
    useful_count = len(useful)
    risk_count = len(risk)
    score = useful_count - 1.5 * risk_count + 0.25 * n
    return {
        **policy,
        "accepted": n,
        "positive": len(positive),
        "strict_useful": useful_count,
        "risk": risk_count,
        "precision_useful": float(useful_count / n) if n else 0.0,
        "risk_rate": float(risk_count / n) if n else 0.0,
        "score": float(score),
        "accepted_images": [str(row["image"]) for row in accepted],
        "useful_images": [str(row["image"]) for row in useful],
        "risk_images": [str(row["image"]) for row in risk],
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "threshold",
        "max_cut_frac",
        "max_rank",
        "min_gradient",
        "min_dist",
        "min_slenderness",
        "max_fill",
        "accepted",
        "positive",
        "strict_useful",
        "risk",
        "precision_useful",
        "risk_rate",
        "score",
    ]
    return {key: row[key] for key in keep}


def constrained_frontiers(results: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    frontiers: dict[str, list[dict[str, Any]]] = {}
    for max_risk in [0, 1, 2, 5, 10, 20]:
        subset = [row for row in results if int(row["risk"]) <= max_risk]
        subset = sorted(
            subset,
            key=lambda row: (
                row["strict_useful"],
                row["precision_useful"],
                row["accepted"],
                -row["risk"],
                row["score"],
            ),
            reverse=True,
        )
        frontiers[f"risk_le_{max_risk}"] = [compact(row) for row in subset[:top_k]]
    for min_useful in [2, 3, 5, 10, 15]:
        subset = [row for row in results if int(row["strict_useful"]) >= min_useful]
        subset = sorted(
            subset,
            key=lambda row: (
                -row["risk"],
                row["precision_useful"],
                row["strict_useful"],
                row["accepted"],
                row["score"],
            ),
            reverse=True,
        )
        frontiers[f"useful_ge_{min_useful}"] = [compact(row) for row in subset[:top_k]]
    return frontiers


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "max_cut_frac",
        "max_rank",
        "min_gradient",
        "min_dist",
        "min_slenderness",
        "max_fill",
        "accepted",
        "positive",
        "strict_useful",
        "risk",
        "precision_useful",
        "risk_rate",
        "score",
        "accepted_images",
        "useful_images",
        "risk_images",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["accepted_images", "useful_images", "risk_images"]:
                out[key] = "|".join(out.get(key) or [])
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    rows, feature_keys = predict_probs(read_candidates(args.candidate_csv), args.checkpoint)
    grouped = group_by_image(rows)
    policies: list[dict[str, float | int]] = []
    for threshold in parse_float_list(args.thresholds):
        for max_cut_frac in parse_float_list(args.max_cut_fracs):
            for max_rank in parse_int_list(args.max_ranks):
                for min_gradient in parse_float_list(args.min_gradients):
                    for min_dist in parse_float_list(args.min_dists):
                        for min_slenderness in parse_float_list(args.min_slenderness):
                            for max_fill in parse_float_list(args.max_fills):
                                policies.append(
                                    {
                                        "threshold": threshold,
                                        "max_cut_frac": max_cut_frac,
                                        "max_rank": max_rank,
                                        "min_gradient": min_gradient,
                                        "min_dist": min_dist,
                                        "min_slenderness": min_slenderness,
                                        "max_fill": max_fill,
                                    }
                                )
    results = [evaluate_policy(grouped, policy) for policy in policies]
    useful_first = sorted(
        results,
        key=lambda row: (
            row["strict_useful"],
            -row["risk"],
            row["precision_useful"],
            row["accepted"],
            row["score"],
        ),
        reverse=True,
    )
    safe_first = sorted(
        results,
        key=lambda row: (
            -row["risk"],
            row["strict_useful"],
            row["precision_useful"],
            row["accepted"],
            row["score"],
        ),
        reverse=True,
    )
    balanced = sorted(results, key=lambda row: row["score"], reverse=True)
    report = {
        "run_id": "R213-candidate-policy-search",
        "warning": "validation candidate-label search only; not deployable evidence and not clean-test-v2 evidence",
        "candidate_csv": str(args.candidate_csv),
        "checkpoint": str(args.checkpoint),
        "feature_keys": feature_keys,
        "num_candidates": len(rows),
        "num_images": len(grouped),
        "num_policies": len(results),
        "total_strict_useful_candidates": int(sum(as_float(row, "strict_useful") > 0.5 for row in rows)),
        "total_risk_candidates": int(sum(as_float(row, "risk") > 0.5 for row in rows)),
        "best_balanced": [compact(row) for row in balanced[: args.top_k]],
        "best_useful_recall": [compact(row) for row in useful_first[: args.top_k]],
        "best_safe": [compact(row) for row in safe_first[: args.top_k]],
        "constrained_frontiers": constrained_frontiers(results, args.top_k),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, balanced[: max(args.top_k, 200)])
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "num_policies": len(results),
                "best_balanced": report["best_balanced"][:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
