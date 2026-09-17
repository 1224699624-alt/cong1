#!/usr/bin/env python3
"""Summarize R211-F2 full-val result and rank visual-audit cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize R211-F2 validation result.")
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r211_f2_remote_fullval_val_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r211_f2_remote_fullval_val_per_image.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r211_f2_remote_fullval_case_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r211_f2_remote_fullval_visual_candidates.csv"))
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rank_cases(rows: list[dict[str, str]], top_k: int) -> list[dict[str, Any]]:
    positives = [row for row in rows if f(row, "accepted") > 0.5]
    safe_gain = sorted(
        positives,
        key=lambda row: (
            -f(row, "delta_gap_region_fp_rate"),
            f(row, "delta_boundary_iou"),
            -f(row, "delta_component_count_mae"),
            -abs(f(row, "delta_recall")),
        ),
        reverse=True,
    )[:top_k]
    boundary_gain = sorted(positives, key=lambda row: f(row, "delta_boundary_iou"), reverse=True)[:top_k]
    component_gain = sorted(positives, key=lambda row: -f(row, "delta_component_count_mae"), reverse=True)[:top_k]
    risk = sorted(
        positives,
        key=lambda row: (
            f(row, "delta_component_count_mae"),
            -f(row, "delta_boundary_iou"),
            -f(row, "delta_dice"),
            -f(row, "delta_recall"),
        ),
        reverse=True,
    )[:top_k]

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for group, group_rows in [
        ("safe_gap_gain", safe_gain),
        ("boundary_gain", boundary_gain),
        ("component_gain", component_gain),
        ("risk_check", risk),
    ]:
        for rank, row in enumerate(group_rows, start=1):
            key = (group, row["image"])
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "group": group,
                    "rank": rank,
                    "image": row["image"],
                    "accepted_cut_pixels": f(row, "accepted_cut_pixels"),
                    "delta_dice": f(row, "delta_dice"),
                    "delta_recall": f(row, "delta_recall"),
                    "delta_boundary_iou": f(row, "delta_boundary_iou"),
                    "delta_boundary_f1": f(row, "delta_boundary_f1"),
                    "delta_gap_region_fp_rate": f(row, "delta_gap_region_fp_rate"),
                    "delta_component_count_mae": f(row, "delta_component_count_mae"),
                    "delta_pred_component_count": f(row, "delta_pred_component_count"),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    rows = read_rows(args.per_image_csv)
    visual_candidates = rank_cases(rows, args.top_k)
    best = summary.get("best", {})
    mean = best.get("mean", {})
    output = {
        "summary_json": str(args.summary_json),
        "per_image_csv": str(args.per_image_csv),
        "val_gate_pass": summary.get("val_gate_pass"),
        "num_evaluated": summary.get("num_evaluated"),
        "best_threshold": best.get("threshold"),
        "best_mean": mean,
        "accepted_images": int(sum(f(row, "accepted") > 0.5 for row in rows)),
        "accepted_rate": float(sum(f(row, "accepted") > 0.5 for row in rows) / max(1, len(rows))),
        "visual_candidates": visual_candidates,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.output_csv, visual_candidates)
    print(json.dumps({k: output[k] for k in ["val_gate_pass", "num_evaluated", "best_threshold", "accepted_images", "accepted_rate"]}, indent=2))


if __name__ == "__main__":
    main()
