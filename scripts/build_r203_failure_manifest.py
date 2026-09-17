#!/usr/bin/env python3
"""Build the R203 failure manifest from R201 per-image metrics.

The manifest is a diagnostic artifact, not a tuning result. It compares R110
against ARAA and nnU-Net risk-audit metrics to identify cases where a future
R110-derived method should reduce bridges/gap false positives without creating
fragmentation or recall loss.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS_DIR = Path("outputs/analysis/r201_unified_eval")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/r203_failure_manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R203 failure manifest from R201 metrics.")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def by_image(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("per_image") or []
    return {str(row["image"]): row for row in rows}


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def manifest_row(image: str, r110: dict[str, Any], araa: dict[str, Any], nnunet: dict[str, Any] | None) -> dict[str, Any]:
    nn = nnunet or {}
    r110_merge = f(r110, "component_merge_rate")
    araa_merge = f(araa, "component_merge_rate")
    r110_comp = f(r110, "component_count_mae")
    araa_comp = f(araa, "component_count_mae")
    r110_gap = f(r110, "gap_region_fp_rate")
    araa_gap = f(araa, "gap_region_fp_rate")
    r110_recall = f(r110, "recall")
    araa_recall = f(araa, "recall")
    r110_dice = f(r110, "dice")
    araa_dice = f(araa, "dice")
    r110_boundary = f(r110, "boundary_iou")
    araa_boundary = f(araa, "boundary_iou")

    component_deficit = max(0.0, r110_comp - min(araa_comp, f(nn, "component_count_mae", araa_comp)))
    merge_deficit = max(0.0, r110_merge - min(araa_merge, f(nn, "component_merge_rate", araa_merge)))
    gap_excess = max(0.0, r110_gap - min(araa_gap, f(nn, "gap_region_fp_rate", araa_gap)))
    recall_buffer = r110_recall - araa_recall
    dice_buffer = r110_dice - araa_dice
    boundary_buffer = r110_boundary - araa_boundary

    priority = 2.0 * merge_deficit + 0.20 * component_deficit + 1.5 * gap_excess
    priority += 0.25 * max(0.0, dice_buffer) + 0.15 * max(0.0, boundary_buffer)
    priority -= 0.5 * max(0.0, -recall_buffer)

    if merge_deficit > 0 and component_deficit <= 2.0 and recall_buffer >= -0.01:
        failure_type = "clean_bridge_candidate"
    elif component_deficit > 2.0:
        failure_type = "fragmentation_risk_high_component_error"
    elif gap_excess > 0.03:
        failure_type = "gap_fp_candidate"
    elif r110_recall < 0.90:
        failure_type = "recall_risk_do_not_cut_aggressively"
    else:
        failure_type = "mixed_or_low_priority"

    return {
        "image": image,
        "priority_score": priority,
        "failure_type": failure_type,
        "r110_dice": r110_dice,
        "araa_dice": araa_dice,
        "nnunet_dice": f(nn, "dice", -1.0),
        "r110_recall": r110_recall,
        "araa_recall": araa_recall,
        "r110_boundary_iou": r110_boundary,
        "araa_boundary_iou": araa_boundary,
        "r110_gap_fp": r110_gap,
        "araa_gap_fp": araa_gap,
        "nnunet_gap_fp": f(nn, "gap_region_fp_rate", -1.0),
        "r110_component_merge": r110_merge,
        "araa_component_merge": araa_merge,
        "nnunet_component_merge": f(nn, "component_merge_rate", -1.0),
        "r110_component_mae": r110_comp,
        "araa_component_mae": araa_comp,
        "nnunet_component_mae": f(nn, "component_count_mae", -1.0),
        "component_deficit": component_deficit,
        "merge_deficit": merge_deficit,
        "gap_excess": gap_excess,
        "recall_buffer_vs_araa": recall_buffer,
        "dice_buffer_vs_araa": dice_buffer,
        "boundary_buffer_vs_araa": boundary_buffer,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["failure_type"])] = counts.get(str(row["failure_type"]), 0) + 1
    top = sorted(rows, key=lambda row: float(row["priority_score"]), reverse=True)[:10]
    return {
        "num_images": len(rows),
        "failure_type_counts": counts,
        "top_images": [
            {
                "image": row["image"],
                "priority_score": row["priority_score"],
                "failure_type": row["failure_type"],
                "r110_dice": row["r110_dice"],
                "r110_gap_fp": row["r110_gap_fp"],
                "r110_component_merge": row["r110_component_merge"],
                "r110_component_mae": row["r110_component_mae"],
            }
            for row in top
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir
    r110 = by_image(load_metrics(metrics_dir / "r110_r100_r108_patch_basic_unified_metrics.json"))
    araa = by_image(load_metrics(metrics_dir / "araa_danet_epoch98_unified_metrics.json"))
    nn_path = metrics_dir / "r202_nnunet2d_unified_metrics.json"
    nnunet = by_image(load_metrics(nn_path)) if nn_path.exists() else {}

    images = sorted(set(r110) & set(araa))
    rows = [manifest_row(image, r110[image], araa[image], nnunet.get(image)) for image in images]
    rows.sort(key=lambda row: float(row["priority_score"]), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "r203_failure_manifest.csv", rows)
    (args.output_dir / "r203_failure_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    summary = {
        "run_id": "R203",
        "purpose": "Select train/val-safe directions for reducing R110 bridge/gap failures without over-erosion.",
        "inputs": {
            "r110": str(metrics_dir / "r110_r100_r108_patch_basic_unified_metrics.json"),
            "araa": str(metrics_dir / "araa_danet_epoch98_unified_metrics.json"),
            "nnunet_risk_audit": str(nn_path) if nn_path.exists() else None,
        },
        "selection_note": "clean-test-v2 diagnostic only; do not tune thresholds from these rows.",
        "summary": summarize(rows),
        "recommended_next": [
            "Use train/val to learn or select a conservative bridge/gap candidate detector.",
            "Reject edits that reduce recall or increase component-count MAE on validation.",
            "Only after locking validation rules, apply once to clean-test-v2 and evaluate with R201.",
        ],
    }
    (args.output_dir / "r203_failure_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
