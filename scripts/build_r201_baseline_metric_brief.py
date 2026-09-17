#!/usr/bin/env python3
"""Build a compact R201 baseline metric brief.

The brief is a current-state snapshot for the ARIS improvement loop. It reads
the R201 unified comparison table and writes a Markdown summary of strong
baselines, metric directions, and the current anatomy-consistency bottleneck.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


METRICS = [
    ("dice", "higher"),
    ("iou", "higher"),
    ("boundary_iou", "higher"),
    ("boundary_f1", "higher"),
    ("surface_dice_2px", "higher"),
    ("surface_dice_5px", "higher"),
    ("hd95_px", "lower"),
    ("assd_px", "lower"),
    ("gap_region_fp_rate", "lower"),
    ("component_merge_rate", "lower"),
    ("component_count_mae", "lower"),
]


PRIMARY_MODELS = [
    "araa_danet_epoch98",
    "r110_r100_r108_patch_basic",
    "r143_highres_medical_recipe_unet",
    "r202_nnunet2d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R201 baseline metric brief.")
    parser.add_argument("--table-csv", type=Path, default=Path("outputs/analysis/r201_unified_eval/r201_unified_comparison_table.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("research-workflow/refine-logs/R201_BASELINE_METRIC_BRIEF.md"))
    return parser.parse_args()


def as_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return None
    return float(value)


def fmt(value: Any) -> str:
    if value in (None, "", "None"):
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def rank_rows(rows: list[dict[str, Any]], metric: str, direction: str) -> list[dict[str, Any]]:
    available = [row for row in rows if as_float(row, metric) is not None and int(float(row.get("num_evaluated") or 0)) >= 81]
    reverse = direction == "higher"
    return sorted(available, key=lambda row: as_float(row, metric) or 0.0, reverse=reverse)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.table_csv)
    by_id = {str(row.get("model_id") or row.get("experiment_slug")): row for row in rows}
    lines = [
        "# R201 Baseline Metric Brief",
        "",
        "## Scope",
        "- Dataset: `TSRS_RSNA-Epiphysis_clean_test_v2/test`",
        "- Evidence: R201 unified comparison table",
        "- Note: `hd95_px` and `assd_px` are lower-is-better pixel-unit fields.",
        "",
        "## Primary Rows",
        "| model | evaluated | Dice | IoU | BIoU | BF1 | SD2 | SD5 | HD95px | ASSDpx | gap FP | merge | count MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in PRIMARY_MODELS:
        row = by_id.get(model)
        if not row:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    f"{row.get('num_evaluated')}/{row.get('num_missing')} missing",
                    fmt(row.get("dice")),
                    fmt(row.get("iou")),
                    fmt(row.get("boundary_iou")),
                    fmt(row.get("boundary_f1")),
                    fmt(row.get("surface_dice_2px")),
                    fmt(row.get("surface_dice_5px")),
                    fmt(row.get("hd95_px")),
                    fmt(row.get("assd_px")),
                    fmt(row.get("gap_region_fp_rate")),
                    fmt(row.get("component_merge_rate")),
                    fmt(row.get("component_count_mae")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Metric Leaders Among Complete R201 Rows"])
    for metric, direction in METRICS:
        ranked = rank_rows(rows, metric, direction)
        leader = ranked[0] if ranked else None
        if leader is None:
            lines.append(f"- `{metric}` ({direction}): no complete rows")
        else:
            lines.append(f"- `{metric}` ({direction}): `{leader.get('model_id')}` = `{fmt(leader.get(metric))}`")
    lines.extend(
        [
            "",
            "## Current Bottleneck",
            "- R110 improves Dice/IoU/Boundary IoU over ARAA, but has worse component merge and component count MAE.",
            "- nnU-Net is a risk-audit upper baseline for overlap/boundary, but has worse gap-region FP than ARAA/R110.",
            "- New R110-based variants should be judged by anatomy diagnostics first: gap FP, component merge, component count MAE, and boundary/surface metrics, while keeping Dice/IoU competitive.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.output_md)


if __name__ == "__main__":
    main()
