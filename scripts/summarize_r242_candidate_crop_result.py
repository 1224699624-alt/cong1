#!/usr/bin/env python3
"""Summarize R242 candidate crop scorer outputs.

This is a lightweight ARIS result parser. It reads the R242 grouped-CV JSON and
threshold grid, reports the best low-risk threshold row, and writes a Markdown
decision note. It does not modify masks or datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize R242 candidate crop scorer result.")
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r242_candidate_crop_scorer_fullval_groupedcv5.json"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r242_candidate_crop_scorer_fullval_threshold_grid.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("research-workflow/refine-logs/R242_CANDIDATE_CROP_SCORER_RESULT.md"))
    parser.add_argument("--max-risk", type=int, default=2)
    parser.add_argument("--min-safe-useful", type=int, default=5)
    parser.add_argument("--min-boundary-iou-delta", type=float, default=1e-4)
    parser.add_argument("--max-recall-drop", type=float, default=1e-8)
    return parser.parse_args()


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return int(float(value))


def read_grid(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def choose_low_risk_best(rows: list[dict[str, Any]], max_risk: int) -> dict[str, Any] | None:
    nonempty = [row for row in rows if as_int(row, "selected") > 0]
    low_risk = [row for row in nonempty if as_int(row, "risk") <= max_risk]
    pool = low_risk if low_risk else nonempty
    if not pool:
        return None
    return sorted(
        pool,
        key=lambda row: (
            -as_int(row, "risk"),
            as_int(row, "safe_useful"),
            as_float(row, "mean_delta_boundary_iou", -999.0),
            -as_float(row, "mean_delta_gap_region_fp_rate", 999.0),
            as_int(row, "selected"),
        ),
        reverse=True,
    )[0]


def gate(best: dict[str, Any] | None, args: argparse.Namespace) -> tuple[bool, list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    if best is None:
        return False, [], ["no non-empty threshold row"]
    checks = [
        ("risk<=max_risk", as_int(best, "risk") <= args.max_risk, f"risk={as_int(best, 'risk')} <= {args.max_risk}"),
        ("safe_useful", as_int(best, "safe_useful") >= args.min_safe_useful, f"safe_useful={as_int(best, 'safe_useful')} >= {args.min_safe_useful}"),
        ("recall_nonworse", as_float(best, "mean_delta_recall") >= -args.max_recall_drop, f"mean_delta_recall={as_float(best, 'mean_delta_recall'):.9f}"),
        (
            "boundary_iou_gain",
            as_float(best, "mean_delta_boundary_iou") >= args.min_boundary_iou_delta,
            f"mean_delta_boundary_iou={as_float(best, 'mean_delta_boundary_iou'):.9f} >= {args.min_boundary_iou_delta}",
        ),
        ("boundary_f1_gain", as_float(best, "mean_delta_boundary_f1") > 0.0, f"mean_delta_boundary_f1={as_float(best, 'mean_delta_boundary_f1'):.9f}"),
        ("gap_fp_gain", as_float(best, "mean_delta_gap_region_fp_rate") < 0.0, f"mean_delta_gap_region_fp_rate={as_float(best, 'mean_delta_gap_region_fp_rate'):.9f}"),
        ("component_count_nonworse", as_float(best, "mean_delta_component_count_mae") <= 0.0, f"mean_delta_component_count_mae={as_float(best, 'mean_delta_component_count_mae'):.9f}"),
        ("low_foreground_cut", as_float(best, "mean_cut_gt_fg_frac") <= 0.25, f"mean_cut_gt_fg_frac={as_float(best, 'mean_cut_gt_fg_frac'):.6f}"),
        ("high_gap_cut", as_float(best, "mean_cut_gt_gap_frac") >= 0.75, f"mean_cut_gt_gap_frac={as_float(best, 'mean_cut_gt_gap_frac'):.6f}"),
    ]
    for name, ok, detail in checks:
        line = f"{name}: {detail}"
        if ok:
            passed.append(line)
        else:
            failed.append(line)
    return not failed, passed, failed


def fmt(value: Any) -> str:
    if value in (None, "", "None"):
        return "NA"
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    grid = read_grid(args.grid_csv)
    best = choose_low_risk_best(grid, args.max_risk)
    passed, passed_checks, failed_checks = gate(best, args)
    decision = "go_original_val_mask_editor_next" if passed else "no_go_candidate_crop_scorer_not_promoted"
    lines = [
        "# R242 Candidate Crop Scorer Result",
        "",
        "## ARIS Status",
        f"- Stage: {summary.get('evidence_level', 'unknown')}",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Split: `{summary.get('split')}`",
        f"- Clean-test-v2 used: `{summary.get('clean_test_v2_used')}`",
        f"- Writes masks: `{summary.get('writes_masks')}`",
        f"- Decision: `{decision}`",
        "",
        "## Data",
        f"- candidate rows: `{summary.get('num_candidate_rows')}`",
        f"- candidate images: `{summary.get('num_candidate_images')}`",
        f"- crops: `{summary.get('num_crops')}`",
        f"- safe useful labels: `{summary.get('num_safe_useful')}`",
        f"- risk labels: `{summary.get('num_risk')}`",
        "",
        "## Best Low-Risk Threshold Row",
    ]
    if best is None:
        lines.append("- no non-empty threshold row")
    else:
        keys = [
            "useful_threshold",
            "risk_threshold",
            "selected",
            "selected_images",
            "safe_useful",
            "risk",
            "mean_delta_dice",
            "mean_delta_iou",
            "mean_delta_recall",
            "mean_delta_boundary_iou",
            "mean_delta_boundary_f1",
            "mean_delta_gap_region_fp_rate",
            "mean_delta_component_count_mae",
            "mean_cut_gt_fg_frac",
            "mean_cut_gt_gap_frac",
        ]
        for key in keys:
            lines.append(f"- {key}: `{fmt(best.get(key))}`")
    lines.extend(["", "## Promotion Gate", "Passed checks:"])
    lines.extend([f"- {line}" for line in passed_checks] or ["- none"])
    lines.append("")
    lines.append("Failed checks:")
    lines.extend([f"- {line}" for line in failed_checks] or ["- none"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "R242 remains candidate-level evidence only. If it passes the gate, the next",
            "step is an isolated original-val mask editor and R201-style metric audit.",
            "If it fails, do not tune thresholds on the same noisy R239 candidate",
            "distribution; move toward cleaner component-merge-targeted candidate",
            "generation closer to the R237/R238 oracle route.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "passed": passed, "best": best, "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
