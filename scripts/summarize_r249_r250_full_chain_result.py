#!/usr/bin/env python3
"""Summarize the R244 -> R249 -> R250 reversed-topology chain.

This is a lightweight ARIS result parser. It reads the R244/R245/R246/R248
JSON artifacts when they exist, reports the current gate status, and writes a
Markdown decision note. It never reads clean-test-v2, never writes masks, and
does not start or stop remote jobs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize R244/R249/R250 full-chain state.")
    parser.add_argument(
        "--state-json",
        type=Path,
        default=Path("outputs/analysis/r244_r249_remote_state_latest.json"),
    )
    parser.add_argument(
        "--r244-json",
        type=Path,
        default=Path("outputs/analysis/r244_gtfree_merge_targeted_fullval_summary.json"),
    )
    parser.add_argument(
        "--r245-json",
        type=Path,
        default=Path("outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json"),
    )
    parser.add_argument(
        "--r246-json",
        type=Path,
        default=Path("outputs/analysis/r246_gtfree_merge_feature_failure.fullval.json"),
    )
    parser.add_argument(
        "--r248-json",
        type=Path,
        default=Path("outputs/analysis/r248_r244_context_crop_scorer_fullval.json"),
    )
    parser.add_argument(
        "--r248-regrid-json",
        type=Path,
        default=Path("outputs/analysis/r248_r244_context_crop_scorer_fullval_regrid.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("research-workflow/refine-logs/R249_R250_FULL_CHAIN_STATUS.md"),
    )
    parser.add_argument("--min-safe-useful", type=int, default=5)
    parser.add_argument("--max-risk", type=int, default=2)
    parser.add_argument("--min-boundary-iou-delta", type=float, default=1e-4)
    parser.add_argument("--max-recall-drop", type=float, default=1e-8)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, "", "None", "nan"):
        return default
    return int(float(value))


def fmt(value: Any) -> str:
    if value in (None, "", "None"):
        return "NA"
    if isinstance(value, bool):
        return str(value).lower()
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return str(value)


def r245_gate(best: dict[str, Any] | None, args: argparse.Namespace) -> tuple[bool, list[str], list[str]]:
    if not best:
        return False, [], ["missing R245 best config"]
    checks = [
        ("selected", as_int(best.get("num_rows")) >= args.min_safe_useful, f"num_rows={as_int(best.get('num_rows'))}"),
        ("safe_useful", as_int(best.get("num_safe_useful")) >= args.min_safe_useful, f"safe={as_int(best.get('num_safe_useful'))} >= {args.min_safe_useful}"),
        ("risk", as_int(best.get("num_risk")) <= args.max_risk, f"risk={as_int(best.get('num_risk'))} <= {args.max_risk}"),
        ("recall_nonworse", as_float(best.get("mean_delta_recall")) >= -args.max_recall_drop, f"recall_delta={fmt(best.get('mean_delta_recall'))}"),
        (
            "boundary_iou_gain",
            as_float(best.get("mean_delta_boundary_iou")) >= args.min_boundary_iou_delta,
            f"boundary_iou_delta={fmt(best.get('mean_delta_boundary_iou'))} >= {args.min_boundary_iou_delta}",
        ),
        ("boundary_f1_gain", as_float(best.get("mean_delta_boundary_f1")) > 0.0, f"boundary_f1_delta={fmt(best.get('mean_delta_boundary_f1'))}"),
        ("gap_fp_gain", as_float(best.get("mean_delta_gap_region_fp_rate")) < 0.0, f"gap_fp_delta={fmt(best.get('mean_delta_gap_region_fp_rate'))}"),
        ("low_foreground_cut", as_float(best.get("mean_cut_gt_fg_frac"), 1.0) <= 0.25, f"cut_gt_fg_frac={fmt(best.get('mean_cut_gt_fg_frac'))}"),
    ]
    passed: list[str] = []
    failed: list[str] = []
    for name, ok, detail in checks:
        line = f"{name}: {detail}"
        if ok:
            passed.append(line)
        else:
            failed.append(line)
    return not failed, passed, failed


def r248_best(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not data:
        return None
    best = nested(data, "grouped_cv", "best")
    return best if isinstance(best, dict) else None


def first_value(data: dict[str, Any] | None, paths: list[tuple[str, ...]]) -> Any:
    for path in paths:
        value = nested(data, *path)
        if value is not None:
            return value
    return None


def infer_decision(
    state: dict[str, Any] | None,
    r245: dict[str, Any] | None,
    r248: dict[str, Any] | None,
    r248_regrid: dict[str, Any] | None,
    r245_passed: bool,
) -> str:
    status = nested(state, "parsed_status", default={}) or {}
    if status.get("r244_running"):
        return "waiting_for_r244_fullval"
    if status.get("r249_waiting") and r245 is None:
        return "waiting_for_r249_full_audit"
    if r245 is None:
        return "missing_r245_full_audit"
    if r245_passed:
        return "r245_candidate_gate_passed_prepare_isolated_val_mask_editor"
    if status.get("r248_running"):
        return "waiting_for_r248_fullval"
    if r248 is None:
        return "r245_no_go_waiting_or_missing_r248"
    regrid_best = nested(r248_regrid, "best") if r248_regrid else None
    if isinstance(regrid_best, dict) and regrid_best.get("passes_gate"):
        return "r248_regrid_candidate_gate_passed_review_before_mask_writing"
    if r248_regrid is not None:
        return "r248_no_go_context_scorer_not_clean_enough"
    best = r248_best(r248)
    return "r248_has_candidate_result_review_gate" if best else "r248_original_grid_unreachable_regrid_required"


def lines_for_group(name: str, group: dict[str, Any] | None) -> list[str]:
    if not group:
        return [f"- {name}: `missing`"]
    keys = [
        "num_rows",
        "num_images",
        "num_safe_useful",
        "num_risk",
        "mean_delta_recall",
        "mean_delta_boundary_iou",
        "mean_delta_boundary_f1",
        "mean_delta_gap_region_fp_rate",
        "mean_delta_component_count_mae",
        "mean_cut_gt_fg_frac",
        "mean_cut_gt_gap_frac",
    ]
    out = [f"- {name}:"]
    out.extend([f"  - {key}: `{fmt(group.get(key))}`" for key in keys if key in group])
    return out


def main() -> None:
    args = parse_args()
    state = read_json(args.state_json)
    r244 = read_json(args.r244_json)
    if r244 is None:
        state_r244 = nested(state, "json", "r244")
        r244 = state_r244 if isinstance(state_r244, dict) else None
    r245 = read_json(args.r245_json)
    r246 = read_json(args.r246_json)
    r248 = read_json(args.r248_json)
    r248_regrid = read_json(args.r248_regrid_json)

    r245_best = nested(r245, "best") if r245 else None
    r245_passed, r245_passed_checks, r245_failed_checks = r245_gate(r245_best, args)
    decision = infer_decision(state, r245, r248, r248_regrid, r245_passed)
    status = nested(state, "parsed_status", default={}) or {}

    lines = [
        "# R249/R250 Full Chain Status",
        "",
        "## ARIS Status",
        f"- Decision: `{decision}`",
        "- Clean-test-v2 used: `false`",
        "- Writes masks: `false`",
        "- Scope: `TSRS_RSNA-Epiphysis original-val diagnostics only`",
        "",
        "## Remote/Artifact Status",
        f"- R244 running: `{fmt(status.get('r244_running'))}`",
        f"- R249 waiting: `{fmt(status.get('r249_waiting'))}`",
        f"- R250 waiting: `{fmt(status.get('r250_waiting'))}`",
        f"- R248 running: `{fmt(status.get('r248_running'))}`",
        f"- R244 log progress: `{nested(status, 'r244_log_progress', 'current')}/{nested(status, 'r244_log_progress', 'total')}`",
        f"- R249 seen summary: `{nested(status, 'r249_seen_summary', 'images')} images / {nested(status, 'r249_seen_summary', 'rows')} rows`",
        f"- Full R245 exists: `{fmt(status.get('full_r245_exists'))}`",
        f"- Full R246 exists: `{fmt(status.get('full_r246_exists'))}`",
        f"- Full R248 exists: `{fmt(status.get('full_r248_exists'))}`",
        "",
        "## R244 Oracle/GT-Free Snapshot",
        f"- candidate images: `{fmt(first_value(r244, [('num_candidate_images',)]))}`",
        f"- candidate rows: `{fmt(first_value(r244, [('num_candidate_rows',)]))}`",
        f"- oracle images: `{fmt(first_value(r244, [('oracle_safe_best_per_image', 'num_images'), ('oracle_images',)]))}`",
        f"- oracle Boundary IoU delta: `{fmt(first_value(r244, [('oracle_safe_best_per_image', 'mean_delta_boundary_iou'), ('oracle_boundary_iou',)]))}`",
        f"- oracle gap FP delta: `{fmt(first_value(r244, [('oracle_safe_best_per_image', 'mean_delta_gap_region_fp_rate'), ('oracle_gap_fp',)]))}`",
        f"- oracle component count MAE delta: `{fmt(first_value(r244, [('oracle_safe_best_per_image', 'mean_delta_component_count_mae'), ('oracle_component_count_mae',)]))}`",
        f"- GT-free clean safe/risk: `{fmt(first_value(r244, [('gtfree_clean_subset', 'num_safe_useful'), ('clean_safe',)]))}` / `{fmt(first_value(r244, [('gtfree_clean_subset', 'num_risk'), ('clean_risk',)]))}`",
        "",
        "## R245 Gate",
        f"- R245 decision field: `{fmt(nested(r245, 'decision'))}`",
        f"- R245 passing configs: `{fmt(nested(r245, 'num_passing_configs'))}`",
    ]
    if r245_best:
        keys = [
            "num_rows",
            "num_images",
            "num_safe_useful",
            "num_risk",
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
        lines.extend([f"- best {key}: `{fmt(r245_best.get(key))}`" for key in keys])
    else:
        lines.append("- best: `missing`")
    lines.extend(["", "Passed checks:"])
    lines.extend([f"- {line}" for line in r245_passed_checks] or ["- none"])
    lines.append("")
    lines.append("Failed checks:")
    lines.extend([f"- {line}" for line in r245_failed_checks] or ["- none"])

    lines.extend(["", "## R246 Feature-Failure Groups"])
    groups = nested(r246, "groups", default={}) or {}
    for group_name in ["oracle_safe_best", "r245_best_selected", "safe_useful", "risk"]:
        lines.extend(lines_for_group(group_name, groups.get(group_name)))

    lines.extend(["", "## R248 Context-Crop Scorer"])
    best248 = r248_best(r248)
    if best248:
        for key in [
            "selected",
            "selected_images",
            "safe_useful",
            "risk",
            "mean_delta_recall",
            "mean_delta_boundary_iou",
            "mean_delta_boundary_f1",
            "mean_delta_gap_region_fp_rate",
            "mean_delta_component_count_mae",
        ]:
            lines.append(f"- best {key}: `{fmt(best248.get(key))}`")
    else:
        lines.append("- best: `missing or null`")
    lines.extend(
        [
            f"- original grid passing configs: `{fmt(nested(r248, 'grouped_cv', 'num_passing_configs'))}`",
            f"- OOF risk probability range: `{fmt(nested(r248_regrid, 'risk_prob_min'))}` to `{fmt(nested(r248_regrid, 'risk_prob_max'))}`",
            f"- reachable regrid nonempty configs: `{fmt(nested(r248_regrid, 'num_nonempty_configs'))}`",
            f"- reachable regrid passing configs: `{fmt(nested(r248_regrid, 'num_passing_configs'))}`",
            f"- reachable regrid decision: `{fmt(nested(r248_regrid, 'decision'))}`",
        ]
    )
    regrid_best = nested(r248_regrid, "best")
    if isinstance(regrid_best, dict):
        for key in [
            "useful_threshold",
            "risk_threshold",
            "selected",
            "selected_images",
            "safe_useful",
            "risk",
            "mean_delta_recall",
            "mean_delta_boundary_iou",
            "mean_delta_boundary_f1",
            "mean_delta_gap_region_fp_rate",
            "mean_delta_component_count_mae",
            "mean_cut_gt_fg_frac",
        ]:
            lines.append(f"- regrid best {key}: `{fmt(regrid_best.get(key))}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "- If decision is `waiting_for_r244_fullval`, do not launch additional CPU-heavy jobs.",
            "- If R245 passes, next step is an isolated original-val mask editor and R201-style audit, not clean-test-v2.",
            "- If R245 remains no-go and R248 is missing, wait for R250 or launch R248 manually only after confirming R250 failed.",
            "- The original R248 risk-threshold grid ended at `0.50`, below every OOF risk probability; `best=null` alone is not a valid model conclusion.",
            "- The reachable R248b regrid is the corrected gate. If it remains no-go, move to explicit seam/background supervision or reduce generator risk before any mask writing.",
            "- clean-test-v2 remains locked for final evaluation/diagnosis only.",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision": decision, "output_md": str(args.output_md)}, indent=2))


if __name__ == "__main__":
    main()
