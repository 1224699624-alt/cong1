#!/usr/bin/env python3
"""Audit whether a validation run is strong enough to promote.

This is a report-only ARIS gate for R212/R213 style acceptance-gate runs.
It reads an existing summary JSON plus optional threshold-audit CSV and does
not touch masks, datasets, or clean-test-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_RISK_IMAGES = [
    "1620.png",
    "3058.png",
    "3468.png",
    "6605.png",
    "7259.png",
    "8253.png",
    "13480.png",
    "13867.png",
    "15040.png",
]

HIGHER_BETTER = [
    "delta_dice",
    "delta_iou",
    "delta_boundary_iou",
    "delta_boundary_f1",
    "delta_surface_dice_2px",
    "delta_surface_dice_5px",
]

LOWER_BETTER = [
    "delta_hd95_px",
    "delta_assd_px",
    "delta_gap_region_fp_rate",
    "delta_component_merge_rate",
    "delta_component_count_mae",
    "delta_instance_merge_count",
    "delta_pred_component_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R212/R213 validation-gate promotion readiness.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--threshold-audit-csv", type=Path, default=Path(""))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=Path(""))
    parser.add_argument("--risk-images", default=",".join(DEFAULT_RISK_IMAGES))
    parser.add_argument("--min-accepted-rate", type=float, default=0.03)
    parser.add_argument("--min-boundary-iou-delta", type=float, default=1e-4)
    parser.add_argument("--min-boundary-f1-delta", type=float, default=1e-4)
    parser.add_argument("--min-surface2-delta", type=float, default=1e-4)
    parser.add_argument("--max-dice-drop", type=float, default=5e-4)
    parser.add_argument("--max-iou-drop", type=float, default=8e-4)
    parser.add_argument("--max-recall-drop", type=float, default=8e-4)
    parser.add_argument("--require-gap-improvement", action="store_true")
    parser.add_argument("--require-component-nonworse", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not str(path) or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def best_mean(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best") or {}
    mean = best.get("mean") or {}
    if not isinstance(mean, dict):
        raise ValueError("summary JSON does not contain best.mean")
    return mean


def threshold_value(summary: dict[str, Any]) -> float | None:
    best = summary.get("best") or {}
    if "threshold" not in best:
        return None
    return float(best["threshold"])


def accepted_images_for_threshold(rows: list[dict[str, Any]], threshold: float | None) -> list[str]:
    if threshold is None:
        return []
    out = []
    for row in rows:
        if abs(as_float(row.get("threshold")) - threshold) < 1e-9 and as_float(row.get("accepted")) > 0.0:
            out.append(str(row.get("image")))
    return out


def audit(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.summary_json)
    rows = read_csv(args.threshold_audit_csv)
    mean = best_mean(summary)
    threshold = threshold_value(summary)
    risk_images = {item.strip() for item in args.risk_images.split(",") if item.strip()}
    accepted_images = accepted_images_for_threshold(rows, threshold)
    risk_hits = sorted(set(accepted_images) & risk_images)

    accepted_rate = as_float(mean.get("accepted"))
    checks = {
        "summary_gate_pass": bool((summary.get("best") or {}).get("gate_pass")),
        "accepted_rate_meaningful": accepted_rate >= args.min_accepted_rate,
        "dice_nonworse": as_float(mean.get("delta_dice")) >= -args.max_dice_drop,
        "iou_nonworse": as_float(mean.get("delta_iou")) >= -args.max_iou_drop,
        "recall_nonworse": as_float(mean.get("delta_recall")) >= -args.max_recall_drop,
        "boundary_iou_meaningful": as_float(mean.get("delta_boundary_iou")) >= args.min_boundary_iou_delta,
        "boundary_f1_meaningful": as_float(mean.get("delta_boundary_f1")) >= args.min_boundary_f1_delta,
        "surface2_meaningful": as_float(mean.get("delta_surface_dice_2px")) >= args.min_surface2_delta,
        "no_risk_image_accepted": not risk_hits,
    }
    if args.require_gap_improvement:
        checks["gap_fp_improves"] = as_float(mean.get("delta_gap_region_fp_rate")) < 0.0
    if args.require_component_nonworse:
        checks["component_merge_nonworse"] = as_float(mean.get("delta_component_merge_rate")) <= 0.0
        checks["component_count_nonworse"] = as_float(mean.get("delta_component_count_mae")) <= 0.0
        checks["pred_component_count_nonworse"] = as_float(mean.get("delta_pred_component_count")) <= 0.0

    promote_ready = all(checks.values())
    metrics = {key: mean.get(key) for key in [*HIGHER_BETTER, "delta_recall", *LOWER_BETTER, "accepted"]}
    return {
        "run_id": "R213-validation-gate-audit",
        "summary_json": str(args.summary_json),
        "threshold_audit_csv": str(args.threshold_audit_csv) if str(args.threshold_audit_csv) else None,
        "warning": "validation gate only; not clean-test-v2 evidence",
        "threshold": threshold,
        "num_evaluated": summary.get("num_evaluated"),
        "accepted_images": accepted_images,
        "risk_images": sorted(risk_images),
        "risk_hits": risk_hits,
        "metrics": metrics,
        "criteria": {
            "min_accepted_rate": args.min_accepted_rate,
            "min_boundary_iou_delta": args.min_boundary_iou_delta,
            "min_boundary_f1_delta": args.min_boundary_f1_delta,
            "min_surface2_delta": args.min_surface2_delta,
            "max_dice_drop": args.max_dice_drop,
            "max_iou_drop": args.max_iou_drop,
            "max_recall_drop": args.max_recall_drop,
            "require_gap_improvement": bool(args.require_gap_improvement),
            "require_component_nonworse": bool(args.require_component_nonworse),
        },
        "checks": checks,
        "promote_ready": promote_ready,
        "decision": "promote_to_visual_audit_only" if promote_ready else "do_not_promote",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# R213 Validation Gate Audit",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "This is validation-only evidence. It is not clean-test-v2 evidence.",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in report["metrics"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Checks", "", "| Check | Pass |", "| --- | ---: |"])
    for key, value in report["checks"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Accepted Images", ""])
    if report["accepted_images"]:
        lines.extend(f"- `{item}`" for item in report["accepted_images"])
    else:
        lines.append("- none")
    lines.extend(["", "## Risk Hits", ""])
    if report["risk_hits"]:
        lines.extend(f"- `{item}`" for item in report["risk_hits"])
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.output_md != Path("."):
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
