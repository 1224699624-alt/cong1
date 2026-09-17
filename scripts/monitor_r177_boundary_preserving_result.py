#!/usr/bin/env python3
"""Monitor and judge R177 boundary-preserving separation result."""

from __future__ import annotations

import json
from pathlib import Path


TARGET = 0.9317660066557425
R110_DICE = 0.9177231563529792
R110_BOUNDARY_IOU = 0.25189601044085763
R110_FALSE_BRIDGE = 0.790123
R110_COMPONENT_ERROR = 2.506173


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    metrics_path = Path("outputs/analysis/r177_boundary_preserving_separation_arbitrator_clean_test_v2_metrics.json")
    val_path = Path("outputs/analysis/r177_boundary_preserving_separation_arbitrator_val_summary.json")
    out_path = Path("outputs/analysis/r177_boundary_preserving_separation_arbitrator_result_summary.json")
    metrics = load_json(metrics_path)
    val = load_json(val_path)
    if metrics is None:
        summary = {
            "status": "waiting_for_metrics",
            "metrics_path": str(metrics_path),
            "val_summary_exists": val is not None,
        }
    else:
        mean = metrics.get("mean", {})
        dice = float(mean.get("dice", 0.0))
        boundary_iou = float(mean.get("boundary_iou", 0.0))
        false_bridge = float(mean.get("false_bridge_flag", 1.0))
        component_error = float(mean.get("component_count_error", 999.0))
        dice_non_drop = dice >= R110_DICE
        boundary_non_drop = boundary_iou >= R110_BOUNDARY_IOU
        structure_improved = false_bridge < R110_FALSE_BRIDGE or component_error < R110_COMPONENT_ERROR
        if dice >= TARGET:
            status = "target_met"
        elif dice_non_drop and boundary_non_drop and structure_improved:
            status = "constraint_win_below_target"
        elif dice > R110_DICE:
            status = "new_best_dice_below_target"
        elif structure_improved and dice >= R110_DICE - 0.001:
            status = "structure_tradeoff_near_best"
        else:
            status = "below_best"
        summary = {
            "status": status,
            "metrics_path": str(metrics_path),
            "val_summary_path": str(val_path),
            "num_evaluated": metrics.get("num_evaluated"),
            "mean": mean,
            "comparisons": {
                "target_dice": TARGET,
                "target_margin": dice - TARGET,
                "r110_dice": R110_DICE,
                "dice_delta_vs_r110": dice - R110_DICE,
                "r110_boundary_iou": R110_BOUNDARY_IOU,
                "boundary_iou_delta_vs_r110": boundary_iou - R110_BOUNDARY_IOU,
                "r110_false_bridge": R110_FALSE_BRIDGE,
                "false_bridge_delta_vs_r110": false_bridge - R110_FALSE_BRIDGE,
                "r110_component_error": R110_COMPONENT_ERROR,
                "component_error_delta_vs_r110": component_error - R110_COMPONENT_ERROR,
            },
            "val_best": None if val is None else val.get("best"),
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
