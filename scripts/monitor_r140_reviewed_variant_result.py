#!/usr/bin/env python3
"""Summarize and gate the R140 reviewed-variant experiment result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = Path("outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_clean_test_v2_metrics.json")
DEFAULT_CONTROL = Path("outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_original_test_metrics.json")
DEFAULT_HISTORY = Path("outputs/timm_instance_sep/r140_reviewed_variant_dinov3_instance_sep/history.json")
DEFAULT_SUMMARY = Path("outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_result_summary.json")
TARGET_DICE = 0.9317660066557425
BEST_VALID_DICE = 0.9177231563529792


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor R140 reviewed-variant result against clean-test-v2 target.")
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--control-metrics-json", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--history-json", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--target-dice", type=float, default=TARGET_DICE)
    parser.add_argument("--best-valid-dice", type=float, default=BEST_VALID_DICE)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric(payload: dict[str, Any], key: str) -> float | None:
    mean = payload.get("mean", {})
    value = mean.get(key)
    return float(value) if value is not None else None


def main() -> None:
    args = parse_args()
    if not args.metrics_json.exists():
        history_epochs = 0
        best_val_dice = None
        if args.history_json.exists():
            history = load_json(args.history_json)
            history_epochs = len(history) if isinstance(history, list) else 0
            vals = [row.get("val_dice") for row in history if isinstance(row, dict) and row.get("val_dice") is not None]
            best_val_dice = max(float(v) for v in vals) if vals else None
        summary = {
            "status": "waiting_for_metrics",
            "metrics_json": str(args.metrics_json),
            "history_json": str(args.history_json),
            "history_epochs": history_epochs,
            "best_val_dice_so_far": best_val_dice,
            "target_dice": args.target_dice,
            "best_valid_dice": args.best_valid_dice,
            "next_action": "wait_for_r140_final_clean_test_v2_metrics",
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    payload = load_json(args.metrics_json)
    control = load_json(args.control_metrics_json) if args.control_metrics_json.exists() else None
    dice = metric(payload, "dice")
    precision = metric(payload, "precision")
    recall = metric(payload, "recall")
    boundary_iou = metric(payload, "boundary_iou")
    if dice is None:
        raise ValueError(f"missing mean.dice in {args.metrics_json}")
    delta_vs_best = dice - args.best_valid_dice
    gap_to_target = args.target_dice - dice
    target_met = dice > args.target_dice
    new_best = dice > args.best_valid_dice
    summary = {
        "status": "target_met" if target_met else ("new_best_below_target" if new_best else "below_best"),
        "metrics_json": str(args.metrics_json),
        "control_metrics_json": str(args.control_metrics_json) if args.control_metrics_json.exists() else None,
        "dataset": payload.get("dataset"),
        "split": payload.get("split"),
        "num_evaluated": payload.get("num_evaluated"),
        "clean_test_v2": {
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "boundary_iou": boundary_iou,
        },
        "original_test_control": control.get("mean") if isinstance(control, dict) else None,
        "target_dice": args.target_dice,
        "best_valid_dice": args.best_valid_dice,
        "delta_vs_best_valid": delta_vs_best,
        "gap_to_target": gap_to_target,
        "target_met": target_met,
        "new_best": new_best,
        "next_action": (
            "run_result_to_claim_and_experiment_audit"
            if target_met
            else ("analyze_r140_failure_modes_before_next_gpu" if new_best else "do_not_continue_without_new_evidence")
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
