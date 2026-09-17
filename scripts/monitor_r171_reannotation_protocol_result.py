#!/usr/bin/env python3
"""Summarize and gate the R171 reannotation-protocol experiment result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = Path("outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_clean_test_v2_metrics.json")
DEFAULT_CONTROL = Path("outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_original_test_metrics.json")
DEFAULT_HISTORY = Path("outputs/timm_instance_sep/r171_reannotation_protocol_dinov3_instance_sep/history.json")
DEFAULT_SUMMARY = Path("outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_result_summary.json")
TARGET_DICE = 0.9317660066557425
BEST_VALID_DICE = 0.9177231563529792


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def history_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "epochs": 0, "best_val_dice": None, "best_epoch": None}
    history = load_json(path)
    if not isinstance(history, list):
        return {"exists": True, "epochs": 0, "best_val_dice": None, "best_epoch": None}
    best_val = None
    best_epoch = None
    for row in history:
        if not isinstance(row, dict) or row.get("val_dice") is None:
            continue
        val = float(row["val_dice"])
        if best_val is None or val > best_val:
            best_val = val
            best_epoch = row.get("epoch")
    return {"exists": True, "epochs": len(history), "best_val_dice": best_val, "best_epoch": best_epoch}


def waiting_summary(args: argparse.Namespace) -> dict[str, Any]:
    hist = history_summary(args.history_json)
    return {
        "status": "waiting_for_metrics",
        "metrics_json": str(args.metrics_json),
        "control_metrics_json": str(args.control_metrics_json),
        "history_json": str(args.history_json),
        "history": hist,
        "target_dice": args.target_dice,
        "best_valid_dice": args.best_valid_dice,
        "next_action": "wait_for_r171_final_clean_test_v2_metrics",
    }


def result_summary(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_json(args.metrics_json)
    control = load_json(args.control_metrics_json) if args.control_metrics_json.exists() else None
    dice = metric(payload, "dice")
    if dice is None:
        raise ValueError(f"missing mean.dice in {args.metrics_json}")
    precision = metric(payload, "precision")
    recall = metric(payload, "recall")
    boundary_iou = metric(payload, "boundary_iou")
    delta_vs_best = dice - args.best_valid_dice
    gap_to_target = args.target_dice - dice
    target_met = dice > args.target_dice
    new_best = dice > args.best_valid_dice
    if target_met:
        status = "target_met"
        next_action = "run_result_to_claim_and_experiment_audit"
    elif new_best:
        status = "new_best_below_target"
        next_action = "analyze_r171_failure_modes_and_decide_next_branch"
    else:
        status = "below_best"
        next_action = "do_not_continue_r171_without_new_evidence"
    return {
        "status": status,
        "metrics_json": str(args.metrics_json),
        "control_metrics_json": str(args.control_metrics_json) if args.control_metrics_json.exists() else None,
        "history_json": str(args.history_json),
        "history": history_summary(args.history_json),
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
        "success_eval_policy": "Only TSRS_RSNA-Epiphysis_clean_test_v2/test decides target success.",
        "next_action": next_action,
    }


def main() -> None:
    args = parse_args()
    summary = result_summary(args) if args.metrics_json.exists() else waiting_summary(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
