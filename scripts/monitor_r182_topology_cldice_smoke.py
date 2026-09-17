#!/usr/bin/env python3
"""Summarize the R182 topology/clDice smoke gate."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    run_id = "r182_topology_cldice_smoke"
    history_path = Path("outputs/timm_instance_sep") / run_id / "history.json"
    metrics_path = Path("outputs/analysis") / f"{run_id}_clean_test_v2_metrics.json"
    out_path = Path("outputs/analysis") / f"{run_id}_gate_summary.json"

    history = load_json(history_path)
    metrics = load_json(metrics_path)
    status = "waiting_for_metrics"
    reason = "history_or_metrics_missing"
    best_val: dict[str, object] | None = None
    clean_mean: dict[str, object] | None = None

    if isinstance(history, list) and history:
        best_val = max(history, key=lambda row: float(row.get("val_dice", -1.0)))
    if isinstance(metrics, dict):
        clean_mean = metrics.get("mean") if isinstance(metrics.get("mean"), dict) else None

    if best_val and clean_mean:
        val_dice = float(best_val.get("val_dice", 0.0))
        val_boundary = float(best_val.get("val_boundary_iou", 0.0))
        val_precision = float(best_val.get("val_precision", 0.0))
        val_recall = float(best_val.get("val_recall", 0.0))
        clean_dice = float(clean_mean.get("dice", 0.0))
        clean_precision = float(clean_mean.get("precision", 0.0))
        clean_recall = float(clean_mean.get("recall", 0.0))

        if val_dice < 0.55:
            status = "gate_failed"
            reason = "val_dice_too_low"
        elif val_boundary < 0.12:
            status = "gate_failed"
            reason = "val_boundary_too_low"
        elif val_precision < 0.35 or val_recall < 0.35:
            status = "gate_failed"
            reason = "val_empty_or_all_foreground_risk"
        elif clean_dice < 0.40 or clean_precision < 0.25 or clean_recall < 0.25:
            status = "gate_failed"
            reason = "clean_diagnostic_collapse"
        else:
            status = "gate_pass"
            reason = "smoke_noncollapsed"

    summary = {
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "history_path": str(history_path),
        "metrics_path": str(metrics_path),
        "best_val": best_val,
        "clean_test_v2_diagnostic_mean": clean_mean,
        "note": "Smoke gate only. clean-test-v2 is diagnostic here and not used for tuning or success claim.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
