#!/usr/bin/env python3
"""Re-evaluate R248 out-of-fold predictions on a reachable threshold grid.

The original R248 full-val run searched risk thresholds only up to 0.50, while
all out-of-fold risk probabilities were above 0.70. This read-only regrid uses
the already generated original-val OOF predictions; it does not retrain, write
masks, or read clean-test-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from train_r248_r244_context_crop_scorer import choose_best, evaluate_grid, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regrid R248 original-val OOF predictions.")
    parser.add_argument(
        "--probed-csv",
        type=Path,
        default=Path("outputs/analysis/r248_r244_context_crop_scorer_fullval_probed_rows.csv"),
    )
    parser.add_argument("--useful-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.84")
    parser.add_argument("--risk-thresholds", default="0.70,0.72,0.74,0.76,0.78,0.80,0.85,0.90,0.95,0.99")
    parser.add_argument("--max-actions-per-image", type=int, default=1)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/analysis/r248_r244_context_crop_scorer_fullval_regrid.json"),
    )
    parser.add_argument(
        "--grid-csv",
        type=Path,
        default=Path("outputs/analysis/r248_r244_context_crop_scorer_fullval_regrid.csv"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def main() -> None:
    args = parse_args()
    rows = read_rows(args.probed_csv)
    eval_args = SimpleNamespace(
        useful_thresholds=args.useful_thresholds,
        risk_thresholds=args.risk_thresholds,
        max_actions_per_image=args.max_actions_per_image,
    )
    grid = evaluate_grid(rows, eval_args)
    best = choose_best(grid)
    useful_probs = finite_values(rows, "r248_useful_prob")
    risk_probs = finite_values(rows, "r248_risk_prob")
    report = {
        "run_id": "R248b-r244-context-crop-scorer-reachable-regrid",
        "source_probed_csv": str(args.probed_csv),
        "dataset": "TSRS_RSNA-Epiphysis",
        "split": "val",
        "clean_test_v2_used": False,
        "writes_masks": False,
        "num_rows": len(rows),
        "num_images": len({str(row.get("image")) for row in rows}),
        "useful_thresholds": args.useful_thresholds,
        "risk_thresholds": args.risk_thresholds,
        "useful_prob_min": min(useful_probs) if useful_probs else None,
        "useful_prob_max": max(useful_probs) if useful_probs else None,
        "risk_prob_min": min(risk_probs) if risk_probs else None,
        "risk_prob_max": max(risk_probs) if risk_probs else None,
        "num_configs": len(grid),
        "num_nonempty_configs": sum(int(row.get("selected") or 0) > 0 for row in grid),
        "num_passing_configs": sum(bool(row.get("passes_gate")) for row in grid),
        "best": best,
        "decision": "passing_config_found" if best and best.get("passes_gate") else "no_go_context_scorer_not_clean_enough",
        "warning": "Original-val OOF threshold diagnosis only; no masks written and clean-test-v2 was not read.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.grid_csv, grid)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
