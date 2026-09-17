#!/usr/bin/env python3
"""Compare old-vs-new reannotation experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("dice", "iou", "precision", "recall", "specificity", "boundary_iou")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare reannotation quality experiment outputs.")
    parser.add_argument("--old-dataset", required=True)
    parser.add_argument("--new-dataset", required=True)
    parser.add_argument("--ablations-root", default="outputs/reannotation_quality_compare")
    parser.add_argument("--refiner-exp", default="boundary_prefgate_trimfirst_refiner")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_mean(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("mean", {})


def load_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return int(payload.get("num_evaluated", 0))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    args = parse_args()
    root = Path(args.ablations_root)
    old_path = root / args.refiner_exp / args.old_dataset / args.split / "metrics.json"
    new_path = root / args.refiner_exp / args.new_dataset / args.split / "metrics.json"
    old_metrics = load_mean(old_path)
    new_metrics = load_mean(new_path)
    old_count = load_count(old_path)
    new_count = load_count(new_path)

    rows = []
    wins = {"old": 0, "new": 0}
    for metric in METRICS:
        old_value = float(old_metrics.get(metric, 0.0))
        new_value = float(new_metrics.get(metric, 0.0))
        delta = new_value - old_value
        if delta > 0:
            wins["new"] += 1
        elif delta < 0:
            wins["old"] += 1
        rows.append((metric, old_value, new_value, delta))

    primary_delta = dict((metric, delta) for metric, _, _, delta in rows)["dice"]
    if primary_delta > 0:
        conclusion = f"New reannotation wins on test Dice by {primary_delta:.6f}."
    elif primary_delta < 0:
        conclusion = f"Old filtered annotation wins on test Dice by {-primary_delta:.6f}."
    else:
        conclusion = "Old and new annotations tie on test Dice."

    lines = [
        "# Old vs New Reannotation Quality",
        "",
        f"- Old dataset: `{args.old_dataset}` ({old_count} evaluated test masks)",
        f"- New dataset: `{args.new_dataset}` ({new_count} evaluated test masks)",
        f"- Refiner: `{args.refiner_exp}`",
        f"- Primary conclusion: {conclusion}",
        "",
        "| metric | old | new | new - old |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, old_value, new_value, delta in rows:
        lines.append(f"| {metric} | {fmt(old_value)} | {fmt(new_value)} | {delta:+.6f} |")
    lines.extend(
        [
            "",
            f"Metric wins: new={wins['new']}, old={wins['old']}.",
            "",
        ]
    )

    output = Path(args.output) if args.output else root / "old_vs_new_reannotation_quality.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
