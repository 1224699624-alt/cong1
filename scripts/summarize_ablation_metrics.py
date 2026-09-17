#!/usr/bin/env python3
"""Summarize ablation metrics into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_EXPERIMENTS = [
    "zero_shot_box_only_conf020_pad005",
    "medsam_zero_shot_box_only_conf020_pad005",
    "zero_shot_dynamic_box_consistency_conf020",
    "zero_shot_mask_to_prompt_refine_conf020_pad005",
    "zero_shot_box_neg_ring_conf020_pad005",
    "zero_shot_box_fgbg_ring_conf020_pad005",
    "maskdec_boxonly_e20_lr5e-6_pad005_jit008",
    "maskdec_boxneg_heat_contrast_e20_lr5e-6_pad005_jit008",
    "maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008",
    "maskdec_prompt_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008_gtbox",
    "maskdec_prompt_gbc_boxfgbg_heat_contrast_e20_lr5e-6_pad005_jit008",
]

METRIC_ORDER = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "boundary_iou",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics into a Markdown table.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--output", default=None, help="Output markdown path.")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("mean", {})


def format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_table(rows: list[dict[str, str]]) -> str:
    headers = ["experiment"] + METRIC_ORDER
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ablations_root = Path(args.ablations_root)
    output = Path(args.output) if args.output else ablations_root / f"{args.dataset}_{args.split}_ablation_summary.md"

    rows: list[dict[str, str]] = []
    for experiment in args.experiments:
        metrics_path = ablations_root / experiment / args.dataset / args.split / "metrics.json"
        metrics = load_metrics(metrics_path)
        row = {"experiment": experiment}
        for metric_name in METRIC_ORDER:
            row[metric_name] = format_metric(metrics.get(metric_name) if metrics else None)
        rows.append(row)

    markdown = "\n".join(
        [
            f"# {args.dataset} {args.split} Ablation Summary",
            "",
            build_table(rows),
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(f"Saved summary: {output}")
    print(markdown)


if __name__ == "__main__":
    main()
