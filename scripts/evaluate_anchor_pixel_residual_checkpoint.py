#!/usr/bin/env python3
"""Evaluate a saved anchor-local pixel residual checkpoint with a small config grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_anchor_pixel_residual import MLP, eval_config, parse_nums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved anchor-local pixel residual checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument(
        "--apply-anchor-exp",
        default=None,
        help="Anchor mask experiment used for apply-dataset evaluation. Defaults to --anchor-exp.",
    )
    parser.add_argument("--candidate-exps", nargs="+", required=True)
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--boundary-radii", default="4,6")
    parser.add_argument("--prob-thresholds", default="0.50,0.60")
    parser.add_argument("--edit-margins", default="0.10,0.20")
    parser.add_argument("--fixed-radius", type=int, default=None, help="Skip tuning and apply this edit radius directly.")
    parser.add_argument("--fixed-threshold", type=float, default=None, help="Skip tuning and apply this probability threshold directly.")
    parser.add_argument("--fixed-margin", type=float, default=None, help="Skip tuning and apply this edit margin directly.")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    payload = torch.load(args.checkpoint, map_location=args.device)
    in_dim = int(payload["in_dim"])
    model = MLP(in_dim, args.hidden).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()

    tried = []
    if args.fixed_radius is not None or args.fixed_threshold is not None or args.fixed_margin is not None:
        if args.fixed_radius is None or args.fixed_threshold is None or args.fixed_margin is None:
            raise ValueError("--fixed-radius, --fixed-threshold, and --fixed-margin must be provided together.")
        best = {
            "radius": args.fixed_radius,
            "threshold": args.fixed_threshold,
            "margin": args.fixed_margin,
            "mean": None,
            "source": "fixed",
        }
    else:
        best = None
        for radius in parse_nums(args.boundary_radii, int):
            for threshold in parse_nums(args.prob_thresholds, float):
                for margin in parse_nums(args.edit_margins, float):
                    mean, _ = eval_config(
                        model,
                        args,
                        roots,
                        args.train_dataset,
                        Path(args.train_raw_root),
                        args.tune_split,
                        None,
                        threshold,
                        margin,
                        radius,
                    )
                    item = {"radius": radius, "threshold": threshold, "margin": margin, "mean": mean}
                    tried.append(item)
                    if best is None or mean["dice"] > best["mean"]["dice"]:
                        best = item
        assert best is not None

    out_dir = Path(args.ablations_roots[0]) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = eval_config(
        model,
        args,
        roots,
        args.apply_dataset,
        Path(args.apply_raw_root),
        args.apply_split,
        args.source_apply_dataset,
        best["threshold"],
        best["margin"],
        best["radius"],
        out_dir,
    )
    summary = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "best": best,
        "tried": tried,
        "output_exp": args.output_exp,
        "checkpoint": args.checkpoint,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"mean": mean, "best": best}, indent=2))


if __name__ == "__main__":
    main()
