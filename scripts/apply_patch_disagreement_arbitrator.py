#!/usr/bin/env python3
"""Apply a trained patch disagreement arbitrator checkpoint to a dataset split."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from train_patch_disagreement_arbitrator import (
    TinyPatchNet,
    collect_eval_cache,
    eval_cached,
    parse_nums,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply patch arbitrator checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--candidate-exps", nargs="+", required=True)
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--feature-mode", default="basic", choices=["basic", "local_stats"])
    parser.add_argument("--zone-mode", default="anchor_candidate_disagreement", choices=["candidate_disagreement", "anchor_candidate_disagreement"])
    parser.add_argument("--radius", type=int, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--margin", type=float, required=True)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def make_eval_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        apply_dataset=args.dataset,
        apply_split=args.split,
        source_apply_dataset=args.source_dataset,
        apply_anchor_exp=args.anchor_exp,
        anchor_exp=args.anchor_exp,
        apply_candidate_exps=args.candidate_exps,
        candidate_exps=args.candidate_exps,
        ablations_roots=args.ablations_roots,
        feature_mode=args.feature_mode,
        zone_mode=args.zone_mode,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = TinyPatchNet(int(ckpt["in_ch"]), int(ckpt["args"].get("base_channels", 20))).to(args.device)
    model.load_state_dict(ckpt["model"])
    roots = [Path(p) for p in args.ablations_roots]
    eval_args = make_eval_args(args)
    cache = collect_eval_cache(
        model,
        eval_args,
        roots,
        args.dataset,
        Path(args.raw_root),
        args.split,
        args.source_dataset,
        args.max_images,
    )
    out_dir = Path(args.ablations_roots[0]) / args.output_exp / args.dataset / args.split / "masks"
    mean, records = eval_cached(
        cache,
        args.threshold,
        args.margin,
        args.radius,
        args.boundary_kernel,
        args.zone_mode,
        out_dir,
    )
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "checkpoint": args.checkpoint,
        "radius": args.radius,
        "threshold": args.threshold,
        "margin": args.margin,
        "output_exp": args.output_exp,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"mean": mean, "num_evaluated": len(records)}, indent=2))


if __name__ == "__main__":
    main()
