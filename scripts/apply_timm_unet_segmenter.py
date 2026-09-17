#!/usr/bin/env python3
"""Apply a trained timm U-Net/FPN segmenter checkpoint to one dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_timm_unet_segmenter import TimmUNet, infer_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a timm encoder segmentation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location=args.device)
    saved_args = state["args"]
    model = TimmUNet(
        saved_args["encoder"],
        bool(saved_args.get("pretrained", False)),
        int(saved_args["decoder_channels"]),
    ).to(args.device)
    model.load_state_dict(state["model"])
    threshold = float(state["threshold"])
    summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.dataset,
        args.split,
        int(saved_args["img_size"]),
        threshold,
        args.device,
        Path(args.pred_root) / args.output_exp / args.dataset / args.split / "masks",
        Path(args.metrics_json),
    )
    print(
        {
            "dataset": args.dataset,
            "split": args.split,
            "threshold": threshold,
            "mean": summary["mean"],
        }
    )


if __name__ == "__main__":
    main()
