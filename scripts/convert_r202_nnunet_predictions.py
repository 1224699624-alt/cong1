#!/usr/bin/env python3
"""Convert R202 nnU-Net predictions into the R201 mask directory format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert nnU-Net R202 predictions to binary PNG masks.")
    parser.add_argument("--pred-dir", default="outputs/nnunet/r202/predictions/Dataset202_TSRS_RSNAEpiphysis2D")
    parser.add_argument(
        "--mask-dir",
        default="outputs/ablations_variants/r202_nnunet2d/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks",
    )
    parser.add_argument("--summary", default="outputs/analysis/r202_nnunet2d_prediction_conversion_summary.json")
    parser.add_argument("--expected-count", type=int, default=81)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    mask_dir = Path(args.mask_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    mask_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for pred_path in sorted(pred_dir.glob("*.png")):
        stem = pred_path.stem
        if stem.endswith("_0000"):
            stem = stem[:-5]
        arr = np.asarray(Image.open(pred_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = (arr > 0).astype(np.uint8) * 255
        out_path = mask_dir / f"{stem}.png"
        Image.fromarray(mask, mode="L").save(out_path)
        converted.append(out_path.name)

    summary = {
        "run_id": "R202",
        "pred_dir": str(pred_dir),
        "mask_dir": str(mask_dir),
        "num_converted": len(converted),
        "expected_count": args.expected_count,
        "count_matches_expected": len(converted) == args.expected_count,
        "converted": converted,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if len(converted) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} masks, converted {len(converted)}")


if __name__ == "__main__":
    main()
