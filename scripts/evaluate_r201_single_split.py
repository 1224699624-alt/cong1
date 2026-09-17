#!/usr/bin/env python3
"""Apply the frozen R201 metric definitions to one explicit prediction folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_r201_unified_eval import compute_metrics, read_binary_mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-count", type=int, required=True)
    args = p.parse_args()
    gt_dir, pred_dir = Path(args.gt_dir), Path(args.pred_dir)
    if "clean-test" in (str(gt_dir) + str(pred_dir)).lower():
        raise RuntimeError("R254 development evaluation forbids clean-test-v2")
    rows = []
    for gt_path in sorted(gt_dir.glob("*.png")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        gt, pred = read_binary_mask(gt_path), read_binary_mask(pred_path)
        if pred.shape != gt.shape:
            from PIL import Image

            pred = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize(gt.shape[::-1], Image.Resampling.NEAREST)) > 0
        row = compute_metrics(pred, gt, 3, 9, 2.0, 5.0)
        row["image"] = gt_path.name
        rows.append(row)
    if len(rows) != args.expected_count:
        raise SystemExit(f"Expected {args.expected_count} evaluated images, got {len(rows)}")
    metric_names = [k for k in rows[0] if k != "image"]
    mean = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in metric_names}
    payload = {"protocol": "R201 metric definitions on original-val", "num_images": len(rows), "mean": mean, "per_image": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()
