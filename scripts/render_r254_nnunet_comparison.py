#!/usr/bin/env python3
"""Render baseline-selected original-val comparison panels for R254."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path: Path, gray: bool = False) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr if gray else arr > 0


def resize(arr: np.ndarray, size=(360, 480), mask=False) -> Image.Image:
    image = Image.fromarray(arr.astype(np.uint8) * 255 if arr.dtype == bool else arr.astype(np.uint8)).convert("RGB")
    return image.resize(size, Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR)


def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = np.stack([image] * 3, axis=-1).astype(np.float32)
    fp, fn = pred & ~gt, gt & ~pred
    base[fp] = 0.35 * base[fp] + 0.65 * np.array([255, 30, 30])
    base[fn] = 0.35 * base[fn] + 0.65 * np.array([255, 220, 20])
    return resize(np.clip(base, 0, 255).astype(np.uint8))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", default="data/raw/TSRS_RSNA-Epiphysis/val")
    p.add_argument("--gt-dir", default="data/raw/TSRS_RSNA-Epiphysis/val_labels")
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--ours-dir", required=True)
    p.add_argument("--baseline-metrics", required=True)
    p.add_argument("--output-dir", default="outputs/visualizations/r254_hapdsp_pcr_original_val")
    p.add_argument("--num-cases", type=int, default=12)
    args = p.parse_args()
    metrics = json.loads(Path(args.baseline_metrics).read_text(encoding="utf-8"))["per_image"]
    # Selection is frozen from baseline errors only, never from ours improvement.
    ranked = sorted(metrics, key=lambda r: (r["component_merge_rate"] < 0.5, r["boundary_iou"]))[: args.num_cases]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_dir, gt_dir = Path(args.image_dir), Path(args.gt_dir)
    baseline_dir, ours_dir = Path(args.baseline_dir), Path(args.ours_dir)
    manifest = []
    for row in ranked:
        name = row["image"]
        candidates = list(image_dir.glob(Path(name).stem + ".*"))
        image = load(candidates[0], gray=True)
        gt, baseline, ours = load(gt_dir / name), load(baseline_dir / name), load(ours_dir / name)
        tiles = [resize(image), resize(gt, mask=True), resize(baseline, mask=True), resize(ours, mask=True), overlay(image, gt, baseline), overlay(image, gt, ours)]
        labels = ["X-ray", "GT", "nnU-Net", "HA-PDSP+PCR", "baseline FP/FN", "ours FP/FN"]
        canvas = Image.new("RGB", (len(tiles) * 360, 520), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (tile, label) in enumerate(zip(tiles, labels)):
            canvas.paste(tile, (i * 360, 40))
            draw.text((i * 360 + 8, 10), label, fill="black")
        out = output / f"{Path(name).stem}_comparison.png"
        canvas.save(out)
        manifest.append({"image": name, "panel": str(out), "selection": "baseline-only merge/boundary risk"})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"rendered {len(manifest)} baseline-selected panels")


if __name__ == "__main__":
    main()
