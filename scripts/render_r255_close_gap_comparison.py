#!/usr/bin/env python3
"""Render fair baseline-selected full-hand and close-gap R255 comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
TILE = (360, 480)


def find_valid_image(root: Path, stem: str) -> Path:
    for extension in EXTENSIONS:
        path = root / f"{stem}{extension}"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            return path
        except OSError:
            continue
    raise FileNotFoundError(f"No valid image for {stem} in {root}")


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def read_mask(path: Path) -> np.ndarray:
    return read_gray(path) > 0


def read_instance(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.int32)


def resize(array: np.ndarray, *, mask: bool = False) -> Image.Image:
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    return Image.fromarray(array.astype(np.uint8)).convert("RGB").resize(
        TILE, Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    )


def overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = np.stack([image] * 3, axis=-1).astype(np.float32)
    false_positive, false_negative = pred & ~gt, gt & ~pred
    base[false_positive] = 0.35 * base[false_positive] + 0.65 * np.array([255, 30, 30])
    base[false_negative] = 0.35 * base[false_negative] + 0.65 * np.array([255, 220, 20])
    return resize(np.clip(base, 0, 255).astype(np.uint8))


def crop_box(instance: np.ndarray, first: int, second: int, margin_fraction: float = 0.45) -> tuple[int, int, int, int]:
    selected = (instance == first) | (instance == second)
    yy, xx = np.nonzero(selected)
    if not len(xx):
        return 0, 0, instance.shape[1], instance.shape[0]
    width, height = int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)
    margin = max(24, int(round(max(width, height) * margin_fraction)))
    x0, x1 = max(0, int(xx.min()) - margin), min(instance.shape[1], int(xx.max()) + margin + 1)
    y0, y1 = max(0, int(yy.min()) - margin), min(instance.shape[0], int(yy.max()) + margin + 1)
    return x0, y0, x1, y1


def select_rows(path: Path, num_cases: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["gap_bin"] in {"1-2", "3-4"}]
    best_by_image: dict[str, dict[str, str]] = {}
    for row in rows:
        current = best_by_image.get(row["image"])
        key = (float(row["baseline_pair_merged"]), float(row["baseline_gap_fp_rate"]), -float(row["gap_px"]))
        if current is None:
            best_by_image[row["image"]] = row
            continue
        current_key = (float(current["baseline_pair_merged"]), float(current["baseline_gap_fp_rate"]), -float(current["gap_px"]))
        if key > current_key:
            best_by_image[row["image"]] = row
    # Selection uses baseline fields only; R255 outcomes never affect ranking.
    ranked = sorted(
        best_by_image.values(),
        key=lambda row: (
            -float(row["baseline_pair_merged"]),
            -float(row["baseline_gap_fp_rate"]),
            float(row["gap_px"]),
            row["image"],
        ),
    )
    return ranked[:num_cases]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val"))
    p.add_argument("--gt-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val_labels"))
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--r255-dir", type=Path, required=True)
    p.add_argument("--pair-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/visualizations/r255_close_gap_original_val"))
    p.add_argument("--num-cases", type=int, default=12)
    args = p.parse_args()
    joined = " ".join(map(str, (args.image_dir, args.gt_dir, args.baseline_dir, args.r255_dir))).lower()
    if "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R255 visualization permits Epiphysis original-val only")
    selected = select_rows(args.pair_csv, args.num_cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["X-ray", "GT", "Matched control", "R255 probe", "Control FP/FN", "R255 FP/FN"]
    manifest = []
    for row in selected:
        name, stem = row["image"], Path(row["image"]).stem
        image = read_gray(find_valid_image(args.image_dir, stem))
        instance = read_instance(args.gt_dir / name)
        gt = instance > 0
        baseline = read_mask(args.baseline_dir / name)
        r255 = read_mask(args.r255_dir / name)
        if baseline.shape != gt.shape or r255.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch for {name}")
        full = [resize(image), resize(gt, mask=True), resize(baseline, mask=True), resize(r255, mask=True), overlay(image, gt, baseline), overlay(image, gt, r255)]
        box = crop_box(instance, int(row["instance_i"]), int(row["instance_j"]))
        x0, y0, x1, y1 = box
        image_z, gt_z = image[y0:y1, x0:x1], gt[y0:y1, x0:x1]
        baseline_z, r255_z = baseline[y0:y1, x0:x1], r255[y0:y1, x0:x1]
        zoom = [resize(image_z), resize(gt_z, mask=True), resize(baseline_z, mask=True), resize(r255_z, mask=True), overlay(image_z, gt_z, baseline_z), overlay(image_z, gt_z, r255_z)]
        header_height, row_height, width = 68, 520, len(labels) * TILE[0]
        canvas = Image.new("RGB", (width, header_height + 2 * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        summary = (
            f"{stem} | GT pair {row['instance_i']}-{row['instance_j']} | gap={float(row['gap_px']):.2f}px | "
            f"gap FP {float(row['baseline_gap_fp_rate']):.3f}->{float(row['r255_gap_fp_rate']):.3f} | "
            f"merged {int(float(row['baseline_pair_merged']))}->{int(float(row['r255_pair_merged']))}"
        )
        draw.text((10, 8), summary, fill="black", font=ImageFont.load_default())
        for column, label in enumerate(labels):
            draw.text((column * TILE[0] + 8, 38), label, fill="black", font=ImageFont.load_default())
            canvas.paste(full[column], (column * TILE[0], header_height))
            canvas.paste(zoom[column], (column * TILE[0], header_height + row_height))
        draw.text((8, header_height + 4), "Full image", fill="black")
        draw.text((8, header_height + row_height + 4), "Close-gap zoom", fill="black")
        output = args.output_dir / f"{stem}_comparison.png"
        canvas.save(output, optimize=True)
        manifest.append({
            "image": name,
            "panel": str(output),
            "selection": "baseline-only 1-4px pair merge then gap-FP risk",
            "instance_i": int(row["instance_i"]),
            "instance_j": int(row["instance_j"]),
            "gap_px": float(row["gap_px"]),
            "baseline_gap_fp_rate": float(row["baseline_gap_fp_rate"]),
            "r255_gap_fp_rate": float(row["r255_gap_fp_rate"]),
            "baseline_pair_merged": float(row["baseline_pair_merged"]),
            "r255_pair_merged": float(row["r255_pair_merged"]),
            "crop_xyxy": list(box),
        })
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"rendered": len(manifest), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
