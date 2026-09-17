#!/usr/bin/env python3
"""Build R206 hard-case visualizations for anatomy-consistency diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


DEFAULT_MODELS = {
    "ARAA": "outputs/ablations_variants/araa_danet_epoch98/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks",
    "R110": "outputs/ablations_variants/r110_r100_r108_patch_basic/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks",
    "R202_nnunet": "outputs/ablations_variants/r202_nnunet2d/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-case overlay panels from R203 manifest.")
    parser.add_argument("--manifest-csv", type=Path, default=Path("outputs/analysis/r203_failure_manifest/r203_failure_manifest.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--label-root", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2/test_labels"))
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r206_hardcase_visualizations"))
    parser.add_argument("--per-type", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--models-json", type=Path, default=None, help="Optional JSON mapping display name to mask dir.")
    parser.add_argument("--panel-width", type=int, default=320)
    return parser.parse_args()


def read_manifest(path: Path, per_type: int, max_cases: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    selected: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for row in rows:
        failure_type = row.get("failure_type", "unknown")
        if counts.get(failure_type, 0) >= per_type:
            continue
        selected.append(row)
        counts[failure_type] = counts.get(failure_type, 0) + 1
        if len(selected) >= max_cases:
            break
    return selected


def find_image(raw_root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = raw_root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found for {stem}")


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = arr > 0
    if shape is not None and mask.shape != shape:
        pil = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(pil.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    return mask


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    return np.logical_xor(dilated, eroded)


def tint_overlay(gray: np.ndarray, pred: np.ndarray | None, gt: np.ndarray) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    gt_b = boundary(gt)
    base[gt_b] = [0, 255, 0]
    if pred is not None:
        pred_b = boundary(pred)
        fp = pred & ~gt
        fn = gt & ~pred
        tp = pred & gt
        base[tp] = 0.70 * base[tp] + 0.30 * np.array([0, 120, 255], dtype=np.float32)
        base[fp] = 0.55 * base[fp] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        base[fn] = 0.55 * base[fn] + 0.45 * np.array([255, 255, 0], dtype=np.float32)
        base[pred_b] = [255, 0, 255]
        base[gt_b] = [0, 255, 0]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    ratio = width / float(image.width)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def label_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    pad = 28 if subtitle else 18
    out = Image.new("RGB", (image.width, image.height + pad), "white")
    out.paste(image.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((6, 3), title, fill=(0, 0, 0))
    if subtitle:
        draw.text((6, 15), subtitle, fill=(70, 70, 70))
    return out


def make_case_panel(
    row: dict[str, str],
    models: dict[str, Path],
    args: argparse.Namespace,
) -> tuple[Image.Image, list[str]]:
    name = row["image"]
    stem = Path(name).stem
    image = read_gray(find_image(args.raw_root, args.dataset, args.split, stem))
    gt = read_mask(args.label_root / name, image.shape)
    panels = [label_panel(resize_panel(tint_overlay(image, None, gt), args.panel_width), "Image + GT", "GT boundary: green")]
    missing_models: list[str] = []
    for model_name, mask_dir in models.items():
        mask_path = mask_dir / name
        if not mask_path.exists():
            missing_models.append(model_name)
            continue
        pred = read_mask(mask_path, image.shape)
        metric_bits = []
        key_prefix = model_name.lower().replace("r202_nnunet", "nnunet")
        for metric in ["dice", "boundary_iou", "gap_fp", "component_merge", "component_mae"]:
            value = row.get(f"{key_prefix}_{metric}")
            if value not in (None, ""):
                metric_bits.append(f"{metric}={float(value):.3f}")
        subtitle = " | ".join(metric_bits[:3])
        panels.append(label_panel(resize_panel(tint_overlay(image, pred, gt), args.panel_width), model_name, subtitle))

    height = max(panel.height for panel in panels)
    width = sum(panel.width for panel in panels)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out, missing_models


def write_html(rows: list[dict[str, Any]], output_dir: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>R206 hard cases</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222} .case{background:white;border:1px solid #ddd;margin:16px 0;padding:12px} img{max-width:100%;height:auto;border:1px solid #ccc} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:4px 6px;font-size:12px}</style>",
        "</head><body><h1>R206 Hard-Case Visualizations</h1>",
        "<p>Color legend: GT boundary green, prediction boundary magenta, TP blue tint, FP red tint, FN yellow tint.</p>",
    ]
    for row in rows:
        meta = html.escape(json.dumps(row["metrics"], ensure_ascii=False))
        parts.append("<div class='case'>")
        parts.append(f"<h2>{html.escape(row['image'])} - {html.escape(row['failure_type'])}</h2>")
        parts.append(f"<p>priority={row['priority_score']:.4f}; missing_models={html.escape(','.join(row['missing_models']))}</p>")
        parts.append(f"<img src='{html.escape(row['panel'])}' alt='{html.escape(row['image'])}'>")
        parts.append(f"<pre>{meta}</pre>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.models_json:
        raw_models = json.loads(args.models_json.read_text(encoding="utf-8"))
    else:
        raw_models = DEFAULT_MODELS
    models = {name: Path(path) for name, path in raw_models.items()}
    rows = read_manifest(args.manifest_csv, args.per_type, args.max_cases)
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        panel, missing = make_case_panel(row, models, args)
        panel_name = f"{rank:03d}_{Path(row['image']).stem}_{row.get('failure_type','unknown')}.jpg"
        panel_path = panels_dir / panel_name
        panel.save(panel_path, quality=92)
        index_rows.append(
            {
                "rank": rank,
                "image": row["image"],
                "failure_type": row.get("failure_type", "unknown"),
                "priority_score": float(row.get("priority_score", 0.0)),
                "panel": str(Path("panels") / panel_name).replace("\\", "/"),
                "missing_models": missing,
                "metrics": row,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r206_hardcase_visualization_index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    with (args.output_dir / "r206_hardcase_visualization_index.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["rank", "image", "failure_type", "priority_score", "panel", "missing_models"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in index_rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "image": row["image"],
                    "failure_type": row["failure_type"],
                    "priority_score": row["priority_score"],
                    "panel": row["panel"],
                    "missing_models": ",".join(row["missing_models"]),
                }
            )
    write_html(index_rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "num_cases": len(index_rows)}, indent=2))


if __name__ == "__main__":
    main()
