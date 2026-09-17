#!/usr/bin/env python3
"""Build R210-F1 original-val hard-case visualizations."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R210-F1 val hard-case panels.")
    parser.add_argument("--manifest-csv", type=Path, default=Path("outputs/analysis/r210_f1_hardcase_visualization_manifest.csv"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("outputs/analysis/r210_f1_neck_candidate_gate_val_fastinner_per_image.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="val")
    parser.add_argument("--r110-dir", type=Path, default=Path("outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks"))
    parser.add_argument("--r210-dir", type=Path, default=Path("outputs/ablations_variants/r210_f1_neck_candidate_gate_val_fastinner/TSRS_RSNA-Epiphysis/val/masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r210_f1_val_hardcase_visualizations"))
    parser.add_argument("--panel-width", type=int, default=300)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_image(raw_root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = raw_root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = arr > 0
    if shape is not None and mask.shape != shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0
    return mask


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def tint_overlay(gray: np.ndarray, pred: np.ndarray | None, gt: np.ndarray) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    gt_b = boundary(gt)
    base[gt_b] = [0, 255, 0]
    if pred is not None:
        pred_b = boundary(pred)
        tp = pred & gt
        fp = pred & ~gt
        fn = gt & ~pred
        base[tp] = 0.70 * base[tp] + 0.30 * np.array([0, 120, 255], dtype=np.float32)
        base[fp] = 0.55 * base[fp] + 0.45 * np.array([255, 0, 0], dtype=np.float32)
        base[fn] = 0.55 * base[fn] + 0.45 * np.array([255, 255, 0], dtype=np.float32)
        base[pred_b] = [255, 0, 255]
        base[gt_b] = [0, 255, 0]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def cut_overlay(gray: np.ndarray, r110: np.ndarray, r210: np.ndarray, gt: np.ndarray) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    removed = r110 & ~r210
    added = r210 & ~r110
    gt_b = boundary(gt)
    r210_b = boundary(r210)
    base[removed] = 0.40 * base[removed] + 0.60 * np.array([255, 80, 0], dtype=np.float32)
    base[added] = 0.40 * base[added] + 0.60 * np.array([0, 220, 255], dtype=np.float32)
    base[r210_b] = [255, 0, 255]
    base[gt_b] = [0, 255, 0]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    ratio = width / float(image.width)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def label_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    pad = 32 if subtitle else 20
    out = Image.new("RGB", (image.width, image.height + pad), "white")
    out.paste(image.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((6, 3), title, fill=(0, 0, 0))
    if subtitle:
        draw.text((6, 17), subtitle[:120], fill=(70, 70, 70))
    return out


def fmt_delta(metrics: dict[str, str], key: str) -> str:
    value = metrics.get(key, "")
    if value == "":
        return ""
    return f"{float(value):+.4f}"


def make_panel(row: dict[str, str], metrics: dict[str, str], args: argparse.Namespace) -> Image.Image:
    name = row["image"]
    stem = Path(name).stem
    gray = read_gray(find_image(args.raw_root, args.dataset, args.split, stem))
    gt = read_mask(args.raw_root / args.dataset / f"{args.split}_labels" / name, gray.shape)
    r110 = read_mask(args.r110_dir / name, gray.shape)
    r210 = read_mask(args.r210_dir / name, gray.shape)
    metric_sub = (
        f"Dice {fmt_delta(metrics, 'delta_dice')} | Gap {fmt_delta(metrics, 'delta_gap_region_fp_rate')} | "
        f"Cnt {fmt_delta(metrics, 'delta_component_count_mae')} | BIou {fmt_delta(metrics, 'delta_boundary_iou')}"
    )
    panels = [
        label_panel(resize_panel(tint_overlay(gray, None, gt), args.panel_width), "Image + GT", "GT boundary green"),
        label_panel(resize_panel(tint_overlay(gray, r110, gt), args.panel_width), "R110 anchor", "TP blue, FP red, FN yellow"),
        label_panel(resize_panel(tint_overlay(gray, r210, gt), args.panel_width), "R210-F1", metric_sub),
        label_panel(resize_panel(cut_overlay(gray, r110, r210, gt), args.panel_width), "Accepted cut", "removed orange, added cyan"),
    ]
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (sum(p.width for p in panels), height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out


def write_html(index_rows: list[dict[str, Any]], output_dir: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>R210-F1 val hard cases</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}.case{background:white;border:1px solid #ddd;margin:16px 0;padding:12px}img{max-width:100%;height:auto;border:1px solid #ccc}pre{font-size:12px;white-space:pre-wrap}</style>",
        "</head><body><h1>R210-F1 Original-Val Hard Cases</h1>",
        "<p>Legend: GT boundary green, prediction boundary magenta, TP blue tint, FP red, FN yellow. Accepted cuts are orange.</p>",
    ]
    for row in index_rows:
        parts.append("<div class='case'>")
        parts.append(f"<h2>{html.escape(row['image'])} - {html.escape(row['group'])}</h2>")
        parts.append(f"<p>{html.escape(row['reason'])}</p>")
        parts.append(f"<img src='{html.escape(row['panel'])}' alt='{html.escape(row['image'])}'>")
        parts.append(f"<pre>{html.escape(json.dumps(row['metrics'], ensure_ascii=False, indent=2))}</pre>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = read_rows(args.manifest_csv)
    metrics_by_image = {row["image"]: row for row in read_rows(args.metrics_csv)}
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(manifest, start=1):
        metrics = metrics_by_image.get(row["image"], {})
        panel = make_panel(row, metrics, args)
        panel_name = f"{idx:02d}_{row['group']}_{Path(row['image']).stem}.jpg"
        panel.save(panels_dir / panel_name, quality=92)
        index_rows.append(
            {
                "rank": idx,
                "image": row["image"],
                "group": row["group"],
                "reason": row["reason"],
                "panel": str(Path("panels") / panel_name).replace("\\", "/"),
                "metrics": metrics,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r210_f1_val_hardcase_visualization_index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    with (args.output_dir / "r210_f1_val_hardcase_visualization_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "image", "group", "reason", "panel"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow({key: row[key] for key in ["rank", "image", "group", "reason", "panel"]})
    write_html(index_rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "num_cases": len(index_rows)}, indent=2))


if __name__ == "__main__":
    main()
