#!/usr/bin/env python3
"""Build R238-A original-val visual audit panels."""

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
    parser = argparse.ArgumentParser(description="Build visual panels for R238-A oracle-best seam edits.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="val")
    parser.add_argument("--anchor-dir", type=Path, default=Path("outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks"))
    parser.add_argument("--edited-dir", type=Path, default=Path("outputs/ablations_variants/r238_r237_oracle_best_mask_editor/TSRS_RSNA-Epiphysis/val/masks"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r238_r237_oracle_best_mask_editor_per_image.csv"))
    parser.add_argument("--selected-csv", type=Path, default=Path("outputs/analysis/r238_r237_oracle_best_mask_editor_selected.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r238_r237_oracle_best_visual_audit"))
    parser.add_argument("--panel-width", type=int, default=320)
    parser.add_argument("--crop-pad", type=int, default=56)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


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


def bbox_slice(mask: np.ndarray, pad: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return slice(0, shape[0]), slice(0, shape[1])
    return (
        slice(max(0, int(ys.min()) - pad), min(shape[0], int(ys.max()) + pad + 1)),
        slice(max(0, int(xs.min()) - pad), min(shape[1], int(xs.max()) + pad + 1)),
    )


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


def edit_overlay(gray: np.ndarray, anchor: np.ndarray, edited: np.ndarray, gt: np.ndarray) -> Image.Image:
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32)
    removed = anchor & ~edited
    removed_gap = removed & ~gt
    removed_fg = removed & gt
    base[removed_gap] = 0.35 * base[removed_gap] + 0.65 * np.array([255, 120, 0], dtype=np.float32)
    base[removed_fg] = 0.25 * base[removed_fg] + 0.75 * np.array([255, 0, 0], dtype=np.float32)
    base[boundary(edited)] = [255, 0, 255]
    base[boundary(gt)] = [0, 255, 0]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    ratio = width / float(image.width)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def label_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    pad = 38 if subtitle else 22
    out = Image.new("RGB", (image.width, image.height + pad), "white")
    out.paste(image.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((6, 3), title, fill=(0, 0, 0))
    if subtitle:
        draw.text((6, 18), subtitle[:135], fill=(70, 70, 70))
    return out


def fmt(value: str, digits: int = 6) -> str:
    if value == "":
        return ""
    return f"{float(value):+.{digits}f}"


def make_panel(name: str, metrics: dict[str, str], selected: dict[str, str], args: argparse.Namespace) -> Image.Image:
    stem = Path(name).stem
    gray = read_gray(find_image(args.raw_root, args.dataset, args.split, stem))
    gt = read_mask(args.raw_root / args.dataset / f"{args.split}_labels" / name, gray.shape)
    anchor = read_mask(args.anchor_dir / name, gray.shape)
    edited = read_mask(args.edited_dir / name, gray.shape)
    removed = anchor & ~edited
    local = bbox_slice(removed, args.crop_pad, gray.shape)
    metric_sub = (
        f"Dice {fmt(metrics.get('delta_dice', ''))} | Recall {fmt(metrics.get('delta_recall', ''))} | "
        f"BIoU {fmt(metrics.get('delta_boundary_iou', ''))} | Gap {fmt(metrics.get('delta_gap_region_fp_rate', ''))} | "
        f"Cnt {fmt(metrics.get('delta_component_count_mae', ''))}"
    )
    cut_sub = (
        f"removed {int(float(metrics.get('cut_pixels') or 0))} px | "
        f"fg {float(selected.get('cut_gt_fg_frac') or 0):.3f} | gap {float(selected.get('cut_gt_gap_frac') or 0):.3f} | "
        f"bg_conn {float(selected.get('background_connected_by_cut') or 0):.0f}"
    )
    panels = [
        label_panel(resize_panel(tint_overlay(gray[local], None, gt[local]), args.panel_width), "Image + GT", "GT boundary green"),
        label_panel(resize_panel(tint_overlay(gray[local], anchor[local], gt[local]), args.panel_width), "R110 anchor", "TP blue, FP red, FN yellow"),
        label_panel(resize_panel(tint_overlay(gray[local], edited[local], gt[local]), args.panel_width), "R238-A oracle upper bound", metric_sub),
        label_panel(resize_panel(edit_overlay(gray[local], anchor[local], edited[local], gt[local]), args.panel_width), "Replayed seam cut", cut_sub),
    ]
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (sum(panel.width for panel in panels), height), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out


def write_html(index_rows: list[dict[str, Any]], output_dir: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>R238-A visual audit</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}.case{background:white;border:1px solid #ddd;margin:16px 0;padding:12px}img{max-width:100%;height:auto;border:1px solid #ccc}pre{font-size:12px;white-space:pre-wrap}</style>",
        "</head><body><h1>R238-A Original-Val Visual Audit</h1>",
        "<p>Oracle upper-bound only. Legend: GT boundary green, prediction boundary magenta, TP blue tint, FP red, FN yellow. Replayed deletions are orange if background/gap and red if GT foreground.</p>",
    ]
    for row in index_rows:
        parts.append("<div class='case'>")
        parts.append(f"<h2>{html.escape(row['image'])}</h2>")
        parts.append(f"<p>{html.escape(row['summary'])}</p>")
        parts.append(f"<img src='{html.escape(row['panel'])}' alt='{html.escape(row['image'])}'>")
        parts.append(f"<pre>{html.escape(json.dumps(row['metrics'], ensure_ascii=False, indent=2))}</pre>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    metrics_rows = {row["image"]: row for row in read_rows(args.per_image_csv)}
    selected_rows = {row["image"]: row for row in read_rows(args.selected_csv)}
    edited = [name for name, row in metrics_rows.items() if float(row.get("edited") or 0.0) > 0.5]
    edited.sort(
        key=lambda name: (
            -float(metrics_rows[name].get("delta_boundary_iou") or 0.0),
            float(metrics_rows[name].get("delta_gap_region_fp_rate") or 0.0),
            float(metrics_rows[name].get("delta_component_count_mae") or 0.0),
        )
    )
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    for idx, name in enumerate(edited, start=1):
        metrics = metrics_rows[name]
        selected = selected_rows.get(name, {})
        panel = make_panel(name, metrics, selected, args)
        panel_name = f"{idx:02d}_{Path(name).stem}_r238_oracle_panel.jpg"
        panel.save(panels_dir / panel_name, quality=94)
        summary = (
            f"cut={metrics.get('cut_pixels')} px, Dice {fmt(metrics.get('delta_dice', ''))}, "
            f"Recall {fmt(metrics.get('delta_recall', ''))}, BIoU {fmt(metrics.get('delta_boundary_iou', ''))}, "
            f"Gap {fmt(metrics.get('delta_gap_region_fp_rate', ''))}, Cnt {fmt(metrics.get('delta_component_count_mae', ''))}"
        )
        index_rows.append(
            {
                "rank": idx,
                "image": name,
                "panel": str(Path("panels") / panel_name).replace("\\", "/"),
                "summary": summary,
                "metrics": metrics,
                "selected": selected,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r238_oracle_best_visual_audit_index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    with (args.output_dir / "r238_oracle_best_visual_audit_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "image", "panel", "summary"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow({key: row[key] for key in ["rank", "image", "panel", "summary"]})
    write_html(index_rows, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "num_cases": len(index_rows)}, indent=2))


if __name__ == "__main__":
    main()
