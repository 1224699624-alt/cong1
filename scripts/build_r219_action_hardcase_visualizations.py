#!/usr/bin/env python3
"""Build R219 action-level hard-case visualizations for R216-R218.

The panels show why soft seam actions are useful under oracle diagnostics but
hard to select safely: each case displays the image/GT, R110 anchor, the action
cut overlay, and the candidate-after-action overlay. This is train/val
diagnostic only and writes no model masks.
"""

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

from audit_r216_soft_seam_action_candidates import soft_action_cut
from run_r210_f1_neck_candidate_gate import candidate_components, read_instance
from train_anchor_pixel_residual import image_path, read_gray, read_mask, resize_like


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build R219 action-level hard-case visualization panels.")
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r216_soft_seam_action_fullval_candidates.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="val")
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r219_action_hardcase_visualizations"))
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=8)
    parser.add_argument("--max-components-per-image", type=int, default=3)
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--panel-width", type=int, default=300)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return (np.clip(image * 255.0, 0, 255).astype(np.uint8), gt, anchor)


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def bbox_slice(mask: np.ndarray, pad: int = 36) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def tint_overlay(gray: np.ndarray, pred: np.ndarray | None, gt: np.ndarray, cut: np.ndarray | None = None) -> Image.Image:
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
    if cut is not None and cut.any():
        cut_b = boundary(cut)
        base[cut] = 0.35 * base[cut] + 0.65 * np.array([255, 120, 0], dtype=np.float32)
        base[cut_b] = [255, 255, 255]
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def crop_image(image: Image.Image, local_slice: tuple[slice, slice] | None) -> Image.Image:
    if local_slice is None:
        return image
    y, x = local_slice
    return image.crop((x.start, y.start, x.stop, y.stop))


def resize_panel(image: Image.Image, width: int) -> Image.Image:
    ratio = width / float(image.width)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.BILINEAR)


def label_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    pad = 34 if subtitle else 20
    out = Image.new("RGB", (image.width, image.height + pad), "white")
    out.paste(image.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(out)
    draw.text((6, 3), title, fill=(0, 0, 0))
    if subtitle:
        draw.text((6, 17), subtitle[:125], fill=(70, 70, 70))
    return out


def action_cut(args: argparse.Namespace, image_float: np.ndarray, anchor: np.ndarray, row: dict[str, str]) -> np.ndarray:
    cuts = candidate_components(anchor, image_float, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
    rank = int(round(as_float(row, "candidate_rank")))
    if rank <= 0 or rank > min(len(cuts), args.max_components_per_image):
        return np.zeros_like(anchor, dtype=bool)
    return np.logical_and(soft_action_cut(image_float, anchor, cuts[rank - 1], as_float(row, "action_frac"), args.min_cut_area), anchor)


def case_score(row: dict[str, str], group: str) -> float:
    if group == "safe_useful":
        return (
            4.0 * as_float(row, "r216_quick_useful")
            - 4.0 * as_float(row, "r216_hard_risk")
            + 500.0 * as_float(row, "delta_boundary_iou")
            - 100.0 * as_float(row, "delta_gap_region_fp_rate")
            - 50.0 * max(0.0, -as_float(row, "delta_dice"))
        )
    if group == "hard_risk":
        return (
            4.0 * as_float(row, "r216_hard_risk")
            + 3.0 * as_float(row, "r216_overerosion_proxy")
            + 50.0 * max(0.0, -as_float(row, "delta_dice"))
            + 20.0 * max(0.0, -as_float(row, "delta_recall"))
        )
    return abs(as_float(row, "delta_boundary_iou")) + abs(as_float(row, "delta_dice")) + as_float(row, "r216_cut_gt_fg_frac")


def select_cases(rows: list[dict[str, str]], per_group: int) -> list[dict[str, str]]:
    groups = {
        "safe_useful": [
            row
            for row in rows
            if as_float(row, "r216_quick_useful") > 0.5
            and as_float(row, "r216_hard_risk") < 0.5
            and as_float(row, "r216_safe_channel_proxy_positive") > 0.5
        ],
        "hard_risk": [
            row
            for row in rows
            if as_float(row, "r216_hard_risk") > 0.5
            and as_float(row, "r216_overerosion_proxy") > 0.5
        ],
        "ambiguous": [
            row
            for row in rows
            if as_float(row, "r216_quick_useful") > 0.5
            and as_float(row, "r216_overerosion_proxy") > 0.5
        ],
    }
    selected = []
    used: set[tuple[str, str, str]] = set()
    for group, group_rows in groups.items():
        ranked = sorted(group_rows, key=lambda row: case_score(row, group), reverse=True)
        count = 0
        for row in ranked:
            key = (row["image"], row["candidate_rank"], row["action_frac"])
            if key in used:
                continue
            out = dict(row)
            out["r219_group"] = group
            selected.append(out)
            used.add(key)
            count += 1
            if count >= per_group:
                break
    return selected


def make_panel(args: argparse.Namespace, row: dict[str, str]) -> Image.Image:
    gray, gt, anchor = load_case(args, row["image"])
    image_float = gray.astype(np.float32) / 255.0
    cut = action_cut(args, image_float, anchor, row)
    candidate = np.logical_and(anchor, ~cut)
    local = bbox_slice(np.logical_or(cut, boundary(anchor)), pad=40)
    metric_sub = (
        f"dD={as_float(row,'delta_dice'):+.5f} dBIoU={as_float(row,'delta_boundary_iou'):+.5f} "
        f"dGap={as_float(row,'delta_gap_region_fp_rate'):+.5f}"
    )
    label_sub = (
        f"rank={as_float(row,'candidate_rank'):.0f} frac={as_float(row,'action_frac'):.2f} "
        f"useful={as_float(row,'r216_quick_useful'):.0f} risk={as_float(row,'r216_hard_risk'):.0f} "
        f"fg={as_float(row,'r216_cut_gt_fg_frac'):.2f} gap={as_float(row,'r216_cut_gt_gap_frac'):.2f}"
    )
    panels = [
        label_panel(resize_panel(crop_image(tint_overlay(gray, None, gt), local), args.panel_width), "Image + GT", "GT boundary green"),
        label_panel(resize_panel(crop_image(tint_overlay(gray, anchor, gt), local), args.panel_width), "R110 anchor", "TP blue, FP red, FN yellow"),
        label_panel(resize_panel(crop_image(tint_overlay(gray, anchor, gt, cut), local), args.panel_width), "Action cut", label_sub),
        label_panel(resize_panel(crop_image(tint_overlay(gray, candidate, gt, cut), local), args.panel_width), "After action", metric_sub),
    ]
    h = max(panel.height for panel in panels)
    out = Image.new("RGB", (sum(panel.width for panel in panels), h), "white")
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    return out


def write_html(index_rows: list[dict[str, Any]], output_dir: Path) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>R219 action hard cases</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f7;color:#222}.case{background:white;border:1px solid #ddd;margin:16px 0;padding:12px}img{max-width:100%;height:auto;border:1px solid #ccc}pre{font-size:12px;white-space:pre-wrap}</style>",
        "</head><body><h1>R219 Action-Level Hard Cases</h1>",
        "<p>Legend: GT boundary green, prediction boundary magenta, TP blue tint, FP red, FN yellow, action cut orange/white.</p>",
    ]
    for row in index_rows:
        parts.append("<div class='case'>")
        parts.append(f"<h2>{html.escape(row['image'])} - {html.escape(row['group'])}</h2>")
        parts.append(f"<img src='{html.escape(row['panel'])}' alt='{html.escape(row['image'])}'>")
        parts.append(f"<pre>{html.escape(json.dumps(row['metrics'], ensure_ascii=False, indent=2))}</pre>")
        parts.append("</div>")
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.candidate_csv)
    selected = select_cases(rows, args.per_group)
    panels_dir = args.output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for idx, row in enumerate(selected, start=1):
        panel = make_panel(args, row)
        panel_name = f"{idx:03d}_{row['r219_group']}_{Path(row['image']).stem}_r{int(as_float(row,'candidate_rank'))}_f{as_float(row,'action_frac'):.2f}.jpg"
        panel.save(panels_dir / panel_name, quality=92)
        index_rows.append(
            {
                "rank": idx,
                "image": row["image"],
                "group": row["r219_group"],
                "candidate_rank": as_float(row, "candidate_rank"),
                "action_frac": as_float(row, "action_frac"),
                "panel": str(Path("panels") / panel_name).replace("\\", "/"),
                "metrics": row,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "r219_action_hardcase_index.json").write_text(json.dumps(index_rows, indent=2), encoding="utf-8")
    with (args.output_dir / "r219_action_hardcase_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "image", "group", "candidate_rank", "action_frac", "panel"])
        writer.writeheader()
        for row in index_rows:
            writer.writerow({key: row[key] for key in ["rank", "image", "group", "candidate_rank", "action_frac", "panel"]})
    write_html(index_rows, args.output_dir)
    summary = {
        "run_id": "R219-action-hardcase-visualizations",
        "candidate_csv": str(args.candidate_csv),
        "output_dir": str(args.output_dir),
        "num_cases": len(index_rows),
        "groups": {group: sum(row["group"] == group for row in index_rows) for group in sorted({row["group"] for row in index_rows})},
        "clean_test_v2_used": False,
        "writes_model_masks": False,
    }
    (args.output_dir / "r219_action_hardcase_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
