#!/usr/bin/env python3
"""Create visual audit panels for hard segmentation cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hard-case visual audit pack.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--alignment-json", default=None)
    parser.add_argument("--anchor-name", default="R110")
    parser.add_argument("--anchor-mask-dir", required=True)
    parser.add_argument("--candidate", action="append", default=[], help="NAME=MASK_DIR")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-width", type=int, default=420)
    return parser.parse_args()


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST)) > 0


def fit_image(image: Image.Image, max_width: int) -> Image.Image:
    if image.width <= max_width:
        return image
    scale = max_width / float(image.width)
    return image.resize((max_width, max(1, int(image.height * scale))), Image.BILINEAR)


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    arr = np.asarray(image).copy()
    mask_rs = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
    arr[mask_rs] = ((1.0 - alpha) * arr[mask_rs] + alpha * np.array(color)).astype(np.uint8)
    return Image.fromarray(arr)


def error_overlay(image: Image.Image, pred: np.ndarray, gt: np.ndarray) -> Image.Image:
    arr = np.asarray(image).copy()
    pred_rs = np.asarray(Image.fromarray(pred.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
    gt_rs = np.asarray(Image.fromarray(gt.astype(np.uint8) * 255).resize(image.size, Image.NEAREST)) > 0
    fp = pred_rs & ~gt_rs
    fn = ~pred_rs & gt_rs
    tp = pred_rs & gt_rs
    arr[tp] = ((0.72 * arr[tp]) + 0.28 * np.array([40, 210, 80])).astype(np.uint8)
    arr[fp] = ((0.45 * arr[fp]) + 0.55 * np.array([255, 55, 55])).astype(np.uint8)
    arr[fn] = ((0.45 * arr[fn]) + 0.55 * np.array([55, 120, 255])).astype(np.uint8)
    return Image.fromarray(arr)


def title_panel(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    title_h = 44 if subtitle else 28
    canvas = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    canvas.paste(image, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(0, 0, 0))
    if subtitle:
        draw.text((8, 25), subtitle, fill=(70, 70, 70))
    return canvas


def hstack(panels: list[Image.Image]) -> Image.Image:
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def parse_candidates(items: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Candidate must be NAME=MASK_DIR: {item}")
        name, path = item.split("=", 1)
        parsed.append((name, Path(path)))
    return parsed


def uniq_records(records: list[dict[str, object]], top_k: int) -> list[dict[str, object]]:
    seen: set[str] = set()
    out = []
    for record in records:
        name = str(record["image"])
        if name in seen:
            continue
        seen.add(name)
        out.append(record)
        if len(out) >= top_k:
            break
    return out


def main() -> None:
    args = parse_args()
    audit = json.loads(Path(args.audit_json).read_text(encoding="utf-8"))
    align_by_image = {}
    if args.alignment_json:
        align = json.loads(Path(args.alignment_json).read_text(encoding="utf-8"))
        align_by_image = {str(item["image"]): item for item in align.get("per_image", [])}

    selected = uniq_records(
        audit.get("low_dice", [])
        + audit.get("overmask_like", [])
        + audit.get("recall_limited", [])
        + audit.get("bridge_like", []),
        args.top_k,
    )

    output_dir = Path(args.output_dir)
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.raw_root) / args.dataset / args.split
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    anchor_dir = Path(args.anchor_mask_dir)
    candidates = parse_candidates(args.candidate)
    rows = []

    for rank, record in enumerate(selected, start=1):
        image_name = str(record["image"])
        stem = Path(image_name).stem
        image_path = find_image(image_dir, stem)
        gt_path = gt_dir / image_name
        anchor_path = anchor_dir / image_name
        if image_path is None or not gt_path.exists() or not anchor_path.exists():
            continue

        base = fit_image(load_rgb(image_path), args.max_width)
        size = load_rgb(image_path).size
        gt = load_mask(gt_path, size)
        anchor = load_mask(anchor_path, size)

        panels = [
            title_panel(base, "Image"),
            title_panel(overlay(base, gt, (40, 210, 80), 0.45), "GT", "green"),
            title_panel(overlay(base, anchor, (255, 55, 55), 0.42), args.anchor_name, "red"),
            title_panel(error_overlay(base, anchor, gt), "Error", "green TP / red FP / blue FN"),
        ]
        for cand_name, cand_dir in candidates:
            mask_path = cand_dir / image_name
            if mask_path.exists():
                cand = load_mask(mask_path, size)
                panels.append(title_panel(error_overlay(base, cand, gt), cand_name, "error overlay"))
        canvas = hstack(panels)
        out_name = f"{rank:02d}_{stem}_audit.jpg"
        canvas.save(panel_dir / out_name, quality=94)
        align_rec = align_by_image.get(image_name, {})
        rows.append({
            "rank": rank,
            "image": image_name,
            "panel": f"panels/{out_name}",
            "dice": float(record.get("dice", 0.0)),
            "precision": float(record.get("precision", 0.0)),
            "recall": float(record.get("recall", 0.0)),
            "boundary_iou": float(record.get("boundary_iou", 0.0)),
            "component_delta": int(record.get("component_delta", 0)),
            "area_frac_delta": float(record.get("area_frac_delta", 0.0)),
            "error_in_disagreement_share": float(align_rec.get("error_in_disagreement_share", 0.0)) if align_rec else None,
            "disagreement_pixel_oracle_dice": float(align_rec.get("disagreement_pixel_oracle_dice", 0.0)) if align_rec else None,
        })

    md = [
        "# R122 R110 Hard-Case Visual Audit",
        "",
        f"Dataset: `{args.dataset}/{args.split}`",
        f"Anchor: `{args.anchor_name}`",
        "",
        "| Rank | Image | Dice | Precision | Recall | Boundary IoU | Comp Delta | Disagree Error Share | Panel |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        share = row["error_in_disagreement_share"]
        share_text = "" if share is None else f"{share:.3f}"
        md.append(
            f"| {row['rank']} | {row['image']} | {row['dice']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['boundary_iou']:.4f} | {row['component_delta']} | "
            f"{share_text} | [{Path(row['panel']).name}]({row['panel']}) |"
        )
    (output_dir / "AUDIT_INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (output_dir / "audit_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "num_panels": len(rows), "index": str(output_dir / "AUDIT_INDEX.md")}, indent=2))


if __name__ == "__main__":
    main()
