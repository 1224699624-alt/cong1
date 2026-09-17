#!/usr/bin/env python3
"""Probe R204 bridge candidate generation on train/val anchors only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe conservative bridge candidate coverage.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="val")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--pred-root", type=Path, default=Path("outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r204_bridge_candidate_probe.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r204_bridge_candidate_probe.csv"))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--gap-radii", default="3,5,7")
    parser.add_argument("--thin-radii", default="1,2,3")
    parser.add_argument("--distance-fracs", default="0.10,0.16,0.22,0.30")
    parser.add_argument("--max-remove-fracs", default="0.00025,0.0005,0.001,0.002")
    parser.add_argument("--min-cut-areas", default="1,2,4,8")
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    label_paths = sorted((args.raw_root / args.dataset / f"{args.split}_labels").glob("*.png"))
    if args.max_images > 0:
        label_paths = label_paths[: args.max_images]
    items = []
    for gt_path in label_paths:
        pred_path = args.pred_root / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        gt = read_mask(gt_path)
        anchor = resize_like(read_mask(pred_path), gt.shape)
        items.append({"image": gt_path.name, "gt": gt, "anchor": anchor})
    if not items:
        raise FileNotFoundError(args.raw_root / args.dataset / f"{args.split}_labels")
    return items


def bridge_pieces(anchor: np.ndarray, gap_radius: int, thin_radius: int, distance_frac: float, min_cut_area: int) -> list[np.ndarray]:
    gap = ndimage.binary_dilation(~anchor, structure=np.ones((gap_radius, gap_radius), dtype=bool))
    dist = cv2.distanceTransform(anchor.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return []
    eroded = ndimage.binary_erosion(anchor, structure=np.ones((max(1, thin_radius), max(1, thin_radius)), dtype=bool))
    thin = anchor & ~eroded
    low_dist = anchor & (dist <= float(distance_frac) * float(dist.max()))
    cand = thin & low_dist & gap
    labels, n = ndimage.label(cand, structure=np.ones((3, 3), dtype=np.uint8))
    pieces = []
    for idx in range(1, n + 1):
        piece = labels == idx
        if int(piece.sum()) >= min_cut_area:
            pieces.append(piece)
    return sorted(pieces, key=lambda piece: int(piece.sum()))


def main() -> None:
    args = parse_args()
    items = load_items(args)
    rows = []
    for gap_radius in parse_ints(args.gap_radii):
        for thin_radius in parse_ints(args.thin_radii):
            for distance_frac in parse_floats(args.distance_fracs):
                for min_cut_area in parse_ints(args.min_cut_areas):
                    image_stats = []
                    for item in items:
                        anchor = item["anchor"]
                        pieces = bridge_pieces(anchor, gap_radius, thin_radius, distance_frac, min_cut_area)
                        areas = [int(piece.sum()) for piece in pieces]
                        image_stats.append(
                            {
                                "image": item["image"],
                                "anchor_area": int(anchor.sum()),
                                "num_pieces": len(areas),
                                "total_area": int(sum(areas)),
                                "min_area": min(areas) if areas else 0,
                                "median_area": float(np.median(areas)) if areas else 0.0,
                            }
                        )
                    for max_remove_frac in parse_floats(args.max_remove_fracs):
                        removable_images = 0
                        removable_pixels = 0
                        for stat in image_stats:
                            max_remove = int(round(max_remove_frac * max(1, stat["anchor_area"])))
                            if stat["num_pieces"] > 0 and stat["min_area"] <= max_remove:
                                removable_images += 1
                                removable_pixels += min(max_remove, stat["total_area"])
                        rows.append(
                            {
                                "gap_radius": gap_radius,
                                "thin_radius": thin_radius,
                                "distance_frac": distance_frac,
                                "min_cut_area": min_cut_area,
                                "max_remove_frac": max_remove_frac,
                                "images_with_pieces": int(sum(stat["num_pieces"] > 0 for stat in image_stats)),
                                "images_with_removable_pieces": removable_images,
                                "mean_num_pieces": float(np.mean([stat["num_pieces"] for stat in image_stats])),
                                "mean_total_area": float(np.mean([stat["total_area"] for stat in image_stats])),
                                "median_min_area_nonzero": float(np.median([stat["min_area"] for stat in image_stats if stat["min_area"] > 0])) if any(stat["min_area"] > 0 for stat in image_stats) else 0.0,
                                "approx_removable_pixels": removable_pixels,
                            }
                        )

    rows = sorted(rows, key=lambda row: (row["images_with_removable_pieces"], row["approx_removable_pixels"]), reverse=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"num_images": len(items), "rows": rows}, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"num_images": len(items), "top_rows": rows[:10]}, indent=2))


if __name__ == "__main__":
    main()
