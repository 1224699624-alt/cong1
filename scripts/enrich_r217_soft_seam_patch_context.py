#!/usr/bin/env python3
"""Add R217 GT-free patch-context features to R216 soft seam actions.

The R216 CSV has metrics but not pixel coordinates. This script regenerates the
same R216 candidate actions from the R110 anchor and merges local image/patch
features back into the candidate-action table by (image, candidate_rank,
action_frac). It writes CSV/JSON only and does not use clean-test-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from tqdm import tqdm

from audit_r216_soft_seam_action_candidates import soft_action_cut
from run_r210_f1_neck_candidate_gate import candidate_components, read_instance
from train_anchor_pixel_residual import image_path, read_gray, read_mask, resize_like


R217_FEATURE_KEYS = [
    "r217_cut_intensity_mean",
    "r217_cut_intensity_std",
    "r217_cut_intensity_p10",
    "r217_cut_intensity_p90",
    "r217_ring_intensity_mean",
    "r217_ring_intensity_std",
    "r217_inner_ring_intensity_mean",
    "r217_outer_ring_intensity_mean",
    "r217_cut_minus_ring_intensity",
    "r217_cut_minus_inner_intensity",
    "r217_cut_minus_outer_intensity",
    "r217_inner_minus_outer_intensity",
    "r217_cut_gradient_mean",
    "r217_cut_gradient_p90",
    "r217_ring_gradient_mean",
    "r217_outer_ring_gradient_mean",
    "r217_cut_ring_gradient_ratio",
    "r217_cut_outer_gradient_ratio",
    "r217_anchor_dist_mean",
    "r217_anchor_dist_p90",
    "r217_bg_dist_mean",
    "r217_bg_dist_p10",
    "r217_local_bg_frac",
    "r217_local_anchor_frac",
    "r217_cut_touches_boundary_frac",
    "r217_cut_neighbor_bg_frac",
    "r217_cut_neighbor_fg_frac",
    "r217_local_h",
    "r217_local_w",
    "r217_local_area_frac_image",
    "r217_cut_y_center_norm",
    "r217_cut_x_center_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich R216 soft seam actions with R217 patch-context features.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=8)
    parser.add_argument("--max-components-per-image", type=int, default=3)
    parser.add_argument("--action-fracs", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--crop-pad", type=int, default=18)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("image") or ""), f"{float(row.get('candidate_rank') or 0.0):.6f}", f"{float(row.get('action_frac') or 0.0):.6f}"


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), instance.shape)
    return image, anchor


def safe_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0}
    values = values.astype(np.float32, copy=False)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def safe_ratio(num: float, den: float, eps: float = 1e-6) -> float:
    return float(num / max(eps, den))


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def patch_features(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray, pad: int) -> dict[str, float]:
    cut = np.logical_and(cut, anchor).astype(bool)
    if int(cut.sum()) <= 0:
        return {key: 0.0 for key in R217_FEATURE_KEYS}

    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0.0:
        grad = grad / float(grad.max())

    dist_in = ndimage.distance_transform_edt(anchor)
    dist_bg = ndimage.distance_transform_edt(~anchor)
    local_slice = bbox_slice(cut, pad)
    local_mask = np.zeros_like(cut, dtype=bool)
    if local_slice is not None:
        local_mask[local_slice] = True

    dil3 = ndimage.binary_dilation(cut, structure=np.ones((3, 3), dtype=bool))
    dil5 = ndimage.binary_dilation(cut, structure=np.ones((5, 5), dtype=bool))
    dil9 = ndimage.binary_dilation(cut, structure=np.ones((9, 9), dtype=bool))
    ring = np.logical_and(dil5, ~cut)
    inner_ring = np.logical_and(ring, anchor)
    outer_ring = np.logical_and(ring, ~anchor)
    wider_outer = np.logical_and(dil9, ~dil3)
    wider_outer = np.logical_and(wider_outer, ~anchor)
    neighbor_bg = np.logical_and(dil3, ~anchor)
    neighbor_fg = np.logical_and(dil3, anchor)
    anchor_boundary = np.logical_xor(
        ndimage.binary_dilation(anchor, structure=np.ones((3, 3), dtype=bool)),
        ndimage.binary_erosion(anchor, structure=np.ones((3, 3), dtype=bool)),
    )

    cut_i = safe_stats(image[cut])
    ring_i = safe_stats(image[ring])
    inner_i = safe_stats(image[inner_ring])
    outer_i = safe_stats(image[outer_ring])
    cut_g = safe_stats(grad[cut])
    ring_g = safe_stats(grad[ring])
    outer_g = safe_stats(grad[np.logical_or(outer_ring, wider_outer)])
    dist_i = safe_stats(dist_in[cut])
    bg_i = safe_stats(dist_bg[cut])

    ys, xs = np.where(cut)
    if local_slice is None:
        local_h = 0
        local_w = 0
        local_area_frac = 0.0
    else:
        local_h = int(local_slice[0].stop - local_slice[0].start)
        local_w = int(local_slice[1].stop - local_slice[1].start)
        local_area_frac = float(local_mask.sum() / max(1, cut.size))

    return {
        "r217_cut_intensity_mean": cut_i["mean"],
        "r217_cut_intensity_std": cut_i["std"],
        "r217_cut_intensity_p10": cut_i["p10"],
        "r217_cut_intensity_p90": cut_i["p90"],
        "r217_ring_intensity_mean": ring_i["mean"],
        "r217_ring_intensity_std": ring_i["std"],
        "r217_inner_ring_intensity_mean": inner_i["mean"],
        "r217_outer_ring_intensity_mean": outer_i["mean"],
        "r217_cut_minus_ring_intensity": float(cut_i["mean"] - ring_i["mean"]),
        "r217_cut_minus_inner_intensity": float(cut_i["mean"] - inner_i["mean"]),
        "r217_cut_minus_outer_intensity": float(cut_i["mean"] - outer_i["mean"]),
        "r217_inner_minus_outer_intensity": float(inner_i["mean"] - outer_i["mean"]),
        "r217_cut_gradient_mean": cut_g["mean"],
        "r217_cut_gradient_p90": cut_g["p90"],
        "r217_ring_gradient_mean": ring_g["mean"],
        "r217_outer_ring_gradient_mean": outer_g["mean"],
        "r217_cut_ring_gradient_ratio": safe_ratio(cut_g["mean"], ring_g["mean"]),
        "r217_cut_outer_gradient_ratio": safe_ratio(cut_g["mean"], outer_g["mean"]),
        "r217_anchor_dist_mean": dist_i["mean"],
        "r217_anchor_dist_p90": dist_i["p90"],
        "r217_bg_dist_mean": bg_i["mean"],
        "r217_bg_dist_p10": bg_i["p10"],
        "r217_local_bg_frac": float(np.logical_and(local_mask, ~anchor).sum() / max(1, int(local_mask.sum()))),
        "r217_local_anchor_frac": float(np.logical_and(local_mask, anchor).sum() / max(1, int(local_mask.sum()))),
        "r217_cut_touches_boundary_frac": float(np.logical_and(cut, anchor_boundary).sum() / max(1, int(cut.sum()))),
        "r217_cut_neighbor_bg_frac": float(neighbor_bg.sum() / max(1, int(dil3.sum()))),
        "r217_cut_neighbor_fg_frac": float(neighbor_fg.sum() / max(1, int(dil3.sum()))),
        "r217_local_h": float(local_h),
        "r217_local_w": float(local_w),
        "r217_local_area_frac_image": local_area_frac,
        "r217_cut_y_center_norm": float((ys.min() + ys.max() + 1) / max(1, 2 * cut.shape[0])),
        "r217_cut_x_center_norm": float((xs.min() + xs.max() + 1) / max(1, 2 * cut.shape[1])),
    }


def build_feature_map(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, float]]:
    wanted_images = sorted({str(row["image"]) for row in rows})
    action_fracs = parse_float_list(args.action_fracs)
    feature_map: dict[tuple[str, str, str], dict[str, float]] = {}
    for name in tqdm(wanted_images, desc=f"r217/enrich/{args.split}"):
        if not anchor_path(args, name).exists():
            continue
        image, anchor = load_case(args, name)
        cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
        for rank, raw_cut in enumerate(cuts[: args.max_components_per_image], start=1):
            for action_frac in action_fracs:
                cut = soft_action_cut(image, anchor, raw_cut, action_frac, args.min_cut_area)
                cut = np.logical_and(cut, anchor)
                if int(cut.sum()) < args.min_cut_area:
                    continue
                feature_map[(name, f"{float(rank):.6f}", f"{float(action_frac):.6f}")] = patch_features(image, anchor, cut, args.crop_pad)
    return feature_map


def main() -> None:
    args = parse_args()
    rows = read_csv(args.candidate_csv)
    feature_map = build_feature_map(args, rows)
    merged = []
    matched = 0
    for row in rows:
        out = dict(row)
        features = feature_map.get(row_key(row))
        if features is not None:
            matched += 1
        for key in R217_FEATURE_KEYS:
            out[key] = str((features or {}).get(key, 0.0))
        out["r217_feature_matched"] = "1.0" if features is not None else "0.0"
        merged.append(out)
    write_csv(args.output_csv, merged)
    report = {
        "run_id": "R217-soft-seam-patch-context-enrichment",
        "candidate_csv": str(args.candidate_csv),
        "output_csv": str(args.output_csv),
        "num_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_matched": matched,
        "num_unmatched": len(rows) - matched,
        "match_rate": float(matched / max(1, len(rows))),
        "feature_keys": [*R217_FEATURE_KEYS, "r217_feature_matched"],
        "clean_test_v2_used": False,
        "writes_masks": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
