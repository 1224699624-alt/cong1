#!/usr/bin/env python3
"""R215 background-connectivity candidate diagnostics.

This is a no-training, no-mask-writing audit for the "background connectivity
equals bone-gap separation" route. It regenerates R210/R211-style candidate
cuts from the R110 anchor on original train/val and measures whether deleting a
candidate cut opens local background channels without eroding GT bone.

clean-test-v2 must not be used by this script.
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

from run_r201_unified_eval import GtCache, boundary as r201_boundary, surface as r201_surface
from run_r210_f1_neck_candidate_gate import candidate_components, read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


INFERENCE_FEATURE_KEYS = [
    "r215_crop_h",
    "r215_crop_w",
    "r215_crop_area",
    "r215_cut_area",
    "r215_cut_frac_crop",
    "r215_bg_components_before",
    "r215_bg_components_after",
    "r215_bg_component_count_delta",
    "r215_cut_adjacent_bg_components_before",
    "r215_cut_adjacent_bg_frac_before",
    "r215_bg_components_merge_potential",
    "r215_cut_component_bg_area_after",
    "r215_cut_component_bg_frac_crop_after",
    "r215_cut_component_border_contacts_after",
    "r215_cut_component_touches_crop_border_after",
    "r215_fg_components_before",
    "r215_fg_components_after",
    "r215_fg_component_count_delta",
    "r215_bg_channel_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R215 local background-connectivity candidate features.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-candidate-csv", type=Path, default=Path(""))
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=16)
    parser.add_argument("--max-components-per-image", type=int, default=5)
    parser.add_argument("--crop-pad", type=int, default=18)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r215_background_connectivity_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r215_background_connectivity_val_summary.json"))
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def build_one_gt_cache(name: str, gt: np.ndarray, boundary_kernel: int, gap_kernel: int, with_surface_distance: bool) -> GtCache:
    gap_structure = np.ones((gap_kernel, gap_kernel), dtype=bool)
    gt_boundary = r201_boundary(gt, boundary_kernel)
    gt_surface = r201_surface(gt)
    gt_surface_distance = ndimage.distance_transform_edt(~gt_surface) if with_surface_distance and gt_surface.any() else None
    gt_gap = np.logical_and(ndimage.binary_dilation(gt, structure=gap_structure), ~gt)
    gt_components, _ = ndimage.label(gt, structure=np.ones((3, 3), dtype=np.uint8))
    return GtCache(
        image=name,
        mask=gt,
        boundary=gt_boundary,
        surface=gt_surface,
        surface_distance=gt_surface_distance,
        gap_region=gt_gap,
        component_count=int(gt_components.max()),
    )


def metric_delta(new: dict[str, float | None], old: dict[str, float | None], key: str) -> float | None:
    if old.get(key) is None or new.get(key) is None:
        return None
    return float(new[key]) - float(old[key])


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def quick_candidate_metrics(pred: np.ndarray, gt: np.ndarray, gt_cache: GtCache, boundary_kernel: int) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    pred_boundary = r201_boundary(pred, boundary_kernel)
    gt_boundary = gt_cache.boundary
    b_tp = float(np.logical_and(pred_boundary, gt_boundary).sum())
    b_fp = float(np.logical_and(pred_boundary, ~gt_boundary).sum())
    b_fn = float(np.logical_and(gt_boundary, ~pred_boundary).sum())
    b_union = float(np.logical_or(pred_boundary, gt_boundary).sum())
    _, pred_n = ndimage.label(pred, structure=np.ones((3, 3), dtype=np.uint8))
    gap_pixels = float(gt_cache.gap_region.sum())
    gap_fp = float(np.logical_and(pred, gt_cache.gap_region).sum())
    return {
        "dice": safe_divide(2.0 * tp, 2.0 * tp + fp + fn, default=1.0),
        "iou": safe_divide(tp, tp + fp + fn, default=1.0),
        "precision": safe_divide(tp, tp + fp, default=1.0),
        "recall": safe_divide(tp, tp + fn, default=1.0),
        "boundary_iou": safe_divide(b_tp, b_union, default=0.0),
        "boundary_f1": safe_divide(2.0 * b_tp, 2.0 * b_tp + b_fp + b_fn, default=0.0),
        "gap_region_fp_rate": safe_divide(gap_fp, gap_pixels, default=0.0),
        "component_count_mae": float(abs(int(pred_n) - int(gt_cache.component_count))),
        "component_delta_mean": float(int(pred_n) - int(gt_cache.component_count)),
    }


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def count_components(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    return int(n)


def border_contacts(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    contacts = 0
    contacts += int(mask[0, :].any())
    contacts += int(mask[-1, :].any())
    contacts += int(mask[:, 0].any())
    contacts += int(mask[:, -1].any())
    return contacts


def local_background_features(anchor: np.ndarray, trial: np.ndarray, cut: np.ndarray, gt: np.ndarray, gt_cache: GtCache, pad: int) -> dict[str, float]:
    cut = cut.astype(bool)
    local_slice = bbox_slice(cut, pad)
    if local_slice is None:
        return {key: 0.0 for key in [*INFERENCE_FEATURE_KEYS, "r215_cut_gt_fg_frac", "r215_cut_gt_bg_frac", "r215_cut_gt_gap_frac"]}

    anchor_l = anchor[local_slice].astype(bool)
    trial_l = trial[local_slice].astype(bool)
    cut_l = cut[local_slice].astype(bool)
    gt_l = gt[local_slice].astype(bool)
    gt_gap_l = gt_cache.gap_region[local_slice].astype(bool)

    bg_before = ~anchor_l
    bg_after = ~trial_l
    bg_labels_before, bg_n_before = ndimage.label(bg_before, structure=np.ones((3, 3), dtype=np.uint8))
    bg_labels_after, bg_n_after = ndimage.label(bg_after, structure=np.ones((3, 3), dtype=np.uint8))

    cut_ring = np.logical_and(ndimage.binary_dilation(cut_l, structure=np.ones((3, 3), dtype=bool)), ~cut_l)
    adjacent_labels = np.unique(bg_labels_before[np.logical_and(cut_ring, bg_before)])
    adjacent_labels = adjacent_labels[adjacent_labels > 0]
    adjacent_count = int(adjacent_labels.size)

    cut_labels_after = np.unique(bg_labels_after[cut_l])
    cut_labels_after = cut_labels_after[cut_labels_after > 0]
    if cut_labels_after.size:
        cut_bg_component = np.isin(bg_labels_after, cut_labels_after)
    else:
        cut_bg_component = np.zeros_like(bg_after, dtype=bool)

    fg_n_before = count_components(anchor_l)
    fg_n_after = count_components(trial_l)
    crop_area = int(cut_l.size)
    cut_area = int(cut_l.sum())
    cut_component_area = int(cut_bg_component.sum())

    return {
        "r215_crop_h": float(anchor_l.shape[0]),
        "r215_crop_w": float(anchor_l.shape[1]),
        "r215_crop_area": float(crop_area),
        "r215_cut_area": float(cut_area),
        "r215_cut_frac_crop": float(cut_area / max(1, crop_area)),
        "r215_bg_components_before": float(bg_n_before),
        "r215_bg_components_after": float(bg_n_after),
        "r215_bg_component_count_delta": float(bg_n_after - bg_n_before),
        "r215_cut_adjacent_bg_components_before": float(adjacent_count),
        "r215_cut_adjacent_bg_frac_before": float(np.logical_and(cut_ring, bg_before).sum() / max(1, int(cut_ring.sum()))),
        "r215_bg_components_merge_potential": float(max(0, adjacent_count - 1)),
        "r215_cut_component_bg_area_after": float(cut_component_area),
        "r215_cut_component_bg_frac_crop_after": float(cut_component_area / max(1, crop_area)),
        "r215_cut_component_border_contacts_after": float(border_contacts(cut_bg_component)),
        "r215_cut_component_touches_crop_border_after": float(border_contacts(cut_bg_component) > 0),
        "r215_fg_components_before": float(fg_n_before),
        "r215_fg_components_after": float(fg_n_after),
        "r215_fg_component_count_delta": float(fg_n_after - fg_n_before),
        "r215_bg_channel_proxy": float(
            border_contacts(cut_bg_component) >= 2
            and cut_component_area / max(1, crop_area) >= 0.20
            and np.logical_and(cut_ring, bg_before).sum() / max(1, int(cut_ring.sum())) >= 0.35
        ),
        "r215_cut_gt_fg_frac": float(np.logical_and(cut_l, gt_l).sum() / max(1, cut_area)),
        "r215_cut_gt_bg_frac": float(np.logical_and(cut_l, ~gt_l).sum() / max(1, cut_area)),
        "r215_cut_gt_gap_frac": float(np.logical_and(cut_l, gt_gap_l).sum() / max(1, cut_area)),
    }


def is_strict_useful(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("delta_dice") or 0.0) >= -5e-4
        and float(row.get("delta_iou") or 0.0) >= -8e-4
        and float(row.get("delta_recall") or 0.0) >= -8e-4
        and float(row.get("delta_boundary_iou") or 0.0) >= 1e-4
        and float(row.get("delta_boundary_f1") or 0.0) >= 1e-4
        and float(row.get("delta_surface_dice_2px") or 0.0) >= 1e-4
        and float(row.get("delta_gap_region_fp_rate") or 0.0) < 0.0
        and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
    )


def is_quick_useful(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("delta_dice") or 0.0) >= -5e-4
        and float(row.get("delta_iou") or 0.0) >= -8e-4
        and float(row.get("delta_recall") or 0.0) >= -8e-4
        and float(row.get("delta_boundary_iou") or 0.0) > 0.0
        and float(row.get("delta_boundary_f1") or 0.0) > 0.0
        and float(row.get("delta_gap_region_fp_rate") or 0.0) < 0.0
        and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
    )


def is_hard_risk(row: dict[str, Any]) -> bool:
    return bool(
        float(row.get("delta_dice") or 0.0) < -5e-4
        or float(row.get("delta_iou") or 0.0) < -8e-4
        or float(row.get("delta_recall") or 0.0) < -8e-4
        or float(row.get("delta_boundary_iou") or 0.0) < 0.0
        or float(row.get("delta_boundary_f1") or 0.0) < 0.0
        or float(row.get("delta_surface_dice_2px") or 0.0) < 0.0
    )


def source_names(args: argparse.Namespace) -> list[str]:
    if str(args.source_candidate_csv) and args.source_candidate_csv != Path("."):
        with args.source_candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        ordered = []
        seen = set()
        for row in rows:
            name = str(row.get("image") or "")
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered[: args.limit or None]
    return names(args.raw_root, args.dataset, args.split)[: args.limit or None]


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(tqdm(source_names(args), desc=f"r215/background-connectivity/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            continue
        image, _instance, gt, anchor = load_case(args, name)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel, with_surface_distance=True)
        old = quick_candidate_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
        for rank, raw_cut in enumerate(cuts[: args.max_components_per_image], start=1):
            cut = np.logical_and(raw_cut, anchor)
            if int(cut.sum()) < args.min_cut_area:
                continue
            trial = np.logical_and(anchor, ~cut)
            new = quick_candidate_metrics(trial, gt, gt_cache, args.boundary_kernel)
            row: dict[str, Any] = {
                "run_id": "R215-background-connectivity-candidate-diagnostic",
                "split": args.split,
                "image": name,
                "candidate_rank": float(rank),
            }
            row.update(local_background_features(anchor, trial, cut, gt, gt_cache, args.crop_pad))
            for key, value in old.items():
                row[f"anchor_{key}"] = value
            for key, value in new.items():
                row[f"candidate_{key}"] = value
                row[f"delta_{key}"] = metric_delta(new, old, key)
            row["r215_strict_useful"] = float(is_strict_useful(row))
            row["r215_quick_useful"] = float(is_quick_useful(row))
            row["r215_hard_risk"] = float(is_hard_risk(row))
            row["r215_bg_merge_proxy_positive"] = float(
                float(row["r215_cut_adjacent_bg_components_before"]) >= 2.0
                and float(row["r215_bg_component_count_delta"]) < 0.0
                and float(row["r215_cut_gt_fg_frac"]) <= 0.5
            )
            row["r215_bg_safe_channel_proxy_positive"] = float(
                float(row["r215_bg_channel_proxy"]) > 0.5
                and float(row["r215_cut_gt_fg_frac"]) <= 0.5
                and float(row["r215_cut_gt_gap_frac"]) >= 0.5
            )
            row["r215_overerosion_proxy"] = float(float(row["r215_cut_gt_fg_frac"]) > 0.5)
            rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.output_csv, rows)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summarize(rows, args), indent=2), encoding="utf-8")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [row.get(key) for row in rows]
    finite = [float(v) for v in vals if isinstance(v, (float, int)) and np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    useful = [row for row in rows if float(row.get("r215_strict_useful") or 0.0) > 0.5]
    quick_useful = [row for row in rows if float(row.get("r215_quick_useful") or 0.0) > 0.5]
    risk = [row for row in rows if float(row.get("r215_hard_risk") or 0.0) > 0.5]
    bg_merge_proxy = [row for row in rows if float(row.get("r215_bg_merge_proxy_positive") or 0.0) > 0.5]
    bg_channel_proxy = [row for row in rows if float(row.get("r215_bg_safe_channel_proxy_positive") or 0.0) > 0.5]
    return {
        "run_id": "R215-background-connectivity-candidate-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "purpose": "train/val-only local background-connectivity proxy for bone-gap separation candidates",
        "anchor_exp": args.anchor_exp,
        "num_candidate_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_strict_useful": len(useful),
        "num_quick_useful": len(quick_useful),
        "num_hard_risk": len(risk),
        "num_bg_merge_proxy_positive": len(bg_merge_proxy),
        "num_bg_safe_channel_proxy_positive": len(bg_channel_proxy),
        "inference_feature_keys": INFERENCE_FEATURE_KEYS,
        "mean": {
            key: mean_value(rows, key)
            for key in [
                "r215_cut_adjacent_bg_components_before",
                "r215_bg_component_count_delta",
                "r215_cut_gt_fg_frac",
                "r215_cut_gt_gap_frac",
                "delta_dice",
                "delta_boundary_iou",
                "delta_gap_region_fp_rate",
            ]
        },
        "bg_safe_channel_proxy_positive_mean": {
            key: mean_value(bg_channel_proxy, key)
            for key in [
                "r215_strict_useful",
                "r215_quick_useful",
                "r215_hard_risk",
                "r215_cut_gt_fg_frac",
                "r215_cut_gt_gap_frac",
                "delta_dice",
                "delta_boundary_iou",
                "delta_gap_region_fp_rate",
            ]
        },
        "warning": "candidate-level train/val diagnostic only; not R201 mask-level evidence and not clean-test-v2 evidence",
    }


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    report = summarize(rows, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(args.output_json), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
