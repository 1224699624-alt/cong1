#!/usr/bin/env python3
"""R216 soft seam-action candidate diagnostics.

R215 showed that background-channel features can identify bone-gap separation
opportunities, but hard deletion is too risky. R216 keeps the same train/val
candidate family and audits softer actions: remove only the most seam-like
fraction of each candidate cut instead of deleting the whole component.

This script writes CSV/JSON diagnostics only. It does not write masks and must
not be used on clean-test-v2 for tuning.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R216 soft seam-action candidates on original train/val only.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-candidate-csv", type=Path, default=Path(""))
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=8)
    parser.add_argument("--max-components-per-image", type=int, default=3)
    parser.add_argument("--action-fracs", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--crop-pad", type=int, default=18)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r216_soft_seam_action_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r216_soft_seam_action_val_summary.json"))
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def build_one_gt_cache(name: str, gt: np.ndarray, boundary_kernel: int, gap_kernel: int) -> GtCache:
    gap_structure = np.ones((gap_kernel, gap_kernel), dtype=bool)
    gt_boundary = r201_boundary(gt, boundary_kernel)
    gt_surface = r201_surface(gt)
    gt_surface_distance = ndimage.distance_transform_edt(~gt_surface) if gt_surface.any() else None
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


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def quick_metrics(pred: np.ndarray, gt: np.ndarray, gt_cache: GtCache, boundary_kernel: int) -> dict[str, float]:
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


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def border_contacts(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    return int(mask[0, :].any()) + int(mask[-1, :].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def local_background_features(anchor: np.ndarray, trial: np.ndarray, cut: np.ndarray, gt: np.ndarray, gt_cache: GtCache, pad: int) -> dict[str, float]:
    local_slice = bbox_slice(cut, pad)
    if local_slice is None:
        return {}
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
    cut_labels_after = np.unique(bg_labels_after[cut_l])
    cut_labels_after = cut_labels_after[cut_labels_after > 0]
    cut_bg_component = np.isin(bg_labels_after, cut_labels_after) if cut_labels_after.size else np.zeros_like(bg_after, dtype=bool)
    crop_area = int(cut_l.size)
    cut_area = int(cut_l.sum())
    channel_area = int(cut_bg_component.sum())
    return {
        "r216_crop_area": float(crop_area),
        "r216_cut_area": float(cut_area),
        "r216_cut_frac_crop": safe_divide(cut_area, crop_area),
        "r216_bg_components_before": float(bg_n_before),
        "r216_bg_components_after": float(bg_n_after),
        "r216_bg_component_count_delta": float(bg_n_after - bg_n_before),
        "r216_cut_adjacent_bg_components_before": float(adjacent_labels.size),
        "r216_cut_adjacent_bg_frac_before": safe_divide(float(np.logical_and(cut_ring, bg_before).sum()), float(cut_ring.sum())),
        "r216_cut_component_bg_area_after": float(channel_area),
        "r216_cut_component_bg_frac_crop_after": safe_divide(channel_area, crop_area),
        "r216_cut_component_border_contacts_after": float(border_contacts(cut_bg_component)),
        "r216_bg_channel_proxy": float(
            border_contacts(cut_bg_component) >= 2
            and safe_divide(channel_area, crop_area) >= 0.20
            and safe_divide(float(np.logical_and(cut_ring, bg_before).sum()), float(cut_ring.sum())) >= 0.35
        ),
        "r216_cut_gt_fg_frac": safe_divide(float(np.logical_and(cut_l, gt_l).sum()), cut_area),
        "r216_cut_gt_gap_frac": safe_divide(float(np.logical_and(cut_l, gt_gap_l).sum()), cut_area),
    }


def seam_score(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray) -> np.ndarray:
    dist_in = ndimage.distance_transform_edt(anchor)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0.0:
        grad = grad / float(grad.max())
    ring = np.logical_and(ndimage.binary_dilation(cut, structure=np.ones((5, 5), dtype=bool)), ~cut)
    ring_bg = np.logical_and(ring, ~anchor)
    bg_near = ndimage.distance_transform_edt(~ring_bg)
    raw = -0.55 * dist_in + 0.30 * grad - 0.15 * bg_near
    out = np.full_like(image, -np.inf, dtype=np.float32)
    out[cut] = raw[cut]
    return out


def soft_action_cut(image: np.ndarray, anchor: np.ndarray, raw_cut: np.ndarray, frac: float, min_cut_area: int) -> np.ndarray:
    cut = np.logical_and(raw_cut, anchor)
    n = int(cut.sum())
    if n < min_cut_area:
        return np.zeros_like(cut, dtype=bool)
    keep_n = max(min_cut_area, int(round(n * frac)))
    keep_n = min(n, keep_n)
    scores = seam_score(image, anchor, cut)
    ys, xs = np.where(cut)
    vals = scores[ys, xs]
    order = np.argsort(vals)[::-1][:keep_n]
    out = np.zeros_like(cut, dtype=bool)
    out[ys[order], xs[order]] = True
    out = ndimage.binary_opening(out, structure=np.ones((2, 2), dtype=bool))
    if int(out.sum()) < min_cut_area:
        out = np.zeros_like(cut, dtype=bool)
        out[ys[order], xs[order]] = True
    return out


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
    action_fracs = parse_float_list(args.action_fracs)
    for index, name in enumerate(tqdm(source_names(args), desc=f"r216/soft-seam/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            continue
        image, _instance, gt, anchor = load_case(args, name)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        old = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        raw_cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
        for rank, raw_cut in enumerate(raw_cuts[: args.max_components_per_image], start=1):
            for action_frac in action_fracs:
                cut = soft_action_cut(image, anchor, raw_cut, action_frac, args.min_cut_area)
                cut = np.logical_and(cut, anchor)
                if int(cut.sum()) < args.min_cut_area:
                    continue
                trial = np.logical_and(anchor, ~cut)
                new = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                row: dict[str, Any] = {
                    "run_id": "R216-soft-seam-action-candidate-diagnostic",
                    "split": args.split,
                    "image": name,
                    "candidate_rank": float(rank),
                    "action_frac": float(action_frac),
                }
                row.update(local_background_features(anchor, trial, cut, gt, gt_cache, args.crop_pad))
                for key, value in old.items():
                    row[f"anchor_{key}"] = value
                for key, value in new.items():
                    row[f"candidate_{key}"] = value
                    row[f"delta_{key}"] = metric_delta(new, old, key)
                row["r216_quick_useful"] = float(is_quick_useful(row))
                row["r216_hard_risk"] = float(is_hard_risk(row))
                row["r216_overerosion_proxy"] = float(float(row.get("r216_cut_gt_fg_frac") or 0.0) > 0.5)
                row["r216_safe_channel_proxy_positive"] = float(
                    float(row.get("r216_bg_channel_proxy") or 0.0) > 0.5
                    and float(row.get("r216_cut_gt_fg_frac") or 0.0) <= 0.5
                    and float(row.get("r216_cut_gt_gap_frac") or 0.0) >= 0.5
                )
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
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "None", "nan"):
            continue
        value = float(value)
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_quick_useful": int(sum(float(row.get("r216_quick_useful") or 0.0) > 0.5 for row in rows)),
        "num_hard_risk": int(sum(float(row.get("r216_hard_risk") or 0.0) > 0.5 for row in rows)),
        "num_safe_channel_proxy_positive": int(sum(float(row.get("r216_safe_channel_proxy_positive") or 0.0) > 0.5 for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_cut_gt_fg_frac": mean_value(rows, "r216_cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "r216_cut_gt_gap_frac"),
    }


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_frac: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_frac.setdefault(f"{float(row['action_frac']):.2f}", []).append(row)
    safe = [row for row in rows if float(row.get("r216_safe_channel_proxy_positive") or 0.0) > 0.5]
    return {
        "run_id": "R216-soft-seam-action-candidate-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "purpose": "train/val-only audit of softer seam actions for reducing over-erosion risk",
        "anchor_exp": args.anchor_exp,
        "num_candidate_action_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "overall": summarize_group(rows),
        "by_action_frac": {key: summarize_group(value) for key, value in sorted(by_frac.items())},
        "safe_channel_proxy_positive": summarize_group(safe),
        "warning": "candidate-action train/val diagnostic only; not R201 mask-level evidence and not clean-test-v2 evidence",
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
