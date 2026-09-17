#!/usr/bin/env python3
"""R225 GT-free pseudo-lobe pairwise seam candidate diagnostic.

R224 showed a strong oracle upper bound when seam candidates are generated
between a pair of true GT instances. R225 removes GT from candidate generation:
large R110 anchor components are split into pseudo-lobes using distance-map
peaks, and seam candidates are generated only between pseudo-lobe pairs.

This script writes JSON/CSV diagnostics only. It uses GT only for evaluating
candidate useful/risk labels on original train/val. It must not be used on
clean-test-v2 for tuning or model selection.
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

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, is_hard_risk, is_quick_useful, quick_metrics
from run_r209_component_preserving_feasibility_audit import read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R225 pseudo-lobe pairwise seam candidates.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.006)
    parser.add_argument("--max-lobes-per-component", type=int, default=5)
    parser.add_argument("--max-pairs-per-component", type=int, default=4)
    parser.add_argument("--max-fallback-seed-pixels", type=int, default=2500)
    parser.add_argument("--seed-min-distance", type=int, default=14)
    parser.add_argument("--seed-dist-percentile", type=float, default=80.0)
    parser.add_argument("--seam-radii", default="2,3,4")
    parser.add_argument("--action-fracs", default="0.25,0.50,0.75")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r225_pseudo_lobe_pairwise_seam_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r225_pseudo_lobe_pairwise_seam_val_summary.json"))
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)),
    )


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def seed_points(component: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int, float]]:
    dist = ndimage.distance_transform_edt(component)
    if not component.any():
        return []
    vals = dist[component]
    thr = max(1.0, float(np.percentile(vals, args.seed_dist_percentile)))
    max_filtered = ndimage.maximum_filter(dist, size=max(3, 2 * args.seed_min_distance + 1))
    peaks = component & (dist >= thr) & (dist == max_filtered)
    labels, n_labels = component_labels(peaks)
    seeds: list[tuple[int, int, float]] = []
    for label_id in range(1, n_labels + 1):
        region = labels == label_id
        if not region.any():
            continue
        ys, xs = np.where(region)
        scores = dist[ys, xs]
        best = int(np.argmax(scores))
        seeds.append((int(ys[best]), int(xs[best]), float(scores[best])))
    if len(seeds) < 2:
        # Fallback: choose separated high-distance pixels greedily.
        ys, xs = np.where(component)
        order = np.argsort(dist[ys, xs])[::-1]
        for idx in order[: args.max_fallback_seed_pixels]:
            y, x = int(ys[idx]), int(xs[idx])
            score = float(dist[y, x])
            if score < 1.0:
                break
            if all((y - sy) ** 2 + (x - sx) ** 2 >= args.seed_min_distance**2 for sy, sx, _ in seeds):
                seeds.append((y, x, score))
            if len(seeds) >= args.max_lobes_per_component:
                break
    seeds.sort(key=lambda item: item[2], reverse=True)
    return seeds[: args.max_lobes_per_component]


def pseudo_lobes(component: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    seeds = seed_points(component, args)
    labels = np.zeros(component.shape, dtype=np.int32)
    if len(seeds) < 2:
        return labels, seeds
    ys, xs = np.where(component)
    seed_y = np.asarray([s[0] for s in seeds], dtype=np.float32)
    seed_x = np.asarray([s[1] for s in seeds], dtype=np.float32)
    # Euclidean assignment is a cheap pseudo-instance approximation.
    d2 = (ys[:, None].astype(np.float32) - seed_y[None, :]) ** 2 + (xs[:, None].astype(np.float32) - seed_x[None, :]) ** 2
    nearest = np.argmin(d2, axis=1) + 1
    labels[ys, xs] = nearest.astype(np.int32)
    return labels, seeds


def border_contacts(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    return int(mask[0, :].any()) + int(mask[-1, :].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def lobe_pair_candidates(component: np.ndarray, lobe_labels: np.ndarray, image: np.ndarray, args: argparse.Namespace) -> list[tuple[int, int, int, np.ndarray]]:
    dist = ndimage.distance_transform_edt(component)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    pairs: list[tuple[float, int, int]] = []
    lobe_ids = [int(v) for v in np.unique(lobe_labels) if int(v) > 0]
    for i, a in enumerate(lobe_ids):
        lobe_a = lobe_labels == a
        for b in lobe_ids[i + 1 :]:
            lobe_b = lobe_labels == b
            ring = ndimage.binary_dilation(lobe_a, structure=np.ones((5, 5), dtype=bool)) & ndimage.binary_dilation(lobe_b, structure=np.ones((5, 5), dtype=bool)) & component
            if not ring.any():
                continue
            score = float(ring.sum()) + float(grad[ring].mean() if ring.any() else 0.0) - 0.25 * float(dist[ring].mean())
            pairs.append((score, a, b))
    pairs.sort(reverse=True)
    out: list[tuple[int, int, int, np.ndarray]] = []
    for _score, a, b in pairs[: args.max_pairs_per_component]:
        lobe_a = lobe_labels == a
        lobe_b = lobe_labels == b
        for radius in parse_int_list(args.seam_radii):
            structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
            cut = component & ndimage.binary_dilation(lobe_a, structure=structure) & ndimage.binary_dilation(lobe_b, structure=structure)
            cut &= dist <= max(1.0, float(np.percentile(dist[component], 35)))
            cut = ndimage.binary_opening(cut, structure=np.ones((2, 2), dtype=bool))
            if int(cut.sum()) >= args.min_cut_area:
                out.append((a, b, radius, cut.astype(bool)))
    return out


def trim_cut_by_score(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray, action_frac: float, min_cut_area: int) -> np.ndarray:
    n = int(cut.sum())
    if n < min_cut_area:
        return np.zeros_like(cut, dtype=bool)
    dist_in = ndimage.distance_transform_edt(anchor)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0:
        grad = grad / float(grad.max())
    score = -0.75 * dist_in + 0.35 * grad
    ys, xs = np.where(cut)
    keep_n = min(n, max(min_cut_area, int(round(n * action_frac))))
    order = np.argsort(score[ys, xs])[::-1][:keep_n]
    out = np.zeros_like(cut, dtype=bool)
    out[ys[order], xs[order]] = True
    out = ndimage.binary_opening(out, structure=np.ones((2, 2), dtype=bool))
    if int(out.sum()) < min_cut_area:
        out = np.zeros_like(cut, dtype=bool)
        out[ys[order], xs[order]] = True
    return out


def cut_features(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray, instance: np.ndarray, lobe_a: int, lobe_b: int, radius: int, action_frac: float) -> dict[str, float]:
    local_slice = bbox_slice(cut, pad=16)
    if local_slice is None:
        local_anchor = np.zeros((0, 0), dtype=bool)
        local_cut = np.zeros((0, 0), dtype=bool)
        local_instance = np.zeros((0, 0), dtype=np.int32)
    else:
        local_anchor = anchor[local_slice].astype(bool)
        local_cut = cut[local_slice].astype(bool)
        local_instance = instance[local_slice]
    bg_before = ~local_anchor
    bg_after = ~(local_anchor & ~local_cut)
    bg_labels_before, bg_n_before = component_labels(bg_before)
    bg_labels_after, bg_n_after = component_labels(bg_after)
    cut_ring = ndimage.binary_dilation(local_cut, structure=np.ones((3, 3), dtype=bool)) & ~local_cut
    adjacent_bg = np.unique(bg_labels_before[cut_ring & bg_before])
    adjacent_bg = adjacent_bg[adjacent_bg > 0]
    cut_after_labels = np.unique(bg_labels_after[local_cut])
    cut_after_labels = cut_after_labels[cut_after_labels > 0]
    cut_bg_component = np.isin(bg_labels_after, cut_after_labels) if cut_after_labels.size else np.zeros_like(bg_after, dtype=bool)
    area = int(cut.sum())
    ys, xs = np.where(cut)
    h = int(ys.max() - ys.min() + 1) if ys.size else 0
    w = int(xs.max() - xs.min() + 1) if xs.size else 0
    cut_img = image[cut] if area else np.asarray([], dtype=np.float32)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    cut_grad = grad[cut] if area else np.asarray([], dtype=np.float32)
    return {
        "lobe_a": float(lobe_a),
        "lobe_b": float(lobe_b),
        "seam_radius": float(radius),
        "action_frac": float(action_frac),
        "cut_area": float(area),
        "cut_bbox_h": float(h),
        "cut_bbox_w": float(w),
        "cut_slenderness": float(max(h, w) / max(1, min(h, w))) if area else 0.0,
        "cut_fill": safe_divide(float(area), float(max(1, h * w))),
        "cut_mean_image": float(np.mean(cut_img)) if cut_img.size else 0.0,
        "cut_std_image": float(np.std(cut_img)) if cut_img.size else 0.0,
        "cut_mean_grad": float(np.mean(cut_grad)) if cut_grad.size else 0.0,
        "cut_p90_grad": float(np.percentile(cut_grad, 90)) if cut_grad.size else 0.0,
        "local_anchor_frac": safe_divide(float(local_anchor.sum()), float(local_anchor.size)),
        "local_bg_frac": safe_divide(float((~local_anchor).sum()), float(local_anchor.size)),
        "bg_components_before": float(bg_n_before),
        "bg_components_after": float(bg_n_after),
        "bg_component_count_delta": float(bg_n_after - bg_n_before),
        "cut_adjacent_bg_components_before": float(adjacent_bg.size),
        "cut_component_bg_area_after": float(cut_bg_component.sum()),
        "cut_component_bg_frac_after": safe_divide(float(cut_bg_component.sum()), float(local_cut.size)),
        "cut_component_border_contacts_after": float(border_contacts(cut_bg_component)),
        "bg_channel_proxy": float(border_contacts(cut_bg_component) >= 2 and safe_divide(float(cut_bg_component.sum()), float(local_cut.size)) >= 0.15),
        "cut_gt_fg_frac": safe_divide(float(np.logical_and(cut, instance > 0).sum()), float(area)),
        "cut_gt_gap_frac": safe_divide(float(np.logical_and(local_cut, local_instance == 0).sum()), float(max(1, area))),
    }


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    action_fracs = parse_float_list(args.action_fracs)
    for index, name in enumerate(tqdm(case_names, desc=f"r225/pseudo-lobe-seam/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            continue
        image, instance, gt, anchor = load_case(args, name)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        labels, n_labels = component_labels(anchor)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        for comp_id in range(1, n_labels + 1):
            comp = labels == comp_id
            comp_area = int(comp.sum())
            if comp_area < args.min_component_area or comp_area > args.max_component_area:
                continue
            lobe_labels, seeds = pseudo_lobes(comp, args)
            if len(seeds) < 2:
                continue
            for lobe_a, lobe_b, radius, raw_cut in lobe_pair_candidates(comp, lobe_labels, image, args):
                for action_frac in action_fracs:
                    cut = trim_cut_by_score(image, anchor, raw_cut, action_frac, args.min_cut_area)
                    cut = cut & anchor
                    if int(cut.sum()) < args.min_cut_area or int(cut.sum()) > max_pixels:
                        continue
                    trial = anchor & ~cut
                    trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                    row: dict[str, Any] = {
                        "run_id": "R225-pseudo-lobe-pairwise-seam-candidate-diagnostic",
                        "split": args.split,
                        "image": name,
                        "pred_component_id": float(comp_id),
                        "num_pseudo_lobes": float(len(seeds)),
                    }
                    row.update(cut_features(image, anchor, cut, instance, lobe_a, lobe_b, radius, action_frac))
                    for metric, value in anchor_metrics.items():
                        row[f"anchor_{metric}"] = value
                    for metric, value in trial_metrics.items():
                        row[f"candidate_{metric}"] = value
                        row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                    row["r225_quick_useful"] = float(is_quick_useful(row))
                    row["r225_hard_risk"] = float(is_hard_risk(row))
                    row["r225_safe_gap_positive"] = float(row["cut_gt_fg_frac"] <= 0.25 and row["cut_gt_gap_frac"] >= 0.75)
                    row["r225_overerosion_proxy"] = float(row["cut_gt_fg_frac"] > 0.5)
                    rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.output_csv, rows)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summarize(rows, args), indent=2), encoding="utf-8")
    return rows


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
        "num_images": len({row["image"] for row in rows}),
        "num_quick_useful": int(sum(float(row.get("r225_quick_useful") or 0.0) > 0.5 for row in rows)),
        "num_hard_risk": int(sum(float(row.get("r225_hard_risk") or 0.0) > 0.5 for row in rows)),
        "num_safe_gap_positive": int(sum(float(row.get("r225_safe_gap_positive") or 0.0) > 0.5 for row in rows)),
        "num_overerosion_proxy": int(sum(float(row.get("r225_overerosion_proxy") or 0.0) > 0.5 for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
    }


def oracle_best_by_image(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    chosen: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        useful = [row for row in image_rows if float(row.get("r225_quick_useful") or 0.0) > 0.5 and float(row.get("r225_hard_risk") or 0.0) <= 0.5]
        if not useful:
            continue
        useful.sort(
            key=lambda row: (
                -float(row.get("delta_boundary_iou") or 0.0),
                float(row.get("delta_gap_region_fp_rate") or 0.0),
                float(row.get("delta_component_count_mae") or 0.0),
                -float(row.get("delta_dice") or 0.0),
            )
        )
        chosen.append(useful[0])
    return chosen


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    safe_gap = [row for row in rows if float(row.get("r225_safe_gap_positive") or 0.0) > 0.5]
    oracle = oracle_best_by_image(rows)
    return {
        "run_id": "R225-pseudo-lobe-pairwise-seam-candidate-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "train_val_gt_free_candidate_diagnostic",
        "anchor_exp": args.anchor_exp,
        "num_candidate_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "overall": summarize_group(rows),
        "safe_gap_positive": summarize_group(safe_gap),
        "oracle_best_per_image": summarize_group(oracle),
        "warning": "GT is used only for diagnostic labels/metrics; pseudo-lobe candidate generation is GT-free and original-val only.",
    }


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


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    report = summarize(rows, args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(args.output_json), "summary": report}, indent=2))


if __name__ == "__main__":
    main()
