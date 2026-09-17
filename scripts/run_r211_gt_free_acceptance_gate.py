#!/usr/bin/env python3
"""R211 GT-free acceptance gate for R210-F1 neck candidates.

This is a train/val-only development audit. It reuses the R210-F1 neck
candidate generator, but its accept/reject policy uses only inference-time
features from the image, anchor mask, and candidate geometry. Ground truth is
used only after prediction for R201 metric evaluation.
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

from run_r201_unified_eval import build_gt_cache, compute_metrics
from run_r209_component_preserving_feasibility_audit import analyze_component_matches
from run_r210_f1_neck_candidate_gate import candidate_components, read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R211 GT-free acceptance gate on train/val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r211_gt_free_acceptance_gate")
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r211_gt_free_acceptance_gate_val_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r211_gt_free_acceptance_gate_val_per_image.csv"))
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.004)
    parser.add_argument("--max-candidates-generated", type=int, default=16)
    parser.add_argument("--max-components-per-image", type=int, default=5)
    parser.add_argument("--min-split-gain", type=int, default=1)
    parser.add_argument("--max-split-gain", type=int, default=4)
    parser.add_argument("--min-piece-area", type=int, default=18)
    parser.add_argument("--min-piece-frac", type=float, default=0.015)
    parser.add_argument("--max-cut-frac-component", type=float, default=0.035)
    parser.add_argument("--min-slenderness", type=float, default=1.7)
    parser.add_argument("--max-fill", type=float, default=0.72)
    parser.add_argument("--min-gradient-ratio", type=float, default=0.92)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--skip-missing-anchor", action="store_true")
    return parser.parse_args()


def find_anchor(args: argparse.Namespace, name: str) -> Path:
    path = anchor_path(args, name)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bbox_features(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"bbox_h": 0.0, "bbox_w": 0.0, "slenderness": 0.0, "fill": 0.0}
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    area = int(mask.sum())
    return {
        "bbox_h": float(h),
        "bbox_w": float(w),
        "slenderness": float(max(h, w) / max(1, min(h, w))),
        "fill": float(area / max(1, h * w)),
    }


def component_merge_count(mask: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> int:
    return int(analyze_component_matches(mask, instance, min_overlap_frac)["merged_pred_components"])


def component_info(anchor: np.ndarray, cut: np.ndarray) -> tuple[np.ndarray, int, int]:
    labels, n_labels = ndimage.label(anchor, structure=np.ones((3, 3), dtype=np.uint8))
    ids, counts = np.unique(labels[cut & (labels > 0)], return_counts=True)
    if ids.size == 0:
        return np.zeros_like(anchor, dtype=bool), 0, int(n_labels)
    comp_id = int(ids[int(np.argmax(counts))])
    return labels == comp_id, comp_id, int(n_labels)


def split_features(current: np.ndarray, cut: np.ndarray, image: np.ndarray) -> dict[str, float]:
    comp, comp_id, before_n = component_info(current, cut)
    cut = np.logical_and(cut, comp)
    comp_area = int(comp.sum())
    cut_area = int(cut.sum())
    bbox = bbox_features(cut)

    trial = np.logical_and(current, ~cut)
    _, after_n = ndimage.label(trial, structure=np.ones((3, 3), dtype=np.uint8))
    split_gain = int(after_n - before_n)

    pieces, piece_n = ndimage.label(np.logical_and(comp, ~cut), structure=np.ones((3, 3), dtype=np.uint8))
    piece_areas = [int((pieces == idx).sum()) for idx in range(1, piece_n + 1)]
    meaningful_piece_areas = [
        area for area in piece_areas if area >= max(1, min(comp_area - cut_area, comp_area) * 0.005)
    ]
    meaningful_piece_areas.sort(reverse=True)
    min_piece_area = float(min(meaningful_piece_areas[:2])) if len(meaningful_piece_areas) >= 2 else 0.0
    min_piece_frac = float(min_piece_area / max(1, comp_area))

    grad_y, grad_x = np.gradient(image.astype(np.float32))
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    comp_grad = grad[comp]
    cut_grad = grad[cut]
    grad_ratio = safe_mean([float(cut_grad.mean())]) / max(1e-6, safe_mean([float(np.percentile(comp_grad, 60))]) if comp_grad.size else 0.0)

    dist = ndimage.distance_transform_edt(comp)
    cut_dist = dist[cut]
    comp_dist = dist[comp]
    dist_ratio = safe_mean([float(cut_dist.mean())]) / max(1e-6, safe_mean([float(np.percentile(comp_dist, 50))]) if comp_dist.size else 0.0)

    return {
        "component_id": float(comp_id),
        "component_area": float(comp_area),
        "cut_area": float(cut_area),
        "cut_frac_component": float(cut_area / max(1, comp_area)),
        "split_gain": float(split_gain),
        "piece_count": float(piece_n),
        "min_piece_area": min_piece_area,
        "min_piece_frac": min_piece_frac,
        "gradient_ratio": float(grad_ratio),
        "dist_ratio": float(dist_ratio),
        **bbox,
    }


def gt_free_accept_candidates(
    args: argparse.Namespace,
    anchor: np.ndarray,
    image: np.ndarray,
    candidates: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    current = anchor.copy()
    accepted_cut = np.zeros_like(anchor, dtype=bool)
    max_cut_pixels = int(max(args.min_cut_area, round(float(anchor.sum()) * args.max_cut_frac)))
    reasons: set[str] = set()
    tried = 0
    accepted_features: list[dict[str, float]] = []
    candidate_rows: list[dict[str, Any]] = []

    for raw_cut in candidates[: args.max_components_per_image]:
        tried += 1
        cut = np.logical_and(raw_cut, current)
        area = int(cut.sum())
        if area < args.min_cut_area:
            reasons.add("too_small")
            continue
        if int(accepted_cut.sum()) + area > max_cut_pixels:
            reasons.add("max_cut_frac")
            continue

        feat = split_features(current, cut, image)
        checks = {
            "split_gain": args.min_split_gain <= feat["split_gain"] <= args.max_split_gain,
            "piece_area": feat["min_piece_area"] >= args.min_piece_area and feat["min_piece_frac"] >= args.min_piece_frac,
            "component_cut_frac": feat["cut_frac_component"] <= args.max_cut_frac_component,
            "neck_shape": feat["slenderness"] >= args.min_slenderness or feat["fill"] <= args.max_fill,
            "gradient_support": feat["gradient_ratio"] >= args.min_gradient_ratio,
        }
        candidate_rows.append(
            {
                **feat,
                **{f"check_{key}": float(ok) for key, ok in checks.items()},
                "check_all": float(all(checks.values())),
                "would_exceed_max_cut_frac": float(int(accepted_cut.sum()) + area > max_cut_pixels),
            }
        )
        if all(checks.values()):
            accepted_cut |= cut
            current = np.logical_and(current, ~cut)
            accepted_features.append(feat)
        else:
            reasons.update(key for key, ok in checks.items() if not ok)

    accepted = bool(accepted_cut.any())
    info: dict[str, Any] = {
        "accepted": float(accepted),
        "reject_reason": "accepted" if accepted else ";".join(sorted(reasons or {"no_candidate"})),
        "num_candidates": float(len(candidates)),
        "num_candidates_tried": float(tried),
        "accepted_cut_pixels": float(accepted_cut.sum()),
        "accepted_components": float(len(accepted_features)),
    }
    for key in [
        "cut_area",
        "cut_frac_component",
        "split_gain",
        "min_piece_area",
        "min_piece_frac",
        "slenderness",
        "fill",
        "gradient_ratio",
        "dist_ratio",
    ]:
        vals = [float(feat[key]) for feat in accepted_features if key in feat]
        info[f"accepted_mean_{key}"] = float(np.mean(vals)) if vals else None
        info[f"accepted_max_{key}"] = float(np.max(vals)) if vals else None
    return current if accepted else anchor, info, candidate_rows


def metric_delta(new: dict[str, float | None], old: dict[str, float | None], key: str) -> float | None:
    if old.get(key) is None or new.get(key) is None:
        return None
    return float(new[key]) - float(old[key])


def evaluate_one(
    args: argparse.Namespace,
    name: str,
    gt_cache: Any,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    anchor = resize_like(read_mask(find_anchor(args, name)), gt.shape)
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    candidates = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
    pred, gate_info, candidate_rows = gt_free_accept_candidates(args, anchor, image, candidates)

    old = compute_metrics(anchor, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    new = compute_metrics(pred, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    old_merge = component_merge_count(anchor, instance, args.min_overlap_frac)
    new_merge = component_merge_count(pred, instance, args.min_overlap_frac)
    row: dict[str, Any] = {
        "image": name,
        **gate_info,
        **{f"anchor_{k}": v for k, v in old.items()},
        **{f"r211_{k}": v for k, v in new.items()},
        "anchor_instance_merge_count": float(old_merge),
        "r211_instance_merge_count": float(new_merge),
        "delta_instance_merge_count": float(new_merge - old_merge),
    }
    for key in old:
        row[f"delta_{key}"] = metric_delta(new, old, key)
    for idx, cand_row in enumerate(candidate_rows, start=1):
        cand_row["image"] = name
        cand_row["candidate_rank"] = idx
    return pred, row, candidate_rows


def mean_record(records: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in records for key in row if key != "image"})
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [row.get(key) for row in records]
        finite = [float(v) for v in vals if isinstance(v, (float, int))]
        out[key] = float(np.mean(finite)) if finite else None
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gate_pass(mean: dict[str, float | None]) -> bool:
    return bool(
        (mean.get("accepted") or 0.0) > 0.0
        and (mean.get("delta_dice") or 0.0) >= -0.0015
        and (mean.get("delta_iou") or 0.0) >= -0.0025
        and (mean.get("delta_recall") or 0.0) >= -0.0025
        and (mean.get("delta_component_count_mae") or 0.0) <= 0.0
        and (
            (mean.get("delta_gap_region_fp_rate") or 0.0) < 0.0
            or (mean.get("delta_instance_merge_count") or 0.0) < 0.0
            or (mean.get("delta_boundary_iou") or 0.0) > 0.0
        )
    )


def main() -> None:
    args = parse_args()
    gt_dir = args.raw_root / args.dataset / f"{args.split}_labels"
    cache = build_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.split / "masks"

    missing: list[str] = []
    for name in tqdm(case_names, desc=f"r211/{args.split}"):
        if args.skip_missing_anchor and not anchor_path(args, name).exists():
            missing.append(name)
            continue
        pred, row, cand_rows = evaluate_one(args, name, cache[name])
        records.append(row)
        candidate_records.extend(cand_rows)
        write_mask(output_dir / name, pred)

    mean = mean_record(records)
    summary = {
        "run_id": "R211",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "purpose": "GT-free acceptance gate audit for R210-F1 neck candidates",
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "num_evaluated": len(records),
        "num_missing": len(missing),
        "missing": missing,
        "gate_pass": gate_pass(mean),
        "gate_type": "gt_free_inference_features_only",
        "settings": {
            "dist_percentile": args.dist_percentile,
            "min_cut_area": args.min_cut_area,
            "max_cut_frac": args.max_cut_frac,
            "max_candidates_generated": args.max_candidates_generated,
            "max_components_per_image": args.max_components_per_image,
            "min_split_gain": args.min_split_gain,
            "max_split_gain": args.max_split_gain,
            "min_piece_area": args.min_piece_area,
            "min_piece_frac": args.min_piece_frac,
            "max_cut_frac_component": args.max_cut_frac_component,
            "min_slenderness": args.min_slenderness,
            "max_fill": args.max_fill,
            "min_gradient_ratio": args.min_gradient_ratio,
        },
        "mean": mean,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.per_image_csv, records)
    if args.candidate_csv is not None:
        write_csv(args.candidate_csv, candidate_records)
    print(json.dumps({"gate_pass": summary["gate_pass"], "mean": mean}, indent=2), flush=True)


if __name__ == "__main__":
    main()
