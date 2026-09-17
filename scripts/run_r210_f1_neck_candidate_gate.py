#!/usr/bin/env python3
"""R210-F1 deterministic neck-candidate gate.

Train/val-only audit for component-preserving split candidates. It generates
thin cut candidates from R110 anchor geometry and accepts them only with the
same R201 metric guardrails used by R210-F0. clean-test-v2 is not used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from run_r201_unified_eval import build_gt_cache, compute_metrics
from run_r209_component_preserving_feasibility_audit import analyze_component_matches
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R210-F1 deterministic neck candidate gate.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r210_f1_neck_candidate_gate")
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r210_f1_neck_candidate_gate_val_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r210_f1_neck_candidate_gate_val_per_image.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dist-percentiles", default="6,8,10,12,15")
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.006)
    parser.add_argument("--max-components-per-image", type=int, default=12)
    parser.add_argument("--max-candidates-generated", type=int, default=48)
    parser.add_argument("--use-instance-merge-in-inner-gate", action="store_true")
    parser.add_argument("--dice-tol", type=float, default=0.003)
    parser.add_argument("--iou-tol", type=float, default=0.003)
    parser.add_argument("--recall-tol", type=float, default=0.005)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list[Any]:
    return [cast(x) for x in text.split(",") if x.strip()]


def read_instance(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def find_anchor(args: argparse.Namespace, name: str) -> Path:
    path = args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def component_merge_count(mask: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> int:
    return int(analyze_component_matches(mask, instance, min_overlap_frac)["merged_pred_components"])


def metric_delta(new: dict[str, float | None], old: dict[str, float | None], key: str) -> float:
    return float(new.get(key) or 0.0) - float(old.get(key) or 0.0)


def safe_divide(numer: float, denom: float, default: float = 1.0) -> float:
    return float(numer / denom) if denom > 0 else default


def quick_metrics(pred: np.ndarray, gt: np.ndarray, gt_cache: Any, args: argparse.Namespace) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    pred_boundary = boundary_band(pred, max(1, args.boundary_kernel // 2))
    gt_boundary = gt_cache.boundary
    b_tp = float(np.logical_and(pred_boundary, gt_boundary).sum())
    b_union = float(np.logical_or(pred_boundary, gt_boundary).sum())
    b_fp = float(np.logical_and(pred_boundary, ~gt_boundary).sum())
    b_fn = float(np.logical_and(gt_boundary, ~pred_boundary).sum())
    _, pred_n = ndimage.label(pred)
    gt_n = gt_cache.component_count
    gap_pixels = float(gt_cache.gap_region.sum())
    gap_fp = float(np.logical_and(pred, gt_cache.gap_region).sum())
    return {
        "dice": safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": safe_divide(tp, tp + fp + fn),
        "recall": safe_divide(tp, tp + fn),
        "boundary_iou": safe_divide(b_tp, b_union, default=0.0),
        "boundary_f1": safe_divide(2.0 * b_tp, 2.0 * b_tp + b_fp + b_fn, default=0.0),
        "gap_region_fp_rate": safe_divide(gap_fp, gap_pixels, default=0.0),
        "component_count_mae": float(abs(int(pred_n) - int(gt_n))),
    }


def boundary_band(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def candidate_components(anchor: np.ndarray, image: np.ndarray, dist_percentile: float, min_cut_area: int) -> list[np.ndarray]:
    labels, n_labels = ndimage.label(anchor, structure=np.ones((3, 3), dtype=np.uint8))
    all_components: list[tuple[float, np.ndarray]] = []
    grad_y, grad_x = np.gradient(image.astype(np.float32))
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    for comp_id in range(1, n_labels + 1):
        comp = labels == comp_id
        area = int(comp.sum())
        if area < 64:
            continue
        dist = ndimage.distance_transform_edt(comp)
        vals = dist[comp]
        if vals.size == 0:
            continue
        thr = float(np.percentile(vals, dist_percentile))
        neck = np.logical_and(comp, dist <= max(1.0, thr))
        # Favor interior neck pixels with image gradient support, but keep the
        # rule deterministic and conservative.
        gvals = grad[neck]
        if gvals.size:
            neck = np.logical_and(neck, grad >= float(np.percentile(gvals, 55)))
        neck = ndimage.binary_opening(neck, structure=np.ones((2, 2), dtype=bool))
        neck_labels, neck_n = ndimage.label(neck, structure=np.ones((3, 3), dtype=np.uint8))
        for neck_id in range(1, neck_n + 1):
            cut = neck_labels == neck_id
            cut_area = int(cut.sum())
            if cut_area < min_cut_area:
                continue
            ys, xs = np.where(cut)
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            slender = max(h, w) / max(1, min(h, w))
            fill = cut_area / max(1, h * w)
            touches_edge = bool(np.logical_and(cut, boundary_band(comp, 1)).any())
            if touches_edge and (slender >= 1.8 or fill <= 0.55):
                score = float(cut_area) * slender
                all_components.append((score, cut))
    all_components.sort(key=lambda x: x[0], reverse=True)
    return [cut for _, cut in all_components]


def accept_candidates(
    args: argparse.Namespace,
    anchor: np.ndarray,
    gt: np.ndarray,
    instance: np.ndarray,
    candidates: list[np.ndarray],
    gt_cache: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    old_quick = quick_metrics(anchor, gt, gt_cache, args)
    old = compute_metrics(anchor, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    old_merge = component_merge_count(anchor, instance, args.min_overlap_frac)
    inner_old_merge = old_merge if args.use_instance_merge_in_inner_gate else 0
    current = anchor.copy()
    accepted_cut = np.zeros_like(anchor, dtype=bool)
    final = old
    final_merge = old_merge
    max_cut = int(max(args.min_cut_area, round(float(anchor.sum()) * args.max_cut_frac)))
    reject_reasons: set[str] = set()
    tried = 0
    for cut in candidates[: args.max_components_per_image]:
        tried += 1
        cut = np.logical_and(cut, current)
        if int(cut.sum()) < args.min_cut_area:
            reject_reasons.add("too_small")
            continue
        if int(accepted_cut.sum()) + int(cut.sum()) > max_cut:
            reject_reasons.add("max_cut_frac")
            continue
        trial = np.logical_and(current, ~cut)
        trial_metrics = quick_metrics(trial, gt, gt_cache, args)
        trial_merge = component_merge_count(trial, instance, args.min_overlap_frac) if args.use_instance_merge_in_inner_gate else inner_old_merge
        checks = {
            "dice_safe": metric_delta(trial_metrics, old_quick, "dice") >= -args.dice_tol,
            "iou_safe": metric_delta(trial_metrics, old_quick, "iou") >= -args.iou_tol,
            "recall_safe": metric_delta(trial_metrics, old_quick, "recall") >= -args.recall_tol,
            "component_count_safe": float(trial_metrics["component_count_mae"] or 0.0) <= float(old_quick["component_count_mae"] or 0.0),
            "has_diagnostic_gain": (
                metric_delta(trial_metrics, old_quick, "boundary_iou") > 0.0
                or metric_delta(trial_metrics, old_quick, "boundary_f1") > 0.0
                or metric_delta(trial_metrics, old_quick, "gap_region_fp_rate") < 0.0
                or trial_merge < inner_old_merge
            ),
        }
        if all(checks.values()):
            accepted_cut |= cut
            current = trial
            final_merge = trial_merge if args.use_instance_merge_in_inner_gate else old_merge
        else:
            reject_reasons.update(key for key, ok in checks.items() if not ok)

    accepted = bool(accepted_cut.any())
    final = compute_metrics(current if accepted else anchor, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    final_merge = component_merge_count(current if accepted else anchor, instance, args.min_overlap_frac)
    info: dict[str, Any] = {
        **{f"anchor_{k}": v for k, v in old.items()},
        **{f"r210_f1_{k}": v for k, v in final.items()},
        "accepted": float(accepted),
        "reject_reason": "accepted" if accepted else ";".join(sorted(reject_reasons or {"no_candidate"})),
        "num_candidates": float(len(candidates)),
        "num_candidates_tried": float(tried),
        "accepted_cut_pixels": float(accepted_cut.sum()),
        "anchor_instance_merge_count": float(old_merge),
        "r210_f1_instance_merge_count": float(final_merge),
    }
    for key in old:
        if isinstance(old[key], (float, int)) or old[key] is None:
            info[f"delta_{key}"] = None if old[key] is None or final[key] is None else float(final[key]) - float(old[key])
    info["delta_instance_merge_count"] = float(final_merge - old_merge)
    return current if accepted else anchor, info


def mean_record(records: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({k for r in records for k in r if k != "image"})
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [r.get(key) for r in records]
        finite = [float(v) for v in vals if isinstance(v, (float, int))]
        out[key] = float(np.mean(finite)) if finite else None
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gate_pass(best: dict[str, Any]) -> bool:
    mean = best["mean"]
    return bool(
        (mean.get("accepted") or 0.0) > 0.0
        and (mean.get("delta_dice") or 0.0) >= -0.003
        and (mean.get("delta_iou") or 0.0) >= -0.003
        and (mean.get("delta_recall") or 0.0) >= -0.005
        and (mean.get("delta_component_count_mae") or 0.0) <= 0.0
        and (
            (mean.get("delta_gap_region_fp_rate") or 0.0) < 0.0
            or (mean.get("delta_instance_merge_count") or 0.0) < 0.0
            or (mean.get("delta_boundary_iou") or 0.0) > 0.0
            or (mean.get("delta_boundary_f1") or 0.0) > 0.0
        )
    )


def main() -> None:
    args = parse_args()
    gt_dir = args.raw_root / args.dataset / f"{args.split}_labels"
    gt_cache = build_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    all_grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_records: list[dict[str, Any]] = []
    best_preds: list[tuple[str, np.ndarray]] = []

    for percentile in parse_nums(args.dist_percentiles, float):
        records: list[dict[str, Any]] = []
        preds: list[tuple[str, np.ndarray]] = []
        for name in tqdm(case_names, desc=f"r210_f1/p{percentile:g}"):
            instance = read_instance(gt_dir / name)
            gt = instance > 0
            anchor = resize_like(read_mask(find_anchor(args, name)), gt.shape)
            image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
            cuts = candidate_components(anchor, image, percentile, args.min_cut_area)[: args.max_candidates_generated]
            pred, rec = accept_candidates(args, anchor, gt, instance, cuts, gt_cache[name])
            rec["image"] = name
            records.append(rec)
            preds.append((name, pred))
        mean = mean_record(records)
        item = {"dist_percentile": percentile, "mean": mean}
        all_grid.append(item)
        score = (
            mean.get("accepted") or 0.0,
            -(mean.get("delta_component_count_mae") or 0.0),
            -(mean.get("delta_gap_region_fp_rate") or 0.0),
            -(mean.get("delta_instance_merge_count") or 0.0),
            mean.get("delta_boundary_iou") or 0.0,
            mean.get("delta_dice") or -999.0,
        )
        if best is None or score > best["_score"]:
            best = {**item, "_score": score}
            best_records = records
            best_preds = preds

    assert best is not None
    best.pop("_score", None)
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.split / "masks"
    for name, pred in best_preds:
        write_mask(output_dir / name, pred)
    summary = {
        "run_id": "R210-F1",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "num_evaluated": len(best_records),
        "val_gate_pass": gate_pass(best),
        "best": best,
        "grid": all_grid,
        "output_exp": args.output_exp,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.per_image_csv, best_records)
    print(json.dumps({"val_gate_pass": summary["val_gate_pass"], "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
