#!/usr/bin/env python3
"""R244-F0 GT-free merge-targeted candidate generator diagnostic.

R242/R243 showed that the broad R239 background-channel candidate distribution
is too noisy. R244-F0 changes the candidate generator: start from thin neck-like
cuts inside R110 connected components, then audit whether local background
connectivity features identify bone-seam candidates without GT.

Candidate generation uses only image + R110 anchor. GT is used only for
original-val audit labels and metrics. This script writes CSV/JSON only and
must not use clean-test-v2 for selection.
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

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, quick_metrics
from audit_r237_component_merge_targeted_candidates import local_bg_connectivity
from run_r209_component_preserving_feasibility_audit import analyze_component_matches, read_instance
from run_r210_f1_neck_candidate_gate import candidate_components
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R244 GT-free merge-targeted neck/background candidates.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dist-percentiles", default="6,8,10,12")
    parser.add_argument("--min-cut-area", type=int, default=3)
    parser.add_argument("--max-cut-frac", type=float, default=0.008)
    parser.add_argument("--max-components-per-image", type=int, default=5)
    parser.add_argument("--max-candidates-generated", type=int, default=64)
    parser.add_argument("--crop-pad", type=int, default=24)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r244_gtfree_merge_targeted_candidates_summary.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r244_gtfree_merge_targeted_candidates.csv"))
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def bbox_stats(mask: np.ndarray) -> dict[str, float]:
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


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def component_match_metrics(mask: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> dict[str, float]:
    match = analyze_component_matches(mask, instance, min_overlap_frac)
    return {
        "pred_component_count_instance_match": float(match["pred_component_count"]),
        "gt_instance_count": float(match["gt_instance_count"]),
        "merged_pred_components": float(match["merged_pred_components"]),
        "gt_instances_covered": float(match["gt_instances_covered"]),
        "gt_instances_multi_covered": float(match["gt_instances_multi_covered"]),
    }


def cut_features(image: np.ndarray, anchor: np.ndarray, component: np.ndarray, cut: np.ndarray) -> dict[str, float]:
    grad_y, grad_x = np.gradient(image.astype(np.float32))
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    if float(grad.max()) > 0:
        grad = grad / float(grad.max())
    dist_in = ndimage.distance_transform_edt(anchor)
    ring = ndimage.binary_dilation(cut, structure=np.ones((5, 5), dtype=bool)) & ~cut
    cut_vals = image[cut] if cut.any() else np.asarray([], dtype=np.float32)
    cut_grad = grad[cut] if cut.any() else np.asarray([], dtype=np.float32)
    cut_dist = dist_in[cut] if cut.any() else np.asarray([], dtype=np.float32)
    out = {
        "component_area": float(component.sum()),
        "cut_area": float(cut.sum()),
        "cut_anchor_area_frac": safe_divide(float(cut.sum()), float(anchor.sum())),
        "cut_mean_image": float(np.mean(cut_vals)) if cut_vals.size else 0.0,
        "cut_std_image": float(np.std(cut_vals)) if cut_vals.size else 0.0,
        "cut_mean_grad": float(np.mean(cut_grad)) if cut_grad.size else 0.0,
        "cut_p90_grad": float(np.percentile(cut_grad, 90)) if cut_grad.size else 0.0,
        "cut_mean_dist_in": float(np.mean(cut_dist)) if cut_dist.size else 0.0,
        "cut_max_dist_in": float(np.max(cut_dist)) if cut_dist.size else 0.0,
        "ring_bg_frac": safe_divide(float((ring & ~anchor).sum()), float(ring.sum())),
        "ring_fg_frac": safe_divide(float((ring & anchor).sum()), float(ring.sum())),
    }
    out.update({f"cut_{key}": value for key, value in bbox_stats(cut).items()})
    out.update({f"component_{key}": value for key, value in bbox_stats(component).items()})
    return out


def is_safe_useful(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(row.get("delta_dice") or 0.0) >= -args.dice_tol
        and float(row.get("delta_iou") or 0.0) >= -args.iou_tol
        and float(row.get("delta_recall") or 0.0) >= -args.recall_tol
        and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
        and float(row.get("cut_gt_fg_frac") or 0.0) <= 0.25
        and float(row.get("cut_gt_gap_frac") or 0.0) >= 0.75
        and (
            float(row.get("delta_boundary_iou") or 0.0) > 0.0
            or float(row.get("delta_boundary_f1") or 0.0) > 0.0
            or float(row.get("delta_gap_region_fp_rate") or 0.0) < 0.0
            or float(row.get("delta_merged_pred_components") or 0.0) < 0.0
        )
    )


def is_risk(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(row.get("cut_gt_fg_frac") or 0.0) > 0.5
        or float(row.get("delta_recall") or 0.0) < -args.recall_tol
        or float(row.get("delta_dice") or 0.0) < -args.dice_tol
        or float(row.get("delta_iou") or 0.0) < -args.iou_tol
        or float(row.get("delta_boundary_iou") or 0.0) < 0.0
        or float(row.get("delta_boundary_f1") or 0.0) < 0.0
    )


def row_key(cut: np.ndarray) -> bytes:
    return np.packbits(cut.astype(bool).ravel()).tobytes()


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    for index, name in enumerate(tqdm(case_names, desc=f"r244/gtfree-merge-targeted/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            skipped.append({"image": name, "reason": "missing_anchor"})
            continue
        try:
            instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
            gt = instance > 0
            image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
            anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
        except Exception as exc:  # noqa: BLE001 - keep diagnostics moving and record the case.
            skipped.append({"image": name, "reason": type(exc).__name__})
            continue
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        anchor_match = component_match_metrics(anchor, instance, args.min_overlap_frac)
        comp_labels, _n_components = component_labels(anchor)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        seen: set[bytes] = set()
        for dist_percentile in parse_float_list(args.dist_percentiles):
            cuts = candidate_components(anchor, image, dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
            for cut_rank, raw_cut in enumerate(cuts, start=1):
                cut = raw_cut & anchor
                if int(cut.sum()) < args.min_cut_area or int(cut.sum()) > max_pixels:
                    continue
                key = row_key(cut)
                if key in seen:
                    continue
                seen.add(key)
                comp_ids = np.unique(comp_labels[cut])
                comp_ids = comp_ids[comp_ids > 0]
                if comp_ids.size == 0:
                    continue
                comp_id = int(comp_ids[np.argmax([(comp_labels[cut] == cid).sum() for cid in comp_ids])])
                component = comp_labels == comp_id
                trial = anchor & ~cut
                trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                trial_match = component_match_metrics(trial, instance, args.min_overlap_frac)
                row: dict[str, Any] = {
                    "run_id": "R244-F0-gtfree-merge-targeted-candidates",
                    "split": args.split,
                    "image": name,
                    "dist_percentile": float(dist_percentile),
                    "candidate_rank": float(cut_rank),
                    "pred_component_id": float(comp_id),
                    "cut_gt_fg_frac": safe_divide(float((cut & gt).sum()), float(cut.sum())),
                    "cut_gt_gap_frac": safe_divide(float((cut & gt_cache.gap_region).sum()), float(cut.sum())),
                }
                row.update(cut_features(image, anchor, component, cut))
                row.update(local_bg_connectivity(anchor, cut, args.crop_pad))
                for metric, value in trial_metrics.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                for metric, value in trial_match.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = float(value) - float(anchor_match[metric])
                row["safe_useful_label"] = float(is_safe_useful(row, args))
                row["risk_label"] = float(is_risk(row, args))
                rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.output_csv, rows)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summarize(rows, skipped), indent=2), encoding="utf-8")
    return rows, skipped


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_safe_useful": int(sum(as_float(row, "safe_useful_label") > 0.5 for row in rows)),
        "num_risk": int(sum(as_float(row, "risk_label") > 0.5 for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_delta_merged_pred_components": mean_value(rows, "delta_merged_pred_components"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
        "mean_cut_area": mean_value(rows, "cut_area"),
        "mean_background_connected_by_cut": mean_value(rows, "background_connected_by_cut"),
    }


def gtfree_clean_subset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if as_float(row, "background_connected_by_cut") < 0.5:
            continue
        if as_float(row, "cut_bg_channel_border_contacts_after") < 2.0:
            continue
        if as_float(row, "cut_bg_channel_frac_after") < 0.15:
            continue
        if as_float(row, "cut_mean_dist_in") > 2.0:
            continue
        if as_float(row, "cut_area") > 24.0:
            continue
        if as_float(row, "cut_fill") > 0.45:
            continue
        selected.append(row)
    return selected


def oracle_best_by_image(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    chosen = []
    for image_rows in grouped.values():
        safe = [row for row in image_rows if as_float(row, "safe_useful_label") > 0.5 and as_float(row, "risk_label") <= 0.5]
        if not safe:
            continue
        safe.sort(
            key=lambda row: (
                as_float(row, "delta_component_count_mae"),
                as_float(row, "delta_merged_pred_components"),
                -as_float(row, "delta_boundary_iou"),
                as_float(row, "delta_gap_region_fp_rate"),
            )
        )
        chosen.append(safe[0])
    return chosen


def summarize(rows: list[dict[str, Any]], skipped: list[dict[str, str]]) -> dict[str, Any]:
    safe = [row for row in rows if as_float(row, "safe_useful_label") > 0.5]
    risk = [row for row in rows if as_float(row, "risk_label") > 0.5]
    clean = gtfree_clean_subset(rows)
    oracle = oracle_best_by_image(rows)
    return {
        "run_id": "R244-F0-gtfree-merge-targeted-candidates",
        "dataset": "TSRS_RSNA-Epiphysis",
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "original_val_gtfree_candidate_generator_diagnostic",
        "num_candidate_rows": len(rows),
        "num_candidate_images": len({row["image"] for row in rows}),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "overall": summarize_group(rows),
        "safe_useful": summarize_group(safe),
        "risk": summarize_group(risk),
        "gtfree_clean_subset": summarize_group(clean),
        "oracle_safe_best_per_image": summarize_group(oracle),
        "decision": "go_next_mask_level_candidate_editor" if len(clean) > 0 and summarize_group(clean)["num_safe_useful"] > summarize_group(clean)["num_risk"] else "no_go_generator_or_filter_needs_revision",
        "warning": "Candidate generation is GT-free; labels/metrics are original-val GT audit only. No masks written.",
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
    rows, skipped = build_rows(args)
    write_csv(args.output_csv, rows)
    report = summarize(rows, skipped)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "decision": report["decision"], "summary": report}, indent=2))


if __name__ == "__main__":
    main()
