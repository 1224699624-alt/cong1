#!/usr/bin/env python3
"""R237 component-merge-targeted background-connectivity candidates.

Original train/val-only diagnostic for the "make bone-gap background connected"
route. The script looks for an R110 predicted connected component that overlaps
multiple GT instances, then proposes small cuts between the paired GT instances.

This is oracle candidate generation: GT instances are used only on original
train/val to diagnose whether this direction can repair adhesion/component
failures. It writes CSV/JSON diagnostics only and must not be used on
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

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, quick_metrics
from audit_r224_pairwise_instance_seam_candidates import (
    component_instance_overlaps,
    pairwise_seam_cut,
    trim_cut_by_score,
)
from run_r209_component_preserving_feasibility_audit import analyze_component_matches, read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


METRIC_KEYS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_iou",
    "boundary_f1",
    "gap_region_fp_rate",
    "component_count_mae",
    "component_delta_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R237 merge-targeted seam candidates on original train/val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--max-merged-components-per-image", type=int, default=6)
    parser.add_argument("--max-pairs-per-component", type=int, default=8)
    parser.add_argument("--seam-radii", default="1,2,3,4,5,6")
    parser.add_argument("--action-fracs", default="0.10,0.15,0.20,0.25,0.35,0.50,0.75")
    parser.add_argument("--min-cut-area", type=int, default=2)
    parser.add_argument("--max-cut-frac", type=float, default=0.012)
    parser.add_argument("--crop-pad", type=int, default=24)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--min-boundary-gain", type=float, default=1e-5)
    parser.add_argument("--min-gap-gain", type=float, default=1e-5)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r237_component_merge_targeted_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r237_component_merge_targeted_val_summary.json"))
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)),
    )


def border_contacts(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    return int(mask[0, :].any()) + int(mask[-1, :].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def local_bg_connectivity(anchor: np.ndarray, cut: np.ndarray, crop_pad: int) -> dict[str, float]:
    local_slice = bbox_slice(cut, crop_pad)
    if local_slice is None:
        return {
            "local_bg_components_before": 0.0,
            "local_bg_components_after": 0.0,
            "local_bg_component_delta": 0.0,
            "cut_adjacent_bg_components_before": 0.0,
            "cut_bg_channel_area_after": 0.0,
            "cut_bg_channel_frac_after": 0.0,
            "cut_bg_channel_border_contacts_after": 0.0,
            "background_connected_by_cut": 0.0,
        }
    anchor_l = anchor[local_slice].astype(bool)
    cut_l = cut[local_slice].astype(bool)
    trial_l = np.logical_and(anchor_l, ~cut_l)
    bg_before = ~anchor_l
    bg_after = ~trial_l
    labels_before, n_before = component_labels(bg_before)
    labels_after, n_after = component_labels(bg_after)
    cut_ring = np.logical_and(ndimage.binary_dilation(cut_l, structure=np.ones((3, 3), dtype=bool)), ~cut_l)
    adjacent = np.unique(labels_before[np.logical_and(cut_ring, bg_before)])
    adjacent = adjacent[adjacent > 0]
    after_labels = np.unique(labels_after[cut_l])
    after_labels = after_labels[after_labels > 0]
    channel = np.isin(labels_after, after_labels) if after_labels.size else np.zeros_like(bg_after, dtype=bool)
    channel_area = float(channel.sum())
    crop_area = float(channel.size)
    contacts = float(border_contacts(channel))
    return {
        "local_bg_components_before": float(n_before),
        "local_bg_components_after": float(n_after),
        "local_bg_component_delta": float(n_after - n_before),
        "cut_adjacent_bg_components_before": float(adjacent.size),
        "cut_bg_channel_area_after": channel_area,
        "cut_bg_channel_frac_after": safe_divide(channel_area, crop_area),
        "cut_bg_channel_border_contacts_after": contacts,
        "background_connected_by_cut": float(adjacent.size >= 2 or contacts >= 2),
    }


def component_match_metrics(mask: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> dict[str, float]:
    match = analyze_component_matches(mask, instance, min_overlap_frac)
    return {
        "pred_component_count_instance_match": float(match["pred_component_count"]),
        "gt_instance_count": float(match["gt_instance_count"]),
        "merged_pred_components": float(match["merged_pred_components"]),
        "gt_instances_covered": float(match["gt_instances_covered"]),
        "gt_instances_multi_covered": float(match["gt_instances_multi_covered"]),
    }


def pair_scores(overlaps: list[dict[str, float]]) -> list[tuple[float, int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(overlaps)):
        for j in range(i + 1, len(overlaps)):
            a = int(overlaps[i]["gt_id"])
            b = int(overlaps[j]["gt_id"])
            score = float(overlaps[i]["intersection"]) + float(overlaps[j]["intersection"])
            score += 0.25 * min(float(overlaps[i]["frac_gt"]), float(overlaps[j]["frac_gt"]))
            pairs.append((score, a, b))
    pairs.sort(reverse=True)
    return pairs


def is_merge_targeted_promotable(row: dict[str, Any], args: argparse.Namespace) -> bool:
    component_gain = float(row.get("delta_merged_pred_components") or 0.0) < 0.0
    count_safe = float(row.get("delta_component_count_mae") or 0.0) <= 0.0
    overlap_safe = (
        float(row.get("delta_dice") or 0.0) >= -args.dice_tol
        and float(row.get("delta_iou") or 0.0) >= -args.iou_tol
        and float(row.get("delta_recall") or 0.0) >= -args.recall_tol
    )
    support_gain = (
        float(row.get("delta_boundary_iou") or 0.0) >= args.min_boundary_gain
        or float(row.get("delta_boundary_f1") or 0.0) >= args.min_boundary_gain
        or float(row.get("delta_gap_region_fp_rate") or 0.0) <= -args.min_gap_gain
        or float(row.get("background_connected_by_cut") or 0.0) > 0.5
    )
    return bool(component_gain and count_safe and overlap_safe and support_gain)


def is_safe_nonmerge_gain(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(row.get("delta_dice") or 0.0) >= -args.dice_tol
        and float(row.get("delta_iou") or 0.0) >= -args.iou_tol
        and float(row.get("delta_recall") or 0.0) >= -args.recall_tol
        and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
        and (
            float(row.get("delta_boundary_iou") or 0.0) >= args.min_boundary_gain
            or float(row.get("delta_gap_region_fp_rate") or 0.0) <= -args.min_gap_gain
        )
    )


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    seam_radii = parse_int_list(args.seam_radii)
    action_fracs = parse_float_list(args.action_fracs)
    for index, name in enumerate(tqdm(case_names, desc=f"r237/merge-targeted/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            skipped.append({"image": name, "reason": "missing_anchor"})
            continue
        try:
            image, instance, gt, anchor = load_case(args, name)
        except Exception as exc:
            skipped.append({"image": name, "reason": f"load_failed:{type(exc).__name__}"})
            continue
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        anchor_match = component_match_metrics(anchor, instance, args.min_overlap_frac)
        labels, n_labels = component_labels(anchor)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        merged_components: list[tuple[int, int, list[dict[str, float]]]] = []
        for comp_id in range(1, n_labels + 1):
            comp = labels == comp_id
            overlaps = component_instance_overlaps(comp, instance, args.min_overlap_frac)
            if len(overlaps) >= 2:
                merged_components.append((len(overlaps), comp_id, overlaps))
        merged_components.sort(reverse=True)
        for _num_overlaps, comp_id, overlaps in merged_components[: args.max_merged_components_per_image]:
            comp = labels == comp_id
            for _score, gt_a, gt_b in pair_scores(overlaps)[: args.max_pairs_per_component]:
                for radius in seam_radii:
                    raw_cut = pairwise_seam_cut(comp, instance, gt_a, gt_b, radius)
                    if int(raw_cut.sum()) < args.min_cut_area:
                        continue
                    for action_frac in action_fracs:
                        cut = trim_cut_by_score(image, anchor, raw_cut, action_frac, args.min_cut_area)
                        cut = np.logical_and(cut, anchor)
                        if int(cut.sum()) < args.min_cut_area or int(cut.sum()) > max_pixels:
                            continue
                        trial = np.logical_and(anchor, ~cut)
                        trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                        trial_match = component_match_metrics(trial, instance, args.min_overlap_frac)
                        row: dict[str, Any] = {
                            "run_id": "R237-component-merge-targeted-candidate-diagnostic",
                            "split": args.split,
                            "image": name,
                            "pred_component_id": float(comp_id),
                            "num_component_gt_instances": float(len(overlaps)),
                            "gt_a": float(gt_a),
                            "gt_b": float(gt_b),
                            "seam_radius": float(radius),
                            "action_frac": float(action_frac),
                            "cut_area": float(cut.sum()),
                            "cut_frac_anchor": safe_divide(float(cut.sum()), float(anchor.sum())),
                            "cut_gt_fg_pixels": float(np.logical_and(cut, gt).sum()),
                            "cut_gt_fg_frac": safe_divide(float(np.logical_and(cut, gt).sum()), float(cut.sum())),
                            "cut_gt_gap_frac": safe_divide(float(np.logical_and(cut, gt_cache.gap_region).sum()), float(cut.sum())),
                        }
                        row.update(local_bg_connectivity(anchor, cut, args.crop_pad))
                        for metric, value in anchor_metrics.items():
                            row[f"anchor_{metric}"] = value
                        for metric, value in trial_metrics.items():
                            row[f"candidate_{metric}"] = value
                            row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                        for metric, value in anchor_match.items():
                            row[f"anchor_{metric}"] = value
                        for metric, value in trial_match.items():
                            row[f"candidate_{metric}"] = value
                            row[f"delta_{metric}"] = float(value) - float(anchor_match[metric])
                        row["merge_targeted_promotable"] = float(is_merge_targeted_promotable(row, args))
                        row["safe_nonmerge_gain"] = float(is_safe_nonmerge_gain(row, args))
                        row["overerosion_proxy"] = float(
                            float(row["cut_gt_fg_frac"]) > 0.5
                            or float(row.get("delta_recall") or 0.0) < -args.recall_tol
                        )
                        rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.output_csv, rows)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summarize(rows, args, skipped), indent=2), encoding="utf-8")
    setattr(args, "_r237_skipped", skipped)
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
        "num_merge_targeted_promotable": int(sum(float(row.get("merge_targeted_promotable") or 0.0) > 0.5 for row in rows)),
        "num_safe_nonmerge_gain": int(sum(float(row.get("safe_nonmerge_gain") or 0.0) > 0.5 for row in rows)),
        "num_overerosion_proxy": int(sum(float(row.get("overerosion_proxy") or 0.0) > 0.5 for row in rows)),
        "mean_cut_area": mean_value(rows, "cut_area"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_delta_merged_pred_components": mean_value(rows, "delta_merged_pred_components"),
        "mean_background_connected_by_cut": mean_value(rows, "background_connected_by_cut"),
    }


def oracle_best_by_image(rows: list[dict[str, Any]], prefer_merge: bool) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    chosen: list[dict[str, Any]] = []
    flag = "merge_targeted_promotable" if prefer_merge else "safe_nonmerge_gain"
    for image_rows in grouped.values():
        useful = [row for row in image_rows if float(row.get(flag) or 0.0) > 0.5]
        if not useful:
            continue
        useful.sort(
            key=lambda row: (
                float(row.get("delta_merged_pred_components") or 0.0),
                float(row.get("delta_component_count_mae") or 0.0),
                -float(row.get("delta_boundary_iou") or 0.0),
                float(row.get("delta_gap_region_fp_rate") or 0.0),
                -float(row.get("delta_dice") or 0.0),
            )
        )
        chosen.append(useful[0])
    return chosen


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace, skipped: list[dict[str, str]] | None = None) -> dict[str, Any]:
    skipped = skipped or list(getattr(args, "_r237_skipped", []))
    promotable = [row for row in rows if float(row.get("merge_targeted_promotable") or 0.0) > 0.5]
    safe_gain = [row for row in rows if float(row.get("safe_nonmerge_gain") or 0.0) > 0.5]
    oracle_merge = oracle_best_by_image(rows, prefer_merge=True)
    oracle_safe = oracle_best_by_image(rows, prefer_merge=False)
    images_with_merge_candidates = len({row["image"] for row in rows})
    return {
        "run_id": "R237-component-merge-targeted-candidate-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "train_val_oracle_candidate_diagnostic",
        "purpose": "test whether cutting within predicted multi-instance components can connect bone-gap background and reduce merge diagnostics",
        "num_candidate_rows": len(rows),
        "num_images_with_merge_candidates": images_with_merge_candidates,
        "num_skipped": len(skipped),
        "skipped": skipped,
        "overall": summarize_group(rows),
        "merge_targeted_promotable": summarize_group(promotable),
        "safe_nonmerge_gain": summarize_group(safe_gain),
        "oracle_merge_best_per_image": summarize_group(oracle_merge),
        "oracle_safe_best_per_image": summarize_group(oracle_safe),
        "decision": "go_build_merge_targeted_editor_or_scorer" if oracle_merge else "no_go_merge_target_weak_signal",
        "warning": "GT instances are used for candidate generation on original train/val only; this is not deployable and not clean-test-v2 evidence.",
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
    print(json.dumps({"decision": report["decision"], "output_csv": str(args.output_csv), "output_json": str(args.output_json), "summary": report}, indent=2))


if __name__ == "__main__":
    main()
