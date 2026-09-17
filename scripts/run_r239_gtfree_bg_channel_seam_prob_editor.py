#!/usr/bin/env python3
"""R239 GT-free background-channel seam-probability editor.

Original-val mask-level development experiment. Candidate generation uses only
the R110 anchor mask, image gradients, local background connectivity, and the
R228 seam-probability map. GT is used only after candidate generation for
train/val evaluation/gating. Do not use this on clean-test-v2 for tuning.
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
from audit_r226_bg_connectivity_seam_scorer import (
    bg_components_near_component,
    bg_contact_seeds,
    candidate_features,
    make_bg_bridge_cut,
    seed_pair_cut,
    trim_cut_by_score,
)
from audit_r228_oracle_pixel_seam_localizer import as_float, fit_model, predict_prob, sample_training
from run_r209_component_preserving_feasibility_audit import read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R239 GT-free bg-channel seam-prob editor on original val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r239_gtfree_bg_channel_seam_prob_editor")
    parser.add_argument("--split", default="val")
    parser.add_argument("--r224-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--source-candidate-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--source-mode", choices=["source-csv", "all-anchors"], default="all-anchors")
    parser.add_argument("--limit-source", type=int, default=0)
    parser.add_argument("--limit-train-rows", type=int, default=300)
    parser.add_argument("--samples-per-image", type=int, default=1024)
    parser.add_argument("--positive-oversample", type=int, default=512)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--max-components-per-image", type=int, default=3)
    parser.add_argument("--component-pad", type=int, default=18)
    parser.add_argument("--max-bg-components", type=int, default=6)
    parser.add_argument("--max-pairs-per-component", type=int, default=4)
    parser.add_argument("--corridor-radii", default="1,2,3")
    parser.add_argument("--action-fracs", default="0.15,0.25,0.35")
    parser.add_argument("--min-cut-area", type=int, default=3)
    parser.add_argument("--max-cut-frac", type=float, default=0.012)
    parser.add_argument("--max-candidates-per-image", type=int, default=120)
    parser.add_argument("--max-cuts-per-image", type=int, default=1)
    parser.add_argument("--seam-prob-threshold", type=float, default=0.35)
    parser.add_argument("--seam-prob-p90-threshold", type=float, default=0.70)
    parser.add_argument("--min-bg-channel-frac", type=float, default=0.10)
    parser.add_argument("--require-bg-channel", action="store_true")
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--boundary-tol", type=float, default=0.0)
    parser.add_argument("--allow-component-mae-worsen", action="store_true")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r239_gtfree_bg_channel_seam_prob_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r239_gtfree_bg_channel_seam_prob_per_image.csv"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r239_gtfree_bg_channel_seam_prob_candidates.csv"))
    parser.add_argument("--seed", type=int, default=202607239)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def application_names(args: argparse.Namespace) -> list[str]:
    if args.source_mode == "all-anchors":
        anchor_dir = args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks"
        ordered = sorted(path.name for path in anchor_dir.glob("*.png"))
        if not ordered:
            ordered = names(args.raw_root, args.dataset, args.split)
        return ordered[: args.limit_source or None]
    ordered: list[str] = []
    seen: set[str] = set()
    if args.source_candidate_csv.exists():
        with args.source_candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("image") or "")
                if name and name not in seen:
                    seen.add(name)
                    ordered.append(name)
    return ordered[: args.limit_source or None]


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def gate_candidate(row: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    if as_float(row, "seam_prob_mean") < args.seam_prob_threshold:
        return False, "low_mean_prob"
    if as_float(row, "seam_prob_p90") < args.seam_prob_p90_threshold:
        return False, "low_p90_prob"
    if args.require_bg_channel and as_float(row, "bg_channel_proxy") <= 0.5:
        return False, "no_bg_channel_proxy"
    if as_float(row, "cut_component_bg_frac_after") < args.min_bg_channel_frac:
        return False, "low_bg_channel_frac"
    if as_float(row, "delta_dice") < -args.dice_tol:
        return False, "dice_drop"
    if as_float(row, "delta_iou") < -args.iou_tol:
        return False, "iou_drop"
    if as_float(row, "delta_recall") < -args.recall_tol:
        return False, "recall_drop"
    if as_float(row, "delta_boundary_iou") < -args.boundary_tol:
        return False, "boundary_iou_drop"
    if as_float(row, "delta_boundary_f1") < -args.boundary_tol:
        return False, "boundary_f1_drop"
    if as_float(row, "delta_gap_region_fp_rate") >= 0.0:
        return False, "no_gap_fp_gain"
    if not args.allow_component_mae_worsen and as_float(row, "delta_component_count_mae") > 0.0:
        return False, "component_mae_worse"
    return True, "accepted"


def build_pair_specs(args: argparse.Namespace, anchor: np.ndarray, component: np.ndarray) -> list[tuple[float, str, dict[str, Any], dict[str, Any]]]:
    pair_specs: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
    bg_infos = bg_components_near_component(anchor, component, args.component_pad, args.max_bg_components)
    for i, a in enumerate(bg_infos):
        for b in bg_infos[i + 1 :]:
            dy = float(a["centroid_y"] - b["centroid_y"])
            dx = float(a["centroid_x"] - b["centroid_x"])
            dist = float(np.hypot(dy, dx))
            score = float(a["contact"] + b["contact"]) - 0.02 * dist
            pair_specs.append((score, "bg_component", a, b))
    seeds = bg_contact_seeds(anchor, component, args.max_bg_components)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            dist = float(np.hypot(float(a["y"]) - float(b["y"]), float(a["x"]) - float(b["x"])))
            if dist >= 12.0:
                pair_specs.append((-abs(dist - 42.0), "bg_seed", a, b))
    pair_specs.sort(key=lambda item: item[0], reverse=True)
    return pair_specs[: args.max_pairs_per_component]


def candidate_rows_for_case(
    args: argparse.Namespace,
    image: np.ndarray,
    instance: np.ndarray,
    gt: np.ndarray,
    anchor: np.ndarray,
    prob: np.ndarray,
    name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
    anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
    comp_labels, n_components = component_labels(anchor)
    component_order = [(int((comp_labels == comp_id).sum()), comp_id) for comp_id in range(1, n_components + 1)]
    component_order.sort(reverse=True)
    max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
    seen: set[bytes] = set()
    for _area, comp_id in component_order[: args.max_components_per_image]:
        component = comp_labels == comp_id
        comp_area = int(component.sum())
        if comp_area < args.min_component_area or comp_area > args.max_component_area:
            continue
        for _score, pair_kind, info_a, info_b in build_pair_specs(args, anchor, component):
            bg_a = np.zeros_like(anchor, dtype=bool)
            bg_b = np.zeros_like(anchor, dtype=bool)
            local_component = component
            if pair_kind == "bg_component":
                local = info_a["local_slice"]
                labels = info_a["labels"]
                local_comp_full = np.zeros_like(anchor, dtype=bool)
                bg_a[local] = labels == int(info_a["label_id"])
                bg_b[local] = labels == int(info_b["label_id"])
                local_comp_full[local] = info_a["local_component"]
                local_component = local_comp_full & component
            else:
                bg_a[int(info_a["y"]), int(info_a["x"])] = True
                bg_b[int(info_b["y"]), int(info_b["x"])] = True
            for radius in parse_int_list(args.corridor_radii):
                if pair_kind == "bg_component":
                    raw_cut, bridge = make_bg_bridge_cut(local_component, bg_a, bg_b, image, radius, args.min_cut_area)
                else:
                    raw_cut, bridge = seed_pair_cut(local_component, info_a, info_b, image, radius, args.min_cut_area)
                raw_cut &= anchor
                if int(raw_cut.sum()) < args.min_cut_area:
                    continue
                for action_frac in parse_float_list(args.action_fracs):
                    cut = trim_cut_by_score(image, anchor, raw_cut, bridge, action_frac, args.min_cut_area) & anchor
                    cut_area = int(cut.sum())
                    if cut_area < args.min_cut_area or cut_area > max_pixels:
                        continue
                    key = np.packbits(cut.ravel()).tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    trial = anchor & ~cut
                    trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                    row: dict[str, Any] = {
                        "run_id": "R239-gtfree-bg-channel-seam-prob-editor",
                        "split": args.split,
                        "image": name,
                        "candidate_family": pair_kind,
                        "pred_component_id": float(comp_id),
                        "seam_prob_mean": float(np.mean(prob[cut])),
                        "seam_prob_p90": float(np.percentile(prob[cut], 90)),
                        "cut_gt_fg_frac": safe_divide(float(np.logical_and(cut, instance > 0).sum()), float(cut_area)),
                        "cut_gt_gap_frac": safe_divide(float(np.logical_and(cut, gt_cache.gap_region).sum()), float(cut_area)),
                        "_cut": cut,
                    }
                    row.update(candidate_features(image, anchor, component, cut, bridge, bg_a, bg_b, radius, action_frac))
                    for metric, value in trial_metrics.items():
                        row[f"candidate_{metric}"] = value
                        row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                    accepted, reason = gate_candidate(row, args)
                    row["accepted"] = float(accepted)
                    row["reject_reason"] = reason
                    row["overerosion_proxy"] = float(row["cut_gt_fg_frac"] > 0.5 or row["delta_recall"] < -args.recall_tol)
                    rows.append(row)
                    if len(rows) >= args.max_candidates_per_image:
                        return rows
    return rows


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(records),
        "num_images": len({str(row.get("image")) for row in records}),
        "num_edited": int(sum(as_float(row, "edited") > 0.5 for row in records)),
        "mean_delta_dice": mean_value(records, "delta_dice"),
        "mean_delta_iou": mean_value(records, "delta_iou"),
        "mean_delta_recall": mean_value(records, "delta_recall"),
        "mean_delta_precision": mean_value(records, "delta_precision"),
        "mean_delta_boundary_iou": mean_value(records, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(records, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(records, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(records, "delta_component_count_mae"),
        "mean_cut_pixels": mean_value(records, "cut_pixels"),
    }


def summarize_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if as_float(row, "accepted") > 0.5]
    return {
        "num_candidate_rows": len(rows),
        "num_accepted_candidates": len(accepted),
        "num_candidate_images": len({row["image"] for row in rows}),
        "num_accepted_images": len({row["image"] for row in accepted}),
        "mean_accepted_seam_prob": mean_value(accepted, "seam_prob_mean"),
        "mean_accepted_cut_gt_fg_frac": mean_value(accepted, "cut_gt_fg_frac"),
        "mean_accepted_cut_gt_gap_frac": mean_value(accepted, "cut_gt_gap_frac"),
        "num_accepted_overerosion_proxy": int(sum(as_float(row, "overerosion_proxy") > 0.5 for row in accepted)),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x, y, train_info = sample_training(args)
    model = fit_model(args, x, y)
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.split / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()
    per_image: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    app_names = application_names(args)
    for name in tqdm(app_names, desc=f"r239/apply/{args.split}"):
        if not anchor_path(args, name).exists():
            continue
        image, instance, gt, anchor = load_case(args, name)
        prob = predict_prob(model, image, anchor)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        current = anchor.copy()
        accepted_cuts = []
        rows = candidate_rows_for_case(args, image, instance, gt, anchor, prob, name)
        rows.sort(
            key=lambda row: (
                -as_float(row, "accepted"),
                -as_float(row, "bg_channel_proxy"),
                -as_float(row, "seam_prob_p90"),
                -as_float(row, "seam_prob_mean"),
                -as_float(row, "delta_boundary_iou"),
            )
        )
        for row in rows:
            cut = row.pop("_cut", None)
            if cut is None:
                candidate_rows.append(row)
                continue
            if as_float(row, "accepted") <= 0.5:
                candidate_rows.append(row)
                continue
            if len(accepted_cuts) >= args.max_cuts_per_image:
                row["accepted"] = 0.0
                row["reject_reason"] = "max_cuts_reached"
                candidate_rows.append(row)
                continue
            cut = cut & current
            if int(cut.sum()) < args.min_cut_area:
                row["accepted"] = 0.0
                row["reject_reason"] = "cut_empty_after_previous"
                candidate_rows.append(row)
                continue
            current = current & ~cut
            accepted_cuts.append(cut)
            candidate_rows.append(row)
        pred_metrics = quick_metrics(current, gt, gt_cache, args.boundary_kernel)
        write_mask(output_dir / name, current)
        record: dict[str, Any] = {
            "image": name,
            "edited": float(bool(accepted_cuts)),
            "num_accepted_cuts": float(len(accepted_cuts)),
            "cut_pixels": float(sum(int(cut.sum()) for cut in accepted_cuts)),
        }
        for metric, value in anchor_metrics.items():
            record[f"anchor_{metric}"] = value
        for metric, value in pred_metrics.items():
            record[f"r239_{metric}"] = value
            record[f"delta_{metric}"] = metric_delta(pred_metrics, anchor_metrics, metric)
        per_image.append(record)
    report = {
        "run_id": "R239-gtfree-bg-channel-seam-prob-editor",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "clean_test_v2_used": False,
        "writes_masks": True,
        "evidence_level": "original_val_mask_level_development",
        "candidate_generation": "GT-free bg components/seeds + seam probability; GT only for val metrics/gate",
        "train_info": {**train_info, "num_samples": int(len(y)), "positive_sample_rate": float(np.mean(y))},
        "output_mask_dir": str(output_dir),
        "num_images": len(per_image),
        "summary": summarize_records(per_image),
        "candidate_summary": summarize_candidates(candidate_rows),
        "per_image": per_image,
        "warning": "Original-val development only; not clean-test-v2 or R201 final evidence.",
    }
    return report, candidate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row if key != "_cut"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    report, candidates = run(args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.per_image_csv, report["per_image"])
    write_csv(args.candidate_csv, candidates)
    print(json.dumps({"summary_json": str(args.summary_json), "per_image_csv": str(args.per_image_csv), "candidate_csv": str(args.candidate_csv), "summary": report["summary"], "candidate_summary": report["candidate_summary"]}, indent=2))


if __name__ == "__main__":
    main()
