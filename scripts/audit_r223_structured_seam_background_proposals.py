#!/usr/bin/env python3
"""R223 structured seam/background proposal diagnostic.

Train/val-only F0 audit for the structured seam-background route. It keeps the
R221/R215/R216 idea of restoring local background channels between adjacent
epiphysis structures, but changes the unit from independent pixels to coherent
proposal components.

This script writes JSON/CSV diagnostics only. It must not be used on
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from audit_r216_soft_seam_action_candidates import (
    build_one_gt_cache,
    is_hard_risk,
    is_quick_useful,
    local_background_features,
    quick_metrics,
    soft_action_cut,
)
from run_r210_f1_neck_candidate_gate import candidate_components, read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


BASE_FEATURE_KEYS = [
    "dist_percentile",
    "candidate_rank",
    "action_frac",
    "cut_area",
    "cut_bbox_h",
    "cut_bbox_w",
    "cut_bbox_area",
    "cut_slenderness",
    "cut_fill",
    "cut_anchor_area_frac",
    "cut_mean_dist_in",
    "cut_min_dist_in",
    "cut_max_dist_in",
    "cut_mean_grad",
    "cut_p90_grad",
    "cut_mean_image",
    "cut_std_image",
    "ring_bg_frac",
    "ring_fg_frac",
    "ring_mean_image",
    "ring_std_image",
    "inner_mean_image",
    "outer_mean_image",
    "inner_outer_abs_diff",
    "local_anchor_frac",
    "local_bg_frac",
    "local_h",
    "local_w",
    "r216_crop_area",
    "r216_cut_area",
    "r216_cut_frac_crop",
    "r216_bg_components_before",
    "r216_bg_components_after",
    "r216_bg_component_count_delta",
    "r216_cut_adjacent_bg_components_before",
    "r216_cut_adjacent_bg_frac_before",
    "r216_cut_component_bg_area_after",
    "r216_cut_component_bg_frac_crop_after",
    "r216_cut_component_border_contacts_after",
    "r216_bg_channel_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R223 structured seam/background proposals.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dist-percentiles", default="6,8,10,12,15")
    parser.add_argument("--action-fracs", default="0.20,0.35,0.50,0.75")
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=24)
    parser.add_argument("--max-components-per-image", type=int, default=6)
    parser.add_argument("--crop-pad", type=int, default=18)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--classifier", choices=["hgb", "logreg"], default="hgb")
    parser.add_argument("--useful-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--risk-thresholds", default="0.05,0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--max-proposals-per-image", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r223_structured_seam_background_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r223_structured_seam_background_val_summary.json"))
    parser.add_argument("--cv-csv", type=Path, default=Path("outputs/analysis/r223_structured_seam_background_val_groupedcv.csv"))
    parser.add_argument("--seed", type=int, default=202607223)
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)),
    )


def finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "None", "nan"):
            continue
        value = float(value)
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else None


def proposal_features(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray, dist_percentile: float, rank: int, action_frac: float, pad: int) -> dict[str, float]:
    cut = cut.astype(bool)
    local_slice = bbox_slice(cut, pad)
    dist_in = ndimage.distance_transform_edt(anchor)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    ys, xs = np.where(cut)
    h = int(ys.max() - ys.min() + 1) if ys.size else 0
    w = int(xs.max() - xs.min() + 1) if xs.size else 0
    area = int(cut.sum())
    bbox_area = max(1, h * w)
    ring = np.logical_and(ndimage.binary_dilation(cut, structure=np.ones((5, 5), dtype=bool)), ~cut)
    ring_bg = np.logical_and(ring, ~anchor)
    ring_fg = np.logical_and(ring, anchor)
    inner = np.logical_and(ndimage.binary_erosion(anchor, structure=np.ones((3, 3), dtype=bool)), ring)
    outer = ring_bg
    if local_slice is None:
        local_anchor = np.zeros((0, 0), dtype=bool)
    else:
        local_anchor = anchor[local_slice].astype(bool)
    cut_img = image[cut] if area else np.asarray([], dtype=np.float32)
    ring_img = image[ring] if ring.any() else np.asarray([], dtype=np.float32)
    inner_img = image[inner] if inner.any() else np.asarray([], dtype=np.float32)
    outer_img = image[outer] if outer.any() else np.asarray([], dtype=np.float32)
    cut_grad = grad[cut] if area else np.asarray([], dtype=np.float32)
    cut_dist = dist_in[cut] if area else np.asarray([], dtype=np.float32)
    inner_mean = float(np.mean(inner_img)) if inner_img.size else 0.0
    outer_mean = float(np.mean(outer_img)) if outer_img.size else 0.0
    return {
        "dist_percentile": float(dist_percentile),
        "candidate_rank": float(rank),
        "action_frac": float(action_frac),
        "cut_area": float(area),
        "cut_bbox_h": float(h),
        "cut_bbox_w": float(w),
        "cut_bbox_area": float(bbox_area),
        "cut_slenderness": float(max(h, w) / max(1, min(h, w))) if area else 0.0,
        "cut_fill": safe_divide(area, bbox_area),
        "cut_anchor_area_frac": safe_divide(area, float(anchor.sum())),
        "cut_mean_dist_in": float(np.mean(cut_dist)) if cut_dist.size else 0.0,
        "cut_min_dist_in": float(np.min(cut_dist)) if cut_dist.size else 0.0,
        "cut_max_dist_in": float(np.max(cut_dist)) if cut_dist.size else 0.0,
        "cut_mean_grad": float(np.mean(cut_grad)) if cut_grad.size else 0.0,
        "cut_p90_grad": float(np.percentile(cut_grad, 90)) if cut_grad.size else 0.0,
        "cut_mean_image": float(np.mean(cut_img)) if cut_img.size else 0.0,
        "cut_std_image": float(np.std(cut_img)) if cut_img.size else 0.0,
        "ring_bg_frac": safe_divide(float(ring_bg.sum()), float(ring.sum())),
        "ring_fg_frac": safe_divide(float(ring_fg.sum()), float(ring.sum())),
        "ring_mean_image": float(np.mean(ring_img)) if ring_img.size else 0.0,
        "ring_std_image": float(np.std(ring_img)) if ring_img.size else 0.0,
        "inner_mean_image": inner_mean,
        "outer_mean_image": outer_mean,
        "inner_outer_abs_diff": abs(inner_mean - outer_mean),
        "local_anchor_frac": safe_divide(float(local_anchor.sum()), float(local_anchor.size)),
        "local_bg_frac": safe_divide(float((~local_anchor).sum()), float(local_anchor.size)),
        "local_h": float(local_anchor.shape[0]),
        "local_w": float(local_anchor.shape[1]),
    }


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    dist_percentiles = parse_float_list(args.dist_percentiles)
    action_fracs = parse_float_list(args.action_fracs)
    for index, name in enumerate(tqdm(case_names, desc=f"r223/proposals/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            continue
        image, _instance, gt, anchor = load_case(args, name)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        seen: set[tuple[int, float, bytes]] = set()
        for dist_percentile in dist_percentiles:
            raw_cuts = candidate_components(anchor, image, dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
            for rank, raw_cut in enumerate(raw_cuts[: args.max_components_per_image], start=1):
                for action_frac in action_fracs:
                    cut = soft_action_cut(image, anchor, raw_cut, action_frac, args.min_cut_area)
                    cut = np.logical_and(cut, anchor)
                    if int(cut.sum()) < args.min_cut_area:
                        continue
                    key = (rank, float(action_frac), np.packbits(cut).tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    trial = np.logical_and(anchor, ~cut)
                    trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                    row: dict[str, Any] = {
                        "run_id": "R223-structured-seam-background-proposal-diagnostic",
                        "split": args.split,
                        "image": name,
                    }
                    row.update(proposal_features(image, anchor, cut, dist_percentile, rank, action_frac, args.crop_pad))
                    row.update(local_background_features(anchor, trial, cut, gt, gt_cache, args.crop_pad))
                    for metric, value in anchor_metrics.items():
                        row[f"anchor_{metric}"] = value
                    for metric, value in trial_metrics.items():
                        row[f"candidate_{metric}"] = value
                        row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                    row["r223_quick_useful"] = float(is_quick_useful(row))
                    row["r223_hard_risk"] = float(is_hard_risk(row))
                    row["r223_safe_channel_proxy_positive"] = float(
                        float(row.get("r216_bg_channel_proxy") or 0.0) > 0.5
                        and float(row.get("r216_cut_gt_fg_frac") or 0.0) <= 0.5
                        and float(row.get("r216_cut_gt_gap_frac") or 0.0) >= 0.5
                    )
                    row["r223_overerosion_proxy"] = float(float(row.get("r216_cut_gt_fg_frac") or 0.0) > 0.5)
                    rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.output_csv, rows)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summarize(rows, [], args), indent=2), encoding="utf-8")
    return rows


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[as_float(row, key) for key in BASE_FEATURE_KEYS] for row in rows], dtype=np.float32)


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("clf", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, random_state=seed, l2_regularization=0.02)),
        ]
    )


def fit_binary_model(rows: list[dict[str, Any]], label_key: str, args: argparse.Namespace, seed: int) -> Pipeline | None:
    y = np.asarray([int(as_float(row, label_key) > 0.5) for row in rows], dtype=np.int32)
    if len(set(y.tolist())) < 2:
        return None
    model = make_model(args.classifier, seed)
    model.fit(matrix(rows), y)
    return model


def add_cv_probs(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    images = np.asarray(sorted({str(row["image"]) for row in rows}))
    if len(images) < 2:
        return [], {"skipped": True, "reason": "not_enough_images"}
    folds = min(args.cv_folds, len(images))
    probed: list[dict[str, Any]] = []
    skipped_folds = 0
    kf = KFold(n_splits=folds, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images), start=1):
        train_images = set(images[train_idx].tolist())
        val_images = set(images[val_idx].tolist())
        train_rows = [row for row in rows if row["image"] in train_images]
        val_rows = [row for row in rows if row["image"] in val_images]
        useful_model = fit_binary_model(train_rows, "r223_quick_useful", args, args.seed + fold_idx)
        risk_model = fit_binary_model(train_rows, "r223_hard_risk", args, args.seed + 100 + fold_idx)
        if useful_model is None or risk_model is None:
            skipped_folds += 1
            continue
        x_val = matrix(val_rows)
        useful_prob = useful_model.predict_proba(x_val)[:, 1]
        risk_prob = risk_model.predict_proba(x_val)[:, 1]
        for row, up, rp in zip(val_rows, useful_prob, risk_prob):
            probed.append({**row, "fold": fold_idx, "r223_useful_prob": float(up), "r223_risk_prob": float(rp)})
    return probed, {"skipped": False, "folds": folds, "skipped_folds": skipped_folds, "num_probed": len(probed)}


def selected_rows(rows: list[dict[str, Any]], useful_thr: float, risk_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        candidates = [
            row
            for row in image_rows
            if as_float(row, "r223_useful_prob") >= useful_thr and as_float(row, "r223_risk_prob") <= risk_thr
        ]
        candidates.sort(key=lambda row: (-as_float(row, "r223_useful_prob"), as_float(row, "r223_risk_prob"), as_float(row, "candidate_rank")))
        selected.extend(candidates[:max_per_image])
    return selected


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_quick_useful": int(sum(as_float(row, "r223_quick_useful") > 0.5 for row in rows)),
        "num_hard_risk": int(sum(as_float(row, "r223_hard_risk") > 0.5 for row in rows)),
        "num_safe_channel_proxy_positive": int(sum(as_float(row, "r223_safe_channel_proxy_positive") > 0.5 for row in rows)),
        "num_overerosion_proxy": int(sum(as_float(row, "r223_overerosion_proxy") > 0.5 for row in rows)),
        "mean_delta_dice": finite_mean(rows, "delta_dice"),
        "mean_delta_iou": finite_mean(rows, "delta_iou"),
        "mean_delta_recall": finite_mean(rows, "delta_recall"),
        "mean_delta_boundary_iou": finite_mean(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": finite_mean(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": finite_mean(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": finite_mean(rows, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": finite_mean(rows, "r216_cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": finite_mean(rows, "r216_cut_gt_gap_frac"),
    }


def evaluate_cv_grid(probed: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for useful_thr in parse_float_list(args.useful_thresholds):
        for risk_thr in parse_float_list(args.risk_thresholds):
            selected = selected_rows(probed, useful_thr, risk_thr, args.max_proposals_per_image)
            summary = summarize_group(selected)
            grid.append(
                {
                    "useful_threshold": useful_thr,
                    "risk_threshold": risk_thr,
                    "accepted": summary["num_rows"],
                    "accepted_images": summary["num_images"],
                    **summary,
                }
            )
    return grid


def oracle_best_by_image(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        useful = [row for row in image_rows if as_float(row, "r223_quick_useful") > 0.5 and as_float(row, "r223_hard_risk") <= 0.5]
        if not useful:
            continue
        useful.sort(
            key=lambda row: (
                -as_float(row, "delta_boundary_iou"),
                as_float(row, "delta_gap_region_fp_rate"),
                as_float(row, "delta_component_count_mae"),
                -as_float(row, "delta_dice"),
            )
        )
        selected.append(useful[0])
    return selected


def summarize(rows: list[dict[str, Any]], cv_grid: list[dict[str, Any]], args: argparse.Namespace, cv_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    safe = [row for row in rows if as_float(row, "r223_safe_channel_proxy_positive") > 0.5]
    oracle = oracle_best_by_image(rows)
    best_cv = None
    if cv_grid:
        best_cv = sorted(
            cv_grid,
            key=lambda row: (
                -int(row["num_hard_risk"]),
                int(row["num_quick_useful"]),
                -(row["mean_delta_gap_region_fp_rate"] or 0.0),
                row["mean_delta_boundary_iou"] or -999.0,
                row["mean_delta_dice"] or -999.0,
            ),
            reverse=True,
        )[0]
    return {
        "run_id": "R223-structured-seam-background-proposal-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "train_val_candidate_grouped_cv_diagnostic",
        "anchor_exp": args.anchor_exp,
        "feature_keys": BASE_FEATURE_KEYS,
        "num_candidate_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "overall": summarize_group(rows),
        "safe_channel_proxy_positive": summarize_group(safe),
        "oracle_best_per_image": summarize_group(oracle),
        "grouped_cv": cv_meta or {},
        "best_cv": best_cv,
        "warning": "candidate-only original train/val diagnostic; not mask-level R201 evidence and not clean-test-v2 evidence",
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
    probed, cv_meta = add_cv_probs(rows, args)
    cv_grid = evaluate_cv_grid(probed, args) if probed else []
    write_csv(args.cv_csv, cv_grid)
    report = summarize(rows, cv_grid, args, cv_meta)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(args.output_json), "cv_csv": str(args.cv_csv), "best_cv": report["best_cv"]}, indent=2))


if __name__ == "__main__":
    main()
