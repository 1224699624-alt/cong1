#!/usr/bin/env python3
"""R211-F1 learned GT-free acceptance gate.

Builds a candidate-level table from R210-F1 neck candidates on train/val,
uses GT only to label candidates during training/validation, then applies a
locked classifier with inference-time features only. clean-test-v2 is not
used by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from run_r201_unified_eval import build_gt_cache, compute_metrics
from run_r201_unified_eval import boundary as r201_boundary
from run_r201_unified_eval import surface as r201_surface
from run_r201_unified_eval import GtCache
from run_r209_component_preserving_feasibility_audit import analyze_component_matches
from run_r210_f1_neck_candidate_gate import candidate_components, quick_metrics, read_instance
from run_r211_gt_free_acceptance_gate import component_info, split_features
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


BASE_FEATURE_KEYS = [
    "component_area",
    "cut_area",
    "cut_frac_component",
    "split_gain",
    "piece_count",
    "min_piece_area",
    "min_piece_frac",
    "bbox_h",
    "bbox_w",
    "slenderness",
    "fill",
    "gradient_ratio",
    "dist_ratio",
    "component_edge_contact_frac",
    "cut_boundary_contact_frac",
    "cut_exposed_boundary_frac",
    "cut_neighbor_fg_frac",
    "cut_centroid_dist_norm",
    "anchor_component_count",
    "anchor_area",
    "candidate_rank",
]

R214_IMAGE_CONTEXT_FEATURE_KEYS = [
    "cut_intensity_mean",
    "cut_intensity_std",
    "cut_intensity_p10",
    "cut_intensity_p90",
    "component_intensity_mean",
    "component_intensity_std",
    "cut_component_intensity_delta",
    "cut_component_intensity_ratio",
    "ring_intensity_mean",
    "ring_intensity_std",
    "cut_ring_intensity_delta",
    "cut_ring_intensity_ratio",
    "bbox_intensity_mean",
    "bbox_intensity_std",
    "cut_gradient_mean",
    "cut_gradient_p90",
    "ring_gradient_mean",
    "ring_gradient_p90",
    "bbox_gradient_mean",
    "bbox_gradient_p90",
    "cut_ring_gradient_ratio",
    "cut_bbox_gradient_ratio",
    "cut_bbox_y_center_norm",
    "cut_bbox_x_center_norm",
    "cut_bbox_area_frac_image",
]

FEATURE_KEYS = BASE_FEATURE_KEYS + R214_IMAGE_CONTEXT_FEATURE_KEYS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R211-F1 learned GT-free acceptance gate.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r211_f1_learned_acceptance_gate")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=16)
    parser.add_argument("--max-components-per-image", type=int, default=5)
    parser.add_argument("--max-cut-frac", type=float, default=0.004)
    parser.add_argument("--max-cut-frac-component", type=float, default=0.03)
    parser.add_argument("--max-accept-candidate-rank", type=int, default=0)
    parser.add_argument("--min-accept-gradient-ratio", type=float, default=0.0)
    parser.add_argument("--min-accept-dist-ratio", type=float, default=0.0)
    parser.add_argument("--min-accept-slenderness", type=float, default=0.0)
    parser.add_argument("--max-accept-fill", type=float, default=1.0)
    parser.add_argument("--max-pred-component-increase", type=int, default=1_000_000)
    parser.add_argument("--max-step-component-increase", type=int, default=1_000_000)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--label-dice-tol", type=float, default=0.0015)
    parser.add_argument("--label-iou-tol", type=float, default=0.0025)
    parser.add_argument("--label-recall-tol", type=float, default=0.0025)
    parser.add_argument("--label-boundary-tol", type=float, default=0.0015)
    parser.add_argument("--label-min-gap-gain", type=float, default=1e-4)
    parser.add_argument("--label-min-component-gain", type=float, default=0.5)
    parser.add_argument("--label-mode", choices=["anatomy_gain", "boundary_positive"], default="anatomy_gain")
    parser.add_argument("--label-min-boundary-gain", type=float, default=0.0)
    parser.add_argument("--label-min-surface2-gain", type=float, default=0.0)
    parser.add_argument("--label-max-hd95-delta", type=float, default=0.0)
    parser.add_argument("--label-max-assd-delta", type=float, default=0.0)
    parser.add_argument("--strict-boundary-gate", action="store_true")
    parser.add_argument("--full-candidate-labels", action="store_true")
    parser.add_argument("--candidate-instance-labels", action="store_true")
    parser.add_argument("--threshold-grid", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--classifier", choices=["logreg", "hgb"], default="logreg")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/r211_f1_learned_acceptance_gate/r211_f1_acceptor.joblib"))
    parser.add_argument("--eval-only-checkpoint", type=Path, default=Path(""))
    parser.add_argument("--train-candidate-csv", type=Path, default=Path("outputs/analysis/r211_f1_train_candidates.csv"))
    parser.add_argument("--val-candidate-csv", type=Path, default=Path("outputs/analysis/r211_f1_val_candidates.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r211_f1_learned_acceptance_gate_val_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r211_f1_learned_acceptance_gate_val_per_image.csv"))
    parser.add_argument("--threshold-audit-csv", type=Path, default=Path(""))
    parser.add_argument("--seed", type=int, default=202607211)
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list[Any]:
    return [cast(item) for item in text.split(",") if item.strip()]


def add_f3_shape_features(features: dict[str, float], current: np.ndarray, cut: np.ndarray) -> dict[str, float]:
    cut = cut.astype(bool)
    if int(cut.sum()) <= 0:
        features.update(
            {
                "component_edge_contact_frac": 0.0,
                "cut_boundary_contact_frac": 0.0,
                "cut_exposed_boundary_frac": 0.0,
                "cut_neighbor_fg_frac": 0.0,
                "cut_centroid_dist_norm": 0.0,
            }
        )
        return features

    comp, _, _ = component_info(current.astype(bool), cut)
    comp_boundary = r201_boundary(comp, 3)
    cut_boundary = r201_boundary(cut, 3)
    dilated_cut = ndimage.binary_dilation(cut, structure=np.ones((3, 3), dtype=bool))
    cut_ring = np.logical_and(dilated_cut, ~cut)
    neighbor_fg = np.logical_and(cut_ring, comp)
    neighbor_bg = np.logical_and(cut_ring, ~comp)

    cy, cx = ndimage.center_of_mass(cut.astype(np.uint8))
    if np.isnan(cy) or np.isnan(cx) or not comp.any():
        centroid_dist_norm = 0.0
    else:
        dist_to_bg = ndimage.distance_transform_edt(comp)
        centroid_dist_norm = float(dist_to_bg[int(round(cy)), int(round(cx))] / max(1.0, float(dist_to_bg[comp].max())))

    features.update(
        {
            "component_edge_contact_frac": float(np.logical_and(cut, comp_boundary).sum() / max(1, int(cut.sum()))),
            "cut_boundary_contact_frac": float(np.logical_and(cut_boundary, comp).sum() / max(1, int(cut_boundary.sum()))),
            "cut_exposed_boundary_frac": float(neighbor_bg.sum() / max(1, int(cut_ring.sum()))),
            "cut_neighbor_fg_frac": float(neighbor_fg.sum() / max(1, int(cut_ring.sum()))),
            "cut_centroid_dist_norm": centroid_dist_norm,
        }
    )
    return features


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


def bbox_slice(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def add_r214_image_context_features(features: dict[str, float], image: np.ndarray, current: np.ndarray, cut: np.ndarray) -> dict[str, float]:
    """Add GT-free local image context so candidate gates can see edge/intensity evidence."""
    cut = cut.astype(bool)
    if int(cut.sum()) <= 0:
        features.update({key: 0.0 for key in R214_IMAGE_CONTEXT_FEATURE_KEYS})
        return features

    comp, _, _ = component_info(current.astype(bool), cut)
    cut = np.logical_and(cut, comp)
    dilated_cut = ndimage.binary_dilation(cut, structure=np.ones((5, 5), dtype=bool))
    ring = np.logical_and(dilated_cut, ~cut)
    local_slice = bbox_slice(cut, pad=6)
    local_mask = np.zeros_like(cut, dtype=bool)
    if local_slice is not None:
        local_mask[local_slice] = True

    grad_y, grad_x = np.gradient(image.astype(np.float32))
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    cut_i = safe_stats(image[cut])
    comp_i = safe_stats(image[comp])
    ring_i = safe_stats(image[ring])
    bbox_i = safe_stats(image[local_mask])
    cut_g = safe_stats(grad[cut])
    ring_g = safe_stats(grad[ring])
    bbox_g = safe_stats(grad[local_mask])

    if local_slice is None:
        y_center_norm = 0.0
        x_center_norm = 0.0
        bbox_area_frac = 0.0
    else:
        ys, xs = np.where(cut)
        y_center_norm = float((ys.min() + ys.max() + 1) / max(1, 2 * cut.shape[0]))
        x_center_norm = float((xs.min() + xs.max() + 1) / max(1, 2 * cut.shape[1]))
        bbox_area_frac = float(local_mask.sum() / max(1, cut.size))

    features.update(
        {
            "cut_intensity_mean": cut_i["mean"],
            "cut_intensity_std": cut_i["std"],
            "cut_intensity_p10": cut_i["p10"],
            "cut_intensity_p90": cut_i["p90"],
            "component_intensity_mean": comp_i["mean"],
            "component_intensity_std": comp_i["std"],
            "cut_component_intensity_delta": float(cut_i["mean"] - comp_i["mean"]),
            "cut_component_intensity_ratio": safe_ratio(cut_i["mean"], comp_i["mean"]),
            "ring_intensity_mean": ring_i["mean"],
            "ring_intensity_std": ring_i["std"],
            "cut_ring_intensity_delta": float(cut_i["mean"] - ring_i["mean"]),
            "cut_ring_intensity_ratio": safe_ratio(cut_i["mean"], ring_i["mean"]),
            "bbox_intensity_mean": bbox_i["mean"],
            "bbox_intensity_std": bbox_i["std"],
            "cut_gradient_mean": cut_g["mean"],
            "cut_gradient_p90": cut_g["p90"],
            "ring_gradient_mean": ring_g["mean"],
            "ring_gradient_p90": ring_g["p90"],
            "bbox_gradient_mean": bbox_g["mean"],
            "bbox_gradient_p90": bbox_g["p90"],
            "cut_ring_gradient_ratio": safe_ratio(cut_g["mean"], ring_g["mean"]),
            "cut_bbox_gradient_ratio": safe_ratio(cut_g["mean"], bbox_g["mean"]),
            "cut_bbox_y_center_norm": y_center_norm,
            "cut_bbox_x_center_norm": x_center_norm,
            "cut_bbox_area_frac_image": bbox_area_frac,
        }
    )
    return features


def build_one_gt_cache(name: str, gt: np.ndarray, boundary_kernel: int, gap_kernel: int, with_surface_distance: bool) -> GtCache:
    gap_structure = np.ones((gap_kernel, gap_kernel), dtype=bool)
    gt_boundary = r201_boundary(gt, boundary_kernel)
    gt_surface = r201_surface(gt)
    gt_surface_distance = ndimage.distance_transform_edt(~gt_surface) if with_surface_distance and gt_surface.any() else None
    gt_gap = np.logical_and(ndimage.binary_dilation(gt, structure=gap_structure), ~gt)
    gt_components, _ = ndimage.label(gt)
    return GtCache(
        image=name,
        mask=gt,
        boundary=gt_boundary,
        surface=gt_surface,
        surface_distance=gt_surface_distance,
        gap_region=gt_gap,
        component_count=int(gt_components.max()),
    )


def anchor_path(args: argparse.Namespace, split: str, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / split / "masks" / name


def load_case(args: argparse.Namespace, split: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, split, name))
    anchor = resize_like(read_mask(anchor_path(args, split, name)), gt.shape)
    return image, instance, gt, anchor


def component_merge_count(mask: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> int:
    return int(analyze_component_matches(mask, instance, min_overlap_frac)["merged_pred_components"])


def metric_delta(new: dict[str, float | None], old: dict[str, float | None], key: str) -> float | None:
    if old.get(key) is None or new.get(key) is None:
        return None
    return float(new[key]) - float(old[key])


def cheap_component_metrics(mask: np.ndarray, gt_component_count: int) -> dict[str, float]:
    _, pred_n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    return {
        "component_merge_rate": float(int(pred_n) < int(gt_component_count)),
        "component_count_mae": float(abs(int(pred_n) - int(gt_component_count))),
        "component_delta_mean": float(int(pred_n) - int(gt_component_count)),
    }


def pred_component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    return int(count)


def candidate_metrics(
    args: argparse.Namespace,
    pred: np.ndarray,
    gt: np.ndarray,
    gt_cache: Any,
) -> dict[str, float | None]:
    if args.full_candidate_labels:
        return compute_metrics(pred, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    row: dict[str, float | None] = quick_metrics(pred, gt, gt_cache, args)
    row.update(cheap_component_metrics(pred, gt_cache.component_count))
    return row


def is_positive_label(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    if args.label_mode == "boundary_positive":
        return bool(
            float(row.get("delta_dice") or 0.0) >= -args.label_dice_tol
            and float(row.get("delta_iou") or 0.0) >= -args.label_iou_tol
            and float(row.get("delta_recall") or 0.0) >= -args.label_recall_tol
            and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
            and float(row.get("delta_boundary_iou") or 0.0) >= args.label_min_boundary_gain
            and float(row.get("delta_boundary_f1") or 0.0) >= args.label_min_boundary_gain
            and float(row.get("delta_surface_dice_2px") or 0.0) >= args.label_min_surface2_gain
            and float(row.get("delta_hd95_px") or 0.0) <= args.label_max_hd95_delta
            and float(row.get("delta_assd_px") or 0.0) <= args.label_max_assd_delta
            and (
                float(row.get("delta_gap_region_fp_rate") or 0.0) <= -args.label_min_gap_gain
                or float(row.get("delta_component_count_mae") or 0.0) <= -args.label_min_component_gain
                or float(row.get("delta_component_merge_rate") or 0.0) < 0.0
                or float(row.get("delta_instance_merge_count") or 0.0) < 0.0
            )
        )
    safe = (
        float(row.get("delta_dice") or 0.0) >= -args.label_dice_tol
        and float(row.get("delta_iou") or 0.0) >= -args.label_iou_tol
        and float(row.get("delta_recall") or 0.0) >= -args.label_recall_tol
        and float(row.get("delta_boundary_iou") or 0.0) >= -args.label_boundary_tol
        and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
    )
    gain = (
        float(row.get("delta_gap_region_fp_rate") or 0.0) <= -args.label_min_gap_gain
        or float(row.get("delta_component_count_mae") or 0.0) <= -args.label_min_component_gain
        or float(row.get("delta_component_merge_rate") or 0.0) < 0.0
        or float(row.get("delta_instance_merge_count") or 0.0) < 0.0
    )
    return bool(safe and gain)


def candidate_trial_row(
    args: argparse.Namespace,
    split: str,
    name: str,
    rank: int,
    anchor: np.ndarray,
    image: np.ndarray,
    instance: np.ndarray,
    gt: np.ndarray,
    gt_cache: Any,
    cut: np.ndarray,
    old_metrics: dict[str, float | None],
    old_merge: int,
    old_instance_merge: int | None,
) -> dict[str, Any]:
    cut = np.logical_and(cut, anchor)
    features = split_features(anchor, cut, image)
    features = add_f3_shape_features(features, anchor, cut)
    features = add_r214_image_context_features(features, image, anchor, cut)
    _, anchor_component_count = ndimage.label(anchor, structure=np.ones((3, 3), dtype=np.uint8))
    features["anchor_component_count"] = float(anchor_component_count)
    features["anchor_area"] = float(anchor.sum())
    features["candidate_rank"] = float(rank)

    if int(cut.sum()) < args.min_cut_area:
        trial = anchor
    else:
        trial = np.logical_and(anchor, ~cut)
    new_metrics = candidate_metrics(args, trial, gt, gt_cache)
    new_merge = int(new_metrics.get("component_merge_rate") or 0)
    new_instance_merge = component_merge_count(trial, instance, args.min_overlap_frac) if args.candidate_instance_labels else old_instance_merge
    row: dict[str, Any] = {
        "split": split,
        "image": name,
        "candidate_rank": float(rank),
        **features,
        "anchor_instance_merge_count": None if old_instance_merge is None else float(old_instance_merge),
        "candidate_instance_merge_count": None if new_instance_merge is None else float(new_instance_merge),
        "delta_instance_merge_count": None if old_instance_merge is None or new_instance_merge is None else float(new_instance_merge - old_instance_merge),
        "anchor_component_merge_rate": float(old_merge),
        "candidate_component_merge_rate": float(new_merge),
    }
    for key, value in old_metrics.items():
        row[f"anchor_{key}"] = value
    for key, value in new_metrics.items():
        row[f"candidate_{key}"] = value
        row[f"delta_{key}"] = metric_delta(new_metrics, old_metrics, key)
    row["label_positive"] = float(is_positive_label(args, row))
    return row


def build_candidate_table(args: argparse.Namespace, split: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_names = names(args.raw_root, args.dataset, split)[: limit or None]
    for name in tqdm(case_names, desc=f"r211_f1/candidates/{split}"):
        if not anchor_path(args, split, name).exists():
            continue
        print(f"[R211-F1] candidates/{split}: {name}", flush=True)
        image, instance, gt, anchor = load_case(args, split, name)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel, with_surface_distance=args.full_candidate_labels)
        old_metrics = candidate_metrics(args, anchor, gt, gt_cache)
        old_merge = int(old_metrics.get("component_merge_rate") or 0)
        old_instance_merge = component_merge_count(anchor, instance, args.min_overlap_frac) if args.candidate_instance_labels else None
        cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
        for rank, cut in enumerate(cuts[: args.max_components_per_image], start=1):
            rows.append(
                candidate_trial_row(
                    args,
                    split,
                    name,
                    rank,
                    anchor,
                    image,
                    instance,
                    gt,
                    gt_cache,
                    cut,
                    old_metrics,
                    old_merge,
                    old_instance_merge,
                )
            )
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


def matrix(rows: list[dict[str, Any]], feature_keys: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    keys = feature_keys or FEATURE_KEYS
    x = np.asarray([[float(row.get(key) or 0.0) for key in keys] for row in rows], dtype=np.float32)
    y = np.asarray([int(float(row.get("label_positive") or 0.0) > 0.5) for row in rows], dtype=np.int32)
    return x, y


def fit_classifier(args: argparse.Namespace, rows: list[dict[str, Any]]) -> Pipeline:
    x, y = matrix(rows)
    if y.sum() == 0:
        raise RuntimeError("No positive candidate labels; cannot train R211-F1 acceptor.")
    if args.classifier == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
        model = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()), ("clf", clf)])
    else:
        clf = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.04, random_state=args.seed, l2_regularization=0.01)
        model = Pipeline([("impute", SimpleImputer()), ("clf", clf)])
    model.fit(x, y)
    return model


def predict_prob(model: Pipeline, row: dict[str, Any], feature_keys: list[str] | None = None) -> float:
    keys = feature_keys or FEATURE_KEYS
    x = np.asarray([[float(row.get(key) or 0.0) for key in keys]], dtype=np.float32)
    return float(model.predict_proba(x)[0, 1])


def passes_inference_accept_filters(args: argparse.Namespace, row: dict[str, Any]) -> bool:
    if args.max_accept_candidate_rank > 0 and float(row.get("candidate_rank") or 0.0) > args.max_accept_candidate_rank:
        return False
    if float(row.get("gradient_ratio") or 0.0) < args.min_accept_gradient_ratio:
        return False
    if float(row.get("dist_ratio") or 0.0) < args.min_accept_dist_ratio:
        return False
    if float(row.get("slenderness") or 0.0) < args.min_accept_slenderness:
        return False
    if float(row.get("fill") or 0.0) > args.max_accept_fill:
        return False
    return True


def accepted_candidate_summary(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    keys = [
        "candidate_rank",
        "candidate_prob",
        "cut_area",
        "cut_frac_component",
        "split_gain",
        "piece_count",
        "min_piece_frac",
        "slenderness",
        "fill",
        "gradient_ratio",
        "dist_ratio",
    ]
    summary = [{key: float(row[key]) for key in keys if key in row and row[key] is not None} for row in rows]
    return json.dumps(summary, sort_keys=True)


def eval_candidate_candidate_metrics(
    args: argparse.Namespace,
    anchor: np.ndarray,
    image: np.ndarray,
    cut: np.ndarray,
    rank: int,
) -> dict[str, Any]:
    cut = np.logical_and(cut, anchor)
    features = split_features(anchor, cut, image)
    features = add_f3_shape_features(features, anchor, cut)
    features = add_r214_image_context_features(features, image, anchor, cut)
    _, anchor_component_count = ndimage.label(anchor, structure=np.ones((3, 3), dtype=np.uint8))
    features["anchor_component_count"] = float(anchor_component_count)
    features["anchor_area"] = float(anchor.sum())
    features["candidate_rank"] = float(rank)
    return features


def apply_model_to_case(
    args: argparse.Namespace,
    model: Pipeline,
    threshold: float,
    split: str,
    name: str,
    gt_cache: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    image, instance, gt, anchor = load_case(args, split, name)
    old = compute_metrics(anchor, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    old_instance_merge = component_merge_count(anchor, instance, args.min_overlap_frac)
    current = anchor.copy()
    accepted_cut = np.zeros_like(anchor, dtype=bool)
    anchor_pred_component_count = pred_component_count(anchor)
    current_pred_component_count = anchor_pred_component_count
    max_cut_pixels = int(max(args.min_cut_area, round(float(anchor.sum()) * args.max_cut_frac)))
    rows = []
    cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
    for rank, cut in enumerate(cuts[: args.max_components_per_image], start=1):
        cut = np.logical_and(cut, current)
        if int(cut.sum()) < args.min_cut_area:
            continue
        feat = eval_candidate_candidate_metrics(args, current, image, cut, rank)
        feat["candidate_prob"] = predict_prob(model, feat)
        feat["cut_mask"] = cut
        rows.append(feat)
    rows.sort(key=lambda row: float(row["candidate_prob"]), reverse=True)
    accepted_probs = []
    accepted_rows = []
    for row in rows:
        cut = row["cut_mask"]
        if float(row["candidate_prob"]) < threshold:
            continue
        if int(accepted_cut.sum()) + int(cut.sum()) > max_cut_pixels:
            continue
        if float(row.get("cut_frac_component") or 1.0) > args.max_cut_frac_component:
            continue
        if not passes_inference_accept_filters(args, row):
            continue
        trial = np.logical_and(current, ~cut)
        trial_pred_component_count = pred_component_count(trial)
        if trial_pred_component_count - anchor_pred_component_count > args.max_pred_component_increase:
            continue
        if trial_pred_component_count - current_pred_component_count > args.max_step_component_increase:
            continue
        accepted_cut |= cut
        current = trial
        current_pred_component_count = trial_pred_component_count
        accepted_probs.append(float(row["candidate_prob"]))
        accepted_rows.append(row)
    pred = current if accepted_cut.any() else anchor
    new = compute_metrics(pred, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    new_instance_merge = component_merge_count(pred, instance, args.min_overlap_frac)
    rec: dict[str, Any] = {
        "image": name,
        "accepted": float(bool(accepted_cut.any())),
        "accepted_cut_pixels": float(accepted_cut.sum()),
        "accepted_components": float(len(accepted_probs)),
        "accepted_mean_prob": float(np.mean(accepted_probs)) if accepted_probs else None,
        "accepted_candidate_summary": accepted_candidate_summary(accepted_rows),
        "anchor_pred_component_count": float(anchor_pred_component_count),
        "r211_f1_pred_component_count": float(pred_component_count(pred)),
        "delta_pred_component_count": float(pred_component_count(pred) - anchor_pred_component_count),
        "num_candidates": float(len(cuts)),
        "num_candidates_scored": float(len(rows)),
        "anchor_instance_merge_count": float(old_instance_merge),
        "r211_f1_instance_merge_count": float(new_instance_merge),
        "delta_instance_merge_count": float(new_instance_merge - old_instance_merge),
        **{f"anchor_{key}": value for key, value in old.items()},
        **{f"r211_f1_{key}": value for key, value in new.items()},
    }
    for key in old:
        rec[f"delta_{key}"] = metric_delta(new, old, key)
    return pred, rec


def prepare_cached_eval_case(
    args: argparse.Namespace,
    model: Pipeline,
    split: str,
    name: str,
) -> dict[str, Any]:
    image, instance, gt, anchor = load_case(args, split, name)
    gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel, with_surface_distance=True)
    old = compute_metrics(anchor, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    old_instance_merge = component_merge_count(anchor, instance, args.min_overlap_frac)
    anchor_pred_component_count = pred_component_count(anchor)
    max_cut_pixels = int(max(args.min_cut_area, round(float(anchor.sum()) * args.max_cut_frac)))
    rows = []
    cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
    for rank, cut in enumerate(cuts[: args.max_components_per_image], start=1):
        cut = np.logical_and(cut, anchor)
        if int(cut.sum()) < args.min_cut_area:
            continue
        feat = eval_candidate_candidate_metrics(args, anchor, image, cut, rank)
        feat["candidate_prob"] = predict_prob(model, feat)
        feat["cut_mask"] = cut
        rows.append(feat)
    rows.sort(key=lambda row: float(row["candidate_prob"]), reverse=True)
    return {
        "name": name,
        "instance": instance,
        "gt": gt,
        "gt_cache": gt_cache,
        "anchor": anchor,
        "old": old,
        "old_instance_merge": old_instance_merge,
        "anchor_pred_component_count": anchor_pred_component_count,
        "max_cut_pixels": max_cut_pixels,
        "rows": rows,
        "num_candidates": float(len(cuts)),
    }


def apply_cached_threshold(args: argparse.Namespace, case: dict[str, Any], threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    anchor = case["anchor"]
    current = anchor.copy()
    accepted_cut = np.zeros_like(anchor, dtype=bool)
    accepted_probs = []
    accepted_rows = []
    current_pred_component_count = int(case["anchor_pred_component_count"])
    for row in case["rows"]:
        cut = np.logical_and(row["cut_mask"], current)
        if int(cut.sum()) < args.min_cut_area:
            continue
        if float(row["candidate_prob"]) < threshold:
            continue
        if int(accepted_cut.sum()) + int(cut.sum()) > int(case["max_cut_pixels"]):
            continue
        if float(row.get("cut_frac_component") or 1.0) > args.max_cut_frac_component:
            continue
        if not passes_inference_accept_filters(args, row):
            continue
        trial = np.logical_and(current, ~cut)
        trial_pred_component_count = pred_component_count(trial)
        if trial_pred_component_count - int(case["anchor_pred_component_count"]) > args.max_pred_component_increase:
            continue
        if trial_pred_component_count - current_pred_component_count > args.max_step_component_increase:
            continue
        accepted_cut |= cut
        current = trial
        current_pred_component_count = trial_pred_component_count
        accepted_probs.append(float(row["candidate_prob"]))
        accepted_rows.append(row)

    pred = current if accepted_cut.any() else anchor
    new = compute_metrics(
        pred,
        case["gt"],
        args.boundary_kernel,
        args.gap_kernel,
        args.surface_tol,
        args.surface_tol_extra,
        case["gt_cache"],
    )
    new_instance_merge = component_merge_count(pred, case["instance"], args.min_overlap_frac)
    old = case["old"]
    rec: dict[str, Any] = {
        "image": case["name"],
        "accepted": float(bool(accepted_cut.any())),
        "accepted_cut_pixels": float(accepted_cut.sum()),
        "accepted_components": float(len(accepted_probs)),
        "accepted_mean_prob": float(np.mean(accepted_probs)) if accepted_probs else None,
        "accepted_candidate_summary": accepted_candidate_summary(accepted_rows),
        "anchor_pred_component_count": float(case["anchor_pred_component_count"]),
        "r211_f1_pred_component_count": float(pred_component_count(pred)),
        "delta_pred_component_count": float(pred_component_count(pred) - int(case["anchor_pred_component_count"])),
        "num_candidates": case["num_candidates"],
        "num_candidates_scored": float(len(case["rows"])),
        "anchor_instance_merge_count": float(case["old_instance_merge"]),
        "r211_f1_instance_merge_count": float(new_instance_merge),
        "delta_instance_merge_count": float(new_instance_merge - case["old_instance_merge"]),
        **{f"anchor_{key}": value for key, value in old.items()},
        **{f"r211_f1_{key}": value for key, value in new.items()},
    }
    for key in old:
        rec[f"delta_{key}"] = metric_delta(new, old, key)
    return pred, rec


def mean_record(records: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in records for key in row if key != "image"})
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [row.get(key) for row in records]
        finite = [float(v) for v in vals if isinstance(v, (float, int))]
        out[key] = float(np.mean(finite)) if finite else None
    return out


def gate_pass(args: argparse.Namespace, mean: dict[str, float | None]) -> bool:
    base_pass = bool(
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
    if not base_pass:
        return False
    if not args.strict_boundary_gate:
        return True
    return bool(
        (mean.get("delta_boundary_iou") or 0.0) >= 0.0
        and (mean.get("delta_boundary_f1") or 0.0) >= 0.0
        and (mean.get("delta_surface_dice_2px") or 0.0) >= 0.0
    )


def threshold_audit_rows(threshold: float, gate: bool, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if float(rec.get("accepted") or 0.0) <= 0.0:
            continue
        rows.append(
            {
                "threshold": threshold,
                "gate_pass": float(gate),
                "image": rec.get("image"),
                "accepted": rec.get("accepted"),
                "accepted_components": rec.get("accepted_components"),
                "accepted_cut_pixels": rec.get("accepted_cut_pixels"),
                "accepted_mean_prob": rec.get("accepted_mean_prob"),
                "accepted_candidate_summary": rec.get("accepted_candidate_summary"),
                "delta_dice": rec.get("delta_dice"),
                "delta_iou": rec.get("delta_iou"),
                "delta_recall": rec.get("delta_recall"),
                "delta_boundary_iou": rec.get("delta_boundary_iou"),
                "delta_boundary_f1": rec.get("delta_boundary_f1"),
                "delta_surface_dice_2px": rec.get("delta_surface_dice_2px"),
                "delta_surface_dice_5px": rec.get("delta_surface_dice_5px"),
                "delta_hd95_px": rec.get("delta_hd95_px"),
                "delta_assd_px": rec.get("delta_assd_px"),
                "delta_gap_region_fp_rate": rec.get("delta_gap_region_fp_rate"),
                "delta_component_merge_rate": rec.get("delta_component_merge_rate"),
                "delta_component_count_mae": rec.get("delta_component_count_mae"),
                "delta_instance_merge_count": rec.get("delta_instance_merge_count"),
                "delta_pred_component_count": rec.get("delta_pred_component_count"),
            }
        )
    return rows


def evaluate_thresholds(
    args: argparse.Namespace, model: Pipeline, split: str
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    case_names = names(args.raw_root, args.dataset, split)[: args.limit_val or None]
    thresholds = parse_nums(args.threshold_grid, float)
    records_by_threshold: dict[float, list[dict[str, Any]]] = {threshold: [] for threshold in thresholds}
    preds_by_threshold: dict[float, list[tuple[str, np.ndarray]]] = {threshold: [] for threshold in thresholds}
    for name in tqdm(case_names, desc=f"r211_f1/eval/{split}/cached", leave=False):
        if not anchor_path(args, split, name).exists():
            continue
        print(f"[R211-F1] eval/{split}/cached: {name}", flush=True)
        case = prepare_cached_eval_case(args, model, split, name)
        for threshold in thresholds:
            pred, rec = apply_cached_threshold(args, case, threshold)
            records_by_threshold[threshold].append(rec)
            preds_by_threshold[threshold].append((name, pred))

    best: dict[str, Any] | None = None
    best_records: list[dict[str, Any]] = []
    grid: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        records = records_by_threshold[threshold]
        preds = preds_by_threshold[threshold]
        mean = mean_record(records)
        item = {"threshold": threshold, "mean": mean, "gate_pass": gate_pass(args, mean)}
        grid.append(item)
        audit_rows.extend(threshold_audit_rows(threshold, bool(item["gate_pass"]), records))
        score = (
            item["gate_pass"],
            (mean.get("accepted") or 0.0) > 0.0,
            -(mean.get("delta_component_count_mae") or 0.0),
            -(mean.get("delta_gap_region_fp_rate") or 0.0),
            -(mean.get("delta_instance_merge_count") or 0.0),
            mean.get("delta_boundary_iou") or 0.0,
            mean.get("delta_dice") or -999.0,
        )
        if best is None or score > best["_score"]:
            best = {**item, "_score": score, "_preds": preds}
            best_records = records
    assert best is not None
    preds = best.pop("_preds")
    best.pop("_score", None)
    out_dir = args.ablations_root / args.output_exp / args.dataset / split / "masks"
    for name, pred in preds:
        write_mask(out_dir / name, pred)
    return best, grid, best_records, audit_rows


def class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = [int(float(row.get("label_positive") or 0.0) > 0.5) for row in rows]
    return {
        "num_candidates": len(rows),
        "num_positive": int(sum(y)),
        "positive_rate": float(sum(y) / max(1, len(y))),
    }


def main() -> None:
    args = parse_args()
    train_rows: list[dict[str, Any]] = []
    if str(args.eval_only_checkpoint):
        payload = joblib.load(args.eval_only_checkpoint)
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        checkpoint_source = str(args.eval_only_checkpoint)
    else:
        train_rows = build_candidate_table(args, args.train_split, args.limit_train)
        write_csv(args.train_candidate_csv, train_rows)
        model = fit_classifier(args, train_rows)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_keys": FEATURE_KEYS, "args": vars(args)}, args.checkpoint)
        checkpoint_source = str(args.checkpoint)
    val_rows = build_candidate_table(args, args.val_split, args.limit_val)
    write_csv(args.val_candidate_csv, val_rows)

    best, grid, records, threshold_rows = evaluate_thresholds(args, model, args.val_split)
    summary = {
        "run_id": "R211-F1",
        "dataset": args.dataset,
        "split": args.val_split,
        "clean_test_v2_used": False,
        "purpose": "learned GT-free acceptance gate for R210-F1 candidates",
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "classifier": args.classifier,
        "label_mode": args.label_mode,
        "strict_boundary_gate": bool(args.strict_boundary_gate),
        "checkpoint": checkpoint_source,
        "eval_only_checkpoint": str(args.eval_only_checkpoint) if str(args.eval_only_checkpoint) else None,
        "features": FEATURE_KEYS,
        "train_candidate_summary": class_summary(train_rows) if train_rows else None,
        "val_candidate_summary": class_summary(val_rows),
        "num_evaluated": len(records),
        "val_gate_pass": best["gate_pass"],
        "best": best,
        "grid": grid,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.per_image_csv, records)
    if str(args.threshold_audit_csv):
        write_csv(args.threshold_audit_csv, threshold_rows)
    print(json.dumps({"val_gate_pass": summary["val_gate_pass"], "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
