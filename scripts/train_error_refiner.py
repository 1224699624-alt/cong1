#!/usr/bin/env python3
"""Train an under/over-segmentation aware residual mask refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight residual refiner for YOLO-SAM masks.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--baseline-exp", default="zero_shot_box_only_conf020_pad005")
    parser.add_argument("--refined-exp", default="zero_shot_mask_to_prompt_refine_pad003_neg2_k5")
    parser.add_argument("--refined-exp-2", default=None, help="Optional second refined candidate mask directory for multi-candidate fusion.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output", default="outputs/error_refiner/TSRS_RSNA-Epiphysis/best.pt")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--aux-loss-weight", type=float, default=0.35)
    parser.add_argument("--contrast-loss-weight", type=float, default=0.10)
    parser.add_argument("--cldice-loss-weight", type=float, default=0.0)
    parser.add_argument("--cldice-iters", type=int, default=3)
    parser.add_argument("--instance-sep-loss-weight", type=float, default=0.20)
    parser.add_argument("--stage2-boundary-loss-weight", type=float, default=0.25)
    parser.add_argument("--structure-keep-loss-weight", type=float, default=1.0)
    parser.add_argument("--structure-gap-loss-weight", type=float, default=1.1)
    parser.add_argument(
        "--target-mode",
        choices=["residual", "mec", "residual_mec_gated", "boundary_band_fgbg", "araa_guided_precision", "allstack_boundary_trim", "allstack_boundary_trim_guarded", "allstack_anatomy_roi_only", "allstack_anatomy_roi_prior_only", "allstack_anatomy_roi_aic_prior_only", "allstack_anatomy_roi_aic_refiner", "allstack_anatomy_roi_refiner", "allstack_anatomy_roi_interaction_refiner", "allstack_anatomy_roi_instance_sep_refiner", "allstack_anatomy_roi_stage2_refiner", "allstack_anatomy_roi_twostage_boundary_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3", "allstack_anatomy_roi_uncertainty_refiner", "allstack_anatomy_roi_refiner_uncertainty_gate", "allstack_anatomy_roi_selector_refiner_v3", "allstack_anatomy_roi_boundary_prefgate_trimfirst"],
        default="residual",
    )
    parser.add_argument("--gate-disagreement-threshold", type=float, default=0.10)
    parser.add_argument("--gate-boundary-weight", type=float, default=0.75)
    parser.add_argument("--gate-uncertainty-weight", type=float, default=0.35)
    parser.add_argument("--gate-smooth-kernel", type=int, default=5)
    parser.add_argument("--boundary-band-kernel", type=int, default=9)
    parser.add_argument("--boundary-band-weight", type=float, default=1.0)
    parser.add_argument("--boundary-fg-weight", type=float, default=0.60)
    parser.add_argument("--boundary-bg-weight", type=float, default=0.50)
    parser.add_argument("--selector-loss-weight", type=float, default=0.20)
    parser.add_argument("--trim-risk-weight", type=float, default=0.70)
    parser.add_argument("--trim-strength-weight", type=float, default=0.55)
    parser.add_argument("--selection-dice-weight", type=float, default=0.30)
    parser.add_argument("--selection-iou-weight", type=float, default=0.25)
    parser.add_argument("--selection-precision-weight", type=float, default=0.25)
    parser.add_argument("--selection-boundary-weight", type=float, default=0.20)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        files.extend(image_dir.glob(f"*{extension}"))
    return sorted(files)


def read_image_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.float32)


def resize_float(array: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(array.astype(np.float32), (size, size), interpolation=interpolation)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (dilated ^ eroded).astype(np.float32)


def distance_prior(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    fg_dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    bg_dist = cv2.distanceTransform(1 - mask_u8, cv2.DIST_L2, 5)
    signed = fg_dist - bg_dist
    denom = np.percentile(np.abs(signed), 95)
    if denom < 1e-6:
        return np.zeros_like(mask, dtype=np.float32)
    return np.clip(signed / denom, -1.0, 1.0).astype(np.float32)


def gradient_map(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    denom = np.percentile(grad, 99)
    if denom < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip(grad / denom, 0.0, 1.0).astype(np.float32)


def core_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    k = max(3, int(kernel_size) | 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    mask_u8 = (mask > 0.5).astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return eroded.astype(np.float32)


def principal_axis_prior(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 8:
        return np.zeros_like(mask, dtype=np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    grid = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=-1) - center
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)
    dist = np.abs(grid[..., 0] * normal[0] + grid[..., 1] * normal[1])
    scale = max(4.0, float(np.sqrt(np.max(eigvals) + 1e-6)) * 0.75)
    prior = np.exp(-(dist * dist) / (2.0 * scale * scale))
    return prior.astype(np.float32)


def gaussian_center_prior(mask: np.ndarray, min_sigma: float = 6.0, scale_factor: float = 0.30) -> np.ndarray:
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 8:
        return np.zeros_like(mask, dtype=np.float32)
    cx = float(xs.mean())
    cy = float(ys.mean())
    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    sigma_x = max(min_sigma, bw * scale_factor)
    sigma_y = max(min_sigma, bh * scale_factor)
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    norm_x = ((xx.astype(np.float32) - cx) ** 2) / (2.0 * sigma_x * sigma_x)
    norm_y = ((yy.astype(np.float32) - cy) ** 2) / (2.0 * sigma_y * sigma_y)
    return np.exp(-(norm_x + norm_y)).astype(np.float32)


def ellipse_shape_prior(mask: np.ndarray, margin: float = 1.20) -> np.ndarray:
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 8:
        return np.zeros_like(mask, dtype=np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    center = coords.mean(axis=0)
    centered = coords - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    radii = np.sqrt(eigvals) * max(1.0, margin)
    rotation = eigvecs.astype(np.float32)
    yy, xx = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    grid = np.stack([xx.astype(np.float32), yy.astype(np.float32)], axis=-1) - center
    aligned = grid @ rotation
    normalized = (aligned[..., 0] / max(radii[0], 1e-3)) ** 2 + (aligned[..., 1] / max(radii[1], 1e-3)) ** 2
    return np.exp(-0.5 * normalized).astype(np.float32)


def build_aic_like_anatomy_score(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    structure_seed = core_mask(mask, kernel_size=5)
    axis_seed = principal_axis_prior(mask)
    center_seed = gaussian_center_prior(mask)
    shape_seed = ellipse_shape_prior(mask)
    grad_seed = gradient_map(image)
    anatomy_score = (
        0.30 * structure_seed
        + 0.22 * axis_seed
        + 0.20 * center_seed
        + 0.18 * shape_seed
        + 0.10 * grad_seed
    )
    return np.clip(anatomy_score, 0.0, 1.0).astype(np.float32)


def dilate_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = max(1, int(kernel_size))
    kernel = np.ones((k, k), dtype=np.uint8)
    mask_u8 = (mask > 0.5).astype(np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=1).astype(np.float32)


def filter_roi_support(
    support_mask: np.ndarray,
    anatomy_score: np.ndarray | None = None,
    min_area_ratio: float = 0.08,
    max_center_distance_factor: float = 1.75,
    max_aspect_ratio: float = 4.5,
    dilation_kernel: int = 3,
) -> np.ndarray:
    """
    Soft anatomy-aware ROI filtering inspired by coarse-to-fine anatomy-aware
    post-processing. The goal is to keep all candidate evidence, but assign lower
    influence to remote or implausible fragments before bbox crop.
    """
    support_u8 = (support_mask > 0.5).astype(np.uint8)
    if support_u8.sum() == 0:
        return support_mask.astype(np.float32)

    merged = dilate_mask(support_u8, kernel_size=dilation_kernel).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged, connectivity=8)
    if num_labels <= 1:
        return merged.astype(np.float32)

    component_indices = [idx for idx in range(1, num_labels)]
    component_indices.sort(key=lambda idx: int(stats[idx, cv2.CC_STAT_AREA]), reverse=True)
    main_idx = component_indices[0]
    main_area = float(stats[main_idx, cv2.CC_STAT_AREA])
    main_center = centroids[main_idx].astype(np.float32)
    main_w = float(stats[main_idx, cv2.CC_STAT_WIDTH])
    main_h = float(stats[main_idx, cv2.CC_STAT_HEIGHT])
    main_scale = max(8.0, 0.5 * np.hypot(main_w, main_h))
    min_area = max(8.0, main_area * float(min_area_ratio))
    support = np.zeros_like(merged, dtype=np.float32)
    anatomy_score = anatomy_score.astype(np.float32) if anatomy_score is not None else None
    for idx in component_indices:
        area = float(stats[idx, cv2.CC_STAT_AREA])
        w = max(1.0, float(stats[idx, cv2.CC_STAT_WIDTH]))
        h = max(1.0, float(stats[idx, cv2.CC_STAT_HEIGHT]))
        aspect = max(w / h, h / w)
        center = centroids[idx].astype(np.float32)
        center_distance = float(np.linalg.norm(center - main_center))
        area_score = 1.0 - np.exp(-area / max(min_area, 1.0))
        distance_sigma = max(4.0, max_center_distance_factor * main_scale)
        distance_score = np.exp(-(center_distance * center_distance) / (2.0 * distance_sigma * distance_sigma))
        aspect_score = np.exp(-max(0.0, aspect - 1.0) / max(max_aspect_ratio, 1e-6))
        anchor_bonus = 0.18 if idx == main_idx else 0.0
        anatomy_component_score = 1.0
        if anatomy_score is not None:
            region_vals = anatomy_score[labels == idx]
            if region_vals.size > 0:
                anatomy_component_score = float(np.clip(np.mean(region_vals), 0.25, 1.0))
        confidence = np.clip(
            0.18 + 0.56 * area_score * distance_score * aspect_score + 0.18 * anatomy_component_score + anchor_bonus,
            0.18,
            1.0,
        )
        support[labels == idx] = confidence

    support = np.maximum(support, 0.15 * merged.astype(np.float32))
    support = cv2.GaussianBlur(support, ksize=(0, 0), sigmaX=0.9, sigmaY=0.9)
    return np.clip(support, 0.0, 1.0).astype(np.float32)


def compute_roi_box(mask: np.ndarray, shape: tuple[int, int], pad_ratio: float = 0.20, support_threshold: float = 0.35) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > support_threshold)
    h, w = shape
    if len(xs) == 0:
        return 0, 0, w, h
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    bw = x1 - x0
    bh = y1 - y0
    side = int(np.ceil(max(bw, bh) * (1.0 + pad_ratio)))
    side = max(side, max(bw, bh) + 12)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1 = x0 + side
    y1 = y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        shift = x1 - w
        x0 = max(0, x0 - shift)
        x1 = w
    if y1 > h:
        shift = y1 - h
        y0 = max(0, y0 - shift)
        y1 = h
    return x0, y0, x1, y1


def crop_array(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return array[y0:y1, x0:x1]


def prepare_roi_sample(
    image: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    refined_2: np.ndarray | None,
    gt: np.ndarray | None,
    size: int,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, int] | None]:
    if target_mode not in {"allstack_anatomy_roi_only", "allstack_anatomy_roi_prior_only", "allstack_anatomy_roi_aic_prior_only", "allstack_anatomy_roi_aic_refiner", "allstack_anatomy_roi_refiner", "allstack_anatomy_roi_interaction_refiner", "allstack_anatomy_roi_instance_sep_refiner", "allstack_anatomy_roi_stage2_refiner", "allstack_anatomy_roi_twostage_boundary_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3", "allstack_anatomy_roi_uncertainty_refiner", "allstack_anatomy_roi_refiner_uncertainty_gate", "allstack_anatomy_roi_selector_refiner_v3", "allstack_anatomy_roi_boundary_prefgate_trimfirst"}:
        image_r = resize_float(image, size, cv2.INTER_LINEAR)
        baseline_r = resize_float(baseline, size, cv2.INTER_NEAREST)
        refined_r = resize_float(refined, size, cv2.INTER_NEAREST)
        refined_2_r = resize_float(refined_2, size, cv2.INTER_NEAREST) if refined_2 is not None else None
        gt_r = resize_float(gt, size, cv2.INTER_NEAREST) if gt is not None else None
        return image_r, baseline_r, refined_r, refined_2_r, gt_r, None

    union = np.maximum(np.maximum(baseline, refined), refined_2) if refined_2 is not None else np.maximum(baseline, refined)
    anatomy_score = None
    if target_mode == "allstack_anatomy_roi_prior_only":
        structure_seed = core_mask(union, kernel_size=5)
        axis_seed = principal_axis_prior(union)
        grad_seed = gradient_map(image)
        anatomy_score = np.clip(0.50 * structure_seed + 0.35 * axis_seed + 0.15 * grad_seed, 0.0, 1.0)
    elif target_mode in {"allstack_anatomy_roi_aic_prior_only", "allstack_anatomy_roi_aic_refiner"}:
        anatomy_score = build_aic_like_anatomy_score(union, image)
    filtered_union = filter_roi_support(union, anatomy_score=anatomy_score)
    box = compute_roi_box(filtered_union, image.shape, support_threshold=0.35)
    image_c = crop_array(image, box)
    baseline_c = crop_array(baseline, box)
    refined_c = crop_array(refined, box)
    refined_2_c = crop_array(refined_2, box) if refined_2 is not None else None
    gt_c = crop_array(gt, box) if gt is not None else None
    image_r = resize_float(image_c, size, cv2.INTER_LINEAR)
    baseline_r = resize_float(baseline_c, size, cv2.INTER_NEAREST)
    refined_r = resize_float(refined_c, size, cv2.INTER_NEAREST)
    refined_2_r = resize_float(refined_2_c, size, cv2.INTER_NEAREST) if refined_2_c is not None else None
    gt_r = resize_float(gt_c, size, cv2.INTER_NEAREST) if gt_c is not None else None
    meta = {"x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3], "h": image.shape[0], "w": image.shape[1]}
    return image_r, baseline_r, refined_r, refined_2_r, gt_r, meta


def masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (image.astype(np.float32) * (mask > 0.5).astype(np.float32)).astype(np.float32)


def boundary_band(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    k = max(3, int(kernel_size) | 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    mask_u8 = (mask > 0.5).astype(np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (dilated ^ eroded).astype(np.float32)


def soft_selector_target(
    baseline: np.ndarray,
    refined: np.ndarray,
    refined_2: np.ndarray | None,
    gt: np.ndarray,
) -> np.ndarray:
    candidate_list = [baseline, refined, refined if refined_2 is None else refined_2]
    score_maps = []
    for candidate in candidate_list:
        error = np.abs(candidate.astype(np.float32) - gt.astype(np.float32))
        score_maps.append(np.exp(-4.0 * error).astype(np.float32))
    scores = np.stack(score_maps, axis=0)
    denom = np.clip(scores.sum(axis=0, keepdims=True), 1e-6, None)
    return (scores / denom).astype(np.float32)


def compute_instance_separation_targets(gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_mask = (gt > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    core = np.zeros_like(gt, dtype=np.float32)
    sep = np.zeros_like(gt, dtype=np.float32)
    hover_x = np.zeros_like(gt, dtype=np.float32)
    hover_y = np.zeros_like(gt, dtype=np.float32)
    if num_labels <= 1:
        return core, sep, hover_x, hover_y

    h, w = gt.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dilated_sum = np.zeros_like(gt, dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for label in range(1, num_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component_u8 = component.astype(np.uint8)
        dist = cv2.distanceTransform(component_u8, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist > 1e-6:
            component_core = dist >= max(1.0, 0.35 * max_dist)
            core[component_core] = 1.0
        cx, cy = centroids[label]
        xs = xx[component].astype(np.float32)
        ys = yy[component].astype(np.float32)
        if xs.size > 0:
            dx = xs - float(cx)
            dy = ys - float(cy)
            max_abs = max(float(np.max(np.abs(dx))), float(np.max(np.abs(dy))), 1.0)
            hover_x[component] = dx / max_abs
            hover_y[component] = dy / max_abs
        dilated_sum += cv2.dilate(component_u8, kernel, iterations=1).astype(np.float32)
    sep[(dilated_sum >= 2.0) & (gt <= 0.5)] = 1.0
    return core, sep, hover_x.astype(np.float32), hover_y.astype(np.float32)


def compute_structure_sensitive_targets(
    gt: np.ndarray,
    baseline: np.ndarray | None = None,
    refined: np.ndarray | None = None,
    refined_2: np.ndarray | None = None,
    prediction_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_mask = (gt > 0.5).astype(np.uint8)
    keep_band = np.zeros_like(gt, dtype=np.float32)
    gap_band = np.zeros_like(gt, dtype=np.float32)
    if gt_mask.sum() == 0:
        return keep_band, gap_band, np.zeros_like(gt, dtype=np.float32)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_mask, connectivity=8)
    dilated_sum = np.zeros_like(gt, dtype=np.float32)
    keep_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for label in range(1, num_labels):
        component = (labels == label).astype(np.uint8)
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        dist = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist > 1e-6:
            thin_inner = max(1.0, min(2.5, 0.45 * max_dist))
            keep_band[(component > 0) & (dist <= thin_inner)] = 1.0
        else:
            keep_band[component > 0] = 1.0
        dilated_sum += cv2.dilate(component, gap_kernel, iterations=1).astype(np.float32)

    touching_gap = (dilated_sum >= 2.0) & (gt_mask == 0)
    if touching_gap.any():
        gap_band[touching_gap] = 1.0

    global_dist_in = cv2.distanceTransform(gt_mask, cv2.DIST_L2, 5)
    global_dist_out = cv2.distanceTransform(1 - gt_mask, cv2.DIST_L2, 5)
    keep_band = keep_band * (global_dist_in <= 3.0).astype(np.float32)
    gap_band = gap_band * (global_dist_out <= 4.0).astype(np.float32)

    keep_band = cv2.dilate(keep_band.astype(np.uint8), keep_kernel, iterations=1).astype(np.float32)
    keep_band = keep_band * gt_mask.astype(np.float32)

    if prediction_aware and baseline is not None and refined is not None:
        candidate_union = np.maximum(baseline.astype(np.float32), refined.astype(np.float32))
        if refined_2 is not None:
            candidate_union = np.maximum(candidate_union, refined_2.astype(np.float32))
        candidate_union_u8 = (candidate_union > 0.5).astype(np.uint8)
        coarse_missing = ((gt_mask > 0) & (candidate_union_u8 == 0)).astype(np.uint8)
        coarse_missing = cv2.dilate(coarse_missing, keep_kernel, iterations=1).astype(np.float32)
        keep_support = cv2.dilate(candidate_union_u8, keep_kernel, iterations=1).astype(np.float32)
        keep_band = keep_band * np.clip(0.35 + 0.65 * np.maximum(coarse_missing, 1.0 - keep_support * gt_mask.astype(np.float32)), 0.0, 1.0)

        gap_intrusion = cv2.dilate(candidate_union_u8, gap_kernel, iterations=1).astype(np.float32)
        gap_dispute = np.zeros_like(gt, dtype=np.float32)
        if refined_2 is not None:
            gap_dispute = np.maximum(np.abs(refined - baseline), np.abs(refined_2 - refined)).astype(np.float32)
        else:
            gap_dispute = np.abs(refined - baseline).astype(np.float32)
        gap_band = gap_band * np.clip(0.25 + 0.75 * np.maximum(gap_intrusion, cv2.dilate((gap_dispute > 0.0).astype(np.uint8), keep_kernel, iterations=1).astype(np.float32)), 0.0, 1.0)

    structure_band = np.clip(np.maximum(keep_band, gap_band), 0.0, 1.0).astype(np.float32)
    return keep_band.astype(np.float32), gap_band.astype(np.float32), structure_band


def build_feature_stack(
    image: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    refined_2: np.ndarray | None = None,
    include_anatomy: bool = False,
) -> np.ndarray:
    if refined_2 is None:
        union = np.maximum(baseline, refined)
        intersection = baseline * refined
        disagreement = np.abs(refined - baseline)
        refined_2_safe = np.zeros_like(refined, dtype=np.float32)
        refined_2_boundary = np.zeros_like(refined, dtype=np.float32)
        refined_2_distance = np.zeros_like(refined, dtype=np.float32)
        refined_pair_boundary = np.zeros_like(refined, dtype=np.float32)
        baseline_only = np.clip(baseline - refined, 0.0, 1.0)
        refined_only = np.clip(refined - baseline, 0.0, 1.0)
        refined_2_only = np.zeros_like(refined, dtype=np.float32)
        consensus = 1.0 - disagreement
    else:
        union = np.maximum(np.maximum(baseline, refined), refined_2)
        intersection = baseline * refined * refined_2
        disagreement = np.maximum(np.abs(refined - baseline), np.abs(refined_2 - baseline))
        refined_2_safe = refined_2.astype(np.float32)
        refined_2_boundary = boundary_map(refined_2_safe)
        refined_2_distance = distance_prior(refined_2_safe)
        refined_pair_boundary = boundary_map(np.abs(refined_2_safe - refined))
        baseline_only = np.clip(baseline - np.maximum(refined, refined_2_safe), 0.0, 1.0)
        refined_only = np.clip(refined - np.maximum(baseline, refined_2_safe), 0.0, 1.0)
        refined_2_only = np.clip(refined_2_safe - np.maximum(baseline, refined), 0.0, 1.0)
        consensus = 1.0 - (
            np.abs(baseline - refined) + np.abs(baseline - refined_2_safe) + np.abs(refined - refined_2_safe)
        ) / 3.0

    baseline_boundary = boundary_map(baseline)
    refined_boundary = boundary_map(refined)
    disagreement_boundary = boundary_map(disagreement)
    refined_distance = distance_prior(refined)
    positive_roi = masked_image(image, intersection)
    negative_roi = masked_image(image, disagreement)
    candidate_band = boundary_band(union, kernel_size=7)
    inner_band = (candidate_band * union).astype(np.float32)
    outer_band = (candidate_band * (1.0 - union)).astype(np.float32)
    consensus = np.clip(consensus, 0.0, 1.0).astype(np.float32)

    channel_list = [
        image,
        baseline,
        refined,
        union,
        intersection,
        disagreement,
        positive_roi,
        negative_roi,
        baseline_boundary,
        refined_boundary,
        disagreement_boundary,
        refined_distance,
        refined_2_safe,
        masked_image(image, refined_2_safe),
        refined_pair_boundary,
        refined_2_boundary,
        refined_2_distance,
        baseline_only,
        refined_only,
        refined_2_only,
        inner_band,
        outer_band,
        consensus,
    ]
    if include_anatomy:
        anatomy_union = union
        channel_list.extend(
            [
                gradient_map(image),
                core_mask(anatomy_union, kernel_size=5),
                boundary_band(anatomy_union, kernel_size=7),
                principal_axis_prior(anatomy_union),
            ]
        )
    return np.stack(channel_list, axis=0).astype(np.float32)


class ErrorRefinerDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        ablations_root: Path,
        dataset: str,
        split: str,
        baseline_exp: str,
        refined_exp: str,
        refined_exp_2: str | None,
        img_size: int,
        target_mode: str,
        boundary_band_kernel: int,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.target_mode = target_mode
        self.boundary_band_kernel = boundary_band_kernel
        self.image_dir = raw_root / dataset / split
        self.label_dir = raw_root / dataset / f"{split}_labels"
        self.baseline_dir = ablations_root / baseline_exp / dataset / split / "masks"
        self.refined_dir = ablations_root / refined_exp / dataset / split / "masks"
        self.refined_dir_2 = ablations_root / refined_exp_2 / dataset / split / "masks" if refined_exp_2 else None
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image split not found: {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label split not found: {self.label_dir}")
        mask_dirs = [self.baseline_dir, self.refined_dir]
        if self.refined_dir_2 is not None:
            mask_dirs.append(self.refined_dir_2)
        for mask_dir in mask_dirs:
            if not mask_dir.exists():
                raise FileNotFoundError(f"Candidate mask dir not found: {mask_dir}")
        self.image_files = find_image_files(self.image_dir)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_files[index]
        stem = image_path.stem
        image = read_image_gray(image_path)
        gt = read_binary_mask(self.label_dir / f"{stem}.png")
        baseline = read_binary_mask(self.baseline_dir / f"{stem}.png")
        refined = read_binary_mask(self.refined_dir / f"{stem}.png")
        refined_2 = read_binary_mask(self.refined_dir_2 / f"{stem}.png") if self.refined_dir_2 is not None else None

        image, baseline, refined, refined_2, gt, _ = prepare_roi_sample(
            image,
            baseline,
            refined,
            refined_2,
            gt,
            self.img_size,
            self.target_mode,
        )

        missing = ((gt > 0.5) & (refined <= 0.5)).astype(np.float32)
        excess = ((refined > 0.5) & (gt <= 0.5)).astype(np.float32)
        channels = build_feature_stack(
            image,
            baseline,
            refined,
            refined_2=refined_2,
            include_anatomy=self.target_mode in {"allstack_anatomy_roi_only", "allstack_anatomy_roi_prior_only", "allstack_anatomy_roi_aic_prior_only", "allstack_anatomy_roi_aic_refiner", "allstack_anatomy_roi_refiner", "allstack_anatomy_roi_interaction_refiner", "allstack_anatomy_roi_instance_sep_refiner", "allstack_anatomy_roi_stage2_refiner", "allstack_anatomy_roi_twostage_boundary_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3", "allstack_anatomy_roi_uncertainty_refiner", "allstack_anatomy_roi_refiner_uncertainty_gate", "allstack_anatomy_roi_selector_refiner_v3", "allstack_anatomy_roi_boundary_prefgate_trimfirst"},
        )
        selector_target = soft_selector_target(baseline, refined, refined_2, gt)
        band = boundary_band(refined if refined_2 is None else np.maximum(refined, refined_2), self.boundary_band_kernel)
        fg_band_target = (missing * band).astype(np.float32)
        bg_band_target = (excess * band).astype(np.float32)
        core_target, sep_target, hover_x, hover_y = compute_instance_separation_targets(gt)
        keep_bone_target, cut_gap_target, structure_band_target = compute_structure_sensitive_targets(
            gt,
            baseline=baseline,
            refined=refined,
            refined_2=refined_2,
            prediction_aware=self.target_mode in {"allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3"},
        )
        targets = np.stack(
            [
                gt,
                missing,
                excess,
                band,
                fg_band_target,
                bg_band_target,
                core_target,
                sep_target,
                hover_x,
                hover_y,
                keep_bone_target,
                cut_gap_target,
                structure_band_target,
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "image": torch.from_numpy(channels),
            "target": torch.from_numpy(targets),
            "selector_target": torch.from_numpy(selector_target),
            "name": image_path.name,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 3,
        base_channels: int = 32,
        embedding_dim: int = 32,
        target_mode: str = "residual",
        gate_disagreement_threshold: float = 0.10,
        gate_boundary_weight: float = 0.75,
        gate_uncertainty_weight: float = 0.35,
        gate_smooth_kernel: int = 5,
        boundary_band_weight: float = 1.0,
        boundary_fg_weight: float = 0.60,
        boundary_bg_weight: float = 0.50,
    ) -> None:
        super().__init__()
        self.target_mode = target_mode
        self.use_keepbone_cutgap_stage2 = target_mode in {"allstack_anatomy_roi_keepbone_cutgap_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3"}
        self.use_keepbone_cutgap_stage2_v2 = target_mode in {"allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3"}
        self.use_keepbone_cutgap_stage2_recall_rescue = target_mode == "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue"
        self.use_keepbone_cutgap_stage2_v3 = target_mode == "allstack_anatomy_roi_keepbone_cutgap_refiner_v3"
        self.gate_disagreement_threshold = gate_disagreement_threshold
        self.gate_boundary_weight = gate_boundary_weight
        self.gate_uncertainty_weight = gate_uncertainty_weight
        self.gate_smooth_kernel = max(1, int(gate_smooth_kernel))
        self.boundary_band_weight = boundary_band_weight
        self.boundary_fg_weight = boundary_fg_weight
        self.boundary_bg_weight = boundary_bg_weight
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.anchor_branch = ConvBlock(c, c)
        self.under_branch = ConvBlock(c, c)
        self.over_branch = ConvBlock(c, c)
        self.guided_fg_branch = ConvBlock(c, c)
        self.guided_bg_branch = ConvBlock(c, c)
        self.trim_risk_branch = ConvBlock(c, c)
        self.trim_delta_branch = ConvBlock(c, c)
        self.guided_fuse = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.anchor_head = nn.Conv2d(c, 1, kernel_size=1)
        self.under_head = nn.Conv2d(c, 1, kernel_size=1)
        self.over_head = nn.Conv2d(c, 1, kernel_size=1)
        self.guided_head = nn.Conv2d(c, 1, kernel_size=1)
        self.trim_risk_head = nn.Conv2d(c, 1, kernel_size=1)
        self.trim_delta_head = nn.Conv2d(c, 1, kernel_size=1)
        self.boundary_fg_head = nn.Conv2d(c, 1, kernel_size=1)
        self.boundary_bg_head = nn.Conv2d(c, 1, kernel_size=1)
        self.selector_branch = ConvBlock(c, c)
        self.selector_head = nn.Conv2d(c, 3, kernel_size=1)
        self.instance_core_head = nn.Conv2d(c, 1, kernel_size=1)
        self.instance_sep_head = nn.Conv2d(c, 1, kernel_size=1)
        self.hover_x_head = nn.Conv2d(c, 1, kernel_size=1)
        self.hover_y_head = nn.Conv2d(c, 1, kernel_size=1)
        self.interaction_cue_encoder = nn.Sequential(
            nn.Conv2d(8, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.interaction_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.interaction_gate_head = nn.Conv2d(c, 1, kernel_size=1)
        self.interaction_delta_head = nn.Conv2d(c, 1, kernel_size=1)
        self.stage2_cue_encoder = nn.Sequential(
            nn.Conv2d(10, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage2_fuse = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.stage2_gate_head = nn.Conv2d(c, 1, kernel_size=1)
        self.stage2_delta_head = nn.Conv2d(c, 1, kernel_size=1)
        self.stage2_band_head = nn.Conv2d(c, 1, kernel_size=1)
        self.stage2_fg_head = nn.Conv2d(c, 1, kernel_size=1)
        self.stage2_bg_head = nn.Conv2d(c, 1, kernel_size=1)
        if self.use_keepbone_cutgap_stage2:
            self.stage2_keep_head = nn.Conv2d(c, 1, kernel_size=1)
            self.stage2_gap_head = nn.Conv2d(c, 1, kernel_size=1)
        self.embed_pool = nn.AdaptiveAvgPool2d(1)
        self.anchor_proj = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, embedding_dim))
        self.under_proj = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, embedding_dim))
        self.over_proj = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, embedding_dim))

    def build_gate(self, x: torch.Tensor, residual_prob: torch.Tensor) -> torch.Tensor:
        disagreement = x[:, 5:6].clamp(0.0, 1.0)
        disagreement_boundary = x[:, 10:11].clamp(0.0, 1.0)
        gate = (disagreement > self.gate_disagreement_threshold).float()
        gate = torch.maximum(gate, disagreement)
        gate = torch.maximum(gate, disagreement_boundary * self.gate_boundary_weight)
        if x.shape[1] > 14:
            cross_candidate_boundary = x[:, 14:15].clamp(0.0, 1.0)
            gate = torch.maximum(gate, cross_candidate_boundary * self.gate_boundary_weight)
        uncertainty = 1.0 - torch.abs(2.0 * residual_prob - 1.0)
        gate = torch.maximum(gate, uncertainty * self.gate_uncertainty_weight)
        if self.gate_smooth_kernel > 1:
            gate = F.avg_pool2d(
                gate,
                kernel_size=self.gate_smooth_kernel,
                stride=1,
                padding=self.gate_smooth_kernel // 2,
            )
        return gate.clamp(0.0, 1.0)

    def branch_embedding(self, feature: torch.Tensor, projector: nn.Module) -> torch.Tensor:
        pooled = self.embed_pool(feature).flatten(1)
        return F.normalize(projector(pooled), dim=1)

    def soft_boundary_focus(self, prob: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        k = max(3, int(kernel_size) | 1)
        dilated = F.max_pool2d(prob, kernel_size=k, stride=1, padding=k // 2)
        eroded = 1.0 - F.max_pool2d(1.0 - prob, kernel_size=k, stride=1, padding=k // 2)
        return (dilated - eroded).clamp(0.0, 1.0)

    def interaction_refine_residual(
        self,
        d1: torch.Tensor,
        anchor_logits: torch.Tensor,
        under_prob: torch.Tensor,
        over_prob: torch.Tensor,
        boundary_fg_logits: torch.Tensor,
        boundary_bg_logits: torch.Tensor,
        disagreement: torch.Tensor,
        anatomy_gradient: torch.Tensor,
        anatomy_core: torch.Tensor,
        anatomy_band: torch.Tensor,
        anatomy_axis: torch.Tensor,
    ) -> torch.Tensor:
        residual_prob = torch.sigmoid(anchor_logits)
        boundary_fg_prob = torch.sigmoid(boundary_fg_logits)
        boundary_bg_prob = torch.sigmoid(boundary_bg_logits)
        uncertainty = (1.0 - torch.abs(2.0 * residual_prob - 1.0)).clamp(0.0, 1.0)
        structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
        cue_maps = torch.cat(
            [
                disagreement,
                anatomy_gradient,
                anatomy_band,
                structure_prior,
                uncertainty,
                boundary_fg_prob,
                boundary_bg_prob,
                torch.abs(under_prob - over_prob),
            ],
            dim=1,
        )
        cue_feature = self.interaction_cue_encoder(cue_maps)
        fused_feature = self.interaction_fuse(torch.cat([d1, cue_feature], dim=1))
        risk_map = (
            0.35 * disagreement
            + 0.20 * anatomy_band
            + 0.15 * anatomy_gradient
            + 0.15 * uncertainty
            + 0.15 * torch.abs(boundary_fg_prob - boundary_bg_prob)
        ).clamp(0.0, 1.0)
        interaction_gate = torch.sigmoid(self.interaction_gate_head(fused_feature)) * risk_map
        interaction_delta = self.interaction_delta_head(fused_feature)
        refined_logits = anchor_logits + interaction_gate * interaction_delta
        return torch.sigmoid(refined_logits)

    def instance_sep_refine_residual(
        self,
        d1: torch.Tensor,
        anchor_logits: torch.Tensor,
        under_prob: torch.Tensor,
        over_prob: torch.Tensor,
        boundary_fg_logits: torch.Tensor,
        boundary_bg_logits: torch.Tensor,
        anatomy_gradient: torch.Tensor,
        anatomy_core: torch.Tensor,
        anatomy_band: torch.Tensor,
        anatomy_axis: torch.Tensor,
        disagreement: torch.Tensor,
        instance_core_logits: torch.Tensor,
        instance_sep_logits: torch.Tensor,
        hover_x_logits: torch.Tensor,
        hover_y_logits: torch.Tensor,
    ) -> torch.Tensor:
        residual_prob = torch.sigmoid(anchor_logits)
        boundary_fg_prob = torch.sigmoid(boundary_fg_logits)
        boundary_bg_prob = torch.sigmoid(boundary_bg_logits)
        instance_core_prob = torch.sigmoid(instance_core_logits)
        instance_sep_prob = torch.sigmoid(instance_sep_logits)
        hover_x = torch.tanh(hover_x_logits)
        hover_y = torch.tanh(hover_y_logits)
        hover_mag = torch.sqrt(torch.clamp(hover_x * hover_x + hover_y * hover_y, min=0.0, max=4.0)) / np.sqrt(2.0)
        structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
        uncertainty = (1.0 - torch.abs(2.0 * residual_prob - 1.0)).clamp(0.0, 1.0)
        cue_maps = torch.cat(
            [
                disagreement,
                anatomy_gradient,
                anatomy_band,
                structure_prior,
                instance_core_prob,
                instance_sep_prob,
                hover_mag.clamp(0.0, 1.0),
                torch.abs(under_prob - over_prob),
            ],
            dim=1,
        )
        cue_feature = self.interaction_cue_encoder(cue_maps)
        fused_feature = self.interaction_fuse(torch.cat([d1, cue_feature], dim=1))
        sep_risk = (
            0.30 * disagreement
            + 0.20 * anatomy_band
            + 0.20 * instance_sep_prob
            + 0.10 * anatomy_gradient
            + 0.10 * hover_mag.clamp(0.0, 1.0)
            + 0.10 * uncertainty
        ).clamp(0.0, 1.0)
        interaction_gate = torch.sigmoid(self.interaction_gate_head(fused_feature)) * sep_risk
        interaction_delta = self.interaction_delta_head(fused_feature)
        refined_logits = anchor_logits + interaction_gate * interaction_delta
        refined_prob = torch.sigmoid(refined_logits)
        protect_core = torch.clamp(instance_core_prob * structure_prior, 0.0, 1.0)
        split_trim = torch.clamp(instance_sep_prob * (1.0 - structure_prior) * residual_prob, 0.0, 1.0)
        refined_prob = torch.maximum(refined_prob, 0.92 * protect_core)
        refined_prob = (refined_prob * (1.0 - 0.25 * split_trim)).clamp(0.0, 1.0)
        return refined_prob

    def two_stage_boundary_correct(
        self,
        x: torch.Tensor,
        e1: torch.Tensor,
        d1: torch.Tensor,
        coarse_prob: torch.Tensor,
        boundary_fg_logits: torch.Tensor,
        boundary_bg_logits: torch.Tensor,
        anatomy_gradient: torch.Tensor,
        anatomy_core: torch.Tensor,
        anatomy_band: torch.Tensor,
        anatomy_axis: torch.Tensor,
        disagreement: torch.Tensor,
        instance_core_logits: torch.Tensor,
        instance_sep_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boundary_fg_prob = torch.sigmoid(boundary_fg_logits)
        boundary_bg_prob = torch.sigmoid(boundary_bg_logits)
        instance_core_prob = torch.sigmoid(instance_core_logits)
        instance_sep_prob = torch.sigmoid(instance_sep_logits)
        uncertainty = (1.0 - torch.abs(2.0 * coarse_prob - 1.0)).clamp(0.0, 1.0)
        structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
        coarse_band = self.soft_boundary_focus(coarse_prob, kernel_size=5)
        narrow_band = torch.maximum(coarse_band, 0.85 * anatomy_band * torch.maximum(disagreement, uncertainty))
        narrow_band = (narrow_band * (0.55 + 0.45 * (1.0 - anatomy_core))).clamp(0.0, 1.0)
        cue_maps = torch.cat(
            [
                x[:, 0:1].clamp(0.0, 1.0),
                coarse_prob,
                disagreement,
                anatomy_gradient,
                anatomy_band,
                structure_prior,
                uncertainty,
                boundary_fg_prob,
                boundary_bg_prob,
                instance_sep_prob,
            ],
            dim=1,
        )
        cue_feature = self.stage2_cue_encoder(cue_maps)
        stage2_feature = self.stage2_fuse(torch.cat([e1, d1, cue_feature], dim=1))
        stage2_band_logits = self.stage2_band_head(stage2_feature)
        stage2_fg_logits = self.stage2_fg_head(stage2_feature)
        stage2_bg_logits = self.stage2_bg_head(stage2_feature)
        stage2_band_prob = torch.sigmoid(stage2_band_logits)
        stage2_fg_prob = torch.sigmoid(stage2_fg_logits)
        stage2_bg_prob = torch.sigmoid(stage2_bg_logits)
        active_band = (narrow_band * (0.55 + 0.45 * stage2_band_prob)).clamp(0.0, 1.0)
        inner_focus = (active_band * (1.0 - coarse_prob) * torch.maximum(structure_prior, 0.75 * boundary_fg_prob)).clamp(0.0, 1.0)
        outer_focus = (
            active_band
            * coarse_prob
            * torch.maximum(1.0 - structure_prior, 0.65 * instance_sep_prob + 0.35 * boundary_bg_prob)
        ).clamp(0.0, 1.0)
        stage2_gate = torch.sigmoid(self.stage2_gate_head(stage2_feature)) * active_band
        stage2_delta = self.stage2_delta_head(stage2_feature)
        corrected_logits = torch.logit(coarse_prob.clamp(1e-4, 1.0 - 1e-4)) + stage2_gate * stage2_delta
        corrected_prob = torch.sigmoid(corrected_logits)
        corrected_prob = corrected_prob + 0.22 * stage2_fg_prob * inner_focus * (1.0 - corrected_prob)
        corrected_prob = corrected_prob - 0.30 * stage2_bg_prob * outer_focus * corrected_prob
        core_floor = (0.92 * instance_core_prob * structure_prior).clamp(0.0, 1.0)
        seam_trim = (0.20 * instance_sep_prob * active_band * corrected_prob * (1.0 - 0.35 * structure_prior)).clamp(0.0, 0.25)
        corrected_prob = torch.maximum(corrected_prob, core_floor)
        corrected_prob = (corrected_prob * (1.0 - seam_trim)).clamp(0.0, 1.0)
        return corrected_prob, {
            "band": stage2_band_logits,
            "fg": stage2_fg_logits,
            "bg": stage2_bg_logits,
        }

    def two_stage_keepbone_cutgap_correct(
        self,
        x: torch.Tensor,
        e1: torch.Tensor,
        d1: torch.Tensor,
        coarse_prob: torch.Tensor,
        boundary_fg_logits: torch.Tensor,
        boundary_bg_logits: torch.Tensor,
        anatomy_gradient: torch.Tensor,
        anatomy_core: torch.Tensor,
        anatomy_band: torch.Tensor,
        anatomy_axis: torch.Tensor,
        disagreement: torch.Tensor,
        instance_core_logits: torch.Tensor,
        instance_sep_logits: torch.Tensor,
        prediction_aware: bool = False,
        recall_rescue: bool = False,
        precision_rescue: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        boundary_fg_prob = torch.sigmoid(boundary_fg_logits)
        boundary_bg_prob = torch.sigmoid(boundary_bg_logits)
        instance_core_prob = torch.sigmoid(instance_core_logits)
        instance_sep_prob = torch.sigmoid(instance_sep_logits)
        uncertainty = (1.0 - torch.abs(2.0 * coarse_prob - 1.0)).clamp(0.0, 1.0)
        structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
        coarse_band = self.soft_boundary_focus(coarse_prob, kernel_size=5)
        candidate_union = x[:, 3:4].clamp(0.0, 1.0)
        candidate_intersection = x[:, 4:5].clamp(0.0, 1.0)
        candidate_overlap_gap = (candidate_union - candidate_intersection).clamp(0.0, 1.0)
        disagreement_boundary = x[:, 10:11].clamp(0.0, 1.0)

        keep_narrow_band = (
            torch.maximum(coarse_band, anatomy_band)
            * torch.maximum(structure_prior, 0.85 * instance_core_prob)
            * (1.0 - 0.55 * anatomy_core)
        ).clamp(0.0, 1.0)
        gap_narrow_band = (
            torch.maximum(anatomy_band, disagreement)
            * torch.maximum(uncertainty, 0.80 * instance_sep_prob)
            * coarse_prob
            * torch.maximum(1.0 - structure_prior, 0.55 * instance_sep_prob)
        ).clamp(0.0, 1.0)
        if prediction_aware:
            keep_missing_focus = ((1.0 - candidate_union) * torch.maximum(structure_prior, instance_core_prob)).clamp(0.0, 1.0)
            keep_narrow_band = torch.maximum(keep_narrow_band, keep_missing_focus * torch.maximum(anatomy_band, coarse_band)).clamp(0.0, 1.0)
            gap_bridge_focus = (
                torch.maximum(candidate_overlap_gap, disagreement_boundary)
                * torch.maximum(instance_sep_prob, 1.0 - structure_prior)
                * torch.maximum(anatomy_band, disagreement)
            ).clamp(0.0, 1.0)
            gap_narrow_band = torch.maximum(gap_narrow_band, gap_bridge_focus).clamp(0.0, 1.0)
        structure_band = torch.maximum(keep_narrow_band, gap_narrow_band).clamp(0.0, 1.0)

        cue_maps = torch.cat(
            [
                x[:, 0:1].clamp(0.0, 1.0),
                coarse_prob,
                disagreement,
                anatomy_gradient,
                anatomy_band,
                structure_prior,
                uncertainty,
                boundary_fg_prob,
                boundary_bg_prob,
                instance_sep_prob,
            ],
            dim=1,
        )
        cue_feature = self.stage2_cue_encoder(cue_maps)
        stage2_feature = self.stage2_fuse(torch.cat([e1, d1, cue_feature], dim=1))
        stage2_band_logits = self.stage2_band_head(stage2_feature)
        stage2_keep_logits = self.stage2_keep_head(stage2_feature)
        stage2_gap_logits = self.stage2_gap_head(stage2_feature)
        stage2_band_prob = torch.sigmoid(stage2_band_logits)
        stage2_keep_prob = torch.sigmoid(stage2_keep_logits)
        stage2_gap_prob = torch.sigmoid(stage2_gap_logits)

        active_band = (structure_band * (0.55 + 0.45 * stage2_band_prob)).clamp(0.0, 1.0)
        keep_focus = (
            active_band
            * (1.0 - coarse_prob)
            * torch.maximum(structure_prior, instance_core_prob)
            * (0.60 + 0.40 * keep_narrow_band)
        ).clamp(0.0, 1.0)
        gap_focus = (
            active_band
            * coarse_prob
            * torch.maximum(1.0 - structure_prior, instance_sep_prob)
            * (0.60 + 0.40 * gap_narrow_band)
        ).clamp(0.0, 1.0)
        if prediction_aware:
            keep_focus = torch.maximum(
                keep_focus,
                (active_band * keep_narrow_band * (1.0 - candidate_union) * torch.maximum(structure_prior, instance_core_prob)).clamp(0.0, 1.0),
            ).clamp(0.0, 1.0)
            gap_focus = torch.maximum(
                gap_focus,
                (active_band * gap_narrow_band * torch.maximum(candidate_overlap_gap, coarse_prob * disagreement) * torch.maximum(instance_sep_prob, 1.0 - structure_prior)).clamp(0.0, 1.0),
            ).clamp(0.0, 1.0)
        if precision_rescue:
            low_gap_risk = (
                1.0
                - torch.maximum(
                    gap_narrow_band,
                    torch.maximum(instance_sep_prob, torch.maximum(candidate_overlap_gap, disagreement_boundary)),
                )
            ).clamp(0.0, 1.0)
            aic_keep_prior = (
                torch.maximum(anatomy_gradient, anatomy_band)
                * torch.maximum(structure_prior, instance_core_prob)
                * (1.0 - candidate_union)
                * low_gap_risk
            ).clamp(0.0, 1.0)
            keep_focus = torch.maximum(keep_focus, (active_band * aic_keep_prior).clamp(0.0, 1.0))
        if recall_rescue:
            low_gap_risk = (
                1.0
                - torch.maximum(
                    gap_narrow_band,
                    torch.maximum(instance_sep_prob, torch.maximum(candidate_overlap_gap, disagreement_boundary)),
                )
            ).clamp(0.0, 1.0)
            candidate_recall_support = torch.maximum(candidate_union, boundary_fg_prob).clamp(0.0, 1.0)
            thin_structure_support = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
            rescue_focus = (
                active_band
                * (1.0 - coarse_prob)
                * candidate_recall_support
                * torch.maximum(structure_prior, instance_core_prob)
                * thin_structure_support
                * low_gap_risk
            ).clamp(0.0, 1.0)
            keep_focus = torch.maximum(keep_focus, rescue_focus).clamp(0.0, 1.0)

        stage2_gate = torch.sigmoid(self.stage2_gate_head(stage2_feature)) * active_band
        stage2_delta = self.stage2_delta_head(stage2_feature)
        corrected_logits = torch.logit(coarse_prob.clamp(1e-4, 1.0 - 1e-4)) + stage2_gate * stage2_delta
        corrected_prob = torch.sigmoid(corrected_logits)
        corrected_prob = corrected_prob + 0.32 * stage2_keep_prob * keep_focus * (1.0 - corrected_prob)
        corrected_prob = corrected_prob - 0.36 * stage2_gap_prob * gap_focus * corrected_prob
        if recall_rescue:
            low_gap_risk = (
                1.0
                - torch.maximum(
                    gap_narrow_band,
                    torch.maximum(instance_sep_prob, torch.maximum(candidate_overlap_gap, disagreement_boundary)),
                )
            ).clamp(0.0, 1.0)
            rescue_gate = (
                keep_focus
                * stage2_keep_prob
                * boundary_fg_prob
                * low_gap_risk
                * (1.0 - 0.55 * boundary_bg_prob)
            ).clamp(0.0, 1.0)
            corrected_prob = corrected_prob + 0.12 * rescue_gate * (1.0 - corrected_prob)

        core_floor = (0.94 * instance_core_prob * torch.maximum(structure_prior, keep_narrow_band)).clamp(0.0, 1.0)
        gap_trim = (0.18 * stage2_gap_prob * gap_focus * (0.70 + 0.30 * instance_sep_prob)).clamp(0.0, 0.28)
        corrected_prob = torch.maximum(corrected_prob, core_floor)
        corrected_prob = (corrected_prob * (1.0 - gap_trim)).clamp(0.0, 1.0)
        if precision_rescue:
            precision_veto = (
                torch.maximum(candidate_overlap_gap, disagreement_boundary)
                * torch.maximum(instance_sep_prob, gap_narrow_band)
                * boundary_bg_prob
                * (1.0 - structure_prior)
                * coarse_prob
                * (1.0 - candidate_union)
            ).clamp(0.0, 1.0)
            extra_trim = (0.14 * stage2_gap_prob * active_band * precision_veto).clamp(0.0, 0.20)
            corrected_prob = (corrected_prob * (1.0 - extra_trim)).clamp(0.0, 1.0)
        return corrected_prob, {
            "band": stage2_band_logits,
            "keep": stage2_keep_logits,
            "gap": stage2_gap_logits,
        }

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        anchor_feature = self.anchor_branch(d1)
        under_feature = self.under_branch(d1)
        over_feature = self.over_branch(d1)
        anchor_logits = self.anchor_head(anchor_feature)
        under_logits = self.under_head(under_feature)
        over_logits = self.over_head(over_feature)
        boundary_fg_logits = self.boundary_fg_head(under_feature)
        boundary_bg_logits = self.boundary_bg_head(over_feature)
        instance_core_logits = self.instance_core_head(anchor_feature)
        instance_sep_logits = self.instance_sep_head(d1)
        hover_x_logits = self.hover_x_head(d1)
        hover_y_logits = self.hover_y_head(d1)
        selector_logits = self.selector_head(self.selector_branch(d1))
        trim_risk_logits = self.trim_risk_head(self.trim_risk_branch(d1))
        trim_delta_logits = self.trim_delta_head(self.trim_delta_branch(d1))
        stage2_aux_logits = None
        if self.target_mode in {"mec", "residual_mec_gated", "boundary_band_fgbg", "araa_guided_precision", "allstack_boundary_trim", "allstack_boundary_trim_guarded", "allstack_anatomy_roi_only", "allstack_anatomy_roi_prior_only", "allstack_anatomy_roi_aic_prior_only", "allstack_anatomy_roi_aic_refiner", "allstack_anatomy_roi_refiner", "allstack_anatomy_roi_interaction_refiner", "allstack_anatomy_roi_instance_sep_refiner", "allstack_anatomy_roi_stage2_refiner", "allstack_anatomy_roi_twostage_boundary_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3", "allstack_anatomy_roi_uncertainty_refiner", "allstack_anatomy_roi_refiner_uncertainty_gate", "allstack_anatomy_roi_selector_refiner_v3", "allstack_anatomy_roi_boundary_prefgate_trimfirst"}:
            anchor_mask = x[:, 2:3].clamp(0.0, 1.0)
            under_prob = torch.sigmoid(under_logits)
            over_prob = torch.sigmoid(over_logits)
            mec_prob = anchor_mask * (1.0 - over_prob) + (1.0 - anchor_mask) * under_prob
            if self.target_mode == "mec":
                final_prob = mec_prob
            elif self.target_mode == "residual_mec_gated":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                final_prob = residual_prob * (1.0 - gate) + mec_prob * gate
            elif self.target_mode == "boundary_band_fgbg":
                residual_prob = torch.sigmoid(anchor_logits)
                candidate_band = torch.maximum(x[:, 8:9].clamp(0.0, 1.0), x[:, 9:10].clamp(0.0, 1.0))
                candidate_band = torch.maximum(candidate_band, x[:, 10:11].clamp(0.0, 1.0))
                if x.shape[1] > 13:
                    candidate_band = torch.maximum(candidate_band, x[:, 13:14].clamp(0.0, 1.0))
                fg_prob = torch.sigmoid(boundary_fg_logits)
                bg_prob = torch.sigmoid(boundary_bg_logits)
                band_refine = residual_prob + self.boundary_fg_weight * fg_prob * candidate_band - self.boundary_bg_weight * bg_prob * candidate_band
                final_prob = residual_prob * (1.0 - candidate_band * self.boundary_band_weight) + band_refine * (candidate_band * self.boundary_band_weight)
            elif self.target_mode == "araa_guided_precision":
                candidate_union = x[:, 3:4].clamp(0.0, 1.0)
                candidate_intersection = x[:, 4:5].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                candidate_band = torch.maximum(x[:, 8:9].clamp(0.0, 1.0), x[:, 9:10].clamp(0.0, 1.0))
                candidate_band = torch.maximum(candidate_band, x[:, 10:11].clamp(0.0, 1.0))
                if x.shape[1] > 13:
                    candidate_band = torch.maximum(candidate_band, x[:, 13:14].clamp(0.0, 1.0))
                guide_mask = candidate_union
                fg_feature = self.guided_fg_branch(d1 * guide_mask)
                bg_feature = self.guided_bg_branch(d1 * (1.0 - guide_mask))
                guided_feature = self.guided_fuse(torch.cat([d1, fg_feature, bg_feature], dim=1))
                guided_prob = torch.sigmoid(self.guided_head(guided_feature))
                disputed_region = (candidate_union - candidate_intersection).clamp(0.0, 1.0)
                outer_band = (candidate_band * (1.0 - candidate_union)).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, candidate_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                trimmed_union = candidate_intersection + disputed_region * guided_prob
                expansion = self.boundary_fg_weight * under_prob * outer_band
                suppression = self.boundary_bg_weight * over_prob * precision_gate * disputed_region
                final_prob = trimmed_union + expansion - suppression
            elif self.target_mode == "allstack_boundary_trim":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                candidate_band = torch.maximum(x[:, 8:9].clamp(0.0, 1.0), x[:, 9:10].clamp(0.0, 1.0))
                candidate_band = torch.maximum(candidate_band, x[:, 10:11].clamp(0.0, 1.0))
                if x.shape[1] > 13:
                    candidate_band = torch.maximum(candidate_band, x[:, 13:14].clamp(0.0, 1.0))
                trim_risk = torch.sigmoid(trim_risk_logits)
                trim_delta = torch.sigmoid(trim_delta_logits)
                precision_gate = torch.maximum(disagreement, candidate_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                trim_gate = (precision_gate * trim_risk).clamp(0.0, 1.0)
                final_prob = allstack_prob * (1.0 - self.boundary_bg_weight * trim_delta * trim_gate)
            elif self.target_mode == "allstack_boundary_trim_guarded":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                candidate_intersection = x[:, 4:5].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                candidate_band = torch.maximum(x[:, 8:9].clamp(0.0, 1.0), x[:, 9:10].clamp(0.0, 1.0))
                candidate_band = torch.maximum(candidate_band, x[:, 10:11].clamp(0.0, 1.0))
                if x.shape[1] > 13:
                    candidate_band = torch.maximum(candidate_band, x[:, 13:14].clamp(0.0, 1.0))
                trim_risk = torch.sigmoid(trim_risk_logits)
                trim_delta = torch.sigmoid(trim_delta_logits)
                precision_gate = torch.maximum(disagreement, candidate_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                low_confidence = ((0.70 - allstack_prob).clamp(min=0.0) / 0.70).clamp(0.0, 1.0)
                over_support = (over_prob - 0.5 * under_prob).clamp(0.0, 1.0)
                trim_gate = (precision_gate * trim_risk * low_confidence * over_support * (1.0 - 0.85 * candidate_intersection)).clamp(0.0, 1.0)
                trim_strength = (0.65 * self.boundary_bg_weight * trim_delta).clamp(0.0, 0.45)
                final_prob = allstack_prob * (1.0 - trim_strength * trim_gate)
            elif self.target_mode == "allstack_anatomy_roi_only":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                final_prob = residual_prob * (1.0 - gate) + mec_prob * gate
            elif self.target_mode == "allstack_anatomy_roi_prior_only":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                prior_gate = torch.clamp(0.45 * anatomy_core + 0.35 * anatomy_axis + 0.20 * anatomy_gradient, 0.0, 1.0)
                context_gate = torch.maximum(prior_gate, anatomy_band * 0.5).clamp(0.0, 1.0)
                final_prob = allstack_prob * (0.85 + 0.15 * context_gate)
                final_prob = final_prob.clamp(0.0, 1.0)
            elif self.target_mode == "allstack_anatomy_roi_aic_prior_only":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                final_prob = residual_prob * (1.0 - gate) + mec_prob * gate
            elif self.target_mode == "allstack_anatomy_roi_aic_refiner":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, anatomy_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                detail_gate = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                suppress = self.boundary_bg_weight * bg_support * precision_gate * (1.0 - structure_prior)
                recover = self.boundary_fg_weight * fg_support * detail_gate * structure_prior * (1.0 - allstack_prob)
                final_prob = allstack_prob * (1.0 - suppress) + recover
            elif self.target_mode == "allstack_anatomy_roi_refiner":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, anatomy_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                detail_gate = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                suppress = self.boundary_bg_weight * bg_support * precision_gate * (1.0 - structure_prior)
                recover = self.boundary_fg_weight * fg_support * detail_gate * structure_prior * (1.0 - allstack_prob)
                final_prob = allstack_prob * (1.0 - suppress) + recover
            elif self.target_mode == "allstack_anatomy_roi_interaction_refiner":
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                refined_residual_prob = self.interaction_refine_residual(
                    d1,
                    anchor_logits,
                    under_prob,
                    over_prob,
                    boundary_fg_logits,
                    boundary_bg_logits,
                    disagreement,
                    anatomy_gradient,
                    anatomy_core,
                    anatomy_band,
                    anatomy_axis,
                )
                gate = self.build_gate(x, refined_residual_prob)
                allstack_prob = refined_residual_prob * (1.0 - gate) + mec_prob * gate
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, anatomy_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                detail_gate = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                suppress = self.boundary_bg_weight * bg_support * precision_gate * (1.0 - structure_prior)
                recover = self.boundary_fg_weight * fg_support * detail_gate * structure_prior * (1.0 - allstack_prob)
                final_prob = allstack_prob * (1.0 - suppress) + recover
            elif self.target_mode == "allstack_anatomy_roi_instance_sep_refiner":
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                refined_residual_prob = self.instance_sep_refine_residual(
                    d1,
                    anchor_logits,
                    under_prob,
                    over_prob,
                    boundary_fg_logits,
                    boundary_bg_logits,
                    anatomy_gradient,
                    anatomy_core,
                    anatomy_band,
                    anatomy_axis,
                    disagreement,
                    instance_core_logits,
                    instance_sep_logits,
                    hover_x_logits,
                    hover_y_logits,
                )
                gate = self.build_gate(x, refined_residual_prob)
                allstack_prob = refined_residual_prob * (1.0 - gate) + mec_prob * gate
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, anatomy_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                detail_gate = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
                instance_core_prob = torch.sigmoid(instance_core_logits)
                instance_sep_prob = torch.sigmoid(instance_sep_logits)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                suppress = self.boundary_bg_weight * bg_support * precision_gate * (1.0 - structure_prior) * (0.6 + 0.4 * instance_sep_prob)
                recover = self.boundary_fg_weight * fg_support * detail_gate * torch.maximum(structure_prior, instance_core_prob) * (1.0 - allstack_prob)
                final_prob = torch.maximum(allstack_prob * (1.0 - suppress) + recover, 0.90 * instance_core_prob * structure_prior)
            elif self.target_mode == "allstack_anatomy_roi_stage2_refiner":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, anatomy_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                detail_gate = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                stage2_zone = (precision_gate * detail_gate * (1.0 - anatomy_core)).clamp(0.0, 1.0)
                keep_prob = (allstack_prob + self.boundary_fg_weight * fg_support * stage2_zone * (0.5 + 0.5 * structure_prior)).clamp(0.0, 1.0)
                suppress_prob = (self.boundary_bg_weight * bg_support * stage2_zone * (1.0 - 0.5 * structure_prior)).clamp(0.0, 1.0)
                final_prob = allstack_prob * (1.0 - suppress_prob) + (1.0 - allstack_prob) * keep_prob * 0.35
            elif self.target_mode == "allstack_anatomy_roi_twostage_boundary_refiner":
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                refined_residual_prob = self.instance_sep_refine_residual(
                    d1,
                    anchor_logits,
                    under_prob,
                    over_prob,
                    boundary_fg_logits,
                    boundary_bg_logits,
                    anatomy_gradient,
                    anatomy_core,
                    anatomy_band,
                    anatomy_axis,
                    disagreement,
                    instance_core_logits,
                    instance_sep_logits,
                    hover_x_logits,
                    hover_y_logits,
                )
                gate = self.build_gate(x, refined_residual_prob)
                coarse_prob = refined_residual_prob * (1.0 - gate) + mec_prob * gate
                final_prob, stage2_aux_logits = self.two_stage_boundary_correct(
                    x,
                    e1,
                    d1,
                    coarse_prob,
                    boundary_fg_logits,
                    boundary_bg_logits,
                    anatomy_gradient,
                    anatomy_core,
                    anatomy_band,
                    anatomy_axis,
                    disagreement,
                    instance_core_logits,
                    instance_sep_logits,
                )
            elif self.target_mode in {"allstack_anatomy_roi_keepbone_cutgap_refiner", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2", "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue", "allstack_anatomy_roi_keepbone_cutgap_refiner_v3"}:
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                refined_residual_prob = self.instance_sep_refine_residual(
                    d1,
                    anchor_logits,
                    under_prob,
                    over_prob,
                    boundary_fg_logits,
                    boundary_bg_logits,
                    anatomy_gradient,
                    anatomy_core,
                    anatomy_band,
                    anatomy_axis,
                    disagreement,
                    instance_core_logits,
                    instance_sep_logits,
                    hover_x_logits,
                    hover_y_logits,
                )
                gate = self.build_gate(x, refined_residual_prob)
                coarse_prob = refined_residual_prob * (1.0 - gate) + mec_prob * gate
                final_prob, stage2_aux_logits = self.two_stage_keepbone_cutgap_correct(
                    x,
                    e1,
                    d1,
                    coarse_prob,
                    boundary_fg_logits,
                    boundary_bg_logits,
                    anatomy_gradient,
                    anatomy_core,
                    anatomy_band,
                    anatomy_axis,
                    disagreement,
                    instance_core_logits,
                    instance_sep_logits,
                    prediction_aware=self.use_keepbone_cutgap_stage2_v2,
                    recall_rescue=self.use_keepbone_cutgap_stage2_recall_rescue,
                    precision_rescue=self.use_keepbone_cutgap_stage2_v3,
                )
            elif self.target_mode == "allstack_anatomy_roi_uncertainty_refiner":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                uncertainty = (1.0 - torch.abs(2.0 * allstack_prob - 1.0)).clamp(0.0, 1.0)
                uncertainty_gate = (uncertainty * torch.maximum(disagreement, anatomy_band) * torch.maximum(anatomy_gradient, 0.5 * anatomy_axis)).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                refine_gain = self.boundary_fg_weight * fg_support * uncertainty_gate * structure_prior * (1.0 - allstack_prob)
                refine_trim = self.boundary_bg_weight * bg_support * uncertainty_gate * (1.0 - structure_prior) * allstack_prob
                final_prob = (allstack_prob + refine_gain - refine_trim).clamp(0.0, 1.0)
            elif self.target_mode == "allstack_anatomy_roi_refiner_uncertainty_gate":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                precision_gate = torch.maximum(disagreement, anatomy_band * self.gate_boundary_weight).clamp(0.0, 1.0)
                detail_gate = torch.maximum(anatomy_gradient, anatomy_band).clamp(0.0, 1.0)
                uncertainty = (1.0 - torch.abs(2.0 * allstack_prob - 1.0)).clamp(0.0, 1.0)
                uncertainty_gate = (uncertainty * precision_gate * detail_gate).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                suppress = self.boundary_bg_weight * bg_support * uncertainty_gate * (1.0 - structure_prior)
                recover = self.boundary_fg_weight * fg_support * uncertainty_gate * structure_prior * (1.0 - allstack_prob)
                final_prob = (allstack_prob * (1.0 - suppress) + recover).clamp(0.0, 1.0)
            elif self.target_mode == "allstack_anatomy_roi_boundary_prefgate_trimfirst":
                residual_prob = torch.sigmoid(anchor_logits)
                gate = self.build_gate(x, residual_prob)
                allstack_prob = residual_prob * (1.0 - gate) + mec_prob * gate
                baseline_prob = x[:, 1:2].clamp(0.0, 1.0)
                sam_prob = x[:, 2:3].clamp(0.0, 1.0)
                candidate_intersection = x[:, 4:5].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                refined_2_prob = x[:, 12:13].clamp(0.0, 1.0)
                baseline_only = x[:, 17:18].clamp(0.0, 1.0)
                sam_only = x[:, 18:19].clamp(0.0, 1.0)
                hq_only = x[:, 19:20].clamp(0.0, 1.0)
                inner_band = x[:, 20:21].clamp(0.0, 1.0)
                outer_band = x[:, 21:22].clamp(0.0, 1.0)
                consensus = x[:, 22:23].clamp(0.0, 1.0)
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                uncertainty = (1.0 - torch.abs(2.0 * allstack_prob - 1.0)).clamp(0.0, 1.0)
                selector_weights = torch.softmax(selector_logits, dim=1)
                local_candidate_prob = (
                    selector_weights[:, 0:1] * baseline_prob
                    + selector_weights[:, 1:2] * sam_prob
                    + selector_weights[:, 2:3] * refined_2_prob
                )
                stable_core = torch.maximum(candidate_intersection, 0.85 * anatomy_core).clamp(0.0, 1.0)
                trust_map = (0.60 * candidate_intersection + 0.25 * anatomy_core + 0.15 * structure_prior).clamp(0.0, 1.0)
                boundary_gate = (
                    0.32 * disagreement
                    + 0.22 * anatomy_band
                    + 0.18 * anatomy_gradient
                    + 0.14 * uncertainty
                    + 0.14 * (1.0 - consensus)
                ).clamp(0.0, 1.0)
                band_focus = torch.maximum(inner_band, outer_band)
                active_boundary_zone = torch.maximum(boundary_gate, band_focus)
                active_boundary_zone = (active_boundary_zone * (1.0 - 0.65 * trust_map)).clamp(0.0, 1.0)
                boundary_pref_prob = allstack_prob * (1.0 - active_boundary_zone) + local_candidate_prob * active_boundary_zone
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                trim_evidence = (
                    0.34 * disagreement
                    + 0.24 * (1.0 - structure_prior)
                    + 0.18 * bg_support
                    + 0.12 * (1.0 - consensus)
                    + 0.12 * baseline_only
                ).clamp(0.0, 1.0)
                recover_evidence = (
                    0.30 * selector_weights[:, 2:3]
                    + 0.22 * selector_weights[:, 1:2]
                    + 0.18 * hq_only
                    + 0.12 * sam_only
                    + 0.10 * fg_support
                    + 0.08 * anatomy_gradient
                ).clamp(0.0, 1.0)
                trim_gate = (
                    active_boundary_zone
                    * outer_band
                    * trim_evidence
                    * (1.0 - 0.80 * stable_core)
                ).clamp(0.0, 1.0)
                recover_gate = (
                    active_boundary_zone
                    * inner_band
                    * recover_evidence
                    * structure_prior
                    * (1.0 - boundary_pref_prob)
                ).clamp(0.0, 1.0)
                trim_strength = (0.80 * self.boundary_bg_weight * bg_support * trim_gate).clamp(0.0, 0.70)
                recover_strength = (0.28 * self.boundary_fg_weight * fg_support * recover_gate).clamp(0.0, 0.35)
                trimmed_prob = boundary_pref_prob * (1.0 - trim_strength)
                recovered_prob = trimmed_prob + (1.0 - trimmed_prob) * recover_strength
                final_prob = torch.maximum(0.98 * stable_core, recovered_prob).clamp(0.0, 1.0)
                final_prob = final_prob.clamp(0.0, 1.0)
            else:
                baseline_prob = x[:, 1:2].clamp(0.0, 1.0)
                refined_prob = x[:, 2:3].clamp(0.0, 1.0)
                refined_2_prob = x[:, 12:13].clamp(0.0, 1.0)
                baseline_only = x[:, 17:18].clamp(0.0, 1.0)
                refined_only = x[:, 18:19].clamp(0.0, 1.0)
                refined_2_only = x[:, 19:20].clamp(0.0, 1.0)
                inner_band = x[:, 20:21].clamp(0.0, 1.0)
                outer_band = x[:, 21:22].clamp(0.0, 1.0)
                consensus = x[:, 22:23].clamp(0.0, 1.0)
                anatomy_gradient = x[:, -4:-3].clamp(0.0, 1.0)
                anatomy_core = x[:, -3:-2].clamp(0.0, 1.0)
                anatomy_band = x[:, -2:-1].clamp(0.0, 1.0)
                anatomy_axis = x[:, -1:].clamp(0.0, 1.0)
                disagreement = x[:, 5:6].clamp(0.0, 1.0)
                selector_weights = torch.softmax(selector_logits, dim=1)
                selected_prob = (
                    selector_weights[:, 0:1] * baseline_prob
                    + selector_weights[:, 1:2] * refined_prob
                    + selector_weights[:, 2:3] * refined_2_prob
                )
                exclusive_mix = torch.maximum(baseline_only, torch.maximum(refined_only, refined_2_only))
                structure_prior = torch.maximum(anatomy_core, 0.5 * anatomy_axis).clamp(0.0, 1.0)
                trust_map = (0.55 * consensus + 0.30 * anatomy_core + 0.15 * structure_prior).clamp(0.0, 1.0)
                uncertainty = (1.0 - torch.abs(2.0 * selected_prob - 1.0)).clamp(0.0, 1.0)
                boundary_gate = (
                    0.35 * disagreement
                    + 0.25 * anatomy_band
                    + 0.15 * anatomy_gradient
                    + 0.15 * (1.0 - consensus)
                    + 0.10 * uncertainty
                ).clamp(0.0, 1.0)
                boundary_gate = (boundary_gate * (1.0 - 0.65 * trust_map)).clamp(0.0, 1.0)
                fg_support = torch.sigmoid(boundary_fg_logits)
                bg_support = torch.sigmoid(boundary_bg_logits)
                add_allowed = (inner_band * (0.65 * structure_prior + 0.35 * exclusive_mix)).clamp(0.0, 1.0)
                trim_allowed = (outer_band * (0.65 * (1.0 - structure_prior) + 0.35 * disagreement)).clamp(0.0, 1.0)
                refine_gain = 0.30 * self.boundary_fg_weight * fg_support * boundary_gate * add_allowed * (1.0 - selected_prob)
                refine_trim = 0.65 * self.boundary_bg_weight * bg_support * boundary_gate * trim_allowed * selected_prob
                final_prob = (selected_prob + refine_gain - refine_trim).clamp(0.0, 1.0)
            final_prob = final_prob.clamp(1e-4, 1.0 - 1e-4)
            final_logits = torch.logit(final_prob)
        else:
            final_logits = anchor_logits
        logits = torch.cat(
            [final_logits, under_logits, over_logits, boundary_fg_logits, boundary_bg_logits, trim_risk_logits, trim_delta_logits],
            dim=1,
        )
        embeddings = {
            "anchor": self.branch_embedding(anchor_feature, self.anchor_proj),
            "under": self.branch_embedding(under_feature, self.under_proj),
            "over": self.branch_embedding(over_feature, self.over_proj),
        }
        return {
            "logits": logits,
            "embeddings": embeddings,
            "selector_logits": selector_logits,
            "stage2_aux_logits": stage2_aux_logits,
            "instance_logits": {
                "core": instance_core_logits,
                "sep": instance_sep_logits,
                "hover_x": hover_x_logits,
                "hover_y": hover_y_logits,
            },
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denominator = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(x.shape)}")
    p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeletonize(x: torch.Tensor, iters: int) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    skeleton = (x - soft_open(x)).clamp_min(0.0)
    for _ in range(max(0, int(iters))):
        x = soft_erode(x)
        delta = (x - soft_open(x)).clamp_min(0.0)
        skeleton = skeleton + (1.0 - skeleton) * delta
    return skeleton.clamp(0.0, 1.0)


def cldice_loss(logits: torch.Tensor, target: torch.Tensor, iters: int = 3, eps: float = 1e-6) -> torch.Tensor:
    pred_prob = torch.sigmoid(logits).clamp(0.0, 1.0)
    target_prob = target.clamp(0.0, 1.0)
    pred_skel = soft_skeletonize(pred_prob, iters)
    target_skel = soft_skeletonize(target_prob, iters)
    tprec = (pred_skel * target_prob).sum(dim=(1, 2, 3)) / (pred_skel.sum(dim=(1, 2, 3)) + eps)
    tsens = (target_skel * pred_prob).sum(dim=(1, 2, 3)) / (target_skel.sum(dim=(1, 2, 3)) + eps)
    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cldice).mean()


def criterion(
    logits: torch.Tensor,
    targets: torch.Tensor,
    embeddings: dict[str, torch.Tensor],
    instance_logits: dict[str, torch.Tensor] | None,
    stage2_aux_logits: dict[str, torch.Tensor] | None,
    selector_logits: torch.Tensor | None,
    selector_target: torch.Tensor | None,
    aux_weight: float,
    contrast_weight: float,
    selector_loss_weight: float,
    cldice_weight: float,
    cldice_iters: int,
    instance_sep_loss_weight: float,
    stage2_boundary_loss_weight: float,
    structure_keep_loss_weight: float,
    structure_gap_loss_weight: float,
) -> torch.Tensor:
    final_loss = F.binary_cross_entropy_with_logits(logits[:, 0:1], targets[:, 0:1]) + dice_loss(logits[:, 0:1], targets[:, 0:1])
    topology_loss = logits.new_tensor(0.0)
    if cldice_weight > 0.0:
        topology_loss = cldice_loss(logits[:, 0:1], targets[:, 0:1], iters=cldice_iters)
        final_loss = final_loss + cldice_weight * topology_loss
    missing_loss = F.binary_cross_entropy_with_logits(logits[:, 1:2], targets[:, 1:2]) + dice_loss(logits[:, 1:2], targets[:, 1:2])
    excess_loss = F.binary_cross_entropy_with_logits(logits[:, 2:3], targets[:, 2:3]) + dice_loss(logits[:, 2:3], targets[:, 2:3])
    boundary_fg_loss = F.binary_cross_entropy_with_logits(logits[:, 3:4], targets[:, 4:5]) + dice_loss(logits[:, 3:4], targets[:, 4:5])
    boundary_bg_loss = F.binary_cross_entropy_with_logits(logits[:, 4:5], targets[:, 5:6]) + dice_loss(logits[:, 4:5], targets[:, 5:6])
    trim_risk_target = ((targets[:, 5:6] > 0.5) | ((targets[:, 3:4] > 0.5) & (targets[:, 2:3] > 0.5))).float()
    trim_delta_target = targets[:, 5:6]
    trim_risk_loss = F.binary_cross_entropy_with_logits(logits[:, 5:6], trim_risk_target) + dice_loss(logits[:, 5:6], trim_risk_target)
    trim_delta_loss = F.binary_cross_entropy_with_logits(logits[:, 6:7], trim_delta_target) + dice_loss(logits[:, 6:7], trim_delta_target)
    contrast_loss = F.triplet_margin_loss(
        embeddings["anchor"],
        embeddings["under"],
        embeddings["over"],
        margin=0.20,
        p=2.0,
    )
    selector_loss = logits.new_tensor(0.0)
    if selector_logits is not None and selector_target is not None:
        selector_log_prob = F.log_softmax(selector_logits, dim=1)
        selector_loss = -(selector_target * selector_log_prob).sum(dim=1).mean()
    instance_sep_loss = logits.new_tensor(0.0)
    if instance_logits is not None and targets.shape[1] >= 10:
        core_target = targets[:, 6:7]
        sep_target = targets[:, 7:8]
        hover_x_target = targets[:, 8:9]
        hover_y_target = targets[:, 9:10]
        core_logits = instance_logits["core"]
        sep_logits = instance_logits["sep"]
        hover_x_logits = instance_logits["hover_x"]
        hover_y_logits = instance_logits["hover_y"]
        core_loss = F.binary_cross_entropy_with_logits(core_logits, core_target) + dice_loss(core_logits, core_target)
        sep_loss = F.binary_cross_entropy_with_logits(sep_logits, sep_target)
        hover_x_loss = F.smooth_l1_loss(torch.tanh(hover_x_logits), hover_x_target)
        hover_y_loss = F.smooth_l1_loss(torch.tanh(hover_y_logits), hover_y_target)
        instance_sep_loss = core_loss + 0.75 * sep_loss + 0.30 * (hover_x_loss + hover_y_loss)
    stage2_boundary_loss = logits.new_tensor(0.0)
    if stage2_aux_logits is not None:
        if "keep" in stage2_aux_logits and "gap" in stage2_aux_logits and targets.shape[1] >= 13:
            band_target = targets[:, 12:13]
            keep_target = targets[:, 10:11]
            gap_target = targets[:, 11:12]
            stage2_band_loss = F.binary_cross_entropy_with_logits(stage2_aux_logits["band"], band_target) + dice_loss(stage2_aux_logits["band"], band_target)
            stage2_keep_loss = F.binary_cross_entropy_with_logits(stage2_aux_logits["keep"], keep_target) + dice_loss(stage2_aux_logits["keep"], keep_target)
            stage2_gap_loss = F.binary_cross_entropy_with_logits(stage2_aux_logits["gap"], gap_target) + dice_loss(stage2_aux_logits["gap"], gap_target)
            stage2_boundary_loss = stage2_band_loss + structure_keep_loss_weight * stage2_keep_loss + structure_gap_loss_weight * stage2_gap_loss
        else:
            band_target = targets[:, 3:4]
            fg_target = targets[:, 4:5]
            bg_target = targets[:, 5:6]
            stage2_band_loss = F.binary_cross_entropy_with_logits(stage2_aux_logits["band"], band_target) + dice_loss(stage2_aux_logits["band"], band_target)
            stage2_fg_loss = F.binary_cross_entropy_with_logits(stage2_aux_logits["fg"], fg_target) + dice_loss(stage2_aux_logits["fg"], fg_target)
            stage2_bg_loss = F.binary_cross_entropy_with_logits(stage2_aux_logits["bg"], bg_target) + dice_loss(stage2_aux_logits["bg"], bg_target)
            stage2_boundary_loss = stage2_band_loss + 0.8 * stage2_fg_loss + 0.8 * stage2_bg_loss
    return (
        final_loss
        + aux_weight * (missing_loss + excess_loss + 0.5 * (boundary_fg_loss + boundary_bg_loss) + 0.75 * trim_risk_loss + trim_delta_loss)
        + contrast_weight * contrast_loss
        + selector_loss_weight * selector_loss
        + instance_sep_loss_weight * instance_sep_loss
        + stage2_boundary_loss_weight * stage2_boundary_loss
    )


def compute_batch_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = torch.sigmoid(logits[:, 0:1]) > 0.5
    gt = target[:, 0:1] > 0.5
    tp = (pred & gt).float().sum(dim=(1, 2, 3))
    fp = (pred & ~gt).float().sum(dim=(1, 2, 3))
    fn = (~pred & gt).float().sum(dim=(1, 2, 3))
    dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)

    pred_f = pred.float()
    gt_f = gt.float()
    pred_dilate = F.max_pool2d(pred_f, kernel_size=3, stride=1, padding=1)
    pred_erode = 1.0 - F.max_pool2d(1.0 - pred_f, kernel_size=3, stride=1, padding=1)
    gt_dilate = F.max_pool2d(gt_f, kernel_size=3, stride=1, padding=1)
    gt_erode = 1.0 - F.max_pool2d(1.0 - gt_f, kernel_size=3, stride=1, padding=1)
    pred_boundary = (pred_dilate - pred_erode) > 0.5
    gt_boundary = (gt_dilate - gt_erode) > 0.5
    boundary_intersection = (pred_boundary & gt_boundary).float().sum(dim=(1, 2, 3))
    boundary_union = (pred_boundary | gt_boundary).float().sum(dim=(1, 2, 3))
    boundary_iou = (boundary_intersection + 1e-6) / (boundary_union + 1e-6)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "boundary_iou": float(boundary_iou.mean().item()),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    aux_weight: float,
    contrast_weight: float,
    selector_loss_weight: float,
    cldice_weight: float,
    cldice_iters: int,
    instance_sep_loss_weight: float,
    stage2_boundary_loss_weight: float,
    structure_keep_loss_weight: float,
    structure_gap_loss_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "boundary_iou": 0.0}
    total_batches = 0
    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, desc="train" if is_train else "val"):
            images = batch["image"].to(device=device, dtype=torch.float32)
            targets = batch["target"].to(device=device, dtype=torch.float32)
            selector_target = batch["selector_target"].to(device=device, dtype=torch.float32)
            outputs = model(images)
            logits = outputs["logits"]
            embeddings = outputs["embeddings"]
            instance_logits = outputs.get("instance_logits")
            stage2_aux_logits = outputs.get("stage2_aux_logits")
            selector_logits = outputs.get("selector_logits")
            loss = criterion(
                logits,
                targets,
                embeddings,
                instance_logits,
                stage2_aux_logits,
                selector_logits,
                selector_target,
                aux_weight,
                contrast_weight,
                selector_loss_weight,
                cldice_weight,
                cldice_iters,
                instance_sep_loss_weight,
                stage2_boundary_loss_weight,
                structure_keep_loss_weight,
                structure_gap_loss_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            batch_metrics = compute_batch_metrics(logits, targets)
            for key, value in batch_metrics.items():
                total_metrics[key] += value
            total_batches += 1
    result = {"loss": total_loss / max(1, total_batches)}
    for key, value in total_metrics.items():
        result[key] = value / max(1, total_batches)
    return result


def selection_score(metrics: dict[str, float], args: argparse.Namespace) -> float:
    return (
        args.selection_dice_weight * float(metrics.get("dice", 0.0))
        + args.selection_iou_weight * float(metrics.get("iou", 0.0))
        + args.selection_precision_weight * float(metrics.get("precision", 0.0))
        + args.selection_boundary_weight * float(metrics.get("boundary_iou", 0.0))
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    train_dataset = ErrorRefinerDataset(
        Path(args.raw_root),
        Path(args.ablations_root),
        args.dataset,
        args.train_split,
        args.baseline_exp,
        args.refined_exp,
        args.refined_exp_2,
        args.img_size,
        args.target_mode,
        args.boundary_band_kernel,
    )
    val_dataset = ErrorRefinerDataset(
        Path(args.raw_root),
        Path(args.ablations_root),
        args.dataset,
        args.val_split,
        args.baseline_exp,
        args.refined_exp,
        args.refined_exp_2,
        args.img_size,
        args.target_mode,
        args.boundary_band_kernel,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    input_channels = int(train_dataset[0]["image"].shape[0])
    model = SmallUNet(
        in_channels=input_channels,
        base_channels=args.base_channels,
        target_mode=args.target_mode,
        gate_disagreement_threshold=args.gate_disagreement_threshold,
        gate_boundary_weight=args.gate_boundary_weight,
        gate_uncertainty_weight=args.gate_uncertainty_weight,
        gate_smooth_kernel=args.gate_smooth_kernel,
        boundary_band_weight=args.boundary_band_weight,
        boundary_fg_weight=args.boundary_fg_weight,
        boundary_bg_weight=args.boundary_bg_weight,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_score = -1.0
    best_dice = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            args.device,
            optimizer,
            args.aux_loss_weight,
            args.contrast_loss_weight,
            args.selector_loss_weight,
            args.cldice_loss_weight,
            args.cldice_iters,
            args.instance_sep_loss_weight,
            args.stage2_boundary_loss_weight,
            args.structure_keep_loss_weight,
            args.structure_gap_loss_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            args.device,
            None,
            args.aux_loss_weight,
            args.contrast_loss_weight,
            args.selector_loss_weight,
            args.cldice_loss_weight,
            args.cldice_iters,
            args.instance_sep_loss_weight,
            args.stage2_boundary_loss_weight,
            args.structure_keep_loss_weight,
            args.structure_gap_loss_weight,
        )
        val_metrics["selection_score"] = selection_score(val_metrics, args)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        if val_metrics["selection_score"] > best_score:
            best_score = val_metrics["selection_score"]
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_dice": best_dice,
                    "best_val_score": best_score,
                    "best_epoch": best_epoch,
                    "input_channels": input_channels,
                    "base_channels": args.base_channels,
                    "target_mode": args.target_mode,
                    "gate_disagreement_threshold": args.gate_disagreement_threshold,
                    "gate_boundary_weight": args.gate_boundary_weight,
                    "gate_uncertainty_weight": args.gate_uncertainty_weight,
                    "gate_smooth_kernel": args.gate_smooth_kernel,
                    "boundary_band_kernel": args.boundary_band_kernel,
                    "boundary_band_weight": args.boundary_band_weight,
                    "boundary_fg_weight": args.boundary_fg_weight,
                    "boundary_bg_weight": args.boundary_bg_weight,
                    "structure_keep_loss_weight": args.structure_keep_loss_weight,
                    "structure_gap_loss_weight": args.structure_gap_loss_weight,
                },
                output,
            )
            print(f"Saved best checkpoint: {output}")
        else:
            epochs_without_improvement += 1
        (output.parent / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch}: no validation selection-score improvement for "
                f"{epochs_without_improvement} epochs (best epoch {best_epoch}, best score {best_score:.6f}, best Dice {best_dice:.6f})."
            )
            break


if __name__ == "__main__":
    main()
