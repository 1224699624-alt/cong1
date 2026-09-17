#!/usr/bin/env python3
"""Run a trained residual mask refiner on candidate YOLO-SAM masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from train_error_refiner import (
    SmallUNet,
    build_feature_stack,
    find_image_files,
    prepare_roi_sample,
    read_binary_mask,
    read_image_gray,
    resize_float,
)

ANATOMY_TARGET_MODES = {
    "allstack_anatomy_roi_only",
    "allstack_anatomy_roi_prior_only",
    "allstack_anatomy_roi_aic_prior_only",
    "allstack_anatomy_roi_aic_refiner",
    "allstack_anatomy_roi_refiner",
    "allstack_anatomy_roi_interaction_refiner",
    "allstack_anatomy_roi_instance_sep_refiner",
    "allstack_anatomy_roi_stage2_refiner",
    "allstack_anatomy_roi_twostage_boundary_refiner",
    "allstack_anatomy_roi_keepbone_cutgap_refiner",
    "allstack_anatomy_roi_keepbone_cutgap_refiner_v2",
    "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue",
    "allstack_anatomy_roi_keepbone_cutgap_refiner_v3",
    "allstack_anatomy_roi_uncertainty_refiner",
    "allstack_anatomy_roi_refiner_uncertainty_gate",
    "allstack_anatomy_roi_selector_refiner_v3",
    "allstack_anatomy_roi_boundary_prefgate_trimfirst",
}

OPTIONAL_LEGACY_HEAD_PREFIXES = (
    "selector_branch.",
    "selector_head.",
    "instance_core_head.",
    "instance_sep_head.",
    "hover_x_head.",
    "hover_y_head.",
    "interaction_cue_encoder.",
    "interaction_fuse.",
    "interaction_gate_head.",
    "interaction_delta_head.",
    "stage2_cue_encoder.",
    "stage2_fuse.",
    "stage2_gate_head.",
    "stage2_delta_head.",
    "stage2_band_head.",
    "stage2_fg_head.",
    "stage2_bg_head.",
    "stage2_keep_head.",
    "stage2_gap_head.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer refined masks with an error-aware residual refiner.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--baseline-exp", default="zero_shot_box_only_conf020_pad005")
    parser.add_argument("--refined-exp", default="zero_shot_mask_to_prompt_refine_pad003_neg2_k5")
    parser.add_argument("--refined-exp-2", default=None, help="Optional second refined candidate mask directory for multi-candidate fusion.")
    parser.add_argument("--checkpoint", default="outputs/error_refiner/TSRS_RSNA-Epiphysis/best.pt")
    parser.add_argument("--output-name", default="error_refiner")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--post-median-ksize", type=int, default=3, help="Odd kernel size for binary median smoothing; <=1 disables.")
    parser.add_argument("--post-open-kernel", type=int, default=3, help="Odd kernel size for morphological opening to remove spikes; <=1 disables.")
    parser.add_argument("--post-min-component", type=int, default=48, help="Remove connected foreground components smaller than this area; <=0 disables.")
    parser.add_argument("--post-min-hole", type=int, default=24, help="Fill background holes smaller than this area; <=0 disables.")
    parser.add_argument(
        "--target-mode",
        default=None,
        help="Override checkpoint target_mode when older checkpoints do not store it.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 0
    return value | 1


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = 1
    return keep.astype(bool)


def fill_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return mask
    inv = (~mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    filled = mask.astype(np.uint8).copy()
    h, w = mask.shape
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        if not touches_border and area <= max_hole_area:
            filled[labels == label] = 1
    return filled.astype(bool)


def postprocess_mask(
    prob: np.ndarray,
    threshold: float,
    post_median_ksize: int,
    post_open_kernel: int,
    post_min_component: int,
    post_min_hole: int,
) -> np.ndarray:
    mask = (prob >= threshold).astype(np.uint8)

    median_ksize = _odd_kernel(post_median_ksize)
    if median_ksize > 1:
        mask = (cv2.medianBlur(mask * 255, median_ksize) > 127).astype(np.uint8)

    open_kernel = _odd_kernel(post_open_kernel)
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask_bool = mask.astype(bool)
    mask_bool = remove_small_components(mask_bool, post_min_component)
    mask_bool = fill_small_holes(mask_bool, post_min_hole)
    return mask_bool


def build_input(
    image: np.ndarray,
    baseline: np.ndarray,
    refined: np.ndarray,
    size: int,
    refined_2: np.ndarray | None = None,
    target_mode: str = "residual",
) -> np.ndarray:
    image_r, baseline_r, refined_r, refined_2_r, _, roi_meta = prepare_roi_sample(
        image,
        baseline,
        refined,
        refined_2,
        None,
        size,
        target_mode,
    )
    return build_feature_stack(
        image_r,
        baseline_r,
        refined_r,
        refined_2=refined_2_r,
        include_anatomy=target_mode in ANATOMY_TARGET_MODES,
    ), roi_meta


def load_model_bundle(
    checkpoint_path: str | Path,
    device: str,
    target_mode_override: str | None = None,
) -> tuple[SmallUNet, int, str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    img_size = int(checkpoint_args.get("img_size", 512))
    base_channels = int(checkpoint.get("base_channels", checkpoint_args.get("base_channels", 32)))
    input_channels = int(checkpoint.get("input_channels", 12))
    target_mode = checkpoint.get("target_mode", checkpoint_args.get("target_mode"))
    if target_mode_override is not None:
        target_mode = target_mode_override
    elif target_mode is None:
        if input_channels == 27:
            raise ValueError(
                "Checkpoint expects 27-channel anatomy features but does not store target_mode. "
                "Re-run with --target-mode set to the training mode, for example "
                "'allstack_anatomy_roi_refiner' or another anatomy ROI variant."
            )
        target_mode = "residual"
    target_mode = str(target_mode)
    gate_disagreement_threshold = float(
        checkpoint.get("gate_disagreement_threshold", checkpoint_args.get("gate_disagreement_threshold", 0.10))
    )
    gate_boundary_weight = float(
        checkpoint.get("gate_boundary_weight", checkpoint_args.get("gate_boundary_weight", 0.75))
    )
    gate_uncertainty_weight = float(
        checkpoint.get("gate_uncertainty_weight", checkpoint_args.get("gate_uncertainty_weight", 0.35))
    )
    gate_smooth_kernel = int(
        checkpoint.get("gate_smooth_kernel", checkpoint_args.get("gate_smooth_kernel", 5))
    )
    boundary_band_weight = float(
        checkpoint.get("boundary_band_weight", checkpoint_args.get("boundary_band_weight", 1.0))
    )
    boundary_fg_weight = float(
        checkpoint.get("boundary_fg_weight", checkpoint_args.get("boundary_fg_weight", 0.60))
    )
    boundary_bg_weight = float(
        checkpoint.get("boundary_bg_weight", checkpoint_args.get("boundary_bg_weight", 0.50))
    )
    model = SmallUNet(
        in_channels=input_channels,
        base_channels=base_channels,
        target_mode=target_mode,
        gate_disagreement_threshold=gate_disagreement_threshold,
        gate_boundary_weight=gate_boundary_weight,
        gate_uncertainty_weight=gate_uncertainty_weight,
        gate_smooth_kernel=gate_smooth_kernel,
        boundary_band_weight=boundary_band_weight,
        boundary_fg_weight=boundary_fg_weight,
        boundary_bg_weight=boundary_bg_weight,
    ).to(device)
    state_dict = checkpoint["model"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        optional_missing = [
            key
            for key in missing
            if key.startswith(OPTIONAL_LEGACY_HEAD_PREFIXES)
        ]
        if unexpected or len(optional_missing) != len(missing):
            raise
        print(
            "Loaded legacy checkpoint with newly added optional heads left initialized: "
            f"{len(optional_missing)} missing optional keys",
            flush=True,
        )
    model.eval()
    return model, img_size, target_mode


def main() -> None:
    args = parse_args()
    model, img_size, target_mode = load_model_bundle(args.checkpoint, args.device, args.target_mode)

    raw_root = Path(args.raw_root)
    ablations_root = Path(args.ablations_root)
    image_dir = raw_root / args.dataset / args.split
    baseline_dir = ablations_root / args.baseline_exp / args.dataset / args.split / "masks"
    refined_dir = ablations_root / args.refined_exp / args.dataset / args.split / "masks"
    refined_dir_2 = ablations_root / args.refined_exp_2 / args.dataset / args.split / "masks" if args.refined_exp_2 else None
    output_dir = ablations_root / args.output_name / args.dataset / args.split / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for image_path in tqdm(find_image_files(image_dir), desc=f"error-refine/{args.dataset}/{args.split}"):
        stem = image_path.stem
        image = read_image_gray(image_path)
        baseline = read_binary_mask(baseline_dir / f"{stem}.png")
        refined = read_binary_mask(refined_dir / f"{stem}.png")
        refined_2 = read_binary_mask(refined_dir_2 / f"{stem}.png") if refined_dir_2 is not None else None
        original_shape = image.shape
        input_array, roi_meta = build_input(
            image,
            baseline,
            refined,
            img_size,
            refined_2=refined_2,
            target_mode=target_mode,
        )
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(args.device)
        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs["logits"]
            prob = torch.sigmoid(logits[:, 0:1])[0, 0].detach().cpu().numpy()
        if roi_meta is None:
            prob = cv2.resize(prob.astype(np.float32), original_shape[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            roi_h = int(roi_meta["y1"] - roi_meta["y0"])
            roi_w = int(roi_meta["x1"] - roi_meta["x0"])
            roi_prob = cv2.resize(prob.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            full_prob = refined.astype(np.float32).copy()
            full_prob[roi_meta["y0"] : roi_meta["y1"], roi_meta["x0"] : roi_meta["x1"]] = roi_prob
            prob = full_prob
        mask = postprocess_mask(
            prob,
            args.threshold,
            args.post_median_ksize,
            args.post_open_kernel,
            args.post_min_component,
            args.post_min_hole,
        )
        Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / f"{stem}.png")
        records.append({"image": image_path.name, "mask_area": int(mask.sum())})

    (output_dir.parent / "inference_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved refined masks to: {output_dir}")


if __name__ == "__main__":
    main()
