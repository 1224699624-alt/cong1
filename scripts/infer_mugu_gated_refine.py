#!/usr/bin/env python3
"""MuGu-style gated refinement for YOLO + SAM/HQ-SAM segmentation.

High-confidence samples keep a lightweight base refined mask.
Low-confidence samples trigger a heavier HQ-SAM refinement branch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from infer_yolo_sam import (
    boxes_to_union_mask,
    expand_boxes,
    find_image_files,
    load_sam,
    parse_float_list,
    predict_mask_to_prompt_refined_masks,
    predict_sam_masks,
    read_rgb,
    save_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuGu-style gated SAM refinement inference.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-root", default="outputs/ablations/mugu_gated_refine")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--box-padding-ratio", type=float, default=0.05)
    parser.add_argument("--min-mask-area", type=int, default=20)
    parser.add_argument("--save-overlays", action="store_true")

    parser.add_argument("--base-sam-checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--base-sam-model-type", default="vit_b")
    parser.add_argument("--base-sam-backend", default="sam", choices=["sam", "hq_sam"])

    parser.add_argument("--heavy-sam-checkpoint", default="checkpoints/sam_hq_vit_b.pth")
    parser.add_argument("--heavy-sam-model-type", default="vit_b")
    parser.add_argument("--heavy-sam-backend", default="hq_sam", choices=["sam", "hq_sam"])

    parser.add_argument("--base-refine-box-padding-ratio", type=float, default=0.03)
    parser.add_argument("--base-refine-num-positive-points", type=int, default=1)
    parser.add_argument("--base-refine-num-negative-points", type=int, default=2)
    parser.add_argument("--base-refine-negative-dilate-kernel", type=int, default=5)
    parser.add_argument("--base-refine-iters", type=int, default=2)
    parser.add_argument("--base-refine-prompt-scheme", default="box+point+mask")

    parser.add_argument("--heavy-refine-box-padding-ratio", type=float, default=0.03)
    parser.add_argument("--heavy-refine-num-positive-points", type=int, default=1)
    parser.add_argument("--heavy-refine-num-negative-points", type=int, default=2)
    parser.add_argument("--heavy-refine-negative-dilate-kernel", type=int, default=9)
    parser.add_argument("--heavy-refine-iters", type=int, default=2)
    parser.add_argument("--heavy-refine-prompt-scheme", default="box+point+mask")

    parser.add_argument("--gating-disagreement-threshold", type=float, default=0.12)
    parser.add_argument("--gating-base-score-threshold", type=float, default=0.58)
    parser.add_argument("--gating-heavy-score-margin", type=float, default=0.01)
    parser.add_argument("--target-mask-box-ratio", type=float, default=0.45)
    parser.add_argument("--max-outside-box-ratio", type=float, default=0.08)
    parser.add_argument("--save-gating-overlays", action="store_true")
    return parser.parse_args()


def score_mask_consistency(
    mask: np.ndarray,
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    target_mask_box_ratio: float,
    max_outside_box_ratio: float,
) -> dict[str, float]:
    mask_bool = mask.astype(bool)
    mask_area = float(mask_bool.sum())
    if mask_area <= 0:
        return {
            "score": 0.0,
            "mask_box_ratio": 0.0,
            "outside_box_ratio": 1.0,
        }
    box_union = boxes_to_union_mask(boxes, image_shape)
    box_area = float(box_union.sum())
    mask_box_ratio = mask_area / box_area if box_area > 0 else 0.0
    outside_ratio = float(np.logical_and(mask_bool, ~box_union).sum() / mask_area)
    area_penalty = min(1.0, abs(mask_box_ratio - target_mask_box_ratio) / max(target_mask_box_ratio, 1e-6))
    outside_penalty = max(0.0, outside_ratio - max_outside_box_ratio) / max(1.0 - max_outside_box_ratio, 1e-6)
    score = 1.0 - 0.65 * area_penalty - 0.35 * outside_penalty
    return {
        "score": float(score),
        "mask_box_ratio": float(mask_box_ratio),
        "outside_box_ratio": float(outside_ratio),
    }


def disagreement_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_xor(a, b).sum() / union)


def save_gating_overlay(
    image_rgb: np.ndarray,
    final_mask: np.ndarray,
    base_mask: np.ndarray,
    heavy_mask: np.ndarray,
    output_path: Path,
) -> None:
    overlay = image_rgb.copy()
    final_region = final_mask > 0
    base_only = (base_mask > 0) & ~(heavy_mask > 0)
    heavy_only = (heavy_mask > 0) & ~(base_mask > 0)
    both = (base_mask > 0) & (heavy_mask > 0)
    overlay[both] = (0.55 * overlay[both] + 0.45 * np.array([255, 210, 0])).astype(np.uint8)
    overlay[base_only] = (0.55 * overlay[base_only] + 0.45 * np.array([40, 170, 255])).astype(np.uint8)
    overlay[heavy_only] = (0.55 * overlay[heavy_only] + 0.45 * np.array([255, 60, 60])).astype(np.uint8)
    overlay[final_region] = (0.7 * overlay[final_region] + 0.3 * np.array([60, 255, 120])).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.raw_root) / args.dataset / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split not found: {image_dir}")

    out_dir = Path(args.out_root) / args.dataset / args.split
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    gating_overlay_dir = out_dir / "gating_overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if args.save_gating_overlays:
        gating_overlay_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.yolo_weights)
    base_predictor = load_sam(
        Path(args.base_sam_checkpoint),
        args.base_sam_model_type,
        args.device,
        backend=args.base_sam_backend,
    )
    heavy_predictor = load_sam(
        Path(args.heavy_sam_checkpoint),
        args.heavy_sam_model_type,
        args.device,
        backend=args.heavy_sam_backend,
    )

    records = []
    for image_path in tqdm(find_image_files(image_dir), desc=f"mugu-gated/{args.dataset}/{args.split}"):
        image_rgb = read_rgb(image_path)
        result = yolo.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            device=args.device,
        )[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
        scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))
        raw_boxes = boxes.astype(np.float32).reshape(-1, 4)
        padded_boxes = expand_boxes(raw_boxes, image_rgb.shape[:2], args.box_padding_ratio)

        if raw_boxes.size == 0:
            final_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
            final_boxes = padded_boxes
            chosen_branch = "empty"
            base_mask = final_mask
            heavy_mask = final_mask
            base_stats = {"score": 0.0, "mask_box_ratio": 0.0, "outside_box_ratio": 1.0}
            heavy_stats = dict(base_stats)
            disagree = 0.0
            trigger_heavy = False
        else:
            base_mask, base_boxes, _ = predict_mask_to_prompt_refined_masks(
                base_predictor,
                image_rgb,
                padded_boxes,
                args.device,
                prompt_mode="box",
                refine_box_padding_ratio=args.base_refine_box_padding_ratio,
                refine_num_positive_points=args.base_refine_num_positive_points,
                refine_num_negative_points=args.base_refine_num_negative_points,
                refine_negative_dilate_kernel=args.base_refine_negative_dilate_kernel,
                refine_iters=args.base_refine_iters,
                refine_prompt_scheme=args.base_refine_prompt_scheme,
                use_mask_input=True,
            )
            base_stats = score_mask_consistency(
                base_mask,
                base_boxes,
                image_rgb.shape[:2],
                args.target_mask_box_ratio,
                args.max_outside_box_ratio,
            )

            heavy_mask, heavy_boxes, _ = predict_mask_to_prompt_refined_masks(
                heavy_predictor,
                image_rgb,
                padded_boxes,
                args.device,
                prompt_mode="box",
                refine_box_padding_ratio=args.heavy_refine_box_padding_ratio,
                refine_num_positive_points=args.heavy_refine_num_positive_points,
                refine_num_negative_points=args.heavy_refine_num_negative_points,
                refine_negative_dilate_kernel=args.heavy_refine_negative_dilate_kernel,
                refine_iters=args.heavy_refine_iters,
                refine_prompt_scheme=args.heavy_refine_prompt_scheme,
                use_mask_input=True,
            )
            heavy_stats = score_mask_consistency(
                heavy_mask,
                heavy_boxes,
                image_rgb.shape[:2],
                args.target_mask_box_ratio,
                args.max_outside_box_ratio,
            )

            disagree = disagreement_ratio(base_mask, heavy_mask)
            trigger_heavy = (
                disagree >= args.gating_disagreement_threshold
                or base_stats["score"] < args.gating_base_score_threshold
            )
            if trigger_heavy and heavy_stats["score"] > base_stats["score"] + args.gating_heavy_score_margin:
                final_mask = heavy_mask
                final_boxes = heavy_boxes
                chosen_branch = "heavy_hqsam_refine"
            else:
                final_mask = base_mask
                final_boxes = base_boxes
                chosen_branch = "base_sam_refine"

        if int(final_mask.sum()) < args.min_mask_area:
            final_mask = np.zeros_like(final_mask, dtype=np.uint8)

        Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_dir / f"{image_path.stem}.png")
        if args.save_overlays:
            save_overlay(image_rgb, final_mask, final_boxes, overlay_dir / f"{image_path.stem}.jpg")
        if args.save_gating_overlays:
            save_gating_overlay(
                image_rgb,
                final_mask,
                base_mask,
                heavy_mask,
                gating_overlay_dir / f"{image_path.stem}.jpg",
            )

        records.append(
            {
                "image": image_path.name,
                "num_boxes": int(len(raw_boxes)),
                "scores": [float(score) for score in scores],
                "mask_area": int(final_mask.sum()),
                "chosen_branch": chosen_branch,
                "trigger_heavy": bool(trigger_heavy),
                "disagreement_ratio": float(disagree),
                "base_score": float(base_stats["score"]),
                "base_mask_box_ratio": float(base_stats["mask_box_ratio"]),
                "base_outside_box_ratio": float(base_stats["outside_box_ratio"]),
                "heavy_score": float(heavy_stats["score"]),
                "heavy_mask_box_ratio": float(heavy_stats["mask_box_ratio"]),
                "heavy_outside_box_ratio": float(heavy_stats["outside_box_ratio"]),
                "base_backend": args.base_sam_backend,
                "heavy_backend": args.heavy_sam_backend,
            }
        )

    (out_dir / "inference_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved masks to: {mask_dir}")


if __name__ == "__main__":
    main()

