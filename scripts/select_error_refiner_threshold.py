#!/usr/bin/env python3
"""Select the best inference threshold for the error refiner on a validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics, mean_metrics, read_binary_mask as read_eval_mask
from infer_error_refiner import build_input, load_model_bundle, postprocess_mask
from train_error_refiner import find_image_files, read_binary_mask, read_image_gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds for a trained error refiner.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--baseline-exp", default="zero_shot_box_only_conf020_pad005")
    parser.add_argument("--refined-exp", default="zero_shot_mask_to_prompt_refine_pad003_neg2_k5")
    parser.add_argument("--refined-exp-2", default=None, help="Optional second refined candidate mask directory for multi-candidate fusion.")
    parser.add_argument("--checkpoint", default="outputs/error_refiner/TSRS_RSNA-Epiphysis/best.pt")
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60")
    parser.add_argument(
        "--target-mode",
        default=None,
        help="Override checkpoint target_mode when older checkpoints do not store it.",
    )
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--selection-dice-weight", type=float, default=0.30)
    parser.add_argument("--selection-iou-weight", type=float, default=0.25)
    parser.add_argument("--selection-precision-weight", type=float, default=0.25)
    parser.add_argument("--selection-boundary-weight", type=float, default=0.20)
    parser.add_argument("--post-median-ksize", type=int, default=3)
    parser.add_argument("--post-open-kernel", type=int, default=3)
    parser.add_argument("--post-min-component", type=int, default=48)
    parser.add_argument("--post-min-hole", type=int, default=24)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    model, img_size, target_mode = load_model_bundle(Path(args.checkpoint), args.device, args.target_mode)

    raw_root = Path(args.raw_root)
    ablations_root = Path(args.ablations_root)
    image_dir = raw_root / args.dataset / args.split
    gt_dir = raw_root / args.dataset / f"{args.split}_labels"
    baseline_dir = ablations_root / args.baseline_exp / args.dataset / args.split / "masks"
    refined_dir = ablations_root / args.refined_exp / args.dataset / args.split / "masks"
    refined_dir_2 = ablations_root / args.refined_exp_2 / args.dataset / args.split / "masks" if args.refined_exp_2 else None

    records_by_threshold: dict[float, list[dict[str, float]]] = {threshold: [] for threshold in thresholds}
    for image_path in tqdm(find_image_files(image_dir), desc=f"threshold-sweep/{args.dataset}/{args.split}"):
        stem = image_path.stem
        image = read_image_gray(image_path)
        baseline = read_binary_mask(baseline_dir / f"{stem}.png")
        refined = read_binary_mask(refined_dir / f"{stem}.png")
        refined_2 = read_binary_mask(refined_dir_2 / f"{stem}.png") if refined_dir_2 is not None else None
        gt = read_eval_mask(gt_dir / f"{stem}.png")
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
            prob = cv2.resize(prob.astype(np.float32), gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
        else:
            roi_h = int(roi_meta["y1"] - roi_meta["y0"])
            roi_w = int(roi_meta["x1"] - roi_meta["x0"])
            roi_prob = cv2.resize(prob.astype(np.float32), (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            full_prob = refined.astype(np.float32).copy()
            full_prob[roi_meta["y0"] : roi_meta["y1"], roi_meta["x0"] : roi_meta["x1"]] = roi_prob
            prob = full_prob
        for threshold in thresholds:
            pred = postprocess_mask(
                prob,
                threshold,
                args.post_median_ksize,
                args.post_open_kernel,
                args.post_min_component,
                args.post_min_hole,
            )
            metrics = compute_metrics(pred, gt, args.boundary_kernel)
            metrics["image"] = image_path.name
            records_by_threshold[threshold].append(metrics)

    threshold_summaries = []
    for threshold in thresholds:
        mean = mean_metrics(records_by_threshold[threshold])
        selection_score = (
            args.selection_dice_weight * float(mean.get("dice", 0.0))
            + args.selection_iou_weight * float(mean.get("iou", 0.0))
            + args.selection_precision_weight * float(mean.get("precision", 0.0))
            + args.selection_boundary_weight * float(mean.get("boundary_iou", 0.0))
        )
        threshold_summaries.append(
            {
                "threshold": threshold,
                "mean": mean,
                "selection_score": selection_score,
                "num_evaluated": len(records_by_threshold[threshold]),
            }
        )

    best = max(threshold_summaries, key=lambda item: float(item.get("selection_score", -1.0)))
    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "baseline_exp": args.baseline_exp,
        "refined_exp": args.refined_exp,
        "refined_exp_2": args.refined_exp_2,
        "target_mode": target_mode,
        "postprocess": {
            "median_ksize": args.post_median_ksize,
            "open_kernel": args.post_open_kernel,
            "min_component": args.post_min_component,
            "min_hole": args.post_min_hole,
        },
        "thresholds": threshold_summaries,
        "best": best,
    }
    output = Path(args.output) if args.output else Path(args.checkpoint).parent / f"threshold_sweep_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))
    print(f"Saved threshold sweep: {output}")


if __name__ == "__main__":
    main()
