#!/usr/bin/env python3
"""Diagnose R191 MaskDINO COCO predictions without using clean-test-v2."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--score-thresholds", default="0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--top-k", default="1,3,5,10,20,40,80")
    return parser.parse_args()


def decode_rle(rle: dict, shape: tuple[int, int] | None = None) -> np.ndarray:
    arr = mask_utils.decode(rle)
    if arr.ndim == 3:
        arr = np.any(arr > 0, axis=2)
    else:
        arr = arr > 0
    if shape is not None and arr.shape != shape:
        raise ValueError(f"mask shape mismatch: got {arr.shape}, expected {shape}")
    return arr


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    denom = pred_sum + gt_sum
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_iou(a: list[int] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    iw = max(0, x1 - x0 + 1)
    ih = max(0, y1 - y0 + 1)
    inter = iw * ih
    area_a = max(0, a[2] - a[0] + 1) * max(0, a[3] - a[1] + 1)
    area_b = max(0, b[2] - b[0] + 1) * max(0, b[3] - b[1] + 1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom else 0.0


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    coco = json.loads(Path(args.coco_json).read_text(encoding="utf-8"))
    preds = json.loads(Path(args.pred_json).read_text(encoding="utf-8"))
    thresholds = [float(x) for x in args.score_thresholds.split(",") if x.strip()]
    top_ks = [int(x) for x in args.top_k.split(",") if x.strip()]

    images = {int(im["id"]): im for im in coco["images"]}
    gt_by_image: dict[int, list[np.ndarray]] = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        im = images[image_id]
        shape = (int(im["height"]), int(im["width"]))
        gt_by_image[image_id].append(decode_rle(ann["segmentation"], shape=shape))

    pred_by_image: dict[int, list[dict]] = defaultdict(list)
    for pred in preds:
        pred_by_image[int(pred["image_id"])].append(pred)
    for items in pred_by_image.values():
        items.sort(key=lambda p: float(p.get("score", 0.0)), reverse=True)

    per_image = []
    all_scores: list[float] = []
    all_areas: list[float] = []
    max_best_bbox_ious: list[float] = []
    threshold_dice = {str(t): [] for t in thresholds}
    threshold_iou = {str(t): [] for t in thresholds}
    threshold_area_ratio = {str(t): [] for t in thresholds}
    topk_dice = {str(k): [] for k in top_ks}
    topk_iou = {str(k): [] for k in top_ks}
    topk_area_ratio = {str(k): [] for k in top_ks}

    for image_id, im in sorted(images.items()):
        shape = (int(im["height"]), int(im["width"]))
        gt_masks = gt_by_image.get(image_id, [])
        gt_union = np.zeros(shape, dtype=bool)
        for mask in gt_masks:
            gt_union |= mask
        gt_bbox = bbox_from_mask(gt_union)
        gt_area = int(gt_union.sum())

        image_preds = pred_by_image.get(image_id, [])
        decoded_preds = []
        for pred in image_preds:
            mask = decode_rle(pred["segmentation"], shape=shape)
            score = float(pred.get("score", 0.0))
            area = int(mask.sum())
            all_scores.append(score)
            all_areas.append(area)
            decoded_preds.append((score, mask, area))

        best_bbox = 0.0
        for _, mask, _ in decoded_preds[:10]:
            best_bbox = max(best_bbox, bbox_iou(bbox_from_mask(mask), gt_bbox))
        max_best_bbox_ious.append(best_bbox)

        image_record = {
            "image_id": image_id,
            "file_name": im["file_name"],
            "gt_instances": len(gt_masks),
            "gt_area": gt_area,
            "pred_count": len(decoded_preds),
            "max_score": max((x[0] for x in decoded_preds), default=0.0),
            "top10_best_bbox_iou": best_bbox,
            "thresholds": {},
            "topk": {},
        }
        for t in thresholds:
            union = np.zeros(shape, dtype=bool)
            kept = 0
            for score, mask, _ in decoded_preds:
                if score >= t:
                    union |= mask
                    kept += 1
            d = dice(union, gt_union)
            j = iou(union, gt_union)
            ratio = float(union.sum() / gt_area) if gt_area else 0.0
            threshold_dice[str(t)].append(d)
            threshold_iou[str(t)].append(j)
            threshold_area_ratio[str(t)].append(ratio)
            image_record["thresholds"][str(t)] = {
                "kept": kept,
                "dice": d,
                "iou": j,
                "pred_area_ratio": ratio,
            }
        for k in top_ks:
            union = np.zeros(shape, dtype=bool)
            for _, mask, _ in decoded_preds[:k]:
                union |= mask
            d = dice(union, gt_union)
            j = iou(union, gt_union)
            ratio = float(union.sum() / gt_area) if gt_area else 0.0
            topk_dice[str(k)].append(d)
            topk_iou[str(k)].append(j)
            topk_area_ratio[str(k)].append(ratio)
            image_record["topk"][str(k)] = {
                "dice": d,
                "iou": j,
                "pred_area_ratio": ratio,
            }
        per_image.append(image_record)

    best_threshold = max(thresholds, key=lambda t: float(np.mean(threshold_dice[str(t)])))
    best_topk = max(top_ks, key=lambda k: float(np.mean(topk_dice[str(k)])))
    out = {
        "status": "completed",
        "inputs": {
            "coco_json": args.coco_json,
            "pred_json": args.pred_json,
            "clean_test_v2_used": False,
            "new_or_reannotated_test_used": False,
        },
        "images": len(images),
        "predictions": len(preds),
        "prediction_count_per_image": summarize([float(r["pred_count"]) for r in per_image]),
        "score_summary": summarize(all_scores),
        "pred_mask_area_summary": summarize(all_areas),
        "top10_best_bbox_iou_summary": summarize(max_best_bbox_ious),
        "threshold_union": {
            str(t): {
                "dice": summarize(threshold_dice[str(t)]),
                "iou": summarize(threshold_iou[str(t)]),
                "pred_area_ratio": summarize(threshold_area_ratio[str(t)]),
            }
            for t in thresholds
        },
        "topk_union": {
            str(k): {
                "dice": summarize(topk_dice[str(k)]),
                "iou": summarize(topk_iou[str(k)]),
                "pred_area_ratio": summarize(topk_area_ratio[str(k)]),
            }
            for k in top_ks
        },
        "best_threshold_by_val_union_dice": {
            "threshold": best_threshold,
            "mean_dice": float(np.mean(threshold_dice[str(best_threshold)])),
            "mean_iou": float(np.mean(threshold_iou[str(best_threshold)])),
        },
        "best_topk_by_val_union_dice": {
            "top_k": best_topk,
            "mean_dice": float(np.mean(topk_dice[str(best_topk)])),
            "mean_iou": float(np.mean(topk_iou[str(best_topk)])),
        },
        "worst_images_by_best_threshold": sorted(
            (
                {
                    "file_name": r["file_name"],
                    "pred_count": r["pred_count"],
                    "max_score": r["max_score"],
                    "dice": r["thresholds"][str(best_threshold)]["dice"],
                    "pred_area_ratio": r["thresholds"][str(best_threshold)]["pred_area_ratio"],
                    "top10_best_bbox_iou": r["top10_best_bbox_iou"],
                }
                for r in per_image
            ),
            key=lambda x: x["dice"],
        )[:12],
        "per_image": per_image,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "status",
        "images",
        "predictions",
        "prediction_count_per_image",
        "score_summary",
        "top10_best_bbox_iou_summary",
        "best_threshold_by_val_union_dice",
        "best_topk_by_val_union_dice",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
