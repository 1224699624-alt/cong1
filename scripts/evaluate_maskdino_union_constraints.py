#!/usr/bin/env python3
"""Evaluate MaskDINO binary-union predictions with boundary/bridge constraints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from pycocotools import mask as mask_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--export-dir", default="")
    parser.add_argument("--score-thresholds", default="0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.4")
    parser.add_argument("--top-k", default="1,2,3,5,10,20")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--separation-radius", type=int, default=9)
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


def summarize(values: list[float]) -> dict[str, float | int]:
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


def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    denom = int(pred.sum()) + int(gt.sum())
    return safe_div(2.0 * float(np.logical_and(pred, gt).sum()), float(denom), default=1.0)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = int(np.logical_or(pred, gt).sum())
    return safe_div(float(np.logical_and(pred, gt).sum()), float(union), default=1.0)


def boundary(mask: np.ndarray, kernel: int) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(pil_mask.filter(ImageFilter.MaxFilter(kernel))) > 0
    eroded = np.asarray(pil_mask.filter(ImageFilter.MinFilter(kernel))) > 0
    return np.logical_xor(dilated, eroded)


def component_count(mask: np.ndarray) -> int:
    from scipy import ndimage

    _, n = ndimage.label(mask)
    return int(n)


def constraint_metrics(gt: np.ndarray, pred: np.ndarray, boundary_kernel: int, separation_radius: int) -> dict[str, float]:
    from scipy import ndimage

    gt_boundary = boundary(gt, boundary_kernel)
    pred_boundary = boundary(pred, boundary_kernel)
    b_tp = float(np.logical_and(gt_boundary, pred_boundary).sum())
    b_fp = float(np.logical_and(~gt_boundary, pred_boundary).sum())
    b_fn = float(np.logical_and(gt_boundary, ~pred_boundary).sum())
    b_union = float(np.logical_or(gt_boundary, pred_boundary).sum())

    structure = np.ones((separation_radius, separation_radius), dtype=bool)
    separation_band = np.logical_and(ndimage.binary_dilation(gt, structure=structure), ~gt)
    sep_pixels = float(separation_band.sum())
    sep_fp = float(np.logical_and(pred, separation_band).sum())

    gt_n = component_count(gt)
    pred_n = component_count(pred)
    return {
        "dice": dice(pred, gt),
        "iou": iou(pred, gt),
        "boundary_iou": safe_div(b_tp, b_union),
        "boundary_f1": safe_div(2.0 * b_tp, 2.0 * b_tp + b_fp + b_fn),
        "separation_band_fp_rate": safe_div(sep_fp, sep_pixels),
        "component_count_error": float(abs(pred_n - gt_n)),
        "component_delta": float(pred_n - gt_n),
        "false_bridge_flag": float(pred_n < gt_n),
        "pred_area_ratio": safe_div(float(pred.sum()), float(gt.sum())),
        "gt_components": float(gt_n),
        "pred_components": float(pred_n),
    }


def union_for_threshold(decoded_preds: list[tuple[float, np.ndarray]], threshold: float, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for score, mask in decoded_preds:
        if score >= threshold:
            out |= mask
    return out


def union_for_topk(decoded_preds: list[tuple[float, np.ndarray]], top_k: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for _, mask in decoded_preds[:top_k]:
        out |= mask
    return out


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
    for image_preds in pred_by_image.values():
        image_preds.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    threshold_rows: dict[str, list[dict[str, float]]] = {str(t): [] for t in thresholds}
    topk_rows: dict[str, list[dict[str, float]]] = {str(k): [] for k in top_ks}
    decoded_cache: dict[int, list[tuple[float, np.ndarray]]] = {}
    gt_cache: dict[int, np.ndarray] = {}
    per_image_meta = []

    for image_id, im in sorted(images.items()):
        shape = (int(im["height"]), int(im["width"]))
        gt = np.zeros(shape, dtype=bool)
        for mask in gt_by_image.get(image_id, []):
            gt |= mask
        gt_cache[image_id] = gt

        decoded = []
        for pred in pred_by_image.get(image_id, []):
            decoded.append((float(pred.get("score", 0.0)), decode_rle(pred["segmentation"], shape=shape)))
        decoded_cache[image_id] = decoded
        per_image_meta.append({
            "image_id": image_id,
            "file_name": im["file_name"],
            "gt_area": int(gt.sum()),
            "gt_components": component_count(gt),
            "pred_count": len(decoded),
            "max_score": max((score for score, _ in decoded), default=0.0),
        })

        for t in thresholds:
            pred_union = union_for_threshold(decoded, t, shape)
            threshold_rows[str(t)].append(constraint_metrics(gt, pred_union, args.boundary_kernel, args.separation_radius))
        for k in top_ks:
            pred_union = union_for_topk(decoded, k, shape)
            topk_rows[str(k)].append(constraint_metrics(gt, pred_union, args.boundary_kernel, args.separation_radius))

    def summarize_group(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
        keys = sorted({key for row in rows for key in row})
        return {key: summarize([row[key] for row in rows]) for key in keys}

    threshold_summary = {key: summarize_group(rows) for key, rows in threshold_rows.items()}
    topk_summary = {key: summarize_group(rows) for key, rows in topk_rows.items()}
    best_threshold = max(thresholds, key=lambda t: float(threshold_summary[str(t)]["dice"]["mean"]))
    best_topk = max(top_ks, key=lambda k: float(topk_summary[str(k)]["dice"]["mean"]))

    selected_mode = "threshold"
    selected_value: float | int = best_threshold
    selected_rows = threshold_rows[str(best_threshold)]
    if float(topk_summary[str(best_topk)]["dice"]["mean"]) > float(threshold_summary[str(best_threshold)]["dice"]["mean"]):
        selected_mode = "topk"
        selected_value = best_topk
        selected_rows = topk_rows[str(best_topk)]

    selected_per_image = []
    export_dir = Path(args.export_dir) if args.export_dir else None
    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)

    for idx, meta in enumerate(per_image_meta):
        image_id = int(meta["image_id"])
        shape = gt_cache[image_id].shape
        if selected_mode == "threshold":
            pred_union = union_for_threshold(decoded_cache[image_id], float(selected_value), shape)
        else:
            pred_union = union_for_topk(decoded_cache[image_id], int(selected_value), shape)
        row = dict(meta)
        row.update(selected_rows[idx])
        selected_per_image.append(row)
        if export_dir:
            out_name = Path(str(meta["file_name"])).with_suffix(".png").name
            Image.fromarray(pred_union.astype(np.uint8) * 255).save(export_dir / out_name)

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
        "threshold_summary": threshold_summary,
        "topk_summary": topk_summary,
        "best_threshold_by_dice": {
            "threshold": best_threshold,
            "dice": threshold_summary[str(best_threshold)]["dice"],
            "boundary_iou": threshold_summary[str(best_threshold)]["boundary_iou"],
            "separation_band_fp_rate": threshold_summary[str(best_threshold)]["separation_band_fp_rate"],
        },
        "best_topk_by_dice": {
            "top_k": best_topk,
            "dice": topk_summary[str(best_topk)]["dice"],
            "boundary_iou": topk_summary[str(best_topk)]["boundary_iou"],
            "separation_band_fp_rate": topk_summary[str(best_topk)]["separation_band_fp_rate"],
        },
        "selected_readout": {
            "mode": selected_mode,
            "value": selected_value,
            "mean": summarize_group(selected_rows),
            "export_dir": str(export_dir) if export_dir else "",
        },
        "worst_images_by_selected_dice": sorted(selected_per_image, key=lambda row: float(row["dice"]))[:20],
        "per_image": selected_per_image,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "images": out["images"],
        "selected_readout": out["selected_readout"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
