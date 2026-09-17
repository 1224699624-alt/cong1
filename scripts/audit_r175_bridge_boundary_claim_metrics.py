#!/usr/bin/env python3
"""Summarize bridge/boundary claim metrics from existing clean-test-v2 results.

The script is intentionally conservative: it can run with metrics JSON files only,
and it computes richer mask-derived metrics only for experiments whose prediction
directories are available locally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_EXPERIMENTS = [
    ("R038 anchor", "outputs/analysis/r038_aggr_r1_t045_m000_clean_test_v2_metrics.json", ""),
    ("R090 patch arbitrator", "outputs/analysis/r090_online_patch_disagreement_arbitrator_clean_test_v2_metrics.json", ""),
    ("R100 patch complement", "outputs/analysis/r100_r095b_r097_patch_basic_clean_test_v2_metrics.json", ""),
    ("R110 current best", "outputs/analysis/r110_r100_r108_patch_basic_clean_test_v2_metrics.json", ""),
    ("R130 bridge penalty source", "outputs/analysis/r130_dinov3_bridge_suppressed_instance_sep_clean_test_v2_metrics.json", ""),
    ("R140 reviewed variant source", "outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_clean_test_v2_metrics.json", ""),
    ("R143 high-res medical recipe", "outputs/analysis/r143_highres_medical_recipe_unet_clean_test_v2_metrics.json", ""),
    ("R165 filtered reannotation source", "outputs/analysis/r165_filtered_reannotated_dinov3_instance_sep_clean_test_v2_metrics.json", ""),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R175 bridge/boundary claim metrics.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--output", default="outputs/analysis/r175_bridge_boundary_claim_metrics.json")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        metavar="NAME=METRICS_JSON[:PRED_DIR]",
        help="Optional experiment spec. If omitted, uses known project results.",
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def boundary(mask: np.ndarray, kernel: int = 3) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(pil_mask.filter(ImageFilter.MaxFilter(kernel))) > 0
    eroded = np.asarray(pil_mask.filter(ImageFilter.MinFilter(kernel))) > 0
    return np.logical_xor(dilated, eroded)


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def components(mask: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    return int(n), labels, np.zeros((int(n) + 1, 5), dtype=np.float32)


def mask_derived_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from scipy import ndimage

    gt_n, _, _ = components(gt)
    pred_n, _, _ = components(pred)

    gt_boundary = boundary(gt, 3)
    pred_boundary = boundary(pred, 3)
    b_tp = float(np.logical_and(gt_boundary, pred_boundary).sum())
    b_fp = float(np.logical_and(~gt_boundary, pred_boundary).sum())
    b_fn = float(np.logical_and(gt_boundary, ~pred_boundary).sum())
    boundary_f1 = safe_divide(2.0 * b_tp, 2.0 * b_tp + b_fp + b_fn)

    structure = np.ones((9, 9), dtype=bool)
    separation_band = np.logical_and(ndimage.binary_dilation(gt, structure=structure), ~gt)

    # Foreground in a GT-adjacent background band is a practical proxy for bone-gap adhesion.
    separation_band_pixels = float(separation_band.sum())
    gap_fp = float(np.logical_and(pred, separation_band).sum())
    gap_fp_rate = safe_divide(gap_fp, separation_band_pixels)
    gap_precision = 1.0 - gap_fp_rate

    false_bridge_flag = float(pred_n < gt_n)
    component_abs_error = float(abs(pred_n - gt_n))
    component_delta = float(pred_n - gt_n)

    return {
        "component_count_error": component_abs_error,
        "component_delta": component_delta,
        "false_bridge_flag": false_bridge_flag,
        "boundary_f1": boundary_f1,
        "separation_band_fp_rate": gap_fp_rate,
        "separation_band_precision": gap_precision,
    }


def parse_experiment_specs(raw_specs: list[str]) -> list[tuple[str, str, str]]:
    if not raw_specs:
        return DEFAULT_EXPERIMENTS
    specs = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError(f"Bad --experiment spec: {raw}")
        name, rest = raw.split("=", 1)
        parts = rest.split(":", 1)
        metrics_json = parts[0]
        pred_dir = parts[1] if len(parts) > 1 else ""
        specs.append((name.strip(), metrics_json.strip(), pred_dir.strip()))
    return specs


def summarize_records(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record if key != "image"})
    return {key: float(mean([float(record[key]) for record in records if key in record])) for key in keys}


def load_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    mean_metrics = data.get("mean", {})
    per_image = data.get("per_image", [])
    counts = {
        "dice_below_090": sum(1 for row in per_image if float(row.get("dice", 1.0)) < 0.90),
        "boundary_iou_below_020": sum(1 for row in per_image if float(row.get("boundary_iou", 1.0)) < 0.20),
    }
    if per_image and any("false_bridge_flag" in row for row in per_image):
        counts["false_bridge_cases"] = sum(1 for row in per_image if float(row.get("false_bridge_flag", 0.0)) >= 0.5)
    return {
        "path": str(path),
        "exists": path.exists(),
        "num_evaluated": int(data.get("num_evaluated", len(per_image))),
        "mean": mean_metrics,
        "counts": counts,
    }


def enrich_from_masks(raw_root: Path, dataset: str, split: str, pred_dir: Path) -> dict:
    gt_dir = raw_root / dataset / f"{split}_labels"
    records: list[dict[str, float]] = []
    missing = []
    for gt_path in sorted(gt_dir.glob("*.png")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            missing.append(gt_path.name)
            continue
        gt = read_mask(gt_path)
        pred = read_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        row = mask_derived_metrics(gt, pred)
        row["image"] = gt_path.name
        records.append(row)
    return {
        "pred_dir": str(pred_dir),
        "num_mask_records": len(records),
        "num_missing_masks": len(missing),
        "missing_masks": missing[:25],
        "mean": summarize_records(records),
        "per_image": records,
    }


def rank_experiments(results: list[dict]) -> dict[str, list[str]]:
    available = [item for item in results if item.get("metrics", {}).get("exists")]

    def metric(item: dict, key: str, default: float, reverse: bool = True) -> tuple[float, str]:
        value = item.get("metrics", {}).get("mean", {}).get(key, default)
        return (float(value), item["name"]) if reverse else (-float(value), item["name"])

    rankings = {
        "dice_desc": [name for _, name in sorted([metric(item, "dice", -1.0) for item in available], reverse=True)],
        "boundary_iou_desc": [name for _, name in sorted([metric(item, "boundary_iou", -1.0) for item in available], reverse=True)],
        "precision_desc": [name for _, name in sorted([metric(item, "precision", -1.0) for item in available], reverse=True)],
    }
    if any("false_bridge_flag" in item.get("metrics", {}).get("mean", {}) for item in available):
        rankings["false_bridge_asc"] = [
            name for _, name in sorted([metric(item, "false_bridge_flag", 1e9, reverse=False) for item in available], reverse=True)
        ]
    return rankings


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    results = []
    for name, metrics_json, pred_dir in parse_experiment_specs(args.experiment):
        path = Path(metrics_json)
        item = {"name": name, "metrics_json": str(path)}
        if path.exists():
            item["metrics"] = load_metrics(path)
        else:
            item["metrics"] = {"exists": False, "path": str(path)}
        if pred_dir:
            pred_path = Path(pred_dir)
            item["mask_derived"] = enrich_from_masks(raw_root, args.dataset, args.split, pred_path) if pred_path.exists() else {
                "pred_dir": str(pred_path),
                "exists": False,
            }
        results.append(item)

    output = {
        "run_id": "R175",
        "purpose": "Bridge/boundary claim metric audit for epiphysis segmentation.",
        "dataset": args.dataset,
        "split": args.split,
        "rule": "clean-test-v2 is final evaluation/diagnostic only; do not tune thresholds from this file.",
        "claim_metrics": {
            "primary": ["dice", "boundary_iou", "false_bridge_flag", "component_count_error"],
            "mask_available_extra": ["boundary_f1", "separation_band_fp_rate", "separation_band_precision"],
        },
        "rankings": rank_experiments(results),
        "experiments": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "rankings": output["rankings"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
