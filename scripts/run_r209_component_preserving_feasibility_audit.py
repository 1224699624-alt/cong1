#!/usr/bin/env python3
"""R209-F0 component-preserving instance feasibility audit.

This train/val-only audit checks whether R110 anatomy failures are better
framed as instance/component separation instead of pixel deletion. It does not
train a model and must not be used on clean-test-v2 for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from run_r201_unified_eval import build_gt_cache, compute_metrics, read_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit component-preserving instance feasibility on train/val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--splits", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--min-merged-image-rate", type=float, default=0.10)
    parser.add_argument("--max-oracle-component-mae-delta", type=float, default=0.0)
    parser.add_argument("--min-oracle-gap-delta", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r209_component_preserving_feasibility_audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r209_component_preserving_feasibility_audit.csv"))
    return parser.parse_args()


def read_instance(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(pil.resize((shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def instance_ids(instance: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(instance) if int(v) > 0]


def analyze_component_matches(anchor: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> dict[str, Any]:
    pred_labels, pred_n = component_labels(anchor)
    ids = instance_ids(instance)
    gt_binary = instance > 0
    gt_labels, gt_n = component_labels(gt_binary)

    ids_arr = np.asarray(ids, dtype=np.int32)
    pred_area = np.bincount(pred_labels.ravel(), minlength=pred_n + 1).astype(np.float64)
    if ids_arr.size:
        gt_area = np.asarray([(instance == gt_id).sum() for gt_id in ids], dtype=np.float64)
        valid = np.logical_and(pred_labels > 0, instance > 0)
        pred_valid = pred_labels[valid].astype(np.int64) - 1
        inst_valid = instance[valid].astype(np.int32)
        cols = np.searchsorted(ids_arr, inst_valid)
        col_valid = np.logical_and(cols < ids_arr.size, ids_arr[cols] == inst_valid)
        overlap = np.zeros((pred_n, ids_arr.size), dtype=np.int64)
        if pred_valid.size and col_valid.any():
            flat = pred_valid[col_valid] * ids_arr.size + cols[col_valid]
            overlap = np.bincount(flat, minlength=pred_n * ids_arr.size).reshape(pred_n, ids_arr.size)
    else:
        gt_area = np.zeros((0,), dtype=np.float64)
        overlap = np.zeros((pred_n, 0), dtype=np.int64)

    pred_rows: list[dict[str, Any]] = []
    merged_pred_components = 0
    single_pred_components = 0
    unmatched_pred_components = 0
    for pred_id in range(1, pred_n + 1):
        comp_area = int(pred_area[pred_id])
        inter_row = overlap[pred_id - 1] if overlap.size else np.zeros((0,), dtype=np.int64)
        nonzero_cols = np.flatnonzero(inter_row > 0)
        overlaps = [
            {
                "gt_instance_id": int(ids_arr[col]),
                "intersection": int(inter_row[col]),
                "frac_of_pred": float(inter_row[col] / max(1, comp_area)),
                "frac_of_gt": float(inter_row[col] / max(1.0, gt_area[col])),
            }
            for col in nonzero_cols
        ]
        meaningful = [o for o in overlaps if o["frac_of_gt"] >= min_overlap_frac or o["frac_of_pred"] >= min_overlap_frac]
        if len(meaningful) >= 2:
            merged_pred_components += 1
        elif len(meaningful) == 1:
            single_pred_components += 1
        else:
            unmatched_pred_components += 1
        pred_rows.append(
            {
                "pred_component_id": pred_id,
                "area": comp_area,
                "num_meaningful_gt_instances": len(meaningful),
                "gt_instance_ids": [o["gt_instance_id"] for o in meaningful],
            }
        )

    gt_covered = 0
    gt_multi_covered = 0
    for col, gt_id in enumerate(ids):
        covering = [
            pred_id + 1
            for pred_id, inter in enumerate(overlap[:, col] if overlap.size else [])
            if float(inter) / max(1.0, gt_area[col]) >= min_overlap_frac
        ]
        if covering:
            gt_covered += 1
        if len(covering) >= 2:
            gt_multi_covered += 1

    return {
        "pred_component_count": int(pred_n),
        "gt_component_count_binary": int(gt_n),
        "gt_instance_count": len(ids),
        "merged_pred_components": merged_pred_components,
        "single_pred_components": single_pred_components,
        "unmatched_pred_components": unmatched_pred_components,
        "gt_instances_covered": gt_covered,
        "gt_instances_multi_covered": gt_multi_covered,
        "pred_rows": pred_rows,
    }


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    return float(np.mean(finite)) if finite else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "anchor_dice",
        "anchor_iou",
        "anchor_recall",
        "anchor_boundary_iou",
        "anchor_gap_region_fp_rate",
        "anchor_component_merge_rate",
        "anchor_component_count_mae",
        "oracle_dice",
        "oracle_iou",
        "oracle_recall",
        "oracle_boundary_iou",
        "oracle_gap_region_fp_rate",
        "oracle_component_merge_rate",
        "oracle_component_count_mae",
        "delta_dice",
        "delta_recall",
        "delta_boundary_iou",
        "delta_gap_region_fp_rate",
        "delta_component_merge_rate",
        "delta_component_count_mae",
        "merged_pred_components",
        "pred_component_count",
        "gt_instance_count",
        "gt_instances_covered",
    ]
    out = {key: mean_or_none([row.get(key) for row in rows]) for key in keys}
    out["num_images"] = len(rows)
    out["images_with_pred_merge"] = int(sum(int(row["merged_pred_components"]) > 0 for row in rows))
    out["pred_merge_image_rate"] = float(out["images_with_pred_merge"] / max(1, len(rows)))
    return out


def evaluate_split(args: argparse.Namespace, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    gt_dir = args.raw_root / args.dataset / f"{split}_labels"
    anchor_dir = args.ablations_root / args.anchor_exp / args.dataset / split / "masks"
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not anchor_dir.exists():
        raise FileNotFoundError(anchor_dir)
    cache = build_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)
    label_paths = sorted(gt_dir.glob("*.png"))
    if args.limit > 0:
        label_paths = label_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    total = len(label_paths)
    for idx, label_path in enumerate(label_paths, start=1):
        if idx == 1 or idx % 10 == 0 or idx == total:
            print(f"[R209-F0] {split}: {idx}/{total} {label_path.name}", flush=True)
        anchor_path = anchor_dir / label_path.name
        if not anchor_path.exists():
            missing.append(label_path.name)
            continue
        instance = read_instance(label_path)
        gt_binary = instance > 0
        anchor = resize_bool(read_binary_mask(anchor_path), gt_binary.shape)
        gt_cache = cache[label_path.name]

        anchor_metrics = compute_metrics(
            anchor,
            gt_binary,
            args.boundary_kernel,
            args.gap_kernel,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
        oracle = gt_binary
        oracle_metrics = compute_metrics(
            oracle,
            gt_binary,
            args.boundary_kernel,
            args.gap_kernel,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
        match = analyze_component_matches(anchor, instance, args.min_overlap_frac)
        row: dict[str, Any] = {
            "split": split,
            "image": label_path.name,
            **{f"anchor_{k}": v for k, v in anchor_metrics.items()},
            **{f"oracle_{k}": v for k, v in oracle_metrics.items()},
            "pred_component_count": match["pred_component_count"],
            "gt_component_count_binary": match["gt_component_count_binary"],
            "gt_instance_count": match["gt_instance_count"],
            "merged_pred_components": match["merged_pred_components"],
            "single_pred_components": match["single_pred_components"],
            "unmatched_pred_components": match["unmatched_pred_components"],
            "gt_instances_covered": match["gt_instances_covered"],
            "gt_instances_multi_covered": match["gt_instances_multi_covered"],
        }
        for key in [
            "dice",
            "iou",
            "recall",
            "boundary_iou",
            "boundary_f1",
            "surface_dice_2px",
            "surface_dice_5px",
            "hd95_px",
            "assd_px",
            "gap_region_fp_rate",
            "component_merge_rate",
            "component_count_mae",
        ]:
            a = anchor_metrics.get(key)
            o = oracle_metrics.get(key)
            row[f"delta_{key}"] = None if a is None or o is None else float(o) - float(a)
        rows.append(row)
    return rows, missing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_rows: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    missing_by_split: dict[str, list[str]] = {}
    for split in splits:
        rows, missing = evaluate_split(args, split)
        all_rows.extend(rows)
        split_summaries[split] = summarize(rows)
        missing_by_split[split] = missing

    val = split_summaries.get("val") or split_summaries.get(splits[0], {})
    decision = "go_component_preserving_model_feasible" if (
        float(val.get("pred_merge_image_rate") or 0.0) >= args.min_merged_image_rate
        and float(val.get("delta_component_count_mae") or 0.0) <= args.max_oracle_component_mae_delta
        and float(val.get("delta_gap_region_fp_rate") or 0.0) <= -args.min_oracle_gap_delta
    ) else "no_go_component_oracle_not_sufficient"

    output = {
        "run_id": "R209-F0",
        "purpose": "train/val-only component-preserving instance feasibility audit",
        "clean_test_v2_used": False,
        "dataset": args.dataset,
        "anchor_exp": args.anchor_exp,
        "gate": {
            "min_overlap_frac": args.min_overlap_frac,
            "min_merged_image_rate": args.min_merged_image_rate,
            "max_oracle_component_mae_delta": args.max_oracle_component_mae_delta,
            "min_oracle_gap_delta": args.min_oracle_gap_delta,
        },
        "split_summaries": split_summaries,
        "missing_by_split": missing_by_split,
        "decision": decision,
        "decision_note": (
            "R110 has frequent multi-instance merges and instance-level supervision has a component-safe oracle upper bound."
            if decision.startswith("go")
            else "The component-preserving oracle/gate is insufficient; avoid training another instance model without a sharper hypothesis."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.output_csv, all_rows)
    print(json.dumps({"decision": decision, "val_summary": val}, indent=2))


if __name__ == "__main__":
    main()
