#!/usr/bin/env python3
"""R238-A oracle-best mask editor replaying R237 candidate specs.

This writes isolated original-val masks by replaying the best R237 candidate per
image according to GT-derived diagnostics. It is an upper-bound development
artifact only, not a deployable method and not clean-test-v2 evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, quick_metrics
from audit_r224_pairwise_instance_seam_candidates import pairwise_seam_cut, trim_cut_by_score
from run_r209_component_preserving_feasibility_audit import read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay R237 oracle-best candidates into isolated val masks.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r238_r237_oracle_best_mask_editor")
    parser.add_argument("--split", default="val")
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r237_component_merge_targeted_bounded_val_candidates.csv"))
    parser.add_argument("--max-cuts-per-image", type=int, default=1)
    parser.add_argument("--min-cut-area", type=int, default=2)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r238_r237_oracle_best_mask_editor_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r238_r237_oracle_best_mask_editor_per_image.csv"))
    parser.add_argument("--selected-csv", type=Path, default=Path("outputs/analysis/r238_r237_oracle_best_mask_editor_selected.csv"))
    return parser.parse_args()


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None", "nan"):
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def choose_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if as_float(row.get("safe_nonmerge_gain")) <= 0.5:
            continue
        if as_float(row.get("overerosion_proxy")) > 0.5:
            continue
        grouped.setdefault(str(row.get("image")), []).append(row)
    chosen: dict[str, list[dict[str, Any]]] = {}
    for image, image_rows in grouped.items():
        image_rows.sort(
            key=lambda row: (
                as_float(row.get("delta_component_count_mae")),
                as_float(row.get("delta_merged_pred_components")),
                -as_float(row.get("delta_boundary_iou")),
                as_float(row.get("delta_gap_region_fp_rate")),
                -as_float(row.get("delta_dice")),
                -as_float(row.get("background_connected_by_cut")),
            )
        )
        chosen[image] = image_rows
    return chosen


def reconstruct_cut(row: dict[str, Any], image: np.ndarray, instance: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    labels, n_labels = component_labels(anchor)
    comp_id = int(round(as_float(row.get("pred_component_id"))))
    if comp_id < 1 or comp_id > n_labels:
        return np.zeros_like(anchor, dtype=bool)
    component = labels == comp_id
    gt_a = int(round(as_float(row.get("gt_a"))))
    gt_b = int(round(as_float(row.get("gt_b"))))
    radius = int(round(as_float(row.get("seam_radius"))))
    action_frac = as_float(row.get("action_frac"), 0.25)
    raw_cut = pairwise_seam_cut(component, instance, gt_a, gt_b, radius)
    cut = trim_cut_by_score(image, anchor, raw_cut, action_frac, min_cut_area=2)
    return np.logical_and(cut, anchor)


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row.get(key), default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(records),
        "num_images": len({row["image"] for row in records}),
        "num_edited": int(sum(as_float(row.get("edited")) > 0.5 for row in records)),
        "mean_delta_dice": mean_value(records, "delta_dice"),
        "mean_delta_iou": mean_value(records, "delta_iou"),
        "mean_delta_recall": mean_value(records, "delta_recall"),
        "mean_delta_precision": mean_value(records, "delta_precision"),
        "mean_delta_boundary_iou": mean_value(records, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(records, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(records, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(records, "delta_component_count_mae"),
        "mean_delta_component_delta_mean": mean_value(records, "delta_component_delta_mean"),
        "mean_cut_pixels": mean_value(records, "cut_pixels"),
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


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    chosen = choose_rows(read_rows(args.candidate_csv))
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.split / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.png"):
        old.unlink()

    per_image: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for name in names(args.raw_root, args.dataset, args.split):
        if not anchor_path(args, name).exists():
            continue
        try:
            image, instance, gt, anchor = load_case(args, name)
        except Exception:
            continue
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        current = anchor.copy()
        accepted = []
        for row in chosen.get(name, [])[: args.max_cuts_per_image]:
            cut = reconstruct_cut(row, image, instance, current)
            if int(cut.sum()) < args.min_cut_area:
                continue
            current = np.logical_and(current, ~cut)
            accepted.append(cut)
            out_row = {key: value for key, value in row.items() if not str(key).startswith("anchor_") and not str(key).startswith("candidate_")}
            out_row["replayed_cut_pixels"] = float(cut.sum())
            selected_rows.append(out_row)
        pred_metrics = quick_metrics(current, gt, gt_cache, args.boundary_kernel)
        write_mask(output_dir / name, current)
        record: dict[str, Any] = {
            "image": name,
            "edited": float(bool(accepted)),
            "num_accepted_cuts": float(len(accepted)),
            "cut_pixels": float(sum(int(cut.sum()) for cut in accepted)),
        }
        for metric, value in anchor_metrics.items():
            record[f"anchor_{metric}"] = value
        for metric, value in pred_metrics.items():
            record[f"r238_{metric}"] = value
            record[f"delta_{metric}"] = metric_delta(pred_metrics, anchor_metrics, metric)
        per_image.append(record)
    report = {
        "run_id": "R238-A-r237-oracle-best-mask-editor",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "clean_test_v2_used": False,
        "writes_masks": True,
        "evidence_level": "original_val_oracle_upper_bound_mask_level",
        "candidate_csv": str(args.candidate_csv),
        "output_mask_dir": str(output_dir),
        "num_images": len(per_image),
        "num_selected_source_images": len(chosen),
        "summary": summarize_records(per_image),
        "per_image": per_image,
        "warning": "Oracle-best replay uses GT-derived R237 candidate labels; this is an upper bound, not a deployable method.",
    }
    return report, selected_rows


def main() -> None:
    args = parse_args()
    report, selected = run(args)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.per_image_csv, report["per_image"])
    write_csv(args.selected_csv, selected)
    print(json.dumps({"summary_json": str(args.summary_json), "per_image_csv": str(args.per_image_csv), "selected_csv": str(args.selected_csv), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
