#!/usr/bin/env python3
"""R205-F0 train/val safe-edit feasibility audit.

This is an oracle-style feasibility audit on train/val only. It asks whether
R110 has enough nonzero, safe local correction signal to justify another
learned anatomy/gap refiner. It does not train a model and must not be run on
clean-test-v2 for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_r201_unified_eval import build_gt_cache, compute_metrics, read_binary_mask


R186_NEAR_ZERO_EDIT_FRAC = 7.656069e-08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R110 train/val safe-edit feasibility for R205.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--dice-drop-tol", type=float, default=0.001)
    parser.add_argument("--recall-drop-tol", type=float, default=0.002)
    parser.add_argument("--min-safe-image-rate", type=float, default=0.10)
    parser.add_argument("--min-edited-frac", type=float, default=1e-5)
    parser.add_argument("--min-diagnostic-delta", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r205_feasibility_safe_edit_audit.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r205_feasibility_safe_edit_audit.csv"))
    return parser.parse_args()


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask.astype(bool)
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(pil.resize((shape[1], shape[0]), Image.NEAREST)) > 0


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    return float(np.mean(finite)) if finite else None


def mean_records(records: list[dict[str, Any]]) -> dict[str, float | None]:
    metric_keys = sorted({key for row in records for key in row if key not in {"split", "image", "variant", "status"}})
    return {key: mean_or_none([row.get(key) for row in records]) for key in metric_keys}


def metric_delta(candidate: dict[str, Any], anchor: dict[str, Any], key: str) -> float | None:
    if candidate.get(key) is None or anchor.get(key) is None:
        return None
    return float(candidate[key]) - float(anchor[key])


def make_row(
    split: str,
    image: str,
    variant: str,
    pred: np.ndarray,
    anchor: np.ndarray,
    gt: np.ndarray,
    metrics: dict[str, Any],
    anchor_metrics: dict[str, Any],
) -> dict[str, Any]:
    edited = pred != anchor
    removed = anchor & ~pred
    added = pred & ~anchor
    row: dict[str, Any] = {
        "split": split,
        "image": image,
        "variant": variant,
        **metrics,
        "edited_pixels": int(edited.sum()),
        "edited_frac": float(edited.sum() / max(1, edited.size)),
        "removed_pixels": int(removed.sum()),
        "added_pixels": int(added.sum()),
        "removed_frac_of_anchor": float(removed.sum() / max(1, int(anchor.sum()))),
        "added_frac_of_gt": float(added.sum() / max(1, int(gt.sum()))),
    }
    for key in [
        "dice",
        "iou",
        "precision",
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
        delta = metric_delta(metrics, anchor_metrics, key)
        row[f"delta_{key}"] = delta
    return row


def evaluate_split(args: argparse.Namespace, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_dir = args.raw_root / args.dataset / f"{split}_labels"
    anchor_dir = args.ablations_root / args.anchor_exp / args.dataset / split / "masks"
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not anchor_dir.exists():
        raise FileNotFoundError(anchor_dir)

    cache = build_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)
    gt_paths = sorted(gt_dir.glob("*.png"))
    if args.limit > 0:
        gt_paths = gt_paths[: args.limit]

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for gt_path in gt_paths:
        anchor_path = anchor_dir / gt_path.name
        if not anchor_path.exists():
            missing.append(gt_path.name)
            continue
        gt = read_binary_mask(gt_path)
        anchor = resize_like(read_binary_mask(anchor_path), gt.shape)
        gt_cache = cache[gt_path.name]

        anchor_metrics = compute_metrics(
            anchor,
            gt,
            args.boundary_kernel,
            args.gap_kernel,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
        rows.append(make_row(split, gt_path.name, "anchor_r110", anchor, anchor, gt, anchor_metrics, anchor_metrics))

        gap_delete = anchor & ~gt_cache.gap_region
        gap_delete_metrics = compute_metrics(
            gap_delete,
            gt,
            args.boundary_kernel,
            args.gap_kernel,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
        rows.append(make_row(split, gt_path.name, "oracle_delete_gap_fp", gap_delete, anchor, gt, gap_delete_metrics, anchor_metrics))

        all_fp_delete = anchor & gt
        all_fp_metrics = compute_metrics(
            all_fp_delete,
            gt,
            args.boundary_kernel,
            args.gap_kernel,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
        rows.append(make_row(split, gt_path.name, "oracle_delete_all_fp", all_fp_delete, anchor, gt, all_fp_metrics, anchor_metrics))

        fn_add = anchor | gt
        fn_add_metrics = compute_metrics(
            fn_add,
            gt,
            args.boundary_kernel,
            args.gap_kernel,
            args.surface_tol,
            args.surface_tol_extra,
            gt_cache,
        )
        rows.append(make_row(split, gt_path.name, "oracle_add_all_fn", fn_add, anchor, gt, fn_add_metrics, anchor_metrics))

    return rows, {"split": split, "num_labels": len(gt_paths), "num_missing": len(missing), "missing": missing[:50]}


def summarize_split(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    summary: dict[str, Any] = {}
    anchor_mean = mean_records(by_variant.get("anchor_r110", []))
    summary["anchor_r110"] = {"mean": anchor_mean, "num_images": len(by_variant.get("anchor_r110", []))}

    for variant, records in sorted(by_variant.items()):
        if variant == "anchor_r110":
            continue
        mean = mean_records(records)
        # Keep this explicit instead of clever: a safe image must actually edit,
        # preserve overlap/recall, and not worsen component count.
        safe_records = [
            row
            for row in records
            if float(row.get("edited_frac") or 0.0) >= args.min_edited_frac
            and float(row.get("delta_dice") or 0.0) >= -args.dice_drop_tol
            and float(row.get("delta_recall") or 0.0) >= -args.recall_drop_tol
            and float(row.get("delta_component_count_mae") or 0.0) <= 0.0
        ]
        n = len(records)
        safe_rate = float(len(safe_records) / max(1, n))
        diagnostic_improved = any(
            (mean.get(f"delta_{key}") is not None and float(mean[f"delta_{key}"]) < -args.min_diagnostic_delta)
            for key in ["gap_region_fp_rate", "component_merge_rate", "component_count_mae", "hd95_px", "assd_px"]
        ) or any(
            (mean.get(f"delta_{key}") is not None and float(mean[f"delta_{key}"]) > args.min_diagnostic_delta)
            for key in ["boundary_iou", "boundary_f1", "surface_dice_2px", "surface_dice_5px"]
        )
        mean_component_safe = bool(
            (mean.get("delta_component_count_mae") is not None)
            and float(mean["delta_component_count_mae"]) <= 0.0
            and (mean.get("delta_component_merge_rate") is not None)
            and float(mean["delta_component_merge_rate"]) <= 0.0
        )
        passes_gate = bool(
            variant == "oracle_delete_gap_fp"
            and safe_rate >= args.min_safe_image_rate
            and (mean.get("edited_frac") or 0.0) >= args.min_edited_frac
            and (mean.get("edited_frac") or 0.0) > 10.0 * R186_NEAR_ZERO_EDIT_FRAC
            and (mean.get("delta_dice") or 0.0) >= -args.dice_drop_tol
            and (mean.get("delta_recall") or 0.0) >= -args.recall_drop_tol
            and mean_component_safe
            and diagnostic_improved
        )
        summary[variant] = {
            "mean": mean,
            "num_images": n,
            "safe_image_count": len(safe_records),
            "safe_image_rate": safe_rate,
            "diagnostic_improved": diagnostic_improved,
            "mean_component_safe": mean_component_safe,
            "passes_feasibility_gate": passes_gate,
        }
    return summary


def main() -> None:
    args = parse_args()
    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    all_rows: list[dict[str, Any]] = []
    split_status: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in splits:
        rows, status = evaluate_split(args, split)
        all_rows.extend(rows)
        split_status.append(status)
        split_summaries[split] = summarize_split(rows, args)

    val_summary = split_summaries.get("val", {})
    val_gap = val_summary.get("oracle_delete_gap_fp", {})
    decision = "go_train_learned_refiner" if val_gap.get("passes_feasibility_gate") else "no_go_do_not_train_r205"

    output = {
        "run_id": "R205-F0",
        "purpose": "train/val-only oracle safe-edit feasibility audit before any new anatomy-aware training",
        "clean_test_v2_used": False,
        "dataset": args.dataset,
        "anchor_exp": args.anchor_exp,
        "metric_protocol": {
            "source": "R201-compatible compute_metrics",
            "boundary_kernel": args.boundary_kernel,
            "gap_kernel": args.gap_kernel,
            "surface_tol": args.surface_tol,
            "surface_tol_extra": args.surface_tol_extra,
        },
        "gate": {
            "dice_drop_tol": args.dice_drop_tol,
            "recall_drop_tol": args.recall_drop_tol,
            "min_safe_image_rate": args.min_safe_image_rate,
            "min_edited_frac": args.min_edited_frac,
            "r186_near_zero_edit_frac": R186_NEAR_ZERO_EDIT_FRAC,
            "min_diagnostic_delta": args.min_diagnostic_delta,
            "mean_component_safe": "delta_component_count_mae <= 0 and delta_component_merge_rate <= 0",
        },
        "split_status": split_status,
        "split_summaries": split_summaries,
        "decision": decision,
        "decision_note": (
            "If no_go, R205 should not train another anatomy-aware refiner; move to baselines/data/new model family."
            if decision.startswith("no_go")
            else "Val oracle gap-FP deletion has enough safe nonzero signal to justify a learned refiner smoke."
        ),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    if all_rows:
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in all_rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    print(json.dumps({"decision": decision, "val_oracle_delete_gap_fp": val_gap}, indent=2))


if __name__ == "__main__":
    main()
