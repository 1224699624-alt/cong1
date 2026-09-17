#!/usr/bin/env python3
"""R184 guarded local arbitrator anchored to true R110 train/val masks.

This script deliberately refuses to apply to clean-test-v2 unless the original
validation split shows a non-destructive local edit over the true R110 anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_anchor_pixel_residual import write_mask
from train_r177_separation_arbitrator import (
    MLP,
    add_structure_metrics,
    eval_cache,
    eval_cached,
    mean_records,
    parse_nums,
    read_manifest,
    sample_training,
    train_model,
)


TARGET = 0.9317660066557425
R110_CLEAN_DICE = 0.9177231563529792
R110_CLEAN_BOUNDARY_IOU = 0.25189601044085763


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R184 true-R110 guarded arbitrator.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--manifest-csv", default="outputs/analysis/r184_r110_bridge_risk_manifest/r177_bridge_risk_manifest.csv")
    parser.add_argument("--output-exp", default="r184_r110_guarded_arbitrator")
    parser.add_argument("--checkpoint", default="outputs/r184_guarded_arbitrator/r184_r110_guarded_arbitrator.pt")
    parser.add_argument("--metrics-json", default="outputs/analysis/r184_r110_guarded_arbitrator_clean_test_v2_metrics.json")
    parser.add_argument("--val-summary-json", default="outputs/analysis/r184_r110_guarded_arbitrator_val_summary.json")
    parser.add_argument("--result-json", default="outputs/analysis/r184_r110_guarded_arbitrator_result_summary.json")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--samples-per-risk-image", type=int, default=4096)
    parser.add_argument("--samples-per-normal-image", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--pos-weight-scale", type=float, default=0.75)
    parser.add_argument("--risk-radius", type=int, default=7)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--zone-radii", default="2,3,5")
    parser.add_argument("--prob-thresholds", default="0.50,0.55,0.60")
    parser.add_argument("--edit-margins", default="0.20,0.25,0.30,0.35")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--val-dice-drop-tol", type=float, default=0.0005)
    parser.add_argument("--val-component-tol", type=float, default=0.25)
    parser.add_argument("--max-edited-frac", type=float, default=0.006)
    parser.add_argument("--allow-clean-test-on-gate-fail", action="store_true")
    parser.add_argument("--seed", type=int, default=202607184)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def anchor_mean(cache: list[dict[str, object]], boundary_kernel: int, radius: int) -> dict[str, float]:
    records = []
    for item in cache:
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        rec = compute_metrics(anchor, gt, boundary_kernel)
        rec = add_structure_metrics(rec, anchor, gt, radius)
        rec["edited_frac"] = 0.0
        rec["image"] = str(item["name"])
        records.append(rec)
    return mean_records(records)


def eval_cached_guarded(
    cache: list[dict[str, object]],
    threshold: float,
    margin: float,
    zone_radius: int,
    boundary_radius: int,
    boundary_kernel: int,
    max_edited_frac: float,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    from train_r177_separation_arbitrator import risk_zone

    records = []
    for item in tqdm(cache, desc=f"eval/r184/t{threshold:.2f}/m{margin:.2f}/z{zone_radius}", leave=False):
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        prob = item["prob"]  # type: ignore[assignment]
        zone_key = f"zone_{zone_radius}"
        if zone_key not in item:
            item[zone_key] = risk_zone(anchor, zone_radius, boundary_radius)
        zone = item[zone_key]  # type: ignore[assignment]

        add = zone & (prob >= threshold + margin) & (~anchor)
        delete = zone & (prob <= threshold - margin) & anchor
        edit = add | delete
        max_pixels = int(round(float(anchor.size) * max_edited_frac))
        if max_pixels > 0 and int(edit.sum()) > max_pixels:
            confidence = np.zeros_like(prob, dtype=np.float32)
            confidence[add] = prob[add] - (threshold + margin)
            confidence[delete] = (threshold - margin) - prob[delete]
            keep_idx = np.argpartition(confidence.ravel(), -max_pixels)[-max_pixels:]
            keep = np.zeros(anchor.size, dtype=bool)
            keep[keep_idx] = confidence.ravel()[keep_idx] > 0
            edit = edit & keep.reshape(anchor.shape)
            add = add & edit
            delete = delete & edit

        pred = anchor.copy()
        pred[add] = True
        pred[delete] = False
        rec = compute_metrics(pred, gt, boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt, zone_radius)
        rec["edited_frac"] = float(edit.sum() / max(1, anchor.size))
        rec["image"] = str(item["name"])
        records.append(rec)
        if write_dir is not None:
            write_mask(write_dir / str(item["name"]), pred)
    return mean_records(records), records


def passes_val_gate(mean: dict[str, float], anchor: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        mean["dice"] >= anchor["dice"] - args.val_dice_drop_tol
        and mean["component_count_error"] <= anchor["component_count_error"] + args.val_component_tol
        and mean["false_bridge_flag"] <= anchor["false_bridge_flag"] + 1e-9
        and mean["boundary_iou"] >= anchor["boundary_iou"] - 1e-9
    )


def main() -> None:
    args = parse_args()
    manifest = read_manifest(Path(args.manifest_csv))
    x, y = sample_training(args, manifest)
    model = train_model(args, x, y)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "in_dim": x.shape[1], "args": vars(args)}, args.checkpoint)

    val_cache = eval_cache(model, args, args.train_dataset, Path(args.train_raw_root), args.tune_split, args.anchor_exp, args.max_val_images)
    val_anchor = anchor_mean(val_cache, args.boundary_kernel, args.risk_radius)
    best = None
    val_grid = []
    for zone_radius in parse_nums(args.zone_radii, int):
        for threshold in parse_nums(args.prob_thresholds, float):
            for margin in parse_nums(args.edit_margins, float):
                mean, _ = eval_cached_guarded(
                    val_cache,
                    threshold,
                    margin,
                    zone_radius,
                    args.boundary_radius,
                    args.boundary_kernel,
                    args.max_edited_frac,
                )
                gated = passes_val_gate(mean, val_anchor, args)
                item = {"zone_radius": zone_radius, "threshold": threshold, "margin": margin, "mean": mean, "passes_gate": gated}
                val_grid.append(item)
                if not gated:
                    continue
                score = (
                    mean["dice"] - val_anchor["dice"],
                    mean["boundary_iou"] - val_anchor["boundary_iou"],
                    -(mean["false_bridge_flag"] - val_anchor["false_bridge_flag"]),
                    -(mean["component_count_error"] - val_anchor["component_count_error"]),
                )
                if best is None:
                    best = item | {"score": score}
                elif score > best["score"]:
                    best = item | {"score": score}

    gate_pass = best is not None
    val_summary = {
        "run_id": "R184",
        "anchor_exp": args.anchor_exp,
        "apply_anchor_exp": args.apply_anchor_exp,
        "val_anchor": val_anchor,
        "gate_pass": gate_pass,
        "gate_rules": {
            "val_dice_drop_tol": args.val_dice_drop_tol,
            "val_component_tol": args.val_component_tol,
            "max_edited_frac": args.max_edited_frac,
        },
        "best": best,
        "grid": val_grid,
    }
    Path(args.val_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_summary_json).write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

    result = {
        "run_id": "R184",
        "status": "val_gate_failed",
        "val_summary_path": args.val_summary_json,
        "metrics_path": args.metrics_json,
        "target_dice": TARGET,
        "r110_clean_dice": R110_CLEAN_DICE,
        "r110_clean_boundary_iou": R110_CLEAN_BOUNDARY_IOU,
    }
    if not gate_pass and not args.allow_clean_test_on_gate_fail:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    assert best is not None
    apply_cache = eval_cache(model, args, args.apply_dataset, Path(args.apply_raw_root), args.apply_split, args.apply_anchor_exp, args.max_apply_images)
    out_dir = Path(args.ablations_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = eval_cached_guarded(
        apply_cache,
        float(best["threshold"]),
        float(best["margin"]),
        int(best["zone_radius"]),
        args.boundary_radius,
        args.boundary_kernel,
        args.max_edited_frac,
        out_dir,
    )
    metrics = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "best": best,
        "output_exp": args.output_exp,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    dice = float(mean.get("dice", 0.0))
    result.update(
        {
            "status": "target_met" if dice >= TARGET else ("new_best_below_target" if dice > R110_CLEAN_DICE else "below_best"),
            "mean": mean,
            "target_margin": dice - TARGET,
            "dice_delta_vs_r110": dice - R110_CLEAN_DICE,
            "boundary_iou_delta_vs_r110": float(mean.get("boundary_iou", 0.0)) - R110_CLEAN_BOUNDARY_IOU,
            "val_best": best,
        }
    )
    Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
