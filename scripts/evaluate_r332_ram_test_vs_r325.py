#!/usr/bin/env python3
"""Locked RAM test audit: R325 nnU-Net versus validation-selected R332.

No threshold, checkpoint, or correction-depth search is performed on test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset, sha256
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, seed_all
from train_r332_ram_joint_iterative_refinement import collect_steps, flatten


def numeric_delta(method: dict, baseline: dict) -> dict:
    result = {}
    for section in ("overall_instance_metrics", "overlap_region_metrics",
                    "overlap_pair_intersection_metrics"):
        result[section] = {}
        for key in ("dsc", "nsd_2px", "voe", "msd_px", "msd_fail_rate", "ravd"):
            left, right = method[section].get(key), baseline[section].get(key)
            result[section][key] = left - right if isinstance(left, (int, float)) and isinstance(right, (int, float)) else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--improved-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correction-steps", type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    seed_all(3321)
    if not torch.cuda.is_available():
        raise RuntimeError("R332 test audit requires CUDA")
    device = torch.device("cuda")
    dataset = NativeWristDataset(args.dataset_root, "test", augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    pairs = top_training_pairs(args.dataset_root, 15)

    baseline_payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline = NnUNetMultiLabelPrior().to(device)
    baseline.load_state_dict(baseline_payload["model"], strict=True)
    baseline_data = collect_steps(baseline, InstanceCompletionRefiner().to(device), loader, device, 0)["step0"]
    baseline_metrics = official_metrics(baseline_data, pairs)
    del baseline, baseline_data, baseline_payload
    torch.cuda.empty_cache()

    improved_payload = torch.load(args.improved_checkpoint, map_location=device, weights_only=False)
    improved = NnUNetMultiLabelPrior().to(device)
    improved.load_state_dict(improved_payload["model"], strict=True)
    refiner = InstanceCompletionRefiner().to(device)
    refiner.load_state_dict(improved_payload["refiner"], strict=True)
    step_data = collect_steps(improved, refiner, loader, device, args.correction_steps)
    improved_steps = {name: official_metrics(data, pairs) for name, data in step_data.items()}
    selected = improved_steps[f"step{args.correction_steps}"]

    result = {
        "experiment": "R332_RAM_LOCKED_TEST_VS_R325",
        "split": "test", "n_images": len(dataset), "test_used": True,
        "test_usage": "one locked final evaluation; no tuning or model selection",
        "threshold": 0.5, "threshold_search": False,
        "correction_steps": args.correction_steps,
        "correction_steps_selected_on": "validation",
        "spatial_preprocessing": "native pixels; right/bottom padding only; no resize",
        "pair_definition": "top 15 overlap pairs determined from train only",
        "baseline": {"name": "R325 plain native nnU-Net", "checkpoint": str(args.baseline_checkpoint),
                     "sha256": sha256(args.baseline_checkpoint), "official_metrics": baseline_metrics,
                     "flat_metrics": flatten(baseline_metrics)},
        "improved": {"name": f"R332 joint iterative, step{args.correction_steps}",
                     "checkpoint": str(args.improved_checkpoint), "sha256": sha256(args.improved_checkpoint),
                     "best_epoch_selected_on_validation": int(improved_payload["epoch"]),
                     "official_metrics": selected, "flat_metrics": flatten(selected)},
        "improved_diagnostic_steps": {
            name: {"official_metrics": metrics, "flat_metrics": flatten(metrics)}
            for name, metrics in improved_steps.items()
        },
        "delta_improved_minus_baseline": numeric_delta(selected, baseline_metrics),
        "flat_delta_improved_minus_baseline": {
            key: flatten(selected)[key] - flatten(baseline_metrics)[key]
            for key in flatten(selected)
        },
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "baseline": result["baseline"]["flat_metrics"],
                      "improved": result["improved"]["flat_metrics"],
                      "delta": result["flat_delta_improved_minus_baseline"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
