#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--native-r201", type=Path, required=True)
    p.add_argument("--active-r201", type=Path, required=True)
    p.add_argument("--close-diagnostics", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    native = json.loads(args.native_r201.read_text(encoding="utf-8"))["mean"]
    active = json.loads(args.active_r201.read_text(encoding="utf-8"))["mean"]
    close = json.loads(args.close_diagnostics.read_text(encoding="utf-8"))
    delta = {key: active[key] - native[key] for key in native}
    pair = close["pair_summary"]["1-4"]
    image = close["image_summary"]
    checks = {
        "close_gap_fp_improves": pair["delta_gap_fp_rate"] < 0,
        "pair_merge_nonworse": pair["delta_pair_merge_rate"] <= 0,
        "precision_stable": delta["precision"] >= -0.002,
        "recall_stable": delta["recall"] >= -0.005,
        "dice_stable": delta["dice"] >= -0.005,
        "boundary_iou_stable": delta["boundary_iou"] >= -0.002,
        "false_split_stable": image["r255_false_split_rate"] - image["baseline_false_split_rate"] <= 0.002,
        "small_instance_recall_stable": image["r255_small_instance_recall_le512"] - image["baseline_small_instance_recall_le512"] >= -0.01,
    }
    payload = {
        "run_id": "R255_GATE_B",
        "decision": "go_to_gate_c" if all(checks.values()) else "no_go",
        "checks": checks,
        "r201_delta": delta,
        "close_1_4": pair,
        "image_summary": image,
        "clean_test_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
