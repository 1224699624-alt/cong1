"""Apply the predeclared R283 original-val gate without threshold search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["mean"]
    result = json.loads(args.result.read_text(encoding="utf-8"))["mean"]
    checks = {
        "gap_fp_not_worse": result["gap_region_fp_rate"] <= baseline["gap_region_fp_rate"],
        "merge_not_worse": result["component_merge_rate"] <= baseline["component_merge_rate"],
        "recall_within_0p004": result["recall"] >= baseline["recall"] - 0.004,
        "dice_within_0p003": result["dice"] >= baseline["dice"] - 0.003,
        "boundary_iou_within_0p002": result["boundary_iou"] >= baseline["boundary_iou"] - 0.002,
        "surface2_within_0p002": result["surface_dice_2px"] >= baseline["surface_dice_2px"] - 0.002,
    }
    report = {
        "decision": "pass" if all(checks.values()) else "no_go_after_spatial_confidence_gating",
        "checks": checks,
        "baseline": baseline,
        "result": result,
        "delta": {key: result[key] - baseline[key] for key in result},
        "clean_test_used": False,
        "threshold_search_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
