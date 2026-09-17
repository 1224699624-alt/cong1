#!/usr/bin/env python3
"""Combine the fixed 1/2/3-step R332 validation arms without touching test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    arms = {}
    for steps in (1, 2, 3):
        path = args.root / f"steps_{steps}" / "result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        final = result["flat_metrics_by_step"][f"step{steps}"]
        baseline = result["r325_baseline_flat"]
        joint_step0 = result["flat_metrics_by_step"]["step0"]
        gate = result["best_row"]["passes_checkpoint_gate"]
        score = (int(gate), final["overlap_nsd_2px"], final["overlap_dsc"],
                 -final["overlap_msd_px"], final["overall_dsc"], final["overall_iou"])
        arms[str(steps)] = {
            "best_epoch": result["best_epoch"], "passes_gate": gate,
            "progression": result["progression"], "r325_baseline": baseline,
            "joint_step0": joint_step0, "final": final,
            "delta_vs_r325": {key: final[key] - baseline[key] for key in final},
            "delta_refinement_vs_joint_step0": {key: final[key] - joint_step0[key] for key in final},
            "selection_score": score,
        }
    winner = max(arms, key=lambda key: tuple(arms[key]["selection_score"]))
    output = {
        "experiment": "R332_RAM_JOINT_ITERATIVE_REFINEMENT_MATRIX",
        "split": "validation", "test_used": False,
        "threshold": 0.5, "threshold_search": False,
        "arms": arms, "selected_correction_steps": int(winner),
        "selection_rule": "macro DSC/IoU gate, then overlap NSD, DSC, MSD, overall DSC/IoU",
    }
    (args.root / "result.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
