#!/usr/bin/env python3
"""Apply the predeclared R316C Dice/IoU and distance hard gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE = {
    "dice": 0.9027585385931185,
    "iou": 0.826195708970261,
    "hd95_px": 15.864083649434889,
    "assd_px": 4.320394541478605,
}
HD95_GATE = REFERENCE["hd95_px"] * .95
ASSD_GATE = REFERENCE["assd_px"] * .95


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analysis-dir", type=Path, default=Path("outputs/analysis"))
    p.add_argument("--output", type=Path, default=Path("outputs/analysis/r316c_dual_route_summary.json"))
    a = p.parse_args()
    candidates = []
    for route in ("continuous", "gated"):
        for alpha in ("020", "035"):
            path = a.analysis_dir / f"r316c_{route}_alpha{alpha}_original_val_r201.json"
            mean = json.loads(path.read_text(encoding="utf-8"))["mean"]
            gates = {
                "dice_strictly_above_r260": mean["dice"] > REFERENCE["dice"],
                "iou_strictly_above_r260": mean["iou"] > REFERENCE["iou"],
                "hd95_at_least_5pct_better": mean["hd95_px"] <= HD95_GATE,
                "assd_at_least_5pct_better": mean["assd_px"] <= ASSD_GATE,
            }
            distance_gain = .5 * ((REFERENCE["hd95_px"] - mean["hd95_px"]) / REFERENCE["hd95_px"] +
                                  (REFERENCE["assd_px"] - mean["assd_px"]) / REFERENCE["assd_px"])
            candidates.append({
                "route": route, "alpha": float(f"0.{alpha}"), "result": str(path),
                "mean": mean, "delta_vs_r260": {k: mean[k] - v for k, v in REFERENCE.items()},
                "gates": gates, "pass": all(gates.values()),
                "mean_relative_hd95_assd_gain": distance_gain,
            })
    passing = [row for row in candidates if row["pass"]]
    selected = max(passing, key=lambda row: (row["mean_relative_hd95_assd_gain"], row["mean"]["dice"])) if passing else None
    payload = {
        "experiment": "R316C_DUAL_ROUTE_HARD_GATE",
        "reference": REFERENCE,
        "hard_gates": {"dice": f"> {REFERENCE['dice']}", "iou": f"> {REFERENCE['iou']}",
                       "hd95_px": f"<= {HD95_GATE}", "assd_px": f"<= {ASSD_GATE}"},
        "clean_test_used": False, "candidates": candidates,
        "num_passing": len(passing), "selected": selected,
        "decision": "r316c_pass" if selected else "r316c_no_go",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
