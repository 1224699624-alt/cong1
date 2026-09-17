#!/usr/bin/env python3
"""Select an R317 original-val checkpoint with region gates before distance gains."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

R260 = {"dice": 0.9027585385931185, "iou": 0.826195708970261,
        "hd95_px": 15.864083649434889, "assd_px": 4.320394541478605}
LIMITS = {"hd95_px": R260["hd95_px"] * .95, "assd_px": R260["assd_px"] * .95}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-glob", default="outputs/analysis/r317_continuous_*_original_val_r201.json")
    p.add_argument("--trainer-dir", type=Path, required=True)
    p.add_argument("--r316c-reference", type=Path,
                   default=Path("outputs/analysis/r316c_continuous_alpha035_original_val_r201.json"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/analysis/r317_continuous_checkpoint_selection.json"))
    return p.parse_args()


def checkpoint_for(metrics:Path, trainer_dir:Path)->tuple[str,Path]:
    match = re.search(r"_epoch(\d+)_original_val", metrics.name)
    if match:
        label = f"epoch{int(match.group(1)):03d}"
        return label, trainer_dir / f"checkpoint_epoch_{int(match.group(1)):03d}.pth"
    if "ema_best" in metrics.name:
        return "ema_best", trainer_dir / "checkpoint_best.pth"
    raise RuntimeError(f"Unrecognized R317 metric filename: {metrics}")


def main():
    a = parse_args()
    paths = [Path(x) for x in sorted(glob.glob(a.metrics_glob))]
    if len(paths) < 2:
        raise RuntimeError(f"Expected multiple R317 checkpoint metrics, got {paths}")
    reference316 = json.loads(a.r316c_reference.read_text())["mean"]
    candidates = []
    for path in paths:
        if "clean-test" in str(path).lower() or "articular" in str(path).lower():
            raise RuntimeError(f"Forbidden evaluation path: {path}")
        payload = json.loads(path.read_text())
        if payload.get("protocol") != "R201 metric definitions on original-val" or payload.get("num_images") != 96:
            raise RuntimeError(f"Invalid R201 payload: {path}")
        mean = payload["mean"]
        label, checkpoint = checkpoint_for(path, a.trainer_dir)
        if not checkpoint.is_file():
            raise RuntimeError(f"Checkpoint missing for {path}: {checkpoint}")
        gates = {
            "dice_strictly_above_r260": mean["dice"] > R260["dice"],
            "iou_strictly_above_r260": mean["iou"] > R260["iou"],
            "hd95_at_least_5pct_better": mean["hd95_px"] <= LIMITS["hd95_px"],
            "assd_at_least_5pct_better": mean["assd_px"] <= LIMITS["assd_px"],
        }
        distance_gain = .5 * ((R260["hd95_px"] - mean["hd95_px"]) / R260["hd95_px"] +
                              (R260["assd_px"] - mean["assd_px"]) / R260["assd_px"])
        candidates.append({"label": label, "checkpoint": str(checkpoint), "metrics": str(path),
                           "mean": mean, "gates": gates, "pass": all(gates.values()),
                           "mean_relative_hd95_assd_gain": distance_gain,
                           "delta_vs_r260": {k: mean[k] - R260[k] for k in R260},
                           "delta_vs_r316c_selected": {k: mean[k] - reference316[k] for k in R260}})
    passing = [x for x in candidates if x["pass"]]
    selected = max(passing, key=lambda x:(x["mean_relative_hd95_assd_gain"], x["mean"]["dice"])) if passing else None
    out = {"experiment": "R317_CONTINUOUS_PRIOR_MATURITY_AND_R201_CHECKPOINT_SELECTION",
           "scope": "TSRS_RSNA-Epiphysis full 875 train / 96 original-val",
           "clean_test_used": False, "reference_r260": R260, "distance_limits": LIMITS,
           "checkpoint_selection": "Dice/IoU hard gates, then mean relative HD95/ASSD gain",
           "candidates": candidates, "num_passing": len(passing), "selected": selected,
           "decision": "r317_pass" if selected else "r317_no_checkpoint_passed"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
