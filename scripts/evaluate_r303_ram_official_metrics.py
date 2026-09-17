#!/usr/bin/env python3
"""R303: RAM-W600 paper-aligned frozen test audit.

Replaces the project-only ``overlap_dice``/``overlap_nsd_2px`` summaries with
the metrics used by the RAM-W600 benchmark (Table 4/Table 13): DSC, NSD@2px,
VOE, symmetric MSD (pixels), and MSD failure rate.  The overlap report is
computed on the logical overlap map (at least two active bone channels), and
also includes the official pair-intersection aggregate for transparency.

This script is diagnostic only: it does not train, select a checkpoint, or
search a threshold using test data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import (
    NnUNetMultiLabelPrior, WristDataset, seed_everything, top_training_pairs,
)
from train_r300_ram_ceb_boundary_gate import apply_gate, collect, examples
from sklearn.ensemble import RandomForestClassifier
from diagnose_r301_ram_ceb_signature import val_examples


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask ^ binary_erosion(mask)


def _official_one(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> dict[str, float | None]:
    """RAM/MONAI-style binary DSC, NSD, VOE, symmetric MSD and RAVD."""
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    p = int(pred.sum()); t = int(target.sum()); inter = int((pred & target).sum())
    dsc = 1.0 if p == 0 and t == 0 else (0.0 if p + t == 0 else 2.0 * inter / (p + t))
    union = int((pred | target).sum())
    voe = 0.0 if union == 0 else 1.0 - inter / union
    ravd = 0.0 if t == 0 else abs(p - t) / float(t)
    if p == 0 and t == 0:
        return {"dsc": dsc, "nsd_2px": 1.0, "voe": voe, "msd_px": 0.0,
                "ravd": ravd, "msd_failed": 0.0}
    if p == 0 or t == 0:
        return {"dsc": dsc, "nsd_2px": 0.0, "voe": voe, "msd_px": None,
                "ravd": ravd, "msd_failed": 1.0}
    ps = _surface(pred); ts = _surface(target)
    dt_t = distance_transform_edt(~ts); dt_p = distance_transform_edt(~ps)
    d_pt = dt_t[ps]; d_tp = dt_p[ts]
    nsd = float(((d_pt <= tolerance).sum() + (d_tp <= tolerance).sum()) /
                max(int(ps.sum()) + int(ts.sum()), 1))
    msd = float((d_pt.mean() + d_tp.mean()) / 2.0)
    return {"dsc": float(dsc), "nsd_2px": nsd, "voe": float(voe),
            "msd_px": msd, "ravd": float(ravd), "msd_failed": 0.0}


def _aggregate(rows: list[dict[str, float | None]]) -> dict[str, float | int]:
    if not rows:
        return {"n": 0, "dsc": None, "nsd_2px": None, "voe": None,
                "msd_px": None, "msd_fail_rate": None, "ravd": None}
    def mean(key: str):
        vals = [float(r[key]) for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None
    return {
        "n": len(rows), "dsc": mean("dsc"), "nsd_2px": mean("nsd_2px"),
        "voe": mean("voe"), "msd_px": mean("msd_px"),
        "msd_fail_rate": float(np.mean([float(r["msd_failed"]) for r in rows])),
        "ravd": mean("ravd"),
    }


def official_metrics(data: dict, pairs: list[tuple[int, int]]) -> dict:
    channel_rows = []
    overlap_rows = []
    pair_rows = []
    for item in data.values():
        p, t = item["pred"], item["target"]
        for k in range(14):
            channel_rows.append(_official_one(p[k], t[k]))
        po, to = p.sum(0) >= 2, t.sum(0) >= 2
        overlap_rows.append(_official_one(po, to))
        for i, j in pairs:
            pp, tt = p[i] & p[j], t[i] & t[j]
            # RAM Table 13/14 only evaluates actual overlapping regions; do not
            # dilute the result with pair/case combinations with empty GT.
            if tt.any():
                pair_rows.append(_official_one(pp, tt))
    return {
        "overall_instance_metrics": _aggregate(channel_rows),
        "overlap_region_metrics": _aggregate(overlap_rows),
        "overlap_pair_intersection_metrics": _aggregate(pair_rows),
        "overlap_pair_count": len(pair_rows),
    }


def main() -> None:
    out = Path("outputs/ram_w600/r303_official_metrics_test_audit")
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(303)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path("outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth")
    base = NnUNetMultiLabelPrior().to(device)
    base.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])
    base.eval()
    root = Path(r"G:\gutou\RAM-W600")
    tr = DataLoader(WristDataset(root, "train", 384, False), batch_size=8, shuffle=False, num_workers=0)
    te = DataLoader(WristDataset(root, "test", 384, False), batch_size=8, shuffle=False, num_workers=0)
    train = collect(base, tr, device); test = collect(base, te, device)
    X, y = examples(train); Xt, yt = val_examples(test)
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3,
                                 class_weight="balanced", random_state=300, n_jobs=-1).fit(X, y)
    gated = apply_gate(test, clf)
    pairs = top_training_pairs(root, 15)
    bm = official_metrics(test, pairs); gm = official_metrics(gated, pairs)
    result = {
        "experiment": "R303", "device": str(device), "test_audit_only": True,
        "test_used_for_selection": False, "checkpoint": str(ckpt),
        "threshold": 0.5, "pair_definition": "top 15 train overlap pairs; empty GT pair/cases excluded",
        "ram_official_metrics": ["DSC", "NSD@2px", "VOE", "symmetric MSD (pixels)", "MSD Fail Rate", "RAVD"],
        "baseline_test": bm, "hard_gate_test": gm,
        "delta_test": {
            section: {k: (gm[section][k] - bm[section][k])
                      if isinstance(bm[section].get(k), (int, float)) and isinstance(gm[section].get(k), (int, float))
                      else None for k in ("dsc", "nsd_2px", "voe", "msd_px", "msd_fail_rate", "ravd")}
            for section in ("overall_instance_metrics", "overlap_region_metrics", "overlap_pair_intersection_metrics")
        },
        "classifier_test_diagnostic": {"candidates": int(len(yt))},
        "warning": "R302 hard gate is an audit-only postprocessor; no promotion or tuning is based on test results.",
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({"experiment": "R303", "protocol": "RAM-W600 Table 4/Table 13 metrics", "test_used_for_selection": False}, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
