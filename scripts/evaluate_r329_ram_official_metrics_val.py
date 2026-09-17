#!/usr/bin/env python3
"""Official RAM-W600 metric audit for R329 validation only.

No training, threshold search, or test-set loading is performed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import (
    InstanceCompletionRefiner, evaluate, per_instance_features, seed_all,
)
from train_r329_ram_instance_prior_vs_generic_adapter import (
    GenericMultiChannelAdapter, closest_generic_width,
)


def _official_one(pred, target, tolerance=2):
    pred = np.asarray(pred, dtype=bool); target = np.asarray(target, dtype=bool)
    p = int(pred.sum()); t = int(target.sum()); inter = int((pred & target).sum())
    dsc = 1.0 if p == 0 and t == 0 else (0.0 if p + t == 0 else 2.0 * inter / (p + t))
    union = int((pred | target).sum())
    voe = 0.0 if union == 0 else 1.0 - inter / union
    ravd = 0.0 if t == 0 else abs(p - t) / float(t)
    if p == 0 and t == 0:
        return {"dsc": dsc, "nsd_2px": 1.0, "voe": voe, "msd_px": 0.0, "ravd": ravd, "msd_failed": 0.0}
    if p == 0 or t == 0:
        return {"dsc": dsc, "nsd_2px": 0.0, "voe": voe, "msd_px": None, "ravd": ravd, "msd_failed": 1.0}
    from scipy.ndimage import binary_erosion, distance_transform_edt
    ps = pred ^ binary_erosion(pred); ts = target ^ binary_erosion(target)
    dt_t = distance_transform_edt(~ts); dt_p = distance_transform_edt(~ps)
    d_pt = dt_t[ps]; d_tp = dt_p[ts]
    nsd = float(((d_pt <= tolerance).sum() + (d_tp <= tolerance).sum()) / max(int(ps.sum()) + int(ts.sum()), 1))
    msd = float((d_pt.mean() + d_tp.mean()) / 2.0)
    return {"dsc": float(dsc), "nsd_2px": nsd, "voe": float(voe), "msd_px": msd, "ravd": float(ravd), "msd_failed": 0.0}


def _aggregate(rows):
    def mean(key):
        vals = [float(r[key]) for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None
    return {"n": len(rows), "dsc": mean("dsc"), "nsd_2px": mean("nsd_2px"),
            "voe": mean("voe"), "msd_px": mean("msd_px"),
            "msd_fail_rate": float(np.mean([float(r["msd_failed"]) for r in rows])) if rows else None,
            "ravd": mean("ravd")}


def official_metrics(data, pairs):
    channel_rows=[]; overlap_rows=[]; pair_rows=[]
    for item in data.values():
        p, t = item["pred"], item["target"]
        for k in range(14): channel_rows.append(_official_one(p[k], t[k]))
        po, to = p.sum(0) >= 2, t.sum(0) >= 2
        overlap_rows.append(_official_one(po, to))
        for i, j in pairs:
            pp, tt = p[i] & p[j], t[i] & t[j]
            if tt.any(): pair_rows.append(_official_one(pp, tt))
    return {"overall_instance_metrics": _aggregate(channel_rows),
            "overlap_region_metrics": _aggregate(overlap_rows),
            "overlap_pair_intersection_metrics": _aggregate(pair_rows),
            "overlap_pair_count": len(pair_rows)}


@torch.inference_mode()
def collect_instance(baseline, refiner, loader, device):
    baseline.eval(); refiner.eval(); out = {}
    for batch in loader:
        image = batch["image"].to(device)
        target = (batch["mask"].to(device) > 0.5).detach().cpu().numpy()[0]
        logits, _ = baseline(image, False)
        probability = torch.sigmoid(logits)
        refined = logits.clone()
        for start in range(0, 14, 3):
            indices = torch.arange(start, min(start + 3, 14), device=device)
            source = probability[:, indices]
            feature = per_instance_features(image, probability, indices, source)
            corrected, _ = refiner(feature, logits[0, indices].unsqueeze(1))
            refined[0, indices] = corrected[:, 0]
        pred = (torch.sigmoid(refined) >= 0.5).cpu().numpy()[0]
        out[str(batch["case"][0])] = {"pred": pred, "target": target}
    return out


@torch.inference_mode()
def collect_generic(baseline, adapter, loader, device):
    baseline.eval(); adapter.eval(); out = {}
    for batch in loader:
        image = batch["image"].to(device)
        target = (batch["mask"].to(device) > 0.5).detach().cpu().numpy()[0]
        base_logits, _ = baseline(image, False)
        logits, _ = adapter(image, base_logits)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[0]
        out[str(batch["case"][0])] = {"pred": pred, "target": target}
    return out


@torch.inference_mode()
def collect_baseline(baseline, loader, device):
    baseline.eval(); out = {}
    for batch in loader:
        image = batch["image"].to(device)
        target = (batch["mask"].to(device) > 0.5).detach().cpu().numpy()[0]
        logits, _ = baseline(image, False)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[0]
        out[str(batch["case"][0])] = {"pred": pred, "target": target}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--baseline-checkpoint", type=Path, required=True)
    ap.add_argument("--instance-checkpoint", type=Path, required=True)
    ap.add_argument("--generic-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    seed_all(3291)
    if not torch.cuda.is_available(): raise RuntimeError("R329 audit requires CUDA")
    device = torch.device("cuda")
    ds = NativeWristDataset(args.dataset_root, "val", augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline = NnUNetMultiLabelPrior().to(device); baseline.load_state_dict(payload["model"], strict=True)
    width, _, _ = closest_generic_width(108914)
    inst = InstanceCompletionRefiner().to(device)
    inst.load_state_dict(torch.load(args.instance_checkpoint, map_location=device, weights_only=False)["refiner"], strict=True)
    generic = GenericMultiChannelAdapter(width).to(device)
    generic.load_state_dict(torch.load(args.generic_checkpoint, map_location=device, weights_only=False)["adapter"], strict=True)
    pairs = top_training_pairs(args.dataset_root, 15)
    datasets = {
        "baseline": collect_baseline(baseline, loader, device),
        "generic_adapter": collect_generic(baseline, generic, loader, device),
        "instance_prior": collect_instance(baseline, inst, loader, device),
    }
    result = {
        "experiment": "R329_RAM_OFFICIAL_METRICS_VALIDATION",
        "split": "validation", "n_images": len(ds), "test_used": False,
        "threshold": 0.5, "threshold_search": False,
        "spatial_preprocessing": "native pixels; right/bottom padding only",
        "metrics": ["DSC", "NSD@2px", "VOE", "symmetric MSD (pixels)", "MSD Fail Rate", "RAVD"],
        "pair_definition": "top 15 train overlap pairs; empty GT pair/cases excluded",
        "results": {name: official_metrics(data, pairs) for name, data in datasets.items()},
    }
    base = result["results"]["baseline"]
    for name in ("generic_adapter", "instance_prior"):
        result["results"][name]["delta_vs_baseline"] = {
            section: {k: (result["results"][name][section].get(k) - base[section].get(k))
                      if isinstance(result["results"][name][section].get(k), (int, float))
                      and isinstance(base[section].get(k), (int, float)) else None
                      for k in ("dsc", "nsd_2px", "voe", "msd_px", "msd_fail_rate", "ravd")}
            for section in ("overall_instance_metrics", "overlap_region_metrics", "overlap_pair_intersection_metrics")
        }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
