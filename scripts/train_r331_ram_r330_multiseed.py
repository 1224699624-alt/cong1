#!/usr/bin/env python3
"""R331: five-seed stability audit of the fixed R330 weak configuration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import collect_baseline, collect_instance, official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, evaluate, seed_all
from train_r330_ram_pair_surface_volume_prior import train_variant


SEEDS = (1024, 2025, 3407, 4096, 5214)
SECTIONS = ("overall_instance_metrics", "overlap_region_metrics", "overlap_pair_intersection_metrics")
METRICS = ("dsc", "nsd_2px", "voe", "msd_px", "msd_fail_rate", "ravd")


def aggregate(seed_results, baseline):
    output = {}
    for section in SECTIONS:
        output[section] = {}
        for metric in METRICS:
            values = np.asarray([seed_results[str(seed)][section][metric] for seed in SEEDS], dtype=float)
            base = float(baseline[section][metric])
            lower = metric in ("voe", "msd_px", "msd_fail_rate", "ravd")
            wins = values < base if lower else values > base
            output[section][metric] = {
                "mean": float(values.mean()), "std": float(values.std(ddof=1)),
                "min": float(values.min()), "max": float(values.max()),
                "baseline": base, "delta_mean_vs_baseline": float(values.mean() - base),
                "win_count": int(wins.sum()), "n": len(values),
                "values": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
            }
    return output


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--baseline-checkpoint",type=Path,required=True)
    ap.add_argument("--r329-checkpoint",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True); args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("R331 requires CUDA")
    device=torch.device("cuda"); train_set=NativeWristDataset(args.dataset_root,"train",augment=True)
    val_set=NativeWristDataset(args.dataset_root,"val",augment=False)
    val_loader=DataLoader(val_set,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    baseline=NnUNetMultiLabelPrior().to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint,map_location=device,weights_only=False)["model"],strict=True);baseline.eval()
    for p in baseline.parameters():p.requires_grad_(False)
    baseline_project=evaluate(baseline,None,val_loader,device)
    initial=torch.load(args.r329_checkpoint,map_location="cpu",weights_only=False)["refiner"]
    pairs=top_training_pairs(args.dataset_root,15)
    baseline_official=official_metrics(collect_baseline(baseline,val_loader,device),pairs)
    seed_results={};best_rows={};checkpoints={}
    partial={"experiment":"R331_RAM_R330_WEAK_FIVE_SEED","split":"validation","test_used":False,
             "seeds":list(SEEDS),"fixed_config":{"surface_weight":0.03,"instance_volume_weight":0.005,"pair_volume_weight":0.01},
             "baseline":baseline_official,"seed_results":seed_results,"best_rows":best_rows}
    for seed in SEEDS:
        name=f"seed_{seed}"
        checkpoint,best=train_variant(name,baseline,train_set,val_loader,baseline_project,initial,device,args.output,
                                      surface_weight=0.03,instance_volume_weight=0.005,pair_volume_weight=0.01,
                                      epochs=18,seed=seed)
        model=InstanceCompletionRefiner().to(device)
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["refiner"],strict=True)
        seed_results[str(seed)]=official_metrics(collect_instance(baseline,model,val_loader,device),pairs)
        best_rows[str(seed)]=best;checkpoints[str(seed)]=str(checkpoint)
        partial["checkpoints"]=checkpoints
        (args.output/"partial_result.json").write_text(json.dumps(partial,indent=2),encoding="utf-8")
        print(json.dumps({"completed_seed":seed,"official":seed_results[str(seed)]}),flush=True)
    result={**partial,"aggregate":aggregate(seed_results,baseline_official),"complete":True,
            "threshold":0.5,"threshold_search":False,"spatial_preprocessing":"native pixels; right/bottom padding only"}
    (args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
