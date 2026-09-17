#!/usr/bin/env python3
"""R330: refine R329 with pair-aware surface and volume supervision.

RAM test is never loaded. Two predeclared validation-only variants are trained
from the same R329 checkpoint and audited with RAM paper-aligned metrics.
"""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import collect_baseline, collect_instance, official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import (
    InstanceCompletionRefiner, completion_loss, evaluate, per_instance_features, seed_all,
)


def choose_pair_instances(probability, target, count):
    """Always include the most erroneous true-overlap pair, then fill randomly."""
    t = target[0] > 0.5; p = probability[0]
    scores = []
    for i in range(14):
        for j in range(i + 1, 14):
            region = t[i] & t[j]
            if region.any():
                error = (torch.abs(p[i] - t[i].float()) + torch.abs(p[j] - t[j].float()))[region].mean()
                scores.append((float(error), i, j))
    chosen = []
    if scores:
        _, i, j = max(scores); chosen.extend((i, j))
    remaining = [i for i in range(14) if i not in chosen]
    random.shuffle(remaining); chosen.extend(remaining[:max(0, count - len(chosen))])
    return torch.tensor(chosen[:count], device=target.device, dtype=torch.long)


def soft_surface_nsd_loss(probability, target, tolerance=2):
    target = target.float()
    pb = (F.max_pool2d(probability, 3, 1, 1) + F.max_pool2d(-probability, 3, 1, 1)).clamp(0, 1)
    tb = (F.max_pool2d(target, 3, 1, 1) + F.max_pool2d(-target, 3, 1, 1)).clamp(0, 1)
    kernel = tolerance * 2 + 1
    tb_tol = F.max_pool2d(tb, kernel, 1, tolerance)
    pb_tol = F.max_pool2d(pb, kernel, 1, tolerance)
    precision = (pb * tb_tol).sum((1, 2, 3)) / pb.sum((1, 2, 3)).clamp_min(1.0)
    recall = (tb * pb_tol).sum((1, 2, 3)) / tb.sum((1, 2, 3)).clamp_min(1.0)
    return (1.0 - 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)).mean()


def volume_losses(probability, target):
    target = target.float()
    pred_mass = probability.sum((1, 2, 3)); true_mass = target.sum((1, 2, 3))
    instance = (torch.abs(pred_mass - true_mass) / (true_mass + 64.0)).mean()
    pair_terms = []
    for i in range(target.shape[0]):
        for j in range(i + 1, target.shape[0]):
            true_pair = target[i, 0] * target[j, 0]
            true_pair_mass = true_pair.sum()
            if true_pair_mass > 0:
                pred_pair_mass = (probability[i, 0] * probability[j, 0]).sum()
                pair_terms.append(torch.abs(pred_pair_mass - true_pair_mass) / (true_pair_mass + 32.0))
    pair = torch.stack(pair_terms).mean() if pair_terms else probability.sum() * 0.0
    return instance, pair


def train_variant(name, baseline, train_set, val_loader, baseline_metrics, initial_state,
                  device, output, surface_weight, instance_volume_weight, pair_volume_weight,
                  epochs=18, seed=3301):
    seed_all(seed); model = InstanceCompletionRefiner().to(device)
    model.load_state_dict(copy.deepcopy(initial_state), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
                        persistent_workers=True, generator=generator)
    history = output / f"{name}_history.jsonl"; checkpoint = output / f"{name}_best.pth"
    best = None; bad = 0
    for epoch in range(1, epochs + 1):
        model.train(); sums = {k: 0.0 for k in ("loss", "base", "surface", "instance_volume", "pair_volume")}
        started = time.time()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True); target = batch["mask"].to(device, non_blocking=True)
            with torch.no_grad():
                base_logits, _ = baseline(image, False); base_probability = torch.sigmoid(base_logits)
                indices = choose_pair_instances(base_probability, target, 4)
                source = base_probability[:, indices]
                feature = per_instance_features(image, base_probability, indices, source)
                selected_target = target[0, indices].unsqueeze(1)
                error = torch.abs(source[0].unsqueeze(1) - selected_target)
                local = F.max_pool2d((error > 0.15).float(), 11, 1, 5)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits, delta = model(feature, base_logits[0, indices].unsqueeze(1))
                base_loss, _ = completion_loss(logits, selected_target, local, delta, 0.05)
                probability = torch.sigmoid(logits.float())
                surface = soft_surface_nsd_loss(probability, selected_target)
                instance_volume, pair_volume = volume_losses(probability, selected_target)
                loss = base_loss + surface_weight * surface + instance_volume_weight * instance_volume + pair_volume_weight * pair_volume
            if not torch.isfinite(loss): raise RuntimeError({"name": name, "epoch": epoch, "loss": float(loss)})
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer); scaler.update()
            for key, value in (("loss", loss), ("base", base_loss), ("surface", surface),
                               ("instance_volume", instance_volume), ("pair_volume", pair_volume)):
                sums[key] += float(value.detach())
        scheduler.step(); metrics = evaluate(baseline, model, val_loader, device)
        passes = metrics["macro_dsc"] >= baseline_metrics["macro_dsc"] and metrics["macro_iou"] >= baseline_metrics["macro_iou"]
        row = {"variant": name, "epoch": epoch, "seconds": time.time()-started,
               "surface_weight": surface_weight, "instance_volume_weight": instance_volume_weight,
               "pair_volume_weight": pair_volume_weight, "passes_macro_gate": passes,
               **{f"train_{k}": v/len(loader) for k,v in sums.items()}, **metrics}
        with history.open("a", encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
        key = (int(passes), metrics["overlap_nsd_2px"], metrics["overlap_dsc"], -metrics["overlap_msd_px"])
        if best is None or key > best[0]:
            best = (key, row); bad = 0; torch.save({"refiner": model.state_dict(), "row": row}, checkpoint)
        else: bad += 1
        if epoch >= 10 and bad >= 6: break
    return checkpoint, best[1]


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--baseline-checkpoint", type=Path, required=True)
    ap.add_argument("--r329-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); args=ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    seed_all(3301); device=torch.device("cuda");
    train_set=NativeWristDataset(args.dataset_root,"train",augment=True); val_set=NativeWristDataset(args.dataset_root,"val",augment=False)
    val_loader=DataLoader(val_set,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    baseline=NnUNetMultiLabelPrior().to(device); baseline.load_state_dict(torch.load(args.baseline_checkpoint,map_location=device,weights_only=False)["model"],strict=True); baseline.eval()
    for p in baseline.parameters(): p.requires_grad_(False)
    baseline_metrics=evaluate(baseline,None,val_loader,device)
    initial=torch.load(args.r329_checkpoint,map_location="cpu",weights_only=False)["refiner"]
    configs={"surface003_volume010":{"surface_weight":0.03,"instance_volume_weight":0.005,"pair_volume_weight":0.01},
             "surface006_volume020":{"surface_weight":0.06,"instance_volume_weight":0.010,"pair_volume_weight":0.02}}
    best={}; paths={}
    for name,cfg in configs.items():
        paths[name],best[name]=train_variant(name,baseline,train_set,val_loader,baseline_metrics,initial,device,args.output,**cfg)
    pairs=top_training_pairs(args.dataset_root,15)
    official={"baseline":official_metrics(collect_baseline(baseline,val_loader,device),pairs)}
    r329=InstanceCompletionRefiner().to(device); r329.load_state_dict(initial,strict=True)
    official["r329"]=official_metrics(collect_instance(baseline,r329,val_loader,device),pairs)
    for name,path in paths.items():
        model=InstanceCompletionRefiner().to(device); model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["refiner"],strict=True)
        official[name]=official_metrics(collect_instance(baseline,model,val_loader,device),pairs)
    result={"experiment":"R330_RAM_PAIR_SURFACE_VOLUME_PRIOR","split":"validation","test_used":False,
            "threshold":0.5,"threshold_search":False,"configs":configs,"best_rows":best,"official_metrics":official}
    (args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
