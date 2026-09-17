#!/usr/bin/env python3
"""R334: storage-light RAM seam adapter on top of frozen R332 step-2 output.

The mature R325 nnU-Net and R332 overlap refiner are immutable.  Only a tiny,
zero-initialized seam-gated residual is trained.  Validation compares against
the actual frozen R332 step-2 prediction, not the pre-refinement backbone.
RAM test is never opened.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, seed_all
from train_r332_ram_joint_iterative_refinement import flatten, iterative_refine, checkpoint_gate


class RamSeamDataset(NativeWristDataset):
    def __init__(self, root, prior_root, split, augment):
        super().__init__(root, split, augment)
        self.prior_root = Path(prior_root)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        case = item["case"]
        npy = self.prior_root / self.split / f"{case}.npy"
        if npy.exists():
            value = torch.from_numpy(np.load(npy).astype(np.float32))[None, None]
        else:
            from PIL import Image
            png = self.prior_root / self.split / f"{case}.png"
            value = torch.from_numpy(np.asarray(Image.open(png), dtype=np.float32) / 255.0)[None, None]
        h, w = item["original_hw"].tolist()
        value = F.interpolate(value, size=(h, w), mode="bilinear", align_corners=False)[0]
        value = F.pad(value, (0, item["image"].shape[-1] - w, 0, item["image"].shape[-2] - h))
        item["seam"] = value
        return item


class FrozenSeamAdapter(nn.Module):
    """Predict only a bounded negative foreground correction near a seam prior."""
    def __init__(self, channels=24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(-3.0))

    def forward(self, image, logits, seam):
        prob = torch.sigmoid(logits.detach())
        union = prob.amax(1, keepdim=True)
        confidence = (2.0 * (union - 0.5).abs()).clamp(0, 1)
        grad_x = F.pad((image[:, :, :, 1:] - image[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((image[:, :, 1:, :] - image[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8)
        raw = self.body(torch.cat([image, union, seam, edge], 1))
        gate = seam.clamp(0, 1).square() * confidence
        magnitude = F.softplus(raw) * torch.sigmoid(self.log_scale) * gate
        # Distribute suppression only over instances already claiming the pixel.
        assignment = prob / prob.sum(1, keepdim=True).clamp_min(1e-4)
        return logits - magnitude * assignment, magnitude


@torch.inference_mode()
def frozen_r332(network, refiner, image, steps):
    base, _ = network(image, False)
    return iterative_refine(image, base, refiner, steps)[0][-1]


def losses(corrected, teacher, target, seam, magnitude, seam_weight, distill_weight):
    prob = torch.sigmoid(corrected.float())
    target = target.float()
    reliable_sep = seam.square() * (target.sum(1, keepdim=True) == 0).float()
    seam_loss = (reliable_sep * prob.amax(1, keepdim=True)).sum() / reliable_sep.sum().clamp_min(1)
    # Exact preservation away from the reliable separation state.
    protect = (1.0 - reliable_sep).expand_as(prob)
    distill = (protect * (prob - torch.sigmoid(teacher.float())).square()).sum() / protect.sum().clamp_min(1)
    # Prevent needless corrections, including on uncertain regions.
    sparse = magnitude.mean()
    total = seam_weight * seam_loss + distill_weight * distill + 0.01 * sparse
    return total, seam_loss, distill, sparse


@torch.inference_mode()
def collect(network, r332, adapter, loader, device, steps, pairs):
    network.eval(); r332.eval(); adapter.eval()
    before, after, fp0, fp1 = {}, {}, [], []
    for batch in loader:
        image = batch["image"].to(device)
        target = (batch["mask"].to(device) > 0.5)
        seam = batch["seam"].to(device)
        teacher = frozen_r332(network, r332, image, steps)
        corrected, _ = adapter(image, teacher, seam)
        p0, p1 = torch.sigmoid(teacher) >= 0.5, torch.sigmoid(corrected) >= 0.5
        case = str(batch["case"][0])
        before[case] = {"pred": p0.cpu().numpy()[0], "target": target.cpu().numpy()[0]}
        after[case] = {"pred": p1.cpu().numpy()[0], "target": target.cpu().numpy()[0]}
        valid = seam.square() * (target.sum(1, keepdim=True) == 0).float()
        fp0.append(float(((p0.any(1, keepdim=True) * valid).sum() / valid.sum().clamp_min(1))))
        fp1.append(float(((p1.any(1, keepdim=True) * valid).sum() / valid.sum().clamp_min(1))))
    return {
        "r332_step2": {"official": official_metrics(before, pairs), "seam_fp": float(np.mean(fp0))},
        "adapted": {"official": official_metrics(after, pairs), "seam_fp": float(np.mean(fp1))},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--prior-root", type=Path, required=True)
    p.add_argument("--r332-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--seam-weight", type=float, default=0.05)
    p.add_argument("--distill-weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=3341)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed); device = torch.device("cuda")
    payload = torch.load(a.r332_checkpoint, map_location=device, weights_only=False)
    network = NnUNetMultiLabelPrior().to(device); network.load_state_dict(payload["model"], strict=True)
    r332 = InstanceCompletionRefiner().to(device); r332.load_state_dict(payload["refiner"], strict=True)
    network.requires_grad_(False); r332.requires_grad_(False)
    adapter = FrozenSeamAdapter().to(device)
    train = RamSeamDataset(a.dataset_root, a.prior_root, "train", True)
    val = RamSeamDataset(a.dataset_root, a.prior_root, "val", False)
    if a.smoke: train.mask_files = train.mask_files[:2]; val.mask_files = val.mask_files[:2]
    tl = DataLoader(train, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=not a.smoke)
    vl = DataLoader(val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=2e-4, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(True); pairs = top_training_pairs(a.dataset_root, 15)
    if a.smoke:
        b = next(iter(tl)); image=b["image"].to(device); target=b["mask"].to(device); seam=b["seam"].to(device)
        with torch.no_grad(): teacher=frozen_r332(network,r332,image,a.steps)
        corrected,mag=adapter(image,teacher,seam); loss,sl,dl,sp=losses(corrected,teacher,target,seam,mag,a.seam_weight,a.distill_weight)
        loss.backward()
        print(json.dumps({"finite":bool(torch.isfinite(loss)),"loss":float(loss),"seam":float(sl),"distill":float(dl),"sparse":float(sp),"trainable":sum(x.numel() for x in adapter.parameters())}))
        return
    baseline = collect(network, r332, adapter, vl, device, a.steps, pairs)["r332_step2"]
    base_flat = flatten(baseline["official"]); best = None; history = a.output / "history.jsonl"
    for epoch in range(1, a.epochs + 1):
        adapter.train(); sums=np.zeros(4,np.float64); started=time.time()
        for b in tl:
            image=b["image"].to(device); target=b["mask"].to(device); seam=b["seam"].to(device)
            with torch.no_grad(): teacher=frozen_r332(network,r332,image,a.steps)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):
                corrected,mag=adapter(image,teacher,seam); terms=losses(corrected,teacher,target,seam,mag,a.seam_weight,a.distill_weight)
            scaler.scale(terms[0]).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(adapter.parameters(),5)
            scaler.step(opt); scaler.update(); sums += [float(x) for x in terms]
        metrics=collect(network,r332,adapter,vl,device,a.steps,pairs); final=flatten(metrics["adapted"]["official"])
        gate,checks=checkpoint_gate(final,base_flat)
        # R332 must also retain its primary overlap surface/area behavior.
        checks.update({
            "overlap_dsc_not_below_r332": final["overlap_dsc"] >= base_flat["overlap_dsc"],
            "overlap_nsd_not_below_r332": final["overlap_nsd_2px"] >= base_flat["overlap_nsd_2px"],
            "overlap_msd_not_above_r332": final["overlap_msd_px"] <= base_flat["overlap_msd_px"],
            "seam_fp_not_above_r332": metrics["adapted"]["seam_fp"] <= baseline["seam_fp"],
        }); gate=bool(gate and all(checks.values()))
        row={"epoch":epoch,"seconds":time.time()-started,"losses":dict(zip(["total","seam","distill","sparse"],(sums/len(tl)).tolist())),"gate":gate,"checks":checks,"baseline":base_flat,"adapted":final,"seam_fp":{"baseline":baseline["seam_fp"],"adapted":metrics["adapted"]["seam_fp"]}}
        with history.open("a") as f: f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        score=(int(gate), final["overall_dsc"]-base_flat["overall_dsc"], final["overlap_nsd_2px"]-base_flat["overlap_nsd_2px"], baseline["seam_fp"]-metrics["adapted"]["seam_fp"])
        if best is None or score > best[0]:
            best=(score,row); torch.save({"adapter":adapter.state_dict(),"epoch":epoch,"row":row},a.output/"best.pth")
    result={"experiment":"R334_RAM_FROZEN_DUAL_INTERACTION","split":"validation","test_used":False,"threshold":0.5,"r332_frozen":True,"config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},"best":best[1]}
    (a.output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"complete":True,"best_epoch":best[1]["epoch"],"gate":best[1]["gate"]}))


if __name__ == "__main__":
    main()
