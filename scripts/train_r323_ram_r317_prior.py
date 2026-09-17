#!/usr/bin/env python3
"""R323: single-seed, long-schedule R317 prior trained natively on RAM-W600.

The R317 architecture and four-channel dual-scale input remain unchanged.
Frozen R285 predictions provide centers; RAM GT is used only to supervise the
pair label and background-seam heatmap on train/validation splits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from r316c_pair_inputs import pair_channels
from train_r256_scale_invariant_pair_prior import make_heat_target
from train_r258b_prediction_relation_prior import RelationPriorNet, heat_loss
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from build_r322_ram_r317_seam_prior_maps import predicted_instances, nearest_pairs


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=24)
def load_case(image_path: str, mask_path: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(image_path).convert("L").resize((size, size), Image.Resampling.BILINEAR))
    masks = (np.load(mask_path) > 0).astype(np.uint8)
    stem = Path(mask_path).stem
    if stem.endswith("_R"):
        image, masks = np.fliplr(image), np.flip(masks, axis=-1)
    resized = np.stack([cv2.resize(x, (size, size), interpolation=cv2.INTER_NEAREST) for x in masks]).astype(bool)
    return np.ascontiguousarray(image), np.ascontiguousarray(resized)


def prediction_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(image.astype(np.float32) / 255.)[None, None]
    value = (value - value.mean((2, 3), keepdim=True)) / (value.std((2, 3), keepdim=True) + 1e-6)
    return value.to(device)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    yy, xx = np.nonzero(boundary)
    if len(xx) > 512:
        take = np.linspace(0, len(xx) - 1, 512).astype(int); yy, xx = yy[take], xx[take]
    return np.stack([yy, xx], 1)


def boundary_gap(a: np.ndarray, b: np.ndarray) -> float:
    ba, bb = mask_boundary(a), mask_boundary(b)
    if not len(ba) or not len(bb): return float("inf")
    distance, _ = cKDTree(bb).query(ba, k=1)
    return max(0., float(distance.min()) - 1.)


def build_samples(root: Path, split: str, baseline: torch.nn.Module, device: torch.device,
                  size: int, threshold: float, neighbors: int, pair_limit: float,
                  seam_limit: float, max_pairs: int, limit: int = 0) -> tuple[list[dict], dict]:
    mask_paths = sorted((root / "BoneSegmentation" / "masks" / split).glob("*.npy"))[:limit or None]
    samples, audit = [], {"images": len(mask_paths), "positive": 0, "negative": 0,
                          "overlap_negative": 0, "cases_without_balanced_pairs": 0}
    baseline.eval()
    for number, mask_path in enumerate(mask_paths, 1):
        image_path = root / "BoneSegmentation" / "images" / f"{mask_path.stem}.bmp"
        image, masks = load_case(str(image_path), str(mask_path), size)
        with torch.no_grad():
            logits, _ = baseline(prediction_tensor(image, device), False)
            probability = torch.sigmoid(logits)[0].cpu().numpy()
        instances = predicted_instances(probability, threshold)
        candidates = []
        for ia, ib in nearest_pairs(instances, neighbors, pair_limit):
            a, b = instances[ia], instances[ib]; ca, cb = int(a["channel"]), int(b["channel"])
            ma, mb = masks[ca], masks[cb]
            if not ma.any() or not mb.any(): continue
            overlap = bool(np.logical_and(ma, mb).any())
            gap = 0. if overlap else boundary_gap(ma, mb)
            local_scale = math.sqrt(.5 * (float(ma.sum()) + float(mb.sum())))
            relative_gap = gap / max(local_scale, 1.)
            positive = (not overlap) and relative_gap <= seam_limit
            distance = float(np.linalg.norm(a["point"] - b["point"]))
            relative_center = distance / max(a["scale"] + b["scale"], 1e-6)
            candidates.append({"positive": positive, "overlap": overlap, "gap": gap,
                               "relative_gap": relative_gap, "relative_center": relative_center,
                               "channel_a": ca, "channel_b": cb,
                               "pa": a["point"].tolist(), "pb": b["point"].tolist(),
                               "psa": float(a["scale"]), "psb": float(b["scale"])})
        positives = sorted((x for x in candidates if x["positive"]), key=lambda x: x["relative_gap"])
        negatives = [x for x in candidates if not x["positive"]]
        cap = min(len(positives), len(negatives), max_pairs)
        if cap == 0:
            audit["cases_without_balanced_pairs"] += 1; continue
        positives = positives[:cap]; available = list(range(len(negatives))); selected = []
        for pos in positives:
            index = min(available, key=lambda k: abs(negatives[k]["relative_center"] - pos["relative_center"]))
            selected.append(negatives[index]); available.remove(index)
        for item in positives + selected:
            samples.append({"case": mask_path.stem, "image": str(image_path), "mask": str(mask_path), **item})
        audit["positive"] += cap; audit["negative"] += cap
        audit["overlap_negative"] += sum(int(x["overlap"]) for x in selected)
        if number % 50 == 0 or number == len(mask_paths):
            print(json.dumps({"stage": "samples", "split": split, "done": number,
                              "total": len(mask_paths), "pairs": len(samples)}), flush=True)
    return samples, audit


class RamPairDataset(Dataset):
    def __init__(self, samples: list[dict], size: int, augment: bool, jitter: float) -> None:
        self.samples, self.size, self.augment, self.jitter = samples, size, augment, jitter
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image, masks = load_case(sample["image"], sample["mask"], 384)
        pa, pb = np.asarray(sample["pa"], np.float64), np.asarray(sample["pb"], np.float64)
        psa, psb = float(sample["psa"]), float(sample["psb"])
        center = .5 * (pa + pb); side = int(math.ceil(max(32., np.linalg.norm(pb-pa) + 1.35*(psa+psb))))
        x0, y0 = int(round(center[0]-side/2)), int(round(center[1]-side/2))
        def crop(array: np.ndarray) -> np.ndarray:
            out = np.zeros((side, side), dtype=array.dtype)
            sx0, sy0, sx1, sy1 = max(0,x0), max(0,y0), min(384,x0+side), min(384,y0+side)
            if sx1>sx0 and sy1>sy0: out[sy0-y0:sy1-y0,sx0-x0:sx1-x0] = array[sy0:sy1,sx0:sx1]
            return out
        scale = self.size / side; ca, cb = (pa-[x0,y0])*scale, (pb-[x0,y0])*scale
        channels, _, _, _ = pair_channels(crop(image), ca, cb, .5*(psa+psb)*scale,
                                           self.size, 128, .15, 3., 12., self.jitter if self.augment else 0.)
        target, valid = np.zeros((self.size,self.size),np.float32), False
        if sample["positive"]:
            ma = crop(masks[int(sample["channel_a"])]); mb = crop(masks[int(sample["channel_b"])])
            fg = crop(masks.any(0))
            target, valid = make_heat_target(ma, mb, fg, self.size, ca, cb, .5*(psa+psb)*scale)
        if self.augment:
            if random.random()<.5: channels,target=channels[:,:,::-1].copy(),target[:,::-1].copy()
            if random.random()<.5: channels,target=channels[:,::-1,:].copy(),target[::-1,:].copy()
            turns=random.randrange(4)
            if turns: channels,target=np.rot90(channels,turns,axes=(1,2)).copy(),np.rot90(target,turns).copy()
            if random.random()<.25: channels[0]=np.clip(channels[0]**random.uniform(.9,1.1),0,1)
        return {"image":torch.from_numpy(channels), "features":torch.zeros(2),
                "label":torch.tensor(float(sample["positive"])), "heatmap":torch.from_numpy(target[None]),
                "heat_valid":torch.tensor(float(valid)),
                "close":torch.tensor(float(sample["positive"] and sample["relative_gap"]<=.20)),
                "contact":torch.tensor(float(sample["positive"] and sample["gap"]<1.)), "case":sample["case"]}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval(); labels, scores, close, contact, heat_dice, valid_heat = [], [], [], [], [], 0
    for batch in loader:
        logits, heat = model(batch["image"].to(device), batch["features"].to(device))
        target=batch["heatmap"].to(device); probability=torch.sigmoid(heat)
        labels.extend(batch["label"].numpy()); scores.extend(torch.sigmoid(logits).cpu().numpy())
        close.extend(batch["close"].numpy()); contact.extend(batch["contact"].numpy())
        valid=(batch["heat_valid"].numpy()>.5)&(batch["label"].numpy()>.5); valid_heat += int(valid.sum())
        dice=((2*(probability*target).sum((1,2,3))+1)/(probability.sum((1,2,3))+target.sum((1,2,3))+1)).cpu().numpy()
        heat_dice.extend(dice[valid])
    y,p=np.asarray(labels),np.asarray(scores); pred=p>=.5; close=np.asarray(close)>.5; contact=np.asarray(contact)>.5
    if len(np.unique(y))<2: raise RuntimeError("Validation requires both pair classes")
    negative=y<.5
    return {"auroc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),
            "accuracy":float((pred==y).mean()),"positive_recall":float(pred[y>.5].mean()),
            "close_pair_recall":float(pred[close].mean()) if close.any() else 0.,
            "contact_pair_recall":float(pred[contact].mean()) if contact.any() else 0.,
            "negative_specificity":float((~pred[negative]).mean()),
            "positive_heatmap_soft_dice":float(np.mean(heat_dice)) if heat_dice else 0.,
            "valid_positive_heat_count":valid_heat}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--baseline-checkpoint",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--result-json",type=Path,required=True); parser.add_argument("--epochs",type=int,default=60)
    parser.add_argument("--seed",type=int,default=3231); parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--workers",type=int,default=4); parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--size",type=int,default=256); parser.add_argument("--threshold",type=float,default=.5)
    parser.add_argument("--neighbors",type=int,default=3); parser.add_argument("--pair-limit",type=float,default=2.5)
    parser.add_argument("--seam-relative-limit",type=float,default=.8); parser.add_argument("--max-pairs",type=int,default=32)
    parser.add_argument("--limit-train",type=int,default=0); parser.add_argument("--limit-val",type=int,default=0)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); args=parser.parse_args()
    if args.seed in (2581,2582,2583): raise RuntimeError("R323 requires an isolated single RAM seed")
    if (args.dataset_root/"BoneSegmentation"/"masks"/"test").is_dir() is False: raise RuntimeError("Unexpected RAM layout")
    seed_all(args.seed); device=torch.device(args.device); args.output_dir.mkdir(parents=True,exist_ok=True)
    baseline=NnUNetMultiLabelPrior().to(device); payload=torch.load(args.baseline_checkpoint,map_location=device,weights_only=False)
    baseline.load_state_dict(payload["model"],strict=True); baseline.eval()
    train,audit_train=build_samples(args.dataset_root,"train",baseline,device,384,args.threshold,args.neighbors,args.pair_limit,args.seam_relative_limit,args.max_pairs,args.limit_train)
    val,audit_val=build_samples(args.dataset_root,"val",baseline,device,384,args.threshold,args.neighbors,args.pair_limit,args.seam_relative_limit,args.max_pairs,args.limit_val)
    del baseline; torch.cuda.empty_cache()
    if not train or not val: raise RuntimeError("No balanced RAM pairs")
    train_loader=DataLoader(RamPairDataset(train,args.size,True,.03),args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val_loader=DataLoader(RamPairDataset(val,args.size,False,0.),args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=RelationPriorNet(4).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda"); history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in train_loader:
            image=batch["image"].to(device); features=batch["features"].to(device); label=batch["label"].to(device)
            target=batch["heatmap"].to(device); valid=batch["heat_valid"].to(device); optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
                logit,heat=model(image,features); loss=F.binary_cross_entropy_with_logits(logit,label)+.35*heat_loss(heat,target,label,valid)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach()))
        metrics=evaluate(model,val_loader,device); row={"epoch":epoch,"train_loss":float(np.mean(losses)),**metrics}
        history.append(row); print(json.dumps(row),flush=True)
        torch.save({"model":model.state_dict(),"seed":args.seed,"epoch":epoch,"use_development":False,
                    "config":vars(args)},args.output_dir/f"ram_image_centers_seed{args.seed}_epoch{epoch:03d}.pt")
    def key(row):
        safe=row["negative_specificity"]>=.75 and row["positive_recall"]>=.70 and row["valid_positive_heat_count"]>0
        return (float(safe),row["positive_heatmap_soft_dice"],row["auroc"],row["average_precision"],-row["epoch"])
    selected=max(history,key=key); source=args.output_dir/f"ram_image_centers_seed{args.seed}_epoch{selected['epoch']:03d}.pt"
    chosen=torch.load(source,map_location="cpu",weights_only=False); chosen["selected_epoch"]=selected["epoch"]
    checkpoint=args.output_dir/f"ram_image_centers_seed{args.seed}_selected.pt"; torch.save(chosen,checkpoint)
    result={"experiment":"R323_RAM_NATIVE_R317_PRIOR","dataset":"RAM-W600","splits":{"train":audit_train,"val":audit_val},
            "test_used":False,"single_seed":args.seed,"epochs_completed":args.epochs,"selected_epoch":selected["epoch"],
            "selected":selected,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"history":history,
            "elapsed_seconds":time.time()-started,"config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    args.result_json.parent.mkdir(parents=True,exist_ok=True); args.result_json.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"complete":True,"selected_epoch":selected["epoch"],"checkpoint":str(checkpoint)},indent=2))


if __name__=="__main__": main()
