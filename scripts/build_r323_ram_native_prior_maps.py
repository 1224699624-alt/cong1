#!/usr/bin/env python3
"""Build GT-free RAM train/val maps from the single frozen R323 prior."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from build_r259_frozen_prior_maps import crop_input, load_checkpoint_portable
from build_r322_ram_r317_seam_prior_maps import (canonical_image, nearest_pairs,
                                                 predicted_instances, prediction_tensor)
from train_r258b_prediction_relation_prior import RelationPriorNet
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior


def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--baseline-checkpoint",type=Path,required=True); p.add_argument("--prior-checkpoint",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True); p.add_argument("--size",type=int,default=384)
    p.add_argument("--threshold",type=float,default=.5); p.add_argument("--nearest-neighbors",type=int,default=3)
    p.add_argument("--relative-pair-limit",type=float,default=2.5); p.add_argument("--limit-train",type=int,default=0)
    p.add_argument("--limit-val",type=int,default=0); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args(); device=torch.device(a.device); a.output_root.mkdir(parents=True,exist_ok=True)
    baseline=NnUNetMultiLabelPrior().to(device); bp=torch.load(a.baseline_checkpoint,map_location=device,weights_only=False)
    baseline.load_state_dict(bp["model"],strict=True); baseline.eval()
    payload=load_checkpoint_portable(a.prior_checkpoint)
    if int(payload.get("seed",-1)) in (2581,2582,2583) or payload.get("use_development") is not False:
        raise RuntimeError("Expected a single RAM-native image-centers-only checkpoint")
    prior=RelationPriorNet(4).to(device); prior.load_state_dict(payload["model"],strict=True); prior.eval()
    audit={"experiment":"R323_RAM_NATIVE_PRIOR_MAPS","source":"prediction-derived centers; GT-free map generation",
           "test_used":False,"baseline_checkpoint":str(a.baseline_checkpoint),"baseline_sha256":sha256(a.baseline_checkpoint),
           "prior_checkpoint":str(a.prior_checkpoint),"prior_sha256":sha256(a.prior_checkpoint),
           "prior_seed":int(payload["seed"]),"selected_epoch":int(payload["selected_epoch"]),"splits":{}}
    for split,limit in (("train",a.limit_train),("val",a.limit_val)):
        allowed={x.stem for x in (a.dataset_root/"BoneSegmentation"/"masks"/split).glob("*.npy")}
        paths=[x for x in sorted((a.dataset_root/"BoneSegmentation"/"images").glob("*.bmp")) if x.stem in allowed][:limit or None]
        out=a.output_root/split; out.mkdir(parents=True,exist_ok=True); rows=[]
        for number,path in enumerate(paths,1):
            image=canonical_image(path,a.size)
            with torch.no_grad(): logits,_=baseline(prediction_tensor(image,device),False); probability=torch.sigmoid(logits)[0].cpu().numpy()
            instances=predicted_instances(probability,a.threshold); pairs=nearest_pairs(instances,a.nearest_neighbors,a.relative_pair_limit)
            full=np.zeros((a.size,a.size),np.float32)
            for i,j in pairs:
                x,y=instances[i],instances[j]; channels,(x0,y0,side)=crop_input(image,x["point"],y["point"],x["scale"],y["scale"],256,True,128,.15,3.,12.)
                with torch.no_grad():
                    relation,heat=prior(torch.from_numpy(channels[None]).to(device),torch.zeros((1,2),device=device))
                    local=(torch.sigmoid(relation)[:,None,None,None]*torch.sigmoid(heat))[0,0].cpu().numpy()
                patch=cv2.resize(local,(side,side),interpolation=cv2.INTER_LINEAR)
                sx0,sy0,sx1,sy1=max(0,x0),max(0,y0),min(a.size,x0+side),min(a.size,y0+side)
                if sx1>sx0 and sy1>sy0:
                    crop=patch[sy0-y0:sy1-y0,sx0-x0:sx1-x0]
                    full[sy0:sy1,sx0:sx1]=np.maximum(full[sy0:sy1,sx0:sx1],crop)
            np.save(out/f"{path.stem}.npy",full); Image.fromarray(np.clip(full*255,0,255).astype(np.uint8)).save(out/f"{path.stem}.png")
            rows.append({"case":path.stem,"instances":len(instances),"pairs":len(pairs),"mean":float(full.mean()),"max":float(full.max())})
            if number%25==0 or number==len(paths): print(json.dumps({"split":split,"done":number,"total":len(paths)}),flush=True)
        audit["splits"][split]={"count":len(rows),"mean_pairs":float(np.mean([x["pairs"] for x in rows])),
                                "mean_prior":float(np.mean([x["mean"] for x in rows])),"zero_maps":sum(x["max"]<=0 for x in rows)}
    (a.output_root/"manifest.json").write_text(json.dumps(audit,indent=2),encoding="utf-8"); print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
