#!/usr/bin/env python3
"""Fast synthetic checks for R257 geometry, targets, peaks, and matching."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from train_r257_image_only_center_scale_detector import (
    CenterScaleUNet,
    boundary_gap,
    dense_targets,
    decide,
    extract_proposals,
    instance_records,
    letterbox,
    match_centers,
    training_loss,
)


def logit(probability: np.ndarray) -> torch.Tensor:
    probability = np.clip(probability, 1e-4, 1-1e-4)
    return torch.from_numpy(np.log(probability/(1-probability)).astype(np.float32))[None,None]


def main() -> None:
    image=np.full((80,120),100,np.uint8); instance=np.zeros((80,120),np.int32)
    instance[25:45,25:45]=1; instance[25:45,49:69]=2; instance[50:65,85:105]=3
    boxed,labels,info=letterbox(image,instance,128)
    assert boxed.shape==(128,128) and labels.shape==(128,128)
    assert abs(info["ratio"]-128/120)<1e-6
    x0,y0=37.5,22.0; xb=x0*info["scale_x"]+info["pad_x"]; yb=y0*info["scale_y"]+info["pad_y"]
    assert abs((xb-info["pad_x"])/info["scale_x"]-x0)<1e-6
    assert abs((yb-info["pad_y"])/info["scale_y"]-y0)<1e-6
    records=instance_records(labels); assert len(records)==3
    mask,center,scale_target,scale_weight=dense_targets(labels,128)
    assert center.max()==1.0 and scale_weight.max()==1.0 and mask.sum()>0
    assert np.all(scale_target[scale_weight>0]>=0)
    args=SimpleNamespace(nms_kernel=7,center_threshold=0.15,mask_gate=0.20,min_center_distance=4,max_proposals=10,image_size=128)
    output={"center":logit(center),"mask":logit(np.maximum(mask,0.01)),"scale":logit(np.clip(scale_target,0.01,0.99))}
    proposals=extract_proposals(output,args); matches=match_centers(proposals,records,0.60)
    assert len(proposals)==3 and len(matches)==3
    assert boundary_gap(records[0],records[1])>=0
    center2=np.full((128,128),0.01,np.float32); center2[10,10]=0.99; center2[70,70]=0.98
    output2={"center":logit(center2),"mask":logit(np.ones((128,128),np.float32)*0.9),"scale":logit(np.ones((128,128),np.float32)*0.1)}
    valid=np.zeros((128,128),bool); valid[40:110,40:110]=True
    kept=extract_proposals(output2,args,valid); assert len(kept)==1 and kept[0]["x"]==70
    ambiguous=[{"x":20.,"y":20.},{"x":21.,"y":20.}]
    truth=[{"x":20.,"y":20.,"scale":10.},{"x":30.,"y":20.,"scale":10.}]
    matched=match_centers(ambiguous,truth,0.6); assert len({g for _,g,_ in matched})==len(matched)
    assert decide({"x":True},limited=True)==(False,"sanity_only")
    model=CenterScaleUNet(4); batch={"mask":torch.zeros(1,1,32,32),"center":torch.zeros(1,1,32,32),"scale_target":torch.zeros(1,1,32,32),"scale_weight":torch.zeros(1,1,32,32)}; batch["center"][0,0,16,16]=1; batch["scale_weight"][0,0,16,16]=1
    train_args=SimpleNamespace(device="cpu",mask_loss_weight=0.35,scale_loss_weight=0.5); loss=training_loss(model(torch.rand(1,1,32,32)),batch,train_args); assert torch.isfinite(loss); loss.backward()
    print({"synthetic_proposals":len(proposals),"matches":len(matches),"ratio":info["ratio"]})
    print("R257 synthetic checks: PASS")


if __name__=="__main__": main()
