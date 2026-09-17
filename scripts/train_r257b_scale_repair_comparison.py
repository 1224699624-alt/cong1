#!/usr/bin/env python3
"""R257b: frozen-center log-scale head versus prediction-derived basin scale."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset

from train_r257_image_only_center_scale_detector import (
    CenterDataset,
    CenterScaleUNet,
    boundary_gap,
    extract_proposals,
    instance_records,
    match_centers,
)


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--variant-root",type=Path,default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--r257-checkpoint",type=Path,default=Path("outputs/center_detector/r257_image_only_center_scale_fullval/center_scale_seed257_final.pt"))
    p.add_argument("--r257-result",type=Path,default=Path("outputs/analysis/r257_image_only_center_scale_fullval.json"))
    p.add_argument("--output-dir",type=Path,default=Path("outputs/scale_repair/r257b_scale_repair"))
    p.add_argument("--log-scale-checkpoint",type=Path,default=Path("outputs/scale_repair/r257b_scale_repair/log_scale_head_seed2571_final.pt"))
    p.add_argument("--result-json",type=Path,default=Path("outputs/analysis/r257b_scale_repair_result.json"))
    p.add_argument("--proposal-csv",type=Path,default=Path("outputs/analysis/r257b_scale_repair_val_proposals.csv"))
    p.add_argument("--image-size",type=int,default=384); p.add_argument("--base-channels",type=int,default=24)
    p.add_argument("--epochs",type=int,default=4); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=6)
    p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--basin-mask-threshold",type=float,default=0.50)
    p.add_argument("--center-threshold",type=float,default=0.15); p.add_argument("--mask-gate",type=float,default=0.20)
    p.add_argument("--nms-kernel",type=int,default=9); p.add_argument("--min-center-distance",type=int,default=5); p.add_argument("--max-proposals",type=int,default=48)
    p.add_argument("--match-tolerance-scale",type=float,default=0.60); p.add_argument("--relative-close-threshold",type=float,default=0.20)
    p.add_argument("--limit-train",type=int,default=0); p.add_argument("--limit-val",type=int,default=0)
    p.add_argument("--seed",type=int,default=2571); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def read_instance(path:Path)->np.ndarray:
    value=np.asarray(Image.open(path)); return (value[...,0] if value.ndim==3 else value).astype(np.int32)


def compute_log_scale_stats(root:Path,size:int,limit:int)->tuple[float,float,int]:
    paths=sorted((root/"train_labels").glob("*.png")); paths=paths[:limit] if limit>0 else paths; values=[]
    for path in paths:
        instance=read_instance(path); height,width=instance.shape; ratio=min(size/height,size/width); resized_h=max(1,int(round(height*ratio))); resized_w=max(1,int(round(width*ratio))); area_scale=(resized_h/height)*(resized_w/width)
        for value in [int(v) for v in np.unique(instance) if int(v)>0]:
            area=int((instance==value).sum())
            if area>=24: values.append(math.log(max(math.sqrt(area*area_scale),1.0)))
    if not values: raise RuntimeError("No train scales")
    return float(np.mean(values)),float(max(np.std(values),0.10)),len(values)


class LogScaleDataset(Dataset):
    def __init__(self,base:CenterDataset,mean:float,std:float): self.base,self.mean,self.std=base,mean,std
    def __len__(self)->int: return len(self.base)
    def __getitem__(self,index:int)->dict[str,torch.Tensor]:
        item=self.base.load(index); target=np.zeros((self.base.size,self.base.size),np.float32); weight=np.zeros_like(target)
        for record in instance_records(item["instance"],min_area=4):
            y=int(np.clip(round(record["y"]),0,self.base.size-1)); x=int(np.clip(round(record["x"]),0,self.base.size-1)); target[y,x]=(math.log(max(record["scale"],1.0))-self.mean)/self.std; weight[y,x]=1.0
        return {"image":torch.from_numpy(item["image"][None].astype(np.float32)/255.0),"target":torch.from_numpy(target[None]),"weight":torch.from_numpy(weight[None])}


def valid_region(info:dict[str,float],size:int)->np.ndarray:
    result=np.zeros((size,size),bool); width=int(round(info["original_width"]*info["scale_x"])); height=int(round(info["original_height"]*info["scale_y"])); x0,y0=int(info["pad_x"]),int(info["pad_y"]); result[y0:y0+height,x0:x0+width]=True; return result


def basin_scales(mask_probability:np.ndarray,proposals:list[dict[str,float]],valid:np.ndarray,threshold:float)->list[float]:
    if not proposals: return []
    foreground=(mask_probability>=threshold)&valid; yy,xx=np.nonzero(foreground)
    if len(xx)==0: return [1.0]*len(proposals)
    centers=np.asarray([[p["x"],p["y"]] for p in proposals]); _,owner=cKDTree(centers).query(np.stack([xx,yy],axis=1),k=1); counts=np.bincount(owner,minlength=len(proposals)); return [math.sqrt(max(float(count),1.0)) for count in counts]


def method_gate(metrics:dict[str,float|None])->dict[str,bool]:
    finite=lambda key: metrics[key] is not None and bool(np.isfinite(metrics[key]))
    return {"required_scale_metrics_finite":all(finite(k) for k in ("median_abs_log_scale_error","close_endpoint_median_abs_log_scale_error","contact_endpoint_median_abs_log_scale_error")),"global_scale_error_le_050":finite("median_abs_log_scale_error") and metrics["median_abs_log_scale_error"]<=0.50,"close_scale_error_le_060":finite("close_endpoint_median_abs_log_scale_error") and metrics["close_endpoint_median_abs_log_scale_error"]<=0.60,"contact_scale_error_le_060":finite("contact_endpoint_median_abs_log_scale_error") and metrics["contact_endpoint_median_abs_log_scale_error"]<=0.60}


@torch.no_grad()
def evaluate(model:nn.Module,dataset:CenterDataset,args:argparse.Namespace,mean:float,std:float)->tuple[dict[str,Any],dict[str,Any],dict[str,float],list[dict[str,Any]]]:
    model.eval(); common={"gt":0,"pred":0,"matched":0,"loc":[],"count_error":[],"close_total":0,"close_covered":0,"contact_total":0,"contact_covered":0}; errors={"log_scale":{"all":[],"close":[],"contact":[]},"basin_scale":{"all":[],"close":[],"contact":[]}}; rows=[]
    for index in range(len(dataset)):
        item=dataset.load(index); image=torch.from_numpy(item["image"][None,None].astype(np.float32)/255.0).to(args.device); output=model(image); valid=valid_region(item["info"],args.image_size); proposals=extract_proposals(output,args,valid)
        raw=output["scale"][0,0].detach().cpu().numpy(); mask_probability=torch.sigmoid(output["mask"])[0,0].cpu().numpy(); basin=basin_scales(mask_probability,proposals,valid,args.basin_mask_threshold)
        for i,p in enumerate(proposals): p["log_scale"]=float(math.exp(np.clip(raw[int(p["y"]),int(p["x"])]*std+mean,0.0,math.log(args.image_size)))); p["basin_scale"]=float(basin[i])
        info=item["info"]; gt_original=instance_records(item["original_instance"],min_area=24); gt=[]
        for record in gt_original:
            transformed=dict(record); transformed["x"]=record["x"]*info["scale_x"]+info["pad_x"]; transformed["y"]=record["y"]*info["scale_y"]+info["pad_y"]; transformed["scale"]=record["scale"]*math.sqrt(info["scale_x"]*info["scale_y"]); gt.append(transformed)
        matches=match_centers(proposals,gt,args.match_tolerance_scale); mapping={g:p for p,g,_ in matches}; common["gt"]+=len(gt); common["pred"]+=len(proposals); common["matched"]+=len(matches); common["count_error"].append(abs(len(proposals)-len(gt))); match_data={}
        for pred_idx,gt_idx,distance in matches:
            common["loc"].append(distance/max(gt[gt_idx]["scale"],1.0)); match_data[gt_idx]=pred_idx
            for method in errors: errors[method]["all"].append(abs(math.log(max(proposals[pred_idx][method],1e-6)/max(gt[gt_idx]["scale"],1e-6))))
        close_endpoints=set(); contact_endpoints=set()
        for i in range(len(gt_original)):
            for j in range(i+1,len(gt_original)):
                gap=boundary_gap(gt_original[i],gt_original[j]); local_scale=math.sqrt(0.5*(gt_original[i]["area"]+gt_original[j]["area"])); covered=i in mapping and j in mapping and mapping[i]!=mapping[j]
                if gap/max(local_scale,1.0)<=args.relative_close_threshold: common["close_total"]+=1; common["close_covered"]+=int(covered); close_endpoints.update((i,j))
                if gap<1.0: common["contact_total"]+=1; common["contact_covered"]+=int(covered); contact_endpoints.update((i,j))
        for gt_idx in close_endpoints&set(match_data):
            pred_idx=match_data[gt_idx]
            for method in errors: errors[method]["close"].append(abs(math.log(max(proposals[pred_idx][method],1e-6)/max(gt[gt_idx]["scale"],1e-6))))
        for gt_idx in contact_endpoints&set(match_data):
            pred_idx=match_data[gt_idx]
            for method in errors: errors[method]["contact"].append(abs(math.log(max(proposals[pred_idx][method],1e-6)/max(gt[gt_idx]["scale"],1e-6))))
        for rank,p in enumerate(proposals): rows.append({"stem":item["stem"],"rank":rank,"x_letterbox":p["x"],"y_letterbox":p["y"],"x_original":(p["x"]-info["pad_x"])/info["scale_x"],"y_original":(p["y"]-info["pad_y"])/info["scale_y"],"score":p["score"],"mask_probability":p["mask_probability"],"log_scale_letterbox":p["log_scale"],"basin_scale_letterbox":p["basin_scale"],"log_scale_original":p["log_scale"]/math.sqrt(info["scale_x"]*info["scale_y"]),"basin_scale_original":p["basin_scale"]/math.sqrt(info["scale_x"]*info["scale_y"])})
    shared={"num_images":float(len(dataset)),"num_gt_centers":float(common["gt"]),"num_proposals":float(common["pred"]),"num_matched":float(common["matched"]),"center_recall":common["matched"]/max(common["gt"],1),"center_precision":common["matched"]/max(common["pred"],1),"mean_normalized_localization_error":float(np.mean(common["loc"])) if common["loc"] else None,"proposal_count_mae":float(np.mean(common["count_error"])),"false_proposals_per_image":(common["pred"]-common["matched"])/max(len(dataset),1),"close_pair_count":float(common["close_total"]),"close_pair_coverage":common["close_covered"]/max(common["close_total"],1),"contact_pair_count":float(common["contact_total"]),"contact_pair_coverage":common["contact_covered"]/max(common["contact_total"],1)}
    method_metrics={}
    for method,data in errors.items(): method_metrics[method]={"matched_count":float(len(data["all"])),"median_abs_log_scale_error":float(np.median(data["all"])) if data["all"] else None,"close_matched_endpoint_count":float(len(data["close"])),"close_endpoint_median_abs_log_scale_error":float(np.median(data["close"])) if data["close"] else None,"contact_matched_endpoint_count":float(len(data["contact"])),"contact_endpoint_median_abs_log_scale_error":float(np.median(data["contact"])) if data["contact"] else None}
    return method_metrics["log_scale"],method_metrics["basin_scale"],shared,rows


def main()->None:
    args=parse_args(); set_seed(args.seed); joined=str(args.variant_root).lower()
    if "clean-test" in joined or "articular" in joined or args.variant_root.name!="TSRS_RSNA-Epiphysis_contrast_v1": raise RuntimeError("R257b permits contrast-v1 Epiphysis train/val only")
    if not args.r257_checkpoint.exists() or not args.r257_result.exists(): raise FileNotFoundError("Missing frozen R257 artifacts")
    limited=args.limit_train>0 or args.limit_val>0; train_base=CenterDataset(args.variant_root,"train",args.image_size,args.limit_train,True); val_ds=CenterDataset(args.variant_root,"val",args.image_size,args.limit_val,False)
    train_stems=[p.stem for p in train_base.labels]; val_stems=[p.stem for p in val_ds.labels]
    if set(train_stems)&set(val_stems) or len(train_stems)!=len(set(train_stems)) or len(val_stems)!=len(set(val_stems)): raise RuntimeError("Invalid split IDs")
    if not limited and (len(train_stems)!=875 or len(val_stems)!=96): raise RuntimeError("Full R257b requires 875/96")
    reference_payload=json.loads(args.r257_result.read_text(encoding="utf-8")); reference_config=reference_payload.get("config",{}); state=torch.load(args.r257_checkpoint,map_location="cpu",weights_only=False); config=state.get("config",{})
    protocol_keys=("image_size","base_channels","center_threshold","mask_gate","nms_kernel","min_center_distance","max_proposals","match_tolerance_scale","relative_close_threshold")
    protocol_values={key:getattr(args,key) for key in protocol_keys if hasattr(args,key)}
    protocol_match=reference_payload.get("run_id")=="R257_IMAGE_ONLY_CENTER_SCALE_DETECTOR" and int(reference_config.get("epochs",-1))==8 and int(config.get("epochs",-1))==int(reference_config.get("epochs",-2))==8 and int(config.get("seed",-1))==int(reference_config.get("seed",-2)) and Path(str(reference_payload.get("checkpoint",""))).as_posix()==args.r257_checkpoint.as_posix()
    for key,value in protocol_values.items(): protocol_match=protocol_match and key in reference_config and abs(float(reference_config[key])-float(value))<=1e-12 and key in config and abs(float(config[key])-float(reference_config[key]))<=1e-12
    if not protocol_match: raise RuntimeError("Frozen R257 checkpoint/result/proposal protocol mismatch")
    if not limited and (args.epochs!=4 or abs(args.basin_mask_threshold-0.50)>1e-12): raise RuntimeError("Full R257b requires 4 epochs and basin threshold 0.50")
    model=CenterScaleUNet(args.base_channels); model.load_state_dict(state["model"])
    for parameter in model.parameters(): parameter.requires_grad=False
    model.scale=nn.Conv2d(args.base_channels,1,1); nn.init.zeros_(model.scale.weight); nn.init.zeros_(model.scale.bias)
    for parameter in model.scale.parameters(): parameter.requires_grad=True
    model=model.to(args.device); mean,std,num_scales=compute_log_scale_stats(args.variant_root,args.image_size,args.limit_train); train_ds=LogScaleDataset(train_base,mean,std); loader=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,generator=torch.Generator().manual_seed(args.seed)); optimizer=torch.optim.AdamW(model.scale.parameters(),lr=args.lr,weight_decay=args.weight_decay); history=[]
    for epoch in range(1,args.epochs+1):
        model.eval(); losses=[]
        for batch in loader:
            image=batch["image"].to(args.device,non_blocking=True); target=batch["target"].to(args.device); weight=batch["weight"].to(args.device); optimizer.zero_grad(set_to_none=True); output=model(image); loss=(F.smooth_l1_loss(output["scale"],target,reduction="none")*weight).sum()/weight.sum().clamp_min(1.0); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        row={"epoch":epoch,"train_log_scale_loss":float(np.mean(losses))}; history.append(row); print(json.dumps(row),flush=True)
    args.output_dir.mkdir(parents=True,exist_ok=True); args.log_scale_checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"config":vars(args),"log_scale_mean":mean,"log_scale_std":std},args.log_scale_checkpoint); checkpoint_hash=sha256(args.log_scale_checkpoint)
    log_metrics,basin_metrics,shared,rows=evaluate(model,val_ds,args,mean,std); log_checks=method_gate(log_metrics); basin_checks=method_gate(basin_metrics); shared_checks={"all_images":shared["num_images"]==float(len(val_ds)) and (limited or shared["num_images"]==96),"center_recall_ge_085":shared["center_recall"]>=0.85,"center_precision_ge_075":shared["center_precision"]>=0.75,"close_coverage_ge_080":shared["close_pair_coverage"]>=0.80,"contact_coverage_ge_075":shared["contact_pair_count"]>0 and shared["contact_pair_coverage"]>=0.75}
    reference=reference_payload["final"]; invariant_keys=("num_gt_centers","num_proposals","num_matched","center_recall","center_precision","mean_normalized_localization_error","proposal_count_mae","false_proposals_per_image","close_pair_count","close_pair_coverage","contact_pair_count","contact_pair_coverage"); invariant_deltas={key:float(shared[key]-reference[key]) for key in invariant_keys}; center_invariance_exact=all(abs(value)<=1e-12 for value in invariant_deltas.values())
    shared_checks["frozen_r257_protocol_match"]=protocol_match; shared_checks["center_invariance_exact"]=center_invariance_exact
    log_pass=all(log_checks.values()) and all(shared_checks.values()); basin_pass=all(basin_checks.values()) and all(shared_checks.values())
    if limited: decision="sanity_only"; gate_pass=False
    elif log_pass and basin_pass: decision="allow_r258_with_both_scale_methods"; gate_pass=True
    elif log_pass: decision="allow_r258_with_log_scale"; gate_pass=True
    elif basin_pass: decision="allow_r258_with_basin_scale"; gate_pass=True
    else: decision="no_go_scale_repair_insufficient"; gate_pass=False
    payload={"run_id":"R257B_SCALE_REPAIR_COMPARISON","scope":"TSRS_RSNA-Epiphysis_contrast_v1 train/original-val only","clean_test_used":False,"config":vars(args),"frozen_r257_checkpoint":str(args.r257_checkpoint),"frozen_r257_checkpoint_sha256":sha256(args.r257_checkpoint),"log_scale_checkpoint":str(args.log_scale_checkpoint),"log_scale_checkpoint_sha256":checkpoint_hash,"log_scale_stats":{"mean":mean,"std":std,"num_instances":num_scales},"history":history,"shared_center_metrics":shared,"center_invariance_deltas":invariant_deltas,"r257_sigmoid_scale_reference":{k:reference[k] for k in ("median_abs_log_scale_error","close_endpoint_median_abs_log_scale_error","contact_endpoint_median_abs_log_scale_error")},"log_scale":log_metrics,"basin_scale":basin_metrics,"shared_checks":shared_checks,"log_scale_checks":log_checks,"basin_scale_checks":basin_checks,"gate_pass":gate_pass,"decision":decision}
    args.result_json.parent.mkdir(parents=True,exist_ok=True); args.result_json.write_text(json.dumps(payload,indent=2,default=str,allow_nan=False),encoding="utf-8"); args.proposal_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.proposal_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else ["stem"]); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"decision":decision,"shared":shared,"log_scale":log_metrics,"basin_scale":basin_metrics,"log_checks":log_checks,"basin_checks":basin_checks},indent=2),flush=True)


if __name__=="__main__": main()
