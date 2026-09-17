#!/usr/bin/env python3
"""Deterministic qualitative comparison on baseline-hard RAM validation cases."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, find_objects, label
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, per_instance_features


COLORS=np.asarray(plt.get_cmap("tab20").colors[:14])
NAMES=["MC1","MC2","MC3","MC4","MC5","Tr","Tz","Sca","Lu","Cap","Ham","Tri","Radius","Ulna"]


@torch.inference_mode()
def refined_logits(baseline, refiner, image):
    logits,_=baseline(image,False); probability=torch.sigmoid(logits); output=logits.clone()
    for start in range(0,14,3):
        indices=torch.arange(start,min(start+3,14),device=image.device)
        source=probability[:,indices];feature=per_instance_features(image,probability,indices,source)
        corrected,_=refiner(feature,logits[0,indices].unsqueeze(1));output[0,indices]=corrected[:,0]
    return logits,output


def overlay(gray,masks):
    rgb=np.repeat(gray[...,None],3,axis=2); weight=np.zeros(gray.shape+(1,),dtype=float); color=np.zeros_like(rgb)
    for k,mask in enumerate(masks):
        color+=mask[...,None]*COLORS[k];weight+=mask[...,None]
    active=weight[...,0]>0;color=np.divide(color,np.maximum(weight,1),where=np.ones_like(color,dtype=bool))
    rgb[active]=0.58*rgb[active]+0.42*color[active]
    for k,mask in enumerate(masks):
        boundary=mask ^ binary_erosion(mask);rgb[boundary]=COLORS[k]
    return np.clip(rgb,0,1)


def focus_box(target,baseline,margin=35):
    to=target.sum(0)>=2;po=baseline.sum(0)>=2;focus=to | (po ^ to)
    lab,n=label(focus)
    if n:
        scores=[]
        for idx,sl in enumerate(find_objects(lab),start=1):
            component=lab==idx;scores.append((int(((po^to)&component).sum())+int(component.sum())//5,sl))
        sl=max(scores,key=lambda x:x[0])[1];y0,y1=sl[0].start,sl[0].stop;x0,x1=sl[1].start,sl[1].stop
    else:
        union=target.any(0);ys,xs=np.where(union);y0,y1=ys.min(),ys.max()+1;x0,x1=xs.min(),xs.max()+1
    h,w=to.shape;return max(0,y0-margin),min(h,y1+margin),max(0,x0-margin),min(w,x1+margin)


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--baseline-checkpoint",type=Path,required=True);ap.add_argument("--r329-checkpoint",type=Path,required=True)
    ap.add_argument("--r330-checkpoint",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--num-cases",type=int,default=6);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda");dataset=NativeWristDataset(args.dataset_root,"val",augment=False)
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    baseline=NnUNetMultiLabelPrior().to(device);baseline.load_state_dict(torch.load(args.baseline_checkpoint,map_location=device,weights_only=False)["model"],strict=True);baseline.eval()
    r329=InstanceCompletionRefiner().to(device);r329.load_state_dict(torch.load(args.r329_checkpoint,map_location=device,weights_only=False)["refiner"],strict=True);r329.eval()
    r330=InstanceCompletionRefiner().to(device);r330.load_state_dict(torch.load(args.r330_checkpoint,map_location=device,weights_only=False)["refiner"],strict=True);r330.eval()
    rows=[]
    for batch in loader:
        image=batch["image"].to(device);h,w=map(int,batch["original_hw"][0]);case=batch["case"][0]
        base_logits,r329_logits=refined_logits(baseline,r329,image);_,r330_logits=refined_logits(baseline,r330,image)
        target=(batch["mask"][0,:,:h,:w]>0.5).numpy();base=(torch.sigmoid(base_logits)[0,:,:h,:w]>=0.5).cpu().numpy()
        p329=(torch.sigmoid(r329_logits)[0,:,:h,:w]>=0.5).cpu().numpy();p330=(torch.sigmoid(r330_logits)[0,:,:h,:w]>=0.5).cpu().numpy()
        to=target.sum(0)>=2;po=base.sum(0)>=2;inter=(to&po).sum();dsc=(2*inter)/(to.sum()+po.sum()+1e-8)
        rows.append({"case":case,"difficulty":float(1-dsc),"target":target,"baseline":base,"r329":p329,"r330":p330})
    selected=sorted(rows,key=lambda x:x["difficulty"],reverse=True)[:args.num_cases]
    manifest=[];contact=[]
    for item in selected:
        case=item["case"];path=args.dataset_root/"BoneSegmentation"/"images"/f"{case}.bmp"
        gray=np.asarray(Image.open(path).convert("L"),dtype=float)/255.0
        if case.endswith("_R"):gray=np.fliplr(gray)
        panels=[gray,overlay(gray,item["target"]),overlay(gray,item["baseline"]),overlay(gray,item["r329"]),overlay(gray,item["r330"])]
        y0,y1,x0,x1=focus_box(item["target"],item["baseline"]);contact.append([p[y0:y1,x0:x1] for p in panels])
        fig,axes=plt.subplots(2,5,figsize=(13,5.4));labels=["Image","Ground truth","R325","R329","R330"]
        for j,panel in enumerate(panels):
            axes[0,j].imshow(panel,cmap="gray" if j==0 else None);axes[0,j].set_title(labels[j],fontsize=10);axes[0,j].axis("off")
            axes[0,j].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color="#ffd43b",linewidth=1.2))
            axes[1,j].imshow(panel[y0:y1,x0:x1],cmap="gray" if j==0 else None);axes[1,j].axis("off")
        fig.tight_layout(pad=0.35);out=args.output/f"{case}_comparison.png";fig.savefig(out,dpi=300,bbox_inches="tight");plt.close(fig)
        manifest.append({"case":case,"baseline_overlap_difficulty":item["difficulty"],"roi":[y0,y1,x0,x1],"file":out.name})
    fig,axes=plt.subplots(len(contact),5,figsize=(12,2.1*len(contact)),squeeze=False)
    labels=["Image","Ground truth","R325","R329","R330"]
    for i,row in enumerate(contact):
        for j,panel in enumerate(row):
            axes[i,j].imshow(panel,cmap="gray" if j==0 else None);axes[i,j].axis("off")
            if i==0:axes[i,j].set_title(labels[j],fontsize=10)
            if j==0:axes[i,j].set_ylabel(selected[i]["case"],fontsize=8)
    fig.tight_layout(pad=0.25);fig.savefig(args.output/"r331_baseline_hard_contact_sheet.png",dpi=300,bbox_inches="tight");plt.close(fig)
    (args.output/"manifest.json").write_text(json.dumps({"selection":"six hardest validation cases by R325 overlap DSC only; not selected by R330 gain","cases":manifest,"colors":dict(zip(NAMES,COLORS.tolist()))},indent=2),encoding="utf-8")
    print(json.dumps({"visualizations":len(manifest),"output":str(args.output)},indent=2),flush=True)


if __name__=="__main__":main()
