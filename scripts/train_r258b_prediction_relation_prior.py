#!/usr/bin/env python3
"""R258 Phase B: relation prior from prediction-derived center/scale proposals.

Train proposals must be patient-disjoint OOF R258 predictions. Validation
proposals must be frozen full-train R257/R257b predictions. Ground truth is
used only after proposal generation to assign pair labels and heat targets.
"""

from __future__ import annotations

import argparse, csv, hashlib, json, math, random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from train_r256_scale_invariant_pair_prior import (Block, boundary_gap, find_image,
    gaussian, instance_records, make_heat_target, paired_case_bootstrap, read_metadata,
    set_seed, stable_seed)

EXPECTED_HASHES={
    "train_proposals":"23ba7f570e86b6a5807c50ca5097e0a8b0040fdf7f72f0748a794ec0f98c056e",
    "val_proposals":"5f17a815078ab7d275f01565cb82f84f6bd346b09b3d8e8e22e1f140c008b634",
    "fold_manifest":"7d2984059411c5a4e8a03c7445596a5d7be6693eacf9c868eb89cdee47559065",
    "inference_manifest":"c451833f0d783398dbb77a0cd9d234f0cc47c09ed46c9d067ad4b0b6fc596d17",
    "train_metadata":"bfb40b05d22a6b2151af3982b9de2e0dd2534b21c661df79037f33247d599857",
    "val_metadata":"d52cb2e32f055c315e47fe3e010ac7e55fbfc2018fe9419ff0da42848d706128",
}


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--variant-root",type=Path,default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--train-metadata",type=Path,default=Path("outputs/metadata/r256/filtered_train.csv"))
    p.add_argument("--val-metadata",type=Path,default=Path("outputs/metadata/r256/filtered_val.csv"))
    p.add_argument("--train-proposals",type=Path,default=Path("outputs/analysis/r258_oof_center_basin_full_proposals.csv"))
    p.add_argument("--val-proposals",type=Path,default=Path("outputs/analysis/r257b_scale_repair_fullval_proposals.csv"))
    p.add_argument("--phase-a-result",type=Path,default=Path("outputs/analysis/r258_oof_center_basin_full.json"))
    p.add_argument("--fold-manifest",type=Path,default=Path("outputs/analysis/r258_oof_center_basin_full_folds.csv"))
    p.add_argument("--inference-manifest",type=Path,default=Path("outputs/analysis/r258_oof_center_basin_full_inference.csv"))
    p.add_argument("--val-proposal-result",type=Path,default=Path("outputs/analysis/r257b_scale_repair_fullval.json"))
    p.add_argument("--output-dir",type=Path,default=Path("outputs/pair_prior/r258b_prediction_relation_prior"))
    p.add_argument("--result-json",type=Path,default=Path("outputs/analysis/r258b_prediction_relation_prior.json"))
    p.add_argument("--pair-manifest-csv",type=Path,default=Path("outputs/analysis/r258b_prediction_relation_prior_pairs.csv"))
    p.add_argument("--selection",type=Path,help="Optional reviewed train/val keep-list for an isolated clean refit")
    p.add_argument("--crop-size",type=int,default=128); p.add_argument("--epochs",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--workers",type=int,default=4)
    p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--crop-margin-scales",type=float,default=1.35)
    p.add_argument("--match-tolerance-scale",type=float,default=0.60)
    p.add_argument("--relative-pair-limit",type=float,default=0.80)
    p.add_argument("--proposal-distance-limit",type=float,default=4.0)
    p.add_argument("--relative-close-threshold",type=float,default=0.20)
    p.add_argument("--max-pairs-per-case",type=int,default=32)
    p.add_argument("--limit-train",type=int,default=0); p.add_argument("--limit-val",type=int,default=0)
    p.add_argument("--seed",type=int,default=2581); p.add_argument("--train-seeds",default="2581,2582,2583")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--r316c",action="store_true",help="Use the isolated R316C 256/128 dual-scale pair representation")
    p.add_argument("--r317",action="store_true",help="Run the isolated R317 convergence audit with per-epoch checkpoints")
    p.add_argument("--image-centers-only",action="store_true",
                   help="R317: remove age/sex conditioning and use the image+center branch as the final prior")
    p.add_argument("--reuse-completed-log",type=Path,action="append",
                   help="R317: reuse only fully completed seed histories/checkpoints from an earlier JSONL-style log")
    p.add_argument("--global-size",type=int,default=128)
    p.add_argument("--center-sigma-rel",type=float,default=.15)
    p.add_argument("--center-sigma-min",type=float,default=3.)
    p.add_argument("--center-sigma-max",type=float,default=12.)
    p.add_argument("--center-jitter-rel",type=float,default=.03)
    return p.parse_args()


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def validate_provenance(args:argparse.Namespace)->dict[str,Any]:
    paths={"train_proposals":args.train_proposals,"val_proposals":args.val_proposals,"fold_manifest":args.fold_manifest,
           "inference_manifest":args.inference_manifest,"train_metadata":args.train_metadata,"val_metadata":args.val_metadata}
    actual={key:sha256(path) for key,path in paths.items()}
    if actual!=EXPECTED_HASHES: raise RuntimeError(f"Pinned artifact hash mismatch: {actual}")
    phase=json.loads(args.phase_a_result.read_text(encoding="utf-8")); val_result=json.loads(args.val_proposal_result.read_text(encoding="utf-8"))
    if phase.get("decision")!="allow_r258_phase_b_relation_prior" or not phase.get("gate_pass"): raise RuntimeError("Phase A did not authorize Phase B")
    if phase.get("artifact_hashes",{}).get("proposal_csv_sha256")!=actual["train_proposals"] or phase.get("artifact_hashes",{}).get("fold_manifest_sha256")!=actual["fold_manifest"] or phase.get("artifact_hashes",{}).get("inference_manifest_sha256")!=actual["inference_manifest"]: raise RuntimeError("Phase A JSON/hash disagreement")
    if val_result.get("decision")!="allow_r258_with_both_scale_methods" or val_result.get("clean_test_used") is not False: raise RuntimeError("Frozen R257b result did not authorize basin proposals")
    def rows(path):
        with path.open(newline="",encoding="utf-8-sig") as f: return list(csv.DictReader(f))
    folds,inference,proposals=rows(args.fold_manifest),rows(args.inference_manifest),rows(args.train_proposals)
    if len(folds)!=875 or len(inference)!=875 or len({r["stem"] for r in folds})!=875 or len({r["stem"] for r in inference})!=875: raise RuntimeError("OOF exact-once manifest failure")
    fold_map={r["stem"]:int(r["fold"]) for r in folds}; inference_map={r["stem"]:int(r["fold"]) for r in inference}
    if fold_map!=inference_map or set(fold_map.values())!=set(range(5)): raise RuntimeError("OOF fold/inference mapping disagreement")
    if set(r["stem"] for r in proposals)!=set(fold_map) or any(int(r["fold"])!=fold_map[r["stem"]] for r in proposals): raise RuntimeError("Proposal stem/fold provenance failure")
    with args.val_proposals.open(newline="",encoding="utf-8-sig") as f: val_rows=list(csv.DictReader(f))
    if len({r["stem"] for r in val_rows})!=96: raise RuntimeError("Frozen validation proposals must cover 96 images")
    return {"hashes":actual,"phase_a_decision":phase["decision"],"val_proposal_decision":val_result["decision"],"train_exact_once":875,"val_unique_images":96,"folds":5}


def read_instance(path:Path)->np.ndarray:
    value=np.asarray(Image.open(path)); return (value[...,0] if value.ndim==3 else value).astype(np.int32)


@lru_cache(maxsize=64)
def read_gray(path:str)->np.ndarray:
    value=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    if value is None: raise ValueError(f"Cannot read {path}")
    return value


@lru_cache(maxsize=32)
def read_label(path:str)->np.ndarray: return read_instance(Path(path))


def load_proposals(path:Path,root:Path,split:str)->dict[str,list[dict[str,float]]]:
    grouped=defaultdict(list)
    with path.open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if not rows or "basin_scale_original" not in rows[0]: raise RuntimeError(f"Invalid proposal CSV: {path}")
    if split=="train" and "fold" not in rows[0]: raise RuntimeError("Train proposals must contain OOF fold provenance")
    for row in rows:
        stem=str(row["stem"])
        if row.get("x_original","") and row.get("y_original",""):
            x,y=float(row["x_original"]),float(row["y_original"])
        else:
            image=np.asarray(Image.open(find_image(root/split,stem)))
            height,width=image.shape[:2]; ratio=min(384/height,384/width)
            resized_h,resized_w=max(1,int(round(height*ratio))),max(1,int(round(width*ratio)))
            pad_y,pad_x=(384-resized_h)//2,(384-resized_w)//2
            x=(float(row["x_letterbox"])-pad_x)/(resized_w/width)
            y=(float(row["y_letterbox"])-pad_y)/(resized_h/height)
        grouped[stem].append({"x":x,"y":y,"scale":float(row["basin_scale_original"]),
                              "score":float(row.get("score",0.0)),"rank":float(row.get("rank",0)),
                              "fold":float(row.get("fold",-1))})
    return grouped


def assign_proposals(proposals:list[dict[str,float]],records:list[dict[str,Any]],tolerance:float)->list[int|None]:
    assigned=[]
    for p in proposals:
        candidates=[(math.hypot(p["x"]-r["cx"],p["y"]-r["cy"])/max(math.sqrt(r["area"]),1.0),i) for i,r in enumerate(records)]
        distance,index=min(candidates,default=(float("inf"),-1))
        assigned.append(index if distance<=tolerance else None)
    return assigned


def pair_geometry(a:dict[str,float],b:dict[str,float])->tuple[float,float]:
    distance=math.hypot(a["x"]-b["x"],a["y"]-b["y"])
    scale=math.sqrt(0.5*(a["scale"]**2+b["scale"]**2))
    return distance/max(scale,1.0),abs(math.log(max(a["scale"],1e-6)/max(b["scale"],1e-6)))


def build_samples(root:Path,split:str,metadata:dict[str,tuple[float,float]],proposal_path:Path,args:argparse.Namespace)->tuple[list[dict[str,Any]],dict[str,float]]:
    grouped=load_proposals(proposal_path,root,split); stems=sorted(metadata)
    limit=args.limit_train if split=="train" else args.limit_val
    if limit>0: stems=stems[:limit]
    if not set(stems)<=set(grouped): raise RuntimeError(f"Missing {split} proposal images: {sorted(set(stems)-set(grouped))[:5]}")
    samples=[]; audit=defaultdict(float)
    for stem in stems:
        label_path=root/f"{split}_labels"/f"{stem}.png"; instance=read_instance(label_path); records=instance_records(instance,min_area=24)
        proposals=grouped[stem]; assigned=assign_proposals(proposals,records,args.match_tolerance_scale); candidates=[]
        for i in range(len(proposals)):
            for j in range(i+1,len(proposals)):
                rel_dist,log_ratio=pair_geometry(proposals[i],proposals[j])
                if rel_dist>args.proposal_distance_limit: continue
                ai,aj=assigned[i],assigned[j]; positive=False; gap=-1.0; relative_gap=-1.0
                if ai is not None and aj is not None and ai!=aj:
                    gap=boundary_gap(records[ai],records[aj]); local_scale=math.sqrt(0.5*(records[ai]["area"]+records[aj]["area"])); relative_gap=gap/max(local_scale,1.0)
                    positive=relative_gap<=args.relative_pair_limit
                candidates.append({"i":i,"j":j,"positive":positive,"a":records[ai]["id"] if ai is not None else 0,
                    "b":records[aj]["id"] if aj is not None else 0,"gap":gap,"relative_gap":relative_gap,
                    "proposal_relative_distance":rel_dist,"proposal_abs_log_scale_ratio":log_ratio})
        positives=sorted((x for x in candidates if x["positive"]),key=lambda x:(x["relative_gap"],x["proposal_relative_distance"]))
        negatives=[x for x in candidates if not x["positive"]]
        cap=min(len(positives),len(negatives),args.max_pairs_per_case if args.max_pairs_per_case>0 else 10**9)
        if cap==0: audit["cases_without_balanced_pairs"]+=1; continue
        positives=positives[:cap]; available=list(range(len(negatives))); selected_neg=[]
        for pos in positives:
            idx=min(available,key=lambda k:(negatives[k]["proposal_relative_distance"]-pos["proposal_relative_distance"])**2+(negatives[k]["proposal_abs_log_scale_ratio"]-pos["proposal_abs_log_scale_ratio"])**2)
            selected_neg.append(negatives[idx]); available.remove(idx)
        image_path=find_image(root/split,stem); age,male=metadata[stem]
        for item in positives+selected_neg:
            pa,pb=proposals[item["i"]],proposals[item["j"]]
            samples.append({"split":split,"stem":stem,"image":str(image_path),"label":str(label_path),"age":age,"male":male,
                "pa":[pa["x"],pa["y"]],"pb":[pb["x"],pb["y"]],"psa":pa["scale"],"psb":pb["scale"],
                "rank_a":int(pa["rank"]),"rank_b":int(pb["rank"]),"fold_a":int(pa["fold"]),"fold_b":int(pb["fold"]),**item})
        audit["included_cases"]+=1; audit["positive_pairs"]+=cap; audit["negative_pairs"]+=cap
        audit["close_positive_pairs"]+=sum(x["relative_gap"]<=args.relative_close_threshold for x in positives)
        audit["contact_positive_pairs"]+=sum(x["gap"]<1.0 for x in positives)
    audit["num_images_with_proposals"]=len(set(stems)&set(grouped)); audit["num_expected_images"]=len(stems)
    audit["num_proposals"]=sum(len(grouped[s]) for s in stems)
    return samples,{k:float(v) for k,v in audit.items()}


class PairDataset(Dataset):
    def __init__(self,samples:list[dict[str,Any]],args:argparse.Namespace,augment:bool,use_development:bool):
        self.samples,self.args,self.augment,self.use_development=samples,args,augment,use_development
        cases=sorted({str(s["stem"]) for s in samples}); shift=max(1,len(cases)//2)
        meta={str(s["stem"]):(float(s["age"]),float(s["male"])) for s in samples}
        self.shuffled={c:meta[cases[(i+shift)%len(cases)]] for i,c in enumerate(cases)}
    def __len__(self): return len(self.samples)
    def __getitem__(self,index:int)->dict[str,Any]:
        s=self.samples[index]; image=read_gray(s["image"]); instance=read_label(s["label"])
        pa,pb=np.asarray(s["pa"],np.float64),np.asarray(s["pb"],np.float64); psa,psb=float(s["psa"]),float(s["psb"])
        center=0.5*(pa+pb); side=int(math.ceil(max(32.0,np.linalg.norm(pb-pa)+self.args.crop_margin_scales*(psa+psb))))
        x0,y0=int(round(center[0]-side/2)),int(round(center[1]-side/2)); x1,y1=x0+side,y0+side
        def crop(array):
            out=np.zeros((side,side),dtype=array.dtype); sx0,sy0,sx1,sy1=max(0,x0),max(0,y0),min(array.shape[1],x1),min(array.shape[0],y1)
            if sx1>sx0 and sy1>sy0: out[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=array[sy0:sy1,sx0:sx1]
            return out
        size=self.args.crop_size; image_crop=cv2.resize(crop(image),(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        scale_xy=size/side; ca=(pa-[x0,y0])*scale_xy; cb=(pb-[x0,y0])*scale_xy
        if self.args.r316c:
            from r316c_pair_inputs import pair_channels
            channels, _, _, _ = pair_channels(
                crop(image), ca, cb, .5 * (psa + psb) * scale_xy, size,
                self.args.global_size, self.args.center_sigma_rel,
                self.args.center_sigma_min, self.args.center_sigma_max,
                self.args.center_jitter_rel if self.augment else 0.0,
            )
        else:
            channels=np.stack([image_crop,gaussian(size,*ca),gaussian(size,*cb)]).astype(np.float32)
        target=np.zeros((size,size),np.float32); valid=False
        if s["positive"]:
            ma=crop(instance==int(s["a"])); mb=crop(instance==int(s["b"])); fg=crop(instance>0)
            target,valid=make_heat_target(ma,mb,fg,size,ca,cb,0.5*(psa+psb)*scale_xy)
        features=np.asarray([float(s["age"])/240.0,float(s["male"])],np.float32)
        if not self.use_development: features[:]=0
        shuffled=np.asarray([self.shuffled[str(s["stem"])][0]/240.0,self.shuffled[str(s["stem"])][1]],np.float32)
        if self.augment:
            if random.random()<0.5: channels,target=channels[:,:,::-1].copy(),target[:,::-1].copy()
            if random.random()<0.5: channels,target=channels[:,::-1,:].copy(),target[::-1,:].copy()
            turns=random.randrange(4)
            if turns: channels,target=np.rot90(channels,turns,axes=(1,2)).copy(),np.rot90(target,turns).copy()
            if random.random()<0.25: channels[0]=np.clip(channels[0]**random.uniform(.9,1.1),0,1)
        return {"image":torch.from_numpy(channels),"features":torch.from_numpy(features),"label":torch.tensor(float(s["positive"])),
            "heatmap":torch.from_numpy(target[None]),"heat_valid":torch.tensor(float(valid)),"close":torch.tensor(float(s["positive"] and s["relative_gap"]<=self.args.relative_close_threshold)),
            "contact":torch.tensor(float(s["positive"] and s["gap"]<1.0)),"case_id":str(s["stem"]),"shuffled_development":torch.from_numpy(shuffled)}


class RelationPriorNet(nn.Module):
    def __init__(self,input_channels:int=3):
        super().__init__(); self.pool=nn.MaxPool2d(2)
        self.e1,self.e2,self.e3,self.e4=Block(input_channels,16),Block(16,32),Block(32,64),Block(64,128)
        self.meta=nn.Sequential(nn.Linear(2,64),nn.SiLU(),nn.Linear(64,128))
        self.classifier=nn.Sequential(nn.Linear(256,64),nn.SiLU(),nn.Linear(64,1))
        self.u3,self.d3=nn.ConvTranspose2d(128,64,2,2),Block(128,64)
        self.u2,self.d2=nn.ConvTranspose2d(64,32,2,2),Block(64,32)
        self.u1,self.d1=nn.ConvTranspose2d(32,16,2,2),Block(32,16); self.heat=nn.Conv2d(16,1,1)
    def forward(self,image,features):
        e1=self.e1(image); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3)); meta=self.meta(features); conditioned=e4+meta[:,:,None,None]
        logit=self.classifier(torch.cat([F.adaptive_avg_pool2d(e4,1).flatten(1),meta],1)).squeeze(1)
        d3=self.d3(torch.cat([self.u3(conditioned),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1)); d1=self.d1(torch.cat([self.u1(d2),e1],1)); return logit,self.heat(d1)


def heat_loss(logits,target,label,valid):
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction="none").mean((1,2,3)); prob=torch.sigmoid(logits)
    dice=1-(2*(prob*target).sum((1,2,3))+1)/(prob.sum((1,2,3))+target.sum((1,2,3))+1)
    supervised=(label<.5)|(valid>.5); value=bce+torch.where((label>.5)&(valid>.5),dice,torch.zeros_like(dice))
    return value[supervised].mean() if supervised.any() else logits.sum()*0


@torch.no_grad()
def evaluate(model,loader,device,ablation="none"):
    model.eval(); labels=[]; scores=[]; close=[]; contact=[]; cases=[]; heat_dice=[]; positive_heat_count=0; valid_positive_heat_count=0
    for batch in loader:
        image=batch["image"].to(device); feat=batch["features"].to(device)
        if ablation=="zero_xray": image[:,0]=0
        elif ablation=="zero_centers": image[:,1:]=0
        elif ablation=="metadata_only": image[:]=0
        elif ablation=="shuffled_metadata": feat=batch["shuffled_development"].to(device)
        logit,heat=model(image,feat); target=batch["heatmap"].to(device); valid=batch["heat_valid"].numpy()>.5
        labels.extend(batch["label"].numpy()); scores.extend(torch.sigmoid(logit).cpu().numpy()); close.extend(batch["close"].numpy()); contact.extend(batch["contact"].numpy()); cases.extend(batch["case_id"])
        hp=torch.sigmoid(heat); dice=((2*(hp*target).sum((1,2,3))+1)/(hp.sum((1,2,3))+target.sum((1,2,3))+1)).cpu().numpy(); positive=batch["label"].numpy()>.5
        positive_heat_count+=int(positive.sum()); valid_positive_heat_count+=int((positive&valid).sum()); dice[positive&~valid]=0.0; heat_dice.extend(dice[positive])
    y,p=np.asarray(labels),np.asarray(scores); pred=p>=.5; close=np.asarray(close)>.5; contact=np.asarray(contact)>.5; negative=y<.5
    if len(np.unique(y))<2: raise RuntimeError("Evaluation requires both pair classes")
    maps={"accuracy":{},"auroc":{},"close_recall":{},"specificity":{}}
    case_array=np.asarray(cases)
    for case in sorted(set(cases)):
        m=case_array==case; cm=m&close; nm=m&negative; maps["accuracy"][case]=float((pred[m]==y[m]).mean()); maps["auroc"][case]=float(roc_auc_score(y[m],p[m])) if len(np.unique(y[m]))==2 else None
        maps["close_recall"][case]=float(pred[cm].mean()) if cm.any() else None; maps["specificity"][case]=float((~pred[nm]).mean()) if nm.any() else None
    return {"num_pairs":float(len(y)),"auroc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),"accuracy":float((pred==y).mean()),
        "positive_recall":float(pred[y>.5].mean()),"close_pair_count":float(close.sum()),"close_pair_recall":float(pred[close].mean()) if close.any() else 0.0,
        "contact_pair_count":float(contact.sum()),"contact_pair_recall":float(pred[contact].mean()) if contact.any() else 0.0,"negative_specificity":float((~pred[negative]).mean()),
        "positive_heatmap_soft_dice":float(np.mean(heat_dice)) if heat_dice else 0.0,"positive_heat_count":float(positive_heat_count),"valid_positive_heat_count":float(valid_positive_heat_count),
        "positive_heat_coverage":valid_positive_heat_count/max(positive_heat_count,1),"case_accuracy_by_id":maps["accuracy"],"case_auroc_by_id":maps["auroc"],
        "case_close_recall_by_id":maps["close_recall"],"case_specificity_by_id":maps["specificity"]}


def _r317_prior_epoch_key(row:dict[str,Any])->tuple[float,...]:
    """Rank only anatomically safe epochs, then favor localization quality."""
    safe=(row["close_pair_recall"]>=.75 and row["contact_pair_count"]>0 and
          row["contact_pair_recall"]>=.70 and row["negative_specificity"]>=.75 and
          row["positive_heat_coverage"]>=.85)
    return (float(safe),row["positive_heatmap_soft_dice"],row["auroc"],
            row["average_precision"],row["negative_specificity"],-float(row["epoch"]))


def train_variant(name,use_development,train_samples,val_samples,args,seed):
    set_seed(seed); train=DataLoader(PairDataset(train_samples,args,True,use_development),args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    val=DataLoader(PairDataset(val_samples,args,False,use_development),args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=RelationPriorNet(4 if args.r316c else 3).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); amp=args.device.startswith("cuda"); scaler=torch.cuda.amp.GradScaler(enabled=amp); history=[]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.reuse_completed_log:
        if not args.r317: raise RuntimeError("Completed-run reuse is restricted to R317")
        by_epoch={}
        for source_log in args.reuse_completed_log:
            if not source_log.is_file(): raise RuntimeError(f"Missing reuse log: {source_log}")
            for line in source_log.read_text(encoding="utf-8",errors="replace").splitlines():
                try: row=json.loads(line)
                except json.JSONDecodeError: continue
                if row.get("variant")==name and int(row.get("seed",-1))==seed and "epoch" in row:
                    # Printed rows add provenance fields that are intentionally absent
                    # from the in-memory history schema. Normalize before aggregation.
                    by_epoch[int(row["epoch"])]={k:v for k,v in row.items() if k not in {"variant","seed"}}
        complete=set(by_epoch)==set(range(1,args.epochs+1))
        selected_file=args.output_dir/f"{name}_seed{seed}_selected.pt"
        epoch_files=[args.output_dir/f"{name}_seed{seed}_epoch{epoch:02d}.pt" for epoch in range(1,args.epochs+1)]
        if complete and selected_file.is_file() and all(path.is_file() for path in epoch_files):
            history=[by_epoch[epoch] for epoch in range(1,args.epochs+1)]
            print(json.dumps({"variant":name,"seed":seed,"reused_complete_epochs":args.epochs,
                              "source_logs":[str(path) for path in args.reuse_completed_log]}),flush=True)
        else:
            # An AdamW continuation without optimizer state is not equivalent to the pinned
            # protocol. Discard only this incomplete seed and retrain it from epoch 1.
            for path in args.output_dir.glob(f"{name}_seed{seed}_*.pt"): path.unlink()
            print(json.dumps({"variant":name,"seed":seed,"reuse_rejected":"incomplete_seed_retrained"}),flush=True)
    for epoch in range(len(history)+1,args.epochs+1):
        model.train(); losses=[]
        for batch in train:
            image=batch["image"].to(args.device); feat=batch["features"].to(args.device); label=batch["label"].to(args.device); target=batch["heatmap"].to(args.device); valid=batch["heat_valid"].to(args.device); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp): logit,heat=model(image,feat); loss=F.binary_cross_entropy_with_logits(logit,label)+.35*heat_loss(heat,target,label,valid)
            if not bool(torch.isfinite(loss).item()): raise RuntimeError(f"Non-finite loss: {name}, seed={seed}, epoch={epoch}")
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        metrics=evaluate(model,val,args.device); history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),**metrics}); print(json.dumps({"variant":name,"seed":seed,**history[-1]},default=str),flush=True)
        if args.r317:
            torch.save({"model":model.state_dict(),"config":vars(args),"seed":seed,
                        "use_development":use_development,"epoch":epoch},
                       args.output_dir/f"{name}_seed{seed}_epoch{epoch:02d}.pt")
    selected=max(history,key=_r317_prior_epoch_key) if args.r317 else history[-1]
    if args.r317:
        selected_path=args.output_dir/f"{name}_seed{seed}_epoch{int(selected['epoch']):02d}.pt"
        selected_payload=torch.load(selected_path,map_location=args.device,weights_only=False)
        model.load_state_dict(selected_payload["model"])
    ablations={m:evaluate(model,val,args.device,m) for m in ("zero_xray","zero_centers","metadata_only","shuffled_metadata")}
    suffix="selected" if args.r317 else "final"; checkpoint=args.output_dir/f"{name}_seed{seed}_{suffix}.pt"
    torch.save({"model":model.state_dict(),"config":vars(args),"seed":seed,"use_development":use_development,
                "selected_epoch":int(selected["epoch"])},checkpoint)
    return {"name":name,"seed":seed,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),
            "history":history,"selected_epoch":int(selected["epoch"]),"final":selected,"ablations":ablations}


def aggregate(runs):
    numeric=[k for k,v in runs[0]["final"].items() if isinstance(v,(int,float))]; final={k:float(np.mean([r["final"][k] for r in runs])) for k in numeric}
    for key in ("case_accuracy_by_id","case_auroc_by_id","case_close_recall_by_id","case_specificity_by_id"):
        common=set.intersection(*[set(r["final"][key]) for r in runs]); final[key]={}
        for c in sorted(common):
            values=[r["final"][key][c] for r in runs if r["final"][key][c] is not None]; final[key][c]=float(np.mean(values)) if values else None
    ablations={}
    for mode in runs[0]["ablations"]:
        keys=[k for k,v in runs[0]["ablations"][mode].items() if isinstance(v,(int,float))]; ablations[mode]={k:float(np.mean([r["ablations"][mode][k] for r in runs])) for k in keys}
    return {"runs":runs,"final":final,"ablations":ablations}


def strict(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:strict(v) for k,v in value.items()}
    if isinstance(value,list): return [strict(v) for v in value]
    return value


def defined_scalars_finite(value)->bool:
    if value is None: return True
    if isinstance(value,(int,float,np.integer,np.floating)): return bool(np.isfinite(value))
    if isinstance(value,dict): return all(defined_scalars_finite(v) for v in value.values())
    if isinstance(value,(list,tuple)): return all(defined_scalars_finite(v) for v in value)
    return True


def main():
    args=parse_args(); set_seed(args.seed); joined=" ".join(map(str,[args.variant_root,args.train_proposals,args.val_proposals])).lower()
    if "clean-test" in joined or "articular" in joined or args.variant_root.name!="TSRS_RSNA-Epiphysis_contrast_v1": raise RuntimeError("R258B permits contrast-v1 Epiphysis train/val only")
    provenance=validate_provenance(args)
    train_meta,val_meta=read_metadata(args.train_metadata),read_metadata(args.val_metadata)
    clean_refit=args.selection is not None
    if clean_refit:
        selection=json.loads(args.selection.read_text(encoding="utf-8"))
        keep=selection.get("keep",selection)
        keep_train,keep_val=set(map(str,keep["train"])),set(map(str,keep["val"]))
        if (len(keep_train),len(keep_val))!=(814,94): raise RuntimeError("Clean refit requires the frozen 814/94 keep-list")
        if not keep_train<=set(train_meta) or not keep_val<=set(val_meta): raise RuntimeError("Clean keep-list is not covered by metadata")
        train_meta={k:v for k,v in train_meta.items() if k in keep_train}
        val_meta={k:v for k,v in val_meta.items() if k in keep_val}
        provenance["clean_selection"]={"path":str(args.selection),"sha256":sha256(args.selection),"train":814,"val":94}
    if set(train_meta)&set(val_meta): raise RuntimeError("Train/val identity overlap")
    limited=args.limit_train>0 or args.limit_val>0
    if args.r317 and not args.r316c: raise RuntimeError("R317 requires the R316C dual-scale representation")
    if args.image_centers_only and not args.r317: raise RuntimeError("--image-centers-only is restricted to R317")
    fixed=({"crop_size":256,"epochs":18,"batch_size":16,"workers":6,"lr":3e-4,"weight_decay":1e-4,"crop_margin_scales":1.35,"match_tolerance_scale":.60,"relative_pair_limit":.80,"proposal_distance_limit":4.0,"relative_close_threshold":.20,"max_pairs_per_case":32,"seed":2581,
            "global_size":128,"center_sigma_rel":.15,"center_sigma_min":3.,"center_sigma_max":12.,"center_jitter_rel":.03}
           if args.r317 else
           {"crop_size":256,"epochs":6,"batch_size":16,"workers":6,"lr":3e-4,"weight_decay":1e-4,"crop_margin_scales":1.35,"match_tolerance_scale":.60,"relative_pair_limit":.80,"proposal_distance_limit":4.0,"relative_close_threshold":.20,"max_pairs_per_case":32,"seed":2581,
            "global_size":128,"center_sigma_rel":.15,"center_sigma_min":3.,"center_sigma_max":12.,"center_jitter_rel":.03}
           if args.r316c else
           {"crop_size":128,"epochs":4,"batch_size":32,"workers":6,"lr":5e-4,"weight_decay":1e-4,"crop_margin_scales":1.35,"match_tolerance_scale":.60,"relative_pair_limit":.80,"proposal_distance_limit":4.0,"relative_close_threshold":.20,"max_pairs_per_case":32,"seed":2581})
    if not limited:
        expected_counts=(814,94) if clean_refit else (875,96)
        if (len(train_meta),len(val_meta))!=expected_counts: raise RuntimeError(f"R258B metadata rows mismatch: {(len(train_meta),len(val_meta))} != {expected_counts}")
        checked_fixed={k:v for k,v in fixed.items() if not (clean_refit and k=="workers")}
        if any(abs(float(getattr(args,k))-float(v))>1e-12 for k,v in checked_fixed.items()): raise RuntimeError("Full R258B fixed protocol mismatch")
    train_samples,train_audit=build_samples(args.variant_root,"train",train_meta,args.train_proposals,args); val_samples,val_audit=build_samples(args.variant_root,"val",val_meta,args.val_proposals,args)
    if not train_samples or not val_samples: raise RuntimeError("No balanced prediction-derived pairs")
    args.pair_manifest_csv.parent.mkdir(parents=True,exist_ok=True)
    fields=["split","stem","positive","a","b","gap","relative_gap","pa","pb","psa","psb","rank_a","rank_b","fold_a","fold_b"]
    with args.pair_manifest_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:s[k] for k in fields} for s in train_samples+val_samples)
    seeds=[int(x) for x in args.train_seeds.split(",") if x.strip()]
    if not limited and seeds!=[2581,2582,2583]: raise RuntimeError("Full R258B requires exact unique seeds 2581,2582,2583")
    baseline=aggregate([train_variant("image_centers",False,train_samples,val_samples,args,s) for s in seeds]); b=baseline["final"]
    if args.image_centers_only:
        development=None; bootstrap=None
        checks={"all_defined_scalars_finite":defined_scalars_finite(baseline),
            "close_recall_ge_075":b["close_pair_recall"]>=.75,
            "contact_recall_ge_070":b["contact_pair_count"]>0 and b["contact_pair_recall"]>=.70,
            "specificity_ge_075":b["negative_specificity"]>=.75,
            "heat_coverage_ge_085":b["positive_heat_coverage"]>=.85,
            "heat_dice_ge_020_including_invalid_as_zero":b["positive_heatmap_soft_dice"]>=.20,
            "exact_three_seeds":seeds==[2581,2582,2583]}
    else:
        development=aggregate([train_variant("image_centers_age_sex",True,train_samples,val_samples,args,s) for s in seeds]); d=development["final"]; abl=development["ablations"]
        bootstrap={"auroc":paired_case_bootstrap(b["case_auroc_by_id"],d["case_auroc_by_id"],args.seed),"close_recall":paired_case_bootstrap(b["case_close_recall_by_id"],d["case_close_recall_by_id"],args.seed+1),"specificity":paired_case_bootstrap(b["case_specificity_by_id"],d["case_specificity_by_id"],args.seed+2)}
        checks={"all_defined_scalars_finite":defined_scalars_finite({"baseline":baseline,"development":development,"bootstrap":bootstrap}),
            "close_recall_ge_075":d["close_pair_recall"]>=.75,"contact_recall_ge_070":d["contact_pair_count"]>0 and d["contact_pair_recall"]>=.70,"specificity_ge_075":d["negative_specificity"]>=.75,
            "heat_coverage_ge_085":d["positive_heat_coverage"]>=.85,"heat_dice_ge_020_including_invalid_as_zero":d["positive_heatmap_soft_dice"]>=.20,"development_adds_value":d["auroc"]-b["auroc"]>=.005 or d["close_pair_recall"]-b["close_pair_recall"]>=.02,
            "specificity_noninferior":d["negative_specificity"]>=b["negative_specificity"]-.02,"metadata_only_not_predictive":abl["metadata_only"]["auroc"]<=.60,
            "real_metadata_beats_shuffled":d["auroc"]>=abl["shuffled_metadata"]["auroc"]+.003,"exact_three_seeds":seeds==[2581,2582,2583]}
    passed=all(checks.values()) and not limited; decision="sanity_only" if limited else ("allow_controlled_prior_integration" if passed else "no_go_relation_prior_not_validated")
    if clean_refit: decision="clean_refit_complete_gate_pass" if passed else "clean_refit_complete_gate_no_go"
    selected_branch="baseline" if args.image_centers_only else "development_conditioned"
    payload={"run_id":"R317_IMAGE_CENTERS_ONLY_CONVERGENCE" if args.image_centers_only else ("R317_CONTINUOUS_PRIOR_CONVERGENCE" if args.r317 else ("R258B_PREDICTION_RELATION_PRIOR_CLEAN" if clean_refit else "R258B_PREDICTION_RELATION_PRIOR")),"scope":"TSRS_RSNA-Epiphysis_contrast_v1 clean 814/94 train/original-val" if clean_refit else "TSRS_RSNA-Epiphysis_contrast_v1 original train/val only","clean_test_used":False,"config":vars(args),
        "provenance":provenance,"pair_manifest_sha256":sha256(args.pair_manifest_csv),"train_pair_audit":train_audit,"val_pair_audit":val_audit,
        "selected_prior_branch":selected_branch,"baseline":baseline,"development_conditioned":development,
        "delta_development_vs_baseline":{} if development is None else {k:d[k]-b[k] for k in d if isinstance(d[k],(int,float)) and k in b},
        "paired_case_bootstrap":bootstrap,"gate_checks":checks,"gate_pass":passed,"decision":decision}
    args.result_json.parent.mkdir(parents=True,exist_ok=True); args.result_json.write_text(json.dumps(strict(payload),indent=2,default=str,allow_nan=False),encoding="utf-8")
    print(json.dumps({"decision":decision,"train_pair_audit":train_audit,"val_pair_audit":val_audit,
                      "selected_prior_branch":selected_branch,"baseline":b,
                      "development":None if development is None else development["final"],
                      "gate_checks":checks},indent=2,default=str),flush=True)


if __name__=="__main__": main()
