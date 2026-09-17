#!/usr/bin/env python3
"""R258 phase A: patient-disjoint OOF image-only center/basin proposals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

from train_r257_image_only_center_scale_detector import (
    CenterDataset,
    CenterScaleUNet,
    boundary_gap,
    instance_records,
    match_centers,
    training_loss,
)
from train_r257b_scale_repair_comparison import evaluate as evaluate_scale_methods


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--variant-root",type=Path,default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--metadata",type=Path,default=Path("outputs/metadata/r256/filtered_train.csv"))
    p.add_argument("--output-dir",type=Path,default=Path("outputs/oof_proposals/r258_center_basin_oof"))
    p.add_argument("--result-json",type=Path,default=Path("outputs/analysis/r258_oof_center_basin_result.json"))
    p.add_argument("--proposal-csv",type=Path,default=Path("outputs/analysis/r258_oof_center_basin_proposals.csv"))
    p.add_argument("--fold-manifest-csv",type=Path,default=Path("outputs/analysis/r258_oof_fold_manifest.csv"))
    p.add_argument("--inference-manifest-csv",type=Path,default=Path("outputs/analysis/r258_oof_inference_manifest.csv"))
    p.add_argument("--num-folds",type=int,default=5); p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--image-size",type=int,default=384); p.add_argument("--base-channels",type=int,default=24); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--workers",type=int,default=6)
    p.add_argument("--lr",type=float,default=5e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--mask-loss-weight",type=float,default=0.35); p.add_argument("--scale-loss-weight",type=float,default=0.50)
    p.add_argument("--center-threshold",type=float,default=0.15); p.add_argument("--mask-gate",type=float,default=0.20); p.add_argument("--nms-kernel",type=int,default=9); p.add_argument("--min-center-distance",type=int,default=5); p.add_argument("--max-proposals",type=int,default=48)
    p.add_argument("--match-tolerance-scale",type=float,default=0.60); p.add_argument("--relative-close-threshold",type=float,default=0.20); p.add_argument("--basin-mask-threshold",type=float,default=0.50)
    p.add_argument("--limit-cases",type=int,default=0); p.add_argument("--seed",type=int,default=258); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


REQUIRED_PROPOSAL_FIELDS={"stem","x_original","y_original","basin_scale_original","score","fold"}


def metadata_rows(path:Path)->dict[str,dict[str,Any]]:
    with path.open(newline="",encoding="utf-8-sig") as handle:
        return {str(row["id"]):{"boneage":float(row["boneage"]),"male":str(row["male"]).strip().lower() in {"1","true","yes"}} for row in csv.DictReader(handle)}


def make_folds(stems:list[str],metadata:dict[str,dict[str,Any]],num_folds:int,seed:int)->tuple[np.ndarray,bool]:
    strata=np.asarray([f"{int(metadata[s]['male'])}_{np.digitize(metadata[s]['boneage'],[60,96,132,168])}" for s in stems]); counts=Counter(strata.tolist()); stratified=min(counts.values())>=num_folds
    splitter=StratifiedKFold(num_folds,shuffle=True,random_state=seed) if stratified else KFold(num_folds,shuffle=True,random_state=seed); folds=np.full(len(stems),-1,np.int32)
    iterator=splitter.split(stems,strata) if stratified else splitter.split(stems)
    for fold,(_,heldout) in enumerate(iterator): folds[heldout]=fold
    if np.any(folds<0): raise RuntimeError("Incomplete fold assignment")
    return folds,stratified


def subset_dataset(root:Path,stems:set[str],size:int,augment:bool)->CenterDataset:
    dataset=CenterDataset(root,"train",size,0,augment); dataset.labels=[path for path in dataset.labels if path.stem in stems]
    if len(dataset)!=len(stems): raise RuntimeError("Dataset/manifest mismatch")
    return dataset


def train_fold(train_ds:CenterDataset,args:argparse.Namespace,fold:int,run_stamp:str)->tuple[CenterScaleUNet,dict[str,Any]]:
    fold_seed=args.seed+fold; set_seed(fold_seed); loader=DataLoader(train_ds,args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,generator=torch.Generator().manual_seed(fold_seed)); model=CenterScaleUNet(args.base_channels).to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); amp=args.device.startswith("cuda"); scaler=torch.amp.GradScaler("cuda",enabled=amp); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for batch in loader:
            image=batch["image"].to(args.device,non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",enabled=amp): output=model(image); loss=training_loss(output,batch,args)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite R258 loss at fold={fold}, epoch={epoch}")
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        row={"fold":fold,"epoch":epoch,"train_loss":float(np.mean(losses))}; history.append(row); print(json.dumps(row),flush=True)
    checkpoint=args.output_dir/f"fold{fold}_seed{fold_seed}_{run_stamp}.pt"; checkpoint.parent.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"config":vars(args),"fold":fold,"fold_seed":fold_seed,"history":history},checkpoint)
    return model,{"fold":fold,"seed":fold_seed,"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"history":history,"num_train":len(train_ds)}


def read_instance(path:Path)->np.ndarray:
    value=np.asarray(Image.open(path)); return (value[...,0] if value.ndim==3 else value).astype(np.int32)


def audit_oof(rows:list[dict[str,Any]],root:Path,stems:list[str],args:argparse.Namespace)->dict[str,Any]:
    grouped=defaultdict(list)
    for row in rows: grouped[str(row["stem"])].append(row)
    total_gt=total_pred=total_match=0; localization=[]; scale_errors=[]; close_errors=[]; contact_errors=[]; count_errors=[]; close_total=close_covered=contact_total=contact_covered=0; per_image=[]
    for stem in stems:
        instance=read_instance(root/"train_labels"/f"{stem}.png"); gt=instance_records(instance,min_area=24); proposals=[{"x":float(r["x_original"]),"y":float(r["y_original"]),"scale":float(r["basin_scale_original"])} for r in grouped[stem]]; matches=match_centers(proposals,gt,args.match_tolerance_scale); mapping={g:p for p,g,_ in matches}; total_gt+=len(gt); total_pred+=len(proposals); total_match+=len(matches); count_errors.append(abs(len(proposals)-len(gt))); match_error={}
        for pred_idx,gt_idx,distance in matches:
            loc=distance/max(gt[gt_idx]["scale"],1.0); err=abs(math.log(max(proposals[pred_idx]["scale"],1e-6)/max(gt[gt_idx]["scale"],1e-6))); localization.append(loc); scale_errors.append(err); match_error[gt_idx]=err
        image_close=image_close_covered=image_contact=image_contact_covered=0; close_endpoints=set(); contact_endpoints=set()
        for i in range(len(gt)):
            for j in range(i+1,len(gt)):
                gap=boundary_gap(gt[i],gt[j]); local_scale=math.sqrt(0.5*(gt[i]["area"]+gt[j]["area"])); covered=i in mapping and j in mapping and mapping[i]!=mapping[j]
                if gap/max(local_scale,1.0)<=args.relative_close_threshold: close_total+=1; image_close+=1; close_covered+=int(covered); image_close_covered+=int(covered); close_endpoints.update((i,j))
                if gap<1.0: contact_total+=1; image_contact+=1; contact_covered+=int(covered); image_contact_covered+=int(covered); contact_endpoints.update((i,j))
        close_errors.extend(match_error[i] for i in close_endpoints&set(match_error)); contact_errors.extend(match_error[i] for i in contact_endpoints&set(match_error)); per_image.append({"stem":stem,"num_gt":len(gt),"num_proposals":len(proposals),"num_matched":len(matches),"close_pairs":image_close,"close_covered":image_close_covered,"contact_pairs":image_contact,"contact_covered":image_contact_covered})
    return {"num_images":float(len(stems)),"num_unique_proposal_images":float(len(grouped)),"num_gt_centers":float(total_gt),"num_proposals":float(total_pred),"num_matched":float(total_match),"center_recall":total_match/max(total_gt,1),"center_precision":total_match/max(total_pred,1),"mean_normalized_localization_error":float(np.mean(localization)) if localization else None,"proposal_count_mae":float(np.mean(count_errors)),"false_proposals_per_image":(total_pred-total_match)/max(len(stems),1),"median_abs_log_basin_scale_error":float(np.median(scale_errors)) if scale_errors else None,"close_pair_count":float(close_total),"close_pair_coverage":close_covered/max(close_total,1),"close_endpoint_median_abs_log_basin_scale_error":float(np.median(close_errors)) if close_errors else None,"contact_pair_count":float(contact_total),"contact_pair_coverage":contact_covered/max(contact_total,1),"contact_endpoint_median_abs_log_basin_scale_error":float(np.median(contact_errors)) if contact_errors else None,"per_image":per_image}


def main()->None:
    args=parse_args(); set_seed(args.seed); joined=str(args.variant_root).lower()
    if "clean-test" in joined or "articular" in joined or args.variant_root.name!="TSRS_RSNA-Epiphysis_contrast_v1": raise RuntimeError("R258 permits contrast-v1 Epiphysis train only")
    metadata=metadata_rows(args.metadata); all_paths=sorted((args.variant_root/"train_labels").glob("*.png")); stems=[p.stem for p in all_paths]
    if set(stems)!=set(metadata): raise RuntimeError("R258 requires exact train metadata/label identity")
    limited=args.limit_cases>0
    if limited: stems=stems[:args.limit_cases]
    fixed_full={"num_folds":5,"epochs":8,"image_size":384,"base_channels":24,"batch_size":8,"lr":5e-4,"weight_decay":1e-4,"mask_loss_weight":0.35,"scale_loss_weight":0.50,"center_threshold":0.15,"mask_gate":0.20,"nms_kernel":9,"min_center_distance":5,"max_proposals":48,"match_tolerance_scale":0.60,"relative_close_threshold":0.20,"basin_mask_threshold":0.50,"seed":258}
    if not limited:
        if len(stems)!=875 or any(abs(float(getattr(args,key))-float(value))>1e-12 for key,value in fixed_full.items()): raise RuntimeError("Full R258 protocol mismatch")
    folds,stratified=make_folds(stems,metadata,args.num_folds,args.seed); run_stamp=datetime.now().strftime("%Y%m%d_%H%M%S"); manifest=[]
    for stem,fold in zip(stems,folds): manifest.append({"stem":stem,"fold":int(fold),"boneage":metadata[stem]["boneage"],"male":metadata[stem]["male"]})
    args.fold_manifest_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.fold_manifest_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    if not limited and not stratified: raise RuntimeError("Full R258 requires stratified folds")
    all_rows=[]; fold_results=[]; inference_manifest=[]
    for fold in range(args.num_folds):
        train_stems={stems[i] for i in range(len(stems)) if folds[i]!=fold}; heldout_stems={stems[i] for i in range(len(stems)) if folds[i]==fold}
        if train_stems&heldout_stems or train_stems|heldout_stems!=set(stems): raise RuntimeError("Fold leakage")
        train_ds=subset_dataset(args.variant_root,train_stems,args.image_size,True); heldout_ds=subset_dataset(args.variant_root,heldout_stems,args.image_size,False); model,fold_record=train_fold(train_ds,args,fold,run_stamp)
        _,basin_metrics,shared,proposal_rows=evaluate_scale_methods(model,heldout_ds,args,0.0,1.0)
        for row in proposal_rows: row["fold"]=fold
        if proposal_rows and not REQUIRED_PROPOSAL_FIELDS.issubset(proposal_rows[0]): raise RuntimeError("OOF proposal schema mismatch")
        counts=Counter(str(row["stem"]) for row in proposal_rows)
        for stem in sorted(heldout_stems): inference_manifest.append({"stem":stem,"fold":fold,"checkpoint":fold_record["checkpoint"],"checkpoint_sha256":fold_record["checkpoint_sha256"],"num_proposals":counts.get(stem,0)})
        all_rows.extend(proposal_rows); fold_record.update({"num_heldout":len(heldout_ds),"heldout_shared":shared,"heldout_basin_scale":basin_metrics}); fold_results.append(fold_record); del model; torch.cuda.empty_cache()
    proposal_images={str(row["stem"]) for row in all_rows}; manifest_counts=Counter(row["stem"] for row in manifest)
    inference_counts=Counter(row["stem"] for row in inference_manifest)
    if set(inference_counts)!=set(stems) or any(v!=1 for v in inference_counts.values()) or any(v!=1 for v in manifest_counts.values()): raise RuntimeError("OOF processing coverage failure")
    checkpoint_by_fold={row["fold"]:(row["checkpoint"],row["checkpoint_sha256"]) for row in fold_results}
    if any((row["checkpoint"],row["checkpoint_sha256"])!=checkpoint_by_fold[row["fold"]] for row in inference_manifest): raise RuntimeError("OOF checkpoint provenance mismatch")
    args.proposal_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.proposal_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(all_rows[0]) if all_rows else ["stem"]); writer.writeheader(); writer.writerows(all_rows)
    args.inference_manifest_csv.parent.mkdir(parents=True,exist_ok=True)
    with args.inference_manifest_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(inference_manifest[0])); writer.writeheader(); writer.writerows(inference_manifest)
    audit=audit_oof(all_rows,args.variant_root,stems,args); finite=lambda key:audit[key] is not None and bool(np.isfinite(audit[key])); heldout_sum=sum(int(row["heldout_shared"]["num_images"]) for row in fold_results); checks={"all_images_processed_once":audit["num_images"]==len(stems) and len(inference_manifest)==len(stems) and heldout_sum==len(stems) and all(v==1 for v in inference_counts.values()) and all(v==1 for v in manifest_counts.values()),"center_recall_ge_085":audit["center_recall"]>=0.85,"center_precision_ge_075":audit["center_precision"]>=0.75,"close_coverage_ge_080":audit["close_pair_coverage"]>=0.80,"contact_coverage_ge_075":audit["contact_pair_count"]>0 and audit["contact_pair_coverage"]>=0.75,"global_basin_scale_le_050":finite("median_abs_log_basin_scale_error") and audit["median_abs_log_basin_scale_error"]<=0.50,"close_basin_scale_le_060":finite("close_endpoint_median_abs_log_basin_scale_error") and audit["close_endpoint_median_abs_log_basin_scale_error"]<=0.60,"contact_basin_scale_le_060":finite("contact_endpoint_median_abs_log_basin_scale_error") and audit["contact_endpoint_median_abs_log_basin_scale_error"]<=0.60,"count_mae_le_3":audit["proposal_count_mae"]<=3.0,"false_proposals_le_3":audit["false_proposals_per_image"]<=3.0}; gate_pass=all(checks.values()) and not limited; decision="sanity_only" if limited else ("allow_r258_phase_b_relation_prior" if gate_pass else "no_go_oof_proposals_insufficient")
    artifact_hashes={"fold_manifest_sha256":sha256(args.fold_manifest_csv),"proposal_csv_sha256":sha256(args.proposal_csv),"inference_manifest_sha256":sha256(args.inference_manifest_csv)}
    payload={"run_id":"R258_PHASE_A_OOF_CENTER_BASIN_PROPOSALS","scope":"TSRS_RSNA-Epiphysis_contrast_v1 train OOF only","clean_test_used":False,"config":vars(args),"run_stamp":run_stamp,"fold_stratified":stratified,"fold_results":fold_results,"fold_manifest":str(args.fold_manifest_csv),"proposal_csv":str(args.proposal_csv),"inference_manifest":str(args.inference_manifest_csv),"artifact_hashes":artifact_hashes,"oof_audit":audit,"gate_checks":checks,"gate_pass":gate_pass,"decision":decision}; args.result_json.parent.mkdir(parents=True,exist_ok=True); args.result_json.write_text(json.dumps(payload,indent=2,default=str,allow_nan=False),encoding="utf-8"); print(json.dumps({"decision":decision,"oof_audit":{k:v for k,v in audit.items() if k!="per_image"},"gate_checks":checks,"artifact_hashes":artifact_hashes},indent=2),flush=True)


if __name__=="__main__": main()
