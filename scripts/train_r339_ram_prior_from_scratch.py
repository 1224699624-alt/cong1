#!/usr/bin/env python3
"""R339: train a RAM instance prior from random initialization to convergence.

No R329/R330/R332 prior weights are loaded.  A frozen R325 segmentation model
supplies real prediction residuals.  The shared per-instance prior and a
parameter-matched generic adapter are both trained from random initialization
for the same fixed 60-epoch budget with no patience stopping.  RAM test is
never loaded.  The run is resumable and emits a promotion gate for R340.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import (
    collect_baseline, collect_generic, collect_instance, official_metrics,
)
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset, sha256
from train_r327_ram_native_instance_completion_prior import (
    InstanceCompletionRefiner, completion_loss, per_instance_features, seed_all,
)
from train_r329_ram_instance_prior_vs_generic_adapter import (
    GenericMultiChannelAdapter, closest_generic_width, parameter_count,
)
from train_r330_ram_pair_surface_volume_prior import (
    choose_pair_instances, soft_surface_nsd_loss, volume_losses,
)


def flat(metrics: dict) -> dict[str, float]:
    overall, overlap, pair = (metrics["overall_instance_metrics"],
                              metrics["overlap_region_metrics"],
                              metrics["overlap_pair_intersection_metrics"])
    return {"overall_dsc": overall["dsc"], "overall_iou": 1.0-overall["voe"],
            "overall_nsd": overall["nsd_2px"], "overall_msd": overall["msd_px"],
            "overall_ravd": overall["ravd"], "overlap_dsc": overlap["dsc"],
            "overlap_iou": 1.0-overlap["voe"], "overlap_nsd": overlap["nsd_2px"],
            "overlap_msd": overlap["msd_px"], "overlap_ravd": overlap["ravd"],
            "pair_dsc": pair["dsc"], "pair_nsd": pair["nsd_2px"],
            "pair_msd": pair["msd_px"], "pair_ravd": pair["ravd"],
            "pair_fail": pair["msd_fail_rate"]}


def metric_key(value: dict, baseline: dict) -> tuple:
    passes = value["overall_dsc"] >= baseline["overall_dsc"] and value["overall_iou"] >= baseline["overall_iou"]
    return (int(passes), value["overlap_nsd"], value["overlap_dsc"],
            -value["overlap_msd"], value["overall_dsc"], -value["overall_ravd"])


def train_arm(name, module, baseline, train_set, val_loader, pairs, baseline_flat,
              output, epochs, lr, seed, generic=False, resume=True):
    device = next(module.parameters()).device
    optimizer = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loader = DataLoader(train_set, 1, shuffle=True, num_workers=2, pin_memory=True,
                        persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    history_path, best_path, last_path = (output/f"{name}_history.jsonl",
                                          output/f"{name}_best.pth", output/f"{name}_last.pth")
    start, best = 1, None
    if resume and last_path.exists():
        state=torch.load(last_path,map_location=device,weights_only=False)
        module.load_state_dict(state["module"],strict=True);optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]);scaler.load_state_dict(state["scaler"])
        start=int(state["epoch"])+1
        if best_path.exists(): best=torch.load(best_path,map_location="cpu",weights_only=False)["best"]
    audits=[]
    if history_path.exists():
        audits=[json.loads(x) for x in history_path.read_text(encoding="utf-8").splitlines()
                if x.strip() and json.loads(x).get("official") is not None]
    for epoch in range(start, epochs+1):
        module.train(); totals={k:0.0 for k in ("loss","base","surface","instance_volume","pair_volume")}; began=time.time()
        for batch in loader:
            image=batch["image"].to(device,non_blocking=True);target=batch["mask"].to(device,non_blocking=True)
            with torch.no_grad():
                base_logits,_=baseline(image,False);prob=torch.sigmoid(base_logits)
                indices=choose_pair_instances(prob,target,4);selected_target=target[:,indices]
                error=torch.abs(prob[:,indices]-selected_target)
                local=F.max_pool2d((error>0.15).float(),11,1,5)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                if generic:
                    corrected,delta=module(image,base_logits); selected_logits=corrected[:,indices]
                    selected_delta=delta[:,indices]
                else:
                    source=prob[:,indices];feature=per_instance_features(image,prob,indices,source)
                    value,selected_delta=module(feature,base_logits[0,indices].unsqueeze(1))
                    selected_logits=value[:,0].unsqueeze(0)
                base_loss,_=completion_loss(selected_logits[0].unsqueeze(1) if not generic else selected_logits,
                                             selected_target[0].unsqueeze(1) if not generic else selected_target,
                                             local[0].unsqueeze(1) if not generic else local,
                                             selected_delta,0.05)
                p=torch.sigmoid(selected_logits[0].unsqueeze(1).float())
                t=selected_target[0].unsqueeze(1)
                surface=soft_surface_nsd_loss(p,t);iv,pv=volume_losses(p,t)
                loss=base_loss+0.03*surface+0.005*iv+0.01*pv
            if not torch.isfinite(loss): raise RuntimeError({"arm":name,"epoch":epoch,"loss":float(loss)})
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(module.parameters(),12.0);scaler.step(optimizer);scaler.update()
            for k,v in (("loss",loss),("base",base_loss),("surface",surface),("instance_volume",iv),("pair_volume",pv)):
                totals[k]+=float(v.detach())
        scheduler.step(); official=None; flattened=None
        if epoch%5==0 or epoch==epochs:
            data=collect_generic(baseline,module,val_loader,device) if generic else collect_instance(baseline,module,val_loader,device)
            official=official_metrics(data,pairs);flattened=flat(official)
        row={"arm":name,"epoch":epoch,"seconds":time.time()-began,"lr":optimizer.param_groups[0]["lr"],
             **{f"train_{k}":v/len(loader) for k,v in totals.items()},"official":official,"flat":flattened}
        with history_path.open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
        print(json.dumps(row),flush=True)
        if flattened is not None:
            audits.append(row);key=metric_key(flattened,baseline_flat)
            if best is None or tuple(best["key"])<key:
                best={"key":list(key),"row":row};torch.save({"module":module.state_dict(),"best":best,
                                                               "random_initialization":True},best_path)
        torch.save({"module":module.state_dict(),"epoch":epoch,"optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict()},last_path)
    return best_path,best,audits


def plateau(audits: list[dict]) -> dict:
    recent=audits[-4:]
    if len(recent)<4:return {"mature":False,"reason":"fewer_than_four_late_audits"}
    vals=[r["flat"] for r in recent]
    spans={k:max(v[k] for v in vals)-min(v[k] for v in vals) for k in
           ("overall_dsc","overall_iou","overlap_dsc","overlap_nsd","overlap_msd")}
    mature=(spans["overall_dsc"]<=0.0005 and spans["overall_iou"]<=0.001 and
            spans["overlap_dsc"]<=0.004 and spans["overlap_nsd"]<=0.006 and spans["overlap_msd"]<=0.12)
    return {"mature":mature,"epochs":[r["epoch"] for r in recent],"spans":spans,
            "definition":"last four 5-epoch audits remain within predeclared metric spans"}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--baseline-checkpoint",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=60);ap.add_argument("--learning-rate",type=float,default=2e-4)
    ap.add_argument("--seed",type=int,default=3391);ap.add_argument("--smoke",action="store_true");args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);seed_all(args.seed)
    if not torch.cuda.is_available():raise RuntimeError("R339 requires CUDA")
    device=torch.device("cuda");train_set=NativeWristDataset(args.dataset_root,"train",augment=True)
    val_set=NativeWristDataset(args.dataset_root,"val",augment=False)
    if args.smoke:train_set.mask_files=train_set.mask_files[:2];val_set.mask_files=val_set.mask_files[:2];args.epochs=1
    val_loader=DataLoader(val_set,1,shuffle=False,num_workers=1,pin_memory=True)
    payload=torch.load(args.baseline_checkpoint,map_location=device,weights_only=False)
    baseline=NnUNetMultiLabelPrior().to(device);baseline.load_state_dict(payload["model"],strict=True);baseline.eval()
    for p in baseline.parameters():p.requires_grad_(False)
    pairs=top_training_pairs(args.dataset_root,15);base_official=official_metrics(collect_baseline(baseline,val_loader,device),pairs)
    base_flat=flat(base_official);instance=InstanceCompletionRefiner().to(device)
    width,generic_params,search=closest_generic_width(parameter_count(instance));generic=GenericMultiChannelAdapter(width).to(device)
    protocol={"experiment":"R339_RAM_PRIOR_FROM_SCRATCH","test_used":False,"epochs_fixed":args.epochs,
              "early_stopping":False,"all_prior_weights_randomly_initialized":True,
              "loaded_prior_checkpoint":None,"baseline_sha256":sha256(args.baseline_checkpoint),
              "parameter_matching":{"instance":parameter_count(instance),"generic":generic_params,"width":width,
                                    "relative_gap":(generic_params-parameter_count(instance))/parameter_count(instance)},
              "baseline":base_official,"config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    ipath,ibest,iaudits=train_arm("instance_prior",instance,baseline,train_set,val_loader,pairs,base_flat,args.output,
                                  args.epochs,args.learning_rate,args.seed,False,not args.smoke)
    gpath,gbest,gaudits=train_arm("generic_control",generic,baseline,train_set,val_loader,pairs,base_flat,args.output,
                                  args.epochs,args.learning_rate,args.seed,True,not args.smoke)
    if args.smoke:print(json.dumps({"smoke":True,"instance":ibest,"generic":gbest}));return
    iv=ibest["row"]["flat"];gv=gbest["row"]["flat"];maturity=plateau(iaudits)
    checks={"fixed_budget_complete":iaudits[-1]["epoch"]==args.epochs,
            "late_plateau":maturity["mature"],"overall_dsc_ge_baseline":iv["overall_dsc"]>=base_flat["overall_dsc"],
            "overall_iou_ge_baseline":iv["overall_iou"]>=base_flat["overall_iou"],
            "overlap_dsc_gt_baseline":iv["overlap_dsc"]>base_flat["overlap_dsc"],
            "overlap_nsd_gt_baseline":iv["overlap_nsd"]>base_flat["overlap_nsd"],
            "overlap_msd_lt_baseline":iv["overlap_msd"]<base_flat["overlap_msd"],
            "instance_composite_gt_generic":metric_key(iv,base_flat)>metric_key(gv,base_flat)}
    promote=all(checks.values())
    result={**protocol,"instance_best":ibest,"generic_best":gbest,"maturity":maturity,
            "promotion_checks":checks,"promote_to_r340":promote,
            "checkpoints":{"instance":str(ipath),"generic":str(gpath)},"status":"complete"}
    (args.output/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"status":"complete","promote_to_r340":promote,"checks":checks,"instance":iv,"generic":gv},indent=2))


if __name__=="__main__":main()
