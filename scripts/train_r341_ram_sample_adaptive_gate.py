#!/usr/bin/env python3
"""R341: sample-adaptive safe loss for a frozen RAM overlap prior.

The independently trained R339 prior is frozen. A small GT-free sample gate
learns whether its proposed correction is useful from image/prediction features.
During training, GT defines a utility target only; inference uses gate features
alone. The interaction loss receives a detached per-sample loss budget so it
cannot dominate the mature R325 segmentation objective. RAM test is never used.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset, sha256
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, image_gradient, per_instance_features, seed_all
from train_r332_ram_joint_iterative_refinement import flatten
from train_r338_ram_frozen_differentiable_prior import confidence_preserve, set_trainability


class SampleUtilityGate(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(8,24),nn.SiLU(),nn.Linear(24,12),nn.SiLU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-1.4)
    def forward(self,x):return self.net(x).flatten()


def refine_all(image, logits, prior):
    probability=torch.sigmoid(logits);pieces=[]
    for start in range(0,14,3):
        idx=torch.arange(start,min(start+3,14),device=logits.device);source=probability[:,idx]
        feature=per_instance_features(image,probability,idx,source)
        corrected,_=prior(feature,logits[0,idx].unsqueeze(1));pieces.append(corrected[:,0].unsqueeze(0))
    refined=torch.cat(pieces,1);return refined,refined-logits


def gate_features(image, logits, delta):
    p=torch.sigmoid(logits.detach());d=delta.detach();multiplicity=F.relu(p.sum(1,keepdim=True)-1.0).clamp(0,1)
    unc=(4*p*(1-p)).mean((1,2,3));grad=image_gradient(image).mean((1,2,3))
    return torch.stack((unc,multiplicity.mean((1,2,3)),d.abs().mean((1,2,3)),
                        F.relu(d).mean((1,2,3)),F.relu(-d).mean((1,2,3)),
                        p.mean((1,2,3)),(p>.9).float().mean((1,2,3)),grad),1)


@torch.no_grad()
def utility_target(logits, proposed, target):
    p0,p1=torch.sigmoid(logits),torch.sigmoid(proposed);region=F.max_pool2d((target.sum(1,keepdim=True)>=2).float(),11,1,5)
    denom=(region.sum()*target.shape[1]).clamp_min(1.0)
    e0=((p0-target).abs()*region).sum()/denom;e1=((p1-target).abs()*region).sum()/denom
    t_overlap=(target.sum(1,keepdim=True)>=2).float();m0=F.relu(p0.sum(1,keepdim=True)-1).clamp(0,1);m1=F.relu(p1.sum(1,keepdim=True)-1).clamp(0,1)
    v0=(m0.sum()-t_overlap.sum()).abs()/(t_overlap.sum()+32);v1=(m1.sum()-t_overlap.sum()).abs()/(t_overlap.sum()+32)
    score=(e0-e1)/0.002-F.relu(v1-v0)/0.01
    return torch.sigmoid(score).reshape(1),{"local_before":float(e0),"local_after":float(e1),"volume_before":float(v0),"volume_after":float(v1)}


@torch.inference_mode()
def collect(model,prior,gate,loader,device):
    model.eval();prior.eval();gate.eval();direct={};final={};gates=[]
    for batch in loader:
        image=batch["image"].to(device);target=(batch["mask"]>0.5).numpy()[0];logits,_=model(image,False)
        proposed,delta=refine_all(image,logits,prior);g=torch.sigmoid(gate(gate_features(image,logits,delta))).view(-1,1,1,1)
        corrected=logits+g*delta;case=str(batch["case"][0])
        direct[case]={"pred":(torch.sigmoid(logits)>=.5).cpu().numpy()[0],"target":target}
        final[case]={"pred":(torch.sigmoid(corrected)>=.5).cpu().numpy()[0],"target":target};gates.append(float(g))
    return direct,final,{"mean":sum(gates)/len(gates),"min":min(gates),"max":max(gates)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--dataset-root",type=Path,required=True)
    ap.add_argument("--baseline-checkpoint",type=Path,required=True);ap.add_argument("--prior-checkpoint",type=Path,required=True)
    ap.add_argument("--r325-cache",type=Path,required=True);ap.add_argument("--r332-result",type=Path)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--epochs",type=int,default=30)
    ap.add_argument("--warmup-epochs",type=int,default=4);ap.add_argument("--head-lr",type=float,default=2e-6)
    ap.add_argument("--decoder-lr",type=float,default=1e-6);ap.add_argument("--gate-lr",type=float,default=1e-4)
    ap.add_argument("--interaction-budget",type=float,default=.10);ap.add_argument("--preserve-weight",type=float,default=.05)
    ap.add_argument("--direct-tolerance",type=float,default=.0001);ap.add_argument("--seed",type=int,default=3411)
    ap.add_argument("--smoke",action="store_true");args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);seed_all(args.seed)
    if not torch.cuda.is_available():raise RuntimeError("R341 requires CUDA")
    device=torch.device("cuda");train=NativeWristDataset(args.dataset_root,"train",True);val=NativeWristDataset(args.dataset_root,"val",False)
    if args.smoke:train.mask_files=train.mask_files[:2];val.mask_files=val.mask_files[:2];args.epochs=1
    tl=DataLoader(train,1,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True);vl=DataLoader(val,1,False,num_workers=1,pin_memory=True)
    payload=torch.load(args.baseline_checkpoint,map_location=device,weights_only=False)
    model=NnUNetMultiLabelPrior().to(device);model.load_state_dict(payload["model"],strict=True)
    anchor=NnUNetMultiLabelPrior().to(device);anchor.load_state_dict(payload["model"],strict=True);anchor.eval()
    for p in anchor.parameters():p.requires_grad_(False)
    pp=torch.load(args.prior_checkpoint,map_location=device,weights_only=False);state=pp.get("module",pp.get("refiner"))
    prior=InstanceCompletionRefiner().to(device);prior.load_state_dict(state,strict=True);prior.eval()
    for p in prior.parameters():p.requires_grad_(False)
    gate=SampleUtilityGate().to(device);eventual=set_trainability(model,True);head=[];decoder=[]
    for n,p in model.named_parameters():
        if n in eventual:(head if n.startswith("backbone.decoder.seg_layers.4.") else decoder).append(p)
    opt=torch.optim.AdamW([{"params":head,"lr":args.head_lr},{"params":decoder,"lr":args.decoder_lr},{"params":gate.parameters(),"lr":args.gate_lr}],weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs);scaler=torch.cuda.amp.GradScaler(True)
    pairs=top_training_pairs(args.dataset_root,15);r325=flatten(json.loads(args.r325_cache.read_text()))
    r332=None
    if args.r332_result and args.r332_result.exists():r332=json.loads(args.r332_result.read_text()).get("flat_metrics_by_step",{}).get("step2")
    protocol={"experiment":"R341_RAM_SAMPLE_ADAPTIVE_GATE","test_used":False,"prior_frozen":True,"prior_sha256":sha256(args.prior_checkpoint),
              "sample_gate_gt_free_at_inference":True,"interaction_loss_budget":args.interaction_budget,"r325":r325,"r332":r332,
              "config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    (args.output/"protocol.json").write_text(json.dumps(protocol,indent=2))
    history=args.output/"history.jsonl";best=None;bestpath=args.output/"best.pth"
    for epoch in range(1,args.epochs+1):
        dec=epoch>args.warmup_epochs;set_trainability(model,dec);model.eval();model.backbone.decoder.seg_layers[4].train()
        if dec:model.backbone.decoder.stages[4].train();model.backbone.decoder.transpconvs[4].train()
        gate.train();sums={k:0. for k in ("loss","base","interaction","preserve","gate_loss","gate","safe_weight","utility")};start=time.time()
        for batch in tl:
            image=batch["image"].to(device);target=batch["mask"].to(device);opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):
                logits,_=model(image,False)
                with torch.no_grad():anchor_logits,_=anchor(image,False)
                proposed,delta=refine_all(image,logits,prior);features=gate_features(image,logits,delta);gate_logit=gate(features);g=torch.sigmoid(gate_logit)
                u,ustats=utility_target(logits.detach(),proposed.detach(),target)
                corrected=logits+g.view(-1,1,1,1)*delta;base=dice_bce_logits(logits,target);inter=dice_bce_logits(corrected,target)
                budget=torch.minimum(g.detach(),args.interaction_budget*base.detach()/(inter.detach()+1e-6))
                preserve=confidence_preserve(logits,anchor_logits,target);gloss=F.binary_cross_entropy_with_logits(gate_logit.float(),u.float())
                loss=base+budget*inter+args.preserve_weight*preserve+.05*gloss
            if not torch.isfinite(loss):raise RuntimeError({"epoch":epoch,"loss":float(loss)})
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
            torch.nn.utils.clip_grad_norm_(gate.parameters(),5.);scaler.step(opt);scaler.update()
            vals={"loss":loss,"base":base,"interaction":inter,"preserve":preserve,"gate_loss":gloss,"gate":g.mean(),"safe_weight":budget.mean(),"utility":u.mean()}
            for k,v in vals.items():sums[k]+=float(v.detach())
        sch.step();official=epoch%5==0 or epoch==args.epochs;flatm=None;gate_stats=None;passes=None
        if official:
            direct,final,gate_stats=collect(model,prior,gate,vl,device);flatm={"direct":flatten(official_metrics(direct,pairs)),"final":flatten(official_metrics(final,pairs))}
            f,d=flatm["final"],flatm["direct"]
            passes=(f["overall_dsc"]>=r325["overall_dsc"] and f["overall_iou"]>=r325["overall_iou"] and
                    d["overall_dsc"]>=r325["overall_dsc"]-args.direct_tolerance and d["overall_iou"]>=r325["overall_iou"]-args.direct_tolerance and
                    f["overlap_ravd"]<=r325["overlap_ravd"]+.003)
        row={"epoch":epoch,"seconds":time.time()-start,**{f"train_{k}":v/len(tl) for k,v in sums.items()},"official":official,
             "flat":flatm,"gate_stats":gate_stats,"passes":passes}
        with history.open("a") as f:f.write(json.dumps(row)+"\n");print(json.dumps(row),flush=True)
        if official:
            f=flatm["final"];score=(int(passes),f["overlap_nsd_2px"],f["overlap_dsc"],-f["overlap_msd_px"],-f["overlap_ravd"])
            if best is None or score>best[0]:best=(score,row);torch.save({"model":model.state_dict(),"gate":gate.state_dict(),"row":row},bestpath)
    result={**protocol,"best":best[1],"checkpoint":str(bestpath),"status":"complete"};(args.output/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({"status":"complete","best":best[1]},indent=2))

if __name__=="__main__":main()
