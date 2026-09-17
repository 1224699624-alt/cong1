#!/usr/bin/env python3
"""R336-RAM: state masks + logit PCGrad + selective teacher distillation."""
from __future__ import annotations

import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, seed_all
from train_r330_ram_pair_surface_volume_prior import choose_pair_instances
from train_r332_ram_joint_iterative_refinement import flatten, iterative_refine, checkpoint_gate
from train_r335_ram_conditional_state_loss import RamSeamDataset, masks, collect
from r336_gradient_surgery import gradient_surrogate, selective_binary_distillation


def loss_terms(student, teacher, target, seam):
    sep, overlap, uncertain = masks(target, seam)
    prob = torch.sigmoid(student.float()); union = prob.amax(1, keepdim=True)
    base = dice_bce_logits(student, target)
    sep_loss = (sep * F.relu(union - .10).square()).sum() / sep.sum().clamp_min(1)
    ov_pos = overlap.expand_as(target) * target
    ov_loss = (ov_pos * F.softplus(-student.float())).sum() / ov_pos.sum().clamp_min(1)
    keep = uncertain.expand_as(target)
    preserve, preserve_mass = selective_binary_distillation(student, teacher, target, keep, .80)
    return base, sep_loss, ov_loss, preserve, preserve_mass


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True)
    p.add_argument('--r332-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--steps',type=int,default=2);p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seam-weight',type=float,default=.025);p.add_argument('--overlap-weight',type=float,default=.04);p.add_argument('--distill-weight',type=float,default=.20)
    p.add_argument('--seed',type=int,default=3361);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    seed_all(a.seed);device=torch.device('cuda');payload=torch.load(a.r332_checkpoint,map_location=device,weights_only=False)
    network=NnUNetMultiLabelPrior().to(device);network.load_state_dict(payload['model'],strict=True);network.requires_grad_(False)
    student=InstanceCompletionRefiner().to(device);student.load_state_dict(payload['refiner'],strict=True)
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    train=RamSeamDataset(a.dataset_root,a.prior_root,'train',True);val=RamSeamDataset(a.dataset_root,a.prior_root,'val',False)
    if a.smoke:train.mask_files=train.mask_files[:2];val.mask_files=val.mask_files[:2]
    tl=DataLoader(train,batch_size=1,shuffle=True,num_workers=0 if a.smoke else 2,pin_memory=True,persistent_workers=not a.smoke);vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    opt=torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=1e-4);pairs=top_training_pairs(a.dataset_root,15)
    baseline=collect(network,teacher,vl,device,a.steps,pairs);base_flat=flatten(baseline['official']);history=a.output/'history.jsonl';best=None
    for epoch in range(1,(1 if a.smoke else a.epochs)+1):
        student.train(); sums=np.zeros(4); conflicts=np.zeros(2); preserve_mass=0.; started=time.time()
        for b in tl:
            image=b['image'].to(device);target=b['mask'].to(device);seam=b['seam'].to(device)
            with torch.no_grad():base_logits,_=network(image,False);idx=choose_pair_instances(torch.sigmoid(base_logits),target,4);t=iterative_refine(image,base_logits,teacher,a.steps,idx)[0][-1]
            s=iterative_refine(image,base_logits,student,a.steps,idx)[0][-1]
            base,sep,ov,keep,km=loss_terms(s,t,target,seam)
            sep_sur,ss=gradient_surrogate(s,sep,base,a.seam_weight);ov_sur,os=gradient_surrogate(s,ov,base,a.overlap_weight)
            loss=base+sep_sur+ov_sur+a.distill_weight*keep
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),8);opt.step()
            sums += [float(x.detach()) for x in (base,sep,ov,keep)];conflicts += [ss['conflict_fraction'],os['conflict_fraction']];preserve_mass+=km
        metrics=collect(network,student,vl,device,a.steps,pairs);final=flatten(metrics['official']);gate,checks=checkpoint_gate(final,base_flat)
        checks.update({'overlap_dsc_not_below_r332':final['overlap_dsc']>=base_flat['overlap_dsc'],'overlap_nsd_not_below_r332':final['overlap_nsd_2px']>=base_flat['overlap_nsd_2px'],'overlap_msd_not_above_r332':final['overlap_msd_px']<=base_flat['overlap_msd_px'],'seam_fp_not_above_r332':metrics['seam_fp']<=baseline['seam_fp']});gate=bool(gate and all(checks.values()))
        row={'epoch':epoch,'seconds':time.time()-started,'loss_terms':dict(zip(['seg','seam','overlap','selective_distill'],(sums/len(tl)).tolist())),'conflict_fraction':{'seam':conflicts[0]/len(tl),'overlap':conflicts[1]/len(tl)},'distill_mask_mass':preserve_mass/len(tl),'gate':gate,'checks':checks,'baseline':base_flat,'adapted':final,'seam_fp':{'baseline':baseline['seam_fp'],'adapted':metrics['seam_fp']}}
        with history.open('a') as f:f.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        score=(int(gate),min(final['overall_dsc']-base_flat['overall_dsc'],final['overlap_dsc']-base_flat['overlap_dsc']),final['overlap_nsd_2px']-base_flat['overlap_nsd_2px'],baseline['seam_fp']-metrics['seam_fp'])
        if best is None or score>best[0]:best=(score,row);torch.save({'refiner':student.state_dict(),'epoch':epoch,'row':row},a.output/'best.pth')
    result={'experiment':'R336_RAM_STATE_PCGRAD_SELECTIVE_DISTILL','split':'validation','test_used':False,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'best':best[1]};(a.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'complete':True,'smoke':a.smoke,'best_epoch':best[1]['epoch'],'gate':best[1]['gate']}))
if __name__=='__main__':main()
