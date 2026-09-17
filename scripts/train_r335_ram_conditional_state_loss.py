#!/usr/bin/env python3
"""R335-RAM: loss-only conditional interaction fine-tuning.

No new inference module is introduced. The mature R325 backbone is frozen and
the existing R332 refiner is fine-tuned with mutually exclusive seam, overlap,
and uncertain-state losses. RAM test is never opened.
"""
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
from conflict_aware_loss import conflict_aware_loss


def conditional_loss_v346(student, teacher, target, seam, weights):
    """R346 routing: overlap wins over seam; teacher only preserves uncertain pixels."""
    base = dice_bce_logits(student, target)
    terms = conflict_aware_loss(student, target, teacher, seam, base, mode="ram",
                                seam_weight=weights[1], overlap_weight=weights[2],
                                preserve_weight=weights[3])
    # Keep the configured segmentation coefficient consistent with the R335
    # branch. The previous R346 call passed an unscaled base loss, making the
    # requested seg_weight=.10 silently behave as 1.0.
    total = weights[0] * terms.base + weights[1] * terms.seam + weights[2] * terms.overlap + weights[3] * terms.preserve
    terms.total = total
    return terms.total, (terms.base, terms.seam, terms.overlap, terms.preserve), {
        "sep_mass": float(terms.seam_gate_mass), "overlap_mass": float(terms.overlap_gate_mass),
        "uncertain_mass": float(terms.uncertain_mass)}


class RamSeamDataset(NativeWristDataset):
    def __init__(self, root, prior_root, split, augment):
        super().__init__(root, split, augment); self.prior_root=Path(prior_root)
    def __getitem__(self,index):
        item=super().__getitem__(index); case=item['case']; npy=self.prior_root/self.split/f'{case}.npy'
        if npy.exists(): value=torch.from_numpy(np.load(npy).astype(np.float32))[None,None]
        else:
            from PIL import Image
            value=torch.from_numpy(np.asarray(Image.open(self.prior_root/self.split/f'{case}.png'),dtype=np.float32)/255.)[None,None]
        h,w=item['original_hw'].tolist(); value=F.interpolate(value,size=(h,w),mode='bilinear',align_corners=False)[0]
        item['seam']=F.pad(value,(0,item['image'].shape[-1]-w,0,item['image'].shape[-2]-h)); return item


def masks(target,seam):
    seam=(seam-seam.amin((-2,-1),keepdim=True))/(seam.amax((-2,-1),keepdim=True)-seam.amin((-2,-1),keepdim=True)+1e-6)
    count=target.sum(1,keepdim=True); overlap=(count>=2).float(); separation=seam.square()*(count<.5).float()
    separation=separation*(1-overlap); interaction=((separation>0)|(overlap>0)).float(); uncertain=1-interaction
    if torch.any((separation>0)&(overlap>0)): raise RuntimeError('state mask conflict')
    return separation,overlap,uncertain


def conditional_loss(student,teacher,target,seam,weights):
    sep,overlap,uncertain=masks(target,seam); prob=torch.sigmoid(student.float()); teacher_prob=torch.sigmoid(teacher.float())
    base=dice_bce_logits(student,target)
    union=prob.amax(1,keepdim=True); sep_loss=(sep*F.relu(union-.10).square()).sum()/sep.sum().clamp_min(1)
    # At true projection overlap, every labelled participating instance must remain foreground.
    ov_pos=overlap.expand_as(target)*target
    ov_loss=(ov_pos*F.softplus(-student.float())).sum()/ov_pos.sum().clamp_min(1)
    # Outside confirmed interaction states, retain the mature R332 output distribution.
    keep=uncertain.expand_as(prob)
    preserve=(keep*(prob-teacher_prob).square()).sum()/keep.sum().clamp_min(1)
    total=weights[0]*base+weights[1]*sep_loss+weights[2]*ov_loss+weights[3]*preserve
    return total,(base,sep_loss,ov_loss,preserve),{'sep_mass':float(sep.mean()),'overlap_mass':float(overlap.mean()),'uncertain_mass':float(uncertain.mean())}


@torch.inference_mode()
def collect(network,refiner,loader,device,steps,pairs):
    network.eval();refiner.eval(); cases={}; seam_fp=[]
    for b in loader:
        image=b['image'].to(device);target=(b['mask'].to(device)>.5);seam=b['seam'].to(device)
        base,_=network(image,False);out=iterative_refine(image,base,refiner,steps)[0][-1];pred=torch.sigmoid(out)>=.5
        cases[str(b['case'][0])]={'pred':pred.cpu().numpy()[0],'target':target.cpu().numpy()[0]}
        sep,_,_=masks(target.float(),seam); seam_fp.append(float((pred.any(1,keepdim=True)*sep).sum()/sep.sum().clamp_min(1)))
    return {'official':official_metrics(cases,pairs),'seam_fp':float(np.mean(seam_fp))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True)
    p.add_argument('--r332-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=18);p.add_argument('--steps',type=int,default=2);p.add_argument('--lr',type=float,default=4e-5)
    p.add_argument('--seg-weight',type=float,default=.10);p.add_argument('--seam-weight',type=float,default=.025)
    p.add_argument('--overlap-weight',type=float,default=.04);p.add_argument('--preserve-weight',type=float,default=1.0)
    p.add_argument('--seed',type=int,default=3351);p.add_argument('--smoke',action='store_true')
    p.add_argument('--loss-version', choices=['r335','r346'], default='r335');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    seed_all(a.seed);device=torch.device('cuda');payload=torch.load(a.r332_checkpoint,map_location=device,weights_only=False)
    network=NnUNetMultiLabelPrior().to(device);network.load_state_dict(payload['model'],strict=True);network.requires_grad_(False)
    student=InstanceCompletionRefiner().to(device);student.load_state_dict(payload['refiner'],strict=True)
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    train=RamSeamDataset(a.dataset_root,a.prior_root,'train',True);val=RamSeamDataset(a.dataset_root,a.prior_root,'val',False)
    if a.smoke:train.mask_files=train.mask_files[:2];val.mask_files=val.mask_files[:2]
    tl=DataLoader(train,batch_size=1,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=not a.smoke);vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    opt=torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(True);pairs=top_training_pairs(a.dataset_root,15)
    weights=(a.seg_weight,a.seam_weight,a.overlap_weight,a.preserve_weight)
    routed_loss = conditional_loss_v346 if a.loss_version == 'r346' else conditional_loss
    if a.smoke:
        b=next(iter(tl));image=b['image'].to(device);target=b['mask'].to(device);seam=b['seam'].to(device);base,_=network(image,False)
        idx=choose_pair_instances(torch.sigmoid(base).detach(),target,4)
        with torch.no_grad(): t=iterative_refine(image,base,teacher,a.steps,idx)[0][-1]
        s=iterative_refine(image,base,student,a.steps,idx)[0][-1];loss,terms,state=routed_loss(s,t,target,seam,weights);loss.backward()
        print(json.dumps({'smoke':True,'finite':bool(torch.isfinite(loss)),'loss':float(loss),'terms':[float(x) for x in terms],**state,'trainable':sum(x.numel() for x in student.parameters())}));return
    baseline=collect(network,teacher,vl,device,a.steps,pairs);base_flat=flatten(baseline['official']);history=a.output/'history.jsonl'
    baseline_row={'epoch':0,'seconds':0.0,'loss_terms':{},'state_mass':{},'gate':False,
                  'checks':{},'baseline':base_flat,'adapted':base_flat,
                  'seam_fp':{'baseline':baseline['seam_fp'],'adapted':baseline['seam_fp']}}
    best=((0,0.0,0.0,0.0),baseline_row)
    torch.save({'refiner':student.state_dict(),'epoch':0,'row':baseline_row},a.output/'best.pth')
    for epoch in range(1,a.epochs+1):
        student.train();sums=np.zeros(4);states=np.zeros(3);started=time.time()
        for b in tl:
            image=b['image'].to(device);target=b['mask'].to(device);seam=b['seam'].to(device);base,_=network(image,False);idx=choose_pair_instances(torch.sigmoid(base).detach(),target,4)
            with torch.no_grad():t=iterative_refine(image,base,teacher,a.steps,idx)[0][-1]
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):s=iterative_refine(image,base,student,a.steps,idx)[0][-1];loss,terms,state=routed_loss(s,t,target,seam,weights)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(student.parameters(),8);scaler.step(opt);scaler.update()
            sums += [float(x) for x in terms];states += [state['sep_mass'],state['overlap_mass'],state['uncertain_mass']]
        metrics=collect(network,student,vl,device,a.steps,pairs);final=flatten(metrics['official']);gate,checks=checkpoint_gate(final,base_flat)
        checks.update({'overlap_dsc_not_below_r332':final['overlap_dsc']>=base_flat['overlap_dsc'],'overlap_nsd_not_below_r332':final['overlap_nsd_2px']>=base_flat['overlap_nsd_2px'],'overlap_msd_not_above_r332':final['overlap_msd_px']<=base_flat['overlap_msd_px'],'seam_fp_not_above_r332':metrics['seam_fp']<=baseline['seam_fp']});gate=bool(gate and all(checks.values()))
        row={'epoch':epoch,'seconds':time.time()-started,'loss_terms':dict(zip(['seg','seam','overlap','preserve'],(sums/len(tl)).tolist())),'state_mass':dict(zip(['seam','overlap','uncertain'],(states/len(tl)).tolist())),'gate':gate,'checks':checks,'baseline':base_flat,'adapted':final,'seam_fp':{'baseline':baseline['seam_fp'],'adapted':metrics['seam_fp']}}
        with history.open('a') as f:f.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        score=(int(gate),final['overall_dsc']-base_flat['overall_dsc'],final['overlap_nsd_2px']-base_flat['overlap_nsd_2px'],baseline['seam_fp']-metrics['seam_fp'])
        if best is None or score>best[0]:best=(score,row);torch.save({'refiner':student.state_dict(),'epoch':epoch,'row':row},a.output/'best.pth')
    result={'experiment':'R335_RAM_CONDITIONAL_STATE_LOSS','split':'validation','test_used':False,'new_inference_module':False,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'best':best[1]};(a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'complete':True,'best_epoch':best[1]['epoch'],'gate':best[1]['gate']}))
if __name__=='__main__':main()
