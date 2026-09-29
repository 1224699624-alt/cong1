#!/usr/bin/env python3
"""Matched post-R332 C00/CS head-only training and RAM validation.

Every update uses all fourteen observed RAM members.  C0, mature refiner and
the R323 seam-map producer stay frozen.  S is loaded from the saved GT-free
canonical maps, but no target, H0, or evaluation mask enters the corrector.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ram_highres_capture import load_frozen_c0
from ram_so_conditional_corrector import PriorMaps,RAMSOConditionalCorrector
from r332_v31_port import iterative_refine_o
from r351_data import AlignedRamDataset
from train_r284_ramw600_overlap_prior import top_training_pairs
from train_r330_ram_pair_surface_volume_prior import soft_surface_nsd_loss,volume_losses
from evaluate_r329_ram_official_metrics_val import _aggregate,_official_one

ARMS=('C00','CS')
SOURCE_SHA='ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384'
S_MANIFEST_SHA='7da1adf0a418cb909f3db1e788de647a3afeea71925c48a440351e8aeb2e0cf4'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def state_digest(module):
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def _capture_frozen(model,refiner,image):
    """Keep regular detached GPU tensors, not inference tensors for backward."""
    hidden=[];calls=0
    def collect(_module,_inputs,output):
        nonlocal calls
        if calls>=5:hidden.append(output.detach())
        calls+=1
    hook=refiner.dec1.register_forward_hook(collect)
    try:
        with torch.no_grad():
            source=model(torch.cat((image,torch.zeros_like(image)),dim=1))
            states,_=iterative_refine_o(image,source,refiner,2,torch.zeros_like(image),chunk=3)
    finally:hook.remove()
    if calls!=10 or len(hidden)!=5 or len(states)!=3:raise RuntimeError('C0 two-step feature hook order')
    h=torch.cat(hidden,dim=0)
    if h.shape!=(14,12,*image.shape[-2:]):raise RuntimeError(f'step2 hidden {tuple(h.shape)}')
    return states[0].detach(),states[1].detach(),states[2].detach(),h

def load_batch_data(lock,split):
    ds=AlignedRamDataset(Path(lock['dataset']),Path(lock['s_prior_root']),split,False)
    expected=425 if split=='train' else 69
    if len(ds)!=expected:raise ValueError((split,len(ds)))
    return ds

def prior(batch,device,lock):
    image=batch['image'].to(device)
    seam=batch['seam'].to(device)
    if image.shape!=seam.shape or not bool(torch.isfinite(seam).all()) or float(seam.min())<0 or float(seam.max())>1:
        raise ValueError('RAM S alignment/range')
    hw=batch['original_hw']
    if hw.ndim==2:hw=hw[0]
    maps=PriorMaps(s=seam,o=None,image_extent=tuple(int(v) for v in hw),
                   metadata_hashes={'S_manifest':S_MANIFEST_SHA})
    return image,seam,maps

def head_forward(head,arm,image,z2,hidden,maps):
    return head(image,z2,hidden,maps,arm=arm,chunk_size=3)

def complete_loss(final,target,valid,hw):
    """One full14 loss, crop padding so R330 geometry counts native pixels."""
    h,w=hw
    logits=final[:,:,:h,:w].float()
    truth=target[:,:,:h,:w].float()
    v=valid[:,:,:h,:w].float()
    if truth.shape[1]!=14 or not bool(v.all()):raise ValueError('native valid/14 target')
    bce=(F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*v).sum()/v.sum().clamp_min(1.)/14
    prob=logits.sigmoid()
    inter=(prob*truth).sum((0,2,3));mass=prob.sum((0,2,3))+truth.sum((0,2,3))
    dice=(1-(2*inter+1)/(mass+1)).mean()
    members=prob[0,:,None];labels=truth[0,:,None]
    surface=soft_surface_nsd_loss(members,labels)
    volume,pair=volume_losses(members,labels)
    loss=.55*bce+.45*dice+.03*surface+.005*volume+.01*pair
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite loss')
    return loss,{'bce':float(bce.detach()),'dice':float(dice.detach()),'surface':float(surface.detach()),
                 'instance_volume':float(volume.detach()),'pair_volume':float(pair.detach())}

def profile(lock,output):
    start=time.monotonic();device=torch.device('cuda');seed(lock['seed'])
    model,refiner,_=load_frozen_c0(Path(lock['c0']),device)
    ds=load_batch_data(lock,'train');split=json.loads(Path(lock['split_manifest']).read_text())
    fit=set(split['groups']['probe_fit']);indices=[i for i,p in enumerate(ds.mask_files) if p.stem in fit][:3]
    head=RAMSOConditionalCorrector(bound=lock['bound']).to(device)
    times=[];peak=0
    for i in indices:
        b=next(iter(DataLoader(Subset(ds,[i]),batch_size=1,num_workers=0)))
        image,seam,maps=prior(b,device,lock);target=b['mask'].to(device);valid=b['valid'].to(device)
        torch.cuda.synchronize();started=time.monotonic()
        source,step1,z2,hidden=_capture_frozen(model,refiner,image)
        new,delta=head_forward(head,'CS',image,z2,hidden,maps)
        hw=tuple(int(v) for v in b['original_hw'][0]);loss,parts=complete_loss(new,target,valid,hw)
        head.zero_grad(set_to_none=True);loss.backward();torch.cuda.synchronize()
        times.append({'case':b['case'][0],'seconds':time.monotonic()-started,'loss':float(loss),
                      'parts':parts,'final_abs_grad':float(head.head.final_layer.weight.grad.abs().sum()),
                      'old_gradient_none':all(p.grad is None for p in model.parameters()) and all(p.grad is None for p in refiner.parameters())})
        peak=max(peak,torch.cuda.max_memory_allocated())
        head.zero_grad(set_to_none=True)
    result={'status':'PASS','profile_only_no_optimizer_step':True,'cases':times,'median_seconds':float(np.median([r['seconds'] for r in times])),
            'peak_cuda_bytes':peak,'total_seconds':time.monotonic()-start,'test_used':False}
    save(output,result);print(json.dumps(result),flush=True)

def _heads_init(lock,device):
    seed(lock['seed']);head=RAMSOConditionalCorrector(bound=lock['bound']).to(device)
    return head

def train(lock,arm,output,lock_path):
    if arm not in ARMS:raise ValueError(arm)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    start=time.monotonic();device=torch.device('cuda')
    model,refiner,_=load_frozen_c0(Path(lock['c0']),device)
    before_old=(state_digest(model),state_digest(refiner))
    head=_heads_init(lock,device);initial={n:p.detach().clone() for n,p in head.named_parameters()}
    init_sha=state_digest(head)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lock['head_lr'],weight_decay=lock['weight_decay'])
    ds=load_batch_data(lock,'train');split=json.loads(Path(lock['split_manifest']).read_text())
    wanted=set(split['groups']['probe_fit'])
    indices=[i for i,p in enumerate(ds.mask_files) if p.stem in wanted]
    if len(indices)!=256:raise ValueError('fit size')
    subset=Subset(ds,indices)
    loader=DataLoader(subset,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(lock['seed']))
    head.train();steps=0;sequence=[];last=None
    with (output/'TRAINING_TRACE.jsonl').open('w',encoding='utf8') as f:
        for epoch in range(1,lock['epochs']+1):
            for batch in loader:
                if time.monotonic()-start>lock['max_seconds_per_arm']:raise TimeoutError('arm budget')
                if steps%25==0 and shutil.disk_usage(output).free<lock['disk_floor_bytes']:raise RuntimeError('disk floor')
                image,seam,maps=prior(batch,device,lock);target=batch['mask'].to(device);valid=batch['valid'].to(device)
                bs=hashlib.sha256(batch['image'].numpy().tobytes()+batch['mask'].numpy().tobytes()+batch['seam'].numpy().tobytes()).hexdigest()
                optimizer.zero_grad(set_to_none=True)
                source,step1,z2,hidden=_capture_frozen(model,refiner,image)
                final,delta=head_forward(head,arm,image,z2,hidden,maps)
                hw=tuple(int(v) for v in batch['original_hw'][0]);loss,parts=complete_loss(final,target,valid,hw)
                loss.backward()
                grad_raw={n:float(p.grad.norm()) for n,p in head.named_parameters()}
                torch.nn.utils.clip_grad_norm_(head.parameters(),lock['clip_grad'])
                grad_received={n:float(p.grad.norm()) for n,p in head.named_parameters()}
                before={n:p.detach().clone() for n,p in head.named_parameters()}
                optimizer.step();steps+=1
                sequence.append({'case':batch['case'][0],'batch_sha256':bs})
                if steps==1 or steps%lock['log_every']==0 or steps==lock['steps_each']:
                    updates={n:float((p.detach()-before[n]).norm()) for n,p in head.named_parameters()}
                    cumulative={n:float((p.detach()-initial[n]).norm()) for n,p in head.named_parameters()}
                    moments={n:{'step':float(optimizer.state[p]['step']),
                                'exp_avg_norm':float(optimizer.state[p]['exp_avg'].norm()),
                                'exp_avg_sq_norm':float(optimizer.state[p]['exp_avg_sq'].norm())}
                             for n,p in head.named_parameters()}
                    rec={'arm':arm,'step':steps,'epoch':epoch,'case':batch['case'][0],'loss':float(loss.detach()),'parts':parts,
                         'grad_raw':grad_raw,'grad_received':grad_received,'delta_theta':updates,'cumulative_offset':cumulative,
                         'Adam':moments,'lr':optimizer.param_groups[0]['lr'],'seam_min_mean_max':[
                             float(seam.min()),float(seam.mean()),float(seam.max())],
                         'delta_mean_abs_max':[float(delta.mean()),float(delta.abs().mean()),float(delta.abs().max())],
                         'delta_positive_fraction':float((delta>0).float().mean()),'delta_negative_fraction':float((delta<0).float().mean()),
                         'final_gt_member_FN':int(((final<0)&(target>.5)&valid).sum()),
                         'final_false_positive_members':int(((final>=0)&(target<.5)&valid).sum()),
                         'successful_step':True,'AMP':False}
                    f.write(json.dumps(rec)+'\n');f.flush();last=rec
                if steps==lock['steps_each']:break
            if steps==lock['steps_each']:break
    if steps!=lock['steps_each']:raise ValueError('steps insufficient')
    if before_old!=(state_digest(model),state_digest(refiner)):raise ValueError('C0 changed')
    ck={'head':head.state_dict(),'optimizer':optimizer.state_dict(),'arm':arm,'steps':steps,
        'C0_sha256':SOURCE_SHA,'S_manifest_sha256':S_MANIFEST_SHA,'run_lock_sha256':sha(lock_path)}
    torch.save(ck,output/'final.pth')
    result={'status':'COMPLETE','arm':arm,'steps':steps,'seconds':time.monotonic()-start,'head_initial_state_sha256':init_sha,
            'checkpoint_sha256':sha(output/'final.pth'),'C0_model_refiner_digests':list(before_old),
            'sequence_sha256':hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),'case_and_batch_sequence':sequence,
            'last_trace':last,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'test_used':False,'old_model_updates':0}
    save(output/'TRAIN_RESULT.json',result);print(json.dumps({k:result[k] for k in ('status','arm','steps','seconds','checkpoint_sha256')}),flush=True)

def _load_head_for_eval(lock,arm,path,lock_path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['arm']!=arm or ck['steps']!=lock['steps_each'] or ck['run_lock_sha256']!=sha(lock_path):raise ValueError('head checkpoint identity')
    head=RAMSOConditionalCorrector(bound=lock['bound']).to(device);head.load_state_dict(ck['head'],strict=True)
    head.eval().requires_grad_(False)
    return head

def _per_case_members(pred,target,valid):
    p=pred&valid[None];t=target&valid[None]
    return {'TP':(p&t).sum((1,2)).tolist(),'FP':(p&~t).sum((1,2)).tolist(),
            'FN':(~p&t).sum((1,2)).tolist()}

def evaluate(lock,split_name,output,lock_path):
    if split_name not in ('holdout','val'):raise ValueError(split_name)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    start=time.monotonic();device=torch.device('cuda')
    model,refiner,_=load_frozen_c0(Path(lock['c0']),device)
    before=(state_digest(model),state_digest(refiner))
    heads={arm:_load_head_for_eval(lock,arm,Path(lock['run_root'])/f'TRAIN_{arm}'/'final.pth',lock_path,device) for arm in ARMS}
    ds=load_batch_data(lock,'val' if split_name=='val' else 'train')
    split=json.loads(Path(lock['split_manifest']).read_text())
    cases=split['val_cases'] if split_name=='val' else split['groups']['probe_holdout']
    wanted=set(cases);indices=[i for i,p in enumerate(ds.mask_files) if p.stem in wanted]
    if len(indices)!=len(cases):raise ValueError('eval cases')
    loader=DataLoader(Subset(ds,indices),batch_size=1,shuffle=False,num_workers=0)
    pairs=top_training_pairs(Path(lock['dataset']),15)
    all_rows={a:[] for a in ('C0',)+ARMS};per_case={a:[] for a in all_rows}
    for batch in loader:
        image,seam,maps=prior(batch,device,lock)
        with torch.no_grad():source,step1,z2,hidden=_capture_frozen(model,refiner,image)
        truth=batch['mask'].numpy()[0]>.5;valid=batch['valid'].numpy()[0,0].astype(bool)
        h,w=(int(v) for v in batch['original_hw'][0])
        case=batch['case'][0]
        for arm in all_rows:
            with torch.no_grad():
                if arm=='C0':new,delta=z2,torch.zeros_like(z2)
                else:new,delta=head_forward(heads[arm],arm,image,z2,hidden,maps)
            predicted=(new>=0).cpu().numpy()[0]
            p=predicted[:,:h,:w];t=truth[:,:h,:w]
            member=[_official_one(p[k],t[k]) for k in range(14)]
            overlap=_official_one(p.sum(0)>=2,t.sum(0)>=2)
            pair=[]
            for i,j in pairs:
                gt_pair=t[i]&t[j]
                if gt_pair.any():pair.append(_official_one(p[i]&p[j],gt_pair))
            all_rows[arm].append({'case':case,'members':member,'overlap':overlap,'pairs':pair})
            # fixed H0 source: C0 only; never fed into head
            h0_path=Path(lock['h0_root'])/('val' if split_name=='val' else 'train')/f'{case}.npz'
            with np.load(h0_path,allow_pickle=False) as npz:
                if tuple(npz['shape'])!=tuple(truth.shape):raise ValueError('H0 shape')
                h0=np.unpackbits(npz['h0'],bitorder='little',count=int(np.prod(truth.shape))).reshape(truth.shape).astype(bool)
            signed=delta.cpu().numpy()[0]
            original=(z2>=0).cpu().numpy()[0]
            counts=_per_case_members(predicted,truth,valid)
            per_case[arm].append({'case':case,'overall':_aggregate(member),'overlap':_aggregate([overlap]),
              'pair':_aggregate(pair),'pair_count':len(pair),'member_TP_FP_FN':counts,
              'fixed_H0_member_count':int(h0.sum()),'fixed_H0_member_recovered':int((predicted&h0).sum()),
              'new_FP_member_vs_C0':int((predicted&~original&~truth&valid[None]).sum()),
              'new_TP_member_vs_C0':int((predicted&~original&truth&valid[None]).sum()),
              'lost_TP_member_vs_C0':int((~predicted&original&truth&valid[None]).sum()),
              'removed_FP_member_vs_C0':int((~predicted&original&~truth&valid[None]).sum()),
              'delta_mean':float(signed.mean()),'delta_abs_mean':float(np.abs(signed).mean()),
              'delta_positive_fraction':float((signed>0).mean()),'delta_negative_fraction':float((signed<0).mean()),
              'delta_max_abs':float(np.abs(signed).max()),'S_mean':float(seam.mean())})
    official={}
    for arm,rows in all_rows.items():
        members=[item for row in rows for item in row['members']]
        overlap=[row['overlap'] for row in rows]
        pairs=[item for row in rows for item in row['pairs']]
        official[arm]={'overall_instance_metrics':_aggregate(members),'overlap_region_metrics':_aggregate(overlap),
                       'overlap_pair_intersection_metrics':_aggregate(pairs),'overlap_pair_count':len(pairs)}
    if before!=(state_digest(model),state_digest(refiner)):raise ValueError('C0 mutated during eval')
    result={'status':'COMPLETE','split':split_name,'cases':len(indices),'official':official,
            'per_case':per_case,'threshold':.5,'R332_steps':2,'pair_selection':'top15 train only',
            'C0_digests_preserved':True,'checkpoint_sha256':{arm:sha(Path(lock['run_root'])/f'TRAIN_{arm}'/'final.pth') for arm in ARMS},
            'seconds':time.monotonic()-start,'test_used':False}
    save(output/'OFFICIAL.json',result)
    print(json.dumps({'status':'COMPLETE','split':split_name,'cases':len(indices),'seconds':result['seconds'],
      'overlap_Dice':{a:official[a]['overlap_region_metrics']['dsc'] for a in official}}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('profile','train','eval'),required=True)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--split',choices=('holdout','val'))
    a=p.parse_args();torch.set_num_threads(8);lock=json.loads(a.lock.read_text())
    if lock['c0_sha256']!=SOURCE_SHA or lock['S_manifest_sha256']!=S_MANIFEST_SHA:raise ValueError('asset lock')
    if sha(lock['c0'])!=SOURCE_SHA or sha(Path(lock['s_prior_root'])/'manifest.json')!=S_MANIFEST_SHA:raise ValueError('asset hash')
    if a.phase=='profile':profile(lock,a.out)
    else:
        if lock['status']!='LOCKED_BEFORE_TRAIN' or sha(Path(__file__))!=lock['runner_sha256']:raise ValueError('runner/lock')
        source_dir=Path(__file__).resolve().parent
        for name,path in (
            ('split_manifest_sha256',Path(lock['split_manifest'])),
            ('corrector_sha256',source_dir/'ram_so_conditional_corrector.py'),
            ('evaluator_sha256',source_dir/'evaluate_r329_ram_official_metrics_val.py'),
            ('R330_geometry_sha256',source_dir/'train_r330_ram_pair_surface_volume_prior.py'),
            ('S_all_494_maps_parity_sha256',Path(lock['run_root']).parent/'S_MAP_ALL_HASHES.json'),
        ):
            if sha(path)!=lock[name]:raise ValueError(f'{name} changed')
        parity=json.loads((Path(lock['run_root']).parent/'S_MAP_PARITY.json').read_text())
        if parity['status']!='PASS' or parity['checked_maps']!=494 or parity['test_used']:
            raise ValueError('S map parity gate')
        if a.phase=='train':
            if a.arm is None:raise ValueError('arm')
            train(lock,a.arm,a.out,a.lock)
        else:
            if a.split is None:raise ValueError('split')
            evaluate(lock,a.split,a.out,a.lock)

if __name__=='__main__':main()
