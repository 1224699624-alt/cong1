#!/usr/bin/env python3
"""Frozen-C0 RAM O generator pair: BCE vs deduplicated H0 spatial weighting.

Only generator parameters are updated. GT/H0 are accepted by loss/data code,
never by its image-only forward. Calibration thresholds precede holdout/val.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from structural_prior_v3 import ImageOnlyOverlapPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from evaluate_r329_ram_official_metrics_val import _official_one, _aggregate

O_SHA='e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5'
C0_SHA='ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384'
ARMS=('O_REF_CONTINUE','O_HARD_WEIGHT')
NBINS=16384

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def save(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8')

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def model_load(path,device):
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p.get('input_dependencies')!=['image']:raise ValueError('not image-only')
    net=ImageOnlyOverlapPrior().to(device).float();net.load_state_dict(p['model'],strict=True)
    return net,p

def h0_read(root,split,case,shape,hw):
    with np.load(Path(root)/split/f'{case}.npz',allow_pickle=False) as z:
        if tuple(z['shape'])!=shape or tuple(z['original_hw'])!=hw:raise ValueError('H0 coordinates')
        x=np.unpackbits(z['h0'],bitorder='little',count=int(np.prod(shape))).reshape(shape).astype(bool)
    return x

def low_targets(y,valid,h_unique,size=192):
    """Valid-aware area mapping. Same operator in both arms; h never a label."""
    v=F.interpolate(valid.float(),(size,size),mode='area')
    t=F.interpolate(valid.float()*(y.sum(1,keepdim=True)>=2).float(),(size,size),mode='area')/v.clamp_min(1e-8)
    h=F.interpolate(valid.float()*h_unique.float(),(size,size),mode='area')/v.clamp_min(1e-8)
    if bool((h>t+1e-6).any()) or bool((h<0).any()) or bool((t>1+1e-6).any()):raise ValueError('h<=t contract')
    return t,v,h

def bce_loss(z,t,v,h,beta):
    w=1+beta*h;loss=F.binary_cross_entropy_with_logits(z,t,reduction='none')
    return (v*w*loss).sum()/(v*w).sum().clamp_min(1e-8)

def load_case(ds,index,h0root):
    b=ds[index];y=b['mask'][None];v=b['valid'][None];hw=tuple(int(x) for x in b['original_hw'])
    h0=h0_read(h0root,ds.split,b['case'],tuple(y.shape[1:]),hw)
    hu=torch.from_numpy(h0.any(0))[None,None]
    if bool((hu&~((y.sum(1,keepdim=True)>=2)&v)).any()):raise ValueError('H0 not real overlap')
    low=F.interpolate(b['image'][None],(192,192),mode='bilinear',align_corners=False)
    t,vg,h=low_targets(y,v,hu)
    old_t=F.interpolate((y.sum(1,keepdim=True)>=2).float(),(192,192),mode='area')
    old_v=F.interpolate(v.float(),(192,192),mode='nearest')
    return b,low,t,vg,h,h0,old_t,old_v

def profile(lock,out):
    start=time.monotonic();device=torch.device('cuda');seed(lock['seed'])
    net,payload=model_load(lock['overlap'],device)
    if sha(lock['overlap'])!=O_SHA or sha(lock['c0'])!=C0_SHA:raise ValueError('checkpoint SHA')
    ds=NativeWristDataset(Path(lock['dataset']),'train',False)
    manifest=json.loads(Path(lock['split_manifest']).read_text())
    fit=set(manifest['groups']['probe_fit'])
    indices=[i for i,p in enumerate(ds.mask_files) if p.stem in fit][:8]
    opt=torch.optim.AdamW(net.parameters(),lr=lock['lr'],weight_decay=lock['weight_decay'])
    records=[]
    for i in indices:
        b,x,t,v,h,_,oldt,oldv=load_case(ds,i,lock['h0_root'])
        x,t,v,h,oldt,oldv=[a.to(device) for a in (x,t,v,h,oldt,oldv)]
        torch.cuda.synchronize();begin=time.monotonic()
        z=net(x);l=bce_loss(z,t,v,h,0)
        plain=(F.binary_cross_entropy_with_logits(z,t,reduction='none')*v).sum()/v.sum()
        if not torch.equal(l,plain):raise ValueError('beta0 loss differs from selected reference')
        legacy=(F.binary_cross_entropy_with_logits(z,oldt,reduction='none')*oldv).sum()/oldv.sum()
        opt.zero_grad(set_to_none=True);l.backward();opt.step();torch.cuda.synchronize()
        records.append({'case':b['case'],'seconds':time.monotonic()-begin,'beta0_abs_diff':float(abs(l-plain)),
                        'legacy_vs_validaware_loss_diff':float(l-legacy),'extra_weight_fraction_beta1':float((v*h).sum()/v.sum()),
                        'h_max':float(h.max()),'t_max':float(t.max())})
    result={'status':'PASS','phase':'isolated_profile_not_formal_pair','profile_steps':len(records),
            'median_gpu_step_seconds':float(np.median([r['seconds'] for r in records])),
            'seconds':time.monotonic()-start,'torch':torch.__version__,
            'O_optimizer_available': 'optimizer' in payload, 'C0_updated':False,
            'records':records,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'test_used':False}
    save(out,result);print(json.dumps(result),flush=True)

def prepare(lock,out):
    start=time.monotonic();out=Path(out)
    if (out/'cache_manifest.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    split=json.loads(Path(lock['split_manifest']).read_text())
    wanted=set(split['groups']['probe_fit']);ds=NativeWristDataset(Path(lock['dataset']),'train',False)
    rows=[]
    for i,p in enumerate(ds.mask_files):
        if p.stem not in wanted:continue
        b,x,t,v,h,h0,_,_=load_case(ds,i,lock['h0_root'])
        f=out/f'{b["case"]}.pt';torch.save({'image':x,'target':t,'valid_mass':v,'hard_mass':h,'case':b['case']},f)
        rows.append({'case':b['case'],'file':str(f),'sha256':sha(f),'h0_members':int(h0.sum()),
                     'h0_unique':int(h0.any(0).sum()),'extra_mass_fraction':float((v*h).sum()/v.sum()),
                     'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()})
    if len(rows)!=256:raise ValueError('fit cases differ')
    result={'status':'COMPLETE','cases':len(rows),'rows':rows,'seconds':time.monotonic()-start,
            'cache_bytes':sum(Path(r['file']).stat().st_size for r in rows),'test_used':False}
    save(out/'cache_manifest.json',result);print(json.dumps({'cases':len(rows),'seconds':result['seconds'],'bytes':result['cache_bytes']}),flush=True)

def trace_layer(named,before,initial,opt,grads,received):
    rows={}
    for name,p in named:
        group=name.split('.')[0]
        r=rows.setdefault(group,{'grad_raw_sq':0.,'grad_received_sq':0.,'delta_sq':0.,'cumulative_sq':0.,
                                'parameter_sq':0.,'changed':0,'n':0,'adam_m1_sq':0.,'adam_m2_sum':0.,'adam_steps':[]})
        delta=p.detach()-before[name]
        r['grad_raw_sq']+=float(grads[name].square().sum());r['grad_received_sq']+=float(received[name].square().sum())
        r['delta_sq']+=float(delta.square().sum());r['cumulative_sq']+=float((p.detach()-initial[name]).square().sum())
        r['parameter_sq']+=float(before[name].square().sum());r['changed']+=int((delta!=0).sum());r['n']+=p.numel()
        state=opt.state[p];r['adam_m1_sq']+=float(state['exp_avg'].square().sum());r['adam_m2_sum']+=float(state['exp_avg_sq'].sum())
        r['adam_steps'].append(float(state['step']))
    for r in rows.values():
        for k in list(r):
            if k.endswith('_sq'):r[k[:-3]+'_l2']=r.pop(k)**.5
        r['changed_fraction']=r['changed']/r['n']
    return rows

def train(lock,arm,out):
    start=time.monotonic();out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);device=torch.device('cuda');seed(lock['seed'])
    net,payload=model_load(lock['overlap'],device);net.train()
    opt=torch.optim.AdamW(net.parameters(),lr=lock['lr'],weight_decay=lock['weight_decay'])
    if 'optimizer' in payload:opt.load_state_dict(copy.deepcopy(payload['optimizer']))
    manifest=json.loads(Path(lock['cache']+'/cache_manifest.json').read_text());rows=manifest['rows']
    cache=[]
    for r in rows:
        if sha(r['file'])!=r['sha256']:raise ValueError('cache SHA')
        cache.append(torch.load(r['file'],map_location=device,weights_only=False))
    named=list(net.named_parameters());initial={n:p.detach().clone() for n,p in named}
    step=0;sequence=[];beta=0 if arm=='O_REF_CONTINUE' else lock['beta']
    generator=torch.Generator().manual_seed(lock['seed'])
    with (out/'SO_PRIOR_TRACE.jsonl').open('w') as f:
        for epoch in range(1,lock['epochs']+1):
            for index in torch.randperm(len(cache),generator=generator).tolist():
                if time.monotonic()-start>lock['max_train_seconds_arm']:raise TimeoutError('budget')
                b=cache[index];x=b['image'];t=b['target'];v=b['valid_mass'];h=b['hard_mass']
                opt.zero_grad(set_to_none=True);z=net(x);ref=bce_loss(z,t,v,h,0);weighted=bce_loss(z,t,v,h,lock['beta'])
                loss=ref if beta==0 else weighted
                # Both arms use identical observation; optional component grads
                # are extracted before the one normal backward, no extra steps.
                region_grad={}
                if step==0 or (step+1)%128==0:
                    positive=-t*F.logsigmoid(z);negative=-(1-t)*F.logsigmoid(-z)
                    components={'H_cells':v*(h>0)*(positive+negative),
                                'ordinary_positive':v*(h==0)*positive,'negative_component':v*negative}
                    for key,c in components.items():
                        g=torch.autograd.grad(c.sum()/v.sum(),[p for _,p in named],retain_graph=True,allow_unused=True)
                        region_grad[key]={n.split('.')[0]:0. for n,_ in named}
                        for (n,_),gg in zip(named,g):
                            if gg is not None:region_grad[key][n.split('.')[0]]+=float(gg.square().sum())
                        region_grad[key]={k:v**.5 for k,v in region_grad[key].items()}
                loss.backward();grads={n:p.grad.detach().clone() for n,p in named}
                if not torch.isfinite(loss) or any(not bool(torch.isfinite(g).all()) for g in grads.values()):raise ValueError('nonfinite')
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.)
                received={n:p.grad.detach().clone() for n,p in named};before={n:p.detach().clone() for n,p in named}
                opt.step();step+=1;sequence.append(b['case'])
                if step==1 or step%lock['log_every']==0 or step==lock['epochs']*len(cache):
                    r={'arm':arm,'step':step,'epoch':epoch,'case':b['case'],'reference_bce':float(ref),
                       'weighted_bce':float(weighted),'loss_used':float(loss),'beta':beta,
                       'extra_weight_mass_fraction':float(lock['beta']*(v*h).sum()/v.sum()),
                       'H_score_mean':float((z.sigmoid()*v*h).sum()/(v*h).sum().clamp_min(1e-8)),
                       'ordinary_positive_score_mean':float((z.sigmoid()*v*t*(h==0)).sum()/(v*t*(h==0)).sum().clamp_min(1e-8)),
                       'region_raw_grad_l2':region_grad,'layers':trace_layer(named,before,initial,opt,grads,received),
                       'lr':opt.param_groups[0]['lr'],'AMP':False,'success':True}
                    f.write(json.dumps(r)+'\n');f.flush()
        if step!=lock['epochs']*256:raise ValueError('steps incomplete')
    torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'arm':arm,'updates':step,
                'input_dependencies':['image'],'source_sha256':O_SHA,'lock_sha256':lock['lock_sha256']},out/'final.pth')
    result={'status':'COMPLETE','arm':arm,'updates':step,'seconds':time.monotonic()-start,'checkpoint_sha256':sha(out/'final.pth'),
            'initial_sha256':O_SHA,'initial_optimizer_policy':'same fresh AdamW; historical O state absent',
            'sequence_sha256':hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),'case_sequence':sequence,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'test_used':False,'C0_updated':False}
    save(out/'TRAIN_RESULT.json',result);print(json.dumps({k:result[k] for k in ('status','arm','updates','seconds','sequence_sha256')}),flush=True)

def hist_metrics(pos,neg):
    tp=np.cumsum(pos[::-1],dtype=np.float64);fp=np.cumsum(neg[::-1],dtype=np.float64)
    totalp=tp[-1];totaln=fp[-1];recall=tp/max(1,totalp);precision=tp/np.maximum(tp+fp,1)
    ap=float(np.sum(np.diff(np.r_[0,recall])*precision))
    auc=float(np.trapz(np.r_[0,recall],np.r_[0,fp/max(1,totaln)]))
    return {'AP_hist_approx':ap,'ROC_AUC_hist_approx':auc,'histogram_bins':NBINS,
            'curve_recall':recall[::128].tolist(),'curve_precision':precision[::128].tolist(),
            'curve_FPR':(fp/max(1,totaln))[::128].tolist()}

def thresholds_calibrate(scores,allowed):
    scores=np.asarray(scores,dtype=np.float32)
    if allowed<0 or allowed>=len(scores):raise ValueError('invalid allowed negative count')
    # All ties on the cut are excluded, so exact count cannot exceed budget.
    cut=np.partition(scores,len(scores)-1-allowed)[len(scores)-1-allowed]
    threshold=float(np.nextafter(cut,np.float32(np.inf)))
    if int((scores>=threshold).sum())>allowed:raise ValueError('cal threshold exceeds FP cap')
    return threshold

def evaluate_group(lock,models,subset,case_ids,thresholds=None,calibrate=False):
    ds=NativeWristDataset(Path(lock['dataset']),'val' if subset=='val' else 'train',False)
    wanted=set(case_ids);rows={arm:[] for arm in models};single_scores={arm:[] for arm in models}
    all_hist={a:{'pos':np.zeros(NBINS,dtype=np.int64),'neg':np.zeros(NBINS,dtype=np.int64)} for a in models}
    per_score_data={a:[] for a in models};start=time.monotonic()
    for index,p in enumerate(ds.mask_files):
        if p.stem not in wanted:continue
        b,x,t,v,h,h0,_,_=load_case(ds,index,lock['h0_root'])
        image=b['image'][None].to('cuda');truth=b['mask'].numpy()>0;valid=b['valid'].numpy()[0].astype(bool)
        y=truth.sum(0)>=2;single=(truth.sum(0)==1)&valid;bg=(truth.sum(0)==0)&valid;hu=h0.any(0)&valid
        for arm,net in models.items():
            net.eval()
            with torch.inference_mode():z=net(x.to('cuda'));o=F.interpolate(z,image.shape[-2:],mode='bilinear',align_corners=False).sigmoid().cpu().numpy()[0,0]
            pr=(o>=.5)&valid;gt=y&valid;values=o[valid];bits=np.minimum((values*NBINS).astype(np.int64),NBINS-1)
            yp=y[valid];all_hist[arm]['pos']+=np.bincount(bits[yp],minlength=NBINS);all_hist[arm]['neg']+=np.bincount(bits[~yp],minlength=NBINS)
            low_bce=float(bce_loss(z,t.to('cuda'),v.to('cuda'),h.to('cuda'),0))
            metric=_official_one(pr,gt)
            r={'case':b['case'],'side':b['case'][-1],'cohort':'EXT' if b['case'].startswith('EXT_') else 'ordinary',
               'valid':int(valid.sum()),'GT_overlap':int(gt.sum()),'GT_single':int(single.sum()),'GT_background':int(bg.sum()),
               'H0_unique':int(hu.sum()),'H0_member':int(h0.sum()),'bce':low_bce,'metric':metric,
               'fixed0_5':{'tp':int((pr&gt).sum()),'fp':int((pr&~gt&valid).sum()),'fn':int((~pr&gt).sum()),
                          'single_fp':int((pr&single).sum()),'background_fp':int((pr&bg).sum()),
                          'H0_unique_selected':int((pr&hu).sum()),'H0_member_selected':int(np.broadcast_to(pr,truth.shape)[h0].sum())}}
            if calibrate:
                single_scores[arm].append(o[single].copy())
                # Cache only per-case score masks for threshold recounting, at
                # most calibration85 (~135MB/arm); no training/test inputs.
                per_score_data[arm].append((o[single].copy(),o[gt].copy(),o[bg].copy(),o[hu].copy(),np.broadcast_to(o,truth.shape)[h0].copy()))
            else:
                threshold=thresholds[arm];pred=(o>=threshold)&valid
                r['matched_threshold']=threshold
                r['matched']={'single_fp':int((pred&single).sum()),'background_fp':int((pred&bg).sum()),
                   'GT_overlap_selected':int((pred&gt).sum()),'H0_unique_selected':int((pred&hu).sum()),
                   'H0_member_selected':int(np.broadcast_to(pred,truth.shape)[h0].sum())}
            rows[arm].append(r)
    if any(len(v)!=len(case_ids) for v in rows.values()):raise ValueError('evaluation case sets')
    if calibrate:
        orig=rows['O_ORIGINAL'];single_total=sum(r['GT_single'] for r in orig);allowed=sum(r['fixed0_5']['single_fp'] for r in orig)
        if allowed==0:raise ValueError('alpha_ref zero, user must review fallback')
        thresholds={'O_ORIGINAL':.5}
        for arm in ARMS:thresholds[arm]=thresholds_calibrate(np.concatenate(single_scores[arm]),allowed)
        for arm in models:
            threshold=thresholds[arm]
            for r,(si,gt,bg,hu,hm) in zip(rows[arm],per_score_data[arm]):
                r['matched_threshold']=threshold;r['matched']={'single_fp':int((si>=threshold).sum()),
                   'background_fp':int((bg>=threshold).sum()),'GT_overlap_selected':int((gt>=threshold).sum()),
                   'H0_unique_selected':int((hu>=threshold).sum()),'H0_member_selected':int((hm>=threshold).sum())}
        save(Path(lock['output'])/'CALIBRATION_THRESHOLDS.json',{'status':'LOCKED_BEFORE_HOLDOUT_VAL','alpha_ref':allowed/single_total,
             'single_negative_n':single_total,'allowed_fp':allowed,'thresholds':thresholds,
             'source':'original frozen O at .5 on calibration85 only','tie_rule':'nextafter partition cutoff, exclude tied cutoff',
             'threshold_dtype':'float32','test_used':False})
    summaries={}
    for arm,rs in rows.items():
        den={key:sum(r[key] for r in rs) for key in ('GT_single','GT_background','GT_overlap','H0_unique','H0_member','valid')}
        f={key:sum(r['fixed0_5'][key] for r in rs) for key in rs[0]['fixed0_5']}
        m={key:sum(r['matched'][key] for r in rs) for key in rs[0]['matched']}
        summaries[arm]={'denominators':den,'bce_case_mean':float(np.mean([r['bce'] for r in rs])),
           'standard_overlap_mask':_aggregate([r['metric'] for r in rs]),'fixed0_5':{**f,
           'precision':f['tp']/max(1,f['tp']+f['fp']),'recall':f['tp']/max(1,f['tp']+f['fn']),
           'H0_unique_recall':f['H0_unique_selected']/max(1,den['H0_unique'])},
           'matched':{**m,'threshold':thresholds[arm],'single_fpr':m['single_fp']/max(1,den['GT_single']),
           'H0_unique_recall':m['H0_unique_selected']/max(1,den['H0_unique']),
           'H0_member_recall':m['H0_member_selected']/max(1,den['H0_member']),
           'GT_overlap_recall':m['GT_overlap_selected']/max(1,den['GT_overlap'])},
           'natural_full_curves':hist_metrics(all_hist[arm]['pos'],all_hist[arm]['neg'])}
    result={'status':'COMPLETE','subset':subset,'cases':len(case_ids),'arms':summaries,'per_case':rows,
       'thresholds':thresholds,'seconds':time.monotonic()-start,'activation':'bilinear_native_logits_then_sigmoid',
       'AUC_AP':'fixed16384-bin approximation on ALL native valid pixels; exact threshold counts',
       'test_used':False,'segmentation_gain':'NOT_TESTED'}
    save(Path(lock['output'])/f'{subset.upper()}_RESULT.json',result)
    print(json.dumps({'subset':subset,'seconds':result['seconds'],'matched':{a:summaries[a]['matched'] for a in models}}),flush=True)
    return thresholds

def evaluate(lock):
    if not all(Path(lock['output'],arm,'TRAIN_RESULT.json').is_file() for arm in ARMS):raise ValueError('both arms needed')
    r=[json.loads(Path(lock['output'],a,'TRAIN_RESULT.json').read_text()) for a in ARMS]
    if r[0]['sequence_sha256']!=r[1]['sequence_sha256'] or r[0]['updates']!=r[1]['updates']:raise ValueError('pair schedule differs')
    models={'O_ORIGINAL':model_load(lock['overlap'],'cuda')[0]}
    for arm in ARMS:
        models[arm]=model_load(Path(lock['output'])/arm/'final.pth','cuda')[0]
    split=json.loads(Path(lock['split_manifest']).read_text())
    thresholds=evaluate_group(lock,models,'calibration',split['groups']['probe_calibration'],calibrate=True)
    threshold_sha=sha(Path(lock['output'])/'CALIBRATION_THRESHOLDS.json')
    evaluate_group(lock,models,'holdout',split['groups']['probe_holdout'],thresholds=thresholds)
    save(Path(lock['output'])/'BEFORE_VAL_LOCK.json',{'status':'LOCKED_BEFORE_VAL','threshold_sha256':threshold_sha,
         'checkpoint_sha256':{arm:sha(Path(lock['output'])/arm/'final.pth') for arm in ARMS},
         'holdout_result_sha256':sha(Path(lock['output'])/'HOLDOUT_RESULT.json'),'test_used':False})
    evaluate_group(lock,models,'val',split['val_cases'],thresholds=thresholds)
    if threshold_sha!=sha(Path(lock['output'])/'CALIBRATION_THRESHOLDS.json'):raise ValueError('threshold altered after val')
    save(Path(lock['output'])/'EVAL_COMPLETE.json',{'status':'COMPLETE','C0_sha256_after':sha(lock['c0']),
           'original_O_sha256_after':sha(lock['overlap']),'thresholds_fixed':True,'test_used':False})

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('profile','prepare','train','eval'),required=True)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--arm',choices=ARMS);p.add_argument('--out',type=Path)
    a=p.parse_args();torch.set_num_threads(8);lock=json.loads(a.lock.read_text());lock['lock_sha256']=sha(a.lock)
    if a.phase=='profile':profile(lock,a.out)
    elif a.phase=='prepare':prepare(lock,a.out)
    else:
        if lock['status']!='LOCKED_BEFORE_TRAIN' or sha(Path(__file__))!=lock['code_sha256']:raise ValueError('code/lock')
        if sha(lock['c0'])!=C0_SHA or sha(lock['overlap'])!=O_SHA:raise ValueError('original assets mutated')
        if shutil.disk_usage(lock['output']).free<4*1024**3:raise ValueError('disk floor')
        if a.phase=='train':
            if a.arm is None:raise ValueError('arm needed')
            train(lock,a.arm,Path(lock['output'])/a.arm)
        else:evaluate(lock)

if __name__=='__main__':main()
