#!/usr/bin/env python3
"""Read-only locked R329 padded-canvas evaluation of C0/C00/CS endpoints."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset

from run_ram_so_c00_cs_pair import load_batch_data,prior,_capture_frozen,_load_head_for_eval,head_forward,sha
from ram_highres_capture import load_frozen_c0
from train_r284_ramw600_overlap_prior import top_training_pairs
from evaluate_r329_ram_official_metrics_val import _aggregate,_official_one

ARMS=('C0','C00','CS')
def save(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def run(lock,split_name,output,lock_path):
    start=time.monotonic();device=torch.device('cuda')
    model,refiner,_=load_frozen_c0(Path(lock['c0']),device)
    heads={a:_load_head_for_eval(lock,a,Path(lock['run_root'])/f'TRAIN_{a}'/'final.pth',lock_path,device) for a in ('C00','CS')}
    ds=load_batch_data(lock,'val' if split_name=='val' else 'train')
    manifest=json.loads(Path(lock['split_manifest']).read_text())
    cases=manifest['val_cases'] if split_name=='val' else manifest['groups']['probe_holdout']
    wanted=set(cases);indices=[i for i,p in enumerate(ds.mask_files) if p.stem in wanted]
    if len(indices)!=len(cases):raise ValueError('case set differs')
    loader=DataLoader(Subset(ds,indices),batch_size=1,shuffle=False,num_workers=0)
    pairs=top_training_pairs(Path(lock['dataset']),15)
    rows={a:{'members':[],'overlap':[],'pairs':[]} for a in ARMS};per_case={a:[] for a in ARMS}
    for batch in loader:
        image,seam,maps=prior(batch,device,lock)
        _,_,z2,hidden=_capture_frozen(model,refiner,image)
        truth=batch['mask'].numpy()[0]>.5
        case=batch['case'][0];valid=batch['valid'].numpy()[0,0].astype(bool)
        with np.load(Path(lock['h0_root'])/('val' if split_name=='val' else 'train')/f'{case}.npz',allow_pickle=False) as z:
            shape=tuple(int(v) for v in z['shape'])
            if shape!=truth.shape or tuple(z['original_hw'])!=tuple(int(v) for v in batch['original_hw'][0]):
                raise ValueError('H0 identity')
            h0=np.unpackbits(z['h0'],bitorder='little',count=int(np.prod(shape))).reshape(shape).astype(bool)
        original=(z2>=0).cpu().numpy()[0]
        for arm in ARMS:
            with torch.no_grad():
                final,delta=(z2,torch.zeros_like(z2)) if arm=='C0' else head_forward(heads[arm],arm,image,z2,hidden,maps)
            pred=(final>=0).cpu().numpy()[0]
            member=[_official_one(pred[k],truth[k]) for k in range(14)]
            overlap=_official_one(pred.sum(0)>=2,truth.sum(0)>=2)
            pair=[_official_one(pred[i]&pred[j],truth[i]&truth[j])
                  for i,j in pairs if (truth[i]&truth[j]).any()]
            rows[arm]['members'].extend(member);rows[arm]['overlap'].append(overlap);rows[arm]['pairs'].extend(pair)
            d=delta.cpu().numpy()[0]
            per_case[arm].append({'case':case,'overall':_aggregate(member),'overlap':_aggregate([overlap]),
                'pair':_aggregate(pair),'pair_count':len(pair),
                'fixed_H0_member_count':int(h0.sum()),'fixed_H0_member_recovered':int((pred&h0).sum()),
                'new_TP_member_vs_C0':int((pred&~original&truth&valid[None]).sum()),
                'new_FP_member_vs_C0':int((pred&~original&~truth&valid[None]).sum()),
                'removed_FP_member_vs_C0':int((~pred&original&~truth&valid[None]).sum()),
                'lost_TP_member_vs_C0':int((~pred&original&truth&valid[None]).sum()),
                'delta_mean':float(d.mean()),'delta_abs_mean':float(np.abs(d).mean()),
                'delta_max_abs':float(np.abs(d).max()),'delta_positive_fraction':float((d>0).mean()),
                'delta_negative_fraction':float((d<0).mean())})
    official={}
    for arm,r in rows.items():
        official[arm]={'overall_instance_metrics':_aggregate(r['members']),
            'overlap_region_metrics':_aggregate(r['overlap']),
            'overlap_pair_intersection_metrics':_aggregate(r['pairs']),
            'overlap_pair_count':len(r['pairs'])}
    if split_name=='val':
        historic=json.loads(Path(lock['historical_C0_val']).read_text())['official']['step2']
        if official['C0']!=historic:raise ValueError('padded C0 does not exactly reproduce locked historical C0 val')
    result={'status':'COMPLETE','subset':split_name,'cases':len(indices),
      'standard':'original evaluate_r329_ram_official_metrics_val.py _official_one/_aggregate on full native padded canvas, train-only top15 pairs, threshold0.5',
      'official':official,'per_case':per_case,'checkpoints':{a:sha(Path(lock['run_root'])/f'TRAIN_{a}'/'final.pth') for a in ('C00','CS')},
      'time_seconds':time.monotonic()-start,'C0_historic_val_exact':split_name=='val','test_used':False}
    save(output,result)
    print(json.dumps({'status':result['status'],'subset':split_name,'cases':len(indices),
      'overlap_Dice':{a:official[a]['overlap_region_metrics']['dsc'] for a in ARMS},'seconds':result['time_seconds']}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True)
    ap.add_argument('--subset',choices=('holdout','val'),required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(8);lock=json.loads(a.lock.read_text())
    if lock['status']!='LOCKED_PADDED_READONLY_AFTER_TRAIN' or sha(lock['training_lock'])!=lock['training_lock_sha256']:
        raise ValueError('read-only lock')
    original=json.loads(Path(lock['training_lock']).read_text());original['historical_C0_val']=lock['historical_C0_val']
    for arm in ('C00','CS'):
        checkpoint=Path(original['run_root'])/f'TRAIN_{arm}'/'final.pth'
        if sha(checkpoint)!=lock['checkpoint_sha256'][arm]:raise ValueError('checkpoint changed')
    if sha(Path(__file__))!=lock['code_sha256']:raise ValueError('evaluation code changed')
    if a.output.exists():raise FileExistsError(a.output)
    run(original,a.subset,a.output,Path(lock['training_lock']))
if __name__=='__main__':main()
