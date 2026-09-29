#!/usr/bin/env python3
"""Read-only audit of the exact 192-area overlap training target at RAM H0."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from structural_prior_v3 import ImageOnlyOverlapPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset


EXPECTED = "e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def h0_load(path, shape, hw):
    with np.load(path,allow_pickle=False) as z:
        if tuple(z['shape']) != tuple(shape) or tuple(z['original_hw']) != tuple(hw):
            raise ValueError(f'H0 geometry mismatch {path}')
        bits=np.unpackbits(z['h0'],bitorder='little')[:np.prod(shape)]
    return bits.reshape(shape).astype(bool)


def empty():
    return {'n':0,'O_sum':0.,'soft_sum':0.,'O_high':0,'soft_high':0,
            'both_high':0,'O_low_soft_high':0,'O_high_soft_low':0,'both_low':0}


def add(row,o,soft):
    o=np.asarray(o,dtype=np.float32);soft=np.asarray(soft,dtype=np.float32)
    if o.shape!=soft.shape:raise ValueError('position alignment')
    oh=o>=.5;sh=soft>=.5
    row['n']+=int(o.size);row['O_sum']+=float(o.sum(dtype=np.float64));row['soft_sum']+=float(soft.sum(dtype=np.float64))
    row['O_high']+=int(oh.sum());row['soft_high']+=int(sh.sum());row['both_high']+=int((oh&sh).sum())
    row['O_low_soft_high']+=int((~oh&sh).sum());row['O_high_soft_low']+=int((oh&~sh).sum())
    row['both_low']+=int((~oh&~sh).sum())


def present(row):
    n=row['n']
    return {**row,'O_mean':row['O_sum']/n if n else None,
            'soft_target_mean':row['soft_sum']/n if n else None,
            'O_high_fraction':row['O_high']/n if n else None,
            'soft_target_high_fraction':row['soft_high']/n if n else None,
            'O_low_given_soft_high':row['O_low_soft_high']/row['soft_high'] if row['soft_high'] else None}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--overlap',type=Path,required=True)
    p.add_argument('--h0-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=('train','val'),required=True)
    a=p.parse_args()
    if sha(a.overlap)!=EXPECTED:raise ValueError('wrong O generator')
    payload=torch.load(a.overlap,map_location='cpu',weights_only=False)
    if payload['input_dependencies']!=['image']:raise ValueError('wrong O dependency')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ImageOnlyOverlapPrior().to(device);model.load_state_dict(payload['model'],strict=True)
    model.eval().requires_grad_(False);torch.set_num_threads(16)
    ds=NativeWristDataset(a.dataset,a.split,False)
    if len(ds)!={'train':425,'val':69}[a.split]:raise ValueError('split count')
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    groups={key:empty() for key in ('H0_member','H0_unique','GT_overlap_non_H0_unique',
                                   'H0_depth_le_2','H0_depth_2_to_5','H0_depth_gt_5')}
    cohort={tag:empty() for tag in ('EXT','ordinary')}
    cases=[]
    for batch in loader:
        case=str(batch['case'][0]);image=batch['image'].to(device)
        target=(batch['mask']>.5).to(device)
        valid=batch['valid'].numpy()[0,0].astype(bool)
        gt=(target.sum(1,keepdim=True)>=2).float()
        with torch.inference_mode():
            # Exact target operator from adapt_round3_overlap.low_batch.
            low=F.interpolate(gt,(192,192),mode='area')
            soft=F.interpolate(low,gt.shape[-2:],mode='bilinear',align_corners=False)[0,0].cpu().numpy()
            o=model(image).sigmoid()[0,0].cpu().numpy()
        truth=target.cpu().numpy()[0]
        overlap=gt.bool().cpu().numpy()[0,0] & valid
        h0=h0_load(a.h0_root/a.split/f'{case}.npz',truth.shape,batch['original_hw'].numpy()[0])
        if np.any(h0 & ~(truth&overlap[None])):raise ValueError(f'H0 outside true overlap {case}')
        unique=h0.any(0)&valid
        member_o=np.broadcast_to(o,truth.shape)[h0]
        member_soft=np.broadcast_to(soft,truth.shape)[h0]
        add(groups['H0_member'],member_o,member_soft)
        add(groups['H0_unique'],o[unique],soft[unique])
        add(groups['GT_overlap_non_H0_unique'],o[overlap&~unique],soft[overlap&~unique])
        depth=distance_transform_edt(overlap)
        for key,condition in [('H0_depth_le_2',depth<=2),('H0_depth_2_to_5',(depth>2)&(depth<=5)),
                              ('H0_depth_gt_5',depth>5)]:
            rows=h0&condition[None]
            add(groups[key],np.broadcast_to(o,truth.shape)[rows],
                np.broadcast_to(soft,truth.shape)[rows])
        tag='EXT' if case.startswith('EXT_') else 'ordinary'
        add(cohort[tag],member_o,member_soft)
        cases.append({'case':case,'H0_member_n':int(h0.sum()),'H0_mean_O':float(member_o.mean()) if h0.any() else None,
                      'H0_mean_soft_target':float(member_soft.mean()) if h0.any() else None,
                      'H0_soft_target_ge_0_5':float((member_soft>=.5).mean()) if h0.any() else None})
    output={'status':'COMPLETE_READ_ONLY','split':a.split,'cases':len(cases),
            'O_sha256':EXPECTED,'GT':'14 binary NPY members >=2, native/padded/canonical coordinates',
            'target_operator':'area resize GT overlap to 192x192 then bilinear upsample to native; diagnostic reconstruction, not a model upper bound',
            'groups':{k:present(v) for k,v in groups.items()},
            'cohorts':{k:present(v) for k,v in cohort.items()},'per_case':cases,
            'test_used':False,'optimizer_updates':0}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'split':a.split,'H0':output['groups']['H0_member'],'other':output['groups']['GT_overlap_non_H0_unique']}),flush=True)


if __name__=='__main__':main()
