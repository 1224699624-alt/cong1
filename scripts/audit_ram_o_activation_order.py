#!/usr/bin/env python3
"""Read-only check of sigmoid/interpolation order for the RAM O generator."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from structural_prior_v3 import ImageOnlyOverlapPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset

EXPECTED='e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--overlap',type=Path,required=True)
    ap.add_argument('--h0-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if sha(a.overlap)!=EXPECTED:raise ValueError('O checkpoint')
    payload=torch.load(a.overlap,map_location='cpu',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=ImageOnlyOverlapPrior().to(device);net.load_state_dict(payload['model'],strict=True)
    net.eval().requires_grad_(False);torch.set_num_threads(16)
    ds=NativeWristDataset(a.dataset,'val',False)
    if len(ds)!=69:raise ValueError('val count')
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    totals={key:0.0 for key in ('H0_n','H0_actual_sum','H0_eval_sum','H0_actual_high','H0_eval_high',
                               'H0_flip_to_high','H0_flip_to_low','GT_n','GT_actual_high','GT_eval_high',
                               'pixel_n','pixel_abs_diff_sum','pixel_max_abs_diff','logit_parity_max')}
    cases=[]
    for b in loader:
        case=b['case'][0]
        image=b['image'].to(device)
        with torch.inference_mode():
            low_image=F.interpolate(image,(192,192),mode='bilinear',align_corners=False)
            low_logits=net(low_image)
            actual=net(image).sigmoid()
            eval_style=F.interpolate(low_logits.sigmoid(),image.shape[-2:],mode='bilinear',align_corners=False)
            logits_ref=F.interpolate(low_logits,image.shape[-2:],mode='bilinear',align_corners=False)
            logit_parity=float((net(image)-logits_ref).abs().max())
        actual=actual.float().cpu().numpy()[0,0]
        eval_style=eval_style.float().cpu().numpy()[0,0]
        target=b['mask'].numpy()[0]>.5
        valid=b['valid'].numpy()[0,0].astype(bool)
        gt=(target.sum(0)>=2)&valid
        z=np.load(a.h0_root/'val'/f'{case}.npz',allow_pickle=False)
        shape=tuple(int(x) for x in z['shape'])
        h0=np.unpackbits(z['h0'],bitorder='little')[:np.prod(shape)].reshape(shape).astype(bool)
        if shape!=target.shape or tuple(z['original_hw'])!=tuple(b['original_hw'].numpy()[0]):raise ValueError('H0 geometry')
        av=np.broadcast_to(actual,target.shape)[h0]
        ev=np.broadcast_to(eval_style,target.shape)[h0]
        aa=av>=.5;ee=ev>=.5
        diff=np.abs(actual[valid]-eval_style[valid])
        totals['H0_n']+=len(av);totals['H0_actual_sum']+=float(av.sum(dtype=np.float64))
        totals['H0_eval_sum']+=float(ev.sum(dtype=np.float64))
        totals['H0_actual_high']+=int(aa.sum());totals['H0_eval_high']+=int(ee.sum())
        totals['H0_flip_to_high']+=int((~aa&ee).sum())
        totals['H0_flip_to_low']+=int((aa&~ee).sum())
        totals['GT_n']+=int(gt.sum())
        totals['GT_actual_high']+=int(((actual>=.5)&gt).sum())
        totals['GT_eval_high']+=int(((eval_style>=.5)&gt).sum())
        totals['pixel_n']+=int(diff.size);totals['pixel_abs_diff_sum']+=float(diff.sum(dtype=np.float64))
        totals['pixel_max_abs_diff']=max(totals['pixel_max_abs_diff'],float(diff.max()))
        totals['logit_parity_max']=max(totals['logit_parity_max'],logit_parity)
        cases.append({'case':case,'H0_n':len(av),'H0_mean_actual':float(av.mean()) if len(av) else None,
                      'H0_mean_eval_style':float(ev.mean()) if len(ev) else None,
                      'H0_threshold_flip_count':int(np.count_nonzero(aa!=ee)),
                      'max_abs_diff':float(diff.max())})
    if totals['H0_n']!=71335:raise ValueError('H0 count')
    output={'status':'COMPLETE_READ_ONLY','cases':69,'test_used':False,'optimizer_updates':0,
            'actual_downstream':'sigmoid(bilinear_upsample_192_logits)',
            'generator_report_eval':'bilinear_upsample(sigmoid_192_logits)',
            'totals':totals,
            'H0_mean_actual':totals['H0_actual_sum']/totals['H0_n'],
            'H0_mean_eval_style':totals['H0_eval_sum']/totals['H0_n'],
            'H0_fraction_actual_ge_0_5':totals['H0_actual_high']/totals['H0_n'],
            'H0_fraction_eval_ge_0_5':totals['H0_eval_high']/totals['H0_n'],
            'valid_pixel_MAE':totals['pixel_abs_diff_sum']/totals['pixel_n'],
            'per_case':cases}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('status','H0_mean_actual','H0_mean_eval_style',
      'H0_fraction_actual_ge_0_5','H0_fraction_eval_ge_0_5','valid_pixel_MAE')}),flush=True)

if __name__=='__main__':main()
