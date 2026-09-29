#!/usr/bin/env python3
"""Read-only fixed-case direct S perturbation of the trained CS head."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from run_ram_so_c00_cs_pair import load_batch_data,prior,_capture_frozen,_load_head_for_eval,head_forward,sha
from ram_highres_capture import load_frozen_c0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.lock.read_text());torch.set_num_threads(8);start=time.monotonic()
    if a.out.exists():raise FileExistsError(a.out)
    device=torch.device('cuda');model,refiner,_=load_frozen_c0(Path(cfg['c0']),device)
    head=_load_head_for_eval(cfg,'CS',Path(cfg['run_root'])/'TRAIN_CS/final.pth',a.lock,device)
    ds=load_batch_data(cfg,'train');split=json.loads(Path(cfg['split_manifest']).read_text())
    holdout=split['groups']['probe_holdout']
    cases=[]
    for prefix in (False,True):
        for side in ('L','R'):
            cases.extend(sorted(c for c in holdout if c.startswith('EXT_')==prefix and c.endswith('_'+side))[:2])
    lookup={p.stem:i for i,p in enumerate(ds.mask_files)}
    rows=[]
    with torch.no_grad():
        for case in cases:
            batch=next(iter(DataLoader(Subset(ds,[lookup[case]]),batch_size=1,num_workers=0)))
            image,seam,maps=prior(batch,device,cfg)
            _,_,z2,hidden=_capture_frozen(model,refiner,image)
            on,delta_on=head_forward(head,'CS',image,z2,hidden,maps)
            off,delta_off=head_forward(head,'C00',image,z2,hidden,maps)
            direct=(on-off).float();active=seam>=.1
            gt=batch['mask'].to(device)>.5;valid=batch['valid'].to(device)
            old=z2>=0;after=on>=0;offmask=off>=0
            rows.append({'case':case,'S_mean':float(seam.mean()),'S_ge_0_1_pixels':int(active.sum()),
              'direct_S_logit_abs_mean':float(direct.abs().mean()),'direct_S_logit_abs_max':float(direct.abs().max()),
              'direct_S_logit_mean_at_S_ge_0_1':float(direct[active.expand_as(direct)].mean()) if bool(active.any()) else None,
              'direct_mask_flip_members':int((after!=offmask).sum()),
              'direct_flips_at_S_ge_0_1':int(((after!=offmask)&active).sum()),
              'direct_TP_added_vs_off':int((after&~offmask&gt&valid).sum()),
              'direct_FP_added_vs_off':int((after&~offmask&~gt&valid).sum()),
              'direct_TP_lost_vs_off':int((~after&offmask&gt&valid).sum()),
              'direct_FP_removed_vs_off':int((~after&offmask&~gt&valid).sum())})
    result={'status':'COMPLETE_READ_ONLY','cases':rows,'seconds':time.monotonic()-start,
      'checkpoint_sha256':sha(Path(cfg['run_root'])/'TRAIN_CS/final.pth'),
      'interpretation':'same trained CS weights, S on vs zero; direct mechanism only, not a matched training benefit',
      'optimizer_steps':0,'test_used':False}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'seconds':result['seconds'],'total_direct_flips':sum(x['direct_mask_flip_members'] for x in rows)}),flush=True)
if __name__=='__main__':main()
