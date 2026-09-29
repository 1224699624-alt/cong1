"""Bounded frozen real-case C0/S/O/corrector checks, no optimizer steps."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from ram_highres_capture import load_frozen_c0,capture_case,forward_c0
from ram_so_conditional_corrector import PriorMaps,RAMSOConditionalCorrector
from run_ram_so_o_pair import model_load
from r351_data import AlignedRamDataset
from audit_ram_s_prior_20260929 import canonical_masks,full_image_gt_gap_target

def digest(module):
    h=hashlib.sha256()
    for name,x in module.state_dict().items():h.update(name.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.lock.read_text());start=time.monotonic();torch.set_num_threads(8)
    device=torch.device('cuda');model,refiner,_=load_frozen_c0(Path(cfg['c0']),device)
    generator,_=model_load(cfg['overlap'],device);generator.eval().requires_grad_(False)
    before=[digest(model),digest(refiner),digest(generator)]
    ds=AlignedRamDataset(Path(cfg['dataset']),Path(cfg['s_prior_root']),'train',False)
    lookup={p.stem:i for i,p in enumerate(ds.mask_files)}
    torch.manual_seed(cfg['seed']);head=RAMSOConditionalCorrector(bound=cfg['bound']).to(device)
    head_initial=digest(head);rows=[]
    for case in cfg['cases']:
        batch=ds[lookup[case]];image=batch['image'][None].to(device)
        reference=forward_c0(model,refiner,image);capture=capture_case(model,refiner,image)
        maxima=[float(np.max(np.abs(reference[k]-capture[k]))) for k in ('source','step1','step2')]
        if maxima!=[0.,0.,0.]:raise ValueError('C0 capture parity')
        z2=torch.from_numpy(capture['step2'])[None].to(device)
        hidden=torch.from_numpy(capture['refiner_h']).to(device)
        s=batch['seam'][None].to(device)
        with torch.inference_mode():o=generator(image).sigmoid()
        maps=PriorMaps(s=s,o=o,image_extent=tuple(int(x) for x in batch['original_hw']),
                       metadata_hashes={'S_manifest':cfg['S_manifest_sha256'],'O':cfg['O_sha256']})
        arms={}
        for arm in ('C00','CS','CO','CSO'):
            with torch.no_grad():final,delta=head(image,z2,hidden,maps,arm=arm,chunk_size=3)
            arms[arm]={'final_minus_z2_max':float((final-z2).abs().max()),'delta_nonzero':int(delta.count_nonzero()),
                       'mask_changed':int(((final.sigmoid()>=.5)!=(z2.sigmoid()>=.5)).sum())}
            if any(arms[arm].values()):raise ValueError('new zero-head parity')
        valid=batch['valid'].numpy()[0].astype(bool);truth=batch['mask'].numpy()>.5
        pred=capture['step2']>=0
        union=truth.any(0);fp=(pred.any(0)&~union)&valid;fg=union&valid;overlap=(truth.sum(0)>=2)&valid
        masks384=canonical_masks(Path(cfg['dataset'])/'BoneSegmentation/masks/train'/f'{case}.npy')
        gap384,_=full_image_gt_gap_target(masks384)
        native_hw=tuple(int(x) for x in batch['original_hw'])
        gap_tensor=F.interpolate(torch.from_numpy(gap384)[None,None],native_hw,mode='bilinear',align_corners=False)[0,0]
        gap_native=F.pad(gap_tensor,(0,valid.shape[1]-native_hw[1],0,valid.shape[0]-native_hw[0])).numpy()
        gap=(gap_native>1e-4)&~union&valid
        sn=s.cpu().numpy()[0,0]
        def stats(mask):return {'n':int(mask.sum()),'mean_S':float(sn[mask].mean()) if mask.any() else None,
                               'fraction_S_ge_0_5':float((sn[mask]>=.5).mean()) if mask.any() else None}
        # Separate original model FP background from correct foreground; these
        # are S diagnostics, not a new seam GT or a forward decision mask.
        row={'case':case,'capture_maxima':maxima,'arms':arms,'S_shape':list(s.shape),'O_shape':list(o.shape),
             'S_stats':{'min':float(s.min()),'max':float(s.max()),'mean':float(s.mean())},
             'C0_union_FP_background':stats(fp),'C0_FP_in_gap_proxy':stats(fp&gap),
             'C0_FP_other_background':stats(fp&~gap),'GT_foreground':stats(fg),'GT_overlap':stats(overlap),
             'GT_gap_proxy':stats(gap),'C0_correct_FG':stats(fg&pred.any(0)),
             'GT_gap_proxy_note':'GT pair geometric background proxy for offline diagnosis only, not clinical seam gold standard'}
        if len(rows)==0:
            head.zero_grad(set_to_none=True)
            target=batch['mask'][None].to(device)
            final,delta=head(image,z2,hidden,maps,arm='CSO',chunk_size=3)
            loss=F.binary_cross_entropy_with_logits(final,target)
            loss.backward()
            row['first_backward']={'BCE14':float(loss),'head_grads':{n:float(p.grad.norm()) if p.grad is not None else None for n,p in head.named_parameters()},
                'old_model_grad_none':all(p.grad is None for p in model.parameters()),
                'old_refiner_grad_none':all(p.grad is None for p in refiner.parameters()),
                'generator_grad_none':all(p.grad is None for p in generator.parameters()),'optimizer_steps':0}
            if not row['first_backward']['old_model_grad_none'] or not row['first_backward']['old_refiner_grad_none']:raise ValueError('C0 grad leak')
            if not any(v and v>0 for n,v in row['first_backward']['head_grads'].items() if 'final_layer' in n):raise ValueError('new head grad absent')
            if any(v and v>0 for n,v in row['first_backward']['head_grads'].items() if 'input_projection' in n):raise ValueError('zero final first-step explanation')
            head.zero_grad(set_to_none=True)
        rows.append(row)
        del capture,hidden,z2,s,o,maps,final,delta
    after=[digest(model),digest(refiner),digest(generator)]
    result={'status':'PASS' if before==after and head_initial==digest(head) else 'FAIL',
        'position':'after mature R332 step2; no F_high','old_state_digest_before':before,'old_state_digest_after':after,
        'head_state_unchanged':head_initial==digest(head),'head_parameter_n':sum(p.numel() for p in head.parameters()),
        'bound':cfg['bound'],'cases':rows,'seconds':time.monotonic()-start,'optimizer_steps':0,'test_used':False}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(rows),'seconds':result['seconds']}),flush=True)
    if result['status']!='PASS':raise ValueError('state changed')

if __name__=='__main__':main()
