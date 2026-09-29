"""Frozen RAM error-conditional O audit. GT is used only after model inference."""
import argparse, hashlib, json, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy


def stat(score,mask):
    x=score[mask].float()
    if not x.numel():return {'n':0,'sum':0.,'sum_sq':0.,'positive_at_05':0,'hist256':[0]*256}
    bins=(x.clamp(0,1)*255).long()
    return {'n':int(x.numel()),'sum':float(x.sum()),'sum_sq':float(x.square().sum()),
            'positive_at_05':int((x>=.5).sum()),'hist256':torch.bincount(bins,minlength=256).cpu().tolist()}


def merge(dst,src):
    for key in ['n','sum','sum_sq','positive_at_05']:dst[key]=dst.get(key,0)+src[key]
    dst['hist256']=[a+b for a,b in zip(dst.get('hist256',[0]*256),src['hist256'])]


def finalize(s):
    n=s['n'];s['mean']=s['sum']/n if n else None
    s['positive_fraction_at_05']=s['positive_at_05']/n if n else None
    return s


def hist_auc(pos,neg):
    p=np.array(pos['hist256'],dtype=np.float64);n=np.array(neg['hist256'],dtype=np.float64)
    if p.sum()==0 or n.sum()==0:return None
    return float((p*(np.cumsum(n)-n+.5*n)).sum()/(p.sum()*n.sum()))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    start=time.monotonic();legacy.seed_all(3321);torch.set_num_threads(2);device=torch.device('cuda')
    _,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    payload=torch.load(lock['reference_checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'],strict=True);refiner.load_state_dict(payload['refiner'],strict=True)
    for module in [model,refiner,generator]:module.eval().requires_grad_(False)
    o_weight_nonzero=int(torch.count_nonzero(refiner.enc1.net[0].weight[:,6]));assert o_weight_nonzero==0
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'val',False)
    manifest=[{'case':p.stem,'mask_sha':legacy.sha(p),'image_sha':legacy.sha(Path(lock['dataset'])/'BoneSegmentation/images'/(p.stem+'.bmp'))} for p in ds.mask_files]
    val_digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    assert val_digest==lock['val_content_sha256'] and len(ds)==69
    aggregates={};totals={};rows=[];conditional=[];mismatch=0;maxdiff=0.;sensitivity=[]
    for index,batch in enumerate(DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)):
        if time.monotonic()-start>lock['max_seconds']:raise TimeoutError('locked wall budget')
        image=batch['image'].to(device)
        base=model(torch.cat((image,torch.zeros_like(image)),1))
        states,_=legacy.iterative_refine_o(image,base,refiner,2,torch.zeros_like(image))
        probability=states[-1].sigmoid();prediction=probability>=.5
        deployed=generator(image).sigmoid()
        low_image=F.interpolate(image,(192,192),mode='bilinear',align_corners=False)
        old_eval=F.interpolate(generator(low_image).sigmoid(),image.shape[-2:],mode='bilinear',align_corners=False)
        second=probability.topk(2,dim=1).values[:,1:2]
        if index<3:
            with_o,_=legacy.iterative_refine_o(image,base,refiner,2,deployed)
            sensitivity.append({'case':batch['case'][0],'logit_max_abs':float((states[-1]-with_o[-1]).abs().max()),
                                'mask_changed':int(((with_o[-1].sigmoid()>=.5)!=prediction).sum())})
        # Ground truth enters only offline diagnostics below, not any forward call.
        truth=batch['mask'].to(device)>.5;valid=batch['valid'].to(device).bool()
        tcount=truth.sum(1,keepdim=True);pcount=prediction.sum(1,keepdim=True)
        true_o=tcount>=2;pred_o=pcount>=2;member_fn=truth&~prediction&valid;member_fp=~truth&prediction&valid
        exact=(prediction==truth).all(1,keepdim=True)
        masks={'all_valid':valid,'true_overlap':true_o&valid,'true_nonoverlap':~true_o&valid,
            'overlap_TP':true_o&pred_o&valid,'overlap_FN':true_o&~pred_o&valid,
            'overlap_FP':~true_o&pred_o&valid,
            'correct_single_member_foreground':(tcount==1)&exact&valid,
            'correct_empty_background':(tcount==0)&exact&valid,
            'true_overlap_with_any_missing_member':true_o&member_fn.any(1,keepdim=True)&valid,
            'true_overlap_without_missing_member':true_o&~member_fn.any(1,keepdim=True)&valid,
            'any_member_error':(member_fn|member_fp).any(1,keepdim=True)&valid}
        resolution=F.interpolate(F.interpolate(true_o.float(),(192,192),mode='area'),image.shape[-2:],mode='bilinear',align_corners=False)
        scores={'deployed_O':deployed,'legacy_eval_O':old_eval,'member_second_probability':second,'GT_resolution_roundtrip_diagnostic':resolution}
        row={'case':batch['case'][0],'native_hw':batch['original_hw'][0].tolist(),'padded_hw':list(image.shape[-2:]),'regions':{}}
        for region,mask in masks.items():
            row['regions'][region]={}
            for name,score in scores.items():
                s=stat(score,mask);merge(aggregates.setdefault(region,{}).setdefault(name,{}),s)
                row['regions'][region][name]=finalize(s)
        counts={'valid_pixels':int(valid.sum()),'true_member_positive':int((truth&valid).sum()),
            'member_FN':int(member_fn.sum()),'member_FP':int(member_fp.sum()),
            'member_FN_in_true_overlap':int((member_fn&true_o).sum()),
            'member_FN_outside_true_overlap':int((member_fn&~true_o).sum()),
            'member_FN_in_true_overlap_already_predicted_overlap':int((member_fn&true_o&pred_o).sum()),
            'member_FN_at_O_positive':int((member_fn&(deployed>=.5)).sum()),
            'member_FP_at_O_negative':int((member_fp&(deployed<.5)).sum()),
            'member_FN_in_true_overlap_at_O_positive':int((member_fn&true_o&(deployed>=.5)).sum()),
            'O_positive_in_correct_single_foreground':int(((deployed>=.5)&masks['correct_single_member_foreground']).sum())}
        row['counts']=counts
        for key,v in counts.items():totals[key]=totals.get(key,0)+v
        diff=(deployed-old_eval).abs();mismatch+=int(((deployed>=.5)!=(old_eval>=.5))[valid].sum());maxdiff=max(maxdiff,float(diff[valid].max()))
        row['O_path_difference']={'max_abs':float(diff[valid].max()),'mean_abs':float(diff[valid].mean()),'threshold_mask_mismatch':int(((deployed>=.5)!=(old_eval>=.5))[valid].sum())}
        # Fixed native second-probability bins, no calibration fitting or threshold search.
        bins=(second*10).long().clamp(0,9)
        row['condition_on_native_second_probability']=[]
        for k in range(10):
            band=(bins==k)&valid
            row['condition_on_native_second_probability'].append({'bin':k,'true_overlap':finalize(stat(deployed,band&true_o)),
                                                                  'nonoverlap':finalize(stat(deployed,band&~true_o))})
        with (a.out/'INFERENCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append({'case':row['case'],**counts,'overlap_FN_O_recall':row['regions']['overlap_FN']['deployed_O']['positive_fraction_at_05'],
                     'overlap_FN_O_mean':row['regions']['overlap_FN']['deployed_O']['mean']})
        print(json.dumps({'case_index':index+1,'case':row['case'],'elapsed':time.monotonic()-start}),flush=True)
    for region in aggregates.values():
        for s in region.values():finalize(s)
    auc={name:hist_auc(aggregates['overlap_FN'][name],aggregates['correct_single_member_foreground'][name]) for name in scores}
    result={'status':'COMPLETE','optimizer_updates':0,'test_used':False,'cases':len(rows),'reference_sha256':legacy.sha(lock['reference_checkpoint']),
        'O_sha256':legacy.sha(lock['overlap']),'val_content_sha256':val_digest,'O_weight_nonzero':o_weight_nonzero,'actual_O_forward_sensitivity':sensitivity,
        'aggregate_regions':aggregates,'member_counts':totals,'per_case_summary':rows,
        'approx_hist256_AUC_missing_overlap_vs_correct_single_foreground':auc,
        'path_difference':{'max_abs':maxdiff,'threshold_mask_mismatch_pixels':mismatch},
        'resolution_caveat':'GT roundtrip is offline representation diagnostic; not inference input or an achievable bound.',
        'scope':'O quality/error coverage only, not new segmentation performance; no correction rule applied.',
        'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
