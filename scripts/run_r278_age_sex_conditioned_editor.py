"""R278 age/sex-conditioned extension of the frozen R277 decoupled editor."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image
from run_r201_unified_eval import compute_metrics, read_binary_mask
from run_r277_decoupled_seam_editor import edit, fast_row, load_fast_cache, mean, delta, read_prob

BASE={'threshold':0.25,'bone_max':0.70,'depth':4,'dilate':1,'max_cut_fraction':0.003,'max_component_increase':2}

def metadata(path: Path):
    out={}
    with path.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f): out[str(r['id'])]={'age':float(r['boneage']),'male':str(r['male']).lower() in {'1','true','yes'}}
    return out

def conditioned(name,meta,age_mean,age_std,params):
    m=meta[Path(name).stem]; z=float(np.clip((m['age']-age_mean)/age_std,-1.5,1.5)); sex=(1.0 if m['male'] else 0.0)-0.5
    cfg=dict(BASE); cfg['threshold']=float(np.clip(BASE['threshold']+params['age_threshold_slope']*z+params['sex_threshold_offset']*sex,0.15,0.45)); cfg['max_cut_fraction']=float(np.clip(BASE['max_cut_fraction']*(1+params['age_cut_slope']*z),0.0015,0.0045)); return cfg

def eval_fast(cache,meta,mu,sd,params):
    rows=[]
    for name,base,gt,bp,sp,gb,gap,gn,dt,before in cache:
        pred,info=edit(base,bp,sp,conditioned(name,meta,mu,sd,params),dt,before); pn=info['components_after'] if info['accepted'] else before; rows.append(fast_row(pred,gt,gb,gap,gn,pn))
    return mean(rows)

def eval_full(names,base_dir,gt_dir,prob_root,meta,mu,sd,params,out_dir=None):
    rows=[]
    if out_dir: out_dir.mkdir(parents=True,exist_ok=True)
    for name in names:
        gt=read_binary_mask(gt_dir/name); base=read_binary_mask(base_dir/name); bp,sp=read_prob(prob_root,Path(name).stem); pred,info=edit(base,bp,sp,conditioned(name,meta,mu,sd,params)); r=compute_metrics(pred,gt,3,9,2.,5.); r['image']=name; r['accepted']=info['accepted']; rows.append(r)
        if out_dir: Image.fromarray(pred.astype(np.uint8)*255).save(out_dir/name)
    return mean(rows),rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gt-dir',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis/val_labels')); p.add_argument('--prob-root',type=Path,required=True); p.add_argument('--control-dir',type=Path,required=True); p.add_argument('--transfer-dir',type=Path,required=True); p.add_argument('--train-csv',type=Path,required=True); p.add_argument('--val-csv',type=Path,required=True); p.add_argument('--output-root',type=Path,default=Path('outputs/analysis/r278_age_sex_editor')); a=p.parse_args()
    train=metadata(a.train_csv); val=metadata(a.val_csv); ages=np.array([m['age'] for m in train.values()]); mu,sd=float(ages.mean()),float(ages.std()+1e-6)
    names=sorted(p.name for p in a.gt_dir.glob('*.png')); ranked=sorted(names,key=lambda x:hashlib.sha256(('R277:'+x).encode()).hexdigest()); calset=set(ranked[:48]); cal=[n for n in names if n in calset]; audit=[n for n in names if n not in calset]
    cache=load_fast_cache(cal,a.control_dir,a.gt_dir,a.prob_root); zero={'age_threshold_slope':0.,'sex_threshold_offset':0.,'age_cut_slope':0.}; baseline=eval_fast(cache,val,mu,sd,zero); grid=[]
    for age_slope in (-0.04,-0.02,0.,0.02):
      for sex_offset in (-0.03,0.,0.03):
       for cut_slope in (0.,0.20,0.35):
        q={'age_threshold_slope':age_slope,'sex_threshold_offset':sex_offset,'age_cut_slope':cut_slope}; m=eval_fast(cache,val,mu,sd,q); d=delta(m,baseline); safe=d['dice']>=-0.0005 and d['recall']>=-0.0007; score=(-4*d['component_merge_rate']-2*d['gap_region_fp_rate']+d['boundary_iou']) if safe else -1e9; grid.append({'params':q,'mean':m,'delta_vs_r277':d,'safe':safe,'score':score})
    best=max(grid,key=lambda x:x['score']); params=best['params']; result={'protocol':'same R277 calibration/audit split; conditioned versus frozen R277','clean_test_used':False,'age_mean':mu,'age_std':sd,'selected':best,'models':{}}
    for model,d in [('control_nnunet',a.control_dir),('transfer_unet',a.transfer_dir)]:
        r277_a,_=eval_full(audit,d,a.gt_dir,a.prob_root,val,mu,sd,zero); r278_a,rows=eval_full(audit,d,a.gt_dir,a.prob_root,val,mu,sd,params,a.output_root/model/'audit_masks'); result['models'][model]={'audit_r277':r277_a,'audit_r278':r278_a,'audit_delta':delta(r278_a,r277_a),'accepted':sum(r['accepted'] for r in rows)}
    a.output_root.mkdir(parents=True,exist_ok=True); (a.output_root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); (a.output_root/'grid.json').write_text(json.dumps(grid,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
