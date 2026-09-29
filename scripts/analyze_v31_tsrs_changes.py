"""Pixel/case attribution for A -> H_input -> B on TSRS val."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from round3_common import sha

def load_mask(path):
    x=np.asarray(Image.open(path));
    if x.ndim!=2:raise ValueError(path)
    return x>0
def prior(path):
    x=np.asarray(Image.open(path),dtype=np.float32);lo=x.min();hi=x.max();return (x-lo)/max(float(hi-lo),1e-6)
def compare(left,right,gt,support):
    deleted=left&~right;added=~left&right;high=support>=.25
    regions={'trusted_seam_background':~gt&high,'high_prior_true_foreground':gt&high,
             'ordinary_true_foreground':gt&~high,'other_background':~gt&~high}
    result={'delete_tp':int((deleted&gt).sum()),'delete_fp':int((deleted&~gt).sum()),
            'add_tp':int((added&gt).sum()),'add_fp':int((added&~gt).sum()),
            'regions':{name:{'delete':int((deleted&mask).sum()),'add':int((added&mask).sum()),
                              'area':int(mask.sum())} for name,mask in regions.items()}}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--round3',type=Path,required=True);p.add_argument('--h',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    paths={'A':a.round3/'tsrs/A/predictions','H':a.h/'predictions','B':a.round3/'tsrs/B/predictions'}
    gt_dir=a.root/'data/raw/TSRS_RSNA-Epiphysis/val_labels';prior_dir=a.root/'outputs/priors/r317_image_centers_only/val'
    ids=sorted(p.stem for p in gt_dir.glob('*.png'))
    rows=[];totals={key:Counter() for key in ['H_minus_A','B_minus_H']}
    for case in ids:
        gt=load_mask(gt_dir/(case+'.png'));s=prior(prior_dir/(case+'.png'));pred={k:load_mask(v/('val_'+case+'.png')) for k,v in paths.items()}
        if any(x.shape!=gt.shape for x in [s,*pred.values()]):raise ValueError(case)
        comps={'H_minus_A':compare(pred['A'],pred['H'],gt,s),'B_minus_H':compare(pred['H'],pred['B'],gt,s)}
        for key,value in comps.items():
            for metric in ['delete_tp','delete_fp','add_tp','add_fp']:totals[key][metric]+=value[metric]
            for region,v in value['regions'].items():
                for action in ['delete','add']:totals[key][region+'_'+action]+=v[action]
        rows.append({'case':case,**comps})
    result={'status':'COMPLETE','cases':len(ids),'definitions':{'H_minus_A':'pure heatmap input effect','B_minus_H':'separation+preserve+routing package effect',
        'high_prior_true_foreground':'GT foreground AND per-image normalized seam>=.25','trusted_seam_background':'GT background AND normalized seam>=.25'},
        'totals':{k:dict(v) for k,v in totals.items()},'per_case':rows,'sources':{k:{'path':str(v),'sha256_count':len(list(v.glob('*.png')))} for k,v in paths.items()},'test_used':False}
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result['totals'],indent=2))
if __name__=='__main__':main()
