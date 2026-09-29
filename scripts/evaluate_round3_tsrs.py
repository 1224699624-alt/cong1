"""Frozen R201 metrics with strict full-case/shape and checkpoint binding."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from run_r201_unified_eval import compute_metrics
from round3_common import sha,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--gt',type=Path,required=True);p.add_argument('--pred',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluator-source',type=Path,required=True);a=p.parse_args()
    if 'test' in a.gt.name.lower():raise ValueError('val only')
    gt={p.stem:p for p in a.gt.glob('*.png')};pred={p.stem.removeprefix('val_'):p for p in a.pred.glob('*.png')}
    if len(gt)!=96 or set(gt)!=set(pred):raise ValueError('full val IDs differ')
    rows=[]
    for key,path in sorted(gt.items()):
        y=np.asarray(Image.open(path));z=np.asarray(Image.open(pred[key]))
        if y.ndim!=2 or z.ndim!=2 or y.shape!=z.shape:raise ValueError(f'shape/mode mismatch {key}')
        rows.append({'case':key,**compute_metrics(z>0,y>0,3,9,2.,5.)})
    means={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k!='case'}
    if a.output.exists():raise FileExistsError(a.output)
    save(a.output,{'status':'COMPLETE','cases':96,'mean':means,'per_case':rows,'checkpoint':str(a.checkpoint),
                   'checkpoint_sha256':sha(a.checkpoint),'evaluator_sha256':sha(a.evaluator_source),
                   'case_shape_strict':True,'test_used':False})
    print(json.dumps(means),flush=True)
if __name__=='__main__':main()
