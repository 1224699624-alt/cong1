"""R277 model-agnostic post-hoc seam editor with calibration/audit isolation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from scipy import ndimage
from run_r201_unified_eval import boundary, compute_metrics, read_binary_mask

def read_prob(root: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    hits = list(root.rglob(f'val_{stem}.npz'))
    if len(hits) != 1: raise RuntimeError(f'expected one probability file for {stem}, got {hits}')
    p = np.load(hits[0])['probabilities']
    if p.ndim == 4: p = p[:, 0]
    return p[1].astype(np.float32), p[2].astype(np.float32)

def component_count(mask: np.ndarray, min_area: int = 20) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return int(sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n)))

def edit(mask: np.ndarray, bone_p: np.ndarray, seam_p: np.ndarray, cfg: dict,
         dt: np.ndarray | None = None, before: int | None = None) -> tuple[np.ndarray, dict]:
    dt = ndimage.distance_transform_edt(mask) if dt is None else dt
    cut = mask & (seam_p >= cfg['threshold']) & (bone_p <= cfg['bone_max']) & (dt <= cfg['depth'])
    if cfg['dilate']:
        cut = ndimage.binary_dilation(cut, iterations=cfg['dilate']) & mask
    max_pixels = max(1, int(round(mask.sum() * cfg['max_cut_fraction'])))
    if cut.sum() > max_pixels:
        yy, xx = np.nonzero(cut); order = np.argsort(seam_p[yy, xx])[::-1][:max_pixels]
        limited = np.zeros_like(cut); limited[yy[order], xx[order]] = True; cut = limited
    proposed = mask & ~cut
    before = component_count(mask) if before is None else before
    after = component_count(proposed)
    accepted = cut.any() and 1 <= after - before <= cfg['max_component_increase']
    return (proposed if accepted else mask), {'cut_pixels': int(cut.sum()), 'accepted': bool(accepted), 'components_before': before, 'components_after': after}

def mean(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in {'image','edit'}]
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}

def evaluate(names, base_dir, gt_dir, prob_root, cfg, out_dir=None):
    rows=[]
    if out_dir: out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        stem=Path(name).stem; gt=read_binary_mask(gt_dir/name); base=read_binary_mask(base_dir/name)
        bp,sp=read_prob(prob_root,stem); edited,info=edit(base,bp,sp,cfg)
        m=compute_metrics(edited,gt,3,9,2.0,5.0); m.update({'image':name,'edit':info}); rows.append(m)
        if out_dir: Image.fromarray(edited.astype(np.uint8)*255).save(out_dir/name)
    return mean(rows),rows

def delta(new, old): return {k: float(new[k]-old[k]) for k in old}

def load_fast_cache(names, base_dir, gt_dir, prob_root):
    cache=[]; gap_structure=np.ones((9,9),dtype=bool)
    for name in names:
        gt=read_binary_mask(gt_dir/name); base=read_binary_mask(base_dir/name); bp,sp=read_prob(prob_root,Path(name).stem)
        _,gt_n=ndimage.label(gt)
        cache.append((name,base,gt,bp,sp,boundary(gt,3),ndimage.binary_dilation(gt,structure=gap_structure)&~gt,int(gt_n),ndimage.distance_transform_edt(base),component_count(base)))
    return cache

def fast_row(pred, gt, gt_boundary, gap, gt_n, pred_n):
    tp=float((pred&gt).sum()); fp=float((pred&~gt).sum()); fn=float((~pred&gt).sum())
    pb=boundary(pred,3); bi=float((pb&gt_boundary).sum()); bu=float((pb|gt_boundary).sum())
    return {'dice':2*tp/max(1,2*tp+fp+fn),'recall':tp/max(1,tp+fn),'boundary_iou':bi/max(1,bu),'gap_region_fp_rate':float((pred&gap).sum())/max(1,float(gap.sum())),'component_merge_rate':float(pred_n<gt_n)}

def evaluate_fast(cache,cfg):
    rows=[]
    for _,base,gt,bp,sp,gb,gap,gn,dt,before in cache:
        pred,info=edit(base,bp,sp,cfg,dt,before); pred_n=info['components_after'] if info['accepted'] else before
        rows.append(fast_row(pred,gt,gb,gap,gn,pred_n))
    return mean(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gt-dir',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis/val_labels')); p.add_argument('--prob-root',type=Path,required=True); p.add_argument('--control-dir',type=Path,required=True); p.add_argument('--transfer-dir',type=Path,required=True); p.add_argument('--output-root',type=Path,default=Path('outputs/analysis/r277_decoupled_editor')); a=p.parse_args()
    if 'clean-test' in str(a).lower() or 'articular' in str(a).lower(): raise RuntimeError('Epiphysis original-val only')
    names=sorted(p.name for p in a.gt_dir.glob('*.png')); ranked=sorted(names,key=lambda x:hashlib.sha256(('R277:'+x).encode()).hexdigest()); calibration=set(ranked[:48]); cal=[n for n in names if n in calibration]; audit=[n for n in names if n not in calibration]
    base_cfg={'threshold':1.1,'bone_max':0.0,'depth':1,'dilate':0,'max_cut_fraction':0.003,'max_component_increase':2}
    cal_cache=load_fast_cache(cal,a.control_dir,a.gt_dir,a.prob_root)
    base_cal=evaluate_fast(cal_cache,base_cfg)
    grid=[]
    for t in (0.25,0.40):
      for bm in (0.50,0.70):
       for depth in (2,4):
        for dil in (0,1):
         cfg={'threshold':t,'bone_max':bm,'depth':depth,'dilate':dil,'max_cut_fraction':0.003,'max_component_increase':2}
         m=evaluate_fast(cal_cache,cfg); d=delta(m,base_cal)
         safe=d['dice']>=-0.001 and d['recall']>=-0.0015
         score=(-4*d['component_merge_rate']-2*d['gap_region_fp_rate']+d['boundary_iou']) if safe else -1e9
         grid.append({'config':cfg,'mean':m,'delta':d,'safe':safe,'score':score})
    best=max(grid,key=lambda x:x['score']); cfg=best['config']
    payload={'protocol':'R277 48-case calibration + 48-case held-out audit','clean_test_used':False,'selected':best,'models':{}}
    for model,base_dir in [('control_nnunet',a.control_dir),('transfer_unet',a.transfer_dir)]:
        model_out=a.output_root/model; base_a,_=evaluate(audit,base_dir,a.gt_dir,a.prob_root,base_cfg); edit_a,rows=evaluate(audit,base_dir,a.gt_dir,a.prob_root,cfg,model_out/'audit_masks'); base_f,_=evaluate(names,base_dir,a.gt_dir,a.prob_root,base_cfg); edit_f,_=evaluate(names,base_dir,a.gt_dir,a.prob_root,cfg,model_out/'full_masks')
        payload['models'][model]={'audit_base':base_a,'audit_edited':edit_a,'audit_delta':delta(edit_a,base_a),'full_base':base_f,'full_edited':edit_f,'full_delta':delta(edit_f,base_f),'accepted_audit':sum(r['edit']['accepted'] for r in rows)}
    a.output_root.mkdir(parents=True,exist_ok=True); (a.output_root/'result.json').write_text(json.dumps(payload,indent=2),encoding='utf-8'); (a.output_root/'grid.json').write_text(json.dumps(grid,indent=2),encoding='utf-8'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
