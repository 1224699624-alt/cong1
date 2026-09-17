#!/usr/bin/env python3
"""Build GT-free full-image R258B prior maps from frozen prediction-derived inputs."""
from __future__ import annotations
import argparse,csv,json,math,os,pathlib
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from train_r256_scale_invariant_pair_prior import find_image,gaussian,read_metadata,set_seed
from train_r258b_prediction_relation_prior import RelationPriorNet,load_proposals,pair_geometry,sha256,validate_provenance

def args():
 p=argparse.ArgumentParser(); p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1')); p.add_argument('--train-metadata',type=Path,default=Path('outputs/metadata/r256/filtered_train.csv')); p.add_argument('--val-metadata',type=Path,default=Path('outputs/metadata/r256/filtered_val.csv')); p.add_argument('--train-proposals',type=Path,default=Path('outputs/analysis/r258_oof_center_basin_full_proposals.csv')); p.add_argument('--val-proposals',type=Path,default=Path('outputs/analysis/r257b_scale_repair_fullval_proposals.csv')); p.add_argument('--phase-a-result',type=Path,default=Path('outputs/analysis/r258_oof_center_basin_full.json')); p.add_argument('--fold-manifest',type=Path,default=Path('outputs/analysis/r258_oof_center_basin_full_folds.csv')); p.add_argument('--inference-manifest',type=Path,default=Path('outputs/analysis/r258_oof_center_basin_full_inference.csv')); p.add_argument('--val-proposal-result',type=Path,default=Path('outputs/analysis/r257b_scale_repair_fullval.json')); p.add_argument('--r258b-result',type=Path,default=Path('outputs/analysis/r258b_prediction_relation_prior_full.json')); p.add_argument('--checkpoint-dir',type=Path,required=True); p.add_argument('--output-root',type=Path,default=Path('outputs/priors/r259_frozen_relation')); p.add_argument('--selection',type=Path); p.add_argument('--crop-size',type=int,default=128); p.add_argument('--nearest-neighbors',type=int,default=4); p.add_argument('--proposal-distance-limit',type=float,default=4.0); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--limit-train',type=int,default=0); p.add_argument('--limit-val',type=int,default=0); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--r316c',action='store_true'); p.add_argument('--r317',action='store_true'); p.add_argument('--global-size',type=int,default=128); p.add_argument('--center-sigma-rel',type=float,default=.15); p.add_argument('--center-sigma-min',type=float,default=3.); p.add_argument('--center-sigma-max',type=float,default=12.); p.add_argument('--png-only',action='store_true'); return p.parse_args()

def crop_input(image,pa,pb,psa,psb,size,r316c=False,global_size=128,sigma_rel=.15,sigma_min=3.,sigma_max=12.):
 center=.5*(pa+pb); side=int(math.ceil(max(32.,np.linalg.norm(pb-pa)+1.35*(psa+psb)))); x0,y0=int(round(center[0]-side/2)),int(round(center[1]-side/2)); x1,y1=x0+side,y0+side
 out=np.zeros((side,side),np.uint8); sx0,sy0,sx1,sy1=max(0,x0),max(0,y0),min(image.shape[1],x1),min(image.shape[0],y1)
 if sx1>sx0 and sy1>sy0: out[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=image[sy0:sy1,sx0:sx1]
 scale=size/side; ca=(pa-[x0,y0])*scale; cb=(pb-[x0,y0])*scale
 if r316c:
  from r316c_pair_inputs import pair_channels
  channels,_,_,_=pair_channels(out,ca,cb,.5*(psa+psb)*scale,size,global_size,sigma_rel,sigma_min,sigma_max,0.)
 else: channels=np.stack([cv2.resize(out,(size,size),interpolation=cv2.INTER_AREA).astype(np.float32)/255.,gaussian(size,*ca),gaussian(size,*cb)])
 return channels.astype(np.float32),(x0,y0,side)

def load_checkpoint_portable(path):
 # Historical R258B checkpoints were serialized on Linux and include a
 # pathlib.PosixPath in metadata. Map only that metadata class while loading
 # on Windows; tensor values and provenance hashes are left unchanged.
 if os.name!='nt': return torch.load(path,map_location='cpu')
 original=pathlib.PosixPath
 try:
  pathlib.PosixPath=pathlib.WindowsPath
  return torch.load(path,map_location='cpu')
 finally:
  pathlib.PosixPath=original

def main():
 a=args(); joined=' '.join(map(str,[a.variant_root,a.output_root])).lower()
 if 'clean-test' in joined or 'articular' in joined or a.variant_root.name!='TSRS_RSNA-Epiphysis_contrast_v1': raise RuntimeError('R259 permits Epiphysis contrast train/val only')
 provenance=validate_provenance(a); result=json.loads(a.r258b_result.read_text(encoding='utf-8'))
 allowed={'allow_controlled_prior_integration','no_go_relation_prior_not_validated','clean_refit_complete_gate_pass','clean_refit_complete_gate_no_go'}
 if result.get('decision') not in allowed or result.get('clean_test_used') is not False: raise RuntimeError('Unexpected frozen R258B result')
 if a.r316c:
  cfg=result.get('config',{}); expected_cfg={'r316c':True,'crop_size':256,'global_size':128,'center_sigma_rel':.15,'center_sigma_min':3.,'center_sigma_max':12.}
  if a.r317: expected_cfg['r317']=True
  if any(cfg.get(k)!=v for k,v in expected_cfg.items()) or a.crop_size!=256: raise RuntimeError('Unexpected R316C continuous-prior protocol')
 image_only=bool(result.get('config',{}).get('image_centers_only',False))
 if image_only and not a.r317: raise RuntimeError('Image-centers-only map generation is restricted to R317')
 branch='baseline' if image_only else 'development_conditioned'; model_name='image_centers' if image_only else 'image_centers_age_sex'; use_development=not image_only
 runs=result[branch]['runs']; expected={int(r['seed']):(r['checkpoint_sha256'],int(r['history'][-1]['epoch']),r['name']) for r in runs}
 if a.r317: expected={int(r['seed']):(r['checkpoint_sha256'],int(r['selected_epoch']),r['name']) for r in runs}
 if set(expected)!={2581,2582,2583}: raise RuntimeError(expected)
 checkpoints=sorted(a.checkpoint_dir.glob(f'{model_name}_seed*_selected.pt' if a.r317 else f'{model_name}_seed*_final.pt'))
 if len(checkpoints)!=3: raise RuntimeError(f'R259 requires exactly three frozen age/sex checkpoints, got {checkpoints}')
 models=[]
 checkpoint_audit=[]
 for p in checkpoints:
  payload=load_checkpoint_portable(p); seed=int(payload['seed']); digest=sha256(p); exp=expected.get(seed)
  expected_epoch=exp[1] if a.r317 and exp is not None else (6 if a.r316c else 4)
  if exp is None or digest!=exp[0] or exp[1]!=expected_epoch or exp[2]!=model_name or payload.get('use_development') is not use_development or (a.r317 and int(payload.get('selected_epoch',-1))!=expected_epoch): raise RuntimeError(f'Frozen checkpoint provenance failure: {p}')
  model=RelationPriorNet(4 if a.r316c else 3); model.load_state_dict(payload['model']); model.to(a.device).eval(); models.append(model); checkpoint_audit.append({'path':str(p),'seed':seed,'sha256':digest,'epoch':expected_epoch})
 keep=None
 if a.selection:
  selection=json.loads(a.selection.read_text(encoding='utf-8')); raw=selection.get('keep',selection); keep={k:set(map(str,raw[k])) for k in ('train','val')}
  if (len(keep['train']),len(keep['val']))!=(814,94): raise RuntimeError('Expected frozen 814/94 clean selection')
 audit={'checkpoints':checkpoint_audit,'input_provenance':provenance,'r258b_result':str(a.r258b_result),'r258b_result_sha256':sha256(a.r258b_result),'selection':str(a.selection) if a.selection else None,'splits':{}}
 for split,meta_path,proposal_path,limit in [('train',a.train_metadata,a.train_proposals,a.limit_train),('val',a.val_metadata,a.val_proposals,a.limit_val)]:
  metadata=read_metadata(meta_path); all_stems=sorted(metadata); proposals=load_proposals(proposal_path,a.variant_root,split)
  if set(proposals)!=set(all_stems) or len(all_stems)!=(875 if split=='train' else 96): raise RuntimeError(f'{split} proposal/metadata coverage mismatch')
  stems=sorted(keep[split]) if keep else all_stems
  if not set(stems)<=set(all_stems): raise RuntimeError(f'{split} clean selection not covered')
  stems=stems[:limit or None]; out_dir=a.output_root/split; out_dir.mkdir(parents=True,exist_ok=True); pair_total=0
  for n,stem in enumerate(stems,1):
   image=np.asarray(Image.open(find_image(a.variant_root/split,stem)).convert('L')); ps=proposals[stem]; pairs=set()
   for i,p in enumerate(ps):
    ranked=sorted((math.hypot(p['x']-q['x'],p['y']-q['y']),j) for j,q in enumerate(ps) if j!=i)
    for _,j in ranked[:a.nearest_neighbors]:
     x=tuple(sorted((i,j)))
     if pair_geometry(ps[x[0]],ps[x[1]])[0]<=a.proposal_distance_limit: pairs.add(x)
   pairs=sorted(pairs); pair_total+=len(pairs); full_models=[]
   for model in models:
    full=np.zeros(image.shape,np.float32)
    for start in range(0,len(pairs),a.batch_size):
     chunk=pairs[start:start+a.batch_size]; inputs=[]; placements=[]
     for i,j in chunk:
      p,q=ps[i],ps[j]; ch,place=crop_input(image,np.array([p['x'],p['y']]),np.array([q['x'],q['y']]),p['scale'],q['scale'],a.crop_size,a.r316c,a.global_size,a.center_sigma_rel,a.center_sigma_min,a.center_sigma_max); inputs.append(ch); placements.append(place)
     if not inputs: continue
     feature_row=[0.,0.] if image_only else [metadata[stem][0]/240.,metadata[stem][1]]
     feat=torch.tensor([feature_row]*len(inputs),dtype=torch.float32,device=a.device)
     with torch.no_grad(): logit,heat=model(torch.from_numpy(np.stack(inputs)).to(a.device),feat); values=(torch.sigmoid(logit)[:,None,None,None]*torch.sigmoid(heat)).cpu().numpy()[:,0]
     for value,(x0,y0,side) in zip(values,placements):
      patch=cv2.resize(value,(side,side),interpolation=cv2.INTER_LINEAR); sx0,sy0,sx1,sy1=max(0,x0),max(0,y0),min(image.shape[1],x0+side),min(image.shape[0],y0+side)
      if sx1>sx0 and sy1>sy0:
       local=patch[sy0-y0:sy1-y0,sx0-x0:sx1-x0]; full[sy0:sy1,sx0:sx1]=np.maximum(full[sy0:sy1,sx0:sx1],local)
    full_models.append(full)
   prior=np.mean(full_models,axis=0).astype(np.float32)
   if not a.png_only: np.save(out_dir/f'{stem}.npy',prior)
   Image.fromarray(np.clip(prior*255,0,255).astype(np.uint8)).save(out_dir/f'{stem}.png')
   if n%50==0 or n==len(stems): print(json.dumps({'split':split,'done':n,'total':len(stems),'pairs':pair_total}),flush=True)
  nonzero=sum(bool(np.asarray(Image.open(out_dir/f'{s}.png')).max()>0) for s in stems)
  if nonzero!=len(stems): raise RuntimeError(f'{split} has zero/nonfinite prior maps: {nonzero}/{len(stems)}')
  audit['splits'][split]={'images':len(stems),'pairs':pair_total,'nonzero_maps':nonzero}
 a.output_root.mkdir(parents=True,exist_ok=True); (a.output_root/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
