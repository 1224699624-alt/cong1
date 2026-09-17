#!/usr/bin/env python3
"""R265: age/sex-conditioned hierarchical carpal relation priors.

Slots 0-3 are carpal-carpal, slot 4 is carpal-metacarpal-base, and slot 5
is carpal-radius/ulna. Validation labels are never opened.
"""
from __future__ import annotations
import argparse,json,math,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

from build_r261_conditional_seam_priors import Bank,features,find_image,load_proposals,read_metadata
from build_r262_metacarpal_valley_priors import refine_center,palm_subset,delaunay_edges,is_adjacent,localize_valley,maps

TYPE_SLOTS={'carpal_carpal':4,'carpal_metacarpal':1,'carpal_radius_ulna':1}
TYPE_WEIGHT={'carpal_carpal':1.0,'carpal_metacarpal':0.45,'carpal_radius_ulna':0.15}

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1'))
 p.add_argument('--train-metadata',type=Path,default=Path('outputs/metadata/r256/filtered_train.csv'))
 p.add_argument('--val-metadata',type=Path,default=Path('outputs/metadata/r256/filtered_val.csv'))
 p.add_argument('--train-proposals',type=Path,default=Path('outputs/analysis/r258_oof_center_basin_full_proposals.csv'))
 p.add_argument('--val-proposals',type=Path,default=Path('outputs/analysis/r257b_scale_repair_fullval_proposals.csv'))
 p.add_argument('--bank-json',type=Path,default=Path('outputs/analysis/r261_conditional_seam_bank.json'))
 p.add_argument('--output-root',type=Path,default=Path('outputs/priors/r265_hierarchical_carpal'))
 p.add_argument('--min-valley-contrast',type=float,default=.015)
 p.add_argument('--min-confidence',type=float,default=.06)
 p.add_argument('--limit-train',type=int,default=0); p.add_argument('--limit-val',type=int,default=0)
 p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def zone(v,scale,carpal_scale):
 # Radius/ulna proposals must be both proximal and substantially larger than
 # the carpal ossification centers. This prevents a rotated proximal-row
 # carpal from being mislabeled as a weak CR relation.
 if .90<v<=1.03 and scale>=1.35*carpal_scale:return 'radius_ulna'
 if .58<=v<.70 and scale>=.65*carpal_scale:return 'metacarpal_base'
 if .68<=v<=.93 and scale<=1.60*carpal_scale:return 'carpal'
 return None

def relation_type(za,zb):
 pair={za,zb}
 if za==zb=='carpal':return 'carpal_carpal'
 if pair=={'carpal','metacarpal_base'}:return 'carpal_metacarpal'
 if pair=={'carpal','radius_ulna'}:return 'carpal_radius_ulna'
 return None

def build_case(image,raw,age,male,bank,args):
 items=[refine_center(image,x) for x in raw]
 all_idx,frame,vn=palm_subset(items,0.,1.03)
 preliminary=[items[i]['scale'] for i,v in enumerate(vn) if .68<=float(v)<=.93]
 carpal_scale=float(np.median(preliminary)) if preliminary else float(np.median([x['scale'] for x in items]))
 zones=[zone(float(v),float(items[i]['scale']),carpal_scale) for i,v in enumerate(vn)]
 roi=[i for i in all_idx if zones[i] is not None]
 candidates=defaultdict(list)
 for i,j in delaunay_edges(items,roi,frame):
  kind=relation_type(zones[i],zones[j])
  if kind is None or not is_adjacent(items,i,j,roi):continue
  f=features(items[i],items[j],frame)
  if f[0]>4.5:continue
  if kind in {'carpal_metacarpal','carpal_radius_ulna'} and f[5]<.55:continue
  loc=localize_valley(image,items[i],items[j])
  if loc is None or loc['contrast']<args.min_valley_contrast:continue
  prior,support_prior,relation=bank.score(f,age,male)
  proposal=math.sqrt(max(items[i].get('score',0),0)*max(items[j].get('score',0),0))
  confidence=float(np.clip(prior*proposal*loc['evidence'],0,1))
  if confidence<args.min_confidence:continue
  candidates[kind].append({'i':i,'j':j,'type':kind,'type_weight':TYPE_WEIGHT[kind],
   'confidence':confidence,'prior':prior,'support_prior':support_prior,'relation':relation,
   'proposal_confidence':proposal,**{k:v for k,v in loc.items() if k!='profile'}})
 selected=[]
 for kind,count in TYPE_SLOTS.items():
  chosen=sorted(candidates[kind],key=lambda x:x['confidence'],reverse=True)[:count]
  selected.extend(chosen+[None]*(count-len(chosen)))
 slots=[]
 for x in selected:
  if x is None: slots.append(None)
  else: slots.append((*maps(image.shape,items[x['i']],items[x['j']],x,x['confidence']),x))
 audit={'all_proposals':len(items),'roi_candidates':len(roi),'zone_counts':{z:zones.count(z) for z in ('metacarpal_base','carpal','radius_ulna')},
  'carpal_scale':carpal_scale,'candidate_counts':{k:len(v) for k,v in candidates.items()},'selected_counts':{k:sum(x is not None and x[2]['type']==k for x in slots) for k in TYPE_SLOTS},
  'selected':sum(x is not None for x in slots),'abstained_slots':sum(x is None for x in slots)}
 return slots,audit

def save_split(split,root,meta_path,proposal_path,bank,args,limit):
 metadata=read_metadata(meta_path); proposals=load_proposals(proposal_path,root,split)
 if not set(metadata)<=set(proposals):raise RuntimeError(f'missing {split} proposals')
 stems=sorted(metadata)[:limit or None]; out=args.output_root/split
 if out.exists() and args.overwrite:shutil.rmtree(out)
 out.mkdir(parents=True,exist_ok=True); totals=defaultdict(float)
 for n,stem in enumerate(stems,1):
  image=np.asarray(Image.open(find_image(root/split,stem)).convert('L'),np.float32)/255.
  age,male=metadata[stem]; slots,audit=build_case(image,proposals[stem],age,male,bank,args); records=[]
  for k,x in enumerate(slots):
   if x is None: seam=support=np.zeros(image.shape,np.float32)
   else: seam,support,info=x; records.append({'slot':k,**info})
   Image.fromarray(np.clip(seam*255,0,255).astype(np.uint8)).save(out/f'{stem}_seam{k}.png',optimize=True)
   Image.fromarray(np.clip(support*255,0,255).astype(np.uint8)).save(out/f'{stem}_support{k}.png',optimize=True)
  (out/f'{stem}_pairs.json').write_text(json.dumps({'stem':stem,'boneage':age,'male':bool(male),'audit':audit,'slots':records},indent=2,default=lambda x:np.asarray(x).tolist()),encoding='utf-8')
  totals['selected']+=audit['selected']; totals['abstained_slots']+=audit['abstained_slots']
  for kind,value in audit['selected_counts'].items():totals[kind]+=value
  if n%100==0 or n==len(stems):print(json.dumps({'split':split,'done':n,'total':len(stems)}),flush=True)
 return {'images':len(stems),**{k:float(v) for k,v in totals.items()},'mean_selected':float(totals['selected']/max(1,len(stems)))}

def main():
 a=parse_args(); joined=' '.join(map(str,vars(a).values())).lower()
 if a.variant_root.name!='TSRS_RSNA-Epiphysis_contrast_v1' or 'clean-test' in joined or 'articular' in joined:raise RuntimeError('R265 original Epiphysis train/original-val only')
 train_meta=read_metadata(a.train_metadata); val_meta=read_metadata(a.val_metadata)
 if set(train_meta)&set(val_meta):raise RuntimeError('train/val identity overlap')
 payload=json.loads(a.bank_json.read_text(encoding='utf-8'))
 if payload.get('num_cases')!=875 or 'train_labels' not in payload.get('source_label_dir',''):raise RuntimeError('full train-only bank required')
 bank=Bank(payload,24.); a.output_root.mkdir(parents=True,exist_ok=True)
 manifest={'run_id':'R265','clean_test_used':False,'slot_schema':TYPE_SLOTS,'type_weights':TYPE_WEIGHT,
  'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'splits':{}}
 manifest['splits']['train']=save_split('train',a.variant_root,a.train_metadata,a.train_proposals,bank,a,a.limit_train)
 manifest['splits']['val']=save_split('val',a.variant_root,a.val_metadata,a.val_proposals,bank,a,a.limit_val)
 (a.output_root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
