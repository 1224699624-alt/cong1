#!/usr/bin/env python3
"""Build train-label-only conditional relation bank and GT-free per-seam maps."""
from __future__ import annotations
import argparse,csv,json,math,shutil
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
from PIL import Image
from scipy.spatial import cKDTree

FEATURE_NAMES=("relative_distance","abs_log_scale_ratio","mid_x","mid_y","abs_dir_x","abs_dir_y","gap_proxy")
EXTENSIONS=(".jpg",".jpeg",".png",".bmp")

def read_metadata(path:Path):
 with path.open(newline='',encoding='utf-8-sig') as f:
  return {str(r['id']).strip():(float(r['boneage']),float(str(r['male']).strip().lower() in {'true','1','yes'})) for r in csv.DictReader(f)}

def find_image(folder:Path,stem:str):
 for ext in EXTENSIONS:
  p=folder/f'{stem}{ext}'
  if p.exists() and p.stat().st_size>0:return p
 raise FileNotFoundError(stem)

def load_proposals(path:Path,root:Path,split:str):
 grouped=defaultdict(list); shapes={}
 with path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 if not rows or 'basin_scale_original' not in rows[0]: raise RuntimeError(f'Invalid proposal CSV: {path}')
 if split=='train' and 'fold' not in rows[0]: raise RuntimeError('Train proposals must be OOF')
 for row in rows:
  stem=str(row['stem'])
  if row.get('x_original','') and row.get('y_original',''): x,y=float(row['x_original']),float(row['y_original'])
  else:
   if stem not in shapes:
    with Image.open(find_image(root/split,stem)) as im: shapes[stem]=(im.height,im.width)
   h,w=shapes[stem]; ratio=min(384/h,384/w); rh,rw=max(1,int(round(h*ratio))),max(1,int(round(w*ratio))); py,px=(384-rh)//2,(384-rw)//2; x=(float(row['x_letterbox'])-px)/(rw/w); y=(float(row['y_letterbox'])-py)/(rh/h)
  grouped[stem].append({'x':x,'y':y,'scale':float(row['basin_scale_original']),'score':float(row.get('score',0.)),'rank':float(row.get('rank',0)),'fold':float(row.get('fold',-1))})
 return grouped

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1')); p.add_argument('--train-label-dir',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis/train_labels')); p.add_argument('--train-metadata',type=Path,default=Path('outputs/metadata/r256/filtered_train.csv')); p.add_argument('--val-metadata',type=Path,default=Path('outputs/metadata/r256/filtered_val.csv')); p.add_argument('--train-proposals',type=Path,default=Path('outputs/analysis/r258_oof_center_basin_full_proposals.csv')); p.add_argument('--val-proposals',type=Path,default=Path('outputs/analysis/r257b_scale_repair_fullval_proposals.csv')); p.add_argument('--output-root',type=Path,default=Path('outputs/priors/r261_conditional_seams')); p.add_argument('--bank-json',type=Path,default=Path('outputs/analysis/r261_conditional_seam_bank.json')); p.add_argument('--slots',type=int,default=6); p.add_argument('--neighbors',type=int,default=3); p.add_argument('--preselect',type=int,default=24); p.add_argument('--age-bandwidth',type=float,default=24.0); p.add_argument('--min-confidence',type=float,default=.10); p.add_argument('--limit-bank',type=int,default=0,help='Sanity only; full protocol requires 0'); p.add_argument('--limit-train',type=int,default=0); p.add_argument('--limit-val',type=int,default=0); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def read_instance(path:Path):
 a=np.asarray(Image.open(path)); return (a[...,0] if a.ndim==3 else a).astype(np.int32)

def records(label,min_area=24):
 out=[]
 for v in np.unique(label):
  if int(v)<=0: continue
  y,x=np.nonzero(label==v)
  if len(x)<min_area: continue
  out.append({'x':float(x.mean()),'y':float(y.mean()),'scale':float(math.sqrt(len(x))),'area':float(len(x))})
 return out

def canonical(points:np.ndarray):
 center=np.median(points,axis=0); q=points-center
 if len(points)>=3:
  _,_,vh=np.linalg.svd(q,full_matrices=False); ey=vh[0]
 else: ey=np.asarray([0.,1.])
 if ey[1]<0: ey=-ey
 ex=np.asarray([ey[1],-ey[0]])
 if ex[0]<0: ex=-ex
 axes=np.stack([ex,ey],axis=1); z=q@axes
 span=np.maximum(np.percentile(np.abs(z),90,axis=0),1.0)
 return center,axes,span

def features(a,b,frame):
 center,axes,span=frame; pa=np.asarray([a['x'],a['y']]); pb=np.asarray([b['x'],b['y']]); za=(pa-center)@axes/span; zb=(pb-center)@axes/span; d=zb-za; dist=float(np.linalg.norm(d)); scale=max(math.sqrt(.5*(a['scale']**2+b['scale']**2))/math.sqrt(float(span.prod())),1e-4); unit=d/max(dist,1e-6)
 return np.asarray([dist/max(scale,1e-4),abs(math.log(max(a['scale'],1e-4)/max(b['scale'],1e-4))),*((za+zb)*.5),abs(unit[0]),abs(unit[1]),max(0.,dist-.5*(a['scale']+b['scale'])/max(float(span.mean()),1.))],np.float64)

def nearest_pairs(items,n=3):
 if len(items)<2:return []
 p=np.asarray([[x['x'],x['y']] for x in items]); d=np.linalg.norm(p[:,None]-p[None,:],axis=-1); np.fill_diagonal(d,np.inf); pairs=set()
 for i in range(len(items)):
  for j in np.argsort(d[i])[:n]: pairs.add(tuple(sorted((i,int(j)))))
 return sorted(pairs)

def build_bank(label_dir,metadata,neighbors,limit=0):
 samples=[]; used=set()
 for lp in sorted(label_dir.glob('*.png'))[:limit or None]:
  stem=lp.stem
  if stem not in metadata: continue
  rs=records(read_instance(lp))
  if len(rs)<2: continue
  frame=canonical(np.asarray([[r['x'],r['y']] for r in rs]))
  age,male=metadata[stem]
  for i,j in nearest_pairs(rs,neighbors): samples.append({'case':stem,'boneage':age,'male':bool(male),'features':features(rs[i],rs[j],frame).tolist()})
  used.add(stem)
 x=np.asarray([s['features'] for s in samples]); return {'run_id':'R261','feature_names':FEATURE_NAMES,'source_label_dir':str(label_dir),'num_cases':len(used),'num_samples':len(samples),'global_mean':x.mean(0).tolist(),'global_std':np.maximum(x.std(0),1e-4).tolist(),'samples':samples,'leakage_guard':'Only original train instance labels and train metadata.'}

class Bank:
 def __init__(self,payload,bandwidth):
  self.x=np.asarray([s['features'] for s in payload['samples']],np.float64); self.age=np.asarray([s['boneage'] for s in payload['samples']],np.float64); self.male=np.asarray([s['male'] for s in payload['samples']],bool); self.case=np.asarray([s['case'] for s in payload['samples']]); self.std=np.maximum(np.asarray(payload['global_std']),1e-4); self.mean=np.asarray(payload['global_mean']); self.bandwidth=bandwidth; counts=dict(zip(*np.unique(self.case,return_counts=True))); self.balance=np.asarray([1/counts[c] for c in self.case]); self.tree=cKDTree((self.x-self.mean)/self.std); unique=np.unique(self.case); first=np.asarray([np.flatnonzero(self.case==c)[0] for c in unique]); self.case_age=self.age[first]; self.case_male=self.male[first]; self.support_cache={}
 def score(self,f,age,male):
  key=(round(float(age),3),bool(male))
  if key not in self.support_cache:
   cw=np.exp(-.5*((self.case_age-age)/self.bandwidth)**2)*np.where(self.case_male==bool(male),1.,.25); self.support_cache[key]=float(np.clip((cw.sum()**2/(np.square(cw).sum()+1e-8))/80.,0,1))
  support=self.support_cache[key]; k=min(128,len(self.x)); distance,index=self.tree.query((f-self.mean)/self.std,k=k); index=np.atleast_1d(index); z2=np.square(np.atleast_1d(distance))/len(FEATURE_NAMES); w=np.exp(-.5*((self.age[index]-age)/self.bandwidth)**2)*np.where(self.male[index]==bool(male),1.,.25)*self.balance[index]; relation=float(np.sum(w*np.exp(-.5*np.clip(z2,0,30)))/(w.sum()+1e-8)); return support*relation,support,relation

def disk_mean(image,p,r):
 y,x=np.ogrid[:image.shape[0],:image.shape[1]]; m=(x-p[0])**2+(y-p[1])**2<=max(r,1.)**2; return float(np.median(image[m])) if m.any() else float(image.mean())

def image_evidence(image,a,b):
 pa=np.asarray([a['x'],a['y']]); pb=np.asarray([b['x'],b['y']]); mid=.5*(pa+pb); r=max(2.,.12*min(a['scale'],b['scale'])); bone=.5*(disk_mean(image,pa,r)+disk_mean(image,pb,r)); gap=disk_mean(image,mid,r); scale=float(np.std(image))+1e-6; return float(1/(1+math.exp(-np.clip((bone-gap)/(.20*scale),-20,20))))

def slot_maps(shape,a,b,confidence):
 h,w=shape; pa=np.asarray([a['x'],a['y']]); pb=np.asarray([b['x'],b['y']]); delta=pb-pa; dist=max(float(np.linalg.norm(delta)),1e-6); n=delta/dist; t=np.asarray([-n[1],n[0]]); mid=.5*(pa+pb); local=max(3.,min(a['scale'],b['scale'])); sigma_n=max(1.,.07*local); sigma_t=max(2.,.48*local); qa=pa+.28*delta; qb=pb-.28*delta; sig=max(2.,.22*local); margin=4*max(sigma_t,sig); x0=max(0,int(math.floor(min(mid[0],qa[0],qb[0])-margin))); x1=min(w,int(math.ceil(max(mid[0],qa[0],qb[0])+margin+1))); y0=max(0,int(math.floor(min(mid[1],qa[1],qb[1])-margin))); y1=min(h,int(math.ceil(max(mid[1],qa[1],qb[1])+margin+1))); seam_full=np.zeros((h,w),np.float32); support_full=np.zeros((h,w),np.float32)
 if x1<=x0 or y1<=y0:return seam_full,support_full
 yy,xx=np.mgrid[y0:y1,x0:x1]; rx=xx-mid[0]; ry=yy-mid[1]; dn=rx*n[0]+ry*n[1]; dt=rx*t[0]+ry*t[1]; seam=np.exp(-.5*((dn/sigma_n)**2+(dt/sigma_t)**2)); support=np.exp(-((xx-qa[0])**2+(yy-qa[1])**2)/(2*sig**2))+np.exp(-((xx-qb[0])**2+(yy-qb[1])**2)/(2*sig**2)); support=np.clip(support,0,1)*(seam<.55); seam_full[y0:y1,x0:x1]=confidence*seam; support_full[y0:y1,x0:x1]=confidence*support; return seam_full,support_full

def proposal_records(ps): return [{'x':float(p['x']),'y':float(p['y']),'scale':float(p['scale']),'proposal_score':float(p.get('score',1.))} for p in ps]

def build_case(image,ps,age,male,bank,args):
 rs=proposal_records(ps); maps=[]; audit={'proposals':len(rs),'pairs':0,'preselected':0,'selected':0,'abstained':False}
 if len(rs)<2:return maps,audit
 frame=canonical(np.asarray([[r['x'],r['y']] for r in rs])); stage=[]
 for i,j in nearest_pairs(rs,args.neighbors):
  f=features(rs[i],rs[j],frame)
  if f[0]>4.0: continue
  prior,support,relation=bank.score(f,age,male); stage.append((prior,i,j,support,relation,f))
 audit['pairs']=len(stage); stage=sorted(stage,reverse=True,key=lambda z:z[0])[:args.preselect]; audit['preselected']=len(stage); rescored=[]
 for prior,i,j,support,relation,f in stage:
  evidence=image_evidence(image,rs[i],rs[j]); proposal=math.sqrt(max(rs[i]['proposal_score'],0)*max(rs[j]['proposal_score'],0)); agreement=max(0.,1.-abs(relation-evidence)); confidence=float(np.clip(max(prior,0)*max(evidence,0)*max(proposal,0)*agreement,0,1)); rescored.append((confidence,i,j,prior,evidence,proposal,support,relation))
 selected=[x for x in sorted(rescored,reverse=True) if x[0]>=args.min_confidence][:args.slots]; audit['selected']=len(selected); audit['abstained']=len(selected)==0; audit['confidence']=[x[0] for x in selected]
 for confidence,i,j,*_ in selected: maps.append((*slot_maps(image.shape,rs[i],rs[j],confidence),{'i':i,'j':j,'confidence':confidence,'a':rs[i],'b':rs[j]}))
 return maps,audit

def save_split(split,root,meta_path,proposal_path,bank,args,limit):
 metadata=read_metadata(meta_path); proposals=load_proposals(proposal_path,root,split)
 if not set(metadata)<=set(proposals): raise RuntimeError(f'missing {split} proposals')
 stems=sorted(metadata)[:limit or None]; out=args.output_root/split
 if out.exists() and args.overwrite: shutil.rmtree(out)
 out.mkdir(parents=True,exist_ok=True); totals=defaultdict(float)
 for n,stem in enumerate(stems,1):
  image=np.asarray(Image.open(find_image(root/split,stem)).convert('L'),np.float32)/255.; age,male=metadata[stem]; slots,audit=build_case(image,proposals[stem],age,male,bank,args)
  manifest=[]
  for k in range(args.slots):
   if k<len(slots): seam,support,info=slots[k]; manifest.append(info)
   else: seam=support=np.zeros(image.shape,np.float32)
   Image.fromarray(np.clip(seam*255,0,255).astype(np.uint8)).save(out/f'{stem}_seam{k}.png',optimize=True); Image.fromarray(np.clip(support*255,0,255).astype(np.uint8)).save(out/f'{stem}_support{k}.png',optimize=True)
  (out/f'{stem}_pairs.json').write_text(json.dumps({'stem':stem,'boneage':age,'male':bool(male),'audit':audit,'slots':manifest},indent=2),encoding='utf-8')
  for key in ('proposals','pairs','preselected','selected'): totals[key]+=audit[key]
  totals['abstained']+=float(audit['abstained'])
  if n%100==0 or n==len(stems): print(json.dumps({'split':split,'done':n,'total':len(stems)}),flush=True)
 return {'images':len(stems),**{k:float(v) for k,v in totals.items()},'mean_selected':float(totals['selected']/max(len(stems),1))}

def main():
 a=parse_args(); joined=' '.join(map(str,[a.variant_root,a.train_label_dir,a.output_root,a.train_proposals,a.val_proposals])).lower()
 if a.variant_root.name!='TSRS_RSNA-Epiphysis_contrast_v1' or 'clean-test' in joined or 'articular' in joined or a.train_label_dir.name!='train_labels': raise RuntimeError('R261 permits original Epiphysis train/original-val only')
 train_meta=read_metadata(a.train_metadata); val_meta=read_metadata(a.val_metadata)
 if set(train_meta)&set(val_meta): raise RuntimeError('train/val identity overlap')
 payload=build_bank(a.train_label_dir,train_meta,a.neighbors,a.limit_bank); a.bank_json.parent.mkdir(parents=True,exist_ok=True); a.bank_json.write_text(json.dumps(payload,indent=2),encoding='utf-8'); bank=Bank(payload,a.age_bandwidth)
 audit={'run_id':'R261','clean_test_used':False,'bank':{'num_cases':payload['num_cases'],'num_samples':payload['num_samples'],'source_label_dir':payload['source_label_dir']},'config':vars(a),'splits':{}}
 audit['config']={k:str(v) if isinstance(v,Path) else v for k,v in audit['config'].items()}; audit['splits']['train']=save_split('train',a.variant_root,a.train_metadata,a.train_proposals,bank,a,a.limit_train); audit['splits']['val']=save_split('val',a.variant_root,a.val_metadata,a.val_proposals,bank,a,a.limit_val); (a.output_root/'manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
