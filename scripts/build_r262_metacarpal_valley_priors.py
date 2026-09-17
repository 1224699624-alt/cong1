#!/usr/bin/env python3
"""R262: palm/metacarpal-restricted adjacent-pair priors with image-valley localization.

Blue overlay markers are refined proposal centers. Green markers are image-derived
bone-side support peaks. Red segments are perpendicular seam valleys. Validation
labels are never read.
"""
from __future__ import annotations
import argparse,csv,json,math,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay
from build_r261_conditional_seam_priors import Bank,canonical,features,find_image,load_proposals,read_metadata

def parse_args():
 p=argparse.ArgumentParser(); p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1')); p.add_argument('--metadata',type=Path,default=Path('outputs/metadata/r256/filtered_val.csv')); p.add_argument('--proposals',type=Path,default=Path('outputs/analysis/r257b_scale_repair_fullval_proposals.csv')); p.add_argument('--bank-json',type=Path,default=Path('outputs/analysis/r261_conditional_seam_bank.json')); p.add_argument('--selection-csv',type=Path,default=Path('outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv')); p.add_argument('--output-root',type=Path,default=Path('outputs/priors/r262_metacarpal_valley_six')); p.add_argument('--slots',type=int,default=6); p.add_argument('--palm-v-min',type=float,default=.38); p.add_argument('--palm-v-max',type=float,default=.92); p.add_argument('--min-valley-contrast',type=float,default=.015); p.add_argument('--min-confidence',type=float,default=.06); p.add_argument('--num-cases',type=int,default=12); p.add_argument('--all-val',action='store_true'); p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def selected_stems(path:Path,n:int):
 with path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 out=[]
 for r in rows:
  stem=Path(r.get('image') or r.get('filename') or r.get('stem')).stem
  if stem not in out: out.append(stem)
 return out[:n]

def bilinear(image:np.ndarray,points:np.ndarray):
 x=points[:,0].astype(np.float32); y=points[:,1].astype(np.float32); return cv2.remap(image.astype(np.float32),x[None],y[None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)[0]

def refine_center(image:np.ndarray,p:dict):
 x,y=float(p['x']),float(p['y']); radius=float(np.clip(.45*p['scale'],5,45)); x0=max(0,int(x-radius)); x1=min(image.shape[1],int(x+radius+1)); y0=max(0,int(y-radius)); y1=min(image.shape[0],int(y+radius+1))
 crop=image[y0:y1,x0:x1]
 if crop.size<9:return {**p,'x_refined':x,'y_refined':y}
 yy,xx=np.mgrid[y0:y1,x0:x1]; spatial=np.exp(-((xx-x)**2+(yy-y)**2)/(2*(.55*radius)**2)); lo,hi=np.percentile(crop,[45,95]); bright=np.clip((crop-lo)/(hi-lo+1e-6),0,1)**2; weight=spatial*(.05+bright); total=weight.sum(); xr=float((weight*xx).sum()/total); yr=float((weight*yy).sum()/total); shift=np.asarray([xr-x,yr-y]); norm=float(np.linalg.norm(shift)); cap=.35*radius
 if norm>cap: shift*=cap/norm; xr,yr=x+shift[0],y+shift[1]
 return {**p,'x':xr,'y':yr,'x_original':x,'y_original':y,'refine_shift':float(np.linalg.norm([xr-x,yr-y]))}

def palm_subset(items,palm_v_min,palm_v_max):
 points=np.asarray([[x['x'],x['y']] for x in items]); center,axes,span=canonical(points); z=(points-center)@axes; v=z[:,1]; lo,hi=np.percentile(v,[3,97]); vn=(v-lo)/(hi-lo+1e-6); keep=np.flatnonzero((vn>=palm_v_min)&(vn<=palm_v_max)); return keep.tolist(),(center,axes,span),vn

def delaunay_edges(items,indices,frame):
 if len(indices)<2:return []
 center,axes,span=frame; pts=(np.asarray([[items[i]['x'],items[i]['y']] for i in indices])-center)@axes/span; edges=set()
 if len(indices)>=3:
  try:
   tri=Delaunay(pts)
   for simplex in tri.simplices:
    for a,b in ((0,1),(1,2),(2,0)): edges.add(tuple(sorted((indices[int(simplex[a])],indices[int(simplex[b])]))))
  except Exception: pass
 if not edges:
  d=np.linalg.norm(pts[:,None]-pts[None,:],axis=-1); np.fill_diagonal(d,np.inf)
  for a,i in enumerate(indices): edges.add(tuple(sorted((i,indices[int(np.argmin(d[a]))]))))
 return sorted(edges)

def segment_distance(point,a,b):
 ab=b-a; t=float(np.clip(np.dot(point-a,ab)/(np.dot(ab,ab)+1e-8),0,1)); return float(np.linalg.norm(point-(a+t*ab))),t

def is_adjacent(items,i,j,palm_indices):
 a=np.asarray([items[i]['x'],items[i]['y']]); b=np.asarray([items[j]['x'],items[j]['y']]); local=min(items[i]['scale'],items[j]['scale'])
 for k in palm_indices:
  if k in (i,j):continue
  d,t=segment_distance(np.asarray([items[k]['x'],items[k]['y']]),a,b)
  if .12<t<.88 and d<.45*local:return False
 return True

def localize_valley(image,a,b):
 pa=np.asarray([a['x'],a['y']],float); pb=np.asarray([b['x'],b['y']],float); t=np.linspace(0,1,161); points=pa[None]*(1-t[:,None])+pb[None]*t[:,None]; profile=gaussian_filter1d(bilinear(image,points),2.0); global_scale=float(np.percentile(image,99)-np.percentile(image,1)+1e-6); central=np.arange(32,129); valley_i=int(central[np.argmin(profile[central])]); left=np.arange(5,max(8,valley_i-5)); right=np.arange(min(153,valley_i+5),156)
 if not len(left) or not len(right):return None
 left_i=int(left[np.argmax(profile[left])]); right_i=int(right[np.argmax(profile[right])]); valley=float(profile[valley_i]); peak_l=float(profile[left_i]); peak_r=float(profile[right_i]); contrast=float((min(peak_l,peak_r)-valley)/global_scale); evidence=float(1/(1+math.exp(-np.clip((contrast-.02)/.025,-20,20))))
 return {'seam_point':points[valley_i],'left_support':points[left_i],'right_support':points[right_i],'valley_t':float(t[valley_i]),'left_t':float(t[left_i]),'right_t':float(t[right_i]),'valley_intensity':valley,'left_peak':peak_l,'right_peak':peak_r,'contrast':contrast,'evidence':evidence,'profile':profile.tolist()}

def maps(shape,a,b,loc,confidence):
 h,w=shape; pa=np.asarray([a['x'],a['y']]); pb=np.asarray([b['x'],b['y']]); delta=pb-pa; dist=max(float(np.linalg.norm(delta)),1e-6); n=delta/dist; tangent=np.asarray([-n[1],n[0]]); seam_p=np.asarray(loc['seam_point']); left=np.asarray(loc['left_support']); right=np.asarray(loc['right_support']); local=max(3.,min(a['scale'],b['scale'])); sigma_n=max(1.,.055*local); sigma_t=max(2.,.42*local); sigma_s=max(2.,.17*local); margin=4*max(sigma_t,sigma_s); allp=np.stack([seam_p,left,right]); x0=max(0,int(allp[:,0].min()-margin)); x1=min(w,int(allp[:,0].max()+margin+1)); y0=max(0,int(allp[:,1].min()-margin)); y1=min(h,int(allp[:,1].max()+margin+1)); seam=np.zeros((h,w),np.float32); support=np.zeros((h,w),np.float32)
 if x1<=x0 or y1<=y0:return seam,support
 yy,xx=np.mgrid[y0:y1,x0:x1]; rx=xx-seam_p[0]; ry=yy-seam_p[1]; dn=rx*n[0]+ry*n[1]; dt=rx*tangent[0]+ry*tangent[1]; sm=np.exp(-.5*((dn/sigma_n)**2+(dt/sigma_t)**2)); sp=np.exp(-((xx-left[0])**2+(yy-left[1])**2)/(2*sigma_s**2))+np.exp(-((xx-right[0])**2+(yy-right[1])**2)/(2*sigma_s**2)); sp=np.clip(sp,0,1)*(sm<.50); seam[y0:y1,x0:x1]=confidence*sm; support[y0:y1,x0:x1]=confidence*sp; return seam,support

def build_case(image,raw,age,male,bank,args):
 items=[refine_center(image,x) for x in raw]; palm,frame,vn=palm_subset(items,args.palm_v_min,args.palm_v_max); candidates=[]
 for i,j in delaunay_edges(items,palm,frame):
  if not is_adjacent(items,i,j,palm):continue
  f=features(items[i],items[j],frame)
  if f[0]>4.5:continue
  loc=localize_valley(image,items[i],items[j])
  if loc is None or loc['contrast']<args.min_valley_contrast:continue
  prior,support,relation=bank.score(f,age,male); proposal=math.sqrt(max(items[i].get('score',0),0)*max(items[j].get('score',0),0)); confidence=float(np.clip(prior*proposal*loc['evidence'],0,1))
  if confidence<args.min_confidence:continue
  candidates.append({'i':i,'j':j,'confidence':confidence,'prior':prior,'support_prior':support,'relation':relation,'proposal_confidence':proposal,**{k:v for k,v in loc.items() if k!='profile'}})
 selected=sorted(candidates,key=lambda x:x['confidence'],reverse=True)[:args.slots]; slots=[]
 for x in selected: slots.append((*maps(image.shape,items[x['i']],items[x['j']],x,x['confidence']),x))
 return items,palm,vn,slots,{'all_proposals':len(items),'palm_candidates':len(palm),'adjacent_valley_candidates':len(candidates),'selected':len(selected),'abstained_slots':args.slots-len(selected)}

def overlay(image,items,palm,slots,audit):
 base=Image.fromarray(np.clip(image*255,0,255).astype(np.uint8)).convert('RGB'); draw=ImageDraw.Draw(base); font=ImageFont.load_default()
 for i in palm:
  x,y=items[i]['x'],items[i]['y']; r=max(4,int(.10*items[i]['scale'])); draw.ellipse((x-r,y-r,x+r,y+r),outline=(0,120,255),width=3)
 for k,(_,_,slot) in enumerate(slots,1):
  a=np.asarray(slot['left_support']); b=np.asarray(slot['right_support']); s=np.asarray(slot['seam_point']); i,j=slot['i'],slot['j']; delta=np.asarray([items[j]['x']-items[i]['x'],items[j]['y']-items[i]['y']]); n=delta/(np.linalg.norm(delta)+1e-8); tangent=np.asarray([-n[1],n[0]]); length=.45*min(items[i]['scale'],items[j]['scale']); q0=s-length*tangent; q1=s+length*tangent
  for p in (a,b): draw.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=(0,255,80),outline=(0,90,0),width=2)
  draw.line((q0[0],q0[1],q1[0],q1[1]),fill=(255,40,40),width=4); draw.text((s[0]+5,s[1]-12),str(k),fill=(255,255,0),font=font)
 draw.rectangle((8,8,430,48),fill=(0,0,0)); draw.text((14,14),f"blue=center green=bone support red=valley | palm={audit['palm_candidates']} selected={audit['selected']}",fill='white',font=font); return base

def main():
 a=parse_args(); joined=' '.join(map(str,[a.variant_root,a.metadata,a.proposals,a.bank_json,a.output_root])).lower()
 if a.variant_root.name!='TSRS_RSNA-Epiphysis_contrast_v1' or 'clean-test' in joined or 'articular' in joined or a.slots!=6:raise RuntimeError('R262 original-val Epiphysis six-slot visualization only')
 metadata=read_metadata(a.metadata); grouped=load_proposals(a.proposals,a.variant_root,'val'); payload=json.loads(a.bank_json.read_text(encoding='utf-8'))
 if 'train_labels' not in payload.get('source_label_dir','') or payload.get('num_cases')!=875:raise RuntimeError('R262 requires full train-only R261 bank')
 bank=Bank(payload,24.); stems=sorted(metadata) if a.all_val else selected_stems(a.selection_csv,a.num_cases)
 if not set(stems)<=set(metadata)&set(grouped):raise RuntimeError('case coverage mismatch')
 if a.output_root.exists() and a.overwrite:shutil.rmtree(a.output_root)
 a.output_root.mkdir(parents=True); manifest=[]
 for stem in stems:
  image=np.asarray(Image.open(find_image(a.variant_root/'val',stem)).convert('L'),np.float32)/255.; age,male=metadata[stem]; items,palm,vn,slots,audit=build_case(image,grouped[stem],age,male,bank,a); records=[]
  for k in range(a.slots):
   if k<len(slots): seam,support,info=slots[k]; records.append(info)
   else: seam=support=np.zeros(image.shape,np.float32)
   Image.fromarray(np.clip(seam*255,0,255).astype(np.uint8)).save(a.output_root/f'{stem}_seam{k}.png',optimize=True); Image.fromarray(np.clip(support*255,0,255).astype(np.uint8)).save(a.output_root/f'{stem}_support{k}.png',optimize=True)
  panel=overlay(image,items,palm,slots,audit); panel.save(a.output_root/f'{stem}_anatomy_overlay.png',optimize=True); case={'stem':stem,'boneage':age,'male':bool(male),'audit':audit,'slots':records,'overlay':str(a.output_root/f'{stem}_anatomy_overlay.png')}; (a.output_root/f'{stem}_pairs.json').write_text(json.dumps(case,indent=2,default=lambda x:np.asarray(x).tolist()),encoding='utf-8'); manifest.append(case)
 (a.output_root/'manifest.json').write_text(json.dumps({'run_id':'R262','scope':'original-val visualization only','clean_test_used':False,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'cases':manifest},indent=2,default=lambda x:np.asarray(x).tolist()),encoding='utf-8'); print(json.dumps({'cases':len(stems),'selected_slots':sum(x['audit']['selected'] for x in manifest),'abstained_slots':sum(x['audit']['abstained_slots'] for x in manifest),'output':str(a.output_root)},indent=2))
if __name__=='__main__':main()
