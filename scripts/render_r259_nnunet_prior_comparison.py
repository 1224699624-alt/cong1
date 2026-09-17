#!/usr/bin/env python3
"""Four-column full-image/close-gap R259 comparison on frozen hard cases."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from render_r255_close_gap_comparison import find_valid_image,read_gray,read_mask,read_instance,resize,crop_box,select_rows,TILE
def dice(a,b): return float((2*(a&b).sum()+1)/(a.sum()+b.sum()+1))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--image-dir',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis/val')); p.add_argument('--gt-dir',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis/val_labels')); p.add_argument('--baseline-dir',type=Path,required=True); p.add_argument('--improved-dir',type=Path,required=True); p.add_argument('--selection-csv',type=Path,default=Path('outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv')); p.add_argument('--output-dir',type=Path,default=Path('outputs/visualizations/r259_nnunet_frozen_prior_original_val')); p.add_argument('--num-cases',type=int,default=12); p.add_argument('--xray-label',default='Original X-ray'); p.add_argument('--baseline-label',default='Matched nnU-Net baseline'); p.add_argument('--improved-label',default='Frozen prior + loss nnU-Net'); a=p.parse_args()
 joined=' '.join(map(str,[a.image_dir,a.gt_dir,a.baseline_dir,a.improved_dir,a.selection_csv,a.output_dir])).lower()
 if 'clean-test' in joined or 'articular' in joined: raise RuntimeError('original-val Epiphysis only')
 rows=select_rows(a.selection_csv,a.num_cases); a.output_dir.mkdir(parents=True,exist_ok=True); manifest=[]; labels=['Original X-ray','Ground truth','Original nnU-Net','Prior + loss nnU-Net']
 if len(rows)!=a.num_cases: raise RuntimeError(f'Expected {a.num_cases} frozen cases, got {len(rows)}')
 for row in rows:
  name=row['image']; stem=Path(name).stem; image=read_gray(find_valid_image(a.image_dir,stem)); instance=read_instance(a.gt_dir/name); gt=instance>0; baseline=read_mask(a.baseline_dir/name); improved=read_mask(a.improved_dir/name)
  if not (gt.shape==baseline.shape==improved.shape): raise RuntimeError(stem)
  box=crop_box(instance,int(row['instance_i']),int(row['instance_j'])); x0,y0,x1,y1=box
  full=[resize(image),resize(gt,mask=True),resize(baseline,mask=True),resize(improved,mask=True)]; zoom=[resize(image[y0:y1,x0:x1]),resize(gt[y0:y1,x0:x1],mask=True),resize(baseline[y0:y1,x0:x1],mask=True),resize(improved[y0:y1,x0:x1],mask=True)]
  labels=[a.xray_label,'Ground truth',a.baseline_label,a.improved_label]
  header,row_h,width=76,520,4*TILE[0]; canvas=Image.new('RGB',(width,header+2*row_h),'white'); draw=ImageDraw.Draw(canvas); draw.text((10,8),f"{stem} | fixed hard pair {row['instance_i']}-{row['instance_j']} | gap={float(row['gap_px']):.2f}px | Dice {dice(baseline,gt):.4f}->{dice(improved,gt):.4f}",fill='black',font=ImageFont.load_default())
  for i,label in enumerate(labels): draw.text((i*TILE[0]+8,42),label,fill='black',font=ImageFont.load_default()); canvas.paste(full[i],(i*TILE[0],header)); canvas.paste(zoom[i],(i*TILE[0],header+row_h))
  draw.text((8,header+4),'Full image',fill='black'); draw.text((8,header+row_h+4),'Close-gap zoom',fill='black'); out=a.output_dir/f'{stem}_comparison.png'; canvas.save(out,optimize=True); manifest.append({'image':name,'panel':str(out),'selection':'frozen R255 baseline-only 1-4px hard ranking','pair':[int(row['instance_i']),int(row['instance_j'])],'gap_px':float(row['gap_px']),'dice_baseline':dice(baseline,gt),'dice_improved':dice(improved,gt),'crop_xyxy':list(box)})
 (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps({'rendered':len(manifest),'output':str(a.output_dir)},indent=2))
if __name__=='__main__': main()
