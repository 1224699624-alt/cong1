#!/usr/bin/env python3
"""Visual gate for R265 hierarchical carpal relation types."""
import argparse,csv,json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from build_r261_conditional_seam_priors import find_image

COLORS={'carpal_carpal':(255,40,40),'carpal_metacarpal':(255,165,0),'carpal_radius_ulna':(0,220,255)}
SHORT={'carpal_carpal':'CC','carpal_metacarpal':'CM','carpal_radius_ulna':'CR'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r265_hierarchical_carpal/val')); p.add_argument('--selection-csv',type=Path,default=Path('outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv')); p.add_argument('--output-root',type=Path,default=Path('outputs/visualizations/r265_hierarchical_carpal_prior_gate')); p.add_argument('--num-cases',type=int,default=12); a=p.parse_args()
 joined=' '.join(map(str,vars(a).values())).lower()
 if 'clean-test' in joined or 'articular' in joined:raise RuntimeError('original-val Epiphysis only')
 with a.selection_csv.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 stems=[]
 for r in rows:
  stem=Path(r.get('image') or r.get('filename') or r.get('stem')).stem
  if stem not in stems:stems.append(stem)
 stems=stems[:a.num_cases]; a.output_root.mkdir(parents=True,exist_ok=True); font=ImageFont.load_default()
 for stem in stems:
  im=Image.open(find_image(a.variant_root/'val',stem)).convert('RGB'); d=ImageDraw.Draw(im)
  payload=json.loads((a.prior_root/f'{stem}_pairs.json').read_text())
  for x in payload['slots']:
   p0=x['left_support']; p1=x['right_support']; s=x['seam_point']; color=COLORS[x['type']]
   d.line((p0[0],p0[1],p1[0],p1[1]),fill=color,width=4)
   for q in (p0,p1):d.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),fill=(0,255,80),outline=(0,80,0),width=2)
   d.text((s[0]+5,s[1]-12),f"{x['slot']+1}:{SHORT[x['type']]}",fill=(255,255,0),font=font)
  d.rectangle((8,8,530,50),fill=(0,0,0)); d.text((14,14),'red=CC orange=CM cyan=CR green=bone support',fill='white',font=font)
  im.save(a.output_root/f'{stem}_r265_relation_types.png',optimize=True)
 print(json.dumps({'cases':len(stems),'output':str(a.output_root)}))
if __name__=='__main__':main()
