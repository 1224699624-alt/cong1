#!/usr/bin/env python3
import argparse,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from build_r261_conditional_seam_priors import find_image,read_metadata
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis')); p.add_argument('--metadata',type=Path,default=Path('outputs/metadata/r256/filtered_val.csv')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r261_conditional_seams/val')); p.add_argument('--output',type=Path,required=True); p.add_argument('--slots',type=int,default=6); a=p.parse_args(); joined=' '.join(map(str,[a.dataset_root,a.metadata,a.prior_root,a.output])).lower()
 if a.dataset_root.name!='TSRS_RSNA-Epiphysis' or 'clean-test' in joined or 'articular' in joined or a.slots!=6: raise RuntimeError('R261 original-val only')
 if a.output.exists(): shutil.rmtree(a.output)
 a.output.mkdir(parents=True); stems=sorted(read_metadata(a.metadata))
 if len(stems)!=96: raise RuntimeError(len(stems))
 for stem in stems:
  image=Image.open(find_image(a.dataset_root/'val',stem)).convert('L'); shape=np.asarray(image).shape; image.save(a.output/f'val_{stem}_0000.png'); channel=1
  for slot in range(a.slots):
   for kind in ('seam','support'):
    src=a.prior_root/f'{stem}_{kind}{slot}.png'; arr=np.asarray(Image.open(src).convert('L'))
    if arr.shape!=shape: raise RuntimeError((stem,kind,slot,arr.shape,shape))
    Image.fromarray(arr.astype(np.uint8)).save(a.output/f'val_{stem}_{channel:04d}.png'); channel+=1
 print({'cases':96,'channels':13})
if __name__=='__main__': main()
