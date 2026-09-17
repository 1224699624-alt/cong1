#!/usr/bin/env python3
import argparse,shutil
from pathlib import Path
from PIL import Image
from train_r256_scale_invariant_pair_prior import find_image
def main():
 p=argparse.ArgumentParser(); p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r259_frozen_relation/val')); p.add_argument('--native-dir',type=Path,required=True); p.add_argument('--active-dir',type=Path,required=True); a=p.parse_args()
 joined=' '.join(map(str,[a.variant_root,a.prior_root,a.native_dir,a.active_dir])).lower()
 if a.variant_root.name!='TSRS_RSNA-Epiphysis_contrast_v1' or 'clean-test' in joined or 'articular' in joined: raise RuntimeError('R259 original-val Epiphysis only')
 for d in (a.native_dir,a.active_dir):
  if d.exists(): shutil.rmtree(d)
  d.mkdir(parents=True)
 labels=sorted((a.variant_root/'val_labels').glob('*.png'))
 if len(labels)!=96: raise RuntimeError(len(labels))
 for lp in labels:
  s=lp.stem; image=Image.open(find_image(a.variant_root/'val',s)).convert('L'); zero=Image.new('L',image.size,0); pp=a.prior_root/f'{s}.png'
  if not pp.exists() or pp.stat().st_size==0: raise RuntimeError(f'Missing prior {s}')
  prior=Image.open(pp).convert('L')
  image.save(a.native_dir/f'val_{s}_0000.png'); zero.save(a.native_dir/f'val_{s}_0001.png'); image.save(a.active_dir/f'val_{s}_0000.png'); prior.save(a.active_dir/f'val_{s}_0001.png')
 print({'cases':96})
if __name__=='__main__': main()
