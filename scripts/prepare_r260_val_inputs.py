#!/usr/bin/env python3
import argparse,shutil
from pathlib import Path
from PIL import Image
from train_r256_scale_invariant_pair_prior import find_image
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r259_frozen_relation/val')); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); joined=' '.join(map(str,[a.dataset_root,a.prior_root,a.output])).lower()
 if a.dataset_root.name!='TSRS_RSNA-Epiphysis' or 'clean-test' in joined or 'articular' in joined: raise RuntimeError('R260 original-val only')
 if a.output.exists(): shutil.rmtree(a.output)
 a.output.mkdir(parents=True); labels=sorted((a.dataset_root/'val_labels').glob('*.png'))
 if len(labels)!=96: raise RuntimeError(len(labels))
 for lp in labels:
  s=lp.stem; image=Image.open(find_image(a.dataset_root/'val',s)).convert('L'); prior=Image.open(a.prior_root/f'{s}.png').convert('L'); image.save(a.output/f'val_{s}_0000.png'); prior.save(a.output/f'val_{s}_0001.png')
 print({'cases':96})
if __name__=='__main__': main()
