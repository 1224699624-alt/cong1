#!/usr/bin/env python3
"""Prepare isolated two-channel Dataset203 for the frozen-prior visual probe."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from train_r256_scale_invariant_pair_prior import find_image
def main():
 p=argparse.ArgumentParser(); p.add_argument('--variant-root',type=Path,default=Path('data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r259_frozen_relation')); p.add_argument('--nnunet-root',type=Path,default=Path('outputs/nnunet/r259_frozen_prior')); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
 joined=' '.join(map(str,[a.variant_root,a.prior_root,a.nnunet_root])).lower()
 if a.variant_root.name!='TSRS_RSNA-Epiphysis_contrast_v1' or 'articular' in joined or 'clean-test' in joined: raise RuntimeError('R259 Epiphysis train/original-val only')
 ds=a.nnunet_root/'nnUNet_raw'/'Dataset203_TSRS_RSNAEpiphysisPrior2D'
 if ds.exists() and a.overwrite: shutil.rmtree(ds)
 images,labels=ds/'imagesTr',ds/'labelsTr'; train_keys=[]; val_keys=[]
 for split,keys in [('train',train_keys),('val',val_keys)]:
  for lp in sorted((a.variant_root/f'{split}_labels').glob('*.png')):
   stem=lp.stem; key=f'{split}_{stem}'; image=Image.open(find_image(a.variant_root/split,stem)).convert('L'); npy=a.prior_root/split/f'{stem}.npy'; png=a.prior_root/split/f'{stem}.png'
   if not npy.exists() or not png.exists(): raise RuntimeError(f'Missing prior {split}/{stem}')
   values=np.load(npy); prior=Image.open(png).convert('L')
   if values.shape!=np.asarray(image).shape or not np.isfinite(values).all() or float(values.max())<=0 or prior.size!=image.size: raise RuntimeError(f'Invalid prior {split}/{stem}')
   lab=np.asarray(Image.open(lp)); lab=lab[...,0] if lab.ndim==3 else lab
   images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True); image.save(images/f'{key}_0000.png'); prior.save(images/f'{key}_0001.png'); Image.fromarray((lab>0).astype(np.uint8)).save(labels/f'{key}.png'); keys.append(key)
 if (len(train_keys),len(val_keys))!=(875,96): raise RuntimeError((len(train_keys),len(val_keys)))
 (ds/'dataset.json').write_text(json.dumps({'channel_names':{'0':'xray','1':'prior'},'labels':{'background':0,'epiphysis':1},'numTraining':971,'file_ending':'.png','overwrite_image_reader_writer':'NaturalImage2DIO'},indent=2))
 split=a.nnunet_root/'splits_final_r259.json'; split.parent.mkdir(parents=True,exist_ok=True); split.write_text(json.dumps([{'train':train_keys,'val':val_keys}],indent=2)); print(json.dumps({'dataset':str(ds),'train':875,'val':96}))
if __name__=='__main__': main()
