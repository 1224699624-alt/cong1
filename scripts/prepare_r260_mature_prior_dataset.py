#!/usr/bin/env python3
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from train_r256_scale_invariant_pair_prior import find_image
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r259_frozen_relation')); p.add_argument('--nnunet-root',type=Path,default=Path('outputs/nnunet/r260_mature_prior/data')); p.add_argument('--supplemental-label-root',type=Path); p.add_argument('--allow-zero-prior',action='store_true'); p.add_argument('--allow-png-only',action='store_true'); p.add_argument('--overwrite',action='store_true'); a=p.parse_args(); joined=' '.join(map(str,[a.dataset_root,a.prior_root,a.nnunet_root,a.supplemental_label_root])).lower()
 if a.dataset_root.name!='TSRS_RSNA-Epiphysis' or 'clean-test' in joined or 'articular' in joined: raise RuntimeError('R260 original Epiphysis train/val only')
 ds=a.nnunet_root/'nnUNet_raw'/'Dataset204_TSRS_RSNAEpiphysisMaturePrior2D'
 if ds.exists() and a.overwrite: shutil.rmtree(ds)
 images,labels=ds/'imagesTr',ds/'labelsTr'; train=[]; val=[]
 supplemental=[]
 for split,keys in [('train',train),('val',val)]:
  label_map={p.stem:p for p in (a.dataset_root/f'{split}_labels').glob('*.png')}
  if a.supplemental_label_root:
   for pth in (a.supplemental_label_root/f'{split}_labels').glob('*.png'):
    if pth.stem not in label_map: label_map[pth.stem]=pth; supplemental.append({'split':split,'stem':pth.stem,'path':str(pth)})
  lps=[label_map[k] for k in sorted(label_map)]
  for lp in lps:
   stem=lp.stem; key=f'{split}_{stem}'; image=Image.open(find_image(a.dataset_root/split,stem)).convert('L'); npy=a.prior_root/split/f'{stem}.npy'; png=a.prior_root/split/f'{stem}.png'
   if not png.exists() or (not npy.exists() and not a.allow_png_only): raise RuntimeError(f'Missing prior {split}/{stem}')
   prior=Image.open(png).convert('L'); prior_arr=np.asarray(prior)
   if npy.exists():
    arr=np.load(npy); valid=arr.shape==np.asarray(image).shape and np.isfinite(arr).all() and (a.allow_zero_prior or float(arr.max())>0)
   else: valid=prior_arr.shape==np.asarray(image).shape and (a.allow_zero_prior or int(prior_arr.max())>0)
   if not valid or prior.size!=image.size: raise RuntimeError(f'Invalid prior {split}/{stem}')
   lab=np.asarray(Image.open(lp)); lab=lab[...,0] if lab.ndim==3 else lab; images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True); image.save(images/f'{key}_0000.png'); prior.save(images/f'{key}_0001.png'); Image.fromarray((lab>0).astype(np.uint8)).save(labels/f'{key}.png'); keys.append(key)
 if (len(train),len(val))!=(875,96): raise RuntimeError((len(train),len(val)))
 (ds/'dataset.json').write_text(json.dumps({'channel_names':{'0':'xray','1':'prior'},'labels':{'background':0,'epiphysis':1},'numTraining':971,'file_ending':'.png','overwrite_image_reader_writer':'NaturalImage2DIO'},indent=2)); split=a.nnunet_root/'splits_final_r260.json'; split.parent.mkdir(parents=True,exist_ok=True); split.write_text(json.dumps([{'train':train,'val':val}],indent=2)); (a.nnunet_root/'supplemental_labels.json').write_text(json.dumps(supplemental,indent=2)); print(json.dumps({'dataset':str(ds),'train':875,'val':96,'supplemental_labels':supplemental}))
if __name__=='__main__': main()
