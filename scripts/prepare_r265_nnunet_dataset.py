#!/usr/bin/env python3
"""Materialize isolated 13-channel nnU-Net Dataset206 for R265."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from build_r261_conditional_seam_priors import find_image

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset-root',type=Path,default=Path('data/raw/TSRS_RSNA-Epiphysis')); p.add_argument('--prior-root',type=Path,default=Path('outputs/priors/r265_hierarchical_carpal')); p.add_argument('--nnunet-root',type=Path,default=Path('outputs/nnunet/r265_hierarchical_carpal/data')); p.add_argument('--overwrite',action='store_true'); a=p.parse_args()
 joined=' '.join(map(str,[a.dataset_root,a.prior_root,a.nnunet_root])).lower()
 if a.dataset_root.name!='TSRS_RSNA-Epiphysis' or 'clean-test' in joined or 'articular' in joined:raise RuntimeError('R265 original Epiphysis only')
 ds=a.nnunet_root/'nnUNet_raw'/'Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D'
 if ds.exists() and a.overwrite:shutil.rmtree(ds)
 images,labels=ds/'imagesTr',ds/'labelsTr'; train=[]; val=[]
 for split,keys,expected in [('train',train,875),('val',val,96)]:
  lps=sorted((a.dataset_root/f'{split}_labels').glob('*.png'))
  if len(lps)!=expected:raise RuntimeError((split,len(lps),expected))
  for lp in lps:
   stem=lp.stem; key=f'{split}_{stem}'; image=Image.open(find_image(a.dataset_root/split,stem)).convert('L'); shape=np.asarray(image).shape
   images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True); image.save(images/f'{key}_0000.png')
   channel=1
   for slot in range(6):
    for kind in ('seam','support'):
     src=a.prior_root/split/f'{stem}_{kind}{slot}.png'; arr=np.asarray(Image.open(src).convert('L'))
     if arr.shape!=shape or not np.isfinite(arr).all():raise RuntimeError(f'invalid {src}')
     Image.fromarray(arr.astype(np.uint8)).save(images/f'{key}_{channel:04d}.png'); channel+=1
   lab=np.asarray(Image.open(lp)); lab=lab[...,0] if lab.ndim==3 else lab; Image.fromarray((lab>0).astype(np.uint8)).save(labels/f'{key}.png'); keys.append(key)
 channels={'0':'xray'}
 schema=['carpal_carpal']*4+['carpal_metacarpal','carpal_radius_ulna']
 for slot,name in enumerate(schema):channels[str(1+2*slot)]=f'{name}_seam_{slot}'; channels[str(2+2*slot)]=f'{name}_support_{slot}'
 (ds/'dataset.json').write_text(json.dumps({'channel_names':channels,'labels':{'background':0,'epiphysis':1},'numTraining':971,'file_ending':'.png','overwrite_image_reader_writer':'NaturalImage2DIO'},indent=2))
 split=a.nnunet_root/'splits_final_r265.json'; split.parent.mkdir(parents=True,exist_ok=True); split.write_text(json.dumps([{'train':train,'val':val}],indent=2)); print(json.dumps({'dataset':str(ds),'train':len(train),'val':len(val),'channels':13}))
if __name__=='__main__':main()
