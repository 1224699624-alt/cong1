#!/usr/bin/env python3
"""R301: validation-only Ceb boundary-signature separability diagnostic.

No mask editing and no test evaluation. Measures whether Ceb-like pairwise
signatures can distinguish true vs false predicted overlap candidates before
choosing an intervention rule.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support,confusion_matrix
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,WristDataset,seed_everything
from train_r300_ram_ceb_boundary_gate import collect,examples,signature
from scipy.ndimage import label as cc_label

def val_examples(data):
 X=[];y=[]
 for item in data.values():
  pred=item['pred']
  for i in range(14):
   for j in range(i+1,14):
    lab,n=cc_label(pred[i]&pred[j])
    for k in range(1,n+1):
     comp=lab==k
     if comp.sum()<3: continue
     gt=item['target'][i]&item['target'][j]; iou=(comp&gt).sum()/max((comp|gt).sum(),1)
     X.append(signature(item,i,j,comp)); y.append(int(iou>=0.5))
 return np.asarray(X,np.float32),np.asarray(y,np.int64)

def main():
 out=Path('outputs/ram_w600/r301_ceb_signature_diagnostic');out.mkdir(parents=True,exist_ok=True);seed_everything(301);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 base=NnUNetMultiLabelPrior().to(device);base.load_state_dict(torch.load('outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth',map_location=device,weights_only=False)['model']);base.eval()
 tr=DataLoader(WristDataset(Path(r'G:\gutou\RAM-W600'),'train',384,False),batch_size=2,shuffle=False,num_workers=0);va=DataLoader(WristDataset(Path(r'G:\gutou\RAM-W600'),'val',384,False),batch_size=2,shuffle=False,num_workers=0)
 td=collect(base,tr,device);vd=collect(base,va,device);Xtr,ytr=examples(td);Xv,yv=val_examples(vd);clf=RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=301,n_jobs=-1).fit(Xtr,ytr);proba=clf.predict_proba(Xv)[:,1];pred=(proba>=0.5).astype(int);pr,re,f,_=precision_recall_fscore_support(yv,pred,average='binary',zero_division=0)
 result={'experiment':'R301','test_evaluated':False,'train_candidates':int(len(ytr)),'val_candidates':int(len(yv)),'train_positive_fraction':float(ytr.mean()),'val_positive_fraction':float(yv.mean()),'val_auc':float(roc_auc_score(yv,proba)),'val_precision':float(pr),'val_recall':float(re),'val_f1':float(f),'val_confusion_matrix':confusion_matrix(yv,pred).tolist(),'interpretation':'signature separability only; no prediction editing performed'}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
