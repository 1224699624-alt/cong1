#!/usr/bin/env python3
"""R302: one-time frozen test audit for the R300 Ceb-inspired gate."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,precision_recall_fscore_support
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,WristDataset,seed_everything,render_comparisons
from train_r300_ram_ceb_boundary_gate import collect,examples,apply_gate,metrics,pack
from diagnose_r301_ram_ceb_signature import val_examples

def main():
 out=Path('outputs/ram_w600/r302_ceb_gate_test_audit');out.mkdir(parents=True,exist_ok=True);seed_everything(300);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 base=NnUNetMultiLabelPrior().to(device);base.load_state_dict(torch.load('outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth',map_location=device,weights_only=False)['model']);base.eval()
 root=Path(r'G:\gutou\RAM-W600');tr=DataLoader(WristDataset(root,'train',384,False),batch_size=2,shuffle=False,num_workers=0);te=DataLoader(WristDataset(root,'test',384,False),batch_size=2,shuffle=False,num_workers=0)
 train=collect(base,tr,device);test=collect(base,te,device);X,y=examples(train);Xt,yt=val_examples(test);clf=RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=300,n_jobs=-1).fit(X,y);prob=clf.predict_proba(Xt)[:,1];pred=(prob>=0.5).astype(int);pr,re,f,_=precision_recall_fscore_support(yt,pred,average='binary',zero_division=0); gated=apply_gate(test,clf); bm=metrics(test);gm=metrics(gated); result={'experiment':'R302','test_audit_only':True,'test_used_for_selection':False,'classifier_train_candidates':int(len(y)),'test_candidates':int(len(yt)),'test_positive_fraction':float(yt.mean()),'test_signature_auc':float(roc_auc_score(yt,prob)),'test_signature_precision':float(pr),'test_signature_recall':float(re),'test_signature_f1':float(f),'baseline_test':bm,'hard_gate_test':gm,'delta_test':{k:gm[k]-bm[k] for k in bm},'warning':'hard gate is evaluated for audit only and is not promoted'}
 (out/'result.json').write_text(json.dumps(result,indent=2));cases=sorted(test,key=lambda c:int((test[c]['target'].sum(0)>=2).sum()),reverse=True)[:12];render_comparisons(root,out/'visualizations',cases,pack(test),pack(gated),384);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
