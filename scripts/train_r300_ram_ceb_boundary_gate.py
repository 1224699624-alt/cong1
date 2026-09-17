#!/usr/bin/env python3
"""R300: Ceb-inspired boundary-level gate for RAM-W600 overlap predictions.

This is a postprocessor, not a new segmentation backbone: candidate connected
overlap regions from the frozen R285 14-channel baseline receive a structured
local signature and a train-only RandomForest boundary/overlap decision.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from scipy.ndimage import label as cc_label, binary_dilation, binary_erosion
from sklearn.ensemble import RandomForestClassifier
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,WristDataset,render_comparisons,seed_everything,surface_dice

def collect(base,loader,device):
    out={}; base.eval()
    with torch.no_grad():
      for batch in loader:
        logits,_=base(batch['image'].to(device),False); prob=torch.sigmoid(logits).cpu().numpy(); target=(batch['mask'].numpy()>0.5)
        pred=prob>=0.5
        for b,c in enumerate(batch['case']): out[str(c)]={'prob':prob[b],'pred':pred[b],'target':target[b],'image':batch['image'][b,0].numpy()}
    return out

def signature(item,i,j,comp):
    p=item['prob']; im=item['image']; m=comp
    ring=binary_dilation(m,iterations=3)&~m; boundary=m^binary_erosion(m)
    gy,gx=np.gradient(im.astype(np.float32)); edge=np.sqrt(gx*gx+gy*gy)
    ys,xs=np.where(m); h=max(1,ys.max()-ys.min()+1); w=max(1,xs.max()-xs.min()+1)
    vals=lambda a,mask: float(a[mask].mean()) if mask.any() else 0.0
    # Boundary signature: candidate boundary, neighbouring foreground/background
    # ring, image edge, and geometry. This follows Ceb's boundary-level object.
    return np.array([m.sum()/im.size, len(ys)/(h*w), h/w,
        vals(p[i],m),vals(p[j],m),vals(p[i]*p[j],m),
        vals(p[i],boundary),vals(p[j],boundary),vals(p[i],ring),vals(p[j],ring),
        vals(edge,boundary),vals(edge,ring)],dtype=np.float32)

def examples(data):
    X=[]; y=[]
    for item in data.values():
      pred=item['pred']; tar=item['target']; p=item['prob']
      for i in range(14):
       for j in range(i+1,14):
        ov=pred[i]&pred[j]; lab,n=cc_label(ov)
        for k in range(1,n+1):
          comp=lab==k
          if comp.sum()<3: continue
          gt=item['target'][i]&item['target'][j]
          inter=(comp&gt).sum(); union=(comp|gt).sum(); iou=inter/max(union,1)
          X.append(signature(item,i,j,comp)); y.append(int(iou>=0.5))
    return np.asarray(X,np.float32),np.asarray(y,np.int64)

def apply_gate(data,clf):
    out={}
    for c,item in data.items():
      pred=item['pred'].copy(); prob=item['prob']; image=item['image']
      for i in range(14):
       for j in range(i+1,14):
        ov=pred[i]&pred[j]; lab,n=cc_label(ov)
        for k in range(1,n+1):
          comp=lab==k
          if comp.sum()<3: continue
          keep=int(clf.predict(signature(item,i,j,comp)[None])[0])
          if keep==0:
            # Suppress only the less confident instance in this candidate.
            mi=float(prob[i][comp].mean()); mj=float(prob[j][comp].mean())
            pred[i][comp]=False if mi<=mj else pred[i][comp]
            pred[j][comp]=False if mj<mi else pred[j][comp]
      out[c]={'pred':pred,'target':item['target'],'prob':prob,'image':image}
    return out

def metrics(data):
    cds=[];ods=[];nsd=[]
    for item in data.values():
      p=item['pred'];t=item['target']; cds.append(np.mean([(2*(p[k]&t[k]).sum()+1)/(p[k].sum()+t[k].sum()+1) for k in range(14)])); po=p.sum(0)>=2;to=t.sum(0)>=2;ods.append((2*(po&to).sum()+1)/(po.sum()+to.sum()+1));nsd.append(surface_dice(po,to,2))
    return {'macro_dice':float(np.mean(cds)),'overlap_dice':float(np.mean(ods)),'overlap_nsd_2px':float(np.mean(nsd))}

def pack(data): return {'predictions':{c:v['pred'] for c,v in data.items()},'targets':{c:v['target'] for c,v in data.items()}}

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,default=Path(r'G:\gutou\RAM-W600'));ap.add_argument('--baseline-checkpoint',type=Path,default=Path('outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth'));ap.add_argument('--output',type=Path,default=Path('outputs/ram_w600/r300_ceb_boundary_gate'));ap.add_argument('--seed',type=int,default=300);a=ap.parse_args();seed_everything(a.seed);a.output.mkdir(parents=True,exist_ok=True);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  base=NnUNetMultiLabelPrior().to(device);base.load_state_dict(torch.load(a.baseline_checkpoint,map_location=device,weights_only=False)['model']);base.eval();tr=DataLoader(WristDataset(a.dataset_root,'train',384,False),batch_size=2,shuffle=False,num_workers=0);va=DataLoader(WristDataset(a.dataset_root,'val',384,False),batch_size=2,shuffle=False,num_workers=0);te=DataLoader(WristDataset(a.dataset_root,'test',384,False),batch_size=2,shuffle=False,num_workers=0)
  train=collect(base,tr,device); val=collect(base,va,device); test=collect(base,te,device);X,y=examples(train); clf=RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_leaf=3,class_weight='balanced',random_state=a.seed,n_jobs=-1);clf.fit(X,y); val_gate=apply_gate(val,clf); bm=metrics(val);vm=metrics(val_gate); result={'experiment':'R300','device':str(device),'train_candidates':int(len(y)),'train_positive_fraction':float(y.mean()) if len(y) else 0.0,'baseline_val':bm,'gated_val':vm,'delta_val':{k:vm[k]-bm[k] for k in bm},'test_evaluated':False}
  # Strict gate: test is evaluated only if validation macro Dice is non-inferior
  # and overlap Dice improves; otherwise preserve the no-go without test use.
  if vm['macro_dice']>=bm['macro_dice']-0.002 and vm['overlap_dice']>bm['overlap_dice']:
    test_gate=apply_gate(test,clf);tm=metrics(test);im=metrics(test_gate);result.update({'test_evaluated':True,'baseline_test':tm,'improved_test':im,'delta_test':{k:im[k]-tm[k] for k in tm}});cases=sorted(test,key=lambda c:int((test[c]['target'].sum(0)>=2).sum()),reverse=True)[:12];render_comparisons(a.dataset_root,a.output/'visualizations',cases,pack(test),pack(test_gate),384)
  else: result['decision']='NO_GO_VALIDATION_GATE'
  (a.output/'result.json').write_text(json.dumps(result,indent=2));(a.output/'manifest.json').write_text(json.dumps({'experiment':'R300','method':'Ceb-inspired boundary-level candidate overlap classifier','classifier':'RandomForest on pairwise boundary signature','test_selection_rule':'test only after validation non-inferiority plus overlap Dice improvement','threshold':0.5},indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
