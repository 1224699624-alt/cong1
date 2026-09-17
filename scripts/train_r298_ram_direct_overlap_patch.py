#!/usr/bin/env python3
"""R298: direct multi-label mask prediction on coarse-predicted overlap patches."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,WristDataset,render_comparisons,seed_everything,surface_dice
from train_r297_ram_overlap_patch_refiner import boxes_from_probability,crop_batch,metrics

class DirectPatchNet(nn.Module):
    def __init__(self):
        super().__init__(); h=32
        self.net=nn.Sequential(nn.Conv2d(15,h,3,padding=1),nn.GroupNorm(8,h),nn.SiLU(),nn.Conv2d(h,h,3,padding=1),nn.GroupNorm(8,h),nn.SiLU(),nn.Conv2d(h,h,3,padding=1),nn.GroupNorm(8,h),nn.SiLU(),nn.Conv2d(h,14,1))
    def forward(self,x): return self.net(x)

def train_loss(logits,target):
    p=torch.sigmoid(logits.float()); t=target.float(); axes=(0,2,3)
    dice=1-((2*(p*t).sum(axes)+1)/(p.sum(axes)+t.sum(axes)+1)).mean(); bce=F.binary_cross_entropy_with_logits(logits.float(),t)
    ov=(t.sum(1)>=2).unsqueeze(1).expand_as(t)
    if ov.any(): od=1-(2*(p*t*ov).sum()+1)/((p*ov).sum()+(t*ov).sum()+1); ob=F.binary_cross_entropy(p[ov],t[ov])
    else: od=logits.new_zeros(()); ob=logits.new_zeros(())
    return dice+bce+0.75*(od+ob)

def infer(net,base,loader,device,patch_size,use_local):
    preds={}; targets={}; net.eval()
    with torch.no_grad():
      for batch in loader:
        image=batch['image'].to(device); coarse,_=base(image,False); prob=torch.sigmoid(coarse); boxes=boxes_from_probability(prob); patch,_,loc=crop_batch(image,prob,boxes,batch['mask'].to(device),patch_size); local=net(patch) if use_local else None
        for b,c in enumerate(batch['case']):
          out=coarse[b].clone()
          if use_local:
            idx=[j for j,z in enumerate(loc) if z[0]==b][0]; _,y0,y1,x0,x1=loc[idx]; lp=F.interpolate(local[idx:idx+1],(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0]; bp=coarse[b,:,y0:y1,x0:x1]; out[:,y0:y1,x0:x1]=0.5*bp+0.5*lp
          preds[str(c)]=(torch.sigmoid(out)>=0.5).cpu().numpy(); targets[str(c)]=(batch['mask'][b].numpy()>0.5)
    return {'predictions':preds,'targets':targets}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,default=Path(r'G:\gutou\RAM-W600')); ap.add_argument('--baseline-checkpoint',type=Path,default=Path('outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth')); ap.add_argument('--output',type=Path,default=Path('outputs/ram_w600/r298_direct_overlap_patch')); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--patch-size',type=int,default=192); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--seed',type=int,default=298); a=ap.parse_args(); seed_everything(a.seed); a.output.mkdir(parents=True,exist_ok=True)
  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=WristDataset(a.dataset_root,'train',384,True); va=WristDataset(a.dataset_root,'val',384,False); te=WristDataset(a.dataset_root,'test',384,False); tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0); vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=0); tel=DataLoader(te,batch_size=a.batch_size,shuffle=False,num_workers=0)
  base=NnUNetMultiLabelPrior().to(device); payload=torch.load(a.baseline_checkpoint,map_location=device,weights_only=False); base.load_state_dict(payload['model']); base.eval(); [setattr(p,'requires_grad',False) for p in base.parameters()]
  net=DirectPatchNet().to(device); opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); baseline_macro=float(payload['metrics']['macro_dice']); best=(-1,None); best_path=a.output/'direct_best.pth'
  for ep in range(a.epochs):
    net.train(); losses=[]; st=time.time()
    for batch in tl:
      image=batch['image'].to(device); target=batch['mask'].to(device)
      with torch.no_grad(): coarse,_=base(image,False); prob=torch.sigmoid(coarse); boxes=boxes_from_probability(prob)
      patch,pt,_=crop_batch(image,prob,boxes,target,a.patch_size); logits=net(patch); loss=train_loss(logits,pt); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5); opt.step(); losses.append(float(loss.detach()))
    sch.step(); vm=metrics(*[z for z in (lambda q:(q['predictions'],q['targets']))(infer(net,base,vl,device,a.patch_size,True))]); row={'epoch':ep,'seconds':time.time()-st,'train_loss':float(np.mean(losses)),'val_macro_dice':vm['macro_dice'],'val_overlap_dice':vm['overlap_dice'],'val_overlap_nsd_2px':vm['overlap_nsd_2px']}; print(json.dumps(row),flush=True); open(a.output/'history.jsonl','a').write(json.dumps(row)+'\n'); key=vm['overlap_dice'] if vm['macro_dice']>=baseline_macro-0.002 else -1
    if key>best[0]: best=(key,ep); torch.save({'net':net.state_dict(),'metrics':vm,'epoch':ep},best_path)
  net.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['net']); raw=infer(net,base,tel,device,a.patch_size,False); imp=infer(net,base,tel,device,a.patch_size,True); bm=metrics(raw['predictions'],raw['targets']); im=metrics(imp['predictions'],imp['targets']); result={'experiment':'R298','device':str(device),'best_epoch':best[1],'baseline_test':bm,'improved_test':im,'delta_test':{k:im[k]-bm[k] for k in bm}}; (a.output/'result.json').write_text(json.dumps(result,indent=2)); cases=sorted(raw['targets'],key=lambda c:int((raw['targets'][c].sum(0)>=2).sum()),reverse=True)[:12]; render_comparisons(a.dataset_root,a.output/'visualizations',cases,raw,imp,384); (a.output/'manifest.json').write_text(json.dumps({'experiment':'R298','architecture':'frozen R285 + direct local 14-channel patch mask','crop_source':'coarse sigmoid overlap prediction','blend':'0.5 base logits + 0.5 direct patch logits','test_used_for_selection':False},indent=2)); print(json.dumps(result),flush=True)
if __name__=='__main__': main()
