#!/usr/bin/env python3
"""R297: actual high-resolution overlap-patch refinement on RAM-W600.

The mature R285 nnU-Net is frozen. A candidate overlap box is generated from
its own coarse sigmoid predictions (never from GT at validation/test time),
cropped at native resolution, refined by a separate patch encoder, and pasted
back to the full-resolution logits. RAM remains 14-channel multi-label.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, WristDataset, render_comparisons, seed_everything, surface_dice


class PatchRefiner(nn.Module):
    def __init__(self, in_channels=15, hidden=32):
        super().__init__()
        self.body=nn.Sequential(
            nn.Conv2d(in_channels,hidden,3,padding=1,bias=False),nn.GroupNorm(8,hidden),nn.SiLU(inplace=True),
            nn.Conv2d(hidden,hidden,3,padding=1,bias=False),nn.GroupNorm(8,hidden),nn.SiLU(inplace=True),
            nn.Conv2d(hidden,hidden,3,padding=1,bias=False),nn.GroupNorm(8,hidden),nn.SiLU(inplace=True),
        )
        self.head=nn.Conv2d(hidden,14,1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x): return torch.tanh(self.head(self.body(x)))


def boxes_from_probability(prob, threshold=0.35, margin=32, min_side=160, max_side=320):
    """Return square boxes (y0,y1,x0,x1) from coarse overlap predictions."""
    out=[]; masks=(prob>threshold).sum(1)>=2
    for m in masks:
        ys,xs=torch.where(m)
        h,w=m.shape
        if ys.numel()==0: cy,cx=h//2,w//2; side=min_side
        else:
            y0,y1=int(ys.min()),int(ys.max())+1; x0,x1=int(xs.min()),int(xs.max())+1
            side=min(max(max(y1-y0,x1-x0)+2*margin,min_side),max_side)
            cy=(y0+y1)//2; cx=(x0+x1)//2
        side=min(side,h,w); y0=max(0,min(h-side,cy-side//2)); x0=max(0,min(w-side,cx-side//2))
        out.append((y0,y0+side,x0,x0+side))
    return out


def crop_batch(image, prob, boxes, target, patch_size):
    xs=[]; ys=[]; loc=[]
    for b,(y0,y1,x0,x1) in enumerate(boxes):
        inp=torch.cat([image[b:b+1],prob[b:b+1]],1)[:,:,y0:y1,x0:x1]
        tar=target[b:b+1,:,y0:y1,x0:x1]
        xs.append(F.interpolate(inp,(patch_size,patch_size),mode='bilinear',align_corners=False))
        ys.append(F.interpolate(tar.float(),(patch_size,patch_size),mode='nearest'))
        loc.append((b,y0,y1,x0,x1))
    return torch.cat(xs,0),torch.cat(ys,0),loc


def metrics(preds,targets):
    cd=[]; od=[]; on=[]
    for c in sorted(preds):
        p=preds[c]; t=targets[c]
        cd.append(float(np.mean([(2*(p[k]&t[k]).sum()+1)/(p[k].sum()+t[k].sum()+1) for k in range(14)])))
        po=p.sum(0)>=2; to=t.sum(0)>=2
        od.append(float((2*(po&to).sum()+1)/(po.sum()+to.sum()+1))); on.append(surface_dice(po,to,2))
    return {'macro_dice':float(np.mean(cd)),'overlap_dice':float(np.mean(od)),'overlap_nsd_2px':float(np.mean(on))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,default=Path(r'G:\gutou\RAM-W600'))
    ap.add_argument('--baseline-checkpoint',type=Path,default=Path('outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth'))
    ap.add_argument('--output',type=Path,default=Path('outputs/ram_w600/r297_overlap_patch_refiner'))
    ap.add_argument('--epochs',type=int,default=18); ap.add_argument('--patch-size',type=int,default=192); ap.add_argument('--batch-size',type=int,default=2); ap.add_argument('--seed',type=int,default=297)
    a=ap.parse_args(); seed_everything(a.seed); a.output.mkdir(parents=True,exist_ok=True)
    if 'TSRS_RSNA' in str(a.dataset_root): raise RuntimeError('R297 is RAM only')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tr=WristDataset(a.dataset_root,'train',384,True); va=WristDataset(a.dataset_root,'val',384,False); te=WristDataset(a.dataset_root,'test',384,False)
    tl=DataLoader(tr,batch_size=a.batch_size,shuffle=True,num_workers=0); vl=DataLoader(va,batch_size=a.batch_size,shuffle=False,num_workers=0); tel=DataLoader(te,batch_size=a.batch_size,shuffle=False,num_workers=0)
    base=NnUNetMultiLabelPrior().to(device); payload=torch.load(a.baseline_checkpoint,map_location=device,weights_only=False); base.load_state_dict(payload['model']); base.eval()
    for p in base.parameters(): p.requires_grad=False
    refiner=PatchRefiner().to(device); opt=torch.optim.AdamW(refiner.parameters(),lr=2e-4,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    baseline_macro=float(payload['metrics']['macro_dice']); best=(-1,None); best_path=a.output/'patch_refiner_best.pth'
    for ep in range(a.epochs):
        refiner.train(); sums=[]; start=time.time()
        for batch in tl:
            image=batch['image'].to(device); target=batch['mask'].to(device)
            with torch.no_grad(): coarse,_=base(image,False); prob=torch.sigmoid(coarse); boxes=boxes_from_probability(prob)
            patch,pt,loc=crop_batch(image,prob,boxes,target,a.patch_size); corr=refiner(patch)
            # Local refined logits; base probability is converted to logits in patch space.
            patch_prob=patch[:,1:]; patch_base_logit=torch.logit(patch_prob.clamp(1e-4,1-1e-4)); out=patch_base_logit+0.9*corr
            pp=torch.sigmoid(out).float(); tt=pt.float(); ov_bool=(tt.sum(1)>=2).unsqueeze(1).expand_as(tt); ov=ov_bool.float()
            axes=(0,2,3); dice=1-((2*(pp*tt).sum(axes)+1)/(pp.sum(axes)+tt.sum(axes)+1)).mean(); bce=F.binary_cross_entropy_with_logits(out,tt)
            if ov_bool.any(): odice=1-(2*(pp*tt*ov).sum()+1)/((pp*ov).sum()+(tt*ov).sum()+1); obce=F.binary_cross_entropy(pp[ov_bool],tt[ov_bool])
            else: odice=out.new_zeros(()); obce=out.new_zeros(())
            loss=dice+bce+0.75*(odice+obce); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(refiner.parameters(),5); opt.step(); sums.append(float(loss.detach()))
        sched.step(); refiner.eval(); preds={}; targets={}
        with torch.no_grad():
            for batch in vl:
                image=batch['image'].to(device); coarse,_=base(image,False); prob=torch.sigmoid(coarse); boxes=boxes_from_probability(prob); patch,_,loc=crop_batch(image,prob,boxes,batch['mask'].to(device),a.patch_size); corr=refiner(patch)
                for j,(b,y0,y1,x0,x1) in enumerate(loc):
                    refined=coarse[b].clone(); base_patch=torch.logit(prob[b,:,y0:y1,x0:x1].clamp(1e-4,1-1e-4)); up=F.interpolate(0.9*corr[j:j+1],(y1-y0,x1-x0),mode='bilinear',align_corners=False)[0]; refined[:,y0:y1,x0:x1]=base_patch+up; preds[str(batch['case'][b])]=(torch.sigmoid(refined)>=0.5).cpu().numpy(); targets[str(batch['case'][b])]=(batch['mask'][b].numpy()>0.5)
        vm=metrics(preds,targets); row={'epoch':ep,'seconds':time.time()-start,'train_loss':float(np.mean(sums)),'val_macro_dice':vm['macro_dice'],'val_overlap_dice':vm['overlap_dice'],'val_overlap_nsd_2px':vm['overlap_nsd_2px']}; print(json.dumps(row),flush=True); open(a.output/'history.jsonl','a',encoding='utf8').write(json.dumps(row)+'\n')
        key=vm['overlap_dice'] if vm['macro_dice']>=baseline_macro-0.002 else -1
        if key>best[0]: best=(key,ep); torch.save({'refiner':refiner.state_dict(),'metrics':vm,'epoch':ep},best_path)
    refiner.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['refiner']); refiner.eval()
    def infer(loader, use_ref):
        preds={}; targets={}
        with torch.no_grad():
            for batch in loader:
                image=batch['image'].to(device); coarse,_=base(image,False); prob=torch.sigmoid(coarse); boxes=boxes_from_probability(prob); corr=None
                if use_ref:
                    patch,_,loc=crop_batch(image,prob,boxes,batch['mask'].to(device),a.patch_size); corr=refiner(patch)
                for b,c in enumerate(batch['case']):
                    refined=coarse[b].clone()
                    if use_ref:
                        item=[z for z in loc if z[0]==b][0]; _,y0,y1,x0,x1=item; idx=[j for j,z in enumerate(loc) if z[0]==b][0]; base_patch=torch.logit(prob[b,:,y0:y1,x0:x1].clamp(1e-4,1-1e-4)); up=F.interpolate(0.9*corr[idx:idx+1],(y1-y0,x1-x0),mode='bilinear',align_corners=False); refined[:,y0:y1,x0:x1]=base_patch+up[0]
                    preds[str(c)]=(torch.sigmoid(refined)>=0.5).cpu().numpy(); targets[str(c)]=(batch['mask'][b].numpy()>0.5)
        return {'predictions':preds,'targets':targets}
    raw=infer(tel,False); imp=infer(tel,True); bm=metrics(raw['predictions'],raw['targets']); im=metrics(imp['predictions'],imp['targets']); result={'experiment':'R297','device':str(device),'baseline_checkpoint':str(a.baseline_checkpoint),'best_epoch':best[1],'baseline_test':bm,'improved_test':im,'delta_test':{k:im[k]-bm[k] for k in bm}}
    (a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8'); cases=sorted(raw['targets'],key=lambda c:int((raw['targets'][c].sum(0)>=2).sum()),reverse=True)[:12]; render_comparisons(a.dataset_root,a.output/'visualizations',cases,raw,imp,384); (a.output/'manifest.json').write_text(json.dumps({'experiment':'R297','architecture':'frozen R285 nnU-Net + native-resolution overlap patch refiner','crop_from':'coarse sigmoid overlap prediction, no GT at val/test','patch_size':a.patch_size,'threshold':0.5,'test_used_for_selection':False},indent=2),encoding='utf8'); print(json.dumps(result),flush=True)

if __name__=='__main__': main()
