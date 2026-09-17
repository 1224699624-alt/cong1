#!/usr/bin/env python3
"""R333-T: frozen R317 seam input plus conservative output completion refiner.

Development uses TSRS train/original-val only. clean-test-v2 is never opened.
"""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset,DataLoader

from r333_dual_interaction_prior import DualInteractionRefiner,absolute_seam_loss,binary_features,initialize_from_six_channel
from run_r201_unified_eval import compute_metrics
from train_r256_scale_invariant_pair_prior import find_image
from train_r327_ram_native_instance_completion_prior import completion_loss,seed_all


class TsrsDataset(Dataset):
    def __init__(self,root,prior_root,split,size=512,augment=False):
        self.root=Path(root);self.prior_root=Path(prior_root);self.split=split;self.size=size;self.augment=augment
        self.labels=sorted((self.root/f'{split}_labels').glob('*.png'))
        if not self.labels:raise RuntimeError(f'no {split} labels')
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        lp=self.labels[i];stem=lp.stem;image=Image.open(find_image(self.root/self.split,stem)).convert('L');label=Image.open(lp).convert('L');prior=Image.open(self.prior_root/self.split/f'{stem}.png').convert('L')
        original=np.asarray(image);target=(np.asarray(label)>0).astype(np.float32);seam=np.asarray(prior,dtype=np.float32)/255.
        image=np.asarray(image.resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32)/255.
        target=np.asarray(Image.fromarray(target).resize((self.size,self.size),Image.Resampling.NEAREST),dtype=np.float32)
        seam=np.asarray(Image.fromarray(seam).resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32)
        if self.augment and random.random()<.5:image=np.fliplr(image).copy();target=np.fliplr(target).copy();seam=np.fliplr(seam).copy()
        image=torch.from_numpy(image)[None];image=(image-image.mean())/(image.std()+1e-6)
        return {'image':image,'target':torch.from_numpy(target)[None],'seam':torch.from_numpy(seam)[None],'case':stem,'original_hw':torch.tensor(original.shape)}


def corrupt(target,seam):
    # GT-based synthetic errors: false bridge only in seam background and local FN inside bone.
    smooth=F.avg_pool2d(target,5,1,2).clamp(.05,.95);dilated=F.max_pool2d(target, nine:=9,1,nine//2)
    bridge=(dilated-target).clamp(0,1)*(seam>.25).float();field=F.interpolate(torch.rand((target.shape[0],1,16,16),device=target.device),size=target.shape[-2:],mode='bicubic',align_corners=False)
    add=bridge*(field>.58).float();boundary=(target-(1-F.max_pool2d(1-target,5,1,2))).clamp(0,1);drop=target*boundary*(field<.35).float()
    source=(smooth+.65*add-.75*drop).clamp(.05,.95);local=F.max_pool2d(((add+drop)>0).float(),11,1,5)
    return source,local


def refine(image,source,seam,model,steps):
    states=[torch.logit(source.clamp(.05,.95))];deltas=[]
    for _ in range(steps):
        prob=torch.sigmoid(states[-1]);z,d=model(binary_features(image,prob,seam),states[-1]);states.append(z);deltas.append(d)
    return states,deltas


def metric_rows(pred_dir,gt_dir):
    rows=[]
    for gt_path in sorted(Path(gt_dir).glob('*.png')):
        pred=np.asarray(Image.open(Path(pred_dir)/gt_path.name))>0;gt=np.asarray(Image.open(gt_path))>0
        rows.append(compute_metrics(pred,gt,3,9,2.,5.))
    keys=rows[0].keys();return {'mean':{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys},'per_image':rows,'n':len(rows)}


@torch.inference_mode()
def evaluate(model,loader,baseline_masks,gt_dir,output,steps):
    model.eval();dirs=[]
    for s in range(steps+1):(output/f'step{s}').mkdir(parents=True,exist_ok=True);dirs.append(output/f'step{s}')
    for b in loader:
        image=b['image'].cuda();seam=b['seam'].cuda();case=b['case'][0];base=np.asarray(Image.open(Path(baseline_masks)/f'{case}.png'))>0
        h,w=b['original_hw'][0].tolist();small=np.asarray(Image.fromarray(base.astype(np.uint8)*255).resize((image.shape[-1],image.shape[-2]),Image.Resampling.NEAREST))>0
        source=torch.from_numpy(np.ascontiguousarray(small.astype(np.float32)))[None,None].cuda()*.8+.1
        states,_=refine(image,source,seam,model,steps)
        for s,z in enumerate(states):
            pred=(torch.sigmoid(z)[0,0].cpu().numpy()>=.5).astype(np.uint8)*255
            pred=np.asarray(Image.fromarray(pred).resize((w,h),Image.Resampling.NEAREST));Image.fromarray(pred).save(dirs[s]/f'{case}.png')
    return {f'step{s}':metric_rows(dirs[s],gt_dir) for s in range(steps+1)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True);p.add_argument('--baseline-masks',type=Path,required=True);p.add_argument('--r332-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--steps',type=int,default=2);p.add_argument('--seam-weight',type=float,default=.08);p.add_argument('--completion-weight',type=float,default=.30);p.add_argument('--seed',type=int,default=3332);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    joined=' '.join(map(str,vars(a).values())).lower()
    if 'clean-test' in joined or 'articular' in joined:raise RuntimeError('R333-T development forbids clean-test/articular')
    seed_all(a.seed);payload=torch.load(a.r332_checkpoint,map_location='cpu',weights_only=False);model=DualInteractionRefiner().cuda();init=initialize_from_six_channel(model,payload['refiner'])
    train=TsrsDataset(a.dataset_root,a.prior_root,'train',512,True);val=TsrsDataset(a.dataset_root,a.prior_root,'val',512,False)
    if a.smoke:train.labels=train.labels[:2];val.labels=val.labels[:2]
    tl=DataLoader(train,batch_size=2,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True);vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(True)
    def loss_batch(b):
        image=b['image'].cuda();target=b['target'].cuda();seam=b['seam'].cuda();source,local=corrupt(target,seam);states,deltas=refine(image,source,seam,model,a.steps);losses=[]
        for s in range(1,a.steps+1):main,_=completion_loss(states[s],target,local,deltas[s-1],.02);losses.append(main)
        separation=(seam-seam.amin((-2,-1),keepdim=True))/(seam.amax((-2,-1),keepdim=True)-seam.amin((-2,-1),keepdim=True)+1e-6);separation=separation.square()*(target<.5)
        seam_loss=absolute_seam_loss(states[-1],separation,.10);mono=sum((F.relu(losses[i]-losses[i-1].detach()) for i in range(1,len(losses))),start=states[0].sum()*0)
        total=a.completion_weight*sum((i+1)*x for i,x in enumerate(losses))/sum(range(1,len(losses)+1))+a.seam_weight*seam_loss+.05*mono
        return total,seam_loss
    if a.smoke:
        b=next(iter(tl));loss,sl=loss_batch(b);loss.backward();print(json.dumps({'smoke':True,'finite':bool(torch.isfinite(loss)),'loss':float(loss),'seam_loss':float(sl),'grad':sum(float(x.grad.abs().sum()) for x in model.parameters() if x.grad is not None),'init':init}));return
    base=metric_rows(a.baseline_masks,a.dataset_root/'val_labels');best=None;history=a.output/'history.jsonl'
    for epoch in range(1,a.epochs+1):
        model.train();tot=[0.,0.];started=time.time()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):loss,sl=loss_batch(b)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),12);scaler.step(opt);scaler.update();tot[0]+=float(loss);tot[1]+=float(sl)
        metrics=evaluate(model,vl,a.baseline_masks,a.dataset_root/'val_labels',a.output/'predictions_current',a.steps);m=metrics[f'step{a.steps}']['mean'];b=base['mean'];gate=m['dice']>=b['dice'] and m['iou']>=b['iou']
        row={'epoch':epoch,'seconds':time.time()-started,'train_loss':tot[0]/len(tl),'train_seam_loss':tot[1]/len(tl),'gate':gate,'metrics':{k:v['mean'] for k,v in metrics.items()}}
        history.open('a').write(json.dumps(row)+'\n');print(json.dumps(row),flush=True);score=(int(gate),m['surface_dice_2px'],-m['hd95_px'],-m['assd_px'],-m['gap_region_fp_rate'])
        if best is None or score>best[0]:best=(score,row);torch.save({'refiner':model.state_dict(),'row':row,'epoch':epoch},a.output/'best.pth')
        torch.save({'refiner':model.state_dict(),'row':row,'epoch':epoch,'optimizer':opt.state_dict(),'scaler':scaler.state_dict()},a.output/'last.pth')
    result={'experiment':'R333_TSRS_DUAL_INTERACTION_PRIOR','split':'original-val','clean_test_used':False,'baseline':base,'best':best[1],'initialization':init,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'complete':True,'best_epoch':best[1]['epoch']},indent=2))
if __name__=='__main__':main()
