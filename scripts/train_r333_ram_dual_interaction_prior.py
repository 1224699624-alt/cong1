#!/usr/bin/env python3
"""R333-R: combine RAM-native seam maps with R332 iterative overlap completion."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,dice_bce_logits,top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import completion_loss,seed_all
from train_r330_ram_pair_surface_volume_prior import choose_pair_instances,soft_surface_nsd_loss,volume_losses
from train_r332_ram_joint_iterative_refinement import flatten,set_joint_trainability,checkpoint_gate
from r333_dual_interaction_prior import DualInteractionRefiner,absolute_seam_loss,initialize_from_six_channel,instance_features,state_masks


class RamSeamDataset(NativeWristDataset):
    def __init__(self,root,prior_root,split,augment): super().__init__(root,split,augment); self.prior_root=Path(prior_root)
    def __getitem__(self,index):
        item=super().__getitem__(index); case=item["case"]; path=self.prior_root/self.split/f"{case}.npy"
        if path.exists(): value=torch.from_numpy(np.load(path).astype(np.float32))[None,None]
        else:
            from PIL import Image
            p=self.prior_root/self.split/f"{case}.png"; value=torch.from_numpy(np.asarray(Image.open(p),dtype=np.float32)/255.)[None,None]
        h,w=item["original_hw"].tolist(); value=F.interpolate(value,size=(h,w),mode="bilinear",align_corners=False)[0]
        value=F.pad(value,(0,item["image"].shape[-1]-w,0,item["image"].shape[-2]-h))
        item["seam"]=value; return item


def refine(image,logits,model,seam,steps,indices=None):
    states=[logits];deltas=[];current=logits
    for _ in range(steps):
        probability=torch.sigmoid(current); correction=torch.zeros_like(current); ds=[]
        groups=[indices] if indices is not None else [torch.arange(s,min(s+3,14),device=image.device) for s in range(0,14,3)]
        for chosen in groups:
            source=probability[:,chosen]; feature=instance_features(image,probability,chosen,source,seam)
            corrected,delta=model(feature,current[0,chosen].unsqueeze(1)); correction[0,chosen]=corrected[:,0]-current[0,chosen];ds.append(delta)
        current=current+correction;states.append(current);deltas.append(torch.cat(ds))
    return states,deltas


@torch.inference_mode()
def collect(network,refiner,loader,device,steps,pairs):
    network.eval();refiner.eval(); out={f"step{s}":{} for s in range(steps+1)}; seam_fp={f"step{s}":[] for s in range(steps+1)}
    for b in loader:
        image=b["image"].to(device);target=(b["mask"].to(device)>0.5);seam=b["seam"].to(device)
        logits,_=network(image,False);states,_=refine(image,logits,refiner,seam,steps);sep,_,_=state_masks(target.float(),seam)
        for s,z in enumerate(states):
            pred=torch.sigmoid(z)>=.5; key=f"step{s}"; valid=sep>0.25
            seam_fp[key].append(float((pred.any(1,keepdim=True)&valid).sum()/valid.sum().clamp_min(1)))
            out[key][str(b["case"][0])]={"pred":pred.cpu().numpy()[0],"target":target.cpu().numpy()[0]}
    return {k:{"official":official_metrics(v,pairs),"seam_fp":float(np.mean(seam_fp[k]))} for k,v in out.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True)
    p.add_argument('--r332-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--freeze-epochs',type=int,default=4);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--seam-weight',type=float,default=.08);p.add_argument('--seed',type=int,default=3331);p.add_argument('--smoke',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    seed_all(a.seed);device=torch.device('cuda');payload=torch.load(a.r332_checkpoint,map_location=device,weights_only=False)
    network=NnUNetMultiLabelPrior().to(device);network.load_state_dict(payload['model'],strict=True)
    refiner=DualInteractionRefiner().to(device);init=initialize_from_six_channel(refiner,payload['refiner'])
    train=RamSeamDataset(a.dataset_root,a.prior_root,'train',True);val=RamSeamDataset(a.dataset_root,a.prior_root,'val',False)
    if a.smoke: train.mask_files=train.mask_files[:2];val.mask_files=val.mask_files[:2]
    tl=DataLoader(train,batch_size=1,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True);vl=DataLoader(val,batch_size=1,shuffle=False,num_workers=1,pin_memory=True)
    names=set_joint_trainability(network,True);decoder=[x for x in network.parameters() if x.requires_grad]
    opt=torch.optim.AdamW([{'params':refiner.parameters(),'lr':8e-5},{'params':decoder,'lr':7e-6}],weight_decay=1e-4);scaler=torch.cuda.amp.GradScaler(True);pairs=top_training_pairs(a.dataset_root,15)
    def batch_loss(b):
        image=b['image'].to(device);target=b['mask'].to(device);seam=b['seam'].to(device);base,_=network(image,False);prob=torch.sigmoid(base)
        idx=choose_pair_instances(prob.detach(),target,4);sel=target[0,idx].unsqueeze(1);local=F.max_pool2d((torch.abs(prob[0,idx].unsqueeze(1).detach()-sel)>.15).float(),11,1,5)
        states,deltas=refine(image,base,refiner,seam,a.steps,idx); losses=[]
        for s in range(1,a.steps+1):
            z=states[s][0,idx].unsqueeze(1);main,_=completion_loss(z,sel,local,deltas[s-1],.05);q=torch.sigmoid(z.float());iv,pv=volume_losses(q,sel)
            losses.append(main+.03*soft_surface_nsd_loss(q,sel)+.005*iv+.01*pv)
        sep,ov,_=state_masks(target,seam); seam_loss=absolute_seam_loss(states[-1],sep,.10)
        mono=sum((F.relu(losses[i]-losses[i-1].detach()) for i in range(1,len(losses))),start=base.sum()*0)
        total=sum((i+1)*x for i,x in enumerate(losses))/sum(range(1,len(losses)+1))+.05*mono+.10*dice_bce_logits(base,target)+a.seam_weight*seam_loss
        return total,seam_loss,float(sep.mean()),float(ov.mean())
    if a.smoke:
        network.train();refiner.train();b=next(iter(tl));loss,sl,sm,om=batch_loss(b);loss.backward();print(json.dumps({'smoke':True,'finite':bool(torch.isfinite(loss)),'loss':float(loss),'seam_loss':float(sl),'separation_mass':sm,'overlap_mass':om,'init':init}));return
    base_eval=collect(network,refiner,vl,device,a.steps,pairs);base_flat=flatten(base_eval['step0']['official']);best=None;history=a.output/'history.jsonl'
    for epoch in range(1,a.epochs+1):
        joint=epoch>a.freeze_epochs;set_joint_trainability(network,joint);network.eval();refiner.train();tot=[0.,0.];started=time.time()
        for b in tl:
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(True):loss,sl,_,_=batch_loss(b)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(refiner.parameters(),12)
            if joint:torch.nn.utils.clip_grad_norm_(decoder,5)
            scaler.step(opt);scaler.update();tot[0]+=float(loss);tot[1]+=float(sl)
        metrics=collect(network,refiner,vl,device,a.steps,pairs);final=flatten(metrics[f'step{a.steps}']['official']);gate,checks=checkpoint_gate(final,base_flat)
        row={'epoch':epoch,'seconds':time.time()-started,'joint':joint,'train_loss':tot[0]/len(tl),'train_seam_loss':tot[1]/len(tl),'gate':gate,'checks':checks,'metrics':{k:{'flat':flatten(v['official']),'seam_fp':v['seam_fp']} for k,v in metrics.items()}}
        history.open('a').write(json.dumps(row)+'\n');print(json.dumps(row),flush=True);score=(int(gate),-metrics[f'step{a.steps}']['seam_fp'],final['overlap_nsd_2px'],final['overlap_dsc'],-final['overlap_msd_px'])
        if best is None or score>best[0]:best=(score,row);torch.save({'model':network.state_dict(),'refiner':refiner.state_dict(),'row':row,'epoch':epoch},a.output/'best.pth')
        torch.save({'model':network.state_dict(),'refiner':refiner.state_dict(),'row':row,'epoch':epoch,'optimizer':opt.state_dict(),'scaler':scaler.state_dict()},a.output/'last.pth')
    result={'experiment':'R333_RAM_DUAL_INTERACTION_PRIOR','split':'validation','test_used':False,'threshold':.5,'threshold_search':False,'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},'initialization':init,'baseline':base_eval,'best':best[1]}
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({'complete':True,'best_epoch':best[1]['epoch']},indent=2))
if __name__=='__main__':main()
