"""Function-preserving R332 port with one appended explicit-O feature."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel,initialize_from_r285
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner,Block,per_instance_features
from train_r332_ram_joint_iterative_refinement import iterative_refine,flatten,audit_progression
from evaluate_r329_ram_official_metrics_val import official_metrics

EXPECTED='eaf3b6918f7b64d90ae4295be1df4ec4c2c314775e68472e85607667a4334655'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()

class R332RefinerWithO(InstanceCompletionRefiner):
    def __init__(self,width=12,residual_limit=2.5):
        super().__init__(width,residual_limit);self.enc1=Block(7,width)

def load_mature_with_zero_o(model,state):
    current=model.state_dict();changed=[]
    for key,value in state.items():
        if key=='enc1.net.0.weight':
            if value.shape!=(12,6,3,3) or current[key].shape!=(12,7,3,3):raise ValueError((key,value.shape,current[key].shape))
            target=torch.zeros_like(current[key]);target[:,:6].copy_(value);current[key]=target;changed.append(key)
        elif key not in current or current[key].shape!=value.shape:raise ValueError((key,value.shape,current.get(key,None)))
        else:current[key]=value.clone()
    model.load_state_dict(current,strict=True)
    return {'copied_tensor_count':len(state),'changed_tensors':changed,
            'original_feature_weights_exact':bool(torch.equal(model.enc1.net[0].weight[:,:6].detach().cpu(),state['enc1.net.0.weight'].detach().cpu())),
            'new_o_weight_nonzero':int(torch.count_nonzero(model.enc1.net[0].weight[:,6])),
            'mature_residual_exact':bool(torch.equal(model.residual.weight.detach().cpu(),state['residual.weight'].detach().cpu()) and torch.equal(model.residual.bias.detach().cpu(),state['residual.bias'].detach().cpu())),
            'mature_gate_exact':bool(torch.equal(model.gate.weight.detach().cpu(),state['gate.weight'].detach().cpu()) and torch.equal(model.gate.bias.detach().cpu(),state['gate.bias'].detach().cpu()))}

def features_with_o(image,probability,indices,source_probability,overlap):
    legacy=per_instance_features(image,probability,indices,source_probability)
    if overlap.shape!=(image.shape[0],1,*image.shape[-2:]):raise ValueError('O shape')
    return torch.cat((legacy,overlap.repeat(len(indices),1,1,1)),1)

def simultaneous_refine_o(image,logits,refiner,indices,overlap):
    probability=torch.sigmoid(logits);source=probability[:,indices]
    feature=features_with_o(image,probability,indices,source,overlap)
    corrected,delta=refiner(feature,logits[0,indices].unsqueeze(1))
    correction=corrected[:,0]-logits[0,indices]
    full=torch.zeros_like(logits);scatter=indices.view(1,-1,1,1).expand(1,-1,*logits.shape[-2:])
    return logits+full.scatter(1,scatter,correction.unsqueeze(0)),delta

def iterative_refine_o(image,base_logits,refiner,steps,overlap,indices=None,chunk=3):
    states=[base_logits];deltas=[];current=base_logits
    for _ in range(steps):
        if indices is not None:
            current,delta=simultaneous_refine_o(image,current,refiner,indices,overlap);deltas.append(delta)
        else:
            pieces=[];norms=[]
            for start in range(0,14,chunk):
                selected=torch.arange(start,min(start+chunk,14),device=image.device)
                updated,delta=simultaneous_refine_o(image,current,refiner,selected,overlap)
                pieces.append(updated-current);norms.append(delta)
            current=current+torch.stack(pieces).sum(0);deltas.append(torch.cat(norms,0))
        states.append(current)
    return states,deltas

@torch.inference_mode()
def parity(args):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True);start=time.time();device=torch.device('cuda')
    if sha(args.checkpoint)!=EXPECTED:raise ValueError('R332 SHA')
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=NnUNetMultiLabelPrior().to(device);model.load_state_dict(payload['model'],strict=True);model.eval()
    port_backbone=RamNnUNetTwoChannel().to(device);backbone_mapping=initialize_from_r285(port_backbone,args.checkpoint,device);port_backbone.eval()
    old=InstanceCompletionRefiner().to(device);old.load_state_dict(payload['refiner'],strict=True);old.eval()
    new=R332RefinerWithO().to(device);mapping=load_mature_with_zero_o(new,payload['refiner']);new.eval()
    ds=NativeWristDataset(args.dataset_root,'val',False);loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=1)
    pairs=top_training_pairs(args.dataset_root,15);old_data={f'step{i}':{} for i in range(3)};new_data={f'step{i}':{} for i in range(3)}
    maxima=[0.,0.,0.];mismatch=[0,0,0];delta_max=[0.,0.];delta_shapes=None;base_max=0.;base_mismatch=0;feature_max=0.;feature_shapes=None;o_feature_nonzero=0;cases=[]
    for batch in loader:
        image=batch['image'].to(device);truth=(batch['mask']>.5).numpy()[0];base=model(image,False)[0]
        port_base=port_backbone(torch.cat((image,torch.zeros_like(image)),1));base_max=max(base_max,float((base-port_base).abs().max()));base_mismatch+=int(torch.count_nonzero((base.sigmoid()>=.5)!=(port_base.sigmoid()>=.5)))
        probability=base.sigmoid();indices=torch.arange(0,3,device=device);source=probability[:,indices]
        legacy_feature=per_instance_features(image,probability,indices,source);port_feature=features_with_o(image,probability,indices,source,torch.zeros_like(image))
        feature_max=max(feature_max,float((legacy_feature-port_feature[:,:6]).abs().max()));o_feature_nonzero+=int(torch.count_nonzero(port_feature[:,6:]));feature_shapes={'legacy':list(legacy_feature.shape),'port':list(port_feature.shape)}
        a,ad=iterative_refine(image,base,old,2);b,bd=iterative_refine_o(image,port_base,new,2,torch.zeros_like(image))
        delta_shapes={'legacy':[list(x.shape) for x in ad],'port':[list(x.shape) for x in bd]}
        for i,(left,right) in enumerate(zip(ad,bd)):delta_max[i]=max(delta_max[i],float((left-right).abs().max()))
        case=batch['case'][0];cases.append(case)
        for i,(left,right) in enumerate(zip(a,b)):
            maxima[i]=max(maxima[i],float((left-right).abs().max()));lp=(left.sigmoid()>=.5).cpu().numpy()[0];rp=(right.sigmoid()>=.5).cpu().numpy()[0]
            mismatch[i]+=int(np.count_nonzero(lp!=rp));old_data[f'step{i}'][case]={'pred':lp,'target':truth};new_data[f'step{i}'][case]={'pred':rp,'target':truth}
    om={k:official_metrics(v,pairs) for k,v in old_data.items()};nm={k:official_metrics(v,pairs) for k,v in new_data.items()}
    metric_diff={step:{k:flatten(nm[step])[k]-flatten(om[step])[k] for k in flatten(om[step])} for step in om}
    official_identical=om==nm
    passed=base_max<=args.tolerance and base_mismatch==0 and feature_max<=args.tolerance and o_feature_nonzero==0 and max(delta_max)<=args.tolerance and max(maxima)<=args.tolerance and sum(mismatch)==0 and official_identical and max(abs(v) for x in metric_diff.values() for v in x.values() if v is not None)<=1e-12
    result={'status':'PASS' if passed else 'FAIL','checkpoint':str(args.checkpoint),'checkpoint_sha256':sha(args.checkpoint),'checkpoint_epoch':payload['epoch'],
            'cases':len(cases),'case_ids':cases,'dtype':'float32','device':str(device),'tolerance':args.tolerance,'mapping':mapping,'backbone_mapping':backbone_mapping,
            'feature_order':['image','own_probability','other_max','other_sum_clamped_div2','uncertainty','image_gradient','explicit_O'],
            'iteration':'2 steps; probabilities and all features recomputed from current logits at every step; chunks read same current state then update simultaneously',
            'base_max_abs_logit_difference':base_max,'base_mask_mismatch_pixels':base_mismatch,
            'feature_first6_max_abs_difference':feature_max,'feature_shapes':feature_shapes,'explicit_o_feature_nonzero':o_feature_nonzero,
            'delta_max_abs_difference':{f'step{i+1}':v for i,v in enumerate(delta_max)},'delta_shapes':delta_shapes,
            'step_max_abs_logit_difference':{f'step{i}':v for i,v in enumerate(maxima)},'step_mask_mismatch_pixels':{f'step{i}':v for i,v in enumerate(mismatch)},
            'metric_difference':metric_diff,'official_metrics_identical':official_identical,'legacy_metrics':om,'port_metrics':nm,'progression':audit_progression([flatten(om[f'step{i}']) for i in range(3)]),
            'mature_residual_preserved':mapping['mature_residual_exact'],'mature_gate_preserved':mapping['mature_gate_exact'],
            'test_used':False,'training_updates':0,'seconds':time.time()-start,'code_sha256':sha(Path(__file__))}
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps({'status':result['status'],'base_max':base_max,'base_mismatch':base_mismatch,'maxima':maxima,'mismatch':mismatch,'seconds':result['seconds']}),flush=True)
    if not passed:raise RuntimeError(result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tolerance',type=float,default=1e-6);a=p.parse_args();parity(a)
if __name__=='__main__':main()
