"""Bounded RAM N0/A/B/C/D Round3 runner. Train/val only."""
from __future__ import annotations
import argparse,copy,json,math,random,tempfile,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel,initialize_from_r285
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior,dice_bce_logits,top_training_pairs
from evaluate_r329_ram_official_metrics_val import official_metrics
from structural_prior_v3 import build_backbone_input,OutputOverlapAdapter
from local_supervision_v3 import local_supervision
from r349_parameter_gradient_gate_v3 import route_parameter_gradients_v3,assign_gradients_v3
from round3_common import sha,save,load_lock,require_manifest,load_accepted_overlap,resource_guard

class DatasetV3(Dataset):
    def __init__(self,root,prior_root,split,use_seam):
        self.base=NativeWristDataset(root,split,False);self.prior_root=prior_root;self.split=split;self.use_seam=use_seam
    def __len__(self):return len(self.base)
    @property
    def mask_files(self):return self.base.mask_files
    def __getitem__(self,i):
        x=self.base[i]
        if not self.use_seam:
            x['seam']=torch.zeros_like(x['image']);x['seam_source']='DISABLED_NO_FILE_READ';return x
        path=self.prior_root/self.split/(x['case']+'.npy')
        if not path.is_file():raise FileNotFoundError(path)
        p=np.load(path)
        if p.shape!=(384,384) or not np.isfinite(p).all():raise ValueError(path)
        h,w=x['original_hw'].tolist();s=F.interpolate(torch.from_numpy(p.copy())[None,None],(h,w),mode='bilinear',align_corners=False)[0]
        x['seam']=F.pad(s,(0,x['image'].shape[-1]-w,0,x['image'].shape[-2]-h));x['seam_source']=str(path)
        return x

def unpack(b,device):
    return [b[k].to(device) for k in ['image','seam','mask','valid']]
def forward(model,refiner,overlap,image,seam,arm):
    logits=model(build_backbone_input(image,seam if arm in 'BD' else None,arm))
    o=None
    if arm in 'CD':
        with torch.no_grad():o=overlap(image).sigmoid()
        logits=refiner.refine(image,logits,o,1)
    return logits,o

@torch.inference_mode()
def collect(model,refiner,overlap,loader,device,arm,pairs):
    model.eval();refiner.eval() if refiner else None;out={};sources=[]
    for b in loader:
        image,seam,target,valid=unpack(b,device);z,o=forward(model,refiner,overlap,image,seam,arm)
        pred=(z.sigmoid()>=.5).cpu().numpy()[0];truth=(target>.5).cpu().numpy()[0]
        case=b['case'][0]
        if case in out:raise ValueError('duplicate')
        out[case]={'pred':pred,'target':truth};sources.append(str(b['seam_source'][0]))
    if len(out)!=69:raise ValueError('not full val')
    aggregate=official_metrics(out,pairs)
    per_case={k:official_metrics({k:v},pairs) for k,v in out.items()}
    return {'official':aggregate,'per_case':per_case,'cases':list(out),'seam_sources':sources}

def save_best(path,model,refiner,arm,epoch,metric,source,prior_report,protocol):
    payload={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'arm':arm,'epoch':epoch,'selection_value':metric,
             'source_checkpoint_sha256':source,'protocol_sha256':sha(protocol),'runner_sha256':sha(Path(__file__))}
    if refiner:payload['output_refiner']={k:v.detach().cpu() for k,v in refiner.state_dict().items()}
    if prior_report:payload['overlap_checkpoint_sha256']=prior_report['checkpoint_sha256']
    torch.save(payload,path);return sha(path)

def build(args,device):
    lock=load_lock(args.protocol);require_manifest(args.protocol)
    if sha(args.native)!=lock['native_sha256']['RAM']:raise ValueError('native sha')
    model=RamNnUNetTwoChannel().to(device);conversion=initialize_from_r285(model,args.native,device)
    refiner=OutputOverlapAdapter().to(device) if args.arm in 'CD' else None
    overlap=prior_report=None
    if args.arm in 'CD':overlap,prior_report=load_accepted_overlap(args.overlap_checkpoint,args.overlap_acceptance,device)
    return lock,model,refiner,overlap,prior_report,conversion

def baseline(args,device):
    lock=load_lock(args.protocol);require_manifest(args.protocol)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);data=args.root/'data/remote_variants/RAM-W600'
    ds=DatasetV3(data,args.prior_root,'val',False);loader=DataLoader(ds,batch_size=1,num_workers=2)
    native=NnUNetMultiLabelPrior().to(device);native.load_state_dict(torch.load(args.native,map_location='cpu',weights_only=False)['model'],strict=True);native.eval()
    converted=RamNnUNetTwoChannel().to(device);conversion=initialize_from_r285(converted,args.native,device);converted.eval()
    pairs=top_training_pairs(data,15);out_native={};out_converted={};maxdiff=0;maskdiff=0
    with torch.inference_mode():
        for b in loader:
            image,seam,target,valid=unpack(b,device);a=native(image,False)[0];z=converted(torch.cat((image,torch.zeros_like(image)),1))
            maxdiff=max(maxdiff,float((a-z).abs().max()));maskdiff+=int(torch.count_nonzero((a.sigmoid()>=.5)!=(z.sigmoid()>=.5)))
            case=b['case'][0];truth=(target>.5).cpu().numpy()[0];out_native[case]={'pred':(a.sigmoid()>=.5).cpu().numpy()[0],'target':truth};out_converted[case]={'pred':(z.sigmoid()>=.5).cpu().numpy()[0],'target':truth}
    if maxdiff>1e-5 or maskdiff:raise RuntimeError('N0 conversion gate')
    report={'status':'PASS','cases':69,'max_abs_logit_difference':maxdiff,'mask_mismatch_pixels':maskdiff,
            'conversion':conversion,'source_sha256':sha(args.native),'official':official_metrics(out_native,pairs),
            'converted_official':official_metrics(out_converted,pairs),'test_used':False}
    report['per_case']={case:official_metrics({case:item},pairs) for case,item in out_native.items()}
    save(args.output/'result.json',report);print(json.dumps(report),flush=True)

def smoke(args,device):
    if args.arm not in 'CD':raise ValueError('smoke C/D')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);lock,model,refiner,overlap,prior_report,conversion=build(args,device)
    data=args.root/'data/remote_variants/RAM-W600';ds=DatasetV3(data,args.prior_root,'train',args.arm=='D');b=next(iter(DataLoader(ds,batch_size=1,shuffle=False)))
    image,seam,target,valid=unpack(b,device);teacher=copy.deepcopy(model).eval().requires_grad_(False);refiner.train();model.train()
    with torch.no_grad():native=teacher(torch.cat((image,torch.zeros_like(image)),1)).sigmoid()
    initial,o=forward(model,refiner,overlap,image,seam,args.arm);identity=float((initial-teacher(build_backbone_input(image,seam if args.arm=='D' else None,args.arm))).abs().max())
    base=dice_bce_logits(initial,target);terms,states=local_supervision(initial,target,torch.ones_like(target,dtype=torch.bool),target.any(1,keepdim=True),valid,valid,seam,native,enable_seam=args.arm=='D',enable_overlap=True)
    params=list(model.parameters())+list(refiner.parameters());aux=[]
    for name,weight in [('separation',.035),('overlap',.035),('preserve',.10)]:
        if name=='separation' and args.arm!='D':continue
        aux.append((name,terms[name],weight*states['attenuation'][name]))
    gradients,rows=route_parameter_gradients_v3(params,base,aux,max_aux_ratio=.15,max_merged_aux_ratio=.15);assign_gradients_v3(params,gradients)
    finite=all(g is None or torch.isfinite(g).all() for g in gradients);refgrad=float(refiner.residual.weight.grad.abs().sum())
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':2e-6},{'params':refiner.parameters(),'lr':2e-5}],weight_decay=1e-4);opt.step()
    with torch.no_grad():
        after,o=forward(model,refiner,overlap,image,seam,args.arm);zero=refiner.refine(image,model(build_backbone_input(image,seam if args.arm=='D' else None,args.arm)),torch.zeros_like(o),1)
        osensitivity=float((after-zero).abs().max())
        if args.arm=='C':
            changed=forward(model,refiner,overlap,image,torch.rand_like(seam),'C')[0];seam_invariance=float((after-changed).abs().max())
        else:seam_invariance=None
    with tempfile.TemporaryDirectory(dir=args.output) as td:
        p=Path(td)/'x.pth';torch.save({'model':model.state_dict(),'refiner':refiner.state_dict()},p);x=torch.load(p,map_location=device,weights_only=False);model.load_state_dict(x['model'],strict=True);refiner.load_state_dict(x['refiner'],strict=True)
        with torch.no_grad():reload=forward(model,refiner,overlap,image,seam,args.arm)[0]
    report={'status':'PASS' if identity==0 and finite and refgrad>0 and osensitivity>0 and (seam_invariance in (None,0.)) and float((after-reload).abs().max())==0 else 'FAIL',
            'arm':args.arm,'case':b['case'][0],'identity_max_abs':identity,'finite_gradients':bool(finite),'refiner_residual_grad_sum':refgrad,
            'overlap_support_mass':states['masses']['overlap'],'separation_support_mass':states['masses']['separation'],
            'overlap_input':{'shape':list(o.shape),'min':float(o.min()),'max':float(o.max()),'mean':float(o.mean())},
            'O_sensitivity_after_step':osensitivity,'C_seam_perturbation_max_abs':seam_invariance,'strict_reload_max_abs':float((after-reload).abs().max()),
            'routing':rows,'overlap_checkpoint_sha256':prior_report['checkpoint_sha256'],'conversion':conversion,'test_used':False}
    save(args.output/'result.json',report);print(json.dumps(report),flush=True)
    if report['status']!='PASS':raise RuntimeError(report)

def pilot(args,device):
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);started=time.time();lock,model,refiner,overlap,prior_report,conversion=build(args,device);cfg=lock['pilot']
    data=args.root/'data/remote_variants/RAM-W600';train=DatasetV3(data,args.prior_root,'train',args.arm in 'BD');val=DatasetV3(data,args.prior_root,'val',args.arm in 'BD')
    generator=torch.Generator().manual_seed(cfg['seed']);train_loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=2,generator=generator);val_loader=DataLoader(val,batch_size=1,num_workers=2)
    teacher=copy.deepcopy(model).eval().requires_grad_(False);params=list(model.parameters())+(list(refiner.parameters()) if refiner else [])
    groups=[{'params':model.parameters(),'lr':cfg['backbone_lr']}]
    if refiner:groups.append({'params':refiner.parameters(),'lr':cfg['output_lr']})
    opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);pairs=top_training_pairs(data,15);best=-math.inf;history=[];updates=0
    for epoch in range(1,cfg['max_epochs']+1):
        factor=(1-(epoch-1)/cfg['max_epochs'])**.9
        for group,base_lr in zip(opt.param_groups,[cfg['backbone_lr']]+([cfg['output_lr']] if refiner else [])):group['lr']=base_lr*factor
        model.train();refiner.train() if refiner else None;iterator=iter(train_loader);tot={k:0. for k in ['base','separation','overlap','preserve']};routing=[]
        for step in range(cfg['updates_per_epoch']):
            resource_guard(args.output,started);b=next(iterator);image,seam,target,valid=unpack(b,device);opt.zero_grad(set_to_none=True)
            z,o=forward(model,refiner,overlap,image,seam,args.arm);base=dice_bce_logits(z,target)
            with torch.no_grad():native=teacher(torch.cat((image,torch.zeros_like(image)),1)).sigmoid()
            terms,states=local_supervision(z,target,torch.ones_like(target,dtype=torch.bool),target.any(1,keepdim=True),valid,valid,seam,native,enable_seam=args.arm in 'BD',enable_overlap=args.arm in 'CD')
            if args.arm=='A':base.backward();rows=[]
            else:
                aux=[]
                for name,weight in [('separation',.035),('overlap',.035),('preserve',.10)]:
                    if name=='separation' and args.arm not in 'BD' or name=='overlap' and args.arm not in 'CD':continue
                    aux.append((name,terms[name],weight*states['attenuation'][name]))
                gradients,rows=route_parameter_gradients_v3(params,base,aux,max_aux_ratio=.15,max_merged_aux_ratio=.15);assign_gradients_v3(params,gradients)
            torch.nn.utils.clip_grad_norm_(params,5.,error_if_nonfinite=True);opt.step();updates+=1
            tot['base']+=float(base.detach());routing.append(rows[-1] if rows else {'name':'base_only'})
            for k in terms:tot[k]+=float(terms[k].detach())
        report=collect(model,refiner,overlap,val_loader,device,args.arm,pairs);value=report['official']['overall_instance_metrics']['dsc']
        row={'epoch':epoch,'updates':updates,'train':{k:v/cfg['updates_per_epoch'] for k,v in tot.items()},'validation':report['official'],'routing_last':routing[-1]}
        history.append(row);save(args.output/'history.json',history);print(json.dumps({'arm':args.arm,'epoch':epoch,'overall_dsc':value}),flush=True)
        if value>best:
            best=value;bestsha=save_best(args.output/'best.pth',model,refiner,args.arm,epoch,value,sha(args.native),prior_report,args.protocol);save(args.output/'selected_validation.json',report)
    result={'status':'COMPLETE','arm':args.arm,'epochs':cfg['max_epochs'],'updates':updates,'best_checkpoint':str(args.output/'best.pth'),'best_checkpoint_sha256':bestsha,
            'selected_epoch':torch.load(args.output/'best.pth',map_location='cpu',weights_only=False)['epoch'],'selection_value':best,'source_sha256':sha(args.native),
            'overlap_checkpoint_sha256':prior_report['checkpoint_sha256'] if prior_report else None,'conversion':conversion,'protocol_sha256':sha(args.protocol),'seconds':time.time()-started,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'test_used':False,'prior_adaptation_cost':prior_report['updates'] if prior_report else 0}
    result['runner_sha256']=sha(Path(__file__))
    save(args.output/'result.json',result)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','smoke','pilot'],required=True);p.add_argument('--arm',choices=list('ABCD'),default='A')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--overlap-checkpoint',type=Path);p.add_argument('--overlap-acceptance',type=Path);a=p.parse_args()
    random.seed(3512);np.random.seed(3512);torch.manual_seed(3512);torch.cuda.manual_seed_all(3512);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda')
    {'baseline':baseline,'smoke':smoke,'pilot':pilot}[a.mode](a,device)
if __name__=='__main__':main()
