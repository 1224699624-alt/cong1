"""Isolated no-update timing and gradient connectivity for F0/F1 on RAM."""
import argparse,copy,json,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.parent.mkdir(parents=True,exist_ok=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    torch.set_num_threads(64);device=torch.device('cuda');rows=[];start=time.monotonic()
    for factor in [0,1]:
        legacy.seed_all(4421)
        _,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
        ck=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);refiner.load_state_dict(ck['refiner']);legacy.set_joint_trainability(model,True)
        extra=list(model.backbone.decoder.transpconvs[2].parameters())+list(model.backbone.decoder.stages[2].parameters())
        for x in extra:x.requires_grad_(bool(factor))
        old_ref=list(refiner.parameters());old_dec=[x for n,x in model.named_parameters() if x.requires_grad and not n.startswith(('backbone.decoder.stages.2.','backbone.decoder.transpconvs.2.'))]
        opt=torch.optim.AdamW([{'params':old_ref,'lr':lock['refiner_lr']},{'params':old_dec,'lr':lock['decoder_lr']}],weight_decay=1e-4)
        opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        if factor:opt.add_param_group({'params':extra,'lr':lock['decoder_lr'],'weight_decay':1e-4})
        scaler=torch.cuda.amp.GradScaler();state=copy.deepcopy(ck['scaler']);state['scale']=1024.;scaler.load_state_dict(state)
        model.eval();refiner.train()
        for module in [model.backbone.decoder.stages[i] for i in [2,3,4]]+[model.backbone.decoder.transpconvs[i] for i in [2,3,4]]+[model.backbone.decoder.seg_layers[4]]:module.train()
        ds=legacy.NativeWristDataset(Path(lock['dataset']),'train',True);loader=iter(DataLoader(ds,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(4421)))
        torch.cuda.reset_peak_memory_stats();elapsed=[];batchhash=[];finite=True;extra_grad=[]
        for step in range(1,lock['profile_steps']+1):
            tick=time.monotonic();batch=next(loader);h=__import__('hashlib').sha256(b''.join(batch[k].numpy().tobytes() for k in ('image','mask','valid'))).hexdigest();batchhash.append(h)
            image,target,valid=[batch[k].to(device) for k in ('image','mask','valid')];opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():loss=legacy.loss_for_batch(model,refiner,generator,image,target,valid,False)[0]
            scaler.scale(loss).backward();scaler.unscale_(opt)
            finite=finite and bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for group in opt.param_groups for p in group['params'])
            extra_grad.append(sum(float(p.grad.float().abs().sum()) for p in extra if p.grad is not None))
            scaler.update()
            torch.cuda.synchronize();elapsed.append(time.monotonic()-tick)
            print(json.dumps({'factor':factor,'step':step,'seconds':elapsed[-1],'loss':float(loss.detach())}),flush=True)
        rows.append({'F':factor,'seconds':elapsed,'median_full_step_seconds':float(np.median(elapsed[1:] if len(elapsed)>1 else elapsed)),'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'finite':finite,'extra_stage2_parameter_count':sum(p.numel() for p in extra),'extra_stage2_grad_abs_sum':extra_grad,'batch_hashes':batchhash,'old_adam_steps':sorted(set(int(s['step']) for s in opt.state.values() if s)),'scaler_initial':state['scale'],'optimizer_updates':0})
        del model,refiner,generator,opt,scaler,ds,loader;torch.cuda.empty_cache()
    assert rows[0]['batch_hashes']==rows[1]['batch_hashes']
    a.out.write_text(json.dumps({'status':'COMPLETE','rows':rows,'seconds':time.monotonic()-start,'test_used':False},indent=2))

if __name__=='__main__':main()
