"""One real gradient, two isolated optimizer steps; no performance claim."""
import argparse, copy, hashlib, json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy
from r332_adam_state_port import restore_r332_adam
from whitebox_ram_diagnosis import grouping, write


def named(model,refiner):
    return [('model.'+n,p) for n,p in model.named_parameters() if p.requires_grad]+[('refiner.'+n,p) for n,p in refiner.named_parameters() if p.requires_grad]


def optimizer(model,refiner,lr):
    return torch.optim.AdamW([{'params':list(refiner.parameters()),'lr':lr},
                              {'params':[p for p in model.parameters() if p.requires_grad],'lr':7e-6}],weight_decay=1e-4)


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists()
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    legacy.seed_all(3321);device=torch.device('cuda')
    _,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    legacy.set_joint_trainability(model,True);model.eval();refiner.train()
    for m in [model.backbone.decoder.stages[3],model.backbone.decoder.stages[4],model.backbone.decoder.transpconvs[3],model.backbone.decoder.transpconvs[4],model.backbone.decoder.seg_layers[4]]:m.train()
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'train',True)
    batch=next(iter(DataLoader(ds,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(3321))))
    digest=hashlib.sha256(b''.join(batch[k].numpy().tobytes() for k in ['image','mask','valid'])).hexdigest()
    reference=json.loads(Path(lock['reference_trace']).read_text().splitlines()[0]);assert digest==reference['batch_sha256']
    image,target,valid=[batch[k].to(device) for k in ['image','mask','valid']]
    origin=optimizer(model,refiner,lock['refiner_lr_override']);scaler=torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():loss=legacy.loss_for_batch(model,refiner,generator,image,target,valid,False)[0]
    scaler.scale(loss).backward();scaler.unscale_(origin)
    torch.nn.utils.clip_grad_norm_(refiner.parameters(),12.);torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.)
    source=named(model,refiner);source_before={n:p.detach().clone() for n,p in source}
    grad={n:p.grad.detach().clone() if p.grad is not None else None for n,p in source}
    assert all(g is None or bool(torch.isfinite(g).all()) for g in grad.values())
    grad_sha=hashlib.sha256(b''.join(g.cpu().numpy().tobytes() for g in grad.values() if g is not None)).hexdigest()
    arms={};updates={}
    for arm in ['fresh','restored']:
        m=copy.deepcopy(model);r=copy.deepcopy(refiner);opt=optimizer(m,r,lock['refiner_lr_override'])
        mapping=restore_r332_adam(opt,m,r,lock['restore_adam_from']) if arm=='restored' else None
        current=named(m,r);before={n:p.detach().clone() for n,p in current}
        for n,p in current:p.grad=None if grad[n] is None else grad[n].clone()
        actual_sha=hashlib.sha256(b''.join(p.grad.cpu().numpy().tobytes() for n,p in current if p.grad is not None)).hexdigest();assert actual_sha==grad_sha
        opt.step()  # The only step on this isolated cloned model.
        groups={};delta={}
        for n,p in current:
            d=p.detach()-before[n];delta[n]=d.cpu();g=grouping(n)
            row=groups.setdefault(g,{'update2':0.,'changed':0,'elements':0,'steps':[]})
            row['update2']+=float(d.float().square().sum());row['changed']+=int((d!=0).sum());row['elements']+=d.numel()
            if opt.state[p]:row['steps'].append(float(opt.state[p]['step']))
        for row in groups.values():
            row['update_norm']=row.pop('update2')**.5;row['changed_fraction']=row.pop('changed')/row['elements']
            row['step_range']=[min(row['steps']),max(row['steps'])];del row['steps']
        arms[arm]={'gradient_sha256':actual_sha,'groups':groups,'mapping_passed':mapping['status'] if mapping else None};updates[arm]=delta
        del m,r,opt,before
    assert all(torch.equal(p.detach(),source_before[n]) for n,p in source)
    differences={}
    for n,d in updates['fresh'].items():
        g=grouping(n);row=differences.setdefault(g,{'difference2':0.,'dot':0.,'fresh2':0.,'restored2':0.});other=updates['restored'][n]
        row['difference2']+=float((d-other).square().sum());row['dot']+=float((d*other).sum());row['fresh2']+=float(d.square().sum());row['restored2']+=float(other.square().sum())
    for row in differences.values():
        row['update_cosine']=row['dot']/max((row['fresh2']*row['restored2'])**.5,1e-30)
        row['restored_to_fresh_norm_ratio']=(row['restored2']/max(row['fresh2'],1e-30))**.5
    write(a.out,{'status':'PASS','mode':'CURRENT_STATE_SHARED_GRADIENT_SHADOW_STEPS','case':batch['case'][0],
                'batch_sha256':digest,'loss':float(loss.detach()),'gradient_sha256':grad_sha,
                'source_model_optimizer_steps':0,'shadow_optimizer_steps':2,'arms':arms,'update_comparison':differences,
                'test_used':False,'performance_claim':False,'lock_sha256':legacy.sha(a.lock)})


if __name__=='__main__':main()
