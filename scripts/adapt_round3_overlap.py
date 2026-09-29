"""Bounded RAM image-only overlap adaptation with pre-registered gate."""
from __future__ import annotations
import argparse,hashlib,json,random,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from structural_prior_v3 import ImageOnlyOverlapPrior
from r351_dual_prior import ImageOverlapPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from r351_data import AlignedRamDataset

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def low_batch(batch,device):
    image=batch['image'].to(device);y=batch['mask'].to(device)
    valid=batch['valid'].float().to(device)
    if y.shape[1]!=14 or not torch.isfinite(y).all() or not ((y==0)|(y==1)).all():
        raise ValueError('expected complete observed 14 binary member labels, unknown values require explicit mask')
    truth=(y.sum(1,keepdim=True)>=2).float()
    return F.interpolate(image,(192,192),mode='bilinear',align_corners=False),F.interpolate(truth,(192,192),mode='area'),F.interpolate(valid,(192,192),mode='nearest'),truth,valid

def evaluate(model,loader,device,constant,legacy=False):
    model.eval();rows=[];totals=dict(tp=0,fp=0,fn=0,valid=0,predicted=0)
    with torch.no_grad():
        for b in loader:
            x,y,v,truth,valid=low_batch(b,device)
            if legacy:
                s=F.interpolate(b['seam'].to(device),(192,192),mode='bilinear',align_corners=False)
                z=model(x,s)
            else:z=model(x)
            loss=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*v).sum()/v.sum().clamp_min(1)
            q=z.sigmoid();pos=y*v;neg=(1-y)*v
            const=-(y*np.log(constant)+(1-y)*np.log(1-constant))
            native=F.interpolate(q,truth.shape[-2:],mode='bilinear',align_corners=False)
            pr=(native>=.5)&valid.bool();gt=truth.bool()&valid.bool()
            tp=int((pr&gt).sum());fp=int((pr&~gt).sum());fn=int((~pr&gt).sum());n=int(valid.sum())
            row={'case':b['case'][0],'bce':float(loss),'constant_bce':float((const*v).sum()/v.sum().clamp_min(1)),
                 'positive_probability':float((q*pos).sum()/pos.sum().clamp_min(1)),
                 'negative_probability':float((q*neg).sum()/neg.sum().clamp_min(1)),
                 'positive_mass':float(pos.sum()),'valid_low_mass':float(v.sum()),'probability_min':float(q.min()),
                 'probability_max':float(q.max()),'probability_std':float(q.std()),'tp':tp,'fp':fp,'fn':fn,
                 'empty_true_overlap':not bool(gt.any()),'finite':bool(torch.isfinite(q).all()),
                 'valid_native_pixels':n,'predicted_positive_pixels':int(pr.sum())}
            rows.append(row)
            for k,val in [('tp',tp),('fp',fp),('fn',fn),('valid',n),('predicted',int(pr.sum()))]:totals[k]+=val
    summary={k:float(np.mean([r[k] for r in rows])) for k in ['bce','constant_bce','positive_probability','negative_probability','probability_std']}
    summary.update({'precision':totals['tp']/max(1,totals['tp']+totals['fp']),
                    'recall':totals['tp']/max(1,totals['tp']+totals['fn']),
                    'predicted_positive_fraction':totals['predicted']/max(1,totals['valid']),
                    'finite_cases':sum(r['finite'] for r in rows),'cases':len(rows),
                    'empty_true_cases':sum(r['empty_true_overlap'] for r in rows),'threshold':.5,
                    'distance_metrics':'NOT_APPLICABLE probability generator evaluation, no distance exclusion',
                    'unknown_pixels':0,'member_observation':'complete binary14 assumption verified by labels; padding excluded'})
    return {'summary':summary,'per_case':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.protocol.read_text());cfg=lock['prior_adaptation']
    from round3_common import require_manifest
    require_manifest(a.protocol)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    assert sha(a.source)==cfg['source_sha256']
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.set_num_threads(2);device=torch.device('cuda');start=time.time()
    data=a.root/'data/remote_variants/RAM-W600'
    sets={s:NativeWristDataset(data,s,False) for s in ['train','val']}
    assert [len(sets[s]) for s in ['train','val']]==[425,69]
    train=DataLoader(sets['train'],batch_size=1,shuffle=True,num_workers=2,
                     generator=torch.Generator().manual_seed(cfg['seed']))
    val=DataLoader(sets['val'],batch_size=1,shuffle=False,num_workers=2)
    # Constant null model is fixed using training labels only, before val scores.
    positive=total=0
    for path in sets['train'].mask_files:
        mask=np.load(path,mmap_mode='r');positive+=int((mask.sum(0)>=2).sum());total+=mask.shape[-1]*mask.shape[-2]
    constant=np.clip(positive/total,1e-6,1-1e-6)
    source=torch.load(a.source,map_location='cpu',weights_only=False)
    model=ImageOnlyOverlapPrior().to(device);init=model.initialize_from_legacy(source['model'])
    legacy=ImageOverlapPrior().to(device);legacy.load_state_dict(source['model'],strict=True)
    legacy_loader=DataLoader(AlignedRamDataset(data,a.root/'outputs/priors/r323_ram_native_r317_single_seed','val'),batch_size=1,num_workers=2)
    legacy_result=evaluate(legacy,legacy_loader,device,constant,True)
    save(a.output/'legacy_validation.json',legacy_result);del legacy
    initial=evaluate(model,val,device,constant);save(a.output/'initial_image_only_validation.json',initial)
    save(a.output/'run_manifest.json',{'status':'RUNNING','protocol_sha256':sha(a.protocol),'source':str(a.source),
        'source_sha256':sha(a.source),'mapping':init,'config':cfg,'constant_training_prevalence':float(constant),
        'code_sha256':sha(Path(__file__)),'module_sha256':sha(Path(__file__).parent/'structural_prior_v3.py'),
        'test_used':False,'overlap_input_dependencies':['image'],'nnunet_updated':False})
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);best=float('inf');updates=0
    for epoch in range(1,cfg['max_epochs']+1):
        model.train();losses=[]
        for b in train:
            if time.time()-start>lock['resources']['max_wall_seconds_prior']:raise TimeoutError('prior wall budget')
            if shutil.disk_usage(a.output).free<4*1024**3:raise RuntimeError('disk floor')
            x,y,v,_,_=low_batch(b,device);opt.zero_grad(set_to_none=True);z=model(x)
            loss=(F.binary_cross_entropy_with_logits(z,y,reduction='none')*v).sum()/v.sum().clamp_min(1)
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();updates+=1;losses.append(float(loss.detach()))
        report=evaluate(model,val,device,constant)
        row={'epoch':epoch,'updates':updates,'train_bce':float(np.mean(losses)),**report['summary'],'elapsed_seconds':time.time()-start}
        with (a.output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if row['bce']<best:
            best=row['bce'];torch.save({'model':model.state_dict(),'epoch':epoch,'updates':updates,
                'version':'image_only_overlap_v3','input_dependencies':['image'],'source_sha256':sha(a.source),
                'protocol_sha256':sha(a.protocol),'metrics':row},a.output/'best.pth')
            save(a.output/'selected_validation.json',report)
        print(json.dumps(row),flush=True)
    selected=torch.load(a.output/'best.pth',map_location='cpu',weights_only=False)
    r=json.loads((a.output/'selected_validation.json').read_text());s=r['summary']
    gates={'full_val_finite':s['cases']==69 and s['finite_cases']==69,
           'beats_constant':s['bce']<s['constant_bce'],
           'positive_exceeds_negative':s['positive_probability']>s['negative_probability'],
           'noncollapsed':0<s['predicted_positive_fraction']<1 and s['probability_std']>0,
           'positive_precision_recall':s['precision']>0 and s['recall']>0}
    result={'status':'ACCEPTED_FOR_BOUNDED_PILOT' if all(gates.values()) else 'TRAINED_NOT_ACCEPTED',
            'accepted':all(gates.values()),'gates':gates,'checkpoint':str(a.output/'best.pth'),
            'checkpoint_sha256':sha(a.output/'best.pth'),'selected_epoch':selected['epoch'],
            'input_dependencies':['image'],'version':'image_only_overlap_v3','updates':updates,
            'max_updates':cfg['max_updates'],'selected':s,'legacy':legacy_result['summary'],
            'initial':initial['summary'],'protocol_sha256':sha(a.protocol),'seconds':time.time()-start,
            'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'test_used':False,'target_source':'RAM observed14 member >=2',
            'not_claimed':['calibration','segmentation improvement','TSRS native overlap supervision']}
    assert updates==cfg['max_updates'];save(a.output/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
