"""Zero-update, fixed training-case counterfactuals; never performance evidence."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy
from whitebox_ram_diagnosis import stats, rng_get, rng_set, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();lock=json.loads(a.lock.read_text())
    if a.out.exists():raise FileExistsError(a.out)
    legacy.seed_all(lock['seed']);device=torch.device('cuda')
    _,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    legacy.set_joint_trainability(model,True);model.eval();refiner.eval()
    train=legacy.NativeWristDataset(Path(lock['dataset']),'train',True)
    loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(lock['seed']))
    named=[('model.'+n,t) for n,t in model.named_parameters() if t.requires_grad]+[('refiner.'+n,t) for n,t in refiner.named_parameters()]
    params=[t for _,t in named];results=[]
    original=legacy.completion_loss
    for index,batch in enumerate(loader):
        if index>=3:break
        image,target,valid=[batch[k].to(device) for k in ['image','mask','valid']]
        changes=[];parts=[];components={}
        def observe(z,t,local,delta,bw):
            value,info=original(z,t,local,delta,bw)
            bce=F.binary_cross_entropy_with_logits(z.float(),t.float(),reduction='none');weight=1+3*local
            v=valid.expand_as(t).float()
            old=(bce*weight).sum()/weight.sum();masked=(bce*weight*v).sum()/(weight*v).sum().clamp_min(1)
            step=len(changes)+1; factor=step/3
            probability=z.float().sigmoid();axes=(1,2,3)
            dice=1-((2*(probability*t).sum(axes)+1)/(probability.sum(axes)+t.sum(axes)+1)).mean()
            conservation=(delta.abs()*(1-local)).mean()
            tb=(F.max_pool2d(t.float(),3,1,1)+F.max_pool2d(-t.float(),3,1,1)).clamp(0,1)
            pb=(F.max_pool2d(probability,3,1,1)+F.max_pool2d(-probability,3,1,1)).clamp(0,1)
            band=F.max_pool2d(tb,7,1,3)
            boundary=((pb-tb).abs()*band).sum()/band.sum().clamp_min(1)
            reconstructed=.55*old+.45*dice+.02*conservation+bw*boundary
            assert float((reconstructed-value).abs())==0
            for key,term,coefficient in [('bce',old,.55),('dice',dice,.45),('conservation',conservation,.02),('boundary',boundary,bw)]:
                components[f'step{step}_{key}']=(term,coefficient*factor)
            changes.append(.55*(masked-old)*factor)
            parts.append({'full_denominator':float(weight.sum()),'valid_denominator':float((weight*v).sum()),
                          'full_bce':float(old),'valid_bce':float(masked)})
            return value,info
        legacy.completion_loss=observe
        s=rng_get()
        with torch.cuda.amp.autocast():loss,_,_,_,_,_=legacy.loss_for_batch(model,refiner,generator,image,target,valid,False)
        # Scaling before backward protects FP16 intermediates; these are functional, not accumulated gradients.
        scale=65536.
        raw=torch.autograd.grad(loss*scale,params,retain_graph=True,allow_unused=True)
        component_gradients={}
        for key,(term,coefficient) in components.items():
            gradients=torch.autograd.grad(term*scale,params,retain_graph=True,allow_unused=True)
            groups={}
            for (name,_),g,total_g in zip(named,gradients,raw):
                if g is None:continue
                group='refiner' if name.startswith('refiner') else 'decoder'
                row=groups.setdefault(group,{'raw_norm2':0.,'weighted_dot_total':0.,'total_norm2':0.})
                g=g.float()/scale;row['raw_norm2']+=float(g.square().sum())
                if total_g is not None:
                    total_g=total_g.float()/scale
                    row['weighted_dot_total']+=float((g*coefficient*total_g).sum());row['total_norm2']+=float(total_g.square().sum())
            for row in groups.values():
                row['raw_norm']=row['raw_norm2']**.5;row['weighted_norm']=row['raw_norm']*coefficient
                row['cosine_to_total']=row['weighted_dot_total']/max(row['weighted_norm']*row['total_norm2']**.5,1e-30)
            component_gradients[key]={'loss':float(term),'coefficient':coefficient,'groups':groups}
        masked=torch.autograd.grad((loss+sum(changes))*scale,params,allow_unused=True)
        gs={}
        for (name,t),g,h in zip(named,raw,masked):
            if g is None or h is None:continue
            group='refiner' if name.startswith('refiner') else 'decoder'
            row=gs.setdefault(group,{'base2':0.,'difference2':0.,'dot':0.,'variant2':0.})
            g=g.float()/scale;h=h.float()/scale
            row['base2']+=float(g.square().sum());row['difference2']+=float((h-g).square().sum())
            row['dot']+=float((g*h).sum());row['variant2']+=float(h.square().sum())
        for row in gs.values():
            row['relative_gradient_difference']=(row['difference2']/max(row['base2'],1e-30))**.5
            row['cosine']=row['dot']/max((row['base2']*row['variant2'])**.5,1e-30)
        legacy.completion_loss=original;rng_set(s)
        with torch.no_grad():
            base=model(torch.cat([image,torch.zeros_like(image)],1));zero=torch.zeros_like(image)
            normal,_=legacy.iterative_refine_o(image,base,refiner,2,zero)
            contrasts={}
            for mode in ['zero_member_context','roll_member_context_16px']:
                def alter(m,inputs):
                    x=inputs[0].clone()
                    x[:,1:4]=0 if mode.startswith('zero') else torch.roll(x[:,1:4],16,dims=-1)
                    return (x,)
                hook=refiner.enc1.register_forward_pre_hook(alter)
                altered,_=legacy.iterative_refine_o(image,base,refiner,2,zero)
                hook.remove()
                contrasts[mode]={'logit_abs_difference':stats(altered[-1]-normal[-1]),
                    'mask_changed_pixels':int(((altered[-1]>=0)!=(normal[-1]>=0)).sum())}
            overlap=target.sum(1,keepdim=True)>=2
            point=torch.nonzero(overlap)[0].tolist() if bool(overlap.any()) else None
            pixel=None
            if point:
                _,_,y,x=point;prob=normal[-1][0,:,y,x].sigmoid();truth=target[0,:,y,x]
                pixel={'yx':[y,x],'target14':truth.tolist(),'probability14':prob.tolist(),
                       'bce_logit_gradient_sign14':torch.sign(prob-truth).tolist()}
        assert all(t.grad is None for t in params)
        results.append({'case':batch['case'][0],'loss':float(loss),'completion_BCE_valid_only_counterfactual_delta':float(sum(changes)),
                        'loss_denominators':parts,'scaled_gradient_comparison':gs,'context_sensitivity':contrasts,
                        'scaled_completion_component_gradients':component_gradients,
                        'true_overlap_pixel':pixel,'optimizer_updates':0})
    write(a.out,{'mode':'CURRENT_INITIAL_STATE_ZERO_UPDATE_DIAGNOSTIC','cases':results,
                'counterfactual_scope':'Only completion BCE excludes padding; all other terms unchanged. Not a full valid-masked method.',
                'sensitivity_caveat':'Out-of-distribution feature perturbations establish dependence only; not matched-training benefit.',
                'test_used':False,'optimizer_updates':0})


if __name__=='__main__':main()
