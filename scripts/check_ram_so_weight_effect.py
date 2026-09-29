"""Isolated train-only gradient check of H0 weighting; no optimizer step."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from run_ram_so_o_pair import model_load,load_case,bce_loss,sha
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.lock.read_text());torch.set_num_threads(8);start=time.monotonic()
    net,_=model_load(cfg['overlap'],'cuda');net.eval()
    ds=NativeWristDataset(Path(cfg['dataset']),'train',False)
    fit=json.loads(Path(cfg['split_manifest']).read_text())['groups']['probe_fit']
    lookup={p.stem:i for i,p in enumerate(ds.mask_files)};named=list(net.named_parameters());params=[p for _,p in named];rows=[]
    # Fixed train-fit cases, no holdout or val-based selection.
    for case in fit[:8]:
        b,x,t,v,h,_,_,_=load_case(ds,lookup[case],cfg['h0_root'])
        x,t,v,h=[z.to('cuda') for z in (x,t,v,h)]
        z=net(x);ref=bce_loss(z,t,v,h,0);hard=bce_loss(z,t,v,h,cfg['beta'])
        gr=torch.autograd.grad(ref,params,retain_graph=True);gh=torch.autograd.grad(hard,params,retain_graph=True)
        vr=torch.cat([g.flatten() for g in gr]);vh=torch.cat([g.flatten() for g in gh])
        mask=b['mask'][None].to('cuda');valid=b['valid'][None].to('cuda').float()
        member_count=mask.sum(1,keepdim=True)
        single=F.interpolate(valid*(member_count==1).float(),(192,192),mode='area')
        bg=F.interpolate(valid*(member_count==0).float(),(192,192),mode='area')
        elem=F.binary_cross_entropy_with_logits(z,t,reduction='none');w=1+cfg['beta']*h;den=(v*w).sum()
        components={'H_cells':v*(h>0)*w*elem,'ordinary_overlap_cells':v*(h==0)*(t>0)*w*elem,
                    'single_bone_mass':single*w*elem,'background_mass':bg*w*elem}
        rg={}
        for key,element in components.items():
            loss=element.sum()/den
            gs=torch.autograd.grad(loss,params,retain_graph=True)
            layer={}
            for (n,_),g in zip(named,gs):layer[n.split('.')[0]]=layer.get(n.split('.')[0],0.)+float(g.square().sum())
            rg[key]={'loss_contribution':float(loss),'generator_layer_grad_l2':{n:value**.5 for n,value in layer.items()}}
        rows.append({'case':case,'ref_bce':float(ref),'hard_bce':float(hard),
            'beta':cfg['beta'],'extra_mass_fraction':float((v*cfg['beta']*h).sum()/v.sum()),
            'weighted_vs_ref_grad_cosine':float(F.cosine_similarity(vr[None],vh[None])),
            'weighted_minus_ref_grad_relative_l2':float((vh-vr).norm()/vr.norm().clamp_min(1e-12)),
            'component_diagnostics':rg,
            'component_note':'H/nonH masks partition grid support; single/bg area masses may share soft cells and are additional descriptive decompositions, not summed twice'})
    result={'status':'COMPLETE','phase':'isolated_initial_O_gradient_check_not_history',
        'rows':rows,'optimizer_steps':0,'test_used':False,'seconds':time.monotonic()-start,
        'C0_checkpoint_sha256_unchanged':sha(cfg['c0'])}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'COMPLETE','cases':len(rows),'seconds':result['seconds']}),flush=True)

if __name__=='__main__':main()
