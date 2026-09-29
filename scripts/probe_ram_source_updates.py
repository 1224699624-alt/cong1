"""Independent one-step forks on fixed RAM training cases; not a val experiment."""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import run_v31_ram_mature as legacy
from whitebox_ram_diagnosis import grouping,grad_summary,write


def roi_stat(value,mask):
    x=value.detach().float()[mask]
    if not x.numel():return {'n':0,'mean':None,'l1':0.,'l2':0.,'negative_fraction':None,'positive_fraction':None,'zero_fraction':None}
    return {'n':x.numel(),'mean':float(x.mean()),'l1':float(x.abs().sum()),'l2':float(x.norm()),
            'negative_fraction':float((x<0).float().mean()),'positive_fraction':float((x>0).float().mean()),'zero_fraction':float((x==0).float().mean())}


def gradients(value,targets):
    # Per-term functional gradients use a documented finite scale. The real
    # optimizer step retains its checkpoint GradScaler and normal backward.
    for scale in [1024.,64.,1.]:
        raw=torch.autograd.grad(value*scale,targets,retain_graph=True,allow_unused=True)
        if all(g is None or bool(torch.isfinite(g).all()) for g in raw):
            return [None if g is None else g.detach().float()/scale for g in raw],scale
    raise RuntimeError('Nonfinite functional gradients')


@torch.no_grad()
def predict(model,refiner,image):
    model.eval();refiner.eval()
    source=model(torch.cat([image,torch.zeros_like(image)],1))
    states,_=legacy.iterative_refine_o(image,source,refiner,2,torch.zeros_like(image))
    return source.detach(),states[-1].detach()


def train_modes(model,refiner):
    model.eval();refiner.train()
    for m in [model.backbone.decoder.stages[3],model.backbone.decoder.stages[4],model.backbone.decoder.transpconvs[3],model.backbone.decoder.transpconvs[4],model.backbone.decoder.seg_layers[4]]:m.train()


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    start=time.monotonic();device=torch.device('cuda');torch.set_num_threads(lock.get('cpu_threads',2));legacy.seed_all(lock['seed'])
    _,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    payload=torch.load(lock['reference_checkpoint'],map_location='cpu',weights_only=False)
    legacy.set_joint_trainability(model,True)
    named=[('model.'+n,p) for n,p in model.named_parameters() if p.requires_grad]+[('refiner.'+n,p) for n,p in refiner.named_parameters() if p.requires_grad]
    params=[p for _,p in named];decoder=[p for p in model.parameters() if p.requires_grad]
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'train',False)
    manifest=[{'case':p.stem,'mask_sha':legacy.sha(p),'image_sha':legacy.sha(Path(lock['dataset'])/'BoneSegmentation/images'/(p.stem+'.bmp'))} for p in ds.mask_files]
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    assert len(ds)==425 and digest==lock['data']['train']['content_sha256']
    order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(lock['seed']))[:lock['cases']].tolist()
    active=[False];box={};terms={};counter=[0]
    def source_hook(m,inputs,output):
        if active[0]:box['source']=output
    hook=model.register_forward_hook(source_hook)
    original_iteration=legacy.iterative_refine_o
    def iteration(*args,**kwargs):
        states,deltas=original_iteration(*args,**kwargs)
        if active[0]:box['states']=states;box['deltas']=deltas
        return states,deltas
    legacy.iterative_refine_o=iteration
    original_completion=legacy.completion_loss;original_surface=legacy.soft_surface_nsd_loss;original_volume=legacy.volume_losses
    def completion(z,target,local,delta,bw):
        loss,stats=original_completion(z,target,local,delta,bw);counter[0]+=1;k=counter[0]
        terms[f'step{k}_completion']=(loss,k/3);box.setdefault('supports',[]).append({'selected_positive_pixels':int(target.sum()),'local_pixels':int(local.sum()),'weighted_BCE_denominator':float((1+3*local).sum())})
        return loss,stats
    def surface(prob,target):
        loss=original_surface(prob,target);terms[f'step{counter[0]}_surface']=(loss,.03*counter[0]/3);return loss
    def volume(prob,target):
        iv,pv=original_volume(prob,target);terms[f'step{counter[0]}_instance_volume']=(iv,.005*counter[0]/3);terms[f'step{counter[0]}_pair_volume']=(pv,.01*counter[0]/3);return iv,pv
    legacy.completion_loss=completion;legacy.soft_surface_nsd_loss=surface;legacy.volume_losses=volume
    rows=[]
    for probe,case_index in enumerate(order,1):
        if time.monotonic()-start>lock['max_seconds']:raise TimeoutError('budget')
        legacy.seed_all(lock['seed']+probe)
        model.load_state_dict(payload['model'],strict=True);refiner.load_state_dict(payload['refiner'],strict=True)
        opt=torch.optim.AdamW([{'params':refiner.parameters(),'lr':lock['refiner_lr']},{'params':decoder,'lr':lock['decoder_lr']}],weight_decay=1e-4)
        # deepcopy is essential: scalar Adam steps must not alias the saved
        # CPU payload across independent forks.
        opt.load_state_dict(copy.deepcopy(payload['optimizer']))
        assert [g['lr'] for g in opt.param_groups]==[lock['refiner_lr'],lock['decoder_lr']]
        assert all(float(s['step'])==100 for s in opt.state.values())
        scaler=torch.cuda.amp.GradScaler();scaler.load_state_dict(copy.deepcopy(payload['scaler']))
        sample=ds[case_index];image=sample['image'][None].to(device);target=sample['mask'][None].to(device);valid=sample['valid'][None].to(device).bool()
        src_before,final_before=predict(model,refiner,image)
        truth=target>.5;pred=final_before.sigmoid()>=.5;true_o=truth.sum(1,keepdim=True)>=2
        fn=truth&~pred&valid;fp=~truth&pred&valid
        hard=fn&true_o&(src_before < -5.00001)
        rois={'hard_overlap_FN':hard,'other_overlap_FN':fn&true_o&~hard,'nonoverlap_FN':fn&~true_o,
              'correct_true_member':truth&pred&valid,'member_FP':fp}
        rois['near_boundary_GT_negative']=(~truth)&valid&(final_before.abs()<.25)
        active[0]=True;box.clear();terms.clear();counter[0]=0;train_modes(model,refiner);opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():loss,parts,stats,o,indices,supervision=legacy.loss_for_batch(model,refiner,generator,image,target,valid,False)
        for name,w in [('global',.1),('monotonic',.05),('contraction',.01)]:terms[name]=(parts[name],w)
        source=box['source'];selected=torch.zeros_like(truth);selected[:,indices]=True
        auxiliary=None
        if lock.get('source_aux_enabled'):
            from r332_hard_source_supervision import hard_source_loss
            extra,auxiliary=hard_source_loss(source,target,valid,indices)
            terms['unselected_hard_source']=(extra,1.)
            loss=loss+extra
        guard_meta=None
        if lock.get('negative_guard_enabled'):
            from r332_hard_source_supervision import local_negative_guard_loss
            guard,guard_meta=local_negative_guard_loss(source,target,valid,final_before,auxiliary['denominator'])
            terms['local_negative_guard']=(guard,1.)
            loss=loss+guard
        rois['hard_selected']=hard&selected;rois['hard_unselected']=hard&~selected
        hard_objective=F.softplus(-source.float())[hard].mean() if bool(hard.any()) else source.sum()*0
        terms['hard_source_BCE_DIAGNOSTIC_ONLY']=(hard_objective,0.)
        terms['total']=(loss,1.)
        grad_records={};parameter_grads={};initial_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone()
        for name,(value,w) in terms.items():
            grads,scale=gradients(value,[source]+params);gz=grads[0]
            grad_records[name]={'raw_loss':float(value.detach()),'weight':w,'functional_scale':scale,
                'source_logit_gradients':{region:{'raw':roi_stat(gz,mask),'weighted':roi_stat(gz*w,mask)} for region,mask in rois.items()},
                'parameter_gradients_raw':grad_summary(named,grads[1:]),'parameter_gradients_weighted':grad_summary(named,[None if g is None else g*w for g in grads[1:]])}
            parameter_grads[name]=[None if g is None else g.cpu() for g in grads[1:]]
        assert all(p.grad is None for p in params)
        assert torch.equal(initial_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
        before={n:p.detach().clone() for n,p in named}
        scaler.scale(loss).backward();scaler.unscale_(opt)
        before_clip=grad_summary(named,[p.grad for p in params])
        torch.nn.utils.clip_grad_norm_(refiner.parameters(),12.);torch.nn.utils.clip_grad_norm_(decoder,5.)
        received=grad_summary(named,[p.grad for p in params]);scale_before=float(scaler.get_scale())
        opt.step() if not scaler.is_enabled() else scaler.step(opt)
        scaler.update();skipped=float(scaler.get_scale())<scale_before
        if skipped:raise RuntimeError('Shadow optimizer step skipped')
        delta=[(p.detach()-before[n]).cpu() for n,p in named]
        updates={}
        for (n,p),d in zip(named,delta):
            group=grouping(n);record=updates.setdefault(group,{'norm2':0.,'changed':0,'numel':0,'steps':[]})
            record['norm2']+=float(d.float().square().sum());record['changed']+=int((d!=0).sum());record['numel']+=p.numel();record['steps'].append(int(opt.state[p]['step']))
        for record in updates.values():
            record['actual_update_norm']=record.pop('norm2')**.5;record['changed_fraction']=record.pop('changed')/record['numel'];record['step_range']=[min(record['steps']),max(record['steps'])];del record['steps']
        directional={}
        for name,grads in parameter_grads.items():
            groups={}
            for (n,_),g,d in zip(named,grads,delta):
                if g is not None:groups[grouping(n)]=groups.get(grouping(n),0.)+float((g*d).sum())
            directional[name]={'raw_g_dot_delta_by_group':groups,'raw_g_dot_delta':sum(groups.values()),'diagnostic_only':name.endswith('DIAGNOSTIC_ONLY')}
        active[0]=False
        src_after,final_after=predict(model,refiner,image)
        row={'probe':probe,'case':sample['case'],'train_case_index':case_index,'mode':'INDEPENDENT_CURRENT_CHECKPOINT_SINGLE_STEP_FORK',
            'optimizer_steps_before':100,'optimizer_steps_after':101,'new_optimizer_steps':1,
            'batch_sha256':hashlib.sha256(sample['image'].numpy().tobytes()+sample['mask'].numpy().tobytes()).hexdigest(),
            'indices':indices.tolist(),'ROI_counts':{k:int(m.sum()) for k,m in rois.items()},'loss_support':box['supports'],
            'loss':float(loss.detach()),'source_auxiliary':auxiliary,'negative_guard':guard_meta,'AMP_train_vs_FP32_source_max_abs':float((source.detach().float()-src_before).abs().max()),
            'loss_gradient_records':grad_records,'gradient_before_clip':before_clip,'gradient_optimizer_receives':received,
            'actual_parameter_updates':updates,'loss_gradient_dot_actual_update':directional,
            'source_logit_change':{k:roi_stat(src_after-src_before,m) for k,m in rois.items()},
            'final_logit_change':{k:roi_stat(final_after-final_before,m) for k,m in rois.items()},
            'hard_source_logit_before':roi_stat(src_before,hard),'hard_source_logit_after':roi_stat(src_after,hard),
            'hard_FN_recovered_members':int((hard&(final_after.sigmoid()>=.5)).sum()),
            'AMP_skipped':skipped,'O_enabled':False,'test_used':False,'validation_performance_claim':False}
        with (a.out/'TRAINING_TRACE.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append({'case':row['case'],'hard_FN':row['ROI_counts']['hard_overlap_FN'],'selected_hard_FN':row['ROI_counts']['hard_selected'],
                     'mean_hard_source_delta':row['source_logit_change']['hard_overlap_FN']['mean'],
                     'hard_objective_g_dot_delta':directional['hard_source_BCE_DIAGNOSTIC_ONLY']['raw_g_dot_delta'],
                     'hard_FN_recovered':row['hard_FN_recovered_members']})
        print(json.dumps(rows[-1]),flush=True)
        del grads,parameter_grads,before,delta
    hook.remove()
    assert all(float(s['step'])==100 for s in payload['optimizer']['state'].values())
    write(a.out/'RESULT.json',{'status':'COMPLETE','new_optimizer_steps':len(rows),'independent_forks':True,'source_checkpoint_unchanged':True,
          'source_checkpoint_sha256':legacy.sha(lock['reference_checkpoint']),'cases':rows,'seconds':time.monotonic()-start,
          'test_used':False,'performance_comparison':'NOT_RUN; single-step training-case diagnostics only','peak_gpu_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
