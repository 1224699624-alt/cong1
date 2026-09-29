"""V3.1 mature R332 continuation with zero vs frozen explicit O."""
from __future__ import annotations
import argparse,hashlib,json,math,random,time,shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel,initialize_from_r285
from train_r284_ramw600_overlap_prior import dice_bce_logits,top_training_pairs
from train_r330_ram_pair_surface_volume_prior import choose_pair_instances,soft_surface_nsd_loss,volume_losses
from train_r327_ram_native_instance_completion_prior import completion_loss
from train_r332_ram_joint_iterative_refinement import set_joint_trainability,flatten,audit_progression,checkpoint_gate
from evaluate_r329_ram_official_metrics_val import official_metrics,_official_one
from structural_prior_v3 import ImageOnlyOverlapPrior
from r332_v31_port import R332RefinerWithO,load_mature_with_zero_o,iterative_refine_o,EXPECTED

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def load_components(args,device):
    if sha(args.r332)!=EXPECTED:raise ValueError('R332 SHA')
    payload=torch.load(args.r332,map_location='cpu',weights_only=False)
    model=RamNnUNetTwoChannel().to(device);backbone_mapping=initialize_from_r285(model,args.r332,device)
    refiner=R332RefinerWithO().to(device);refiner_mapping=load_mature_with_zero_o(refiner,payload['refiner'])
    generator=ImageOnlyOverlapPrior().to(device);op=torch.load(args.overlap,map_location='cpu',weights_only=False)
    if sha(args.overlap)!=args.overlap_sha or op['input_dependencies']!=['image']:raise ValueError('O checkpoint')
    generator.load_state_dict(op['model'],strict=True);generator.eval().requires_grad_(False)
    return payload,model,refiner,generator,backbone_mapping,refiner_mapping

def get_o(generator,image,use_o):
    with torch.no_grad():return generator(image).sigmoid() if use_o else torch.zeros_like(image)

def loss_for_batch(model,refiner,generator,image,target,valid,use_o):
    base_logits=model(torch.cat((image,torch.zeros_like(image)),1));base_probability=base_logits.sigmoid()
    with torch.no_grad():
        indices=choose_pair_instances(base_probability.detach(),target,4)
        selected_target=target[0,indices].unsqueeze(1)
        initial_error=torch.abs(base_probability[0,indices].unsqueeze(1).detach()-selected_target)
        local=F.max_pool2d((initial_error>.15).float(),11,1,5)
        overlap=get_o(generator,image,use_o)
    states,deltas=iterative_refine_o(image,base_logits,refiner,2,overlap,indices=indices)
    step_losses=[];stats=[]
    for step in range(1,3):
        selected_logits=states[step][0,indices].unsqueeze(1)
        base,values=completion_loss(selected_logits,selected_target,local,deltas[step-1],.05)
        probability=selected_logits.float().sigmoid();surface=soft_surface_nsd_loss(probability,selected_target)
        instance_volume,pair_volume=volume_losses(probability,selected_target)
        loss=base+.03*surface+.005*instance_volume+.01*pair_volume
        step_losses.append(loss);stats.append({**values,'surface':float(surface.detach()),
          'instance_volume':float(instance_volume.detach()),'pair_volume':float(pair_volume.detach()),
          'delta_abs':float(deltas[step-1].detach().abs().mean())})
    deep=step_losses[0]/3+step_losses[1]*2/3
    zero=base_logits.sum()*0;monotonic=F.relu(step_losses[1]-step_losses[0].detach())
    contraction=F.relu(deltas[1].abs().mean()-deltas[0].detach().abs().mean())
    global_loss=dice_bce_logits(base_logits,target);total=deep+.05*monotonic+.01*contraction+.10*global_loss
    valid_member=valid.expand_as(target).bool();selected_valid=valid_member[0,indices].unsqueeze(1)
    supervision={'indices':indices.tolist(),'positive_full':[int(x.sum()) for x in selected_target],
      'negative_full':[int(x.numel()-x.sum()) for x in selected_target],
      'positive_valid':[int((x.bool()&v).sum()) for x,v in zip(selected_target,selected_valid)],
      'negative_valid':[int((~x.bool()&v).sum()) for x,v in zip(selected_target,selected_valid)],
      'local_pixels':[int(x.sum()) for x in local]}
    return total,{'global':global_loss,'monotonic':monotonic,'contraction':contraction},stats,overlap,indices,supervision

@torch.inference_mode()
def collect(model,refiner,generator,loader,device,use_o,pairs):
    model.eval();refiner.eval();outputs={f'step{i}':{} for i in range(3)};diag=[]
    totals={k:np.zeros(14,dtype=np.int64) for k in ['recovered_fn','introduced_fn','removed_fp','introduced_fp','base_volume','final_volume','target_volume']}
    member_rows={'base':[[] for _ in range(14)],'final':[[] for _ in range(14)]}
    for batch in loader:
        image=batch['image'].to(device);target_t=batch['mask'].to(device)>.5;truth=target_t.cpu().numpy()[0]
        base=model(torch.cat((image,torch.zeros_like(image)),1));o=get_o(generator,image,use_o);states,deltas=iterative_refine_o(image,base,refiner,2,o)
        preds=[(z.sigmoid()>=.5).cpu().numpy()[0] for z in states];case=batch['case'][0]
        for i,pred in enumerate(preds):outputs[f'step{i}'][case]={'pred':pred,'target':truth}
        b,f=preds[0],preds[-1]
        for k in range(14):member_rows['base'][k].append(_official_one(b[k],truth[k]));member_rows['final'][k].append(_official_one(f[k],truth[k]))
        totals['recovered_fn']+=(~b&f&truth).sum((1,2));totals['introduced_fn']+=(b&~f&truth).sum((1,2))
        totals['removed_fp']+=(b&~f&~truth).sum((1,2));totals['introduced_fp']+=(~b&f&~truth).sum((1,2))
        totals['base_volume']+=b.sum((1,2));totals['final_volume']+=f.sum((1,2));totals['target_volume']+=truth.sum((1,2))
        true_overlap=target_t.sum(1,keepdim=True)>=2;base_pred=torch.from_numpy(b).to(device)[None]
        member_fn=(target_t&~base_pred).any(1,keepdim=True)
        masks={'true_overlap_with_member_fn':true_overlap&member_fn,'true_overlap_without_member_fn':true_overlap&~member_fn,'non_overlap':~true_overlap}
        diag.append({'case':case,'O':{k:(float(o[m].mean()) if bool(m.any()) else None) for k,m in masks.items()},
                     'step_delta_abs':[float(x.abs().mean()) for x in deltas]})
    metrics={step:official_metrics(data,pairs) for step,data in outputs.items()}
    def aggregate(rows):
        result={}
        for metric in ['dsc','nsd_2px','voe','msd_px','ravd']:
            values=[float(r[metric]) for r in rows if r[metric] is not None and np.isfinite(float(r[metric]))]
            result[metric]=float(np.mean(values)) if values else None
        result['msd_fail_rate']=float(np.mean([float(r['msd_failed']) for r in rows]));return result
    member_surface=[{'member':k,'base':aggregate(member_rows['base'][k]),'final':aggregate(member_rows['final'][k])} for k in range(14)]
    per_case={case:{step:official_metrics({case:outputs[step][case]},pairs) for step in outputs} for case in outputs['step0']}
    return {'official':metrics,'flat':{k:flatten(v) for k,v in metrics.items()},'per_case':diag,
            'per_case_official':per_case,'member_surface':member_surface,
            'member_corrections':{k:v.tolist() for k,v in totals.items()},'cases':list(outputs['step0'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['M_ref_continue','M_O'],required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--r332',type=Path,required=True);p.add_argument('--overlap',type=Path,required=True);p.add_argument('--overlap-sha',required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--parity',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    parity=json.loads(a.parity.read_text());lock=json.loads(a.protocol.read_text());cfg=lock['RAM_mature_runs']
    if lock['status']!='LOCKED_BEFORE_UPDATE' or lock['test_used'] or lock['source_code_hashes']['run_v31_ram_mature.py']!=sha(Path(__file__)):
        raise ValueError('V3.1 protocol/code gate')
    if lock['component_hashes']['image_only_O']!=sha(a.overlap):raise ValueError('O hash differs from lock')
    if parity['status']!='PASS' or parity['checkpoint_sha256']!=sha(a.r332):raise ValueError('parity gate')
    a.output.mkdir(parents=True);device=torch.device('cuda');seed_all(cfg['seed']);start=time.time();use_o=a.arm=='M_O'
    payload,model,refiner,generator,bmap,rmap=load_components(a,device)
    initial_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    initial_refiner={k:v.detach().cpu().clone() for k,v in refiner.state_dict().items()}
    trainable_names=set_joint_trainability(model,True);decoder=[x for x in model.parameters() if x.requires_grad]
    optimizer=torch.optim.AdamW([{'params':refiner.parameters(),'lr':cfg['refiner_lr']},{'params':decoder,'lr':cfg['decoder_lr']}],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['epochs']);scaler=torch.cuda.amp.GradScaler(enabled=True)
    data=a.root/'data/remote_variants/RAM-W600';train=NativeWristDataset(data,'train',True);val=NativeWristDataset(data,'val',False)
    loader=DataLoader(train,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))
    val_loader=DataLoader(val,batch_size=1,shuffle=False,num_workers=1);pairs=top_training_pairs(data,15);history=[];best=None
    manifest={'status':'SMOKE' if a.smoke else 'RUNNING','arm':a.arm,'r332':str(a.r332),'r332_sha256':sha(a.r332),'source_epoch':payload['epoch'],
      'overlap_sha256':sha(a.overlap),'O_enabled':use_o,'protocol_sha256':sha(a.protocol),'parity_sha256':sha(a.parity),
      'runner_sha256':sha(Path(__file__)),'backbone_mapping':bmap,'refiner_mapping':rmap,'config':cfg,'test_used':False}
    save(a.output/'run_manifest.json',manifest)
    m0_flat=flatten(parity['port_metrics']['step2'])
    save(a.output/'M0_preupdate_validation.json',{'status':'REUSED_FROM_ZERO_UPDATE_FULL69_PARITY','official':parity['port_metrics'],'step2_flat':m0_flat,
                                                  'parity_sha256':sha(a.parity),'test_used':False})
    if a.smoke:
        batch=next(iter(loader));image=batch['image'].to(device);target=batch['mask'].to(device);valid=batch['valid'].to(device);optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):loss,parts,stats,o,indices,supervision=loss_for_batch(model,refiner,generator,image,target,valid,use_o)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        report={'status':'PASS','arm':a.arm,'case':batch['case'][0],'finite':bool(torch.isfinite(loss)),
          'loss':float(loss.detach()),'decoder_grad_sum':sum(float(x.grad.abs().sum()) for x in decoder if x.grad is not None),
          'refiner_grad_sum':sum(float(x.grad.abs().sum()) for x in refiner.parameters() if x.grad is not None),
          'o_channel_grad_sum':float(refiner.enc1.net[0].weight.grad[:,6].abs().sum()),
          'expected_o_gradient':'zero' if not use_o else 'positive','optimizer_steps':0,'indices':indices.tolist(),
          'O_stats':{'min':float(o.min()),'max':float(o.max()),'mean':float(o.mean())},'step_stats':stats,'supervision':supervision,'test_used':False}
        if not report['finite'] or report['decoder_grad_sum']<=0 or report['refiner_grad_sum']<=0 or (use_o and report['o_channel_grad_sum']<=0) or (not use_o and report['o_channel_grad_sum']!=0):report['status']='FAIL'
        save(a.output/'smoke.json',report);print(json.dumps(report),flush=True)
        if report['status']!='PASS':raise RuntimeError(report)
        return
    for epoch in range(1,cfg['epochs']+1):
        model.eval();model.backbone.decoder.stages[3].train();model.backbone.decoder.stages[4].train();model.backbone.decoder.transpconvs[3].train();model.backbone.decoder.transpconvs[4].train();model.backbone.decoder.seg_layers[4].train();refiner.train()
        totals={'loss':0.,'global':0.,'monotonic':0.,'contraction':0.,'o_weight_grad':0.};order=[];member_indices=[];batch_hashes=[];iterator=iter(loader)
        supervision_totals={k:np.zeros(14,dtype=np.int64) for k in ['positive_full','negative_full','positive_valid','negative_valid','local_pixels']}
        for _ in range(cfg['updates_per_epoch']):
            if time.time()-start>cfg['max_seconds']:raise TimeoutError('budget')
            if shutil.disk_usage(a.output).free<cfg['disk_floor_bytes']:raise RuntimeError('disk floor')
            batch=next(iterator);order.append(batch['case'][0]);batch_hashes.append(hashlib.sha256(batch['image'].numpy().tobytes()+batch['mask'].numpy().tobytes()+batch['valid'].numpy().tobytes()).hexdigest());image=batch['image'].to(device);target=batch['mask'].to(device);valid=batch['valid'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):loss,parts,stats,o,indices,supervision=loss_for_batch(model,refiner,generator,image,target,valid,use_o)
            member_indices.append(indices.tolist())
            for position,index in enumerate(supervision['indices']):
                for key in supervision_totals:supervision_totals[key][index]+=supervision[key][position]
            if not torch.isfinite(loss):raise ValueError('nonfinite')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            totals['o_weight_grad']+=float(refiner.enc1.net[0].weight.grad[:,6].abs().sum())
            torch.nn.utils.clip_grad_norm_(refiner.parameters(),12.);torch.nn.utils.clip_grad_norm_(decoder,5.)
            scaler.step(optimizer);scaler.update();totals['loss']+=float(loss.detach())
            for k in parts:totals[k]+=float(parts[k].detach())
        scheduler.step();validation=collect(model,refiner,generator,val_loader,device,use_o,pairs)
        flats=[validation['flat'][f'step{i}'] for i in range(3)];progression=audit_progression(flats);passes,checks=checkpoint_gate(flats[-1],payload['row']['r325_baseline'])
        m0_checks={'overall_dsc_not_below_M0':flats[-1]['overall_dsc']>=m0_flat['overall_dsc'],
                   'overall_iou_not_below_M0':flats[-1]['overall_iou']>=m0_flat['overall_iou']}
        m0_pass=all(m0_checks.values())
        score=(int(m0_pass),int(passes),progression['passed_checks'],flats[-1]['overlap_nsd_2px'],flats[-1]['overlap_dsc'],-flats[-1]['overlap_msd_px'],-flats[-1]['overlap_ravd'])
        row={'epoch':epoch,'updates':epoch*cfg['updates_per_epoch'],'case_order':order,'batch_tensor_sha256':batch_hashes,'member_indices':member_indices,
             'member_supervision_counts':{k:v.tolist() for k,v in supervision_totals.items()},'train':{k:v/cfg['updates_per_epoch'] for k,v in totals.items()},
             'validation':validation,'progression':progression,'R325_gate':checks,'passes_R325_gate':passes,
             'M0_gate':m0_checks,'passes_M0_region_gate':m0_pass,'selection_score':list(score)}
        history.append(row);save(a.output/'history.json',history);print(json.dumps({'arm':a.arm,'epoch':epoch,'step2':flats[-1],'O_grad':row['train']['o_weight_grad']}),flush=True)
        if best is None or score>best:
            best=score;torch.save({'model':model.state_dict(),'refiner':refiner.state_dict(),'epoch':epoch,'arm':a.arm,'row':row,
              'r332_sha256':sha(a.r332),'overlap_sha256':sha(a.overlap),'protocol_sha256':sha(a.protocol),'runner_sha256':sha(Path(__file__))},a.output/'best.pth')
            save(a.output/'selected_validation.json',validation)
    expected_updates=cfg['epochs']*cfg['updates_per_epoch']
    if history[-1]['updates']!=expected_updates:raise RuntimeError('update budget mismatch')
    all_cases=[x for row in history for x in row['case_order']];all_batches=[x for row in history for x in row['batch_tensor_sha256']];all_indices=[x for row in history for x in row['member_indices']]
    sequence={'case_order_sha256':hashlib.sha256(json.dumps(all_cases).encode()).hexdigest(),
              'batch_tensor_sha256':hashlib.sha256(json.dumps(all_batches).encode()).hexdigest(),
              'member_indices_sha256':hashlib.sha256(json.dumps(all_indices).encode()).hexdigest()}
    if a.arm=='M_O':
        reference=json.loads((a.output.parent/'M_ref_continue/history.json').read_text())
        ref_cases=[x for row in reference for x in row['case_order']];ref_batches=[x for row in reference for x in row['batch_tensor_sha256']];ref_indices=[x for row in reference for x in row['member_indices']]
        sequence.update({'case_order_matches_M_ref':all_cases==ref_cases,'batch_tensors_match_M_ref':all_batches==ref_batches,
                         'member_selection_difference_count':sum(x!=y for x,y in zip(all_indices,ref_indices)),
                         'member_selection_note':'dynamic current-student R332 selection retained; differences after model divergence are algorithm state, not RNG/data order'})
        if not sequence['case_order_matches_M_ref'] or not sequence['batch_tensors_match_M_ref']:raise RuntimeError('matched data/augmentation sequence failed')
    selected=torch.load(a.output/'best.pth',map_location='cpu',weights_only=False);manifest.update({'status':'COMPLETE','selected_epoch':selected['epoch'],
      'selected_checkpoint':str(a.output/'best.pth'),'selected_checkpoint_sha256':sha(a.output/'best.pth'),'updates':expected_updates,
      'seconds':time.time()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'test_used':False})
    manifest['sequence_audit']=sequence
    manifest['actual_update_norms']={
      'backbone_l2':float(sum((selected['model'][k].float()-initial_model[k].float()).square().sum() for k in initial_model).sqrt()),
      'refiner_l2':float(sum((selected['refiner'][k].float()-initial_refiner[k].float()).square().sum() for k in initial_refiner).sqrt()),
      'explicit_o_weight_l2':float(selected['refiner']['enc1.net.0.weight'][:,6].float().norm()),
      'explicit_o_weight_nonzero':int(torch.count_nonzero(selected['refiner']['enc1.net.0.weight'][:,6]))}
    save(a.output/'run_manifest.json',manifest);print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
