"""Frozen feature/margin/spatial audit and training-case functional gradients."""
import argparse,copy,hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion,distance_transform_edt
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy
from whitebox_ram_diagnosis import grouping


def summarize(x):
    x=x.detach().float().reshape(-1)
    if not x.numel():return {'n':0,'sum':0.,'sum_abs':0.,'near_zero_025':0,'positive':0}
    return {'n':x.numel(),'sum':float(x.sum()),'sum_abs':float(x.abs().sum()),
            'near_zero_025':int((x.abs()<.25).sum()),'positive':int((x>=0).sum())}


def add(dst,key,values):
    row=dst.setdefault(key,{})
    for k,v in values.items():row[k]=row.get(k,0)+v


def selected_features(f,coords,native_hw,cap=128):
    # Deterministic striding, fixed before outcomes; never fit a classifier.
    if len(coords)>cap:coords=coords[::int(np.ceil(len(coords)/cap))]
    h,w=native_hw;y=coords[:,0];x=coords[:,1]
    grid=torch.stack((2*(x.float()+.5)/w-1,2*(y.float()+.5)/h-1),-1)[None,None]
    values=F.grid_sample(f,grid,mode='bilinear',align_corners=False)[0,:,0].T
    return values,coords


def cosine(a,b):
    den=a.norm()*b.norm()
    return float((a*b).sum()/den) if float(den)>0 else None


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    start=time.monotonic();torch.set_num_threads(lock.get('cpu_threads',2));legacy.seed_all(4421);device=torch.device('cuda')
    _,template,rt,_,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    models={};features={};handles=[]
    for arm,path in lock['checkpoints'].items():
        m=copy.deepcopy(template);r=copy.deepcopy(rt);state=torch.load(path,map_location='cpu',weights_only=False)
        m.load_state_dict(state['model']);r.load_state_dict(state['refiner']);m.eval().requires_grad_(False);r.eval().requires_grad_(False);models[arm]=(m,r);features[arm]={}
        for level in [3,4]:
            def hook(module,inputs,output,arm=arm,level=level):features[arm][level]=output.detach()
            handles.append(m.backbone.decoder.stages[level].register_forward_hook(hook))
    def forward(image):
        result={}
        for arm,(m,r) in models.items():
            src=m(torch.cat([image,torch.zeros_like(image)],1));states,_=legacy.iterative_refine_o(image,src,r,2,torch.zeros_like(image));result[arm]=(src,states[-1])
        return result
    def masks(outputs,truth,valid):
        si,zi=outputs['initial'];_,zr=outputs['reference'];_,za=outputs['source_aux'];pi=zi.sigmoid()>=.5;pr=zr.sigmoid()>=.5;pa=za.sigmoid()>=.5
        return {'hard_FN':truth&(truth.sum(1,keepdim=True)>=2)&~pi&valid&(si < -5.00001),
                'treatment_only_FP':~truth&~pr&pa&valid,
                'candidate_new_FP_vs_initial':~truth&~pi&pa&valid,
                'correct_positive':truth&pr&pa&valid}
    data=Path(lock['dataset']);val=legacy.NativeWristDataset(data,'val',False)
    totals={};counts={};rows=[];feature_rows=[];reconstruction=[];surface_rows=[]
    with torch.no_grad():
        for i,batch in enumerate(DataLoader(val,batch_size=1,shuffle=False,num_workers=0)):
            if time.monotonic()-start>lock['max_seconds']:raise TimeoutError('budget')
            image=batch['image'].to(device);truth=batch['mask'].to(device)>.5;valid=batch['valid'].to(device).bool();outputs=forward(image);rois=masks(outputs,truth,valid)
            row={'case':batch['case'][0],'ROI_counts':{k:int(v.sum()) for k,v in rois.items()},'stats':{},'surface_distances':{}}
            for name,mask in rois.items():
                counts[name]=counts.get(name,0)+int(mask.sum())
                for arm in ['reference','source_aux']:
                    for level,values in [('source',outputs[arm][0]),('final',outputs[arm][1])]:
                        key=name+'/'+arm+'/'+level;s=summarize(values[mask]);add(totals,key,s);row['stats'][key]=s
                s=summarize((outputs['source_aux'][0]-outputs['reference'][0])[mask]);add(totals,name+'/source_shift',s)
            # Counterfactual component swaps: existing checkpoints only.
            sref=outputs['reference'][0];saux=outputs['source_aux'][0]
            backbone_swap=legacy.iterative_refine_o(image,saux,models['reference'][1],2,torch.zeros_like(image))[0][-1]
            refiner_swap=legacy.iterative_refine_o(image,sref,models['source_aux'][1],2,torch.zeros_like(image))[0][-1]
            fp=rois['treatment_only_FP'];row['swap_treatment_FP']={'n':int(fp.sum()),'aux_backbone_ref_refiner':int((fp&(backbone_swap.sigmoid()>=.5)).sum()),'ref_backbone_aux_refiner':int((fp&(refiner_swap.sigmoid()>=.5)).sum())}
            for k,v in row['swap_treatment_FP'].items():counts['swap_'+k]=counts.get('swap_'+k,0)+v
            pred_ref=(outputs['reference'][1].sigmoid()>=.5)[0].cpu().numpy()
            pred_aux=(outputs['source_aux'][1].sigmoid()>=.5)[0].cpu().numpy()
            for member in range(14):
                tm=truth[0,member].cpu().numpy();boundary=tm^binary_erosion(tm)
                pr=pred_ref[member];pa=pred_aux[member]
                if not boundary.any():
                    for name in ['hard_FN','treatment_only_FP','candidate_new_FP_vs_initial']:
                        counts[name+'_boundary_unavailable']=counts.get(name+'_boundary_unavailable',0)+int(rois[name][0,member].sum())
                    surface_rows.append({'case':row['case'],'member':member,'reference':{'nsd':1. if not pr.any() else 0.},'source_aux':{'nsd':1. if not pa.any() else 0.}})
                    continue
                distance=distance_transform_edt(~boundary)
                sr={'case':row['case'],'member':member}
                for label,prediction in [('reference',pr),('source_aux',pa)]:
                    ps=prediction^binary_erosion(prediction);pn=int(ps.sum());tn=int(boundary.sum())
                    pmatch=int((distance[ps]<=2).sum())
                    tmatch=int((distance_transform_edt(~ps)[boundary]<=2).sum()) if pn else 0
                    sr[label]={'pred_surface':pn,'GT_surface':tn,'matched_pred_surface':pmatch,'matched_GT_surface':tmatch,'nsd':(pmatch+tmatch)/(pn+tn)}
                surface_rows.append(sr)
                for name in ['hard_FN','treatment_only_FP','candidate_new_FP_vs_initial']:
                    region=rois[name][0,member].cpu().numpy();d=distance[region]
                    if not d.size:continue
                    item={'n':int(d.size),'sum':float(d.sum()),'within1':int((d<=1).sum()),'within2':int((d<=2).sum()),'within5':int((d<=5).sum())}
                    add(totals,name+'/GT_boundary_distance',item);add(row['surface_distances'],name,item)
                hard_coords=torch.nonzero(rois['hard_FN'][0,member]);fp_coords=torch.nonzero(rois['treatment_only_FP'][0,member]);pos_coords=torch.nonzero(rois['correct_positive'][0,member])
                if len(hard_coords) and len(fp_coords) and len(pos_coords):
                    item={'case':row['case'],'member':member,'levels':{}}
                    for level in [3,4]:
                        fh,_=selected_features(features['reference'][level],hard_coords,image.shape[-2:]);ff,_=selected_features(features['reference'][level],fp_coords,image.shape[-2:]);fg,_=selected_features(features['reference'][level],pos_coords,image.shape[-2:])
                        item['levels'][str(level)]={'hard_fp_mean_cosine':cosine(fh.mean(0),ff.mean(0)),
                            'hard_positive_mean_cosine':cosine(fh.mean(0),fg.mean(0)),'fp_positive_mean_cosine':cosine(ff.mean(0),fg.mean(0)),
                            'sample_counts':[len(fh),len(ff),len(fg)]}
                    feature_rows.append(item)
                # Exact linear-head algebra on native-resolution stage4 samples.
                for name,coords in [('hard_FN',hard_coords),('treatment_only_FP',fp_coords)]:
                    if not len(coords):continue
                    if len(coords)>128:coords=coords[::int(np.ceil(len(coords)/128))]
                    y,x=coords[:,0],coords[:,1];fr=features['reference'][4][0,:,y,x].T.double();fa=features['source_aux'][4][0,:,y,x].T.double()
                    hr=models['reference'][0].backbone.decoder.seg_layers[4];ha=models['source_aux'][0].backbone.decoder.seg_layers[4]
                    wr=hr.weight[member,:,0,0].double();wa=ha.weight[member,:,0,0].double();br=hr.bias[member].double();ba=ha.bias[member].double()
                    feature=(fa-fr)@wr;weight=fr@(wa-wr);interaction=(fa-fr)@(wa-wr);bias=torch.ones_like(feature)*(ba-br)
                    actual=(outputs['source_aux'][0][0,member,y,x]-outputs['reference'][0][0,member,y,x]).double()
                    residual=actual-feature-weight-interaction-bias
                    reconstruction.append({'case':row['case'],'member':member,'ROI':name,'n':len(y),
                        'feature_sum':float(feature.sum()),'head_weight_sum':float(weight.sum()),'interaction_sum':float(interaction.sum()),'head_bias_sum':float(bias.sum()),
                        'actual_sum':float(actual.sum()),'max_abs_reconstruction_error':float(residual.abs().max())})
            with (a.out/'INFERENCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            rows.append(row);print(json.dumps({'val_case':i+1,'elapsed':time.monotonic()-start}),flush=True)
    for h in handles:h.remove()
    # Three fixed training cases, functional backward only, no optimizer objects.
    train=legacy.NativeWristDataset(data,'train',False);order=torch.randperm(len(train),generator=torch.Generator().manual_seed(4421))[:3].tolist();gradient_rows=[]
    m=models['reference'][0];legacy.set_joint_trainability(m,True)
    # Additional frozen decoder stage2 is observed only, never trained/saved.
    m.backbone.decoder.stages[2].requires_grad_(True)
    named=[(n,p) for n,p in m.named_parameters() if p.requires_grad];params=[p for _,p in named]
    for index in order:
        b=train[index];image=b['image'][None].to(device);truth=b['mask'][None].to(device)>.5;valid=b['valid'][None].to(device).bool()
        with torch.no_grad():outputs=forward(image);rois=masks(outputs,truth,valid)
        z=m(torch.cat([image,torch.zeros_like(image)],1));objectives={};vectors={};roi_counts={}
        for label,region in [('hard_FN',rois['hard_FN']),('new_FP',rois['treatment_only_FP'])]:
            roi_counts[label]=int(region.sum())
            objective=F.softplus(-z)[region].mean() if label=='hard_FN' else F.softplus(z)[region].mean()
            if not bool(region.any()):objective=z.sum()*0
            grads=torch.autograd.grad(objective,params,retain_graph=True,allow_unused=True)
            grouped={}
            for (name,p),g in zip(named,grads):
                if g is not None:grouped.setdefault(grouping('model.'+name),[]).append(g.detach().float().reshape(-1))
            vectors[label]={k:torch.cat(v) for k,v in grouped.items()};objectives[label]=float(objective.detach())
        comparison={}
        for key in vectors['hard_FN']:
            left=vectors['hard_FN'][key];right=vectors['new_FP'][key]
            comparison[key]={'FN_gradient_norm':float(left.norm()),'FP_gradient_norm':float(right.norm()),'cosine':cosine(left,right)}
        assert all(p.grad is None for p in params)
        gradient_rows.append({'case':b['case'],'ROI_counts':roi_counts,'source_BCE_objectives':objectives,'groups':comparison,'optimizer_updates':0})
    result={'status':'COMPLETE','optimizer_updates':0,'test_used':False,'cpu_threads':torch.get_num_threads(),'val_cases':len(rows),'train_gradient_cases':gradient_rows,
        'ROI_counts':counts,'aggregate':totals,'feature_mean_comparisons':feature_rows,'head_decomposition_samples':reconstruction,'member_surface_decomposition':surface_rows,
        'scope':'Feature means and local gradient cosines do not prove representational inseparability or validation benefit of an update.',
        'stage2_note':'frozen stage2 requires_grad enabled only in isolated zero-update observation; no model saved',
        'seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
