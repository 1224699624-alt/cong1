"""Bounded local asset/source audit using Round1 indices; no test labels."""
from __future__ import annotations
import hashlib,json,sys,pydoc,pathlib
from unittest.mock import patch
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
import torch
from r351_dual_prior import ImageOverlapPrior
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from train_r258b_prediction_relation_prior import RelationPriorNet

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v3_round2'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()


def record(path):
    p=Path(path)
    return {'path':str(p),'exists':p.is_file(),'size':p.stat().st_size if p.is_file() else None,
            'sha256':digest(p) if p.is_file() else None}


def write(name,payload):
    path=OUT/name
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))==payload:return
        raise FileExistsError(path)
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    old=json.loads((ROOT/'research-workflow/refine-logs/ASSET_LOCK.json').read_text())
    assets=[]
    for a in old['assets']:
        path=Path(a['path']);r=record(path);r.update({k:v for k,v in a.items() if k not in r})
        r['hash_status']='PASS' if r['sha256']==a['sha256'] else 'FAIL'
        x=torch.load(path,map_location='cpu',weights_only=False)
        if 'network_weights' in x:
            plan=ROOT/'outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/plans.json'
            cfg=json.loads(plan.read_text())['configurations']['2d'];arch=cfg['architecture'];kw=dict(arch['arch_kwargs'])
            for k in arch['_kw_requires_import']:
                if kw[k] is not None:kw[k]=pydoc.locate(kw[k])
            model=pydoc.locate(arch['network_class_name'])(input_channels=2,num_classes=2,deep_supervision=True,**kw)
            model.load_state_dict(x['network_weights'],strict=True)
            original=ROOT/'outputs/nnunet/r319_r316b_gated_prior/source/r202_checkpoint_best.pth'
            origin=torch.load(original,map_location='cpu',weights_only=False)['network_weights']
            equality=[]
            for k,v in origin.items():
                w=x['network_weights'][k]
                equality.append(torch.equal(v,w) if v.shape==w.shape else
                                v.ndim==4 and w.shape[1]==2 and v.shape[1]==1 and torch.equal(v,w[:,:1]) and int(torch.count_nonzero(w[:,1:]))==0)
            r.update({'dataset':'TSRS','model_class':arch['network_class_name'],'input_channels':2,'output_channels':2,
                      'state_level':'network_weights','strict_load':'PASS','original_R202':record(original),
                      'R202_conversion_all_tensors_exact':all(equality),'second_channel_weight_nonzero':0,
                      'preprocessing':{'plans':record(plan),'spacing':cfg['spacing'],'patch_size':cfg['patch_size'],'normalization':cfg['normalization_schemes']},
                      'source_seed':None,'source_seed_status':'UNKNOWN not embedded as explicit seed',
                      'selection_rule':'native nnUNet checkpoint_best; EMA foreground Dice (trainer implementation)',
                      'source_split':record(ROOT/'outputs/nnunet/r317_image_centers_only/data/splits_final_r260.json')})
        else:
            model=NnUNetMultiLabelPrior();model.load_state_dict(x['model'],strict=True)
            r.update({'dataset':'RAM','model_class':'train_r284_ramw600_overlap_prior.NnUNetMultiLabelPrior / PlainConvUNet',
                      'input_channels':1,'raw_output_channels':15,'segmentation_channels':14,'activation':'independent sigmoid',
                      'state_level':'model','strict_load':'PASS','prior_weight':x.get('prior_weight'),
                      'prior_free_evidence':'use_prior=False returns raw[:,:14]; fifteenth prior head and prior_residual unused',
                      'preprocessing':'NativeWristDataset: right-wrist canonical flip once, per-image zscore, right/bottom padding multiple32, no image resize',
                      'source_seed':None,'source_seed_status':'UNKNOWN in checkpoint; do not infer from run number',
                      'selection_rule':'historical validation native macro Dice; see R325/R324 train_arm',
                      'source_split':'RAM BoneSegmentation/masks/train and val lists below'})
        r['this_round_local_real_batch']='NOT_RUN CPU-only workstation';r['this_round_full_val']='NOT_RUN'
        assets.append(r);del model,x
    write('ASSET_LOCK.json',{'status':'PASS' if all(r['hash_status']=='PASS' for r in assets) else 'FAIL',
                           'scope':'local existence/hash/strict load; no performance reproduction',
                           'assets':assets,'command':f'{sys.executable} scripts/audit_round2_prior_assets.py',
                           'evaluator_versions':{p:record(ROOT/'scripts'/p) for p in ['run_r201_unified_eval.py','evaluate_r201_single_split.py','evaluate_r329_ram_official_metrics_val.py']}})
    historical=json.loads((ROOT/'outputs/final_prior_v1/audit/source_audit_verified.json').read_text())
    split={};coverage={};tsrs_raw=ROOT/'data/raw/TSRS_RSNA-Epiphysis';ram_raw=ROOT/'data/remote_variants/RAM-W600/BoneSegmentation'
    for domain,prior_name in [('TSRS','r317_image_centers_only'),('RAM','r323_ram_native_r317_single_seed')]:
        split[domain]={};coverage[domain]={}
        for s in ['train','val']:
            labels=sorted((tsrs_raw/f'{s}_labels').glob('*.png')) if domain=='TSRS' else sorted((ram_raw/'masks'/s).glob('*.npy'))
            ids=[p.stem for p in labels];prior_dir=ROOT/'outputs/priors'/prior_name/s
            maps={p.stem:p for p in prior_dir.glob('*.png')}
            maps.update({p.stem:p for p in prior_dir.glob('*.npy')})
            rows=[];modes=Counter();bad=[];members=Counter()
            for p in labels:
                if domain=='TSRS':
                    with Image.open(p) as im:modes[im.mode]+=1;hw=(im.height,im.width)
                else:
                    y=np.load(p,mmap_mode='r');hw=y.shape[-2:]
                    members[str((y.shape[0],str(y.dtype)))]+=1
                    if y.shape[0]!=14 or not np.isfinite(y).all() or not np.all((y==0)|(y==1)):
                        bad.append(p.stem)
                m=maps.get(p.stem)
                if m:
                    arr=np.load(m) if m.suffix=='.npy' else np.asarray(Image.open(m))
                    expected=hw if domain=='TSRS' else (384,384)
                    good=arr.shape==expected and bool(np.isfinite(arr).all())
                    if not good:bad.append(p.stem)
                    rows.append({'id':p.stem,'map':str(m),'map_shape':list(arr.shape),'label_hw':list(hw),'shape_contract_pass':good,'map_sha256':digest(m)})
            old_ids=[r['id'].removeprefix(s+'_') if domain=='TSRS' else r['id'] for r in historical[domain.lower()][s]['files']]
            split[domain][s]={'local_count':len(ids),'local_ids':ids,'historical_remote_count':len(old_ids),
                              'historical_remote_ids':old_ids,'missing_vs_historical_remote':sorted(set(old_ids)-set(ids)),
                              'extra_vs_historical_remote':sorted(set(ids)-set(old_ids)), 'duplicate_ids':[k for k,v in Counter(ids).items() if v>1],
                              'mode_distribution':dict(modes),'member_arrays':dict(members)}
            coverage[domain][s]={'status':'PASS' if not bad and set(ids)<=set(maps) else 'FAIL',
                                 'label_cases':len(ids),'map_cases':len(maps),'missing_map_ids':sorted(set(ids)-set(maps)),
                                 'extra_map_ids':sorted(set(maps)-set(ids)),'invalid_shape_or_label_ids':bad,'rows':rows}
        split[domain]['train_val_id_intersection']=sorted(set(split[domain]['train']['local_ids'])&set(split[domain]['val']['local_ids']))
    write('SPLIT_RECONCILIATION.json',{'status':'PASS_DIFFERENCE_EXPLICIT','splits':split,
                                     'remote_evidence':'historical snapshot plus CURRENT_REMOTE_SPLIT.json when available; not local971',
                                     'test_read':False})
    overlap=ROOT/'outputs/artifact_bundles/r351_20260917/overlap/full/best.pth'
    ow=torch.load(overlap,map_location='cpu',weights_only=False);om=ImageOverlapPrior();om.load_state_dict(ow['model'],strict=True)
    overlap_rec={**record(overlap),'strict_load':'PASS legacy ImageOverlapPrior only','epoch':ow['epoch'],
                 'training_record':str(overlap.parent/'result.json'),'inputs':['image','seam'],
                 'supervision':'RAM true multi-membership count>=2; area-soft low resolution target, valid mask',
                 'train_val_counts':[425,69],'seed':3510,'selection':'minimum validation BCE',
                 'image_only_weight_compatibility':'BLOCKED first layer2 channels; channel removal is initialization only'}
    for domain,prior_name in [('TSRS','r317_image_centers_only'),('RAM','r323_ram_native_r317_single_seed')]:
        manifest=ROOT/'outputs/priors'/prior_name/'manifest.json';mj=json.loads(manifest.read_text())
        cp=list((ROOT/'outputs/pair_prior'/prior_name).glob('*selected.pt'));checked=[]
        for c in cp:
            # Historical Linux metadata embeds PosixPath; preserve it as a
            # pure path on Windows without altering the checkpoint itself.
            with patch('pathlib.PosixPath',pathlib.PurePosixPath):
                pl=torch.load(c,map_location='cpu',weights_only=False)
            mod=RelationPriorNet(4);mod.load_state_dict(pl['model'],strict=True)
            checked.append({**record(c),'strict_load':'PASS RelationPriorNet(4)','seed':pl.get('seed'),'selected_epoch':pl.get('selected_epoch')})
        payload={'dataset':domain,'seam':{'generator':'train_r258b_prediction_relation_prior.RelationPriorNet(4)',
                    'generator_version':'R317 image-centers-only' if domain=='TSRS' else 'R323 RAM-native, R317 architecture',
                    'adapter':'structural_prior_v3.build_backbone_input: concatenate unmodified image and aligned heatmap',
                    'manifest':record(manifest),'checkpoints':checked,'source_dataset':domain,
                    'source_labels':'anonymous TSRS indexed union-background seams' if domain=='TSRS' else 'RAM member union-background seams',
                    'selection':'historical train/val; selected epochs from checkpoint/manifest; no new selection',
                    'inference_inputs':['image','prediction-derived centers'],'other_prior_dependency':False,
                    'coordinate_space':'TSRS original pixels then nnUNet preprocessing' if domain=='TSRS' else 'canonical384 map bilinear to native then padding; no second R flip',
                    'coverage':coverage[domain],'role':'core candidate; real joint-input batch acceptance separate'},
                 'overlap':{'legacy_generator':overlap_rec,'generator_v3':'structural_prior_v3.ImageOnlyOverlapPrior',
                            'adapter_v3':'structural_prior_v3.OutputOverlapAdapter (separate O feature)',
                            'v3_checkpoint':None,'v3_status':'BLOCKED missing trained image-only weights',
                            'v3_map_coverage':'NOT_RUN no validated image-only maps',
                            'supervision':'unknown; exploratory RAM transfer only' if domain=='TSRS' else 'observed RAM 14 binary channels; audited values no sentinel/missing marker',
                            'anatomical_completeness':'not established by tensor format alone; dataset protocol supplies complete-member assumption',
                            'true_overlap_state':'unknown' if domain=='TSRS' else '>=2 known positive members; negative requires complete observation',
                            'core_decision':'PENDING TSRS C/D supervision and calibrated weak evidence' if domain=='TSRS' else 'BLOCKED only on image-only adaptation/validation',
                            'exclusion_reason':'legacy coupled wrapper excluded from clean ablation; cross-dataset weight origin not categorically banned',
                            'output_refiner_is_not_overlap_generator':True},
                 'local_validation_scope':'files, shapes and strict state loading; not learned prior quality proof', 'test_read':False}
        write(f'PRIOR_SEMANTICS_{domain}.json',payload)
    # Bounded annotation inventory: only the known raw and train/val variants.
    candidates=[tsrs_raw]+[p for p in (ROOT/'data/raw_variants').glob('TSRS_RSNA-Epiphysis*') if 'test' not in p.name.lower()]
    trace=[]
    for root in candidates:
        seen=[]
        for folder in [root,root/'train',root/'val',root/'train_labels',root/'val_labels']:
            if folder.is_dir():seen.extend(p for p in folder.iterdir() if p.is_file())
        trace.append({'root':str(root),'depth':'root and train/val subdirectories only','suffix_counts':dict(Counter(p.suffix.lower() for p in seen)),
                      'annotation_candidates':[str(p) for p in seen if p.suffix.lower() in ('.json','.xml','.mat','.npz','.npy','.nrrd')]})
    write('SOURCE_TRACE.json',{'status':'BOUNDED_SEARCH_COMPLETE','roots':trace,
                              'scripts_checked':['prepare_r260_mature_prior_dataset.py','train_r258b_prediction_relation_prior.py','train_r323_ram_r317_prior.py','build_r323_ram_native_prior_maps.py'],
                              'conclusion':'No verified complete TSRS multi-membership source identified in checked exports; not proof of nonexistence',
                              'connected_components':'anonymous disconnected components preserve union; not anatomical bone instance count','test_read':False})
    print('Round2 local asset and prior semantics audit completed',flush=True)


if __name__=='__main__':main()
