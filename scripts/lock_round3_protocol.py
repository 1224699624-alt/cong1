"""Lock local case identities/budgets before any Round3 optimizer update."""
import json,hashlib
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/v3_round3'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'outputs/v3_round2/CURRENT_REMOTE_SPLIT.json').read_text())
    data={}
    for domain in ['TSRS','RAM']:
        data[domain]={}
        for split in ['train','val']:
            rows=[]
            for case in old['datasets'][domain][split]['ids']:
                if domain=='TSRS':
                    raw=ROOT/'data/raw/TSRS_RSNA-Epiphysis'
                    image=next(raw/split/(case+ext) for ext in ['.jpg','.jpeg','.png','.bmp'] if (raw/split/(case+ext)).exists())
                    label=raw/(split+'_labels')/(case+'.png')
                    if not label.exists():
                        if case!='1833' or split!='train':raise FileNotFoundError(label)
                        label=ROOT/'data/raw_variants/TSRS_RSNA-Epiphysis_badlabel_filtered_v1/train_labels/1833.png'
                        assert sha(label)==old['1833_label']['sha256']
                    prior=ROOT/'outputs/priors/r317_image_centers_only'/split/(case+'.png')
                    shapes=[Image.open(p).size for p in [image,label,prior]]
                    assert len(set(shapes))==1,(case,shapes)
                    row={'case':case,'image':str(image),'label':str(label),'prior':str(prior),
                         'image_sha256':sha(image),'label_sha256':sha(label),'prior_sha256':sha(prior),'hw':list(shapes[0][::-1]),
                         'preprocessed_id':split+'_'+case}
                else:
                    r=ROOT/'data/remote_variants/RAM-W600/BoneSegmentation'
                    image=r/'images'/(case+'.bmp');label=r/'masks'/split/(case+'.npy')
                    prior=ROOT/'outputs/priors/r323_ram_native_r317_single_seed'/split/(case+'.npy')
                    row={'case':case,'image':str(image),'label':str(label),'prior':str(prior),
                         'image_sha256':sha(image),'label_sha256':sha(label),'prior_sha256':sha(prior)}
                rows.append(row)
            data[domain][split]=rows
        assert not set(r['case'] for r in data[domain]['train'])&set(r['case'] for r in data[domain]['val'])
    write('DATA_MANIFEST.json',{'schema':'ROUND3_DATA_V1','datasets':data,'raw_modified':False,'test_used':False})
    protocol={'schema':'ROUND3_PROTOCOL_V1','status':'LOCKED_BEFORE_TRAINING','data_manifest_sha256':sha(OUT/'DATA_MANIFEST.json'),
      'data_counts':{'TSRS':{'train':875,'val':96},'RAM':{'train':425,'val':69}},
      'native_sha256':{'TSRS':'55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b','RAM':'88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9'},
      'prior_adaptation':{'enabled':True,'max_epochs':6,'updates_per_epoch':425,'max_updates':2550,'batch':1,'seed':3510,
        'optimizer':'AdamW','lr':3e-4,'weight_decay':1e-4,'scheduler':'constant','augmentation':'none',
        'validation':'all69 after each epoch, minimum proper valid BCE','checkpoint_limit':1,'resolution':[192,192],
        'source_sha256':'ab08b0f6a2e4c99bccd38e9136e09fb4b6650b425e93e5855e7d8c314f31e725',
        'target':'area-resampled true observed >=2 member overlap; valid nearest mask; full14observed dataset contract',
        'quality_gate':{'all_cases_finite':True,'bce_better_than_training_prevalence_constant':True,
          'positive_probability_above_negative':True,'native_pixel_precision_gt_zero':True,'native_pixel_recall_gt_zero':True,
          'predicted_positive_fraction_between_0_and_1':True,'native_threshold':.5,
          'interpretation':'minimal noncollapsed informative prior for bounded pilot only; no segmentation improvement or calibration claim'}},
      'pilot':{'max_epochs':2,'updates_per_epoch':100,'max_updates_per_arm':200,'seed':3512,'batch':1,
        'trainability':'joint_finetune','optimizer':'AdamW','backbone_lr':2e-6,'output_lr':2e-5,'weight_decay':1e-4,
        'scheduler':'poly exponent0.9 at epoch start, pergroup ratios preserved',
        'TSRS_augmentation':'existing nnUNet joint spatial augmentation; A consumes same loader then zero seam at backbone',
        'RAM_augmentation':'none','TSRS_precision':'bf16 train, native predictor inference','RAM_precision':'fp32 TF32 true',
        'validation_every_epochs':1,'TSRS_val_iterations':25,'selection':{'TSRS':'nnUNet EMA foreground Dice','RAM':'val overall DSC'},
        'checkpoint_limit_per_arm':1,'full_validation_counts':{'TSRS':96,'RAM':69},'TSRS_arms':['A','B'],
        'RAM_arms':['A','B','C','D'],'TSRS_C_D':'PENDING supervision; not authorized as native true overlap result',
        'separation_weight':.035,'overlap_weight':.035,'preserve_weight':.10,'attenuation_reference_mass':.01,
        'routing_individual_cap':.15,'routing_merged_cap':.15,'output_refiner_init':'fresh same-seed zero residual C/D; no mature backbone',
        'refinement_steps':1,'gradient_clip_norm':5.,'segmentation_threshold':.5,'postprocessing':'native unchanged, no new postprocess',
        'gate_before_C_D':['accepted image-only prior','C/D real smoke PASS'],
        'performance_acceptance':'continuous deltas to N0 and A; no posthoc tolerance or significance claims'},
      'resources':{'host':'connect.westc.seetacloud.com:52807','remote_output':'/root/v3_round3','gpu':0,
        'observed_system_free_GiB_approx':23,'observed_data_free_GiB_approx':4.7,'minimum_free_GiB':4,
        'max_new_output_GiB':8,'max_wall_seconds_prior':7200,'max_wall_seconds_each_pilot':7200,
        'sequential_gpu_jobs':True,'delete_history':False,'cache_full_logits':False},
      'test_used':False,'other_backbones':False,
      'source_hashes':{f:sha(ROOT/'scripts'/f) for f in ['structural_prior_v3.py','local_supervision_v3.py','r349_parameter_gradient_gate_v3.py','run_r201_unified_eval.py','evaluate_r329_ram_official_metrics_val.py']},
      'entrypoint_hash_policy':'new runner hash and runtime lock hash saved before each run; frozen numerical protocol may not change after results'}
    write('ROUND3_PROTOCOL_LOCK.json',protocol)
    print(json.dumps({'status':'LOCKED','counts':protocol['data_counts'],'1833_explicit_source':data['TSRS']['train'][old['datasets']['TSRS']['train']['ids'].index('1833')]}))

if __name__=='__main__':main()
