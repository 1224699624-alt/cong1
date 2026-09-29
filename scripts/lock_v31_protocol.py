"""Freeze V3.1/Round4 incremental protocol before optimizer updates."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/v4_1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'R4_1_PROTOCOL_LOCK.json'
    if path.exists():raise FileExistsError(path)
    files=['r332_v31_port.py','run_v31_ram_mature.py','nnunet_trainers/nnUNetTrainerV31HeatmapInput.py','analyze_v31_tsrs_changes.py',
           'r349_parameter_gradient_gate_v3.py','structural_prior_v3.py','round3_common.py','run_r201_unified_eval.py','evaluate_r329_ram_official_metrics_val.py']
    round3=ROOT/'outputs/v3_round3';a=json.loads((round3/'TSRS_A_MANIFEST.json').read_text());b=json.loads((round3/'TSRS_B_MANIFEST.json').read_text())
    value={'schema':'V3_1_ROUND4_LOCK_V1','status':'LOCKED_BEFORE_UPDATE','research_question':{
      'TSRS':'H_input-A isolates joint heatmap input; B-H isolates separation+preserve+routing package',
      'RAM':'M_O-M_ref_continue isolates explicit frozen O conditioning on the same mature R332 combination'},
      'component_hashes':{
        'R317_checkpoint':sha(ROOT/'outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_r201_selected.pth'),
        'R332_checkpoint':sha(ROOT/'outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth'),
        'R330_refiner_source':'7fd0281d976ea2d5b9e997ed2e6e6447f52655fbdf4163f9fb29a00dfdc3c34c',
        'R325_native':sha(ROOT/'outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth'),
        'image_only_O':sha(round3/'ram_image_only_overlap_best.pth'),
        'R350_corrected_table':sha(ROOT/'outputs/analysis/latest_best_vs_baseline_20260824.csv'),
        'Round3_A_manifest':sha(round3/'TSRS_A_MANIFEST.json'),'Round3_B_manifest':sha(round3/'TSRS_B_MANIFEST.json'),
        'Round3_A_checkpoint':a['selected_checkpoint_sha256'],'Round3_B_checkpoint':b['selected_checkpoint_sha256'],
        'Round3_data_manifest':sha(round3/'DATA_MANIFEST.json'),'Round3_protocol':sha(round3/'ROUND3_PROTOCOL_LOCK.json')},
      'source_code_hashes':{**{f:sha(ROOT/'scripts'/f) for f in files},
                            'run_v3_1_tsrs_h.sh':sha(ROOT/'run_v3_1_tsrs_h.sh'),
                            'lock_v31_protocol.py':sha(ROOT/'scripts/lock_v31_protocol.py')},
      'parity':{'required_before_RAM_updates':True,'full_val_cases':69,'dtype':'float32','logit_max_abs_tolerance':1e-6,
        'mask_mismatch_pixels':0,'metric_max_abs_tolerance':1e-12,'steps':[0,1,2],
        'required_preservation':['paired R332 model state','14 independent sigmoid members','first six refiner features and order','mature residual/gate','two iterative feature recomputations'],
        'new_O_feature':'seventh refiner input, weight zero only; mature residual/gate never reset'},
      'TSRS_H_input':{'runs':['H_input'],'reference_runs':['Round3_A','Round3_B'],'reference_reused':True,
        'native_source':a['source_sha256'],'dataset_split_sha256':a['split_sha256'],'seed':3512,'updates':200,'epochs':2,'updates_per_epoch':100,'batch':1,
        'input':'[original normalized image, aligned R317 seam heatmap]','loss':'native nnUNet CE+Dice deep supervision only','auxiliary_losses':[],
        'routing':False,'optimizer':'AdamW','lr':2e-6,'weight_decay':1e-4,'scheduler':'nnUNet Poly epoch schedule exponent .9',
        'augmentation':'same existing nnUNet joint spatial pipeline as Round3 A/B','trainability':'all backbone parameters joint finetune',
        'selection':'EMA foreground Dice from 25 validation iterations/epoch','evaluator':'strict 96-case R201; threshold/geometry unchanged',
        'checkpoint_limit':1,'pixel_analysis':['delete TP','delete FP','add TP','add FP','high-prior true foreground','trusted seam background','ordinary foreground']},
      'RAM_mature_runs':{'runs':['M_ref_continue','M_O'],'shared_R332_sha256':'eaf3b6918f7b64d90ae4295be1df4ec4c2c314775e68472e85607667a4334655',
        'source_epoch':30,'seed':3321,'epochs':2,'updates_per_epoch':100,'updates':200,'batch':1,'steps':2,'chunk':3,'instances_per_image':4,
        'input':'two-channel port [image,zero], first image weights copied; hidden prior head removed because R332 use_prior=False',
        'M_ref_continue_O':'zero','M_O_O':'frozen image-only generator e7b738...','generator_retrained':False,
        'data':'RAM train425 augment=True, val69 augment=False; NativeWristDataset native/padding/canonical flip','loader_workers':0,
        'matched_sample_policy':'same case shuffle/RNG and augmentation; original R332 current-student choose_pair_instances/local support retained; per-update selected member indices logged and may diverge as method states diverge',
        'trainable_decoder':['decoder stages3/4','transposed conv3/4','seg head4'],'refiner_trainable':True,
        'historical_freeze_schedule':'not replayed: mature continuation starts from epoch30 post-freeze state, so identical selected decoder parameters are enabled from update1 in both arms',
        'optimizer':'AdamW','refiner_lr':8e-5,'decoder_lr':7e-6,'weight_decay':1e-4,'scheduler':'CosineAnnealingLR T_max=2 stepped per epoch',
        'loss':{'completion':'0.55 BCE + 0.45 Dice + .02 conservation + .05 boundary_l1','surface':.03,'instance_volume':.005,'pair_volume':.01,
                'step_weights':[1/3,2/3],'monotonic':.05,'contraction':.01,'global_native':.10},
        'gradient_clip':{'refiner':12.,'decoder':5.},'precision':'CUDA AMP fp16; same both arms',
        'selection':'first require M0 overall DSC/IoU non-degradation, then original R332 checkpoint_gate vs R325, progression passed checks, overlap NSD/DSC, -MSD, -RAVD',
        'validation':'full69 each100 updates; R329 official metrics, threshold .5, top15 train pairs','checkpoint_limit':1,
        'max_seconds':7200,'disk_floor_bytes':4*1024**3},
      'budget':{'new_runs':3,'updates_per_run':200,'total_updates':600,'maximum_runs_if_reference_invalid':4,'maximum_updates':800,
                'N0_retrain':False,'generator_retrain':False,'test':False,'other_backbones':False,'three_seeds':False,'long_training':False},
      'runtime':{'remote_root':'/root/v4_1','gpu':0,'sequential_jobs':True,'system_free_floor_GiB':4,'delete_history':False,
                 'code_snapshot_required':True,'logs_and_JSON_required':True},
      'TSRS_overlap':{'status':'UNKNOWN','C_D':'NOT_RUN','known_external_source':'G:/gutou/标注/掌骨数据标注作业返回/flat_output_suffix',
                      'source_access':'SOURCE_NOT_ACCESSED; outside current workspace authorization','migration_O':'exploratory candidate, not TSRS-native GT'},
      'evidence_boundary':'single-seed 200-update diagnostics; no p-values/CI/stability claims; no test tuning','test_used':False}
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False));print(path,sha(path))
if __name__=='__main__':main()
