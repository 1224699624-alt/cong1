"""Delivery audit for the single-seed matched C00/CS RAM experiment."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    load=lambda name:json.loads((r/name).read_text())
    lock=load('RUN_LOCK.json');before=load('BEFORE_VAL_LOCK.json');v2=load('PADDED_EVALUATION_LOCK_v2.json')
    train={arm:load(f'results/TRAIN_{arm}/TRAIN_RESULT.json') for arm in ('C00','CS')}
    padded={s:load('PADDED_HOLDOUT_OFFICIAL.json' if s=='HOLDOUT' else 'PADDED_VAL_OFFICIAL_v2.json') for s in ('HOLDOUT','VAL')}
    cropped={s:load(f'results/EVAL_{s}/OFFICIAL.json') for s in ('HOLDOUT','VAL')}
    checks={}
    checks['lock_status']=lock['status']=='LOCKED_BEFORE_TRAIN'
    checks['training_lock_identical_to_beforeval']=sha(r/'RUN_LOCK.json')==before['run_lock_sha256']==v2['training_lock_sha256']
    checks['C0_checkpoint_SHA']=sha('outputs/whitebox_adam_20260922/fresh_adam/diagnostic_final.pth')==lock['c0_sha256']
    checks['split_manifest_SHA']=sha('outputs/ram_highres_readout_20260928/PROBE_SPLIT_MANIFEST_v2_PATIENT.json')==lock['split_manifest_sha256']
    checks['runner_code_SHA']=sha('scripts/run_ram_so_c00_cs_pair.py')==lock['runner_sha256']
    checks['corrector_code_SHA']=sha('scripts/ram_so_conditional_corrector.py')==lock['corrector_sha256']
    checks['evaluator_code_SHA']=sha('scripts/evaluate_r329_ram_official_metrics_val.py')==lock['evaluator_sha256']
    checks['geometry_code_SHA']=sha('scripts/train_r330_ram_pair_surface_volume_prior.py')==lock['R330_geometry_sha256']
    checks['S_494_local_remote_before']=load('S_MAP_PARITY.json')['status']=='PASS' and load('S_MAP_PARITY.json')['checked_maps']==494
    checks['S_494_local_remote_after']=load('S_MAP_PARITY_AFTER.json')['status']=='PASS' and load('S_MAP_PARITY_AFTER.json')['checked_maps']==494
    checks['H0_153_remote_SHA']=load('H0_CACHE_PARITY.json')['status']=='PASS' and load('H0_CACHE_PARITY.json')['checked_H0_files']==153
    for arm,t in train.items():
        ck=r/'results'/f'TRAIN_{arm}'/'final.pth'
        checks[arm+'_512_success']=t['status']=='COMPLETE' and t['steps']==512
        checks[arm+'_checkpoint_SHA']=sha(ck)==t['checkpoint_sha256']==before['checkpoints'][arm]==v2['checkpoint_sha256'][arm]
        checks[arm+'_old_frozen']=t['old_model_updates']==0 and len(t['C0_model_refiner_digests'])==2
        payload=torch.load(ck,map_location='cpu',weights_only=False)
        checks[arm+'_head_only_checkpoint']=set(payload)=={'head','optimizer','arm','steps','C0_sha256','S_manifest_sha256','run_lock_sha256'}
        checks[arm+'_Adam512']=all(float(v['step'])==512 for v in payload['optimizer']['state'].values())
        traces=[json.loads(x) for x in (r/'results'/f'TRAIN_{arm}'/'TRAINING_TRACE.jsonl').read_text().splitlines()]
        checks[arm+'_actual_trace']=traces[0]['step']==1 and traces[-1]['step']==512 and all(x['successful_step'] and np.isfinite(x['loss']) for x in traces)
    checks['same_head_initial_state']=train['C00']['head_initial_state_sha256']==train['CS']['head_initial_state_sha256']
    checks['same_512_batch_sequence']=train['C00']['sequence_sha256']==train['CS']['sequence_sha256'] and train['C00']['case_and_batch_sequence']==train['CS']['case_and_batch_sequence']
    checks['holdout_before_val']=sha(r/'results/EVAL_HOLDOUT/OFFICIAL.json')==before['holdout_sha256']
    for subset,count in [('HOLDOUT',84),('VAL',69)]:
        p,c=padded[subset],cropped[subset]
        checks[subset+'_matched_cases']=p['cases']==c['cases']==count and all(len(p['per_case'][x])==count for x in ('C0','C00','CS'))
        checks[subset+'_same_checkpoints']=p['checkpoints']==c['checkpoint_sha256']
        checks[subset+'_no_test']=p['test_used'] is False and c['test_used'] is False
        checks[subset+'_pair_count_equal']=len({p['official'][x]['overlap_pair_count'] for x in ('C0','C00','CS')})==1
        checks[subset+'_paired_H0_C0_zero']=sum(x['fixed_H0_member_recovered'] for x in p['per_case']['C0'])==0
    checks['current_host_legacy_C0_exact']=padded['VAL']['official']['C0']==load('LEGACY_COLLECT_PARITY.json')['step2']
    checks['padded_v2_code_SHA']=sha('scripts/audit_ram_so_padded_official_v2.py')==v2['code_sha256']
    checks['same_host_legacy_SHA']=sha(r/'LEGACY_COLLECT_PARITY.json')==v2['current_host_legacy_sha256']
    checks['padded_v2_val_checkpoint_SHA']=all(padded['VAL']['checkpoints'][arm]==v2['checkpoint_sha256'][arm] for arm in ('C00','CS'))
    checks['padded_v2_same_host_flag']=padded['VAL']['C0_same_host_legacy_exact'] is True
    checks['historic_drift_disclosed']=padded['VAL']['C0_historic_val_exact'] is False and load('LEGACY_COLLECT_PARITY.json')['historic_equal'] is False
    checks['capture_five_case_exact']=load('CAPTURE_CONTEXT_PARITY.json')['status']=='PASS'
    checks['direct_S8_readonly']=load('DIRECT_S_EFFECT_HOLDOUT8.json')['optimizer_steps']==0
    checks['budget_under2h']=int((r/'RUN_WALL_SECONDS.txt').read_text())+300+455+220+120 <7200
    checks['no_test_artifacts']=not any('TEST' in p.name for p in r.glob('*.json') if p.name not in ('TEST_RESULTS.json',))
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'failed':[k for k,v in checks.items() if not v],
            'test_used':False,'new_full_four_arm_training':'NOT_RUN',
            'original_run_wall_seconds':int((r/'RUN_WALL_SECONDS.txt').read_text())}
    (r/'DELIVERY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'failed':result['failed']}))
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
