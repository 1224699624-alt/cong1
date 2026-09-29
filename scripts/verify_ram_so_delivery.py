import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    load=lambda p:json.loads((r/p).read_text())
    lock=load('SO_PRIOR_RUN_LOCK.json');cal=load('results/CALIBRATION_THRESHOLDS.json')
    trains={x:load(f'results/{x}/TRAIN_RESULT.json') for x in ('O_REF_CONTINUE','O_HARD_WEIGHT')}
    checks={}
    for arm,t in trains.items():
        checks[arm+'_2048_steps']=t['status']=='COMPLETE' and t['updates']==2048
        ck=r/'results'/arm/'final.pth';checks[arm+'_SHA']=sha(ck)==t['checkpoint_sha256']
        payload=torch.load(ck,map_location='cpu',weights_only=False)
        checks[arm+'_C0_not_in_generator']=payload['input_dependencies']==['image'] and 'refiner' not in payload
        checks[arm+'_optimizer2048']=all(float(x['step'])==2048 for x in payload['optimizer']['state'].values())
        trace=[json.loads(x) for x in (r/'results'/arm/'SO_PRIOR_TRACE.jsonl').read_text().splitlines()]
        checks[arm+'_trace']=len(trace)==83 and trace[-1]['step']==2048 and all(x['success'] and np.isfinite(x['loss_used']) for x in trace)
    checks['matched_sequence']=trains['O_REF_CONTINUE']['sequence_sha256']==trains['O_HARD_WEIGHT']['sequence_sha256']
    checks['patient_split_reused']=sha('outputs/ram_highres_readout_20260928/PROBE_SPLIT_MANIFEST_v2_PATIENT.json')==lock['split_manifest_sha256']
    checks['data988']=load('DATA_PARITY.json')['checked_files']==988 and load('DATA_PARITY.json')['mismatch_count']==0
    before=load('results/BEFORE_VAL_LOCK.json');checks['threshold_lock']=sha(r/'results/CALIBRATION_THRESHOLDS.json')==before['threshold_sha256']
    checks['holdout_lock']=sha(r/'results/HOLDOUT_RESULT.json')==before['holdout_result_sha256']
    for arm in trains:checks[arm+'_beforeval_SHA']=sha(r/'results'/arm/'final.pth')==before['checkpoint_sha256'][arm]
    for subset,expected in [('CALIBRATION',85),('HOLDOUT',84),('VAL',69)]:
        res=load(f'results/{subset}_RESULT.json')
        checks[subset+'_cases']=res['cases']==expected and all(len(x)==expected for x in res['per_case'].values())
        checks[subset+'_thresholds_unchanged']=res['thresholds']==cal['thresholds']
        checks[subset+'_same_denominators']=len({json.dumps(x['denominators'],sort_keys=True) for x in res['arms'].values()})==1
        checks[subset+'_no_test']=res['test_used']==False and res['segmentation_gain']=='NOT_TESTED'
    res=load('results/CALIBRATION_RESULT.json')
    checks['cal_exact_same_single_FP']=all(x['matched']['single_fp']==cal['allowed_fp'] for x in res['arms'].values())
    end=load('results/EVAL_COMPLETE.json')
    checks['C0_original_SHA_unchanged']=end['C0_sha256_after']==lock['C0_sha256']
    checks['O_original_SHA_unchanged']=end['original_O_sha256_after']==lock['O_sha256']
    corrector=load('CORRECTOR_REAL_RESULT.json')
    checks['real_corrector8_pass']=corrector['status']=='PASS' and len(corrector['cases'])==8
    checks['old_states_preserved']=corrector['old_state_digest_before']==corrector['old_state_digest_after']
    checks['head_no_update']=corrector['head_state_unchanged'] and corrector['optimizer_steps']==0
    checks['real_zero_allarms']=all(not any(v.values()) for case in corrector['cases'] for v in case['arms'].values())
    checks['first_backward_no_old_grad']=all(corrector['cases'][0]['first_backward'][k] for k in ('old_model_grad_none','old_refiner_grad_none','generator_grad_none'))
    checks['only737_new_params']=corrector['head_parameter_n']==737
    s=load('S_SOURCE_AUDIT.json')
    checks['S_RAM_source']=(s['source']['dataset_root'].replace('\\','/')=='G:/gutou/RAM-W600' and
                             s['s_value_status']=='CONDITIONAL_VALUE_PENDING')
    smap=load('S_MAP_PARITY.json')
    checks['S_local_remote19_maps_match']=smap['status']=='PASS' and smap['checked_maps']==19 and not smap['mismatches']
    checks['under2h']=int((r/'PAIR_WALL_SECONDS.txt').read_text())+60 <7200
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'test_used':False,
            'new_segmentation_training':'NOT_RUN','O_pair_seconds':int((r/'PAIR_WALL_SECONDS.txt').read_text())}
    (r/'DELIVERY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v]}))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
