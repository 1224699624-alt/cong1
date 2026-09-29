#!/usr/bin/env python3
"""Independent local integrity checks for the frozen RAM P0/P1 delivery."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--c0',type=Path,required=True)
    a=p.parse_args()
    root=a.root
    split=json.loads((root/'PROBE_SPLIT_MANIFEST_v2_PATIENT.json').read_text())
    extract=json.loads((root/'SAMPLE_EXTRACTION_RESULT_v1.json').read_text())
    extract_manifest=json.loads((root/'EXTRACTION_MANIFEST_v1.json').read_text())
    stream=json.loads((root/'stream_v3/STREAM_RESULT.json').read_text())
    cal=json.loads((root/'NATURAL_Q_probe_calibration_v1.json').read_text())
    before_val=json.loads((root/'stream_v3/HOLDOUT_COMPLETE_BEFORE_VAL.json').read_text())
    lock=json.loads((root/'HIGHRES_STREAM_EVAL_LOCK_v3.json').read_text())
    fit=json.loads((root/'PROBE_FIT_RESULT_v1.json').read_text())
    checks={}
    checks['C0_checkpoint_SHA']=sha(a.c0)==lock['C0_sha256']
    checks['extract_frozen']=extract_manifest['C0_frozen_state_preserved'] and extract_manifest['C0_updates']==0
    checks['stream_frozen']=stream['C0_frozen_state_preserved'] and stream['C0_updates']==0
    checks['H0_exact_train_val']=extract['train_H0_members']==228695 and extract['val_H0_members']==71335
    checks['patient_disjoint']=all(x==0 for x in split['cross_patient_overlap'].values())
    checks['actual_dataset_source']=split['actual_dataset_root'].replace('\\','/').lower().endswith('g:/gutou/ram-w600')
    checks['same_steps']=fit['arms']['P0']['steps']==fit['arms']['P1']['steps']==1300
    checks['same_capacity']=fit['trainable_parameters']==3361 and fit['same_initial_state'] and fit['same_sample_schedule']
    checks['calibration_threshold_unchanged']=lock['thresholds']==cal['thresholds_from_calibration']==before_val['thresholds']
    checks['full_Q_identical_arms']=all(
        stream['results'][subset]['arms']['P0']['q_exact']==stream['results'][subset]['arms']['P1']['q_exact']
        for subset in ('probe_holdout','val'))
    checks['full_Q_val_H0']=stream['results']['val']['arms']['P0']['h0_exact']==71335
    checks['val_after_holdout_lock']=before_val['status']=='HOLDOUT_COMPLETE_BEFORE_VAL' and not before_val['val_used_for_choice']
    checks['no_test_TSRS']=not split['test_files_accessed'] and not stream['test_used'] and not stream['TSRS_used']
    checks['remote_data_parity']=json.loads((root/'REMOTE_DATA_PREFLIGHT.json').read_text())['mismatch_count']==0
    sample_hashes={}
    case_sets={}
    for subset in ('probe_fit','probe_calibration','probe_holdout','val'):
        path=root/'run_v1'/f'SAMPLES_{subset}.npz'
        sample_hashes[subset]=sha(path)
        checks[f'{subset}_cache_hash']=sample_hashes[subset]==extract['subsets'][subset]['cache_sha256']
        with np.load(path,allow_pickle=False) as z:
            case_sets[subset]=set(z['case'].astype(str))
            checks[f'{subset}_B_H_shape']=z['B'].shape[1]==57 and z['H'].shape[1]==32
            checks[f'{subset}_same_sample_rows']=len(z['B'])==len(z['H'])==len(z['label'])==len(z['member'])
    checks['sample_case_sets_exact']=all(case_sets[k]==set(split['groups'][k]) for k in ('probe_fit','probe_calibration','probe_holdout')) and case_sets['val']==set(split['val_cases'])
    checks['sample_case_sets_disjoint']=all(not case_sets[a]&case_sets[b] for a,b in
        (('probe_fit','probe_calibration'),('probe_fit','probe_holdout'),
         ('probe_calibration','probe_holdout'),('probe_fit','val'),
         ('probe_calibration','val'),('probe_holdout','val')))
    with np.load(root/'run_v1/SAMPLES_probe_fit.npz',allow_pickle=False) as fit_rows, \
         np.load(root/'run_v1/NORMALIZATION_AND_PROJECTION.npz',allow_pickle=False) as transform:
        checks['normalization_fit_only']=bool(np.allclose(
            fit_rows['B'].astype(np.float64).mean(0).astype(np.float32),transform['b_mean'],atol=1e-6)
            and np.allclose(fit_rows['H'].astype(np.float64).mean(0).astype(np.float32),transform['h_mean'],atol=1e-6))
    for arm in ('P0','P1'):
        path=root/'run_v1'/f'{arm}_PROBE.pth'
        payload=torch.load(path,map_location='cpu',weights_only=False)
        checks[f'{arm}_checkpoint_lock']=payload['steps']==1300 and payload['C0_sha256']==lock['C0_sha256']
        checks[f'{arm}_checkpoint_sha']=sha(path)==before_val[f'{arm}_checkpoint_sha256']
    trace=[json.loads(line) for line in (root/'run_v1/PROBE_TRAINING_TRACE.jsonl').open() if line.strip()]
    checks['trace_2600_successful']=len(trace)==2600 and all(row['success'] for row in trace)
    checks['round_under_two_GPU_hours']=lock['spent_before_stream_seconds']+stream['elapsed_seconds']<7200
    result={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
            'all_pass':all(checks.values()),'C0_sha256':lock['C0_sha256'],
            'patient_split_counts':split['counts'],'sample_cache_sha256':sample_hashes,
            'gpu_seconds_total':lock['spent_before_stream_seconds']+stream['elapsed_seconds'],
            'gpu_hours_total':(lock['spent_before_stream_seconds']+stream['elapsed_seconds'])/3600,
            'test_used':False,'C0_updates':0,
            'probe_steps_each':1300}
    (root/'FREEZE_LEAKAGE_AND_DELIVERY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'failed':[k for k,v in checks.items() if not v],
                      'checks':len(checks),'gpu_hours':result['gpu_hours_total']}))
    if not all(checks.values()):raise SystemExit(1)


if __name__=='__main__':main()
