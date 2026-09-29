"""Summarize the locked RAM optimizer-state pair from completed raw artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/whitebox_adam_20260922'));a=p.parse_args();out=a.output
    arms=['fresh_adam','restored_adam'];traces={};results={};validations={};locks={}
    for arm in arms:
        results[arm]=load(out/arm/'RESULT.json')
        assert results[arm]['status']=='COMPLETE' and results[arm]['updates']==100
        traces[arm]=[json.loads(x) for x in (out/arm/'TRAINING_TRACE.jsonl').read_text().splitlines()]
        assert len(traces[arm])==100
        locks[arm]=load(out/('DIAGNOSTIC_RUN_LOCK_'+arm+'.json'))
        validations[arm]={12:load(out/arm/'VALIDATION_step0012.json'),100:load(out/arm/'FINAL_VALIDATION.json')}
    fresh=traces['fresh_adam'];restored=traces['restored_adam']
    batch_match=all(x['batch_sha256']==y['batch_sha256'] for x,y in zip(fresh,restored))
    assert batch_match
    mapping=load(out/'restored_adam/ADAM_STATE_MAPPING.json');assert mapping['status']=='PASS'
    rows=[];per_case=[]
    for step in [12,100]:
        av,bv=[validations[arm][step] for arm in arms]
        assert av['cases']==bv['cases'] and len(av['cases'])==69
        for stage in ['step0','step1','step2']:
            for arm,v in [('fresh_adam',av),('restored_adam',bv)]:
                for metric,value in v['flat'][stage].items():
                    rows.append({'arm':arm,'updates':step,'stage':stage,'metric':metric,'value':value,
                        'delta_restored_minus_fresh':bv['flat'][stage][metric]-av['flat'][stage][metric],
                        'dataset':'RAM-W600','split':'val69','evaluator':'R329_official','threshold':.5,
                        'initialization_sha256':locks[arm]['file_hashes'][locks[arm]['r332']],
                        'optimizer_state':'fresh' if arm=='fresh_adam' else 'R332 historical moments+step only',
                        'refiner_lr':locks[arm]['refiner_lr_override'],'decoder_lr':7e-6,'seed':3321,
                        'successful_updates':traces[arm][step-1]['successful_optimizer_steps'],
                        'evidence':f'{arm}/'+('FINAL_VALIDATION.json' if step==100 else 'VALIDATION_step0012.json')})
        for case in av['cases']:
            for key in ['overall_instance_metrics','overlap_region_metrics','overlap_pair_intersection_metrics']:
                for metric in ['dsc','nsd_2px','voe','msd_px','ravd','msd_fail_rate']:
                    left=av['per_case_official'][case]['step2'][key].get(metric)
                    right=bv['per_case_official'][case]['step2'][key].get(metric)
                    per_case.append({'updates':step,'case':case,'region':key,'metric':metric,'fresh':left,'restored':right,
                        'delta_restored_minus_fresh':None if left is None or right is None else right-left})
    for filename,records in [('RUN_CONFIG_AND_RESULT_DIFF.csv',rows),('PER_CASE_STANDARD_DELTAS.csv',per_case)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    merged=[{'arm':arm,**row} for arm in arms for row in traces[arm]]
    (out/'TRAINING_TRACE.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in merged))
    af,bf=[validations[arm][100]['flat']['step2'] for arm in arms]
    effect_gate=bf['overlap_dsc']>af['overlap_dsc'] and bf['overall_dsc']>=af['overall_dsc'] and bf['overlap_nsd_2px']>=af['overlap_nsd_2px']
    norms={}
    for group in fresh[0]['actual_parameter_updates']:
        norms[group]={arm:{'first_update':traces[arm][0]['actual_parameter_updates'][group]['update_norm'],
                          'final_cumulative_offset':traces[arm][-1]['actual_parameter_updates'][group]['cumulative_norm'],
                          'first_optimizer_step_range':traces[arm][0]['actual_parameter_updates'][group]['optimizer_step_range'],
                          'last_optimizer_step_range':traces[arm][-1]['actual_parameter_updates'][group]['optimizer_step_range']} for arm in arms}
    checkpoints={}
    for arm in arms:
        path=out/arm/'diagnostic_final.pth'
        checkpoints[arm]=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else 'REMOTE_ONLY_PENDING_LOCAL_COPY'
    shadow=load(out/'SHARED_GRADIENT_PROBE.json')
    initial_grad_diff=max(abs(fresh[0]['gradient_before_clip'][k]['norm']-restored[0]['gradient_before_clip'][k]['norm'])/max(fresh[0]['gradient_before_clip'][k]['norm'],1e-30) for k in fresh[0]['gradient_before_clip'])
    summary={'status':'COMPLETE','primary_dataset':'RAM-W600','additional_dataset':'TSRS','new_training_updates':200,'new_shadow_updates':2,'new_updates_total':202,
        'batch_hashes_equal':batch_match,'initial_loss_equal':fresh[0]['loss']==restored[0]['loss'],
        'initial_gradient_summary_equal':fresh[0]['gradient_before_clip']==restored[0]['gradient_before_clip'],
        'initial_gradient_norm_max_relative_difference':initial_grad_diff,
        'shared_gradient_shadow_passed':shadow['status']=='PASS' and shadow['arms']['fresh']['gradient_sha256']==shadow['arms']['restored']['gradient_sha256'],
        'dynamic_member_selection_difference_steps':sum(x['supervision']['indices']!=y['supervision']['indices'] for x,y in zip(fresh,restored)),
        'per_arm_successful_updates':{arm:results[arm]['updates'] for arm in arms},
        'AMP_skip_counts':{arm:sum(x['AMP_skipped'] for x in traces[arm]) for arm in arms},
        'LR_constant':all(all(r['lr']==[2.1912418526906397e-7,7e-6] for r in traces[arm]) for arm in arms),
        'effect_gate':effect_gate,'effect_gate_definition':'final overlap DSC improves; overall DSC and overlap NSD do not decline versus matched fresh reference',
        'per_step_metrics':{str(step):{arm:validations[arm][step]['flat']['step2'] for arm in arms} for step in [12,100]},
        'final_delta_restored_minus_fresh':{k:bf[k]-af[k] for k in af},'actual_updates_by_group':norms,
        'checkpoint_hashes':checkpoints,'measured_pair_wall_seconds':sum(results[arm]['seconds'] for arm in arms),
        'conservative_total_seconds_upper_bound':sum(results[arm]['seconds'] for arm in arms)+120,
        'stable_gain_claim':False,'historical_replay_claim':False,'test_used':False,
        'notes':['Different historical step counters preserved for refiner and decoder.',
                 'Both optimizer groups restored together; this does not isolate refiner versus decoder moments.',
                 'New O channel moments are zero but share the tensor historical step; O remains disabled.',
                 'Sampling membership is dynamic model-dependent R332 policy; input data hashes, not member identity, are the exogenous matching gate.']}
    (out/'PAIRED_ACCEPTANCE.json').write_text(json.dumps(summary,indent=2))
    # Fixed selection rule: three highest/lowest per-case overlap DSC deltas at final.
    candidates=[r for r in per_case if r['updates']==100 and r['region']=='overlap_region_metrics' and r['metric']=='dsc' and r['delta_restored_minus_fresh'] is not None]
    candidates.sort(key=lambda r:(r['delta_restored_minus_fresh'],r['case']))
    (out/'CASE_REVIEW_SELECTION.json').write_text(json.dumps({'rule':'all69 cases ranked by final overlap DSC delta; show3worst/3best, no exclusion','worst':candidates[:3],'best':candidates[-3:]},indent=2))
    print(json.dumps({'effect_gate':effect_gate,'delta':summary['final_delta_restored_minus_fresh'],'wall_seconds':summary['measured_pair_wall_seconds']},indent=2))


if __name__=='__main__':main()
