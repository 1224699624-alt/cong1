"""Verify fixed-schedule 2x2 completion and write original-metric contrasts."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

ARMS=('SRC_F0S0','SRC_F0S1','SRC_F1S0','SRC_F1S1')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--schedule',type=Path,required=True);a=p.parse_args()
    root=a.root;schedule=json.loads(a.schedule.read_text());n=len(schedule['steps']);assert n>0
    oracle=json.loads(a.oracle.read_text());assert oracle['status']=='COMPLETE' and oracle['test_used'] is False
    metrics={};traces={};manifests={};valids={}
    for arm in ARMS:
        base=root/arm;manifest=json.loads((base/'run_manifest.json').read_text());assert manifest['status']=='COMPLETE' and manifest['optimizer_steps']==n,(arm,manifest.get('status'))
        lines=[json.loads(x) for x in (base/'TRAINING_TRACE.jsonl').read_text().splitlines()];assert len(lines)==n
        assert all(x['optimizer_steps']['successful']==x['step'] and not x['AMP']['skipped'] for x in lines)
        assert all(x['AMP']['scale_before']==1024. and x['AMP']['scale_after']==1024. for x in lines)
        assert all(x['fixed_indices']==step['indices'] and x['case']==step['case'] and x['batch_sha256']==step['batch_sha256'] for x,step in zip(lines,schedule['steps']))
        assert all(x['h0_exposure']['valid_total']==x['h0_exposure']['total'] for x in lines)
        if arm.endswith('S0'):
            assert all(math.isclose(x['total_loss'],x['native_loss'],abs_tol=1e-7) for x in lines)
        else:
            assert all(math.isclose(x['total_loss'],x['native_loss']+x['extra_source_loss'],abs_tol=1e-6) for x in lines)
        val=json.loads((base/'FINAL_VALIDATION.json').read_text());assert val['status']=='COMPLETE'
        output=val['result'];assert len(output['cases'])==69
        metrics[arm]=output['flat'];traces[arm]=lines;manifests[arm]=manifest;valids[arm]=output
    assert all(valids[x]['cases']==valids[ARMS[0]]['cases'] for x in ARMS)
    start_native=[traces[x][0]['native_loss'] for x in ARMS]
    assert max(start_native)-min(start_native)<=1e-5,start_native
    base_oracle=oracle['splits']['val']['flatten']['step2']
    rows=[]
    for stage in ('step0','step1','step2'):
        for metric in metrics[ARMS[0]][stage]:
            v={arm:metrics[arm][stage][metric] for arm in ARMS}
            row={'stage':stage,'metric':metric,**v,
                 'S_effect_F0':v['SRC_F0S1']-v['SRC_F0S0'],'S_effect_F1':v['SRC_F1S1']-v['SRC_F1S0'],
                 'F_effect_S0':v['SRC_F1S0']-v['SRC_F0S0'],'F_effect_S1':v['SRC_F1S1']-v['SRC_F0S1'],
                 'interaction':v['SRC_F1S1']-v['SRC_F1S0']-v['SRC_F0S1']+v['SRC_F0S0'],
                 'C0_zero_update_reference':base_oracle[metric] if stage=='step2' else None,
                 'updates_each':n,'dataset':'RAM_val69','evaluator':'original_R329','threshold':.5,
                 'shared_start':'C0_fresh100','seed':schedule['seed'],'test_used':False}
            rows.append(row)
    with (root/'FACTORIAL_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    ref=metrics['SRC_F0S0']['step2'];gates={}
    for arm in ARMS[1:]:
        curr=metrics[arm]['step2'];gates[arm]={'overlap_dsc_improved':curr['overlap_dsc']>ref['overlap_dsc'],
            'overall_dsc_preserved':curr['overall_dsc']>=ref['overall_dsc'],
            'overall_iou_preserved':curr['overall_iou']>=ref['overall_iou'],
            'overlap_nsd_preserved':curr['overlap_nsd_2px']>=ref['overlap_nsd_2px'],
            'pair_dsc_preserved':curr['pair_dsc']>=ref['pair_dsc'],
            'overlap_dsc_gain_1pp':curr['overlap_dsc']-ref['overlap_dsc']>=.01}
        gates[arm]['balanced_gate']=all(gates[arm][k] for k in ['overlap_dsc_improved','overall_dsc_preserved','overall_iou_preserved','overlap_nsd_preserved','pair_dsc_preserved'])
    case_deltas={}
    for arm in ARMS[1:]:
        values=[]
        for case in valids[ARMS[0]]['cases']:
            a0=valids[ARMS[0]]['per_case_official'][case]['step2']['overlap_region_metrics']['dsc']
            a1=valids[arm]['per_case_official'][case]['step2']['overlap_region_metrics']['dsc']
            values.append({'case':case,'F0S0':a0,'arm':a1,'delta':a1-a0})
        arr=np.asarray([x['delta'] for x in values]);rng=np.random.default_rng(4421)
        boot=np.mean(arr[rng.integers(0,len(arr),(2000,len(arr)))],axis=1)
        case_deltas[arm]={'mean':float(arr.mean()),'improved_cases':int((arr>0).sum()),'worse_cases':int((arr<0).sum()),
            'case_bootstrap_interval_95pct':np.quantile(boot,[.025,.975]).tolist(),
            'scope':'descriptive paired cases, patient IDs/seed independence unverified','cases':values}
    (root/'PAIRED_CASE_RESULTS.json').write_text(json.dumps(case_deltas,indent=2))
    summary={'status':'COMPLETE','arms':list(ARMS),'updates_each':n,'C0_zero_update_reference':base_oracle,
        'step2':{arm:metrics[arm]['step2'] for arm in ARMS},'initial_native_loss_range':[min(start_native),max(start_native)],
        'all_input_hashes_and_fixed_members_match':True,'optimizer_steps':{arm:manifests[arm]['optimizer_steps'] for arm in ARMS},
        'GPU_wall_seconds_four_arms':sum(manifests[arm]['seconds'] for arm in ARMS),
        'gates':gates,'test_used':False,'scope':'one seed fixed selector; no test or convergence claim'}
    (root/'FACTORIAL_ACCEPTANCE.json').write_text(json.dumps(summary,indent=2))
    with (root/'TRAINING_TRACE.jsonl').open('w') as f:
        for arm in ARMS:
            for x in traces[arm]:f.write(json.dumps({'arm':arm,**x})+'\n')
    print(json.dumps({'status':'COMPLETE','updates_each':n,'overlap_dsc':{arm:metrics[arm]['step2']['overlap_dsc'] for arm in ARMS},'gates':gates},indent=2))


if __name__=='__main__':main()
