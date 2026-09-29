"""Matched identity-context pair integrity and prespecified performance gate."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'outputs/ram_identity_causal_20260922'


def main():
    arms=['scrambled','canonical'];traces={};val={};results={};audits={}
    for arm in arms:
        path=ROOT/(arm+'_scale1024')
        results[arm]=json.loads((path/'RESULT.json').read_text());assert results[arm]['updates']==850
        val[arm]=json.loads((path/'FINAL_VALIDATION.json').read_text())
        traces[arm]=[json.loads(x) for x in (path/'TRAINING_TRACE.jsonl').read_text().splitlines()]
        audits[arm]=json.loads((path/'PARAMETER_AUDIT.json').read_text())
        assert len(traces[arm])==850 and all(not x['AMP_skipped'] for x in traces[arm])
    a,b=[traces[x] for x in arms]
    assert all(x['batch_sha256']==y['batch_sha256'] for x,y in zip(a,b))
    assert a[0]['loss']==b[0]['loss'];assert val[arms[0]]['cases']==val[arms[1]]['cases']
    initial_sha=[]
    for arm in arms:
        initial_sha.append(json.loads((ROOT/(arm+'_scale1024')/'INITIAL_PARAMETER_SHA.json').read_text())['parameters'])
    assert initial_sha[0]==initial_sha[1]
    for arm in arms:
        for row in traces[arm]:
            assert row['amp_scale_before']==1024. and row['amp_scale_after']==1024.
            assert all(g['nonfinite_elements']==0 for g in row['gradient_optimizer_receives'].values())
            for name,u in row['actual_parameter_updates'].items():
                expected=row['step'] if name=='refiner.relation_weight' else row['step']+100
                assert u['optimizer_step_range']==[expected,expected]
        assert set(x['case'] for x in traces[arm][:425])==set(x['case'] for x in traces[arm][425:])
        assert len(set(x['case'] for x in traces[arm]))==425
    base=json.loads((ROOT.parent/'whitebox_adam_20260922/fresh_adam/FINAL_VALIDATION.json').read_text())['flat']['step2']
    av,bv=[val[x]['flat']['step2'] for x in arms];delta={k:bv[k]-av[k] for k in av}
    rows=[]
    for stage in ['step0','step1','step2']:
        for k in val['scrambled']['flat'][stage]:
            x,y=[val[arm]['flat'][stage][k] for arm in arms]
            rows.append({'stage':stage,'metric':k,'scrambled':x,'canonical':y,'canonical_minus_scrambled':y-x,'same_initial_sha':'ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384','updates_each':850,'seed':4421,'train425_val69':'unchanged','evaluator':'R329official','only_factor':'stable_vs_per_image_scrambled_member_identity','new_relation_params':2352,'relation_lr':8e-5,'decoder_lr':7e-6,'refiner_lr':2.1912418526906397e-7,'sampling':'two_complete_passes','cpu_threads':64,'AMP_initial_scale':1024,'failed_and_replay_updates_excluded_from_pair':322})
    with (ROOT/'RUN_CONFIG_AND_RESULT_DIFF.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    gate=delta['overlap_dsc']>=.01 and bv['overlap_dsc']-base['overlap_dsc']>=.01 and all(bv[k]>=max(av[k],base[k]) for k in ['overall_dsc','overall_iou','overlap_nsd_2px','pair_dsc'])
    case_delta=[]
    for case in val['canonical']['cases']:
        x,y=[val[arm]['per_case_official'][case]['step2']['overlap_region_metrics']['dsc'] for arm in arms]
        case_delta.append({'case':case,'scrambled_overlap_dsc':x,'canonical_overlap_dsc':y,'delta':y-x})
    diffs=np.asarray([x['delta'] for x in case_delta]);rng=np.random.default_rng(3324421)
    bootstrap=np.mean(diffs[rng.integers(0,len(diffs),(2000,len(diffs)))],1)
    paired={'mean':float(diffs.mean()),'improved_cases':int((diffs>0).sum()),'worse_cases':int((diffs<0).sum()),'tied_cases':int((diffs==0).sum()),'case_bootstrap_95pct_interval':np.quantile(bootstrap,[.025,.975]).tolist(),'limits':'descriptive case bootstrap; not confirmed independent patients or multi-seed evidence','cases':case_delta}
    (ROOT/'PAIRED_CASE_ANALYSIS.json').write_text(json.dumps(paired,indent=2))
    internal={}
    for arm in arms:
        tt=traces[arm]
        internal[arm]={'perpass_mean_original_training_loss':[float(np.mean([x['loss'] for x in part])) for part in [tt[:425],tt[425:]]],
            'final_relation_norm':tt[-1]['actual_parameter_updates']['refiner.relation_weight']['cumulative_norm'],
            'final_group_updates':tt[-1]['actual_parameter_updates'],
            'context_snapshots':{str(k):tt[k-1]['relation_context'] for k in [1,425,850]},
            'nonfinite_gradients':sum(g['nonfinite_elements'] for x in tt for g in x['gradient_optimizer_receives'].values()),
            'successful_updates':tt[-1]['successful_optimizer_steps']}
    (ROOT/'INTERNAL_TRAINING_COMPARISON.json').write_text(json.dumps(internal,indent=2))
    result={'status':'COMPLETE','effect_gate_1pp':gate,'scrambled':av,'canonical':bv,'delta':delta,'common_initial_anchor':base,'same850batchhashes':True,'same_initial_loss':True,'member_selection_difference_steps':sum(x['supervision']['indices']!=y['supervision']['indices'] for x,y in zip(a,b)),'new_optimizer_steps':1700,'seconds':sum(x['seconds'] for x in results.values()),'test_used':False,'limits':'one seed two-pass pilot; scrambled control tests stable semantics under this coding intervention, not every capacity-only architecture or convergence; mature anchor has fewer updates and is not causal continuation control'}
    (ROOT/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'TRAINING_TRACE.jsonl').open('w') as f:
        for arm in arms:
            for row in traces[arm]:f.write(json.dumps({'arm':arm,**row})+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
