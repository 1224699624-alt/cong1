import csv,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'outputs/ram_source_probe_20260922'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    arms=['pilot_reference','pilot_source_aux'];traces={};valid={};results={}
    for arm in arms:
        results[arm]=load(ROOT/arm/'RESULT.json');assert results[arm]['status']=='COMPLETE' and results[arm]['updates']==100
        traces[arm]=[json.loads(x) for x in (ROOT/arm/'TRAINING_TRACE.jsonl').read_text().splitlines()];assert len(traces[arm])==100
        valid[arm]=load(ROOT/arm/'FINAL_VALIDATION.json')
        for row in traces[arm]:
            assert not row['AMP_skipped']
            for g in row['actual_parameter_updates'].values():assert g['optimizer_step_range']==[100+row['step']]*2
            expected_extra=row['source_auxiliary']['weighted_loss'] if arm=='pilot_source_aux' else 0.
            assert math.isclose(row['loss']-row['original_loss_before_source_aux'],expected_extra,abs_tol=1e-7)
    a,b=[traces[x] for x in arms]
    assert all(x['batch_sha256']==y['batch_sha256'] for x,y in zip(a,b))
    assert a[0]['original_loss_before_source_aux']==b[0]['original_loss_before_source_aux']
    assert valid[arms[0]]['cases']==valid[arms[1]]['cases'] and len(valid[arms[0]]['cases'])==69
    rows=[]
    for stage in ['step0','step1','step2']:
        for key in valid[arms[0]]['flat'][stage]:
            av=valid[arms[0]]['flat'][stage][key];bv=valid[arms[1]]['flat'][stage][key]
            rows.append({'stage':stage,'metric':key,'reference':av,'source_aux':bv,'delta':bv-av,'split':'RAMval69','evaluator':'R329_official','updates_each':100,'initial_optimizer_step':100,'final_optimizer_step':200})
    with (ROOT/'RUN_CONFIG_AND_RESULT_DIFF.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    combined=[]
    for arm in arms:
        combined.extend({'arm':arm,**row} for row in traces[arm])
    (ROOT/'PILOT_TRAINING_TRACE.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in combined))
    af,bf=[valid[x]['flat']['step2'] for x in arms];delta={k:bf[k]-af[k] for k in af}
    gate=delta['overlap_dsc']>0 and all(delta[k]>=0 for k in ['overall_dsc','overall_iou','overlap_nsd_2px'])
    summary={'status':'COMPLETE','effect_gate':gate,'primary_dataset':'RAM','additional_dataset':'TSRS','training_updates':200,'local_probe_updates':6,'total_new_optimizer_updates':206,
        'reference':af,'source_aux':bf,'delta':delta,'batch_hashes_equal':True,
        'member_selection_difference_steps':sum(x['supervision']['indices']!=y['supervision']['indices'] for x,y in zip(a,b)),
        'source_aux_member_exposures':sum(x['source_auxiliary']['unselected_hard_members'] for x in b),
        'source_aux_nonempty_steps':sum(x['source_auxiliary']['unselected_hard_members']>0 for x in b),
        'pair_seconds':sum(results[x]['seconds'] for x in arms),'test_used':False,'scope':'single-seed100-update controlled pilot; not convergence/stability proof'}
    (ROOT/'PAIRED_ACCEPTANCE.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    (ROOT/'PILOT_ARTIFACT_TESTS.json').write_text(json.dumps({'status':'PASS','checks':['both100successful','same100batchhashes','sameinitialoriginal_loss','pairedoptimizer100->200','lossdelta equals exactlythe intended auxiliary','same69valcases'],'effect_gate_independent_of_tests':gate},indent=2))


if __name__=='__main__':main()
