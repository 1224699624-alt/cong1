"""Accept/reject the prespecified guard pilot without replacing historical scores."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'outputs'


def main():
    root=ROOT/'ram_negative_guard_20260922'
    paths={'original':ROOT/'ram_source_probe_20260922/pilot_reference','positive':ROOT/'ram_source_probe_20260922/pilot_source_aux','guard':root/'pilot_guard'}
    val={k:json.loads((p/'FINAL_VALIDATION.json').read_text()) for k,p in paths.items()}
    traces={k:[json.loads(x) for x in (p/'TRAINING_TRACE.jsonl').read_text().splitlines()] for k,p in paths.items()}
    a,b=traces['positive'],traces['guard'];assert len(a)==len(b)==100
    assert all(x['batch_sha256']==y['batch_sha256'] for x,y in zip(a,b))
    assert a[0]['original_loss_before_source_aux']==b[0]['original_loss_before_source_aux']
    assert a[0]['source_auxiliary']==b[0]['source_auxiliary']
    assert all(val[k]['cases']==val['original']['cases'] for k in val)
    for x in b:
        assert not x['AMP_skipped']
        assert all(g['optimizer_step_range']==[100+x['step']]*2 for g in x['actual_parameter_updates'].values())
        assert math.isclose(x['loss'],x['original_loss_before_source_aux']+x['source_auxiliary']['weighted_loss']+x['negative_guard']['weighted_loss'],abs_tol=1e-7)
    assert b[0]['observation_equivalence']['gradient_max_abs']==0
    metrics={k:v['flat']['step2'] for k,v in val.items()};o,p,g=[metrics[k] for k in ['original','positive','guard']]
    gate=g['overlap_nsd_2px']>p['overlap_nsd_2px'] and g['overall_nsd_2px']>p['overall_nsd_2px'] and g['overlap_dsc']>o['overlap_dsc'] and g['overall_dsc']>=o['overall_dsc'] and g['overall_iou']>=o['overall_iou']
    rows=[]
    for stage in ['step0','step1','step2']:
        for metric in val['original']['flat'][stage]:
            values={k:val[k]['flat'][stage][metric] for k in val}
            rows.append({'stage':stage,'metric':metric,**values,'guard_minus_positive':values['guard']-values['positive'],'guard_minus_original':values['guard']-values['original'],'split':'RAMval69','new_updates_each':100,'initial_optimizer_step':100,'final_optimizer_step':200,'only_variable_vs_positive':'local_negative_source_guard','initial_sha256':'ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384','refiner_lr':2.1912418526906397e-7,'decoder_lr':7e-6,'optimizer':'paired_AdamW','scheduler':'constant','seed':4421,'train_cases':425,'cpu_threads':64,'evaluator':'unchanged_R329official','threshold':.5,'selection':'fixed100'})
    with (root/'RUN_CONFIG_AND_RESULT_DIFF.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'status':'COMPLETE','artifact_checks':'PASS','effect_gate':gate,'metrics':metrics,'new_optimizer_steps':106,'reused_reference':str(paths['positive']),'same100batchhashes':True,'same_initial_native_and_positive_losses':True,'member_selection_difference_steps':sum(x['supervision']['indices']!=y['supervision']['indices'] for x,y in zip(a,b)),'guard_member_exposures':sum(x['negative_guard']['negative_members'] for x in b),'guard_nonempty_steps':sum(x['negative_guard']['negative_members']>0 for x in b),'test_used':False,'scope':'one seed100-update pilot; not convergence or final model'}
    (root/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
    with (root/'TRAINING_TRACE.jsonl').open('w') as f:
        for arm in ['positive','guard']:
            for row in traces[arm]:f.write(json.dumps({'arm':arm,'reused_historical_trace':arm=='positive',**row})+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
