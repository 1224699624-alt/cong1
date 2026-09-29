#!/usr/bin/env python3
"""Combine locked RAM P0/P1 outputs without changing probe/evaluator rules."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();b=a.root
    fit=json.loads((b/'PROBE_FIT_RESULT_v1.json').read_text())
    matched={s:json.loads((b/f'MATCHED_{s}_v1.json').read_text())
             for s in ('probe_fit','probe_calibration','probe_holdout')}
    matched['val']=json.loads((b/'stream_v3/MATCHED_val_STREAM.json').read_text())
    natural={'probe_calibration':json.loads((b/'NATURAL_Q_probe_calibration_v1.json').read_text()),
             'probe_holdout':json.loads((b/'stream_v3/NATURAL_Q_probe_holdout_STREAM.json').read_text()),
             'val':json.loads((b/'stream_v3/NATURAL_Q_val_STREAM.json').read_text())}
    rows=[]
    for subset in ('probe_fit','probe_calibration','probe_holdout','val'):
        for arm in ('P0','P1'):
            m=matched[subset]['arms'][arm]
            rows.append({'subset':subset,'view':'balanced_matched_diagnostic','arm':arm,
                         'candidate_count':m['n'],'positive':m['positive'],'negative':m['negative'],
                         'auc':m['auc'],'ap':m['ap'],'logloss':m['logloss'],
                         'auc_ap_exact':True,'threshold':None,
                         'tp':None,'fp':None,'precision':None,'recall':None,'fpr':None,
                         'h0_total':None,'h0_in_q':None,'h0_selected':None})
    for subset in ('probe_calibration','probe_holdout','val'):
        source=natural[subset]
        for arm in ('P0','P1'):
            n=source['arms'][arm]
            pop=source['population_exact'] if subset=='probe_calibration' else None
            rows.append({'subset':subset,'view':'natural_full_Q','arm':arm,
                'candidate_count':n.get('q_exact',pop['q_members'] if pop else None),
                'positive':n.get('positive_exact',pop['q_positive'] if pop else None),
                'negative':n.get('negative_exact',pop['q_negative'] if pop else None),
                'auc':n.get('auc',n.get('auc_hist_approx')),
                'ap':n.get('ap',n.get('ap_hist_approx')),
                'logloss':None,'auc_ap_exact':subset=='probe_calibration',
                'threshold':source['thresholds_from_calibration'][arm] if subset=='probe_calibration' else source['calibrated_thresholds'][arm],
                'tp':n.get('threshold_tp_exact',n.get('weighted_tp_estimate')),
                'fp':n.get('threshold_fp_exact',n.get('weighted_fp_estimate')),
                'precision':n.get('threshold_precision_exact',n.get('sample_precision_weighted')),
                'recall':n.get('threshold_recall_exact',n.get('sample_recall_weighted')),
                'fpr':n.get('threshold_fpr_exact',n.get('sample_fpr_weighted')),
                'h0_total':n.get('h0_exact',pop['h0_members'] if pop else None),
                'h0_in_q':n.get('h0_in_q_exact',pop['h0_in_q'] if pop else None),
                'h0_selected':n.get('h0_selected_exact',n.get('H0_full_recall_estimate')*(pop['h0_members']) if pop else None)})
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with (b/'PROBE_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    cases=pd.read_csv(b/'stream_v3/NATURAL_Q_PER_CASE_MEMBER_STREAM.csv')
    output=[]
    for (subset,case,member),group in cases.groupby(['subset','case','member'],sort=True):
        if set(group.arm)!={'P0','P1'}:
            raise ValueError(f'missing arm for {subset}/{case}/{member}')
        p0=group[group.arm=='P0'].iloc[0];p1=group[group.arm=='P1'].iloc[0]
        for field in ('q_exact','positive_exact','negative_exact','h0_exact','h0_in_q_exact'):
            if int(p0[field])!=int(p1[field]):
                raise ValueError(f'Q/GT denominator mismatch {subset}/{case}/{member}/{field}')
        output.append({'subset':subset,'case':case,'member':member,
                       'q_exact':int(p0.q_exact),'positive_exact':int(p0.positive_exact),
                       'negative_exact':int(p0.negative_exact),'h0_exact':int(p0.h0_exact),
                       'h0_in_q_exact':int(p0.h0_in_q_exact),
                       'delta_h0_selected_P1_minus_P0':int(p1.h0_selected_exact-p0.h0_selected_exact),
                       'delta_fp_P1_minus_P0':int(p1.threshold_fp_exact-p0.threshold_fp_exact),
                       'delta_tp_P1_minus_P0':int(p1.threshold_tp_exact-p0.threshold_tp_exact),
                       'delta_ap_hist_P1_minus_P0':float(p1.ap_hist_approx-p0.ap_hist_approx)
                       if pd.notna(p0.ap_hist_approx) and pd.notna(p1.ap_hist_approx) else None})
    pd.DataFrame(output).to_csv(b/'PER_CASE_MEMBER_DIFF.csv',index=False)
    split=json.loads((b/'PROBE_SPLIT_MANIFEST_v2_PATIENT.json').read_text())
    token=split['case_patient_token']
    ci={}
    rng=np.random.default_rng(4421)
    for subset in ('probe_holdout','val'):
        c=pd.DataFrame(output)
        c=c[(c.subset==subset)&(c.case!='ALL')&(c.member==-1)].copy()
        c['patient']=c.case.map(token)
        if c.patient.isna().any():raise ValueError('missing patient token')
        groups=c.groupby('patient').agg({'h0_exact':'sum','negative_exact':'sum',
            'delta_h0_selected_P1_minus_P0':'sum','delta_fp_P1_minus_P0':'sum'})
        points=groups.to_numpy(dtype=np.float64)
        bootstrap=[]
        for _ in range(1000):
            take=points[rng.integers(0,len(points),size=len(points))].sum(axis=0)
            bootstrap.append([take[2]/max(take[0],1),take[3]/max(take[1],1)])
        boot=np.asarray(bootstrap)
        ci[subset]={'patients':len(groups),
            'observed_delta_h0_recall':float(points[:,2].sum()/points[:,0].sum()),
            'observed_delta_fpr':float(points[:,3].sum()/points[:,1].sum()),
            'patient_bootstrap_1000_seed4421':{
                'delta_h0_recall_q025_q50_q975':np.quantile(boot[:,0],[.025,.5,.975]).tolist(),
                'delta_fpr_q025_q50_q975':np.quantile(boot[:,1],[.025,.5,.975]).tolist()},
            'interpretation':'descriptive; not multi-seed or independent test evidence'}
    (b/'PATIENT_BOOTSTRAP_DIFF.json').write_text(json.dumps(ci,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'results_rows':len(rows),'diff_rows':len(output),
                      'holdout':ci['probe_holdout'],'val':ci['val']}))


if __name__=='__main__':main()
