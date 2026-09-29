"""Summarize the completed RAM S/O prior pair without retuning any thresholds."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    results=root/'results';rows=[];diff=[];bootstrap={};rng=np.random.default_rng(4421)
    split=json.loads(Path('outputs/ram_highres_readout_20260928/PROBE_SPLIT_MANIFEST_v2_PATIENT.json').read_text())
    patient=split['case_patient_token']
    for subset in ('CALIBRATION','HOLDOUT','VAL'):
        r=json.loads((results/f'{subset}_RESULT.json').read_text())
        for arm,v in r['arms'].items():
            rows.append({'subset':subset,'arm':arm,'cases':r['cases'],'prior_mask_Dice':v['standard_overlap_mask']['dsc'],
                'prior_mask_NSD2px':v['standard_overlap_mask']['nsd_2px'],'prior_mask_MSDpx':v['standard_overlap_mask']['msd_px'],
                'BCE':v['bce_case_mean'],'AP_hist_approx':v['natural_full_curves']['AP_hist_approx'],
                'fixed0_5_precision':v['fixed0_5']['precision'],'fixed0_5_GT_overlap_recall':v['fixed0_5']['recall'],
                'fixed0_5_H0_unique_recall':v['fixed0_5']['H0_unique_recall'],
                'matched_threshold':v['matched']['threshold'],'matched_single_FPR':v['matched']['single_fpr'],
                'matched_single_FP':v['matched']['single_fp'],'matched_background_FP':v['matched']['background_fp'],
                'matched_H0_unique_selected':v['matched']['H0_unique_selected'],'H0_unique_total':v['denominators']['H0_unique'],
                'matched_H0_unique_recall':v['matched']['H0_unique_recall'],'matched_H0_member_recall':v['matched']['H0_member_recall'],
                'matched_GT_overlap_recall':v['matched']['GT_overlap_recall'],'segmentation_gain':'NOT_TESTED'})
        ref={x['case']:x for x in r['per_case']['O_REF_CONTINUE']}
        candidate={x['case']:x for x in r['per_case']['O_HARD_WEIGHT']}
        for case,c in candidate.items():
            old=ref[case]
            diff.append({'subset':subset,'case':case,'cohort':c['cohort'],'side':c['side'],
                         'delta_H0_unique':c['matched']['H0_unique_selected']-old['matched']['H0_unique_selected'],
                         'delta_single_FP':c['matched']['single_fp']-old['matched']['single_fp'],
                         'delta_background_FP':c['matched']['background_fp']-old['matched']['background_fp'],
                         'delta_GT_overlap_selected':c['matched']['GT_overlap_selected']-old['matched']['GT_overlap_selected'],
                         'delta_prior_mask_Dice':c['metric']['dsc']-old['metric']['dsc'],
                         'delta_prior_mask_NSD':c['metric']['nsd_2px']-old['metric']['nsd_2px']})
        if subset!='CALIBRATION':
            groups={}
            for case,c in candidate.items():
                p=patient[case];g=groups.setdefault(p,np.zeros(4,dtype=float));old=ref[case]
                g += [c['matched']['H0_unique_selected']-old['matched']['H0_unique_selected'],c['H0_unique'],
                      c['matched']['single_fp']-old['matched']['single_fp'],c['GT_single']]
            values=np.stack(list(groups.values()));samples=[]
            for _ in range(2000):
                sample=values[rng.integers(0,len(values),len(values))].sum(0)
                samples.append([sample[0]/max(1,sample[1]),sample[2]/max(1,sample[3])])
            bootstrap[subset]={'patients':len(groups),'delta_H0_unique_recall':float(values[:,0].sum()/values[:,1].sum()),
                'paired_patient_95pct_delta_H0_unique_recall':np.quantile(samples,[.025,.975],axis=0)[:,0].tolist(),
                'paired_patient_95pct_delta_singleFPR':np.quantile(samples,[.025,.975],axis=0)[:,1].tolist(),
                'note':'descriptive single-seed adaptation comparison, not fresh blind validation'}
    for name,content in [('SO_PRIOR_RESULTS.csv',rows),('O_PER_CASE_COST_DIFF.csv',diff)]:
        with (root/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(content[0]));w.writeheader();w.writerows(content)
    (root/'O_PATIENT_BOOTSTRAP.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    # Consolidate real optimizer traces; no generated or interpolated rows.
    with (root/'SO_PRIOR_TRACE.jsonl').open('w',encoding='utf8') as output:
        for arm in ('O_REF_CONTINUE','O_HARD_WEIGHT'):
            output.write((results/arm/'SO_PRIOR_TRACE.jsonl').read_text())
    print(json.dumps({'rows':len(rows),'per_case_diff':len(diff),'bootstrap':bootstrap}))

if __name__=='__main__':main()
