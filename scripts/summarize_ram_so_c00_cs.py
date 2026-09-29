"""Combine locked C00/CS official endpoints and patient-paired differences."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    split=json.loads(Path('outputs/ram_highres_readout_20260928/PROBE_SPLIT_MANIFEST_v2_PATIENT.json').read_text())
    patients=split['case_patient_token'];rng=np.random.default_rng(4421)
    rows=[];deltas=[];bootstrap={}
    for subset in ('HOLDOUT','VAL'):
        source=(root/('PADDED_VAL_OFFICIAL_v2.json' if subset=='VAL' else 'PADDED_HOLDOUT_OFFICIAL.json'))
        result=json.loads(source.read_text())
        per=result['per_case'];official=result['official']
        for arm in ('C0','C00','CS'):
            m=official[arm]
            row={'subset':subset,'arm':arm,'cases':result['cases']}
            for prefix,section in (('Overall','overall_instance_metrics'),('Overlap','overlap_region_metrics'),('Pair','overlap_pair_intersection_metrics')):
                for metric in ('dsc','nsd_2px','msd_px','msd_fail_rate','ravd','voe'):
                    row[f'{prefix}_{metric}']=m[section][metric]
                row[f'{prefix}_iou']=1-m[section]['voe']
                row[f'{prefix}_dsc_delta_vs_C00']=m[section]['dsc']-official['C00'][section]['dsc']
            for key in ('fixed_H0_member_count','fixed_H0_member_recovered','new_TP_member_vs_C0','new_FP_member_vs_C0',
                        'lost_TP_member_vs_C0','removed_FP_member_vs_C0'):
                row[key]=sum(r[key] for r in per[arm])
            rows.append(row)
        left={r['case']:r for r in per['CS']};right={r['case']:r for r in per['C00']}
        if set(left)!=set(right):raise ValueError('paired case mismatch')
        for case,l in left.items():
            r=right[case]
            d={'subset':subset,'case':case,'cohort':'EXT' if case.startswith('EXT_') else 'ordinary','side':case[-1]}
            for region in ('overall','overlap','pair'):
                for metric in ('dsc','nsd_2px','msd_px'):
                    x,y=l[region][metric],r[region][metric]
                    d[f'{region}_{metric}_CS_minus_C00']=x-y if x is not None and y is not None else None
            for key in ('fixed_H0_member_recovered','new_TP_member_vs_C0','new_FP_member_vs_C0','lost_TP_member_vs_C0','removed_FP_member_vs_C0'):
                d[key+'_CS_minus_C00']=l[key]-r[key]
            deltas.append(d)
        bootstrap[subset]={}
        for region in ('overall','overlap','pair'):
            for metric in ('dsc','nsd_2px'):
                grouped={}
                for case,l in left.items():
                    r=right[case];n=l[region]['n']
                    if not n or l[region][metric] is None or r[region][metric] is None:continue
                    v=grouped.setdefault(patients[case],np.zeros(2,dtype=np.float64))
                    v += [(l[region][metric]-r[region][metric])*n,n]
                data=np.stack(list(grouped.values()));values=[]
                for _ in range(2000):
                    sampled=data[rng.integers(0,len(data),len(data))].sum(0)
                    values.append(sampled[0]/max(1,sampled[1]))
                bootstrap[subset][region+'_'+metric]={'patients':len(data),
                    'delta_CS_minus_C00':float(data[:,0].sum()/data[:,1].sum()),
                    'paired_patient_95pct':np.quantile(values,[.025,.975]).tolist(),
                    'interpretation':'single-seed fixed-budget adaptation, descriptive patient resampling only'}
    for name,data in [('STANDARD_RESULTS.csv',rows),('PER_CASE_DIFF.csv',deltas)]:
        with (root/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (root/'PATIENT_BOOTSTRAP.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
    with (root/'TRAINING_TRACE.jsonl').open('w',encoding='utf8') as out:
        for arm in ('C00','CS'):
            out.write((root/'results'/f'TRAIN_{arm}'/'TRAINING_TRACE.jsonl').read_text())
    print(json.dumps({'rows':len(rows),'cases':len(deltas),'VAL_bootstrap':bootstrap['VAL']}))

if __name__=='__main__':main()
