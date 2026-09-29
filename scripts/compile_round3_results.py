"""Compile completed Round3 artifacts; unavailable arms remain status rows."""
import csv,json,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/v3_round3'
def read(name):return json.loads((D/name).read_text())
def flat_ram(x):
    o=x['overall_instance_metrics'];r=x['overlap_region_metrics'];p=x['overlap_pair_intersection_metrics']
    return {'dice':o['dsc'],'iou':1-o['voe'],'nsd':o['nsd_2px'],'voe':o['voe'],'msd':o['msd_px'],'msd_fail':o['msd_fail_rate'],'ravd':o['ravd'],
            'overlap_dsc':r['dsc'],'overlap_nsd':r['nsd_2px'],'overlap_voe':r['voe'],'overlap_msd':r['msd_px'],'overlap_fail':r['msd_fail_rate'],'overlap_ravd':r['ravd'],
            'pair_dsc':p['dsc'],'pair_nsd':p['nsd_2px'],'pair_voe':p['voe'],'pair_msd':p['msd_px'],'pair_fail':p['msd_fail_rate'],'pair_ravd':p['ravd']}
def diff(a,b):return {k:a[k]-b[k] for k in a if a[k] is not None and b.get(k) is not None}
def summary(values):
    return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),'min':float(np.min(values)),'max':float(np.max(values)),
            'positive':sum(x>0 for x in values),'negative':sum(x<0 for x in values),'zero':sum(x==0 for x in values)}

def main():
    rows=[];payload={'TSRS':{},'RAM':{}}
    ts={arm:read(f'TSRS_{arm}_R201.json') for arm in ['N0','A','B']}
    tman={'A':read('TSRS_A_MANIFEST.json'),'B':read('TSRS_B_MANIFEST.json')}
    for arm in ['N0','A','B','C','D']:
        if arm in ts:
            m=ts[arm]['mean'];entry={'status':'COMPLETE','metrics':m,'selected_epoch':tman.get(arm,{}).get('selected_epoch'),'checkpoint_sha256':ts[arm]['checkpoint_sha256'],'cases':96}
        else:entry={'status':'SUPERVISION_PENDING_NOT_RUN','metrics':{},'selected_epoch':None,'checkpoint_sha256':None,'cases':0}
        payload['TSRS'][arm]=entry;rows.append({'dataset':'TSRS','arm':arm,**entry['metrics'],'status':entry['status'],'selected_epoch':entry['selected_epoch'],'checkpoint_sha256':entry['checkpoint_sha256']})
    rn0=read('RAM_N0_STRICT.json');rv={'N0':{'official':rn0['official'],'per_case':rn0['per_case']}}
    for arm in 'ABCD':rv[arm]=read(f'RAM_{arm}_SELECTED_VALIDATION.json')
    for arm in ['N0','A','B','C','D']:
        metrics=flat_ram(rv[arm]['official']);res={} if arm=='N0' else read(f'RAM_{arm}_RESULT.json')
        entry={'status':'COMPLETE','metrics':metrics,'selected_epoch':res.get('selected_epoch'),'checkpoint_sha256':res.get('best_checkpoint_sha256',rn0['source_sha256']),'cases':69}
        payload['RAM'][arm]=entry;rows.append({'dataset':'RAM','arm':arm,**metrics,'status':'COMPLETE','selected_epoch':entry['selected_epoch'],'checkpoint_sha256':entry['checkpoint_sha256']})
    fieldnames=sorted(set().union(*(r.keys() for r in rows)))
    with (D/'PILOT_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)
    paired={'TSRS':{},'RAM':{}}
    for arm in ['A','B']:
        paired['TSRS'][arm]={}
        for base in ['N0']+(['A'] if arm!='A' else []):
            per={r['case']:r for r in ts[arm]['per_case']};bp={r['case']:r for r in ts[base]['per_case']}
            paired['TSRS'][arm][f'vs_{base}']={'mean_delta':diff(ts[arm]['mean'],ts[base]['mean']),
                'per_case':{k:summary([per[c][k]-bp[c][k] for c in sorted(per)]) for k in ts[arm]['mean']}}
    for arm in 'ABCD':
        paired['RAM'][arm]={}
        for base in ['N0']+(['A'] if arm!='A' else []):
            cases=sorted(rv[arm]['per_case']);case_metrics={}
            for section,prefix in [('overall_instance_metrics','overall'),('overlap_region_metrics','overlap'),('overlap_pair_intersection_metrics','pair')]:
                for metric in ['dsc','nsd_2px','voe','msd_px','msd_fail_rate','ravd']:
                    vals=[]
                    for c in cases:
                        x=rv[arm]['per_case'][c][section][metric];y=rv[base]['per_case'][c][section][metric]
                        if x is not None and y is not None:vals.append(x-y)
                    if vals:case_metrics[prefix+'_'+metric]=summary(vals)
            paired['RAM'][arm][f'vs_{base}']={'aggregate_delta':diff(flat_ram(rv[arm]['official']),flat_ram(rv[base]['official'])),'per_case':case_metrics}
    for domain in payload:
        for arm,entry in payload[domain].items():
            if entry['status']!='COMPLETE':entry['scientific_status']='NOT_RUN'
            elif arm=='N0':entry['scientific_status']='REFERENCE'
            else:
                n0=payload[domain]['N0']['metrics'];m=entry['metrics'];region=m['dice']>=n0['dice'] and m['iou']>=n0['iou']
                relative_a=True if arm=='A' else m['dice']>=payload[domain]['A']['metrics']['dice'] and m['iou']>=payload[domain]['A']['metrics']['iou']
                entry['scientific_status']='PILOT_POSITIVE_REGION_GATE' if region and relative_a else 'SCIENTIFIC_NEGATIVE_OR_INCONCLUSIVE'
                entry['region_gate_vs_N0']=region;entry['region_gate_vs_A']=relative_a
    (D/'PILOT_RESULTS.json').write_text(json.dumps(payload,indent=2))
    (D/'PAIRED_DELTAS.json').write_text(json.dumps(paired,indent=2))
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
