"""Read-only C0 floating threshold equivalence check; no training/test use."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from run_ram_so_c00_cs_pair import load_batch_data,prior,_capture_frozen
from ram_highres_capture import load_frozen_c0
from evaluate_r329_ram_official_metrics_val import _official_one,_aggregate
from train_r284_ramw600_overlap_prior import top_training_pairs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.lock.read_text());torch.set_num_threads(8);start=time.monotonic()
    model,refiner,_=load_frozen_c0(Path(cfg['c0']),'cuda');ds=load_batch_data(cfg,'val')
    pairs=top_training_pairs(Path(cfg['dataset']),15)
    members=[];overlaps=[];pairrows=[];cases=[];mismatch=0
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
            image,_,_=prior(batch,'cuda',cfg);_,_,z2,_=_capture_frozen(model,refiner,image)
            mathematical=z2>=0;historical=z2.sigmoid()>=.5
            diff=mathematical!=historical;count=int(diff.sum());mismatch+=count
            if count:
                value=z2[diff].float()
                cases.append({'case':batch['case'][0],'mismatch_members':count,
                              'min_logit':float(value.min()),'max_logit':float(value.max())})
            p=historical.cpu().numpy()[0];t=batch['mask'].numpy()[0]>.5
            members.extend(_official_one(p[k],t[k]) for k in range(14))
            overlaps.append(_official_one(p.sum(0)>=2,t.sum(0)>=2))
            pairrows.extend(_official_one(p[i]&p[j],t[i]&t[j]) for i,j in pairs if (t[i]&t[j]).any())
    official={'overall_instance_metrics':_aggregate(members),'overlap_region_metrics':_aggregate(overlaps),
              'overlap_pair_intersection_metrics':_aggregate(pairrows),'overlap_pair_count':len(pairrows)}
    old=json.loads(Path(cfg['historical_C0_val']).read_text())['official']['step2']
    result={'status':'PASS' if official==old else 'HISTORIC_MISMATCH_UNRESOLVED',
            'logit_vs_sigmoid_member_mismatches':mismatch,'cases':cases,'official_sigmoid_padded':official,
            'historical_equal':official==old,'time_seconds':time.monotonic()-start,'test_used':False,'optimizer_steps':0}
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'mismatch_members':mismatch,'cases':len(cases),'seconds':result['time_seconds']}),flush=True)
    if not result['historical_equal']:raise ValueError('additional evaluator mismatch remains')
if __name__=='__main__':main()
