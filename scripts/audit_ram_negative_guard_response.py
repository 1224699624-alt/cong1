"""Fixed-initial ROI response audit; read-only val inference, no fitting."""
import argparse, copy, json, time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    torch.set_num_threads(64);legacy.seed_all(4421);device=torch.device('cuda');start=time.monotonic()
    _,model,refiner,_,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    models={}
    for name,path in lock['checkpoints'].items():
        m=copy.deepcopy(model);r=copy.deepcopy(refiner);ck=torch.load(path,map_location='cpu',weights_only=False)
        m.load_state_dict(ck['model']);r.load_state_dict(ck['refiner']);m.eval().requires_grad_(False);r.eval().requires_grad_(False);models[name]=(m,r)
    del model,refiner
    totals={};cases=[]
    with torch.no_grad():
        for batch in DataLoader(legacy.NativeWristDataset(Path(lock['dataset']),'val',False),batch_size=1,num_workers=0):
            if time.monotonic()-start>lock['max_seconds']:raise TimeoutError('read-only audit cap')
            image=batch['image'].to(device);truth=batch['mask'].to(device)>.5;valid=batch['valid'].to(device).bool();outputs={}
            for name,(m,r) in models.items():
                src=m(torch.cat([image,torch.zeros_like(image)],1));final=legacy.iterative_refine_o(image,src,r,2,torch.zeros_like(image))[0][-1];outputs[name]=(src,final)
            si,zi=outputs['initial'];pi=zi>=0;hard=truth&~pi&valid&(truth.sum(1,keepdim=True)>=2)&(si < -5.00001)
            risk=~truth&valid&(zi.abs()<.25)
            row={'case':batch['case'][0],'hard_FN':int(hard.sum()),'frozen_teacher_negative_band':int(risk.sum()),'arms':{}}
            for name in ['positive','guard']:
                src,z=outputs[name];pred=z>=0
                counts={'FN':int((truth&~pred&valid).sum()),'FP':int((~truth&pred&valid).sum()),'new_FP':int((~truth&~pi&pred&valid).sum()),'new_FN':int((truth&pi&~pred&valid).sum()),'hard_recovered':int((hard&pred).sum()),'hard_source_shift_sum':float((src-si)[hard].sum()),'risk_final_shift_sum':float((z-zi)[risk].sum())}
                row['arms'][name]=counts
                dest=totals.setdefault(name,{})
                for k,v in counts.items():dest[k]=dest.get(k,0)+v
            row['positive_only_FP_removed_by_guard']=int((~truth&valid&(outputs['positive'][1]>=0)&(outputs['guard'][1]<0)).sum())
            row['guard_only_FP_added']=int((~truth&valid&(outputs['positive'][1]<0)&(outputs['guard'][1]>=0)).sum())
            cases.append(row)
            with (a.out/'INFERENCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    result={'status':'COMPLETE','cases':len(cases),'totals':totals,'hard_FN':sum(x['hard_FN'] for x in cases),'frozen_teacher_negative_band':sum(x['frozen_teacher_negative_band'] for x in cases),'positive_only_FP_removed_by_guard':sum(x['positive_only_FP_removed_by_guard'] for x in cases),'guard_only_FP_added':sum(x['guard_only_FP_added'] for x in cases),'test_used':False,'seconds':time.monotonic()-start,'scope':'diagnostic counts on valid members; official evaluation remains unchanged'}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
