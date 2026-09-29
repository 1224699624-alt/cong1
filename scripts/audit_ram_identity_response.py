"""Read-only endpoint response and conditioning dependency, never efficacy alone."""
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
import run_v31_ram_mature as legacy
from r332_member_identity import IdentityContextRefiner,iterative_refine_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    start=time.monotonic();torch.set_num_threads(64);legacy.seed_all(4421);device=torch.device('cuda')
    _,template,rt,_,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    models={}
    for name,path in lock['checkpoints'].items():
        m=copy.deepcopy(template);r=copy.deepcopy(rt) if name=='initial' else IdentityContextRefiner.from_mature(rt)
        ck=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);r.load_state_dict(ck['refiner']);m.eval().requires_grad_(False);r.eval().requires_grad_(False)
        if name!='initial':r.identity_mode=name
        models[name]=(m,r)
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'val',False);totals={};rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            if time.monotonic()-start>500:raise TimeoutError('response cap')
            b=ds[i];image=b['image'][None].to(device);zero=torch.zeros_like(image);truth=b['mask'][None].to(device)>.5;valid=b['valid'][None].to(device).bool();out={}
            for name,(m,r) in models.items():
                source=m(torch.cat([image,zero],1));fn=legacy.iterative_refine_o if name=='initial' else iterative_refine_identity
                out[name]=(source,fn(image,source,r,2,zero)[0][-1])
            m,r=models['canonical'];source=out['canonical'][0];r.relation_enabled=False
            off=iterative_refine_identity(image,source,r,2,zero)[0][-1];r.relation_enabled=True;r.identity_mode='scrambled'
            wrong=iterative_refine_identity(image,source,r,2,zero)[0][-1];r.identity_mode='canonical'
            initial=out['initial'][1].sigmoid()>=.5;hard=truth&~initial&valid&(truth.sum(1,keepdim=True)>=2)&(out['initial'][0]<-5.00001)
            row={'case':b['case'],'hard_FN':int(hard.sum()),'arms':{},'canonical_off_mask_changes':int(((off.sigmoid()>=.5)!=(out['canonical'][1].sigmoid()>=.5)).logical_and(valid).sum()),'canonical_wrong_identity_mask_changes':int(((wrong.sigmoid()>=.5)!=(out['canonical'][1].sigmoid()>=.5)).logical_and(valid).sum()),'canonical_off_max_logit_change':float((off-out['canonical'][1]).abs().max()),'canonical_wrong_identity_max_logit_change':float((wrong-out['canonical'][1]).abs().max())}
            for name,(src,z) in out.items():
                pred=z.sigmoid()>=.5;v={'FN':int((truth&~pred&valid).sum()),'FP':int((~truth&pred&valid).sum()),'new_FN':int((truth&initial&~pred&valid).sum()),'new_FP':int((~truth&~initial&pred&valid).sum()),'hard_recovered':int((hard&pred).sum()),'hard_source_shift_sum':float((src-out['initial'][0])[hard].sum())}
                row['arms'][name]=v;dest=totals.setdefault(name,{})
                for k,x in v.items():dest[k]=dest.get(k,0)+x
            rows.append(row)
            with (a.out/'INFERENCE_TRACE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    result={'status':'COMPLETE','cases':len(rows),'totals':totals,'hard_FN':sum(x['hard_FN'] for x in rows),'canonical_off_mask_changes':sum(x['canonical_off_mask_changes'] for x in rows),'canonical_wrong_identity_mask_changes':sum(x['canonical_wrong_identity_mask_changes'] for x in rows),'seconds':time.monotonic()-start,'test_used':False,'optimizer_updates':0,'limits':'valid-member diagnostic counts. Turning learned context off or scrambling it is distribution shift and measures dependency only; causal effect comparison is matched training pair.'}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
