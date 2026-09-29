"""Full-val zero-branch parity; equal masks imply unchanged deterministic metrics."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
import run_v31_ram_mature as legacy
from r332_member_identity import IdentityContextRefiner,iterative_refine_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();lock=json.loads(a.lock.read_text());assert not a.out.exists()
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    torch.set_num_threads(64);legacy.seed_all(4421);device=torch.device('cuda');start=time.monotonic()
    _,m,r,_,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    ck=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);r.load_state_dict(ck['refiner']);m.eval();r.eval();n=IdentityContextRefiner.from_mature(r)
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'val',False);maxima={k:[0.,0.,0.] for k in ['canonical','scrambled']};mismatch={k:[0,0,0] for k in maxima}
    with torch.no_grad():
        for i in range(len(ds)):
            if time.monotonic()-start>300:raise TimeoutError('parity cap')
            b=ds[i];x=b['image'][None].to(device);o=torch.zeros_like(x);source=m(torch.cat([x,o],1));old=legacy.iterative_refine_o(x,source,r,2,o)[0]
            for mode in maxima:
                n.identity_mode=mode;new=iterative_refine_identity(x,source,n,2,o)[0]
                for k,(aa,bb) in enumerate(zip(old,new)):
                    maxima[mode][k]=max(maxima[mode][k],float((aa-bb).abs().max()));mismatch[mode][k]+=int(((aa>=0)!=(bb>=0)).sum())
    result={'status':'PASS' if max(v for x in maxima.values() for v in x)<=1e-6 and sum(v for x in mismatch.values() for v in x)==0 else 'FAIL','cases':len(ds),'max_abs_logits':maxima,'mask_mismatch':mismatch,'metric_parity':'identical prediction masks and targets imply identical deterministic official metrics; no separate metric recomputation','seconds':time.monotonic()-start,'optimizer_updates':0}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result));assert result['status']=='PASS'


if __name__=='__main__':main()
