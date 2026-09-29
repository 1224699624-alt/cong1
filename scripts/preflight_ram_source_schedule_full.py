"""Validate every immutable augmented batch and C0 reduction without updates."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
import run_ram_source_correctability_2x2 as run
import run_v31_ram_mature as legacy


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.lock.read_text());assert not a.out.exists()
    for path,h in cfg['file_hashes'].items():assert run.sha256_file(Path(path))==h,path
    start=time.monotonic();torch.set_num_threads(64);run.seed_all(cfg['seed'])
    steps,_,_=run._load_schedule(cfg,a.lock);device=torch.device('cuda')
    args=SimpleNamespace(r332=Path(cfg['r332']),overlap=Path(cfg['overlap']),overlap_sha=cfg['overlap_sha256'])
    _,m,r,g,_,_=legacy.load_components(args,device)
    ck=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);r.load_state_dict(ck['refiner'])
    m.eval().requires_grad_(False);r.eval().requires_grad_(False)
    loaded=run._validate_schedule_against_c0(m,steps,device)
    gaps=[abs(row['denominator_c0_actual']-row['denominator_c0']) for row in loaded]
    report={'status':'PASS','optimizer_updates':0,'cases':len({row['case'] for row in loaded}),'steps':len(loaded),
        'batch_hashes_verified':True,'H0_tensor_hashes_verified':True,'C0_denominators_verified':True,
        'max_denominator_abs_gap':max(gaps),'seconds':time.monotonic()-start,'test_used':False}
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
