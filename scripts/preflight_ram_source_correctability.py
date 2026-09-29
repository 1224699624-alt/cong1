"""No-update C0/Adam/D2/H0 contract check on the frozen first schedule batch."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import run_ram_source_correctability_2x2 as run


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();start=time.monotonic()
    for path,h in lock['file_hashes'].items():assert run.sha256_file(Path(path))==h,path
    torch.set_num_threads(64);device=torch.device('cuda');schedule=json.loads(Path(lock['schedule_manifest']).read_text());record=schedule['steps'][0]
    row={'path':str(Path(lock['schedule_manifest']).parent/record['path']),'step':1,'case':record['case'],'batch_sha256':record['batch_sha256'],'indices':record['indices'],'denominator_c0':record['denominator_c0'],**{k:record[k] for k in ['image_sha256','mask_sha256','valid_sha256','h0_sha256']}}
    payload,_=run._load_step_payload(row);reports={}
    for arm in ['SRC_F0S0','SRC_F1S0']:
        run.seed_all(4421)
        _,m,r,g,opt,scale,groups,_,_,_=run._prepare_components(lock,a.lock,arm,device)
        run._configure_modes(m,r)
        x=payload['image'][None].to(device);truth=payload['mask'][None].to(device);h0=payload['h0'][None].to(device)
        actual=run._c0_denominator(m,x,truth,record['indices'])
        if abs(actual-record['denominator_c0'])>1e-4:raise ValueError(('D_c0_mismatch',actual,record['denominator_c0']))
        with torch.cuda.amp.autocast():source=run._source_logits(m,x)
        q=F.softplus(-source.float())[h0].mean()
        value={'D_c0':actual,'source_sha256':hashlib.sha256(source.detach().float().cpu().numpy().tobytes()).hexdigest(),
               'Q_H':float(q.detach()),'H0_members':int(h0.sum()),'old_optimizer_group_count':len(opt.param_groups),
               'old_parameter_steps':sorted(set(int(opt.state[p]['step']) for name in ['refiner','decoder_existing'] for _,p in groups[name])),
               'new_D2_requires_grad':sum(p.requires_grad for _,p in groups['decoder_D2']),
               'new_D2_initial_steps':sorted(set(int(opt.state[p]['step']) for _,p in groups['decoder_D2'] if p in opt.state)),
               'scaler':float(scale.get_scale()),'D2_param_count':sum(p.numel() for _,p in groups['decoder_D2'])}
        if arm=='SRC_F1S0':
            grads=torch.autograd.grad(q*1024.,[p for _,p in groups['decoder_D2']],allow_unused=True)
            value['D2_QH_grad_abs_sum']=sum(float(v.detach().float().abs().sum())/1024. for v in grads if v is not None)
            assert value['D2_QH_grad_abs_sum']>0
        reports[arm]=value
        del m,r,g,opt,scale;torch.cuda.empty_cache()
    assert reports['SRC_F0S0']['source_sha256']==reports['SRC_F1S0']['source_sha256']
    assert reports['SRC_F0S0']['old_parameter_steps']==reports['SRC_F1S0']['old_parameter_steps']==[100]
    assert reports['SRC_F0S0']['scaler']==reports['SRC_F1S0']['scaler']==1024.
    a.out.write_text(json.dumps({'status':'PASS','optimizer_updates':0,'case':record['case'],'batch_sha256':record['batch_sha256'],'factors':reports,'seconds':time.monotonic()-start},indent=2))
    print(json.dumps({'status':'PASS','seconds':time.monotonic()-start,'D2_QH_grad_abs_sum':reports['SRC_F1S0']['D2_QH_grad_abs_sum']}),flush=True)


if __name__=='__main__':main()
