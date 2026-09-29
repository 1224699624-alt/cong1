"""Frozen validation responses on ROIs fixed by the common initial model."""
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists()
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    start=time.monotonic();torch.set_num_threads(2);legacy.seed_all(4421);device=torch.device('cuda')
    _,template,ref_template,_,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    models={}
    for arm,path in lock['checkpoints'].items():
        m=copy.deepcopy(template);r=copy.deepcopy(ref_template);payload=torch.load(path,map_location='cpu',weights_only=False)
        m.load_state_dict(payload['model']);r.load_state_dict(payload['refiner']);m.eval().requires_grad_(False);r.eval().requires_grad_(False);models[arm]=(m,r)
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'val',False);rows=[];totals={a:{} for a in ['reference','source_aux']}
    for batch in DataLoader(ds,batch_size=1,shuffle=False,num_workers=0):
        image=batch['image'].to(device);outputs={}
        for arm,(m,r) in models.items():
            src=m(torch.cat([image,torch.zeros_like(image)],1));states,_=legacy.iterative_refine_o(image,src,r,2,torch.zeros_like(image));outputs[arm]=(src,states[-1])
        truth=batch['mask'].to(device)>.5;valid=batch['valid'].to(device).bool();true_o=truth.sum(1,keepdim=True)>=2
        initial_src,initial_final=outputs['initial'];initial_pred=initial_final.sigmoid()>=.5
        rois={'initial_hard_overlap_FN':truth&true_o&~initial_pred&valid&(initial_src < -5.00001),
              'initial_other_overlap_FN':truth&true_o&~initial_pred&valid&(initial_src >= -5.00001),
              'initial_nonoverlap_FN':truth&~true_o&~initial_pred&valid}
        row={'case':batch['case'][0],'arms':{}}
        for arm in ['reference','source_aux']:
            src,final=outputs[arm];pred=final.sigmoid()>=.5;values={}
            for name,roi in rois.items():
                values[name+'_count']=int(roi.sum());values[name+'_source_shift_sum']=float((src-initial_src)[roi].sum())
                values[name+'_recovered']=int((roi&pred).sum());values[name+'_source_above_minus5']=int((roi&(src >= -5.)).sum())
            values.update({'new_member_FN':int((truth&initial_pred&~pred&valid).sum()),'new_member_FP':int((~truth&~initial_pred&pred&valid).sum()),
                           'removed_member_FP':int((~truth&initial_pred&~pred&valid).sum()),'remaining_member_FN':int((truth&~pred&valid).sum())})
            row['arms'][arm]=values
            for key,value in values.items():totals[arm][key]=totals[arm].get(key,0)+value
        rows.append(row)
    a.out.write_text(json.dumps({'status':'COMPLETE','optimizer_updates':0,'cases':len(rows),'test_used':False,'totals':totals,'per_case':rows,
                                'scope':'Frozen initial-case/member ROIs, valid pixels only. Diagnostic counts, not replacement standard metrics.',
                                'seconds':time.monotonic()-start},indent=2))


if __name__=='__main__':main()
