"""Frozen-source two-step residual range diagnostic; no achievable-gain claim."""
import argparse,json,time
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
    start=time.monotonic();torch.set_num_threads(2);legacy.seed_all(3321);device=torch.device('cuda')
    _,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    state=torch.load(lock['reference_checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(state['model']);refiner.load_state_dict(state['refiner'])
    for m in [model,refiner,generator]:m.eval().requires_grad_(False)
    steps=2;bound=steps*float(refiner.residual_limit);assert bound==5.
    rows=[];total={};largest=0.
    for batch in DataLoader(legacy.NativeWristDataset(Path(lock['dataset']),'val',False),batch_size=1,shuffle=False,num_workers=0):
        image=batch['image'].to(device);base=model(torch.cat([image,torch.zeros_like(image)],1))
        states,_=legacy.iterative_refine_o(image,base,refiner,steps,torch.zeros_like(image))
        pred=states[-1].sigmoid()>=.5;o=generator(image).sigmoid()
        largest=max(largest,float((states[-1]-base).abs().max()))
        t=batch['mask'].to(device)>.5;v=batch['valid'].to(device).bool();true_o=t.sum(1,keepdim=True)>=2
        fn=t&~pred&v;fp=~t&pred&v
        blocked_fn=fn&(base < -bound-1e-5);blocked_fp=fp&(base > bound+1e-5)
        overlap_fn=true_o&~(pred.sum(1,keepdim=True)>=2)&v
        # Even with optimal bounded correction, enough true members must be reachable.
        potential_true_members=(t&(base >= -bound-1e-5)).sum(1,keepdim=True)
        blocked_overlap=overlap_fn&(potential_true_members<2)
        counts={'member_FN':int(fn.sum()),'member_FP':int(fp.sum()),
            'member_FN_beyond_positive_residual_bound':int(blocked_fn.sum()),
            'member_FP_beyond_negative_residual_bound':int(blocked_fp.sum()),
            'true_overlap_member_FN':int((fn&true_o).sum()),
            'true_overlap_member_FN_beyond_bound':int((blocked_fn&true_o).sum()),
            'true_overlap_member_FN_Opositive':int((fn&true_o&(o>=.5)).sum()),
            'true_overlap_member_FN_Opositive_beyond_bound':int((blocked_fn&true_o&(o>=.5)).sum()),
            'overlap_FN_pixels':int(overlap_fn.sum()),'overlap_FN_fewer_than_two_true_members_reachable':int(blocked_overlap.sum())}
        rows.append({'case':batch['case'][0],**counts})
        for k,value in counts.items():total[k]=total.get(k,0)+value
    assert largest<=bound+1e-4
    a.out.write_text(json.dumps({'status':'COMPLETE','optimizer_updates':0,'test_used':False,'cases':len(rows),
        'step_count':steps,'per_step_absolute_limit':float(refiner.residual_limit),'total_absolute_logit_limit':bound,
        'measured_max_absolute_logit_change':largest,'threshold_margin':1e-5,'counts':total,'per_case':rows,
        'scope':'Necessary range condition with source logits held fixed. Joint backbone training can move source logits; not a ceiling for the full method.',
        'definition':'For true overlap recovery require at least2 TRUE member logits reachable, not arbitrary false-positive members.',
        'seconds':time.monotonic()-start},indent=2))


if __name__=='__main__':main()
