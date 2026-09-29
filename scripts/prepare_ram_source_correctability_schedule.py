"""Select train-only diagnostic cases and freeze C0-augmented 4-member batches."""
import argparse,hashlib,json,math,random,shutil,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ram_source_correctability_data import NativeWristH0Dataset,build_deterministic_schedule
from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def case_bucket(case):
    return int(hashlib.sha256(case.encode()).hexdigest()[:8],16)%5


def presence(path):
    with np.load(path,allow_pickle=False) as archive:
        shape=tuple(int(v) for v in archive['shape']);bits=np.unpackbits(archive['h0'],bitorder='little')[:math.prod(shape)]
    return set(np.flatnonzero(bits.reshape(shape).reshape(14,-1).any(1)).tolist())


def choose(candidates,k):
    if len(candidates)<k:raise ValueError(('too_few_cases',len(candidates),k))
    ordered=sorted(candidates,key=lambda row:(row['h0_member_pixels'],row['case']))
    selected=[];anchors=[.10,.25,.50,.75,.90]
    for q in anchors:
        row=ordered[round(q*(len(ordered)-1))]
        if row not in selected:selected.append(row)
    universe=set().union(*(x['members'] for x in ordered));covered=set().union(*(x['members'] for x in selected))
    frequency={m:sum(m in x['members'] for x in ordered) for m in universe}
    while len(selected)<k and covered!=universe:
        options=[x for x in ordered if x not in selected]
        best=max(options,key=lambda x:(sum(1/frequency[m] for m in x['members']-covered),
                                       -abs(math.log1p(x['h0_member_pixels'])-math.log1p(ordered[len(ordered)//2]['h0_member_pixels'])),
                                       x['case']))
        if not (best['members']-covered):break
        selected.append(best);covered|=best['members']
    for i in np.linspace(0,len(ordered)-1,min(len(ordered),k*8),dtype=int):
        if len(selected)>=k:break
        row=ordered[int(i)]
        if row not in selected:selected.append(row)
    for row in ordered:
        if len(selected)>=k:break
        if row not in selected:selected.append(row)
    assert len(selected)==k
    return selected,sorted(universe-covered)


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True);start=time.monotonic()
    for path,h in lock['file_hashes'].items():assert sha(path)==h,path
    torch.set_num_threads(64)
    snapshots=[json.loads(line) for line in Path(lock['train_snapshot_jsonl']).read_text().splitlines()]
    assert len(snapshots)==425
    h0_root=Path(lock['h0_dir']);rows=[]
    for row in snapshots:
        case=row['case'];n=int(row['h0']['member_pixels']);rows.append({'case':case,'h0_member_pixels':n,
            'h0_unique_pixels':int(row['h0']['unique_pixels']),'members':presence(h0_root/'train'/(case+'.npz'))})
    pool=[x for x in rows if case_bucket(x['case'])!=0];held=[x for x in rows if case_bucket(x['case'])==0]
    chosen,uncovered=choose(pool,lock['update_cases']);hold,hold_uncovered=choose(held,lock['heldout_cases'])
    chosen_cases=[x['case'] for x in chosen];held_cases=[x['case'] for x in hold]
    assert set(chosen_cases).isdisjoint(held_cases)
    h0_per_cycle=sum(x['h0_member_pixels'] for x in chosen)
    repeats=max(lock['minimum_cycles'],math.ceil(lock['target_h0_exposures']/h0_per_cycle))
    if repeats>lock['maximum_cycles']:raise ValueError({'support_insufficient_under_budget':repeats,'max_cycles':lock['maximum_cycles']})
    ds=NativeWristH0Dataset(Path(lock['dataset']), 'train', True,h0_dir=h0_root)
    id_to_index={path.stem:i for i,path in enumerate(ds.mask_files)};rng=random.Random(lock['seed']);order=[]
    for cycle in range(repeats):
        cycle_cases=chosen_cases.copy();rng.shuffle(cycle_cases)
        if order and cycle_cases[0]==order[-1]:cycle_cases[0],cycle_cases[1]=cycle_cases[1],cycle_cases[0]
        order.extend(cycle_cases)
    case_indices=[id_to_index[case] for case in order]
    sample=ds[case_indices[0]];sample_bytes=sum(sample[key].numel()*sample[key].element_size() for key in ['image','mask','valid','h0'])
    projected_bytes=int(len(order)*sample_bytes*1.25)
    free=shutil.disk_usage(a.out).free
    if projected_bytes+5*1024**3>free:raise RuntimeError({'disk_free_bytes':free,'projected_schedule_bytes':projected_bytes})
    device=torch.device('cuda');model=RamNnUNetTwoChannel().to(device);ck=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True);model.eval().requires_grad_(False)
    s=build_deterministic_schedule(ds,model,a.out/'schedule',lock['seed'],case_order=case_indices)
    del model;torch.cuda.empty_cache()
    first=s['steps'][0];assert len(s['steps'])==len(order) and len({x['case'] for x in s['steps']})==len(chosen)
    metadata={'status':'COMPLETE','dataset':'RAM_train425','C0_sha256':lock['file_hashes'][lock['checkpoint']],
      'selection_rule':'case SHA256 modulo5 heldout; update/heldout case lists selected by H0 burden quantiles and rare-member coverage from train only; no patient ID assumed',
      'update_cases':chosen_cases,'heldout_cases':held_cases,'update_case_H0_member_pixels':{x['case']:x['h0_member_pixels'] for x in chosen},
      'holdout_case_H0_member_pixels':{x['case']:x['h0_member_pixels'] for x in hold},'update_member_coverage':sorted(set().union(*(x['members'] for x in chosen))),
      'heldout_member_coverage':sorted(set().union(*(x['members'] for x in hold))),
      'uncovered_update_members':uncovered,'uncovered_holdout_members':hold_uncovered,
      'repeats':repeats,'updates':len(order),'unaugmented_H0_exposures':h0_per_cycle*repeats,
      'actual_augmented_H0_exposures':sum(int(torch.load((a.out/'schedule'/x['path']),map_location='cpu',weights_only=False)['h0'].sum()) for x in s['steps']),
      'first_step':{'case':first['case'],'indices':first['indices'],'denominator_c0':first['denominator_c0']},
      'schedule_manifest':str(a.out/'schedule/schedule.json'),'schedule_sha256':sha(a.out/'schedule/schedule.json'),
      'source_code_sha256':lock['file_hashes'][lock['code_path']],'estimated_schedule_bytes':projected_bytes,'remaining_free_bytes':shutil.disk_usage(a.out).free,
      'GPU_seconds':time.monotonic()-start,'test_used':False}
    (a.out/'SELECTION_AND_SCHEDULE.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({'status':'COMPLETE','updates':len(order),'repeats':repeats,'H0_exposures':metadata['actual_augmented_H0_exposures'],'seconds':metadata['GPU_seconds']}),flush=True)


if __name__=='__main__':main()
