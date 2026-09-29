"""Count H0 membership and overlap opportunity from immutable C0 mask caches."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from audit_ram_source_correctability_oracle import unpack_bool


def analyze(target,valid,pred,h0):
    target=np.asarray(target,bool);valid=np.asarray(valid,bool)
    pred=np.asarray(pred,bool);h0=np.asarray(h0,bool)
    assert target.shape==pred.shape==h0.shape and target.shape[0]==14
    gt_n=target.sum(0);p_n=pred.sum(0);h_n=h0.sum(0)
    gt_o=(gt_n>=2)&valid;single=(gt_n==1)&valid;region=(h_n>0)&valid
    old_overlap_fn=gt_o&(p_n<2);oracle_n=(pred|h0).sum(0)
    return {'H0_member_pixels':int(h0.sum()),'H0_unique_pixels':int(region.sum()),
        'H0_pred_members_0':int((region&(p_n==0)).sum()),'H0_pred_members_1':int((region&(p_n==1)).sum()),
        'H0_pred_members_2plus':int((region&(p_n>=2)).sum()),
        'global_overlap_FN_pixels_before':int(old_overlap_fn.sum()),
        'global_overlap_FN_recovered_by_oracle':int((old_overlap_fn&(oracle_n>=2)).sum()),
        'one_H0_member_sufficient':int((region&(p_n==1)&(h_n>=1)&gt_o).sum()),
        'at_least_two_H0_members_needed_and_available':int((region&(p_n==0)&(h_n>=2)&gt_o).sum()),
        'H0_alone_insufficient_from_pred0':int((region&(p_n==0)&(h_n==1)&gt_o).sum()),
        'H0_members_at_pred2plus':int(h0[:,region&(p_n>=2)].sum()),
        'overlap_member_FN':int((target&~pred&gt_o[None]).sum()),
        'overlap_GT_positive_members':int((target&gt_o[None]).sum()),
        'nonoverlap_member_FN':int((target&~pred&single[None]).sum()),
        'nonoverlap_GT_positive_members':int((target&single[None]).sum())}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True);start=time.monotonic()
    for path,digest in lock['file_hashes'].items():assert sha(path)==digest,path
    root=Path(lock['predictions_dir']);totals={}
    for split,count in lock['split_counts'].items():
        paths=sorted((root/split).glob('*.npz'));assert len(paths)==count,(split,len(paths))
        aggregate={}
        for path in paths:
            with np.load(path,allow_pickle=False) as x:
                shape=tuple(int(v) for v in x['shape'])
                truth=unpack_bool(x['target'],shape);valid=unpack_bool(x['valid'],shape[-2:]);pred=unpack_bool(x['step2'],shape);h0=unpack_bool(x['h0'],shape)
            row={'case':path.stem,**analyze(truth,valid,pred,h0)}
            with (a.out/(split+'_per_case.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
            for k,v in row.items():
                if k!='case':aggregate[k]=aggregate.get(k,0)+v
        aggregate['cases']=count
        for region in ['overlap','nonoverlap']:
            aggregate[region+'_conditional_member_FN_rate']=aggregate[region+'_member_FN']/aggregate[region+'_GT_positive_members']
        totals[split]=aggregate
    (a.out/'RESULT.json').write_text(json.dumps({'status':'COMPLETE','scope':'read-only C0 masks; diagnostic counts, not model metrics','splits':totals,'seconds':time.monotonic()-start,'optimizer_updates':0,'test_used':False},indent=2))
    print(json.dumps({'status':'COMPLETE','seconds':time.monotonic()-start,'val_H0':totals['val']['H0_member_pixels']}),flush=True)


if __name__=='__main__':main()
