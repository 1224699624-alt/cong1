"""Train-only, case-held-out association screen. No segmentation optimizer steps."""
import argparse,hashlib,json,time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import run_v31_ram_mature as legacy


def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text());assert not a.out.exists();a.out.mkdir(parents=True)
    for path,h in lock['file_hashes'].items():assert legacy.sha(path)==h,path
    start=time.monotonic();torch.set_num_threads(64);legacy.seed_all(4421);device=torch.device('cuda')
    _,model,_,_,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    ck=torch.load(lock['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval().requires_grad_(False)
    ds=legacy.NativeWristDataset(Path(lock['dataset']),'train',False);samples=[];case_rows=[]
    with torch.no_grad():
        for i in range(len(ds)):
            if time.monotonic()-start>lock['max_seconds']:raise TimeoutError('audit cap')
            b=ds[i];image=b['image'][None].to(device);prob=model(torch.cat([image,torch.zeros_like(image)],1)).sigmoid()[0]
            target=b['mask'].to(device)>.5;valid=b['valid'].to(device).bool();top=prob.argmax(0);mx=prob.max(0).values
            # Own p<.05 and max>.5 guarantees that top is another member.
            candidate=(prob<.05)&(mx>.5)[None]&valid
            case=b['case'];seed=int(hashlib.sha256(case.encode()).hexdigest()[:8],16);rng=np.random.default_rng(seed)
            perm=rng.permutation(14);heldout=seed%5==0;counts=[]
            for label in [0,1]:
                coords=torch.nonzero(candidate&(target if label else ~target));counts.append(len(coords))
                if not len(coords):continue
                # Fixed stratified diagnostic sampling, explicitly not population prevalence.
                take=np.linspace(0,len(coords)-1,min(64,len(coords)),dtype=int);coords=coords[torch.as_tensor(take,device=device)]
                for member,y,x in coords.tolist():
                    own=float(prob[member,y,x]);othermax=float(mx[y,x]);othersum=float((prob[:,y,x].sum()-prob[member,y,x]).clamp(max=2))
                    key=(int(np.digitize(own,[.0001,.001,.0067,.02])),int(np.digitize(othermax,[.6,.8,.95,.99])),int(np.digitize(othersum,[.6,1.,1.5])))
                    other=int(top[y,x]);samples.append((case,heldout,label,key,(member,other),(int(perm[member]),int(perm[other]))))
            case_rows.append({'case':case,'heldout':heldout,'negative_candidates':counts[0],'positive_candidates':counts[1]})
            if i%50==0:print(json.dumps({'cases':i+1,'sampled':len(samples)}),flush=True)
    base=defaultdict(lambda:[0,0]);tables={k:defaultdict(lambda:[0,0]) for k in ['canonical','scrambled']}
    for case,held,y,key,canonical,scrambled in samples:
        if held:continue
        base[key][0]+=y;base[key][1]+=1
        for name,ident in [('canonical',canonical),('scrambled',scrambled)]:tables[name][key+ident][0]+=y;tables[name][key+ident][1]+=1
    predictions={k:[] for k in ['compressed_bins','canonical','scrambled']};labels=[];bycase=defaultdict(list)
    for case,held,y,key,canonical,scrambled in samples:
        if not held:continue
        pos,n=base[key];bp=(pos+1)/(n+2);ps={'compressed_bins':bp}
        for name,ident in [('canonical',canonical),('scrambled',scrambled)]:
            pos,n=tables[name][key+ident];ps[name]=(pos+20*bp)/(n+20)
        labels.append(y)
        for name,v in ps.items():predictions[name].append(v)
        bycase[case].append({name:-(y*np.log(v)+(1-y)*np.log1p(-v)) for name,v in ps.items()})
    y=np.asarray(labels);metrics={}
    for name,pred in predictions.items():
        pred=np.asarray(pred);metrics[name]={'log_loss':float(-(y*np.log(pred)+(1-y)*np.log1p(-pred)).mean()),'auc':float(roc_auc_score(y,pred))}
    better=int(sum(np.mean([r['canonical']-r['scrambled'] for r in rows])<0 for rows in bycase.values()))
    screen=bool(metrics['canonical']['log_loss']<metrics['scrambled']['log_loss'] and metrics['canonical']['auc']-metrics['scrambled']['auc']>=.01 and better>=.6*len(bycase))
    result={'status':'COMPLETE','association_screen_pass':screen,'metrics':metrics,'heldout_cases':len(bycase),'canonical_better_cases':better,'fit_cases':sum(not x['heldout'] for x in case_rows),'sampled_holdout':len(y),'sampled_holdout_positives':int(y.sum()),'seconds':time.monotonic()-start,'optimizer_updates':0,'test_used':False,'validation_used':False,'limits':'GT-stratified sampled diagnostic, not segmentation performance; only local scalar bins, not all legacy spatial features. Association is not causal or capacity evidence; canonical vs scrambled has same count-table capacity/smoothing.'}
    (a.out/'RESULT.json').write_text(json.dumps(result,indent=2));(a.out/'CASE_TRACE.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in case_rows));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
