"""Immutable Round3 protocol, data identity, and production prior gates."""
import hashlib,json,time,shutil,os
from pathlib import Path
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def load_lock(path):
    lock=json.loads(Path(path).read_text())
    if lock['schema']!='ROUND3_PROTOCOL_V1' or lock['test_used']:raise ValueError('invalid protocol')
    return lock
def validate_manifest(root,protocol):
    root=Path(root);protocol=Path(protocol);lock=load_lock(protocol)
    manifest=protocol.parent/'DATA_MANIFEST.json'
    if sha(manifest)!=lock['data_manifest_sha256']:raise ValueError('manifest hash mismatch')
    data=json.loads(manifest.read_text())['datasets'];checked={};start=time.time()
    for d,splits in data.items():
        checked[d]={}
        for s,rows in splits.items():
            ids=[r['case'] for r in rows]
            if len(ids)!=lock['data_counts'][d][s] or len(ids)!=len(set(ids)):raise ValueError('case count/duplicate')
            for r in rows:
                for kind in ['image','label','prior']:
                    rel=r[kind].replace('\\','/').split('/YOLO+SAM/',1)[1]
                    # Server has the exact original1833 label; local manifest
                    # uses same-SHA supplementary copy without replacing images.
                    if d=='TSRS' and r['case']=='1833' and kind=='label':
                        rel='data/raw/TSRS_RSNA-Epiphysis/train_labels/1833.png'
                    p=root/rel
                    if sha(p)!=r[kind+'_sha256']:raise ValueError(f'{d}/{s}/{r["case"]}/{kind} hash mismatch')
            checked[d][s]={'count':len(ids),'case_ids':ids}
        if set(checked[d]['train']['case_ids'])&set(checked[d]['val']['case_ids']):raise ValueError('train val overlap')
    result={'status':'PASS','manifest_sha256':sha(manifest),'protocol_sha256':sha(protocol),
            'datasets':checked,'seconds':time.time()-start,'test_used':False}
    path=protocol.parent/'DATA_MANIFEST_REMOTE_VERIFIED.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['manifest_sha256']!=result['manifest_sha256']:raise ValueError('existing audit mismatch')
    else:save(path,result)
    return result

def require_manifest(protocol):
    protocol=Path(protocol);lock=load_lock(protocol)
    x=json.loads((protocol.parent/'DATA_MANIFEST_REMOTE_VERIFIED.json').read_text())
    if x['status']!='PASS' or x['manifest_sha256']!=lock['data_manifest_sha256']:raise ValueError('data not locked')
    return x

def load_accepted_overlap(path,acceptance,device):
    from structural_prior_v3 import ImageOnlyOverlapPrior
    report=json.loads(Path(acceptance).read_text())
    if not report.get('accepted') or report.get('status')!='ACCEPTED_FOR_BOUNDED_PILOT':raise ValueError('prior not accepted')
    if report['checkpoint_sha256']!=sha(path):raise ValueError('prior SHA mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['input_dependencies']!=['image'] or payload['version']!='image_only_overlap_v3':raise ValueError('prior incompatible')
    model=ImageOnlyOverlapPrior().to(device);model.load_state_dict(payload['model'],strict=True)
    model.eval().requires_grad_(False)
    return model,report

def resource_guard(output,start,seconds=7200):
    if time.time()-start>seconds:raise TimeoutError('locked wall-time cap')
    if shutil.disk_usage(output).free<4*1024**3:raise RuntimeError('4GiB disk floor')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--protocol',required=True);a=p.parse_args()
    print(json.dumps(validate_manifest(a.root,a.protocol)))
