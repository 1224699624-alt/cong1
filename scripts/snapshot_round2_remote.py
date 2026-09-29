"""Read-only remote train/val inventory; writes only its evidence JSON."""
import argparse,json,hashlib,shutil,subprocess
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report={'execution':'REMOTE current read-only snapshot','test_read':False,'datasets':{}}
    for domain,root,ext in [('TSRS',a.root/'data/raw/TSRS_RSNA-Epiphysis','.png'),('RAM',a.root/'data/remote_variants/RAM-W600/BoneSegmentation/masks','.npy')]:
        report['datasets'][domain]={}
        for split in ('train','val'):
            folder=root/(split+'_labels' if domain=='TSRS' else split)
            files=sorted(folder.glob('*'+ext));ids=[f.stem for f in files]
            report['datasets'][domain][split]={'count':len(files),'ids':ids,'duplicate_ids':[],
                                              'id_list_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        report['datasets'][domain]['id_overlap']=sorted(set(report['datasets'][domain]['train']['ids'])&set(report['datasets'][domain]['val']['ids']))
    for key,rel in [('1833_label','data/raw/TSRS_RSNA-Epiphysis/train_labels/1833.png'),('1833_image','data/raw/TSRS_RSNA-Epiphysis/train/1833.jpg')]:
        q=a.root/rel;report[key]={'path':str(q),'exists':q.exists(),'sha256':sha(q) if q.exists() else None}
    report['disk']={str(p):dict(zip(('total','used','free'),shutil.disk_usage(p))) for p in (Path('/'),Path('/root/autodl-tmp'))}
    report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.write_text(json.dumps(report,indent=2));print('remote snapshot complete')


if __name__=='__main__':main()
