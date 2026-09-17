"""Chunked nnU-Net inference exporting only seam probabilities for R279."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    p=argparse.ArgumentParser(); p.add_argument('--images',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--work',type=Path,required=True); p.add_argument('--chunk-size',type=int,default=64); p.add_argument('--checkpoint',default='checkpoint_best.pth'); a=p.parse_args()
    files=sorted(a.images.glob('*_0000.png')); a.output.mkdir(parents=True,exist_ok=True); manifest=[]
    for start in range(0,len(files),a.chunk_size):
        chunk=files[start:start+a.chunk_size]; inp=a.work/'input'; pred=a.work/'pred'; shutil.rmtree(a.work,ignore_errors=True); inp.mkdir(parents=True); pred.mkdir(parents=True)
        for src in chunk: os.symlink(src.resolve(),inp/src.name)
        cmd=['nnUNetv2_predict','-i',str(inp),'-o',str(pred),'-d','274','-c','2d','-f','0','-tr','nnUNetTrainerR276WeightedSeam','-chk',a.checkpoint,'--save_probabilities','-npp','2','-nps','2']
        subprocess.run(cmd,check=True)
        for src in chunk:
            case=src.name[:-9]; npz=pred/f'{case}.npz'; prob=np.load(npz)['probabilities']; seam=prob[2,0] if prob.ndim==4 else prob[2]
            split,stem=case.split('_',1); out=a.output/split/f'{stem}.png'; out.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(np.clip(seam*255,0,255).astype(np.uint8)).save(out); manifest.append({'case':case,'split':split,'max':float(seam.max()),'mean':float(seam.mean())})
        print(json.dumps({'done':min(start+len(chunk),len(files)),'total':len(files)}),flush=True)
    shutil.rmtree(a.work,ignore_errors=True); (a.output/'manifest.json').write_text(json.dumps({'cases':len(manifest),'clean_test_used':False,'rows':manifest},indent=2),encoding='utf-8')
if __name__=='__main__': main()
