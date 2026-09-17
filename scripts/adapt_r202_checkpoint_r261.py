#!/usr/bin/env python3
"""Expand mature R202 first convolution from one to thirteen channels."""
import argparse,hashlib,json
from pathlib import Path
import torch
EXPECTED='65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7'
TARGET_KEYS={'encoder.stages.0.0.convs.0.conv.weight','encoder.stages.0.0.convs.0.all_modules.0.weight','decoder.encoder.stages.0.0.convs.0.conv.weight','decoder.encoder.stages.0.0.convs.0.all_modules.0.weight'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if sha(a.input)!=EXPECTED: raise RuntimeError('R202 checkpoint hash mismatch')
 c=torch.load(a.input,map_location='cpu',weights_only=False); source={k:v.clone() for k,v in c['network_weights'].items()}; changed=[]
 for k,v in list(c['network_weights'].items()):
  if k in TARGET_KEYS:
   if tuple(v.shape)!=(32,1,3,3): raise RuntimeError((k,v.shape))
   w=torch.zeros((32,13,3,3),dtype=v.dtype); w[:,0]=v[:,0]; c['network_weights'][k]=w; changed.append(k)
 if set(changed)!=TARGET_KEYS: raise RuntimeError(changed)
 a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(c,a.output); state=torch.load(a.output,map_location='cpu',weights_only=False)['network_weights']
 if set(state)!=set(source): raise RuntimeError('network key set changed')
 for k,v in source.items():
  if k in TARGET_KEYS:
   if not torch.equal(state[k][:,0],v[:,0]) or int(torch.count_nonzero(state[k][:,1:])): raise RuntimeError(k)
  elif not torch.equal(state[k],v): raise RuntimeError(k)
 heads=[k for k in state if 'seg_layers' in k]
 if not heads: raise RuntimeError('segmentation heads missing')
 print(json.dumps({'input_sha256':EXPECTED,'output_sha256':sha(a.output),'changed':sorted(changed),'unchanged_tensors':len(state)-len(changed),'xray_exact':True,'new_channels_nonzero':0,'segmentation_head_tensors':len(heads)},indent=2))
if __name__=='__main__': main()

