#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import torch
EXPECTED='65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 if sha(a.input)!=EXPECTED: raise RuntimeError('R202 checkpoint hash mismatch')
 c=torch.load(a.input,map_location='cpu',weights_only=False); source={k:v.clone() for k,v in c['network_weights'].items()}; changed=[]
 expected_keys={
  'encoder.stages.0.0.convs.0.conv.weight',
  'encoder.stages.0.0.convs.0.all_modules.0.weight',
  'decoder.encoder.stages.0.0.convs.0.conv.weight',
  'decoder.encoder.stages.0.0.convs.0.all_modules.0.weight',
 }
 for k,v in list(c['network_weights'].items()):
  if k in expected_keys:
   if tuple(v.shape)!=(32,1,3,3): raise RuntimeError((k,v.shape))
   new=torch.zeros((32,2,3,3),dtype=v.dtype); new[:,0]=v[:,0]; c['network_weights'][k]=new; changed.append(k)
 if set(changed)!=expected_keys: raise RuntimeError({'changed':changed,'expected':sorted(expected_keys)})
 a.output.parent.mkdir(parents=True,exist_ok=True); torch.save(c,a.output); check=torch.load(a.output,map_location='cpu',weights_only=False)
 state=check['network_weights']
 if set(state)!=set(source): raise RuntimeError('network key set changed during adaptation')
 for k,w0 in source.items():
  w1=state[k]
  if k in expected_keys:
   if not torch.equal(w1[:,0],w0[:,0]) or int(torch.count_nonzero(w1[:,1]))!=0: raise RuntimeError(f'bad expanded tensor: {k}')
  elif not torch.equal(w1,w0): raise RuntimeError(f'unexpected tensor change: {k}')
 heads=[k for k in state if 'seg_layers' in k]
 if not heads: raise RuntimeError('segmentation heads missing')
 print(json.dumps({'input_sha256':EXPECTED,'output_sha256':sha(a.output),'changed':sorted(changed),'network_key_count':len(state),'unchanged_tensor_count':len(state)-len(changed),'xray_channel_exact':True,'prior_channel_nonzero':0,'segmentation_head_tensor_count':len(heads)},indent=2))
if __name__=='__main__': main()
