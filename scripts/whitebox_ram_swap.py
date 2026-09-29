"""Fixed-checkpoint train/val weight swaps; zero optimizer updates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import run_v31_ram_mature as legacy
from r332_v31_port import load_mature_with_zero_o
from whitebox_ram_diagnosis import write


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--lock',type=Path,required=True);p.add_argument('--trained',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();lock=json.loads(a.lock.read_text())
    if a.out.exists():raise FileExistsError(a.out)
    legacy.seed_all(lock['seed']);device=torch.device('cuda')
    payload,model,refiner,generator,_,_=legacy.load_components(SimpleNamespace(r332=Path(lock['r332']),overlap=Path(lock['overlap']),overlap_sha=lock['overlap_sha']),device)
    initial_model={k:v.cpu().clone() for k,v in model.state_dict().items()};initial_ref={k:v.cpu().clone() for k,v in refiner.state_dict().items()}
    trained=torch.load(a.trained,map_location='cpu',weights_only=False)
    data=Path(lock['dataset']);val=legacy.NativeWristDataset(data,'val',False);pairs=legacy.top_training_pairs(data,15)
    configurations={'initial':(initial_model,initial_ref),'new_backbone_old_refiner':(trained['model'],initial_ref),
                    'old_backbone_new_refiner':(initial_model,trained['refiner'])}
    result={'optimizer_updates':0,'mode':'CURRENT_CHECKPOINT_WEIGHT_SWAP_NOT_MATCHED_TRAINING','test_used':False,
            'trained_sha256':legacy.sha(a.trained),'configurations':{}}
    for name,(backbone_state,refiner_state) in configurations.items():
        model.load_state_dict(backbone_state);refiner.load_state_dict(refiner_state);model.eval();refiner.eval()
        predictions={};changes={k:np.zeros(14,dtype=np.int64) for k in ['recovered_fn','introduced_fn','removed_fp','introduced_fp']}
        for batch in DataLoader(val,batch_size=1,shuffle=False,num_workers=0):
            image=batch['image'].to(device);target=batch['mask'].numpy()[0]>.5
            base=model(torch.cat([image,torch.zeros_like(image)],1))
            states,_=legacy.iterative_refine_o(image,base,refiner,2,torch.zeros_like(image))
            b=(base.sigmoid()>=.5).cpu().numpy()[0];f=(states[-1].sigmoid()>=.5).cpu().numpy()[0]
            predictions[batch['case'][0]]={'pred':f,'target':target}
            changes['recovered_fn']+=(~b&f&target).sum((1,2));changes['introduced_fn']+=(b&~f&target).sum((1,2))
            changes['removed_fp']+=(b&~f&~target).sum((1,2));changes['introduced_fp']+=(~b&f&~target).sum((1,2))
        official=legacy.official_metrics(predictions,pairs)
        result['configurations'][name]={'official':official,'flat':legacy.flatten(official),
                                      'member_corrections':{k:v.tolist() for k,v in changes.items()}}
        print(json.dumps({'configuration':name,'metrics':result['configurations'][name]['flat']}),flush=True)
    write(a.out,result)


if __name__=='__main__':main()
