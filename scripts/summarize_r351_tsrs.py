"""Bind a TSRS R201 result to its actual selected checkpoint and baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--inference-policy', choices=['trained','positive_seam_guard'], default='trained')
    a=p.parse_args()
    basepath=Path('outputs/analysis/r350_native_nnunet_matched_original_val_r201.json')
    current=json.loads((a.run/'r201.json').read_text());baseline=json.loads(basepath.read_text())
    if current['num_images']!=96 or baseline['num_images']!=96:raise RuntimeError('case count mismatch')
    if {x['image'] for x in current['per_image']}!={x['image'] for x in baseline['per_image']}:raise RuntimeError('case set mismatch')
    b,m=baseline['mean'],current['mean']
    anchor=json.loads(Path('outputs/analysis/r350_tsrs_unified_best_original_val_r201.json').read_text())['mean']
    delta={k:m[k]-b[k] for k in b}
    checks={'dice_not_below_native':m['dice']>=b['dice'],'iou_not_below_native':m['iou']>=b['iou'],
            'gap_fp_relative_reduction_5pct':m['gap_region_fp_rate']<=b['gap_region_fp_rate']*.95,
            'merge_relative_reduction_5pct':m['component_merge_rate']<=b['component_merge_rate']*.95}
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    backbone=Path(os.environ['R351_TSRS_INIT'])
    compact={'adapter_weights':{k:v for k,v in payload['network_weights'].items() if not k.startswith('backbone.')},
             'frozen_backbone':str(backbone),'frozen_backbone_sha256':hashlib.sha256(backbone.read_bytes()).hexdigest(),
             'completed_epochs':payload['current_epoch']}
    torch.save(compact,a.run/'adapters.pth')
    result={'experiment':'R351_TSRS_DUAL_PRIOR_PILOT','split':'original-val','num_images':96,'test_used':False,
            'inference_policy':a.inference_policy,
            'checkpoint':str(a.checkpoint),'sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'checkpoint_completed_epochs':payload['current_epoch'],'checks':checks,'accepted':all(checks.values()),
            'baseline_file':str(basepath),'baseline_sha256':hashlib.sha256(basepath.read_bytes()).hexdigest(),
            'baseline':b,'metrics':m,'delta':delta,'r350_anchor':anchor,'delta_vs_r350':{k:m[k]-anchor[k] for k in m},
            'overlap_labels':'unknown; transferred prior not measured true TSRS overlap',
            'initialization':'R350 mature checkpoint; not native control'}
    (a.run/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
