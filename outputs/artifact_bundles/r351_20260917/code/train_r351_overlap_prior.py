"""Learn transferable image-conditioned overlap evidence from RAM train only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from r351_data import AlignedRamDataset
from r351_dual_prior import ImageOverlapPrior
from train_r327_ram_native_instance_completion_prior import seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--prior-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--audit', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'result.json').exists():
        raise RuntimeError('Refusing to overwrite completed run')
    seed_all(3510)
    model = ImageOverlapPrior().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    datasets = {s: AlignedRamDataset(a.dataset_root, a.prior_root, s) for s in ('train','val')}
    if a.audit:
        for d in datasets.values(): d.mask_files = d.mask_files[:2]
    loaders = {s:DataLoader(d,batch_size=1,shuffle=s=='train',num_workers=0) for s,d in datasets.items()}
    best = float('inf')
    history=[]
    for epoch in range(1,(1 if a.audit else a.epochs)+1):
        row={'epoch':epoch}
        for split,loader in loaders.items():
            model.train(split=='train')
            sums={'bce':0.,'positive_probability':0.,'negative_probability':0.,'overlap_fraction':0.}
            for b in loader:
                image,seam,valid=[b[k].cuda() for k in ('image','seam','valid')]
                target=(b['mask'].sum(1,keepdim=True)>=2).float().cuda()
                # Train a low-resolution local overlap map with soft coverage
                # targets, never labels derived from a teacher prediction.
                image=F.interpolate(image,(192,192),mode='bilinear',align_corners=False)
                seam=F.interpolate(seam,(192,192),mode='bilinear',align_corners=False)
                target=F.interpolate(target,(192,192),mode='area')
                valid=F.interpolate(valid.float(),(192,192),mode='nearest')
                with torch.set_grad_enabled(split=='train'):
                    z=model(image,seam)
                    # Unweighted proper scoring rule: avoid calling a class-
                    # weighted sigmoid a calibrated overlap probability.
                    loss=(F.binary_cross_entropy_with_logits(z,target,reduction='none')*valid).sum()/valid.sum().clamp_min(1)
                    if split=='train':
                        opt.zero_grad(set_to_none=True);loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                q=torch.sigmoid(z.detach()); pos=target*valid; neg=(1-target)*valid
                sums['bce']+=float(loss.detach())
                sums['positive_probability']+=float((q*pos).sum()/pos.sum().clamp_min(1))
                sums['negative_probability']+=float((q*neg).sum()/neg.sum().clamp_min(1))
                sums['overlap_fraction']+=float(pos.sum()/valid.sum().clamp_min(1))
            row[split]={k:v/len(loader) for k,v in sums.items()}
        history.append(row); print(json.dumps(row),flush=True)
        if row['val']['bce']<best:
            best=row['val']['bce']
            torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':row,
                        'source':'RAM train true multi-label overlap','tsrs_overlap_labels':'unknown'},a.output/'best.pth')
        (a.output/'history.json').write_text(json.dumps(history,indent=2))
    result={'test_used':False,'seed':3510,'train_cases':len(datasets['train']),'val_cases':len(datasets['val']),
            'best':torch.load(a.output/'best.pth',map_location='cpu',weights_only=False)['metrics'],
            'sha256':hashlib.sha256((a.output/'best.pth').read_bytes()).hexdigest(),
            'tsrs_transfer':'unvalidated weak prior, not observed overlap annotation'}
    (a.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__': main()
