#!/usr/bin/env python3
"""R290: frozen nnU-Net + high-resolution local overlap residual adapter.

RAM-W600 only. 14 independent sigmoid channels are preserved. The mature
R285 multi-label nnU-Net is frozen; only a small image/probability adapter is
trained on overlap-weighted supervision. This deliberately avoids the prior
R286-R288 full-image relation heads and gap/overlap/uncertain classifiers.
"""
from __future__ import annotations

import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import (
    NnUNetMultiLabelPrior, WristDataset, BONE_NAMES, seed_everything,
    validate, test_metrics_and_predictions, render_comparisons,
)


class LocalOverlapAdapter(nn.Module):
    """Small full-resolution residual branch; gate is learned from local evidence."""
    def __init__(self, in_channels: int = 15, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 15, 1),
        )
        # Start as exact baseline: both residual and gate are zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor, base_prob: torch.Tensor):
        raw = self.net(torch.cat([image, base_prob], dim=1))
        gate = torch.sigmoid(raw[:, 14:15])
        residual = torch.tanh(raw[:, :14]) * 0.75
        return residual, gate


def overlap_mask(target: torch.Tensor) -> torch.Tensor:
    return (target.sum(1, keepdim=True) >= 2).float()


def weighted_multilabel_loss(logits: torch.Tensor, base_logits: torch.Tensor,
                             target: torch.Tensor, gate: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Global BCE+Dice plus overlap-only BCE/Dice; consistency outside gate."""
    prob = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    tgt = target.float()
    axes = (0, 2, 3)
    dice = 1 - ((2 * (prob*tgt).sum(axes) + 1) / (prob.sum(axes)+tgt.sum(axes)+1)).mean()
    bce = F.binary_cross_entropy_with_logits(logits.float(), tgt)
    ov = overlap_mask(tgt).bool()
    # Count each bone channel independently in overlap pixels: multi-label target preserved.
    ov_expand = ov.expand_as(tgt)
    ov_prob = prob[ov_expand]
    ov_tgt = tgt[ov_expand]
    overlap_bce = F.binary_cross_entropy(ov_prob, ov_tgt) if ov_prob.numel() else prob.new_zeros(())
    inter = (prob * tgt * ov_expand).sum()
    pred_mass = (prob * ov_expand).sum(); tgt_mass = (tgt * ov_expand).sum()
    overlap_dice = 1 - (2*inter + 1) / (pred_mass+tgt_mass+1)
    outside = (1 - gate).expand_as(tgt)
    consistency = ((logits.float()-base_logits.float()).abs() * outside).mean()
    total = dice + bce + 0.50*overlap_bce + 0.50*overlap_dice + 0.05*consistency
    return total, {"base": float((dice+bce).detach()), "overlap": float((overlap_bce+overlap_dice).detach()), "consistency": float(consistency.detach())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', type=Path, default=Path(r'G:\gutou\RAM-W600'))
    ap.add_argument('--baseline-checkpoint', type=Path, default=Path('outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth'))
    ap.add_argument('--output', type=Path, default=Path('outputs/ram_w600/r290_local_overlap_adapter'))
    ap.add_argument('--size', type=int, default=384); ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=24); ap.add_argument('--seed', type=int, default=290)
    args = ap.parse_args(); seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    if 'TSRS_RSNA' in str(args.dataset_root): raise RuntimeError('R290 is RAM-W600 only')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = WristDataset(args.dataset_root, 'train', args.size, True)
    val = WristDataset(args.dataset_root, 'val', args.size, False)
    test = WristDataset(args.dataset_root, 'test', args.size, False)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type=='cuda')
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    tel = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = NnUNetMultiLabelPrior().to(device)
    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload['model']); model.eval()
    for p in model.parameters(): p.requires_grad = False
    adapter = LocalOverlapAdapter().to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = None; best_path = args.output/'adapter_best.pth'; history = []
    for epoch in range(args.epochs):
        adapter.train(); sums = {k:0.0 for k in ('loss','base','overlap','consistency')}; start=time.time()
        for batch in tl:
            image=batch['image'].to(device); target=batch['mask'].to(device)
            with torch.no_grad(): base_logits,_=model(image,False); base_prob=torch.sigmoid(base_logits)
            residual,gate=adapter(image,base_prob)
            logits=base_logits + residual*gate
            loss,parts=weighted_multilabel_loss(logits,base_logits,target,gate)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(adapter.parameters(),5.0); opt.step()
            sums['loss']+=float(loss.detach());
            for k in ('base','overlap','consistency'): sums[k]+=parts[k]
        sched.step(); adapter.eval()
        # validation predictions use exactly fixed 0.5 threshold; no test used.
        inter=torch.zeros(14,device=device); pm=torch.zeros(14,device=device); tm=torch.zeros(14,device=device)
        ov_i=ov_p=ov_t=0.0
        with torch.no_grad():
            for batch in vl:
                image=batch['image'].to(device); target=batch['mask'].to(device)
                base,_=model(image,False); res,gate=adapter(image,torch.sigmoid(base)); out=base+res*gate
                pred=torch.sigmoid(out)>=0.5; tb=target>0.5
                inter+=(pred&tb).sum((0,2,3)); pm+=pred.sum((0,2,3)); tm+=tb.sum((0,2,3))
                po=pred.sum(1)>=2; to=tb.sum(1)>=2; ov_i+=float((po&to).sum()); ov_p+=float(po.sum()); ov_t+=float(to.sum())
        vd=float(((2*inter+1)/(pm+tm+1)).mean()); vod=float((2*ov_i+1)/(ov_p+ov_t+1))
        row={'epoch':epoch,'seconds':time.time()-start,'lr':opt.param_groups[0]['lr'],'val_macro_dice':vd,'val_overlap_dice':vod,**{f'train_{k}':v/len(tl) for k,v in sums.items()}}
        history.append(row); (args.output/'history.jsonl').open('a',encoding='utf8').write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
        key=(vod if vd >= float(payload['metrics']['macro_dice'])-0.002 else -1.0)
        if best is None or key>best[0]: best=(key,epoch); torch.save({'adapter':adapter.state_dict(),'metrics':row,'epoch':epoch},best_path)
    # One final evaluation after validation-only selection.
    adapter.load_state_dict(torch.load(best_path,map_location=device,weights_only=False)['adapter']); adapter.eval()
    def collect(loader):
        preds={}; targets={}
        with torch.no_grad():
            for batch in loader:
                base_t=batch['image'].to(device); base,_=model(base_t,False); res,gate=adapter(base_t,torch.sigmoid(base)); out=base+res*gate
                for b,c in enumerate(batch['case']): preds[str(c)]=(torch.sigmoid(out[b])>=0.5).cpu().numpy(); targets[str(c)]=(batch['mask'][b].numpy()>0.5)
        return {'predictions':preds,'targets':targets}
    base_pred=collect(tel)
    # baseline predictions from the frozen model
    def collect_base(loader):
        preds={}; targets={}
        with torch.no_grad():
            for batch in loader:
                out,_=model(batch['image'].to(device),False)
                for b,c in enumerate(batch['case']): preds[str(c)]=(torch.sigmoid(out[b])>=0.5).cpu().numpy(); targets[str(c)]=(batch['mask'][b].numpy()>0.5)
        return {'predictions':preds,'targets':targets}
    raw=collect_base(tel)
    # reuse metric helper through a lightweight wrapper is unnecessary; compute official-like values directly.
    def metrics(bundle):
        pred=bundle['predictions']; tar=bundle['targets']; cd=[]; od=[]; on=[]
        from train_r284_ramw600_overlap_prior import surface_dice
        for c in sorted(pred):
            p=pred[c]; t=tar[c]; cd.append(float(np.mean([(2*(p[k]&t[k]).sum()+1)/(p[k].sum()+t[k].sum()+1) for k in range(14)])))
            po=p.sum(0)>=2; to=t.sum(0)>=2; od.append(float((2*(po&to).sum()+1)/(po.sum()+to.sum()+1))); on.append(surface_dice(po,to,2))
        return {'macro_dice':float(np.mean(cd)),'overlap_dice':float(np.mean(od)),'overlap_nsd_2px':float(np.mean(on))}
    result={'experiment':'R290','dataset':str(args.dataset_root),'device':str(device),'baseline_checkpoint':str(args.baseline_checkpoint),'best_epoch':best[1],'baseline_test':metrics(raw),'improved_test':metrics(base_pred),'delta_test':{k:metrics(base_pred)[k]-metrics(raw)[k] for k in metrics(raw)}}
    (args.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    cases=sorted(raw['targets'],key=lambda c:int((raw['targets'][c].sum(0)>=2).sum()),reverse=True)[:12]
    render_comparisons(args.dataset_root,args.output/'visualizations',cases,raw,base_pred,args.size)
    (args.output/'manifest.json').write_text(json.dumps({'experiment':'R290','architecture':'frozen R285 nnU-Net + local image/probability residual adapter','overlap_loss':'0.5 BCE + 0.5 Dice on multi-label overlap pixels','test_used_for_selection':False,'threshold':0.5},indent=2),encoding='utf8')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
