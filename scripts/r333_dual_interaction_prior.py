"""Shared interaction-prior components for R333 RAM and TSRS adapters."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_r327_ram_native_instance_completion_prior import Block, image_gradient


class DualInteractionRefiner(nn.Module):
    """Seven-channel shared refiner: image, instance/context, uncertainty, gradient, seam."""

    def __init__(self, width: int = 12, residual_limit: float = 2.5) -> None:
        super().__init__(); self.residual_limit = residual_limit
        self.enc1 = Block(7, width); self.enc2 = Block(width, width * 2, 2)
        self.enc3 = Block(width * 2, width * 4, 2); self.mid = Block(width * 4, width * 4)
        self.dec2 = Block(width * 6, width * 2); self.dec1 = Block(width * 3, width)
        self.residual = nn.Conv2d(width, 1, 1); self.gate = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, feature: torch.Tensor, source_logit: torch.Tensor):
        e1=self.enc1(feature); e2=self.enc2(e1); e3=self.enc3(e2); value=self.mid(e3)
        value=F.interpolate(value,size=e2.shape[-2:],mode="bilinear",align_corners=False)
        value=self.dec2(torch.cat((value,e2),1))
        value=F.interpolate(value,size=e1.shape[-2:],mode="bilinear",align_corners=False)
        value=self.dec1(torch.cat((value,e1),1)); gate=torch.sigmoid(self.gate(value))
        delta=self.residual_limit*torch.tanh(self.residual(value))*gate
        return source_logit+delta,delta


def initialize_from_six_channel(model: DualInteractionRefiner, state: dict) -> dict:
    """Strictly transfer R330/R332 refiner, zero-initializing only the seam channel."""
    new=model.state_dict(); copied=[]
    for key,value in state.items():
        if key == "enc1.net.0.weight":
            if value.shape[1] != 6 or new[key].shape[1] != 7: raise RuntimeError((key,value.shape,new[key].shape))
            new[key].zero_(); new[key][:,:6].copy_(value); copied.append(key); continue
        if key not in new or new[key].shape != value.shape: raise RuntimeError((key,value.shape,new.get(key, None)))
        new[key].copy_(value); copied.append(key)
    model.load_state_dict(new,strict=True)
    return {"copied_tensors":len(copied),"seam_channel_nonzero":int(torch.count_nonzero(new["enc1.net.0.weight"][:,6]))}


def instance_features(image, probabilities, indices, source_probability, seam):
    selected=source_probability[0].unsqueeze(1); contexts=[]
    for index in indices.tolist():
        other=torch.cat((probabilities[:,:index],probabilities[:,index+1:]),1)
        contexts.append(torch.stack((other.max(1).values[0],other.sum(1)[0].clamp(0,2)/2),0))
    context=torch.stack(contexts); count=len(indices); uncertainty=4*selected*(1-selected)
    return torch.cat((image.repeat(count,1,1,1),selected,context,uncertainty,
                      image_gradient(image).repeat(count,1,1,1),seam.repeat(count,1,1,1)),1)


def binary_features(image, probability, seam):
    zero=torch.zeros_like(probability); uncertainty=4*probability*(1-probability)
    return torch.cat((image,probability,zero,zero,uncertainty,image_gradient(image),seam),1)


def state_masks(target: torch.Tensor, seam: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    """Mutually exclusive separation/overlap/uncertain masks from supported labels."""
    seam=(seam-seam.amin((-2,-1),keepdim=True))/(seam.amax((-2,-1),keepdim=True)-seam.amin((-2,-1),keepdim=True)+1e-6)
    membership=target.sum(1,keepdim=True)
    overlap=(membership>=2).float()
    separation=seam.square()*(membership<0.5).float()*(1-overlap)
    uncertain=(1-(separation>0).float()-overlap).clamp(0,1)
    if torch.any((separation>0)&(overlap>0)): raise RuntimeError("separation/overlap state conflict")
    return separation,overlap,uncertain


def absolute_seam_loss(logits: torch.Tensor, separation: torch.Tensor, ceiling: float=0.10):
    """Penalize foreground odds only above an absolute probability ceiling."""
    probability=torch.sigmoid(logits.float())
    excess=F.relu(probability-ceiling)
    weight=separation.expand(-1,logits.shape[1],-1,-1)
    return (weight*excess.square()).sum()/weight.sum().clamp_min(1.0)

