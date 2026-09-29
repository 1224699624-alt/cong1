"""V3 joint image/heatmap input and independently image-conditioned overlap.

No historical weights are advertised as trained image-only priors. C/D are
blocked outside mechanical tests until a compatible validated checkpoint exists.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from r351_dual_prior import ImageOverlapPrior
from r333_dual_interaction_prior import DualInteractionRefiner
from train_r327_ram_native_instance_completion_prior import Block, image_gradient


def build_backbone_input(image, seam, arm):
    if arm not in 'ABCD' or len(arm)!=1:
        raise ValueError('arm must be A/B/C/D')
    if image.ndim!=4 or image.shape[1]!=1:
        raise ValueError('image must be B1HW')
    # A/C do not inspect or load even a malformed seam cache.
    if arm in ('A','C'):
        return torch.cat((image,torch.zeros_like(image)),1)
    if seam is None or seam.shape!=image.shape or not torch.isfinite(seam).all():
        raise ValueError('missing or misaligned heatmap; no implicit resize')
    return torch.cat((image,seam),1)


class ImageOnlyOverlapPrior(ImageOverlapPrior):
    version='image_only_overlap_v3_untrained'

    def __init__(self):
        super().__init__(); self.first=Block(1,8)

    def initialize_from_legacy(self,state):
        copied={k:v.clone() for k,v in state.items()}
        key='first.net.0.weight'
        if copied[key].shape[1]!=2:
            raise ValueError('expected legacy image+seam first layer')
        copied[key]=copied[key][:,:1].clone()
        self.load_state_dict(copied,strict=True)
        return {'status':'BLOCKED_NOT_ADAPTED','operation':'copy image channel; remove seam channel',
                'changed_key':key,'trained_image_only':False}

    def forward(self,image):
        shape=image.shape[-2:]
        x=F.interpolate(image,(192,192),mode='bilinear',align_corners=False)
        first=self.first(x); context=self.third(self.second(first))
        context=F.interpolate(context,first.shape[-2:],mode='bilinear',align_corners=False)
        return F.interpolate(self.last(torch.cat((first,context),1)),shape,mode='bilinear',align_corners=False)


class OutputOverlapAdapter(DualInteractionRefiner):
    """Reuse existing seven-channel refiner, seventh feature is O itself."""
    def refine(self,image,logits,overlap,steps=1):
        if overlap.shape!=(image.shape[0],1,*image.shape[-2:]):
            raise ValueError('overlap coordinates differ')
        if logits.shape[0]!=image.shape[0] or logits.shape[-2:]!=image.shape[-2:]:
            raise ValueError('image/logits coordinates differ')
        if steps<1:
            raise ValueError('steps must be positive')
        current=logits
        edge=image_gradient(image)
        for _ in range(steps):
            probability=current.sigmoid(); pieces=[]
            for k in range(logits.shape[1]):
                own=probability[:,k:k+1]
                other=torch.cat((probability[:,:k],probability[:,k+1:]),1)
                mx=other.amax(1,keepdim=True) if other.shape[1] else torch.zeros_like(own)
                sm=other.sum(1,keepdim=True).clamp(0,2)/2 if other.shape[1] else torch.zeros_like(own)
                feature=torch.cat((image,own,mx,sm,4*own*(1-own),edge,overlap),1)
                if self.training and torch.is_grad_enabled():
                    corrected,_=checkpoint(super().forward,feature,current[:,k:k+1],use_reentrant=False)
                else:
                    corrected,_=super().forward(feature,current[:,k:k+1])
                pieces.append(corrected)
            current=torch.cat(pieces,1)
        return current


class StructuralPriorV3(nn.Module):
    def __init__(self,arm,*,overlap_generator=None,output_refiner=None,mechanical_test_only=False):
        super().__init__()
        if arm not in ('A','B','C','D'):
            raise ValueError(arm)
        self.arm=arm
        if arm in ('C','D'):
            if not mechanical_test_only:
                raise RuntimeError('BLOCKED: no validated image-only checkpoint loader yet')
            if overlap_generator is None or output_refiner is None:
                raise ValueError('C/D mechanical tests need explicitly labelled test modules')
            self.overlap_generator=overlap_generator
            self.output_refiner=output_refiner
        else:
            self.overlap_generator=None;self.output_refiner=None

    def backbone_input(self,image,seam=None):
        return build_backbone_input(image,seam,self.arm)

    def refine_output(self,image,logits):
        if self.arm in ('A','B'):
            return logits
        overlap=self.overlap_generator(image).sigmoid()
        return self.output_refiner.refine(image,logits,overlap)
