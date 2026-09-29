"""Function-preserving post-normalization member-relation conditioning for RAM14."""
import copy
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from r332_v31_port import R332RefinerWithO, features_with_o


class IdentityContextRefiner(R332RefinerWithO):
    @classmethod
    def from_mature(cls, mature):
        # Deepcopy without constructor RNG or reinitialization of mature layers.
        result=copy.deepcopy(mature)
        result.__class__=cls
        width=mature.residual.in_channels
        result.register_parameter('relation_weight',torch.nn.Parameter(
            torch.zeros((14,width,14),device=mature.residual.weight.device,dtype=mature.residual.weight.dtype)))
        result.identity_mode='canonical'
        result.relation_enabled=True
        result.last_context_stats={}
        return result

    def forward_context(self, feature, source_logit, probability, indices, permutation):
        e1=self.enc1(feature)
        inverse=torch.argsort(permutation)
        # Consistently relabel own query and all context channels; GT is never used.
        weights=self.relation_weight[inverse[indices]]
        context=torch.einsum('ncf,fhw->nchw',weights,probability[0,permutation])
        if not self.relation_enabled:context=context*0
        self.last_context_stats={'mean_abs':float(context.detach().abs().mean()),
            'max_abs':float(context.detach().abs().max()),'legacy_e1_mean_abs':float(e1.detach().abs().mean())}
        e1=e1+context
        e2=self.enc2(e1);e3=self.enc3(e2);value=self.mid(e3)
        value=F.interpolate(value,size=e2.shape[-2:],mode='bilinear',align_corners=False)
        value=self.dec2(torch.cat((value,e2),1))
        value=F.interpolate(value,size=e1.shape[-2:],mode='bilinear',align_corners=False)
        value=self.dec1(torch.cat((value,e1),1))
        delta=self.residual_limit*torch.tanh(self.residual(value))*torch.sigmoid(self.gate(value))
        return source_logit+delta,delta


def identity_permutation(image, mode):
    if mode=='canonical':return torch.arange(14,device=image.device)
    if mode!='scrambled':raise ValueError(mode)
    digest=hashlib.sha256(image.detach().float().cpu().contiguous().numpy().tobytes()).digest()
    # Private NumPy generator: leaves sampling/augmentation/Python/Torch RNG alone.
    seed=int.from_bytes(digest[:8],'little')
    return torch.as_tensor(np.random.default_rng(seed).permutation(14),device=image.device)


def iterative_refine_identity(image,base_logits,refiner,steps,overlap,indices=None,chunk=3):
    permutation=identity_permutation(image,refiner.identity_mode)
    states=[base_logits];deltas=[];current=base_logits
    for _ in range(steps):
        probability=current.sigmoid();pieces=[];all_delta=[]
        groups=[indices] if indices is not None else [torch.arange(s,min(s+chunk,14),device=image.device) for s in range(0,14,chunk)]
        for chosen in groups:
            feature=features_with_o(image,probability,chosen,probability[:,chosen],overlap)
            corrected,delta=refiner.forward_context(feature,current[0,chosen].unsqueeze(1),probability,chosen,permutation)
            correction=corrected[:,0]-current[0,chosen]
            scatter=chosen.view(1,-1,1,1).expand(1,-1,*current.shape[-2:])
            pieces.append(torch.zeros_like(current).scatter(1,scatter,correction.unsqueeze(0)));all_delta.append(delta)
        current=current+torch.stack(pieces).sum(0);states.append(current);deltas.append(torch.cat(all_delta,0))
    return states,deltas
