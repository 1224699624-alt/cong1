"""Isolated full-parameter-space routing; legacy R349 remains unchanged."""
from __future__ import annotations
import math
import torch


def _dot(a, b, device):
    return sum(((x.float()*y.float()).sum() for x,y in zip(a,b) if x is not None and y is not None),
               torch.zeros((),device=device))


def _norm(a, device):
    return _dot(a,a,device).clamp_min(0).sqrt()


def _grad(loss, params):
    if not params or not loss.requires_grad:
        return [None]*len(params)
    return [None if g is None else g.detach() for g in
            torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)]


def route_parameter_gradients_v3(parameters, base_loss, auxiliaries, *, max_aux_ratio=.15,
                                 max_merged_aux_ratio=.15, eps=1e-12):
    """Project in full parameter space, then cap individual and summed aux.

    None is zero for vector algebra, but remains None for optimizer assignment
    if no enabled loss uses that parameter. Near-zero base disables auxiliaries.
    The aggregate .15 cap is a new V3 rule, not a historical R349 guarantee.
    """
    for value in [max_aux_ratio, max_merged_aux_ratio, *[w for _,_,w in auxiliaries]]:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError('weights/caps must be finite and non-negative')
    if eps <= 0:
        raise ValueError('eps must be positive')
    params=[p for p in parameters if p.requires_grad]
    device=base_loss.device
    base=_grad(base_loss,params); bn=_norm(base,device)
    if not torch.isfinite(bn):
        raise ValueError('nonfinite base gradient')
    merged=[None]*len(params); rows=[]; raw_vectors=[]
    for name,loss,weight in auxiliaries:
        raw=_grad(loss,params) if weight and float(bn)>eps else [None]*len(params)
        rn=_norm(raw,device); dot=_dot(base,raw,device)
        if not torch.isfinite(rn):
            raise ValueError('nonfinite auxiliary gradient')
        coeff=dot.clamp_max(0)/bn.square().clamp_min(eps**2)
        projected=[]
        for b,g in zip(base,raw):
            if g is None:
                projected.append(-coeff*b if b is not None and float(dot)<0 else None)
            else:
                projected.append(g if b is None else g-coeff*b)
        pn=_norm(projected,device)
        scale=min(float(weight),float(max_aux_ratio*bn/pn.clamp_min(eps))) if float(bn)>eps else 0.
        applied=[None if g is None or scale==0 else g*scale for g in projected]
        merged=[g if a is None else a if g is None else a+g for a,g in zip(merged,applied)]
        rows.append({'name':name,'base_norm':float(bn),'raw_norm':float(rn),
                     'projected_norm':float(pn),'applied_norm_before_merged_cap':float(pn)*scale,
                     'cosine_before':float(dot/(bn*rn).clamp_min(eps)),
                     'cosine_after':float(_dot(base,projected,device)/(bn*pn).clamp_min(eps)),
                     'projected_dot_base':float(_dot(base,projected,device)), 'scale':scale,
                     'skipped':weight==0 or float(bn)<=eps})
        raw_vectors.append(raw)
    mn=_norm(merged,device)
    merged_scale=1. if max_merged_aux_ratio is None else min(1.,float(max_merged_aux_ratio*bn/mn.clamp_min(eps)))
    merged=[None if g is None else g*merged_scale for g in merged]
    combined=[g if b is None else b if g is None else b+g for b,g in zip(base,merged)]
    pairs={f'{auxiliaries[i][0]}:{auxiliaries[j][0]}':float(_dot(raw_vectors[i],raw_vectors[j],device)/
           (_norm(raw_vectors[i],device)*_norm(raw_vectors[j],device)).clamp_min(eps))
           for i in range(len(raw_vectors)) for j in range(i+1,len(raw_vectors))}
    rows.append({'name':'merged_summary','base_norm':float(bn),'individual_cap':max_aux_ratio,
                 'aggregate_cap':max_merged_aux_ratio,'merged_scale':merged_scale,
                 'merged_aux_norm':float(_norm(merged,device)),
                 'merged_aux_base_ratio':float(_norm(merged,device)/bn.clamp_min(eps)),
                 'merged_aux_dot_base':float(_dot(base,merged,device)),
                 'raw_auxiliary_pair_cosines':pairs,'unused_parameters':sum(g is None for g in combined)})
    return combined,rows


def assign_gradients_v3(parameters, gradients):
    params=[p for p in parameters if p.requires_grad]
    if len(params)!=len(gradients):
        raise ValueError('gradient/parameter length mismatch')
    for p,g in zip(params,gradients):
        p.grad=None if g is None else g.to(dtype=p.dtype,device=p.device)
