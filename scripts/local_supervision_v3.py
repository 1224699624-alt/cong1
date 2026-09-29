"""Explicit observation masks for V3 training losses, never inference inputs."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from r350_unified_interaction_loss import active_mean, separation_loss


def local_supervision(logits: torch.Tensor, target: torch.Tensor,
                      member_known: torch.Tensor, union: torch.Tensor,
                      union_known: torch.Tensor, valid: torch.Tensor,
                      seam: torch.Tensor, teacher_probability: torch.Tensor,
                      *, enable_seam: bool, enable_overlap: bool,
                      threshold: float = .25, reference_mass: float = .01):
    """Return auxiliary terms/states without inventing missing memberships.

    target contains member labels when member_known is true. For TSRS its
    union channel must have member_known=false: one foreground channel is
    not a complete list of bone members. No weak-local threshold is invented.
    """
    if target.shape != logits.shape or member_known.shape != target.shape:
        raise ValueError('member target/known/logit shapes differ')
    shape = (logits.shape[0], 1, *logits.shape[-2:])
    for field in (union, union_known, valid, seam):
        if tuple(field.shape) != shape:
            raise ValueError('union/valid/seam coordinate mismatch')
    if reference_mass <= 0:
        raise ValueError('reference_mass must be positive')
    valid = valid.bool()
    member_known = member_known.bool() & valid
    union_known = union_known.bool() & valid
    known_positive = (target > .5) & member_known
    positive_overlap = known_positive.sum(1, keepdim=True) >= 2
    complete_members = member_known.all(1, keepdim=True)
    overlap_known = (positive_overlap | complete_members) & valid
    # Ignore padded values even in the support normalizer.
    lo = seam.masked_fill(~valid, float('inf')).amin((-2, -1), keepdim=True)
    hi = seam.masked_fill(~valid, -float('inf')).amax((-2, -1), keepdim=True)
    any_valid = valid.flatten(2).any(-1, keepdim=True).unsqueeze(-1)
    lo = torch.where(any_valid, lo, torch.zeros_like(lo))
    hi = torch.where(any_valid, hi, torch.zeros_like(hi))
    normalized = ((seam - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1).detach()
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    sep = union_known & ~union.bool() & (normalized >= threshold) if enable_seam else torch.zeros_like(valid)
    overlap = positive_overlap & overlap_known if enable_overlap else torch.zeros_like(valid)
    if torch.any(sep & overlap):
        raise ValueError('observed member union contradicts supplied union target')
    preserve = union_known & ~sep & ~overlap
    probability = logits.float().sigmoid()
    member_weight = overlap & member_known
    terms = {
        'separation': separation_loss(probability.amax(1, keepdim=True), sep),
        'overlap': active_mean(F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction='none'), member_weight),
        'preserve': active_mean((probability - teacher_probability.detach()).square(), preserve),
    }
    denominator = valid.sum().clamp_min(1)
    masses = {name: float(mask.float().sum().detach() / denominator)
              for name, mask in [('separation', sep), ('overlap', overlap), ('preserve', preserve)]}
    states = {'separation': sep, 'overlap': overlap, 'preserve': preserve,
              'overlap_known': overlap_known, 'member_weight': member_weight,
              'masses': masses, 'attenuation': {k: min(1., v / reference_mass) for k, v in masses.items()},
              'overlap_supervision': 'observed' if bool(overlap_known.any()) else 'unknown',
              'weak_local_term': 'NOT_VALIDATED_DISABLED'}
    return terms, states
