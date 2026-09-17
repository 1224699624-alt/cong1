"""Shared logit-space gradient surgery and selective distillation for R336."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def project_conflicting_gradient(reference: torch.Tensor, auxiliary: torch.Tensor, eps: float = 1e-12):
    """Remove only the auxiliary component that opposes the reference gradient.

    Gradients are handled per sample at the prediction interface.  The returned
    tensor is detached and can be injected through ``(g * logits).sum()``.
    """
    g0 = reference.detach().float()
    g1 = auxiliary.detach().float()
    dims = tuple(range(1, g0.ndim))
    dot = (g0 * g1).sum(dims, keepdim=True)
    denom = g0.square().sum(dims, keepdim=True).clamp_min(eps)
    conflict = dot < 0
    projected = torch.where(conflict, g1 - dot / denom * g0, g1)
    stats = {
        "conflict_fraction": float(conflict.float().mean()),
        "cosine_before": float((dot / (denom.sqrt() * g1.square().sum(dims, keepdim=True).clamp_min(eps).sqrt())).mean()),
    }
    return projected.to(auxiliary.dtype), stats


def gradient_surrogate(logits: torch.Tensor, loss: torch.Tensor, reference_loss: torch.Tensor, weight: float):
    """Create a scalar whose logit gradient is a conflict-projected auxiliary gradient."""
    g0 = torch.autograd.grad(reference_loss, logits, retain_graph=True, allow_unused=False)[0]
    g1 = torch.autograd.grad(loss, logits, retain_graph=True, allow_unused=False)[0]
    projected, stats = project_conflicting_gradient(g0, g1)
    # Mean-based losses already encode their normalization in g1.
    surrogate = float(weight) * (projected.detach() * logits).sum()
    return surrogate, stats


def selective_binary_distillation(student_logits, teacher_logits, target, eligible, confidence=0.80):
    """Distil only teacher-correct, confident pixels inside an eligible mask."""
    with torch.no_grad():
        tp = torch.sigmoid(teacher_logits.float())
        correct = ((tp >= .5) == (target > .5)).float()
        confident = ((tp >= confidence) | (tp <= 1 - confidence)).float()
        mask = eligible.float() * correct * confident
    per_pixel = F.binary_cross_entropy_with_logits(student_logits.float(), tp, reduction="none")
    loss = (mask * per_pixel).sum() / mask.sum().clamp_min(1)
    return loss, float(mask.mean())


def selective_softmax_distillation(student_logits, teacher_logits, target, eligible, confidence=0.80):
    """Multiclass counterpart used by the TSRS nnU-Net trainer."""
    with torch.no_grad():
        tp = torch.softmax(teacher_logits.float(), dim=1)
        conf, pred = tp.max(1)
        mask = eligible.float() * (pred == target.long()).float() * (conf >= confidence).float()
    kl = -(tp * torch.log_softmax(student_logits.float(), dim=1)).sum(1)
    loss = (mask * kl).sum() / mask.sum().clamp_min(1)
    return loss, float(mask.mean())
