"""Parameter-space gradient routing for the loss-only R349 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class RoutedGradientStats:
    name: str
    base_norm: float
    raw_norm: float
    projected_norm: float
    applied_norm: float
    cosine_before: float
    cosine_after: float
    conflicted: bool
    scale: float


def _dot(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> torch.Tensor:
    values = [(a.float() * b.float()).sum() for a, b in zip(left, right) if a is not None and b is not None]
    if not values:
        device = next((x.device for x in left + right if x is not None), torch.device("cpu"))
        return torch.zeros((), device=device)
    return torch.stack(values).sum()


def _norm(values: list[torch.Tensor | None]) -> torch.Tensor:
    return _dot(values, values).clamp_min(0).sqrt()


def route_parameter_gradients(
    parameters: Iterable[torch.nn.Parameter],
    base_loss: torch.Tensor,
    auxiliaries: list[tuple[str, torch.Tensor, float]],
    *,
    max_aux_ratio: float = .15,
    eps: float = 1e-12,
) -> tuple[list[torch.Tensor | None], list[RoutedGradientStats]]:
    """Project conflicting auxiliary gradients and cap each against the base norm.

    This operates on trainable parameters rather than logits. A non-conflicting
    logit gradient can still become conflicting after the network Jacobian, so
    parameter-space routing is the quantity that protects the mature model.
    """
    params = [p for p in parameters if p.requires_grad]
    base = list(torch.autograd.grad(base_loss, params, retain_graph=True, allow_unused=True))
    base_norm = _norm(base).clamp_min(eps)
    combined = [None if g is None else g.detach().clone() for g in base]
    stats: list[RoutedGradientStats] = []

    for index, (name, loss, requested_weight) in enumerate(auxiliaries):
        # The caller consumes detached routed gradients, so the graph can be
        # released after the final auxiliary instead of surviving optimizer.step.
        retain_graph = index < len(auxiliaries) - 1
        raw = list(torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True))
        raw_norm = _norm(raw)
        dot_before = _dot(base, raw)
        cosine_before = dot_before / (base_norm * raw_norm.clamp_min(eps))
        conflicted = bool(dot_before.detach() < 0)
        if conflicted:
            coefficient = dot_before / base_norm.square().clamp_min(eps)
            projected = [None if g is None else g - coefficient * (b if b is not None else 0) for b, g in zip(base, raw)]
        else:
            projected = raw
        projected_norm = _norm(projected)
        ratio_cap = max_aux_ratio * base_norm / projected_norm.clamp_min(eps)
        scale_t = torch.minimum(torch.as_tensor(float(requested_weight), device=base_norm.device), ratio_cap)
        applied = [None if g is None else g.detach() * scale_t for g in projected]
        for index, value in enumerate(applied):
            if value is None:
                continue
            combined[index] = value.clone() if combined[index] is None else combined[index] + value
        dot_after = _dot(base, projected)
        cosine_after = dot_after / (base_norm * projected_norm.clamp_min(eps))
        stats.append(RoutedGradientStats(
            name=name,
            base_norm=float(base_norm.detach()),
            raw_norm=float(raw_norm.detach()),
            projected_norm=float(projected_norm.detach()),
            applied_norm=float((projected_norm * scale_t).detach()),
            cosine_before=float(cosine_before.detach()),
            cosine_after=float(cosine_after.detach()),
            conflicted=conflicted,
            scale=float(scale_t.detach()),
        ))
    return combined, stats


def assign_gradients(parameters: Iterable[torch.nn.Parameter], gradients: list[torch.Tensor | None]) -> None:
    params = [p for p in parameters if p.requires_grad]
    if len(params) != len(gradients):
        raise ValueError("gradient/parameter length mismatch")
    for parameter, gradient in zip(params, gradients):
        parameter.grad = None if gradient is None else gradient.to(parameter.dtype)
