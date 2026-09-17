"""Shared state-conditioned structural losses for the R350 TSRS/RAM pair.

The datasets use different output representations, but they share the same
three mutually exclusive states and the same mass-aware weighting rule:
separation, overlap/completion, and preserve.  Dataset adapters are only
responsible for producing the state masks and a positive foreground logit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class StateWeights:
    separation: float
    overlap: float
    preserve: float


def assert_disjoint(separation: torch.Tensor, overlap: torch.Tensor) -> None:
    if separation.shape != overlap.shape:
        raise ValueError(f"state shape mismatch: {separation.shape} != {overlap.shape}")
    if torch.any((separation > 0) & (overlap > 0)):
        raise RuntimeError("separation and overlap states are not mutually exclusive")


def preserve_state(separation: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
    assert_disjoint(separation, overlap)
    return 1.0 - ((separation > 0) | (overlap > 0)).float()


def active_mean(values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    weight = state.float()
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(1)
    if weight.shape[1] == 1 and values.shape[1] != 1:
        weight = weight.expand(-1, values.shape[1], -1, -1)
    return (weight * values).sum() / weight.sum().clamp_min(1.0)


def separation_loss(foreground_probability: torch.Tensor, state: torch.Tensor,
                    ceiling: float = 0.10) -> torch.Tensor:
    return active_mean(F.relu(foreground_probability.float() - ceiling).square(), state)


def completion_loss(positive_logit: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Encourage foreground membership only inside a trusted completion state."""
    return active_mean(F.softplus(-positive_logit.float()), state)


def distribution_preserve_loss(student: torch.Tensor, teacher: torch.Tensor,
                               state: torch.Tensor) -> torch.Tensor:
    return active_mean((student.float() - teacher.float()).square(), state)


def mass_emphasis(state: torch.Tensor, reference_mass: float = 0.01) -> float:
    """Dataset-agnostic attenuation for rare states.

    Losses are normalized by active pixels so small structures remain
    learnable.  This multiplier still prevents a tiny, uncertain state from
    receiving the same global influence as a prevalent state.  It saturates at
    one and contains no dataset-name branch.
    """
    if reference_mass <= 0:
        raise ValueError("reference_mass must be positive")
    mass = state.detach().float().mean()
    scale = (mass / reference_mass).clamp(min=0.0, max=1.0)
    return float(scale)


def state_weights(separation: torch.Tensor, overlap: torch.Tensor, preserve: torch.Tensor,
                  structural_weight: float, preserve_weight: float,
                  reference_mass: float = 0.01) -> StateWeights:
    return StateWeights(
        separation=structural_weight * mass_emphasis(separation, reference_mass),
        overlap=structural_weight * mass_emphasis(overlap, reference_mass),
        preserve=preserve_weight * mass_emphasis(preserve, reference_mass),
    )
