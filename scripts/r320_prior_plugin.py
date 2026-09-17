"""Shared R320 prior-loss utilities for architecture-independent experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def foreground_logits(output: object) -> torch.Tensor:
    """Return one-channel foreground-vs-background logit odds."""
    if isinstance(output, dict):
        output = output["mask"]
    if isinstance(output, (list, tuple)):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 4:
        raise TypeError(f"unsupported segmentation output: {type(output)!r}")
    if output.shape[1] == 1:
        return output
    if output.shape[1] == 2:
        return output[:, 1:2].float() - output[:, 0:1].float()
    raise ValueError(f"expected one or two output channels, got {tuple(output.shape)}")


def continuous_background_prior_loss(
    output: object,
    target: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    """Exact R317 background-only continuous-prior loss."""
    logits = foreground_logits(output).float()
    prior = prior.float()
    lo = prior.amin(dim=(2, 3), keepdim=True)
    hi = prior.amax(dim=(2, 3), keepdim=True)
    normalized = (prior - lo) / (hi - lo + 1e-6)
    weight = normalized.square() * (target < 0.5).float()
    numerator = (weight * F.softplus(logits)).flatten(1).sum(1)
    denominator = weight.flatten(1).sum(1)
    valid = denominator > 1e-6
    return (
        (numerator[valid] / (denominator[valid] + 1e-6)).mean()
        if valid.any()
        else logits.sum() * 0.0
    )


def binary_dice_bce_loss(output: object, target: torch.Tensor) -> torch.Tensor:
    logits = foreground_logits(output)
    positive_fraction = target.mean().detach().clamp(1e-4, 0.5)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=(1.0 - positive_fraction) / positive_fraction,
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * target).flatten(1).sum(1)
    denominator = probability.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


class ZeroInitializedPriorAdapter(nn.Module):
    """Map [X-ray, prior] to one channel while preserving X-ray-only output."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.projection.weight.zero_()
            self.projection.weight[0, 0, 0, 0] = 1.0

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.projection(data)

    def audit(self) -> dict[str, float]:
        weight = self.projection.weight.detach().cpu()
        return {
            "xray_weight": float(weight[0, 0, 0, 0]),
            "prior_weight": float(weight[0, 1, 0, 0]),
        }
