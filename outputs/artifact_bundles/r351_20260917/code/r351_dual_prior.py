"""Backbone-independent input/output priors and observable local supervision.

No target enters a forward method. TSRS overlap annotations are unknown, not
zero: its final union is supervised, but no binary foreground is relabelled
as a genuine overlapping instance membership.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from train_r327_ram_native_instance_completion_prior import Block, image_gradient
from r333_dual_interaction_prior import DualInteractionRefiner, initialize_from_six_channel
from r350_unified_interaction_loss import active_mean, separation_loss, completion_loss


def normalize_prior(prior: torch.Tensor) -> torch.Tensor:
    low = prior.amin((-2, -1), keepdim=True)
    high = prior.amax((-2, -1), keepdim=True)
    return ((prior - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)


class ImageOverlapPrior(nn.Module):
    """Image/seam-conditioned probability, trained on RAM multi-label truth."""

    def __init__(self):
        super().__init__()
        self.first = Block(2, 8)
        self.second = Block(8, 16, 2)
        self.third = Block(16, 24, 2)
        self.last = nn.Sequential(nn.Conv2d(32, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 1))

    def forward(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        shape = image.shape[-2:]
        # Fixed low-resolution context makes the prior independent of backbone
        # feature maps; native output predictions are never resized here.
        x = F.interpolate(torch.cat((image, normalize_prior(seam)), 1), (192, 192), mode="bilinear", align_corners=False)
        first = self.first(x)
        context = self.third(self.second(first))
        context = F.interpolate(context, first.shape[-2:], mode="bilinear", align_corners=False)
        logits = self.last(torch.cat((first, context), 1))
        return F.interpolate(logits, shape, mode="bilinear", align_corners=False)


class SeamInputAdapter(nn.Module):
    """Identity-initialized image residual before any segmentation backbone."""

    def __init__(self, limit: float = .10):
        super().__init__()
        self.limit = limit
        self.net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.SiLU(), nn.Conv2d(8, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor, seam: torch.Tensor, overlap: torch.Tensor) -> torch.Tensor:
        seam = normalize_prior(seam)
        change = self.limit * torch.tanh(self.net(torch.cat((image, seam, overlap), 1)))
        return image + change * seam * (1 - overlap)


class OutputCompletionPrior(DualInteractionRefiner):
    """Shared per-member refiner, including union-only tasks.

    The seventh input is a continuous interaction context: seam evidence
    attenuated by the frozen learned overlap probability. The inherited six
    channels retain the original RAM completion prior representation.
    """

    def load_mature(self, state: dict, identity: bool = False) -> dict:
        report = initialize_from_six_channel(self, state)
        if identity:
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)
        return {**report, "zero_residual": identity}

    def refine(self, image: torch.Tensor, logits: torch.Tensor, seam: torch.Tensor,
               overlap: torch.Tensor, steps: int = 1, chunk: int = 3) -> torch.Tensor:
        if image.shape[0] != 1:
            return torch.cat([self.refine(image[i:i+1], logits[i:i+1], seam[i:i+1], overlap[i:i+1], steps, chunk)
                              for i in range(image.shape[0])], 0)
        interaction = normalize_prior(seam) * (1 - overlap)
        edge = image_gradient(image)
        current = logits
        for _ in range(steps):
            probability = torch.sigmoid(current)
            pieces = []
            for start in range(0, current.shape[1], chunk):
                indices = list(range(start, min(start + chunk, current.shape[1])))
                own = probability[0, indices, None]
                others = []
                for k in indices:
                    other = torch.cat((probability[:, :k], probability[:, k+1:]), 1)
                    if other.shape[1]:
                        others.append(torch.stack((other.max(1).values[0], other.sum(1)[0].clamp(0, 2) / 2), 0))
                    else:
                        others.append(torch.zeros((2, *current.shape[-2:]), device=current.device, dtype=current.dtype))
                n = len(indices)
                feature = torch.cat((image.repeat(n, 1, 1, 1), own, torch.stack(others),
                                     4 * own * (1 - own), edge.repeat(n, 1, 1, 1),
                                     interaction.repeat(n, 1, 1, 1)), 1)
                source = current[0, indices, None]
                if self.training and torch.is_grad_enabled():
                    corrected, _ = checkpoint(self.forward, feature, source, use_reentrant=False)
                else:
                    corrected, _ = self(feature, source)
                pieces.append(corrected[:, 0][None])
            current = torch.cat(pieces, 1)
        return current


def local_terms(logits: torch.Tensor, target: torch.Tensor, teacher_probability: torch.Tensor,
                seam: torch.Tensor, overlap_probability: torch.Tensor,
                valid: torch.Tensor, membership_observed: bool) -> tuple[dict, dict]:
    """Same objective with explicit availability of membership observations.

    RAM multi-label target supplies positive overlap membership. A binary
    union cannot supply that target, so its overlap term is masked out;
    learned overlap knowledge still operates in the forward adapters.
    """
    probability = torch.sigmoid(logits.float())
    foreground = (target.sum(1, keepdim=True) > .5).float()
    corridor = (normalize_prior(seam) >= .25).float()
    separation = corridor * (1 - foreground) * valid
    if membership_observed:
        overlap = (target.sum(1, keepdim=True) >= 2).float() * valid
        positive = overlap * target
    else:
        overlap = torch.zeros_like(foreground)
        positive = torch.zeros_like(target)
    # In unobserved states preserve includes uncertain and potential overlap;
    # it is not evidence that these pixels contain no projection overlap.
    preserve = valid * (1 - torch.maximum(separation, overlap))
    bce = active_mean(F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none"), valid)
    dims = (0, 2, 3)
    p, y = probability * valid, target * valid
    dice = 1 - ((2 * (p * y).sum(dims) + 1) / (p.sum(dims) + y.sum(dims) + 1)).mean()
    # Additional foreground preservation acts on observed GT, not on an
    # invented TSRS-overlap mask. It protects recall across both datasets.
    foreground_recall = active_mean(F.softplus(-logits.float()), target * valid)
    terms = {
        "base": .5 * bce + .5 * dice + .05 * foreground_recall,
        "separation": separation_loss(probability.amax(1, keepdim=True), separation),
        "overlap": completion_loss(logits, positive),
        "preserve": active_mean((probability - teacher_probability.detach()).square(), preserve),
    }
    denominator = valid.sum().clamp_min(1)
    states = {"separation_mass": float(separation.sum().detach() / denominator),
              "overlap_observed_mass": float(overlap.sum().detach() / denominator),
              "overlap_labels_observed": membership_observed,
              "overlap_prior_mean": float((overlap_probability * valid).sum().detach() / denominator)}
    return terms, states
