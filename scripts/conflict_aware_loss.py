"""State-routed loss for the fixed seam and overlap priors.

The prior modules are deliberately outside this file.  This module only
decides which structural objective is allowed to act at each pixel and keeps
the two objectives mutually exclusive.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class ConflictAwareTerms:
    total: torch.Tensor
    base: torch.Tensor
    seam: torch.Tensor
    overlap: torch.Tensor
    preserve: torch.Tensor
    seam_gate_mass: torch.Tensor
    overlap_gate_mass: torch.Tensor
    uncertain_mass: torch.Tensor


def mutually_exclusive_states(target: torch.Tensor, seam_prior: torch.Tensor,
                               mode: str, seam_threshold: float = .25) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return seam, overlap and uncertain gates in [B,1,H,W].

    ``mode='tsrs'`` intentionally has no overlap state: TSRS colour IDs are
    not treated as projection-overlap labels.  In RAM, overlap is derived from
    the number of participating instance masks and takes precedence over the
    seam prior at intersecting pixels.
    """
    if mode not in {"tsrs", "ram"}:
        raise ValueError(f"unknown state mode: {mode}")
    seam = seam_prior.float()
    if seam.ndim == 3:
        seam = seam[:, None]
    lo, hi = seam.amin((-2, -1), keepdim=True), seam.amax((-2, -1), keepdim=True)
    seam = ((seam - lo) / (hi - lo + 1e-6)).clamp(0, 1)
    if not 0.0 < seam_threshold < 1.0:
        raise ValueError(f"seam_threshold must be in (0, 1), got {seam_threshold}")
    # Use an explicit corridor. Testing ``seam > 0`` makes almost every
    # min-max-normalized tail pixel an intervention state and removes teacher
    # protection from large background regions.
    corridor = (seam >= seam_threshold).float()
    foreground = (target.sum(1, keepdim=True) > 0.5).float()
    overlap = (target.sum(1, keepdim=True) >= 2).float() if mode == "ram" else torch.zeros_like(seam)
    seam_gate = corridor * (1 - foreground) * (1 - overlap)
    active = ((corridor > 0) | (overlap > 0)).float()
    uncertain = 1 - active
    return seam_gate, overlap, uncertain


def _foreground_probability(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "tsrs":
        return torch.softmax(logits.float(), dim=1)[:, 1:2]
    return torch.sigmoid(logits.float()).amax(1, keepdim=True)


def conflict_aware_loss(logits: torch.Tensor, target: torch.Tensor, teacher_logits: torch.Tensor,
                        seam_prior: torch.Tensor, base_loss: torch.Tensor, *, mode: str,
                        seam_weight: float = .02, overlap_weight: float = .04,
                        preserve_weight: float = 1.0, seam_ceiling: float = .10,
                        seam_threshold: float = .25) -> ConflictAwareTerms:
    """Compute routed structural terms without changing either prior module."""
    seam_gate, overlap_gate, uncertain = mutually_exclusive_states(
        target, seam_prior, mode, seam_threshold=seam_threshold)
    prob = _foreground_probability(logits, mode)
    seam = (seam_gate * F.relu(prob - seam_ceiling).square()).sum() / seam_gate.sum().clamp_min(1)
    if mode == "ram":
        positive = overlap_gate.expand_as(target) * target
        overlap = (positive * F.softplus(-logits.float())).sum() / positive.sum().clamp_min(1)
    else:
        overlap = logits.sum() * 0
    if mode == "tsrs":
        # Preserve the complete two-class distribution outside the explicit
        # corridor. Foreground-only MSE protects recall but leaves background
        # free to expand, which was the observed R346 failure mode.
        student_dist = torch.softmax(logits.float(), dim=1)
        teacher_dist = torch.softmax(teacher_logits.float(), dim=1)
        keep = uncertain.expand(-1, logits.shape[1], -1, -1)
        preserve = (keep * (student_dist - teacher_dist).square()).sum() / keep.sum().clamp_min(1)
    else:
        teacher_prob = _foreground_probability(teacher_logits, mode)
        preserve = (uncertain * (prob - teacher_prob).square()).sum() / uncertain.sum().clamp_min(1)
    total = base_loss + seam_weight * seam + overlap_weight * overlap + preserve_weight * preserve
    return ConflictAwareTerms(total, base_loss, seam, overlap, preserve,
                              seam_gate.mean(), overlap_gate.mean(), uncertain.mean())


def gradient_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine between two gradients, safely returning zero for absent grads."""
    aa, bb = a.reshape(-1), b.reshape(-1)
    return (aa @ bb) / (aa.norm() * bb.norm()).clamp_min(1e-12)
