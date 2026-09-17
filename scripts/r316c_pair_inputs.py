#!/usr/bin/env python3
"""Shared R316C dual-scale pair input construction."""
from __future__ import annotations

import random

import cv2
import numpy as np

from train_r256_scale_invariant_pair_prior import gaussian


def adaptive_sigma(scale_pixels: float, rel: float = 0.15,
                   minimum: float = 3.0, maximum: float = 12.0) -> float:
    return float(np.clip(rel * max(float(scale_pixels), 1.0), minimum, maximum))


def pair_channels(crop: np.ndarray, center_a: np.ndarray, center_b: np.ndarray,
                  proposal_scale_pixels: float, output_size: int,
                  global_size: int = 128, sigma_rel: float = 0.15,
                  sigma_min: float = 3.0, sigma_max: float = 12.0,
                  jitter_rel: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return high-detail, low-resolution context, and two center maps.

    Centers may be perturbed during training only. The returned centers are the
    perturbed input centers; supervision should continue to use the unperturbed
    geometry so the network learns proposal-error tolerance.
    """
    ca = np.asarray(center_a, dtype=np.float64).copy()
    cb = np.asarray(center_b, dtype=np.float64).copy()
    if jitter_rel > 0:
        jitter_std = jitter_rel * max(float(proposal_scale_pixels), 1.0)
        ca += np.asarray([random.gauss(0.0, jitter_std), random.gauss(0.0, jitter_std)])
        cb += np.asarray([random.gauss(0.0, jitter_std), random.gauss(0.0, jitter_std)])
    high = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    low_side = min(int(global_size), int(output_size))
    low = cv2.resize(crop, (low_side, low_side), interpolation=cv2.INTER_AREA)
    context = cv2.resize(low, (output_size, output_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    sigma = adaptive_sigma(proposal_scale_pixels, sigma_rel, sigma_min, sigma_max)
    channels = np.stack([
        high,
        context,
        gaussian(output_size, *ca, sigma=sigma),
        gaussian(output_size, *cb, sigma=sigma),
    ]).astype(np.float32)
    return channels, ca, cb, sigma
