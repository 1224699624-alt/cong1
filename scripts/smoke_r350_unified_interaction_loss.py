"""Deterministic CPU smoke for the shared R350 loss construction."""
from __future__ import annotations

import json

import torch

from r350_unified_interaction_loss import (
    completion_loss,
    distribution_preserve_loss,
    preserve_state,
    separation_loss,
    state_weights,
)


def run() -> dict[str, object]:
    torch.manual_seed(3500)
    logits = torch.randn(1, 1, 100, 100, requires_grad=True)
    probability = torch.sigmoid(logits)
    teacher = torch.sigmoid(torch.randn_like(logits))

    tsrs_sep = torch.zeros_like(logits); tsrs_sep[:, :, 2:9, 2:9] = 1
    tsrs_ov = torch.zeros_like(logits); tsrs_ov[:, :, 12:14, 12:14] = 1
    ram_sep = torch.zeros_like(logits); ram_sep[:, :, 2:4, 2:4] = 1
    ram_ov = torch.zeros_like(logits); ram_ov[:, :, 8:16, 8:16] = 1

    rows = {}
    for name, separation, overlap in (("tsrs", tsrs_sep, tsrs_ov), ("ram", ram_sep, ram_ov)):
        preserve = preserve_state(separation, overlap)
        weights = state_weights(separation, overlap, preserve, .035, .10)
        total = (
            weights.separation * separation_loss(probability, separation)
            + weights.overlap * completion_loss(logits, overlap)
            + weights.preserve * distribution_preserve_loss(probability, teacher, preserve)
        )
        total.backward(retain_graph=True)
        rows[name] = {
            "finite": bool(torch.isfinite(total)),
            "separation_mass": float(separation.mean()),
            "overlap_mass": float(overlap.mean()),
            "weights": vars(weights),
        }
    assert rows["tsrs"]["weights"]["separation"] > rows["tsrs"]["weights"]["overlap"]
    assert rows["ram"]["weights"]["overlap"] > rows["ram"]["weights"]["separation"]
    return rows


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
