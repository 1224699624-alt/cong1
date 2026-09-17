"""Small deterministic smoke test for the conflict-aware loss routing."""
from __future__ import annotations
import json
import torch
from conflict_aware_loss import conflict_aware_loss, mutually_exclusive_states


def run(mode: str) -> dict:
    torch.manual_seed(3460)
    if mode == "tsrs":
        target = torch.zeros(2, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1
        logits = torch.randn(2, 2, 16, 16, requires_grad=True)
        teacher = torch.randn(2, 2, 16, 16)
    else:
        target = torch.zeros(2, 3, 16, 16)
        target[:, 0, 4:12, 2:8] = 1
        target[:, 1, 4:12, 7:13] = 1  # overlap strip
        logits = torch.randn(2, 3, 16, 16, requires_grad=True)
        teacher = torch.randn(2, 3, 16, 16)
    seam = torch.zeros(2, 16, 16)
    seam[:, 1:4, 1:4] = 1
    seam_gate, overlap_gate, uncertain = mutually_exclusive_states(target, seam, mode)
    assert not torch.any((seam_gate > 0) & (overlap_gate > 0)), "gate conflict"
    base = logits.square().mean()
    terms = conflict_aware_loss(logits, target, teacher, seam, base, mode=mode)
    assert torch.isfinite(terms.total) and torch.isfinite(terms.seam)
    terms.total.backward()
    return {"mode": mode, "finite": True, "seam_mass": float(terms.seam_gate_mass),
            "overlap_mass": float(terms.overlap_gate_mass), "uncertain_mass": float(terms.uncertain_mass),
            "gate_sum": float((seam_gate + overlap_gate).max()), "grad_norm": float(logits.grad.norm())}


def check_tsrs_corridor_scope() -> dict:
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 3:5, 3:5] = 1
    seam = torch.full((1, 8, 8), .10)
    seam[:, 1:3, 1:3] = 1.0
    gate, overlap, outside = mutually_exclusive_states(
        target, seam, "tsrs", seam_threshold=.25)
    assert not torch.any(overlap)
    assert torch.all(gate[:, :, 1:3, 1:3] == 1)
    assert outside[0, 0, 0, 0] == 1, "weak prior tail must remain protected"
    assert outside[0, 0, 1, 1] == 0, "corridor must allow intervention"

    logits = torch.zeros(1, 2, 8, 8, requires_grad=True)
    teacher = torch.zeros_like(logits)
    teacher[:, 0, 0, 0] = 3.0
    terms = conflict_aware_loss(
        logits, target, teacher, seam, logits.sum() * 0, mode="tsrs",
        seam_weight=0.0, preserve_weight=1.0, seam_threshold=.25)
    terms.total.backward()
    outside_grad = logits.grad[0, :, 0, 0]
    corridor_grad = logits.grad[0, :, 1, 1]
    assert torch.all(outside_grad != 0), "both TSRS classes must be preserved outside corridor"
    assert torch.all(corridor_grad == 0), "preserve gradient must not block corridor intervention"
    return {"outside_mass": float(outside.mean()),
            "outside_grad": outside_grad.tolist(), "corridor_grad": corridor_grad.tolist()}


if __name__ == "__main__":
    print(json.dumps({"smoke": [run("tsrs"), run("ram")],
                      "tsrs_corridor": check_tsrs_corridor_scope()}, indent=2))
