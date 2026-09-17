#!/usr/bin/env python3
"""R349 RAM: loss-only, all-instance, parameter-space conflict routing."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint

from evaluate_r329_ram_official_metrics_val import official_metrics
from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits, top_training_pairs
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, per_instance_features, seed_all
from train_r332_ram_joint_iterative_refinement import flatten, iterative_refine, checkpoint_gate
from train_r335_ram_conditional_state_loss import RamSeamDataset


def checkpointed_simultaneous_refine(
    image: torch.Tensor,
    logits: torch.Tensor,
    refiner: InstanceCompletionRefiner,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Apply one all-instance correction while recomputing refiner activations."""
    probability = torch.sigmoid(logits)
    source = probability[:, indices]
    feature = per_instance_features(image, probability, indices, source)
    source_logits = logits[0, indices].unsqueeze(1)
    corrected, _ = checkpoint(refiner, feature, source_logits, use_reentrant=False)
    correction = corrected[:, 0] - logits[0, indices]
    full = torch.zeros_like(logits)
    scatter_index = indices.view(1, -1, 1, 1).expand(1, -1, logits.shape[-2], logits.shape[-1])
    return logits + full.scatter(1, scatter_index, correction.unsqueeze(0))


def checkpointed_iterative_refine(
    image: torch.Tensor,
    base_logits: torch.Tensor,
    refiner: InstanceCompletionRefiner,
    steps: int,
    chunk: int = 3,
) -> torch.Tensor:
    current = base_logits
    for _ in range(steps):
        pieces = []
        for start in range(0, 14, chunk):
            selected = torch.arange(start, min(start + chunk, 14), device=image.device)
            updated = checkpointed_simultaneous_refine(image, current, refiner, selected)
            pieces.append(updated - current)
        current = current + torch.stack(pieces).sum(0)
    return current


def state_masks(target: torch.Tensor, seam_prior: torch.Tensor, threshold: float,
                teacher_logits: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seam = seam_prior.float()
    lo = seam.amin((-2, -1), keepdim=True)
    hi = seam.amax((-2, -1), keepdim=True)
    corridor = ((seam - lo) / (hi - lo + 1e-6) >= threshold).float()
    if teacher_logits is None:
        overlap = torch.zeros_like(corridor)
    else:
        # The gate comes from the immutable R332 prediction and is therefore
        # available at inference. GT supplies the supervised membership target,
        # but never decides which instances or pixels enter the refiner.
        probability = torch.sigmoid(teacher_logits.float())
        second_probability = torch.topk(probability, k=2, dim=1).values[:, 1:2]
        overlap = (second_probability >= threshold).float()
    foreground = (target.sum(1, keepdim=True) > .5).float()
    separation = corridor * (1 - foreground) * (1 - overlap)
    active = ((corridor > 0) | (overlap > 0)).float()
    preserve = 1 - active
    if torch.any((separation > 0) & (overlap > 0)):
        raise RuntimeError("RAM seam and overlap gates conflict")
    return separation, overlap, preserve


def losses(student: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor,
           seam_prior: torch.Tensor, seam_threshold: float) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    separation, overlap, preserve = state_masks(target, seam_prior, seam_threshold, teacher)
    probability = torch.sigmoid(student.float())
    base = dice_bce_logits(student, target)
    union = probability.amax(1, keepdim=True)
    seam = (separation * F.relu(union - .10).square()).sum() / separation.sum().clamp_min(1)
    positive = overlap.expand_as(target) * target
    completion = (positive * F.softplus(-student.float())).sum() / positive.sum().clamp_min(1)
    teacher_probability = torch.sigmoid(teacher.float())
    keep = preserve.expand_as(target)
    distill = (keep * (probability - teacher_probability).square()).sum() / keep.sum().clamp_min(1)
    return {"base": base, "seam": seam, "overlap": completion, "preserve": distill}, {
        "seam_mass": float(separation.mean()), "overlap_mass": float(overlap.mean()),
        "preserve_mass": float(preserve.mean()), "gate_overlap": float(((separation > 0) & (overlap > 0)).float().mean())}


@torch.inference_mode()
def collect(network, refiner, loader, device, steps, pairs):
    network.eval(); refiner.eval(); cases = {}; seam_fp = []
    for batch in loader:
        image = batch["image"].to(device); target = batch["mask"].to(device) > .5
        base, _ = network(image, False)
        output = iterative_refine(image, base, refiner, steps)[0][-1]
        prediction = torch.sigmoid(output) >= .5
        separation, _, _ = state_masks(target.float(), batch["seam"].to(device), .25, output)
        seam_fp.append(float((prediction.any(1, keepdim=True) * separation).sum() / separation.sum().clamp_min(1)))
        cases[str(batch["case"][0])] = {"pred": prediction.cpu().numpy()[0], "target": target.cpu().numpy()[0]}
    return {"official": official_metrics(cases, pairs), "seam_fp": float(np.mean(seam_fp))}


@torch.inference_mode()
def identity_audit(network, student, teacher, loader, device, steps) -> dict:
    maximum = 0.0
    for index, batch in enumerate(loader):
        if index >= 2: break
        image = batch["image"].to(device); base, _ = network(image, False)
        left = iterative_refine(image, base, student, steps)[0][-1]
        right = iterative_refine(image, base, teacher, steps)[0][-1]
        maximum = max(maximum, float((left - right).abs().max()))
    return {"cases": min(2, len(loader)), "max_abs_logit_difference": maximum, "passed": maximum == 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--r332-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--seam-weight", type=float, default=.025)
    parser.add_argument("--overlap-weight", type=float, default=.04)
    parser.add_argument("--preserve-weight", type=float, default=.10)
    parser.add_argument("--max-aux-ratio", type=float, default=.15)
    parser.add_argument("--seam-threshold", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=3491)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed); device = torch.device("cuda")
    payload = torch.load(args.r332_checkpoint, map_location=device, weights_only=False)
    network = NnUNetMultiLabelPrior().to(device); network.load_state_dict(payload["model"], strict=True); network.requires_grad_(False)
    student = InstanceCompletionRefiner().to(device); student.load_state_dict(payload["refiner"], strict=True)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    train = RamSeamDataset(args.dataset_root, args.prior_root, "train", True)
    validation = RamSeamDataset(args.dataset_root, args.prior_root, "val", False)
    if args.audit:
        # Smoke the largest serialized cases: the previous first-two audit did
        # not cover the native-resolution peak that caused the full run OOM.
        train.mask_files = sorted(train.mask_files, key=lambda path: path.stat().st_size, reverse=True)[:2]
        validation.mask_files = sorted(validation.mask_files, key=lambda path: path.stat().st_size, reverse=True)[:2]
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0 if args.audit else 2, pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0 if args.audit else 1, pin_memory=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    pairs = top_training_pairs(args.dataset_root, 15)

    identity = identity_audit(network, student, teacher, validation_loader, device, args.steps)
    (args.output / "identity_audit.json").write_text(json.dumps(identity, indent=2))
    if not identity["passed"]: raise RuntimeError(identity)

    if args.audit:
        batch = next(iter(train_loader)); image = batch["image"].to(device); target = batch["mask"].to(device); seam = batch["seam"].to(device)
        with torch.no_grad():
            base_logits, _ = network(image, False)
            teacher_logits = iterative_refine(image, base_logits, teacher, args.steps)[0][-1]
        student_logits = checkpointed_iterative_refine(image, base_logits, student, args.steps)
        terms, states = losses(student_logits, teacher_logits, target, seam, args.seam_threshold)
        gradients, routed = route_parameter_gradients(student.parameters(), terms["base"], [
            ("seam", terms["seam"], args.seam_weight),
            ("overlap", terms["overlap"], args.overlap_weight),
            ("preserve", terms["preserve"], args.preserve_weight),
        ], max_aux_ratio=args.max_aux_ratio)
        assign_gradients(student.parameters(), gradients)
        result = {"passed": all(torch.isfinite(value) for value in terms.values()) and states["gate_overlap"] == 0.0,
                  "identity": identity, "all_instances_trained": student_logits.shape[1] == 14,
                  "losses": {key: float(value.detach()) for key, value in terms.items()}, "states": states,
                  "gradient_routing": [vars(row) for row in routed]}
        result["passed"] = bool(result["passed"] and result["all_instances_trained"] and
                                all(row.cosine_after >= -1e-6 and
                                    row.applied_norm <= args.max_aux_ratio * row.base_norm + 1e-6
                                    for row in routed))
        (args.output / "smoke_audit.json").write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
        if not result["passed"]: raise RuntimeError("R349 RAM audit failed")
        return

    baseline = collect(network, teacher, validation_loader, device, args.steps, pairs)
    base_flat = flatten(baseline["official"]); history = args.output / "history.jsonl"
    baseline_row = {"epoch": 0, "gate": False, "adapted": base_flat, "baseline": base_flat,
                    "seam_fp": {"baseline": baseline["seam_fp"], "adapted": baseline["seam_fp"]}}
    best = ((0, 0.0, 0.0), baseline_row)
    torch.save({"refiner": student.state_dict(), "epoch": 0, "row": baseline_row}, args.output / "best.pth")

    for epoch in range(1, args.epochs + 1):
        student.train(); totals = {key: 0.0 for key in ("base", "seam", "overlap", "preserve")}; routing = []; started = time.time()
        ramp = min(1.0, epoch / 5.0)
        for batch in train_loader:
            image = batch["image"].to(device); target = batch["mask"].to(device); seam = batch["seam"].to(device)
            with torch.no_grad():
                base_logits, _ = network(image, False)
                teacher_logits = iterative_refine(image, base_logits, teacher, args.steps)[0][-1]
            student_logits = checkpointed_iterative_refine(image, base_logits, student, args.steps)
            terms, _ = losses(student_logits, teacher_logits, target, seam, args.seam_threshold)
            optimizer.zero_grad(set_to_none=True)
            gradients, rows = route_parameter_gradients(student.parameters(), terms["base"], [
                ("seam", terms["seam"], args.seam_weight * ramp),
                ("overlap", terms["overlap"], args.overlap_weight * ramp),
                ("preserve", terms["preserve"], args.preserve_weight * ramp),
            ], max_aux_ratio=args.max_aux_ratio)
            assign_gradients(student.parameters(), gradients); torch.nn.utils.clip_grad_norm_(student.parameters(), 8); optimizer.step()
            for key, value in terms.items(): totals[key] += float(value.detach())
            routing.extend(vars(row) for row in rows)
        metrics = collect(network, student, validation_loader, device, args.steps, pairs); final = flatten(metrics["official"])
        gate, checks = checkpoint_gate(final, base_flat)
        checks.update({"overlap_dsc_not_below_r332": final["overlap_dsc"] >= base_flat["overlap_dsc"],
                       "overlap_nsd_not_below_r332": final["overlap_nsd_2px"] >= base_flat["overlap_nsd_2px"],
                       "overlap_msd_not_above_r332": final["overlap_msd_px"] <= base_flat["overlap_msd_px"],
                       "seam_fp_not_above_r332": metrics["seam_fp"] <= baseline["seam_fp"]})
        gate = bool(gate and all(checks.values()))
        row = {"epoch": epoch, "seconds": time.time() - started, "ramp": ramp, "gate": gate, "checks": checks,
               "losses": {key: value / len(train_loader) for key, value in totals.items()},
               "conflict_rate": {name: float(np.mean([x["conflicted"] for x in routing if x["name"] == name])) for name in ("seam", "overlap", "preserve")},
               "baseline": base_flat, "adapted": final,
               "seam_fp": {"baseline": baseline["seam_fp"], "adapted": metrics["seam_fp"]}}
        with history.open("a") as handle: handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        score = (int(gate), min(final["overall_dsc"] - base_flat["overall_dsc"], final["overlap_dsc"] - base_flat["overlap_dsc"]),
                 final["overlap_nsd_2px"] - base_flat["overlap_nsd_2px"])
        if score > best[0]: best = (score, row); torch.save({"refiner": student.state_dict(), "epoch": epoch, "row": row}, args.output / "best.pth")
    result = {"experiment": "R349_RAM_PARAMETER_PCGRAD", "split": "validation", "test_used": False,
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "best": best[1]}
    (args.output / "result.json").write_text(json.dumps(result, indent=2)); print(json.dumps({"complete": True, "best_epoch": best[1]["epoch"], "gate": best[1]["gate"]}))


if __name__ == "__main__":
    main()
