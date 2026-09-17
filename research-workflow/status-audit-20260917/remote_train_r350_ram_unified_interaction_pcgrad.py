#!/usr/bin/env python3
"""R350 RAM: the same mass-emphasized three-state PCGrad used by TSRS."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients
from r350_unified_interaction_loss import (
    completion_loss,
    distribution_preserve_loss,
    preserve_state,
    separation_loss,
    state_weights,
)
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, seed_all
from train_r332_ram_joint_iterative_refinement import flatten, iterative_refine, checkpoint_gate
from train_r335_ram_conditional_state_loss import RamSeamDataset
from train_r349_ram_parameter_pcgrad import checkpointed_iterative_refine


def state_masks(target: torch.Tensor, seam_prior: torch.Tensor, threshold: float,
                teacher_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seam = seam_prior.float()
    lo = seam.amin((-2, -1), keepdim=True); hi = seam.amax((-2, -1), keepdim=True)
    corridor = (((seam - lo) / (hi - lo + 1e-6)) >= threshold).float()
    probability = torch.sigmoid(teacher_logits.float())
    second_probability = torch.topk(probability, k=2, dim=1).values[:, 1:2]
    overlap = (second_probability >= threshold).float()
    foreground = (target.sum(1, keepdim=True) > .5).float()
    separation = corridor * (1 - foreground) * (1 - overlap)
    preserve = preserve_state(separation, overlap)
    return separation, overlap, preserve


def losses(student: torch.Tensor, teacher: torch.Tensor, target: torch.Tensor,
           seam_prior: torch.Tensor, seam_threshold: float, structural_weight: float,
           preserve_weight: float, reference_mass: float):
    separation, overlap, preserve = state_masks(target, seam_prior, seam_threshold, teacher)
    probability = torch.sigmoid(student.float())
    seam = separation_loss(probability.amax(1, keepdim=True), separation, .10)
    positive_overlap = overlap.expand_as(target) * target
    completion = completion_loss(student, positive_overlap)
    teacher_probability = torch.sigmoid(teacher.float())
    stable = distribution_preserve_loss(probability, teacher_probability, preserve)
    weights = state_weights(separation, overlap, preserve, structural_weight,
                            preserve_weight, reference_mass)
    states = {
        "separation_mass": float(separation.mean()), "overlap_mass": float(overlap.mean()),
        "preserve_mass": float(preserve.mean()),
        "gate_conflict": float(((separation > 0) & (overlap > 0)).float().mean()),
    }
    return {"base": None, "separation": seam, "overlap": completion, "preserve": stable}, weights, states


@torch.inference_mode()
def collect(network, refiner, loader, device, steps, pairs, seam_threshold):
    network.eval(); refiner.eval(); cases = {}; seam_fp = []
    for batch in loader:
        image = batch["image"].to(device); target = batch["mask"].to(device) > .5
        base, _ = network(image, False)
        output = iterative_refine(image, base, refiner, steps)[0][-1]
        prediction = torch.sigmoid(output) >= .5
        separation, _, _ = state_masks(target.float(), batch["seam"].to(device), seam_threshold, output)
        seam_fp.append(float((prediction.any(1, keepdim=True) * separation).sum()
                             / separation.sum().clamp_min(1)))
        cases[str(batch["case"][0])] = {"pred": prediction.cpu().numpy()[0],
                                        "target": target.cpu().numpy()[0]}
    return {"official": official_metrics(cases, pairs), "seam_fp": float(np.mean(seam_fp))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--r332-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--structural-weight", type=float, default=.035)
    parser.add_argument("--preserve-weight", type=float, default=.10)
    parser.add_argument("--max-aux-ratio", type=float, default=.15)
    parser.add_argument("--seam-threshold", type=float, default=.25)
    parser.add_argument("--reference-mass", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=3501)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed); device = torch.device("cuda")
    payload = torch.load(args.r332_checkpoint, map_location=device, weights_only=False)
    network = NnUNetMultiLabelPrior().to(device); network.load_state_dict(payload["model"], strict=True)
    network.requires_grad_(False)
    student = InstanceCompletionRefiner().to(device); student.load_state_dict(payload["refiner"], strict=True)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    train = RamSeamDataset(args.dataset_root, args.prior_root, "train", True)
    validation = RamSeamDataset(args.dataset_root, args.prior_root, "val", False)
    if args.audit:
        train.mask_files = sorted(train.mask_files, key=lambda path: path.stat().st_size, reverse=True)[:2]
        validation.mask_files = sorted(validation.mask_files, key=lambda path: path.stat().st_size, reverse=True)[:2]
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=0 if args.audit else 2,
                              pin_memory=True)
    validation_loader = DataLoader(validation, batch_size=1, shuffle=False,
                                   num_workers=0 if args.audit else 1, pin_memory=True)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    pairs = top_training_pairs(args.dataset_root, 15)

    if args.audit:
        batch = None; observed = []
        for candidate in train_loader:
            candidate_image = candidate["image"].to(device)
            candidate_target = candidate["mask"].to(device)
            candidate_seam = candidate["seam"].to(device)
            with torch.no_grad():
                candidate_base, _ = network(candidate_image, False)
                candidate_teacher = iterative_refine(candidate_image, candidate_base, teacher, args.steps)[0][-1]
            candidate_states = state_masks(candidate_target, candidate_seam, args.seam_threshold,
                                           candidate_teacher)
            masses = [float(value.mean()) for value in candidate_states[:2]]
            observed.append(masses)
            if all(value > 0 for value in masses):
                batch = candidate
                break
        if batch is None:
            raise RuntimeError({"reason": "no RAM audit batch covered both states", "observed": observed})
        image = batch["image"].to(device); target = batch["mask"].to(device)
        seam_prior = batch["seam"].to(device)
        with torch.no_grad():
            base_logits, _ = network(image, False)
            teacher_logits = iterative_refine(image, base_logits, teacher, args.steps)[0][-1]
        student_logits = checkpointed_iterative_refine(image, base_logits, student, args.steps)
        terms, weights, states = losses(student_logits, teacher_logits, target, seam_prior,
                                        args.seam_threshold, args.structural_weight,
                                        args.preserve_weight, args.reference_mass)
        from train_r284_ramw600_overlap_prior import dice_bce_logits
        terms["base"] = dice_bce_logits(student_logits, target)
        gradients, routed = route_parameter_gradients(student.parameters(), terms["base"], [
            ("separation", terms["separation"], weights.separation),
            ("overlap", terms["overlap"], weights.overlap),
            ("preserve", terms["preserve"], weights.preserve),
        ], max_aux_ratio=args.max_aux_ratio)
        assign_gradients(student.parameters(), gradients)
        result = {
            "passed": bool(all(torch.isfinite(value) for value in terms.values())
                           and states["gate_conflict"] == 0.0
                           and student_logits.shape[1] == 14
                           and all(row.cosine_after >= -1e-3
                                   and row.applied_norm <= args.max_aux_ratio * row.base_norm + 1e-6
                                   for row in routed)),
            "losses": {key: float(value.detach()) for key, value in terms.items()},
            "states": states, "effective_weights": vars(weights),
            "gradient_routing": [vars(row) for row in routed],
        }
        (args.output / "smoke_audit.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise RuntimeError("R350 RAM audit failed")
        return

    from train_r284_ramw600_overlap_prior import dice_bce_logits
    baseline = collect(network, teacher, validation_loader, device, args.steps, pairs, args.seam_threshold)
    base_flat = flatten(baseline["official"]); history = args.output / "history.jsonl"
    baseline_row = {"epoch": 0, "gate": False, "adapted": base_flat, "baseline": base_flat,
                    "seam_fp": {"baseline": baseline["seam_fp"], "adapted": baseline["seam_fp"]}}
    best = ((0, 0.0, 0.0), baseline_row)
    torch.save({"refiner": student.state_dict(), "epoch": 0, "row": baseline_row}, args.output / "best.pth")

    for epoch in range(1, args.epochs + 1):
        student.train(); totals = {key: 0.0 for key in ("base", "separation", "overlap", "preserve")}
        state_totals = {key: 0.0 for key in ("separation_mass", "overlap_mass", "preserve_mass")}
        weight_totals = {key: 0.0 for key in ("separation", "overlap", "preserve")}
        routing = []; started = time.time(); ramp = min(1.0, epoch / 5.0)
        for batch in train_loader:
            image = batch["image"].to(device); target = batch["mask"].to(device)
            seam_prior = batch["seam"].to(device)
            with torch.no_grad():
                base_logits, _ = network(image, False)
                teacher_logits = iterative_refine(image, base_logits, teacher, args.steps)[0][-1]
            student_logits = checkpointed_iterative_refine(image, base_logits, student, args.steps)
            terms, weights, states = losses(student_logits, teacher_logits, target, seam_prior,
                                            args.seam_threshold, args.structural_weight,
                                            args.preserve_weight, args.reference_mass)
            terms["base"] = dice_bce_logits(student_logits, target)
            optimizer.zero_grad(set_to_none=True)
            gradients, rows = route_parameter_gradients(student.parameters(), terms["base"], [
                ("separation", terms["separation"], weights.separation * ramp),
                ("overlap", terms["overlap"], weights.overlap * ramp),
                ("preserve", terms["preserve"], weights.preserve * ramp),
            ], max_aux_ratio=args.max_aux_ratio)
            assign_gradients(student.parameters(), gradients)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 8); optimizer.step()
            for key, value in terms.items(): totals[key] += float(value.detach())
            for key in state_totals: state_totals[key] += states[key]
            for key, value in vars(weights).items(): weight_totals[key] += value
            routing.extend(vars(row) for row in rows)

        metrics = collect(network, student, validation_loader, device, args.steps, pairs, args.seam_threshold)
        final = flatten(metrics["official"]); gate, checks = checkpoint_gate(final, base_flat)
        checks.update({
            "overlap_dsc_not_below_r332": final["overlap_dsc"] >= base_flat["overlap_dsc"],
            "overlap_nsd_not_below_r332": final["overlap_nsd_2px"] >= base_flat["overlap_nsd_2px"],
            "overlap_msd_not_above_r332": final["overlap_msd_px"] <= base_flat["overlap_msd_px"],
            "seam_fp_not_above_r332": metrics["seam_fp"] <= baseline["seam_fp"],
        }); gate = bool(gate and all(checks.values()))
        row = {
            "epoch": epoch, "seconds": time.time() - started, "ramp": ramp, "gate": gate,
            "checks": checks,
            "losses": {key: value / len(train_loader) for key, value in totals.items()},
            "state_mass": {key: value / len(train_loader) for key, value in state_totals.items()},
            "effective_weights": {key: value / len(train_loader) for key, value in weight_totals.items()},
            "conflict_rate": {name: float(np.mean([x["conflicted"] for x in routing if x["name"] == name]))
                              for name in ("separation", "overlap", "preserve")},
            "baseline": base_flat, "adapted": final,
            "seam_fp": {"baseline": baseline["seam_fp"], "adapted": metrics["seam_fp"]},
        }
        with history.open("a") as handle: handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        score = (int(gate), min(final["overall_dsc"] - base_flat["overall_dsc"],
                                final["overlap_dsc"] - base_flat["overlap_dsc"]),
                 final["overlap_nsd_2px"] - base_flat["overlap_nsd_2px"])
        if score > best[0]:
            best = (score, row)
            torch.save({"refiner": student.state_dict(), "epoch": epoch, "row": row}, args.output / "best.pth")
    result = {"experiment": "R350_RAM_UNIFIED_INTERACTION_PCGRAD", "split": "validation",
              "test_used": False,
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "best": best[1]}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"complete": True, "best_epoch": best[1]["epoch"], "gate": best[1]["gate"]}))


if __name__ == "__main__":
    main()
