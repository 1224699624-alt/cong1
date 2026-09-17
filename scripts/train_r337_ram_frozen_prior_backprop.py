#!/usr/bin/env python3
"""R337-A: frozen RAM instance prior as a training constraint, not post-processing.

The mature shared per-instance refiner is frozen.  It proposes bounded local
logit corrections from a detached copy of the segmentation prediction.  GT is
used during training only to retain corrections that reduce local probability
error.  The selected, stop-gradient correction becomes a conservative teacher
target.  Gradients update only the last two high-resolution decoder stages and
the segmentation head.  Validation reports both model-only and assisted
outputs; checkpoint selection is based on model-only output, so a positive
result cannot be attributed to inference-time post-processing.

RAM test is never enumerated or loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset, sha256
from train_r327_ram_native_instance_completion_prior import (
    InstanceCompletionRefiner, MetricAccumulator, per_instance_features, seed_all,
)
from train_r332_ram_joint_iterative_refinement import set_joint_trainability


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    axes = (0, 2, 3)
    dice = 1.0 - ((2.0 * (probability * target).sum(axes) + 1.0) /
                  (probability.sum(axes) + target.sum(axes) + 1.0)).mean()
    return 0.55 * bce + 0.45 * dice


@torch.no_grad()
def prior_correct(image: torch.Tensor, logits: torch.Tensor,
                  prior: InstanceCompletionRefiner, chunk: int = 3) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    corrected = logits.clone()
    for start in range(0, 14, chunk):
        indices = torch.arange(start, min(start + chunk, 14), device=logits.device)
        source = probability[:, indices]
        feature = per_instance_features(image, probability, indices, source)
        value, _ = prior(feature, logits[0, indices].unsqueeze(1))
        corrected[0, indices] = value[:, 0]
    return corrected


def beneficial_teacher(image: torch.Tensor, logits: torch.Tensor, target: torch.Tensor,
                       prior: InstanceCompletionRefiner, margin: float,
                       max_delta: float) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with torch.no_grad():
        source = logits.detach()
        proposed = prior_correct(image, source, prior)
        proposed = source + (proposed - source).clamp(-max_delta, max_delta)
        p0, p1 = torch.sigmoid(source), torch.sigmoid(proposed)
        error0, error1 = (p0 - target).abs(), (p1 - target).abs()
        # RAM's reliable overlap definition is multi-membership in independent
        # GT channels.  Dilate only for local context; never supervise test data.
        overlap = (target.sum(1, keepdim=True) >= 2).float()
        interaction = F.max_pool2d(overlap, 11, 1, 5)
        beneficial = ((error1 + margin) < error0).float() * interaction
        # Protect high-confidence correct cores/background from needless edits.
        reliable = (((p0 >= 0.9) & (target > 0.5)) |
                    ((p0 <= 0.1) & (target <= 0.5))).float()
        gate = beneficial * (1.0 - reliable)
        teacher = source + gate * (proposed - source)
        stats = {
            "gate_fraction": float(gate.mean()),
            "proposal_abs": float((proposed - source).abs().mean()),
            "accepted_abs": float((teacher - source).abs().mean()),
            "overlap_context_fraction": float(interaction.mean()),
        }
    return teacher, gate, stats


@torch.inference_mode()
def validate(model, prior, loader, device) -> tuple[dict, dict]:
    model.eval(); prior.eval()
    direct, assisted = MetricAccumulator(device), MetricAccumulator(device)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True) > 0.5
        logits, _ = model(image, False)
        corrected = prior_correct(image, logits, prior)
        direct.add(torch.sigmoid(logits) >= 0.5, target)
        assisted.add(torch.sigmoid(corrected) >= 0.5, target)
    return direct.result(), assisted.result()


def score(metrics: dict, baseline: dict) -> tuple:
    passes = (metrics["macro_dsc"] >= baseline["macro_dsc"] and
              metrics["macro_iou"] >= baseline["macro_iou"])
    return (int(passes), metrics["overlap_nsd_2px"], metrics["overlap_dsc"],
            -metrics["overlap_msd_px"], metrics["macro_dsc"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--baseline-checkpoint", type=Path, required=True)
    ap.add_argument("--prior-checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs/ram_w600/r337_frozen_prior_backprop"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--learning-rate", type=float, default=7e-6)
    ap.add_argument("--prior-weight", type=float, default=0.08)
    ap.add_argument("--benefit-margin", type=float, default=0.002)
    ap.add_argument("--max-delta", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=3371)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("R337 requires CUDA")
    device = torch.device("cuda")

    train_set = NativeWristDataset(args.dataset_root, "train", augment=True)
    val_set = NativeWristDataset(args.dataset_root, "val", augment=False)
    if args.smoke:
        train_set.mask_files = train_set.mask_files[:2]
        val_set.mask_files = val_set.mask_files[:2]
    train_loader = DataLoader(train_set, 1, shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=1, pin_memory=True)

    base_payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    model = NnUNetMultiLabelPrior().to(device)
    model.load_state_dict(base_payload["model"], strict=True)
    trainable_names = set_joint_trainability(model, True)
    prior_payload = torch.load(args.prior_checkpoint, map_location=device, weights_only=False)
    prior = InstanceCompletionRefiner().to(device)
    prior.load_state_dict(prior_payload["refiner"], strict=True)
    prior.eval()
    for parameter in prior.parameters(): parameter.requires_grad_(False)

    baseline, baseline_assisted = validate(model, prior, val_loader, device)
    protocol = {
        "experiment": "R337A_RAM_FROZEN_PRIOR_BACKPROP",
        "dataset": "RAM-W600 train/validation only", "test_used": False,
        "spatial": "native pixels; right/bottom padding only", "threshold": 0.5,
        "baseline_checkpoint": str(args.baseline_checkpoint), "baseline_sha256": sha256(args.baseline_checkpoint),
        "prior_checkpoint": str(args.prior_checkpoint), "prior_sha256": sha256(args.prior_checkpoint),
        "prior_frozen": True, "checkpoint_selection_output": "model_only",
        "trainable_model_tensors": sorted(trainable_names),
        "baseline_model_only": baseline, "baseline_assisted_diagnostic": baseline_assisted,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    history_path, best_path = args.output / "history.jsonl", args.output / "best.pth"
    best = None
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.train(); started = time.time()
        totals = {"loss": 0.0, "seg": 0.0, "prior": 0.0, "gate_fraction": 0.0,
                  "accepted_abs": 0.0}
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits, _ = model(image, False)
                teacher, gate, stats = beneficial_teacher(image, logits, target, prior,
                                                           args.benefit_margin, args.max_delta)
                seg_loss = dice_bce(logits, target)
                probability, teacher_probability = torch.sigmoid(logits.float()), torch.sigmoid(teacher.float())
                prior_loss = ((probability - teacher_probability).abs() * gate).sum() / gate.sum().clamp_min(1.0)
                loss = seg_loss + args.prior_weight * prior_loss
            if not torch.isfinite(loss): raise RuntimeError({"epoch": epoch, "loss": float(loss)})
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 12.0)
            scaler.step(optimizer); scaler.update()
            totals["loss"] += float(loss.detach()); totals["seg"] += float(seg_loss.detach())
            totals["prior"] += float(prior_loss.detach()); totals["gate_fraction"] += stats["gate_fraction"]
            totals["accepted_abs"] += stats["accepted_abs"]
        scheduler.step()
        direct, assisted = validate(model, prior, val_loader, device)
        row = {"epoch": epoch, "seconds": time.time() - started,
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v / len(train_loader) for k, v in totals.items()},
               "model_only": direct, "assisted_diagnostic": assisted,
               "passes_model_only_macro_gate": direct["macro_dsc"] >= baseline["macro_dsc"] and
                                                 direct["macro_iou"] >= baseline["macro_iou"]}
        with history_path.open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = score(direct, baseline)
        if best is None or key > best[0]:
            best = (key, row)
            torch.save({"model": model.state_dict(), "row": row,
                        "prior_frozen": True, "prior_sha256": protocol["prior_sha256"]}, best_path)
    result = {**protocol, "best_row": best[1], "checkpoint": str(best_path),
              "decision": "validation_only_pending_gate_audit"}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "best": best[1]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
