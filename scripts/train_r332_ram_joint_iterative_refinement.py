#!/usr/bin/env python3
"""R332: joint last-decoder fine-tuning with iterative instance refinement.

The RAM test split is never opened. Native pixels are retained and only padded.
Each arm is trained for a fixed budget (no patience early stopping), while the
validation set is used to select checkpoints and the inference correction depth.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import (
    InstanceCompletionRefiner,
    MetricAccumulator,
    completion_loss,
    per_instance_features,
    seed_all,
)
from train_r330_ram_pair_surface_volume_prior import (
    choose_pair_instances,
    soft_surface_nsd_loss,
    volume_losses,
)


CORE_DIRECTIONS = {
    "overall_dsc": 1,
    "overall_iou": 1,
    "overlap_dsc": 1,
    "overlap_nsd_2px": 1,
    "overlap_msd_px": -1,
}


def set_joint_trainability(model: torch.nn.Module, enabled: bool) -> list[str]:
    """Unfreeze only the two highest-resolution decoder stages and output head."""
    trainable = []
    for name, parameter in model.named_parameters():
        use = enabled and (
            name.startswith("backbone.decoder.stages.3.")
            or name.startswith("backbone.decoder.stages.4.")
            or name.startswith("backbone.decoder.transpconvs.3.")
            or name.startswith("backbone.decoder.transpconvs.4.")
            or name.startswith("backbone.decoder.seg_layers.4.")
        )
        parameter.requires_grad_(use)
        if use:
            trainable.append(name)
    return trainable


def simultaneous_refine(
    image: torch.Tensor,
    logits: torch.Tensor,
    refiner: InstanceCompletionRefiner,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probability = torch.sigmoid(logits)
    source = probability[:, indices]
    feature = per_instance_features(image, probability, indices, source)
    corrected, delta = refiner(feature, logits[0, indices].unsqueeze(1))
    correction = corrected[:, 0] - logits[0, indices]
    full = torch.zeros_like(logits)
    scatter_index = indices.view(1, -1, 1, 1).expand(1, -1, logits.shape[-2], logits.shape[-1])
    full = full.scatter(1, scatter_index, correction.unsqueeze(0))
    return logits + full, delta


def iterative_refine(
    image: torch.Tensor,
    base_logits: torch.Tensor,
    refiner: InstanceCompletionRefiner,
    steps: int,
    indices: torch.Tensor | None = None,
    chunk: int = 3,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    states, deltas = [base_logits], []
    current = base_logits
    for _ in range(steps):
        if indices is not None:
            current, delta = simultaneous_refine(image, current, refiner, indices)
            deltas.append(delta)
        else:
            # All chunks see the same state; their corrections are applied together.
            pieces = []
            norms = []
            for start in range(0, 14, chunk):
                selected = torch.arange(start, min(start + chunk, 14), device=image.device)
                updated, delta = simultaneous_refine(image, current, refiner, selected)
                pieces.append(updated - current)
                norms.append(delta)
            current = current + torch.stack(pieces).sum(0)
            deltas.append(torch.cat(norms, 0))
        states.append(current)
    return states, deltas


@torch.inference_mode()
def collect_steps(model, refiner, loader, device, steps: int) -> dict[str, dict]:
    model.eval(); refiner.eval()
    output = {f"step{step}": {} for step in range(steps + 1)}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = (batch["mask"] > 0.5).numpy()[0]
        base_logits, _ = model(image, False)
        states, _ = iterative_refine(image, base_logits, refiner, steps)
        case = str(batch["case"][0])
        for step, logits in enumerate(states):
            prediction = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[0]
            output[f"step{step}"][case] = {"pred": prediction, "target": target}
    return output


@torch.inference_mode()
def fast_validate_steps(model, refiner, loader, device, steps: int) -> list[dict]:
    """Cheap per-epoch trend metrics; complete official metrics run at fixed audits."""
    model.eval(); refiner.eval()
    accumulators = [MetricAccumulator(device) for _ in range(steps + 1)]
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True) > 0.5
        base_logits, _ = model(image, False)
        states, _ = iterative_refine(image, base_logits, refiner, steps)
        for step, logits in enumerate(states):
            accumulators[step].add(torch.sigmoid(logits) >= 0.5, target)
    return [accumulator.result() for accumulator in accumulators]


def flatten(metrics: dict) -> dict[str, float]:
    overall = metrics["overall_instance_metrics"]
    overlap = metrics["overlap_region_metrics"]
    pair = metrics["overlap_pair_intersection_metrics"]
    return {
        "overall_dsc": overall["dsc"],
        "overall_iou": 1.0 - overall["voe"],
        "overall_nsd_2px": overall["nsd_2px"],
        "overall_msd_px": overall["msd_px"],
        "overall_ravd": overall["ravd"],
        "overlap_dsc": overlap["dsc"],
        "overlap_iou": 1.0 - overlap["voe"],
        "overlap_nsd_2px": overlap["nsd_2px"],
        "overlap_msd_px": overlap["msd_px"],
        "overlap_ravd": overlap["ravd"],
        "pair_dsc": pair["dsc"],
        "pair_iou": 1.0 - pair["voe"],
        "pair_nsd_2px": pair["nsd_2px"],
        "pair_msd_px": pair["msd_px"],
        "pair_ravd": pair["ravd"],
        "pair_msd_fail_rate": pair["msd_fail_rate"],
    }


def audit_progression(rows: list[dict[str, float]], tolerance: float = 1e-7) -> dict:
    transitions = []
    for index in range(1, len(rows)):
        checks = {}
        for key, direction in CORE_DIRECTIONS.items():
            delta = rows[index][key] - rows[index - 1][key]
            checks[key] = bool(direction * delta >= -tolerance)
        transitions.append({"from": index - 1, "to": index, "checks": checks,
                            "passed": sum(checks.values()), "total": len(checks)})
    return {
        "transitions": transitions,
        "all_core_metrics_monotonic": all(t["passed"] == t["total"] for t in transitions),
        "passed_checks": sum(t["passed"] for t in transitions),
        "total_checks": sum(t["total"] for t in transitions),
    }


def checkpoint_gate(row: dict[str, float], baseline: dict[str, float]) -> tuple[bool, dict]:
    checks = {
        "overall_dsc_not_below_r325": row["overall_dsc"] >= baseline["overall_dsc"],
        "overall_iou_not_below_r325": row["overall_iou"] >= baseline["overall_iou"],
        "overlap_ravd_within_0.003": row["overlap_ravd"] <= baseline["overlap_ravd"] + 0.003,
        "pair_fail_rate_within_0.001": row["pair_msd_fail_rate"] <= baseline["pair_msd_fail_rate"] + 0.001,
    }
    return all(checks.values()), checks


def save_payload(path, model, refiner, epoch, row, optimizer=None, scheduler=None, scaler=None):
    payload = {"model": model.state_dict(), "refiner": refiner.state_dict(),
               "epoch": epoch, "row": row}
    if optimizer is not None:
        payload.update({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict()})
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--refiner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--freeze-epochs", type=int, default=4)
    parser.add_argument("--refiner-lr", type=float, default=8e-5)
    parser.add_argument("--decoder-lr", type=float, default=7e-6)
    parser.add_argument("--seed", type=int, default=3321)
    parser.add_argument("--instances-per-image", type=int, default=4)
    parser.add_argument("--official-every", type=int, default=5)
    parser.add_argument("--baseline-metrics-cache", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("R332 requires CUDA")
    device = torch.device("cuda")

    train_set = NativeWristDataset(args.dataset_root, "train", augment=True)
    val_set = NativeWristDataset(args.dataset_root, "val", augment=False)
    if args.smoke:
        train_set.mask_files = train_set.mask_files[:2]
        val_set.mask_files = val_set.mask_files[:2]
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2,
                              pin_memory=True, persistent_workers=True,
                              generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    model = NnUNetMultiLabelPrior().to(device)
    model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device,
                                     weights_only=False)["model"], strict=True)
    refiner = InstanceCompletionRefiner().to(device)
    refiner.load_state_dict(torch.load(args.refiner_checkpoint, map_location=device,
                                       weights_only=False)["refiner"], strict=True)
    trainable_names = set_joint_trainability(model, True)
    decoder_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": refiner.parameters(), "lr": args.refiner_lr},
        {"params": decoder_parameters, "lr": args.decoder_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    pairs = top_training_pairs(args.dataset_root, 15)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    audit = {
        "experiment": "R332_RAM_JOINT_ITERATIVE_REFINEMENT",
        "split": "train/validation only", "test_used": False,
        "threshold": 0.5, "threshold_search": False,
        "spatial_preprocessing": "native pixels; right/bottom padding only; no resize",
        "fixed_budget_no_patience_early_stop": True,
        "trainable_decoder_parameters": trainable_names,
        "frozen_parameter_count": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "trainable_decoder_parameter_count": sum(p.numel() for p in decoder_parameters),
        "refiner_parameter_count": sum(p.numel() for p in refiner.parameters()),
        "config": config,
    }
    (args.output / "protocol.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    def loss_for_batch(image, target):
        base_logits, _ = model(image, False)
        base_probability = torch.sigmoid(base_logits)
        indices = choose_pair_instances(base_probability.detach(), target, args.instances_per_image)
        selected_target = target[0, indices].unsqueeze(1)
        initial_error = torch.abs(base_probability[0, indices].unsqueeze(1).detach() - selected_target)
        local = F.max_pool2d((initial_error > 0.15).float(), 11, 1, 5)
        states, deltas = iterative_refine(image, base_logits, refiner, args.steps, indices=indices)
        step_losses, stats = [], []
        for step in range(1, args.steps + 1):
            selected_logits = states[step][0, indices].unsqueeze(1)
            base_loss, values = completion_loss(selected_logits, selected_target, local,
                                                deltas[step - 1], 0.05)
            probability = torch.sigmoid(selected_logits.float())
            surface = soft_surface_nsd_loss(probability, selected_target)
            instance_volume, pair_volume = volume_losses(probability, selected_target)
            value = base_loss + 0.03 * surface + 0.005 * instance_volume + 0.01 * pair_volume
            step_losses.append(value)
            stats.append({**values, "surface": float(surface.detach()),
                          "instance_volume": float(instance_volume.detach()),
                          "pair_volume": float(pair_volume.detach()),
                          "delta_abs": float(deltas[step - 1].detach().abs().mean())})
        weights = torch.tensor([step / sum(range(1, args.steps + 1))
                                for step in range(1, args.steps + 1)], device=device)
        deep = sum(weights[index] * value for index, value in enumerate(step_losses))
        zero = base_logits.sum() * 0.0
        monotonic = sum((F.relu(step_losses[index] - step_losses[index - 1].detach())
                         for index in range(1, len(step_losses))), start=zero)
        contraction = sum((F.relu(deltas[index].abs().mean()
                                  - deltas[index - 1].detach().abs().mean())
                           for index in range(1, len(deltas))), start=zero)
        global_loss = dice_bce_logits(base_logits, target)
        total = deep + 0.05 * monotonic + 0.01 * contraction + 0.10 * global_loss
        return total, global_loss, monotonic, contraction, stats

    if args.smoke:
        model.train(); refiner.train(); set_joint_trainability(model, True)
        batch = next(iter(train_loader)); image = batch["image"].to(device); target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            loss, global_loss, monotonic, contraction, stats = loss_for_batch(image, target)
        loss.backward()
        decoder_grad = sum(float(p.grad.abs().sum()) for p in decoder_parameters if p.grad is not None)
        refiner_grad = sum(float(p.grad.abs().sum()) for p in refiner.parameters() if p.grad is not None)
        frozen_grad = sum(float(p.grad.abs().sum()) for p in model.parameters()
                          if not p.requires_grad and p.grad is not None)
        result = {"smoke": True, "finite": bool(torch.isfinite(loss)), "loss": float(loss),
                  "decoder_grad_sum": decoder_grad, "refiner_grad_sum": refiner_grad,
                  "frozen_grad_sum": frozen_grad, "step_stats": stats}
        if not result["finite"] or decoder_grad <= 0 or refiner_grad <= 0 or frozen_grad != 0:
            raise RuntimeError(result)
        print(json.dumps(result, indent=2), flush=True)
        return

    # Capture/cache the immutable R325 reference before any resume or joint update.
    if args.baseline_metrics_cache is not None and args.baseline_metrics_cache.exists():
        r325_official = json.loads(args.baseline_metrics_cache.read_text(encoding="utf-8"))
    else:
        r325_data = collect_steps(model, refiner, val_loader, device, 0)["step0"]
        r325_official = official_metrics(r325_data, pairs)
        if args.baseline_metrics_cache is not None:
            args.baseline_metrics_cache.parent.mkdir(parents=True, exist_ok=True)
            args.baseline_metrics_cache.write_text(json.dumps(r325_official, indent=2), encoding="utf-8")
    r325_flat = flatten(r325_official)

    history_path = args.output / "history.jsonl"
    last_path = args.output / "last.pth"
    best_path = args.output / "best.pth"
    start_epoch, best = 1, None
    if last_path.exists():
        resume = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(resume["model"], strict=True); refiner.load_state_dict(resume["refiner"], strict=True)
        optimizer.load_state_dict(resume["optimizer"]); scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"]); start_epoch = int(resume["epoch"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)["row"]

    for epoch in range(start_epoch, args.epochs + 1):
        joint_enabled = epoch > args.freeze_epochs
        set_joint_trainability(model, joint_enabled)
        model.eval()
        if joint_enabled:
            model.backbone.decoder.stages[3].train(); model.backbone.decoder.stages[4].train()
            model.backbone.decoder.transpconvs[3].train(); model.backbone.decoder.transpconvs[4].train()
            model.backbone.decoder.seg_layers[4].train()
        refiner.train()
        totals = {"loss": 0.0, "global": 0.0, "monotonic": 0.0, "contraction": 0.0}
        started = time.time()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                loss, global_loss, monotonic, contraction, _ = loss_for_batch(image, target)
            if not torch.isfinite(loss):
                raise RuntimeError({"epoch": epoch, "loss": float(loss)})
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 12.0)
            if joint_enabled:
                torch.nn.utils.clip_grad_norm_(decoder_parameters, 5.0)
            scaler.step(optimizer); scaler.update()
            totals["loss"] += float(loss.detach()); totals["global"] += float(global_loss.detach())
            totals["monotonic"] += float(monotonic.detach()); totals["contraction"] += float(contraction.detach())
        scheduler.step()

        fast_steps = fast_validate_steps(model, refiner, val_loader, device, args.steps)
        official_audit = epoch % args.official_every == 0 or epoch == args.epochs
        if official_audit:
            collected = collect_steps(model, refiner, val_loader, device, args.steps)
            official = {name: official_metrics(data, pairs) for name, data in collected.items()}
            flat_steps = [flatten(official[f"step{step}"]) for step in range(args.steps + 1)]
            progression = audit_progression(flat_steps)
            final_flat = flat_steps[-1]
            passes, gate_checks = checkpoint_gate(final_flat, r325_flat)
        else:
            official = None; flat_steps = None; progression = None
            final_flat = None; passes = None; gate_checks = None
        row = {
            "epoch": epoch, "seconds": time.time() - started, "steps": args.steps,
            "joint_decoder_enabled": joint_enabled,
            "official_audit": official_audit,
            "refiner_lr": optimizer.param_groups[0]["lr"],
            "decoder_lr": optimizer.param_groups[1]["lr"],
            **{f"train_{key}": value / len(train_loader) for key, value in totals.items()},
            "r325_baseline": r325_flat, "fast_step_metrics": fast_steps,
            "official_step_metrics": flat_steps,
            "progression": progression, "passes_checkpoint_gate": passes,
            "checkpoint_gate_checks": gate_checks,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if official_audit:
            score = (int(passes), progression["passed_checks"], final_flat["overlap_nsd_2px"],
                     final_flat["overlap_dsc"], -final_flat["overlap_msd_px"], -final_flat["overlap_ravd"])
            best_score = tuple(best["selection_score"]) if best is not None else None
            row["selection_score"] = list(score)
            if best_score is None or score > best_score:
                best = row
                save_payload(best_path, model, refiner, epoch, row)
        save_payload(last_path, model, refiner, epoch, row, optimizer, scheduler, scaler)
        if epoch in (10, 20, 30):
            save_payload(args.output / f"epoch_{epoch:03d}.pth", model, refiner, epoch, row)

    best_payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    refiner.load_state_dict(best_payload["refiner"], strict=True)
    final_data = collect_steps(model, refiner, val_loader, device, args.steps)
    final_official = {name: official_metrics(data, pairs) for name, data in final_data.items()}
    result = {
        **audit, "best_epoch": best_payload["epoch"], "best_row": best_payload["row"],
        "r325_baseline_official": r325_official, "r325_baseline_flat": r325_flat,
        "official_metrics_by_step": final_official,
        "flat_metrics_by_step": {name: flatten(value) for name, value in final_official.items()},
        "progression": audit_progression([flatten(final_official[f"step{s}"])
                                           for s in range(args.steps + 1)]),
        "checkpoints": {"best": str(best_path), "last": str(last_path)},
        "status": "complete",
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "best_epoch": result["best_epoch"],
                      "flat_metrics_by_step": result["flat_metrics_by_step"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
