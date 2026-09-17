#!/usr/bin/env python3
"""R338: frozen differentiable RAM prior that backpropagates into nnU-Net.

Unlike R337, the refiner forward pass is part of the autograd graph: refiner
parameters are frozen, but d(refined_logits)/d(base_logits) is retained.  The
R325 anchor and the independently trained R330 refiner remain immutable.  Only
the nnU-Net output head is trained initially; the final high-resolution decoder
stage is enabled after a warm-up.  Checkpoints are selected on RAM validation
official metrics and RAM test is never loaded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits, top_training_pairs
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset, sha256
from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner, seed_all
from train_r330_ram_pair_surface_volume_prior import choose_pair_instances, soft_surface_nsd_loss, volume_losses
from train_r332_ram_joint_iterative_refinement import collect_steps, flatten


def set_trainability(model: torch.nn.Module, decoder_enabled: bool) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        use = name.startswith("backbone.decoder.seg_layers.4.") or (decoder_enabled and (
            name.startswith("backbone.decoder.stages.4.") or
            name.startswith("backbone.decoder.transpconvs.4.")))
        parameter.requires_grad_(use)
        if use: names.append(name)
    return names


def selected_refine(image: torch.Tensor, logits: torch.Tensor,
                    prior: InstanceCompletionRefiner, indices: torch.Tensor,
                    alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable w.r.t. logits/image; prior weights remain frozen."""
    probability = torch.sigmoid(logits)
    source = probability[:, indices]
    # Imported implementation contains no detach/no_grad and therefore keeps
    # the source-logit and contextual-probability gradient paths.
    from train_r327_ram_native_instance_completion_prior import per_instance_features
    feature = per_instance_features(image, probability, indices, source)
    corrected, delta = prior(feature, logits[0, indices].unsqueeze(1))
    bounded = alpha * (corrected[:, 0] - logits[0, indices])
    scatter_index = indices.view(1, -1, 1, 1).expand(1, -1, logits.shape[-2], logits.shape[-1])
    full_delta = torch.zeros_like(logits).scatter(1, scatter_index, bounded.unsqueeze(0))
    return logits + full_delta, delta


def confidence_preserve(logits: torch.Tensor, anchor_logits: torch.Tensor,
                        target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        anchor_p = torch.sigmoid(anchor_logits)
        confident = ((anchor_p >= 0.9) | (anchor_p <= 0.1)).float()
        overlap_context = F.max_pool2d((target.sum(1, keepdim=True) >= 2).float(), 11, 1, 5)
        mask = confident * (1.0 - overlap_context)
    value = F.binary_cross_entropy_with_logits(logits.float(), torch.sigmoid(anchor_logits.float()), reduction="none")
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def checkpoint_gate(final: dict, direct: dict, r325: dict,
                    tolerance: float) -> tuple[bool, dict]:
    checks = {
        "final_overall_dsc_ge_r325": final["overall_dsc"] >= r325["overall_dsc"],
        "final_overall_iou_ge_r325": final["overall_iou"] >= r325["overall_iou"],
        "direct_dsc_within_tolerance": direct["overall_dsc"] >= r325["overall_dsc"] - tolerance,
        "direct_iou_within_tolerance": direct["overall_iou"] >= r325["overall_iou"] - tolerance,
        "overlap_ravd_within_0.003": final["overlap_ravd"] <= r325["overlap_ravd"] + 0.003,
        "pair_fail_rate_within_0.001": final["pair_msd_fail_rate"] <= r325["pair_msd_fail_rate"] + 0.001,
    }
    return all(checks.values()), checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--baseline-checkpoint", type=Path, required=True)
    ap.add_argument("--prior-checkpoint", type=Path, required=True)
    ap.add_argument("--r325-cache", type=Path, required=True)
    ap.add_argument("--r332-result", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup-epochs", type=int, default=4)
    ap.add_argument("--head-lr", type=float, default=2e-6)
    ap.add_argument("--decoder-lr", type=float, default=1e-6)
    ap.add_argument("--final-weight", type=float, default=0.20)
    ap.add_argument("--surface-weight", type=float, default=0.006)
    ap.add_argument("--instance-volume-weight", type=float, default=0.001)
    ap.add_argument("--pair-volume-weight", type=float, default=0.002)
    ap.add_argument("--preserve-weight", type=float, default=0.05)
    ap.add_argument("--alpha-max", type=float, default=0.5)
    ap.add_argument("--alpha-warmup", type=int, default=5)
    ap.add_argument("--direct-tolerance", type=float, default=0.0001)
    ap.add_argument("--instances-per-image", type=int, default=4)
    ap.add_argument("--official-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=3381)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("R338 requires CUDA")
    device = torch.device("cuda")

    train_set = NativeWristDataset(args.dataset_root, "train", augment=True)
    val_set = NativeWristDataset(args.dataset_root, "val", augment=False)
    if args.smoke:
        train_set.mask_files = train_set.mask_files[:2]; val_set.mask_files = val_set.mask_files[:2]
    train_loader = DataLoader(train_set, 1, shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=1, pin_memory=True)

    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    model = NnUNetMultiLabelPrior().to(device); model.load_state_dict(payload["model"], strict=True)
    anchor = NnUNetMultiLabelPrior().to(device); anchor.load_state_dict(payload["model"], strict=True)
    anchor.eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    prior_payload = torch.load(args.prior_checkpoint, map_location=device, weights_only=False)
    prior_state = prior_payload.get("refiner", prior_payload.get("module"))
    if prior_state is None:
        raise RuntimeError("Prior checkpoint contains neither 'refiner' nor 'module' state")
    prior = InstanceCompletionRefiner().to(device); prior.load_state_dict(prior_state, strict=True)
    prior.eval()
    for p in prior.parameters(): p.requires_grad_(False)

    # Optimizer contains the eventual head+last-stage tensors. requires_grad is
    # toggled by epoch, retaining a single reproducible scheduler.
    eventual = set_trainability(model, True)
    head_params, decoder_params = [], []
    for name, p in model.named_parameters():
        if name in eventual:
            (head_params if name.startswith("backbone.decoder.seg_layers.4.") else decoder_params).append(p)
    optimizer = torch.optim.AdamW([{"params": head_params, "lr": args.head_lr},
                                   {"params": decoder_params, "lr": args.decoder_lr}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    pairs = top_training_pairs(args.dataset_root, 15)

    r325_official = json.loads(args.r325_cache.read_text(encoding="utf-8"))
    r325_flat = flatten(r325_official)
    r332_reference = None
    if args.r332_result and args.r332_result.exists():
        old = json.loads(args.r332_result.read_text(encoding="utf-8"))
        # Validation-selected R332 can be represented by either result layout.
        r332_reference = old.get("flat_metrics_by_step", {}).get("step2")
    protocol = {
        "experiment": "R338_RAM_FROZEN_DIFFERENTIABLE_PRIOR", "test_used": False,
        "dataset": "RAM-W600 train/validation", "threshold": 0.5,
        "baseline": str(args.baseline_checkpoint), "baseline_sha256": sha256(args.baseline_checkpoint),
        "prior": str(args.prior_checkpoint), "prior_sha256": sha256(args.prior_checkpoint),
        "prior_parameters_frozen": True, "prior_input_gradient_retained": True,
        "anchor_parameters_frozen": True, "checkpoint_output": "model_plus_frozen_prior",
        "r325_reference": r325_flat, "r332_validation_reference": r332_reference,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    def batch_loss(image, target, alpha):
        logits, _ = model(image, False)
        with torch.no_grad(): anchor_logits, _ = anchor(image, False)
        indices = choose_pair_instances(torch.sigmoid(logits.detach()), target, args.instances_per_image)
        refined, delta = selected_refine(image, logits, prior, indices, alpha)
        base_loss = dice_bce_logits(logits, target)
        final_loss = dice_bce_logits(refined, target)
        selected_p = torch.sigmoid(refined[0, indices].unsqueeze(1).float())
        selected_t = target[0, indices].unsqueeze(1)
        surface = soft_surface_nsd_loss(selected_p, selected_t)
        instance_volume, pair_volume = volume_losses(selected_p, selected_t)
        preserve = confidence_preserve(logits, anchor_logits, target)
        total = (base_loss + args.final_weight * final_loss + args.surface_weight * surface +
                 args.instance_volume_weight * instance_volume + args.pair_volume_weight * pair_volume +
                 args.preserve_weight * preserve)
        return total, {"base": base_loss, "final": final_loss, "surface": surface,
                       "instance_volume": instance_volume, "pair_volume": pair_volume,
                       "preserve": preserve, "delta_abs": delta.abs().mean()}

    if args.smoke:
        set_trainability(model, False); model.train(); prior.eval(); anchor.eval()
        batch = next(iter(train_loader)); image = batch["image"].to(device); target = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, stats = batch_loss(image, target, min(args.alpha_max, args.alpha_max / args.alpha_warmup))
        loss.backward()
        model_grad = sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
        prior_grad = sum(float(p.grad.abs().sum()) for p in prior.parameters() if p.grad is not None)
        anchor_grad = sum(float(p.grad.abs().sum()) for p in anchor.parameters() if p.grad is not None)
        result = {"finite": bool(torch.isfinite(loss)), "loss": float(loss), "model_grad": model_grad,
                  "prior_grad": prior_grad, "anchor_grad": anchor_grad,
                  "stats": {k: float(v.detach()) for k, v in stats.items()}}
        if not result["finite"] or model_grad <= 0 or prior_grad != 0 or anchor_grad != 0: raise RuntimeError(result)
        print(json.dumps(result, indent=2)); return

    history = args.output / "history.jsonl"; best_path = args.output / "best.pth"; last_path = args.output / "last.pth"
    best = None
    for epoch in range(1, args.epochs + 1):
        decoder_enabled = epoch > args.warmup_epochs
        trainable = set_trainability(model, decoder_enabled)
        model.eval(); model.backbone.decoder.seg_layers[4].train()
        if decoder_enabled:
            model.backbone.decoder.stages[4].train(); model.backbone.decoder.transpconvs[4].train()
        prior.eval(); anchor.eval()
        alpha = args.alpha_max * min(epoch / max(args.alpha_warmup, 1), 1.0)
        totals = {k: 0.0 for k in ("loss", "base", "final", "surface", "instance_volume",
                                          "pair_volume", "preserve", "delta_abs")}
        started = time.time()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True); target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True): loss, stats = batch_loss(image, target, alpha)
            if not torch.isfinite(loss): raise RuntimeError({"epoch": epoch, "loss": float(loss)})
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer); scaler.update()
            totals["loss"] += float(loss.detach())
            for k, v in stats.items(): totals[k] += float(v.detach())
        scheduler.step()

        official_audit = epoch % args.official_every == 0 or epoch == args.epochs
        flat = None; passes = None; checks = None
        if official_audit:
            data = collect_steps(model, prior, val_loader, device, 1)
            official = {name: official_metrics(value, pairs) for name, value in data.items()}
            flat = {name: flatten(value) for name, value in official.items()}
            passes, checks = checkpoint_gate(flat["step1"], flat["step0"], r325_flat, args.direct_tolerance)
        row = {"epoch": epoch, "seconds": time.time() - started, "alpha": alpha,
               "decoder_enabled": decoder_enabled, "trainable_tensors": trainable,
               "head_lr": optimizer.param_groups[0]["lr"], "decoder_lr": optimizer.param_groups[1]["lr"],
               **{f"train_{k}": v / len(train_loader) for k, v in totals.items()},
               "official_audit": official_audit, "flat_metrics": flat,
               "passes_checkpoint_gate": passes, "gate_checks": checks}
        with history.open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if official_audit:
            final = flat["step1"]; direct = flat["step0"]
            beats_r332 = bool(r332_reference and
                              final["overall_dsc"] >= r332_reference["overall_dsc"] and
                              final["overall_iou"] >= r332_reference["overall_iou"] and
                              final["overlap_nsd_2px"] >= r332_reference["overlap_nsd_2px"] and
                              final["overlap_msd_px"] <= r332_reference["overlap_msd_px"])
            score = (int(passes), int(beats_r332), final["overlap_nsd_2px"], final["overlap_dsc"],
                     -final["overlap_msd_px"], final["overall_dsc"])
            row["beats_r332_core_gate"] = beats_r332; row["selection_score"] = list(score)
            if best is None or score > best[0]:
                best = (score, row)
                torch.save({"model": model.state_dict(), "epoch": epoch, "row": row,
                            "frozen_prior_sha256": protocol["prior_sha256"]}, best_path)
        torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict()}, last_path)

    result = {**protocol, "best_epoch": best[1]["epoch"], "best_row": best[1],
              "checkpoints": {"best": str(best_path), "last": str(last_path)}, "status": "complete"}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "best_epoch": result["best_epoch"],
                      "best": result["best_row"]}, indent=2), flush=True)


if __name__ == "__main__": main()
