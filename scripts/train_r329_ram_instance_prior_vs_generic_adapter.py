#!/usr/bin/env python3
"""R329: RAM instance-conditioned prior versus a parameter-matched adapter.

Both arms refine the same frozen R325 native-resolution nnU-Net, use only its
real training predictions, and optimize the same segmentation, conservation,
and boundary losses. The sole mechanism difference is representation:

* instance prior: one shared refiner is applied independently to each bone;
* generic adapter: all fourteen channels are refined jointly by an ordinary
  multi-channel residual network with automatically matched parameter count.

RAM test data are never loaded.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset
from train_r327_ram_native_instance_completion_prior import (
    Block,
    InstanceCompletionRefiner,
    MetricAccumulator,
    choose_instances,
    completion_loss,
    evaluate,
    image_gradient,
    per_instance_features,
    seed_all,
    sha256,
    train_arm,
)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


class GenericMultiChannelAdapter(nn.Module):
    """Ordinary whole-output residual adapter without per-instance sharing."""

    def __init__(self, width: int, residual_limit: float = 2.5) -> None:
        super().__init__()
        self.width = width
        self.residual_limit = residual_limit
        self.enc1 = Block(17, width)
        self.enc2 = Block(width, width * 2, 2)
        self.enc3 = Block(width * 2, width * 4, 2)
        self.mid = Block(width * 4, width * 4)
        self.dec2 = Block(width * 6, width * 2)
        self.dec1 = Block(width * 3, width)
        self.residual = nn.Conv2d(width, 14, 1)
        self.gate = nn.Conv2d(width, 14, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probability = torch.sigmoid(base_logits.detach())
        multiplicity = probability.sum(1, keepdim=True).clamp(0, 2) / 2
        feature = torch.cat((image, probability, multiplicity, image_gradient(image)), dim=1)
        e1 = self.enc1(feature)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        value = self.mid(e3)
        value = nn.functional.interpolate(value, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec2(torch.cat((value, e2), dim=1))
        value = nn.functional.interpolate(value, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(torch.cat((value, e1), dim=1))
        gate = torch.sigmoid(self.gate(value))
        delta = self.residual_limit * torch.tanh(self.residual(value)) * gate
        return base_logits.detach() + delta, delta


def closest_generic_width(target_parameters: int) -> tuple[int, int, dict[int, int]]:
    candidates = {width: parameter_count(GenericMultiChannelAdapter(width)) for width in range(4, 25)}
    width = min(candidates, key=lambda value: abs(candidates[value] - target_parameters))
    return width, candidates[width], candidates


@torch.inference_mode()
def evaluate_generic(baseline: nn.Module, adapter: nn.Module, loader: DataLoader,
                     device: torch.device) -> dict:
    baseline.eval()
    adapter.eval()
    metrics = MetricAccumulator(device)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True) > 0.5
        base_logits, _ = baseline(image, False)
        logits, _ = adapter(image, base_logits)
        metrics.add(torch.sigmoid(logits) >= 0.5, target)
    return metrics.result()


def train_generic(baseline: nn.Module, width: int, train_set: NativeWristDataset,
                  val_loader: DataLoader, baseline_metrics: dict, device: torch.device,
                  output: Path, epochs: int, learning_rate: float, min_epochs: int,
                  patience: int, seed: int, instances_per_image: int,
                  boundary_weight: float) -> tuple[Path, dict]:
    seed_all(seed)
    adapter = GenericMultiChannelAdapter(width).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2,
                        pin_memory=True, persistent_workers=True, generator=generator)
    checkpoint = output / "generic_adapter_best.pth"
    history = output / "generic_adapter_history.jsonl"
    best = None
    bad = 0
    for epoch in range(1, epochs + 1):
        adapter.train()
        totals = {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0, "conservation": 0.0,
                  "boundary_l1": 0.0, "local_fraction": 0.0}
        started = time.time()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.no_grad():
                base_logits, _ = baseline(image, False)
                base_probability = torch.sigmoid(base_logits)
                indices = choose_instances(base_probability, target, instances_per_image)
                selected_target = target[:, indices]
                error = torch.abs(base_probability[:, indices] - selected_target)
                local = nn.functional.max_pool2d((error > 0.15).float(), 11, 1, 5)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                logits, delta = adapter(image, base_logits)
                loss, stats = completion_loss(logits[:, indices], selected_target, local,
                                              delta[:, indices], boundary_weight)
            if not torch.isfinite(loss):
                raise RuntimeError({"arm": "generic_adapter", "epoch": epoch, "loss": float(loss)})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for key, value in stats.items(): totals[key] += value
        scheduler.step()
        metrics = evaluate_generic(baseline, adapter, val_loader, device)
        passes = metrics["macro_dsc"] >= baseline_metrics["macro_dsc"] and metrics["macro_iou"] >= baseline_metrics["macro_iou"]
        row = {"arm": "generic_adapter", "epoch": epoch, "seconds": time.time() - started,
               "learning_rate": optimizer.param_groups[0]["lr"], "boundary_weight": boundary_weight,
               "passes_macro_non_degradation": passes,
               **{f"train_{key}": value / len(loader) for key, value in totals.items()}, **metrics}
        with history.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = (int(passes), metrics["overlap_dsc"] if passes else metrics["macro_dsc"],
               metrics["overlap_nsd_2px"], -metrics["overlap_msd_px"])
        if best is None or key > best[0]:
            best = (key, row)
            bad = 0
            torch.save({"adapter": adapter.state_dict(), "row": row, "width": width}, checkpoint)
        else:
            bad += 1
        if epoch >= min_epochs and bad >= patience: break
    assert best is not None
    return checkpoint, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ram_w600/r329_instance_prior_vs_generic"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--instances-per-image", type=int, default=4)
    parser.add_argument("--boundary-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3291)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("R329 requires CUDA")
    device = torch.device("cuda")
    train_set = NativeWristDataset(args.dataset_root, "train", augment=True)
    val_set = NativeWristDataset(args.dataset_root, "val", augment=False)
    if args.smoke:
        train_set.mask_files = train_set.mask_files[:2]
        val_set.mask_files = val_set.mask_files[:2]
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline = NnUNetMultiLabelPrior().to(device)
    baseline.load_state_dict(payload["model"], strict=True)
    baseline.eval()
    for parameter in baseline.parameters(): parameter.requires_grad_(False)
    baseline_metrics = evaluate(baseline, None, val_loader, device)

    instance_parameters = parameter_count(InstanceCompletionRefiner())
    generic_width, generic_parameters, search = closest_generic_width(instance_parameters)
    matching = {"instance_parameters": instance_parameters, "generic_width": generic_width,
                "generic_parameters": generic_parameters,
                "relative_gap": (generic_parameters - instance_parameters) / instance_parameters,
                "width_search": search}
    (args.output / "parameter_matching.json").write_text(json.dumps(matching, indent=2), encoding="utf-8")

    if args.smoke:
        sample = train_set[0]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"].unsqueeze(0).to(device)
        with torch.no_grad(): base_logits, _ = baseline(image, False); probability = torch.sigmoid(base_logits)
        indices = choose_instances(probability, target, args.instances_per_image)
        generic = GenericMultiChannelAdapter(generic_width).to(device)
        generic_logits, generic_delta = generic(image, base_logits)
        local = nn.functional.max_pool2d((torch.abs(probability[:, indices] - target[:, indices]) > 0.15).float(), 11, 1, 5)
        generic_loss, _ = completion_loss(generic_logits[:, indices], target[:, indices], local,
                                          generic_delta[:, indices], args.boundary_weight)
        generic_loss.backward()
        instance = InstanceCompletionRefiner().to(device)
        source = probability[:, indices]
        feature = per_instance_features(image, probability, indices, source)
        instance_logits, instance_delta = instance(feature, base_logits[0, indices].unsqueeze(1))
        instance_loss, _ = completion_loss(instance_logits, target[0, indices].unsqueeze(1), local,
                                           instance_delta, args.boundary_weight)
        instance_loss.backward()
        print(json.dumps({"smoke": True, "matching": matching,
                          "generic_loss": float(generic_loss), "instance_loss": float(instance_loss)}))
        return

    seed_all(args.seed)
    initial_instance = copy.deepcopy(InstanceCompletionRefiner().state_dict())
    instance_ckpt, instance_best = train_arm(
        "instance_prior", False, baseline, initial_instance, train_set, val_loader,
        baseline_metrics, device, args.output, args.epochs, args.learning_rate,
        args.min_epochs, args.patience, args.seed, args.instances_per_image,
        "generic", 0.0, 0.0, args.boundary_weight)
    generic_ckpt, generic_best = train_generic(
        baseline, generic_width, train_set, val_loader, baseline_metrics, device,
        args.output, args.epochs, args.learning_rate, args.min_epochs, args.patience,
        args.seed, args.instances_per_image, args.boundary_weight)

    instance = InstanceCompletionRefiner().to(device)
    instance.load_state_dict(torch.load(instance_ckpt, map_location=device, weights_only=False)["refiner"], strict=True)
    instance_metrics = evaluate(baseline, instance, val_loader, device)
    generic = GenericMultiChannelAdapter(generic_width).to(device)
    generic.load_state_dict(torch.load(generic_ckpt, map_location=device, weights_only=False)["adapter"], strict=True)
    generic_metrics = evaluate_generic(baseline, generic, val_loader, device)
    keys = ("macro_dsc", "macro_iou", "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px")
    result = {"experiment": "R329_RAM_INSTANCE_PRIOR_VS_GENERIC_ADAPTER", "split": "validation",
              "test_used": False, "threshold": 0.5, "threshold_search": False,
              "spatial_preprocessing": "native pixels; right/bottom padding only",
              "baseline": baseline_metrics, "instance_prior": instance_metrics,
              "generic_adapter": generic_metrics, "parameter_matching": matching,
              "delta_instance_minus_baseline": {key: instance_metrics[key] - baseline_metrics[key] for key in keys},
              "delta_generic_minus_baseline": {key: generic_metrics[key] - baseline_metrics[key] for key in keys},
              "delta_instance_minus_generic": {key: instance_metrics[key] - generic_metrics[key] for key in keys},
              "best_rows": {"instance_prior": instance_best, "generic_adapter": generic_best},
              "initialization": {"baseline": str(args.baseline_checkpoint), "sha256": sha256(args.baseline_checkpoint),
                                 "baseline_frozen": True, "arm_seed": args.seed},
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
