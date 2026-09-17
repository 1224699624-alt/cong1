#!/usr/bin/env python3
"""R327: native-resolution RAM instance-completion prior.

A mature native-resolution R325 nnU-Net is frozen. A shared per-instance
refiner learns bounded logit corrections. The capacity control sees only the
frozen model predictions; the completion-prior arm additionally trains on
algorithmically corrupted instance probabilities (overlap dropout, boundary
dropout, and local false membership) and clean identity samples.

Only RAM train/validation are loaded. The RAM test directory is never opened.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        groups = 4 if cout % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(groups, cout), nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class InstanceCompletionRefiner(nn.Module):
    """Shared image-conditioned refiner used identically for all 14 bones."""

    def __init__(self, width: int = 12, residual_limit: float = 2.5) -> None:
        super().__init__()
        self.residual_limit = residual_limit
        self.enc1 = Block(6, width)
        self.enc2 = Block(width, width * 2, 2)
        self.enc3 = Block(width * 2, width * 4, 2)
        self.mid = Block(width * 4, width * 4)
        self.dec2 = Block(width * 6, width * 2)
        self.dec1 = Block(width * 3, width)
        self.residual = nn.Conv2d(width, 1, 1)
        self.gate = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, feature: torch.Tensor, source_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(feature)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        value = self.mid(e3)
        value = F.interpolate(value, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec2(torch.cat((value, e2), dim=1))
        value = F.interpolate(value, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        value = self.dec1(torch.cat((value, e1), dim=1))
        gate = torch.sigmoid(self.gate(value))
        delta = self.residual_limit * torch.tanh(self.residual(value)) * gate
        return source_logit + delta, delta


def image_gradient(image: torch.Tensor) -> torch.Tensor:
    gx = F.pad(torch.abs(image[..., 1:] - image[..., :-1]), (0, 1, 0, 0))
    gy = F.pad(torch.abs(image[..., 1:, :] - image[..., :-1, :]), (0, 0, 0, 1))
    value = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return value / value.amax((-2, -1), keepdim=True).clamp_min(1e-6)


def per_instance_features(image: torch.Tensor, probabilities: torch.Tensor,
                          indices: torch.Tensor, source_probability: torch.Tensor) -> torch.Tensor:
    # batch size is one; indices select semantic channels and become a shared
    # per-instance batch without injecting a bone-name embedding.
    # source_probability is already ordered like ``indices`` and has shape
    # [1, number_selected, H, W].
    selected = source_probability[0].unsqueeze(1)
    contexts = []
    for index in indices.tolist():
        other = torch.cat((probabilities[:, :index], probabilities[:, index + 1:]), dim=1)
        contexts.append(torch.stack((other.max(1).values[0], other.sum(1)[0].clamp(0, 2) / 2), dim=0))
    context = torch.stack(contexts, dim=0)
    uncertainty = 4.0 * selected * (1.0 - selected)
    count = len(indices)
    return torch.cat((image.repeat(count, 1, 1, 1), selected, context,
                      uncertainty, image_gradient(image).repeat(count, 1, 1, 1)), dim=1)


def smooth_field(count: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    small_h, small_w = max(4, height // 64), max(4, width // 64)
    field = torch.rand((count, 1, small_h, small_w), device=device)
    return F.interpolate(field, size=(height, width), mode="bicubic", align_corners=False)


def corrupt_probabilities(base_probability: torch.Tensor, target: torch.Tensor,
                          indices: torch.Tensor, profile: str = "generic",
                          synthetic_fraction: float = 0.30,
                          identity_fraction: float = 0.20) -> tuple[torch.Tensor, torch.Tensor]:
    """Prediction-space augmentation that mimics missing/merged instances."""
    selected_target = target[0, indices].unsqueeze(1).float()
    selected_base = base_probability[0, indices].unsqueeze(1).detach()
    count, _, height, width = selected_target.shape
    membership = target.sum(1, keepdim=True)
    overlap = (membership >= 2).float().repeat(count, 1, 1, 1)
    other_union = []
    for index in indices.tolist():
        other = torch.cat((target[:, :index], target[:, index + 1:]), dim=1)
        other_union.append(other.max(1).values[0])
    other_union = torch.stack(other_union).unsqueeze(1).float()

    eroded = 1.0 - F.max_pool2d(1.0 - selected_target, 5, stride=1, padding=2)
    boundary = (selected_target - eroded).clamp(0, 1)
    if profile == "fn_focus":
        # RAM's dominant correctable error is a missing secondary membership
        # inside a genuine projection overlap. Start from the mature model's
        # own probabilities and attenuate only GT-overlap foreground. Existing
        # false-negative shapes seed the corruption; a small low-confidence
        # extension supplies cases where the baseline happened to be correct.
        genuine = selected_target * overlap
        real_missing = genuine * (selected_base < 0.5).float()
        low_confidence = genuine * (selected_base < 0.85).float()
        extension = low_confidence * (smooth_field(count, height, width, target.device) > 0.68).float()
        drop = F.max_pool2d((real_missing + extension).clamp(0, 1), 5, stride=1, padding=2) * genuine
        soft_corrupt = (selected_base * (1.0 - 0.85 * drop)).clamp(0.01, 0.99)
    elif profile == "generic":
        field1 = smooth_field(count, height, width, target.device)
        field2 = smooth_field(count, height, width, target.device)
        drop = selected_target * (((overlap > 0) & (field1 > 0.42)) | ((boundary > 0) & (field2 > 0.72))).float()
        drop = F.max_pool2d(drop, 5, stride=1, padding=2) * selected_target
        dilated = F.max_pool2d(selected_target, 9, stride=1, padding=4)
        false_zone = (dilated - selected_target).clamp(0, 1) * other_union
        add = false_zone * (smooth_field(count, height, width, target.device) > 0.78).float()
        hard_corrupt = (selected_target * (1.0 - drop) + add).clamp(0, 1)
        soft_corrupt = F.avg_pool2d(hard_corrupt, 5, stride=1, padding=2).clamp(0.01, 0.99)
    else:
        raise ValueError(f"unknown corruption profile: {profile}")

    baseline_fraction = 1.0 - synthetic_fraction - identity_fraction
    if baseline_fraction < 0:
        raise ValueError("synthetic_fraction + identity_fraction must be <= 1")
    # A mixture of real baseline predictions, focused synthetic corruption,
    # and clean identity samples teaches both correction and non-interference.
    mode = torch.rand((count, 1, 1, 1), device=target.device)
    clean_soft = F.avg_pool2d(selected_target, 3, stride=1, padding=1).clamp(0.01, 0.99)
    source = torch.where(mode < baseline_fraction, selected_base,
                         torch.where(mode < baseline_fraction + synthetic_fraction,
                                     soft_corrupt, clean_soft))
    changed = torch.abs(source - selected_target)
    local = F.max_pool2d((changed > 0.15).float() + boundary, 11, stride=1, padding=5).clamp(0, 1)
    return source[:, 0].unsqueeze(0).clamp(0.01, 0.99), local


def choose_instances(base_probability: torch.Tensor, target: torch.Tensor, count: int) -> torch.Tensor:
    hard = base_probability >= 0.5
    overlap = (target.sum(1, keepdim=True) >= 2)
    local_error = ((hard != (target > 0.5)) & overlap).sum((0, 2, 3)).float()
    top_count = min(2, count)
    chosen = torch.topk(local_error, k=top_count).indices.tolist()
    remaining = [index for index in range(14) if index not in chosen]
    random.shuffle(remaining)
    chosen.extend(remaining[:max(0, count - len(chosen))])
    return torch.tensor(chosen, device=target.device, dtype=torch.long)


def completion_loss(logits: torch.Tensor, target: torch.Tensor, local: torch.Tensor,
                    delta: torch.Tensor, boundary_weight: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    probability = torch.sigmoid(logits.float())
    weight = 1.0 + 3.0 * local
    bce = (F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none") * weight).sum() / weight.sum()
    axes = (1, 2, 3)
    dice = 1.0 - ((2 * (probability * target).sum(axes) + 1) /
                  (probability.sum(axes) + target.sum(axes) + 1)).mean()
    conservation = (torch.abs(delta) * (1.0 - local)).mean()
    target_boundary = (F.max_pool2d(target.float(), 3, 1, 1)
                       + F.max_pool2d(-target.float(), 3, 1, 1)).clamp(0, 1)
    probability_boundary = (F.max_pool2d(probability, 3, 1, 1)
                            + F.max_pool2d(-probability, 3, 1, 1)).clamp(0, 1)
    boundary_band = F.max_pool2d(target_boundary, 7, 1, 3)
    boundary_l1 = (torch.abs(probability_boundary - target_boundary) * boundary_band).sum() / boundary_band.sum().clamp_min(1.0)
    loss = 0.55 * bce + 0.45 * dice + 0.02 * conservation + boundary_weight * boundary_l1
    return loss, {"bce": float(bce.detach()), "dice_loss": float(dice.detach()),
                  "conservation": float(conservation.detach()), "boundary_l1": float(boundary_l1.detach()),
                  "local_fraction": float(local.mean())}


def surfaces(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> tuple[float, float, bool]:
    if not target.any():
        return (1.0 if not pred.any() else 0.0), (0.0 if not pred.any() else float("nan")), bool(pred.any())
    if not pred.any():
        return 0.0, float("nan"), True
    ps, ts = pred ^ binary_erosion(pred), target ^ binary_erosion(target)
    dt, dp = distance_transform_edt(~ts), distance_transform_edt(~ps)
    nsd = float(((dt[ps] <= tolerance).sum() + (dp[ts] <= tolerance).sum()) / max(int(ps.sum() + ts.sum()), 1))
    return nsd, float(np.concatenate((dt[ps], dp[ts])).mean()), False


class MetricAccumulator:
    def __init__(self, device: torch.device) -> None:
        self.inter = torch.zeros(14, device=device)
        self.pred = torch.zeros(14, device=device)
        self.true = torch.zeros(14, device=device)
        self.ov_inter = self.ov_pred = self.ov_true = 0.0
        self.nsd: list[float] = []
        self.msd: list[float] = []
        self.failures = 0

    def add(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        self.inter += (prediction & target).sum((0, 2, 3))
        self.pred += prediction.sum((0, 2, 3))
        self.true += target.sum((0, 2, 3))
        po, to = prediction.sum(1) >= 2, target.sum(1) >= 2
        self.ov_inter += float((po & to).sum())
        self.ov_pred += float(po.sum())
        self.ov_true += float(to.sum())
        for batch_index in range(po.shape[0]):
            nsd, msd, failed = surfaces(po[batch_index].cpu().numpy(), to[batch_index].cpu().numpy())
            self.nsd.append(nsd)
            if np.isfinite(msd): self.msd.append(msd)
            self.failures += int(failed)

    def result(self) -> dict[str, float | list[float]]:
        dice = (2 * self.inter + 1) / (self.pred + self.true + 1)
        iou = (self.inter + 1) / (self.pred + self.true - self.inter + 1)
        od = (2 * self.ov_inter + 1) / (self.ov_pred + self.ov_true + 1)
        oi = (self.ov_inter + 1) / (self.ov_pred + self.ov_true - self.ov_inter + 1)
        return {"macro_dsc": float(dice.mean()), "macro_iou": float(iou.mean()),
                "overlap_dsc": float(od), "overlap_iou": float(oi),
                "overlap_nsd_2px": float(np.mean(self.nsd)),
                "overlap_msd_px": float(np.mean(self.msd)) if self.msd else float("nan"),
                "overlap_msd_fail_rate": self.failures / max(len(self.nsd), 1),
                "class_dsc": dice.cpu().tolist()}


@torch.inference_mode()
def evaluate(baseline: nn.Module, refiner: nn.Module | None, loader: DataLoader,
             device: torch.device, oracle: str | None = None, chunk: int = 3) -> dict:
    baseline.eval()
    if refiner is not None: refiner.eval()
    metrics = MetricAccumulator(device)
    for batch in loader:
        image = batch["image"].to(device)
        target = batch["mask"].to(device) > 0.5
        logits, _ = baseline(image, False)
        probability = torch.sigmoid(logits)
        prediction = probability >= 0.5
        if oracle is not None:
            true_overlap = target.sum(1, keepdim=True) >= 2
            pred_overlap = prediction.sum(1, keepdim=True) >= 2
            region = true_overlap if oracle == "fn_only" else (true_overlap | pred_overlap)
            if oracle == "fn_only":
                prediction = prediction | (target & region)
            else:
                prediction = torch.where(region, target, prediction)
        elif refiner is not None:
            refined = logits.clone()
            for start in range(0, 14, chunk):
                indices = torch.arange(start, min(start + chunk, 14), device=device)
                source = probability[:, indices]
                feature = per_instance_features(image, probability, indices, source)
                source_logit = logits[0, indices].unsqueeze(1)
                corrected, _ = refiner(feature, source_logit)
                refined[0, indices] = corrected[:, 0]
            prediction = torch.sigmoid(refined) >= 0.5
        metrics.add(prediction, target)
    return metrics.result()


def train_arm(name: str, completion: bool, baseline: nn.Module, initial: dict,
              train_set: NativeWristDataset, val_loader: DataLoader, baseline_metrics: dict,
              device: torch.device, output: Path, epochs: int, learning_rate: float,
              min_epochs: int, patience: int, seed: int, instances_per_image: int,
              corruption_profile: str = "generic", synthetic_fraction: float = 0.30,
              identity_fraction: float = 0.20, boundary_weight: float = 0.0) -> tuple[Path, dict]:
    seed_all(seed)
    refiner = InstanceCompletionRefiner().to(device)
    refiner.load_state_dict(copy.deepcopy(initial), strict=True)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
                        persistent_workers=True, generator=loader_generator)
    checkpoint = output / f"{name}_best.pth"
    history = output / f"{name}_history.jsonl"
    best = None
    bad = 0
    for epoch in range(1, epochs + 1):
        refiner.train()
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
                if completion:
                    source_probability, local = corrupt_probabilities(
                        base_probability, target, indices, corruption_profile,
                        synthetic_fraction, identity_fraction)
                else:
                    source_probability = base_probability[0, indices].unsqueeze(0).detach()
                    selected_target = target[0, indices].unsqueeze(1)
                    error = torch.abs(source_probability[0].unsqueeze(1) - selected_target)
                    local = F.max_pool2d((error > 0.15).float(), 11, stride=1, padding=5)
                source_logit = torch.logit(source_probability[0].unsqueeze(1).clamp(0.01, 0.99))
                feature = per_instance_features(image, base_probability, indices, source_probability)
                selected_target = target[0, indices].unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                corrected, delta = refiner(feature, source_logit)
                loss, stats = completion_loss(corrected, selected_target, local, delta, boundary_weight)
            if not torch.isfinite(loss):
                raise RuntimeError({"arm": name, "epoch": epoch, "loss": float(loss)})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            for key, value in stats.items(): totals[key] += value
        scheduler.step()
        metrics = evaluate(baseline, refiner, val_loader, device)
        passes = metrics["macro_dsc"] >= baseline_metrics["macro_dsc"] and metrics["macro_iou"] >= baseline_metrics["macro_iou"]
        row = {"arm": name, "completion_prior": completion, "epoch": epoch,
               "seconds": time.time() - started, "learning_rate": optimizer.param_groups[0]["lr"],
               "corruption_profile": corruption_profile if completion else "none",
               "synthetic_fraction": synthetic_fraction if completion else 0.0,
               "identity_fraction": identity_fraction if completion else 0.0,
               "boundary_weight": boundary_weight,
               "passes_macro_non_degradation": passes,
               **{f"train_{key}": value / len(loader) for key, value in totals.items()}, **metrics}
        with history.open("a", encoding="utf-8") as handle: handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = (int(passes), metrics["overlap_dsc"] if passes else metrics["macro_dsc"],
               metrics["overlap_nsd_2px"], -metrics["overlap_msd_px"])
        if best is None or key > best[0]:
            best = (key, row)
            bad = 0
            torch.save({"refiner": refiner.state_dict(), "row": row, "arm": name,
                        "completion_prior": completion}, checkpoint)
        else:
            bad += 1
        if epoch >= min_epochs and bad >= patience: break
    assert best is not None
    return checkpoint, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ram_w600/r327_native_instance_completion_prior"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--instances-per-image", type=int, default=4)
    parser.add_argument("--corruption-profile", choices=("generic", "fn_focus"), default="generic")
    parser.add_argument("--synthetic-fraction", type=float, default=0.30)
    parser.add_argument("--identity-fraction", type=float, default=0.20)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--experiment-name", default="R327_RAM_NATIVE_INSTANCE_COMPLETION_PRIOR")
    parser.add_argument("--seed", type=int, default=3271)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError("R327 requires CUDA")
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
    oracle_fn = evaluate(baseline, None, val_loader, device, oracle="fn_only")
    oracle_full = evaluate(baseline, None, val_loader, device, oracle="full")
    oracle = {"baseline": baseline_metrics, "fn_only": oracle_fn, "full_local": oracle_full,
              "delta_fn_only": {key: oracle_fn[key] - baseline_metrics[key] for key in
                                ("macro_dsc", "macro_iou", "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px")},
              "delta_full_local": {key: oracle_full[key] - baseline_metrics[key] for key in
                                  ("macro_dsc", "macro_iou", "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px")}}
    (args.output / "oracle.json").write_text(json.dumps(oracle, indent=2), encoding="utf-8")
    print(json.dumps({"oracle": oracle}, indent=2), flush=True)
    if args.smoke:
        refiner = InstanceCompletionRefiner().to(device)
        sample = train_set[0]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["mask"].unsqueeze(0).to(device)
        with torch.no_grad(): logits, _ = baseline(image, False); probability = torch.sigmoid(logits)
        indices = choose_instances(probability, target, args.instances_per_image)
        source, local = corrupt_probabilities(probability, target, indices, args.corruption_profile,
                                              args.synthetic_fraction, args.identity_fraction)
        feature = per_instance_features(image, probability, indices, source)
        corrected, delta = refiner(feature, torch.logit(source[0].unsqueeze(1)))
        loss, stats = completion_loss(corrected, target[0, indices].unsqueeze(1), local, delta,
                                      args.boundary_weight)
        loss.backward()
        print(json.dumps({"smoke": True, "loss": float(loss), "stats": stats,
                          "feature": list(feature.shape), "finite": bool(torch.isfinite(loss))}), flush=True)
        return

    initial_module = InstanceCompletionRefiner().to(device)
    initial = copy.deepcopy(initial_module.state_dict())
    del initial_module
    capacity_ckpt, capacity_best = train_arm(
        "capacity_control", False, baseline, initial, train_set, val_loader, baseline_metrics,
        device, args.output, args.epochs, args.learning_rate, args.min_epochs, args.patience,
        args.seed, args.instances_per_image, args.corruption_profile, args.synthetic_fraction,
        args.identity_fraction, args.boundary_weight)
    completion_ckpt, completion_best = train_arm(
        "completion_prior", True, baseline, initial, train_set, val_loader, baseline_metrics,
        device, args.output, args.epochs, args.learning_rate, args.min_epochs, args.patience,
        args.seed, args.instances_per_image, args.corruption_profile, args.synthetic_fraction,
        args.identity_fraction, args.boundary_weight)

    def final(path: Path) -> dict:
        refiner = InstanceCompletionRefiner().to(device)
        refiner.load_state_dict(torch.load(path, map_location=device, weights_only=False)["refiner"], strict=True)
        return evaluate(baseline, refiner, val_loader, device)

    capacity_metrics, completion_metrics = final(capacity_ckpt), final(completion_ckpt)
    keys = ("macro_dsc", "macro_iou", "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px")
    result = {"experiment": args.experiment_name, "split": "validation",
              "test_used": False, "threshold": 0.5, "threshold_search": False,
              "spatial_preprocessing": "native pixels; right/bottom padding only",
              "baseline": baseline_metrics, "oracle": oracle, "capacity_control": capacity_metrics,
              "completion_prior": completion_metrics,
              "delta_completion_minus_baseline": {key: completion_metrics[key] - baseline_metrics[key] for key in keys},
              "delta_completion_minus_capacity": {key: completion_metrics[key] - capacity_metrics[key] for key in keys},
              "best_rows": {"capacity_control": capacity_best, "completion_prior": completion_best},
              "initialization": {"baseline": str(args.baseline_checkpoint), "sha256": sha256(args.baseline_checkpoint),
                                 "baseline_frozen": True, "same_refiner_initialization": True},
              "corruption_mix": {"baseline_prediction": 1.0 - args.synthetic_fraction - args.identity_fraction,
                                 "algorithmic_corruption": args.synthetic_fraction,
                                 "clean_identity": args.identity_fraction,
                                 "profile": args.corruption_profile},
              "boundary_weight": args.boundary_weight,
              "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
