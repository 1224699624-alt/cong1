#!/usr/bin/env python3
"""R286: pair-specific, relation-conditioned local interface prior on RAM-W600.

The official test split is evaluated once after validation-only checkpoint selection.
No threshold search is performed. The mature R285 baseline is reused as a frozen
teacher and as the exact initialization of the improved model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import (
    BONE_NAMES,
    NnUNetMultiLabelPrior,
    WristDataset,
    dice_bce_logits,
    render_comparisons,
    seed_everything,
    surface_dice,
    top_training_pairs,
)


class PairRelationInterfaceNnUNet(nn.Module):
    """nnU-Net with pair-wise gap/overlap/uncertain maps at the local interface."""

    def __init__(
        self,
        pairs: list[tuple[int, int]],
        directional_confidence_gate: bool = False,
    ) -> None:
        super().__init__()
        legacy = NnUNetMultiLabelPrior()
        self.backbone = legacy.backbone
        self.pairs = pairs
        self.pair_count = len(pairs)
        self.directional_confidence_gate = directional_confidence_gate

        # The last two high-resolution decoder stages contain 64 and 32 channels.
        self.half_projection = nn.Conv2d(64, 32, 1, bias=False)
        self.relation_fusion = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, self.pair_count * 3, 1),
        )
        # Exact zero makes the initial prediction identical to the mature baseline.
        # Gains are projected to the non-negative half-line after every optimizer step.
        self.gap_gain = nn.Parameter(torch.zeros(self.pair_count))
        self.overlap_gain = nn.Parameter(torch.zeros(self.pair_count))
        incidence = torch.zeros(self.pair_count, 14)
        for pair_index, (first, second) in enumerate(pairs):
            incidence[pair_index, first] = 1.0
            incidence[pair_index, second] = 1.0
        self.register_buffer("incidence", incidence)
        self._decoder_features: dict[str, torch.Tensor] = {}
        self.backbone.decoder.stages[-2].register_forward_hook(self._capture("half"))
        self.backbone.decoder.stages[-1].register_forward_hook(self._capture("full"))

    def _capture(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._decoder_features[name] = output
        return hook

    def load_r285_baseline(self, checkpoint: Path, device: torch.device) -> dict:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        legacy = NnUNetMultiLabelPrior().to(device)
        legacy.load_state_dict(payload["model"])
        self.backbone.load_state_dict(legacy.backbone.state_dict())
        return payload

    def set_trainable_phase(self, phase: str) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        for module in (self.half_projection, self.relation_fusion):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.gap_gain.requires_grad = True
        self.overlap_gain.requires_grad = True
        if phase == "decoder_finetune":
            for stage in self.backbone.decoder.stages[-2:]:
                for parameter in stage.parameters():
                    parameter.requires_grad = True
            for layer in self.backbone.decoder.seg_layers[-2:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._decoder_features.clear()
        raw = self.backbone(image)
        baseline_logits = raw[:, :14]
        half = F.interpolate(
            self.half_projection(self._decoder_features["half"]),
            size=self._decoder_features["full"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        relation_input = torch.cat([half, self._decoder_features["full"]], dim=1)
        relation_logits = self.relation_fusion(relation_input).reshape(
            image.shape[0], self.pair_count, 3, *image.shape[-2:]
        )
        relation_probability = torch.softmax(relation_logits.float(), dim=2)
        gap = relation_probability[:, :, 0]
        overlap = relation_probability[:, :, 1]
        if self.directional_confidence_gate:
            baseline_probability = torch.sigmoid(baseline_logits.float()).detach()
            uncertainty = 4.0 * baseline_probability * (1.0 - baseline_probability)
            bone_delta: list[torch.Tensor | float] = [0.0 for _ in range(14)]
            for pair_index, (first, second) in enumerate(self.pairs):
                relation_overlap = overlap[:, pair_index]
                relation_gap = gap[:, pair_index]
                first_delta = (
                    self.overlap_gain[pair_index] * relation_overlap
                    * baseline_probability[:, second] * uncertainty[:, first]
                    - self.gap_gain[pair_index] * relation_gap * uncertainty[:, first]
                )
                second_delta = (
                    self.overlap_gain[pair_index] * relation_overlap
                    * baseline_probability[:, first] * uncertainty[:, second]
                    - self.gap_gain[pair_index] * relation_gap * uncertainty[:, second]
                )
                bone_delta[first] = bone_delta[first] + first_delta
                bone_delta[second] = bone_delta[second] + second_delta
            delta = torch.stack([
                value if isinstance(value, torch.Tensor)
                else torch.zeros_like(baseline_logits[:, 0])
                for value in bone_delta
            ], dim=1)
        else:
            pair_delta = (
                self.overlap_gain[None, :, None, None] * overlap
                - self.gap_gain[None, :, None, None] * gap
            )
            delta = torch.einsum("bphw,pc->bchw", pair_delta, self.incidence)
        return baseline_logits + delta.to(baseline_logits.dtype), relation_logits


def interface_targets(
    target: torch.Tensor,
    pairs: list[tuple[int, int]],
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relation class, ROI, overlap and gap masks for every selected pair."""
    binary = target > 0.5
    dilated = F.max_pool2d(binary.float(), kernel_size=2 * radius + 1, stride=1, padding=radius) > 0.5
    labels: list[torch.Tensor] = []
    rois: list[torch.Tensor] = []
    overlaps: list[torch.Tensor] = []
    gaps: list[torch.Tensor] = []
    for first, second in pairs:
        first_mask, second_mask = binary[:, first], binary[:, second]
        overlap = first_mask & second_mask
        roi = (dilated[:, first] & dilated[:, second]) | overlap
        gap = roi & ~first_mask & ~second_mask
        label = torch.full_like(first_mask, 2, dtype=torch.long)  # uncertain/support
        label[gap] = 0
        label[overlap] = 1
        labels.append(label)
        rois.append(roi)
        overlaps.append(overlap)
        gaps.append(gap)
    return (
        torch.stack(labels, dim=1),
        torch.stack(rois, dim=1),
        torch.stack(overlaps, dim=1),
        torch.stack(gaps, dim=1),
    )


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    return (value.float() * weight).sum() / weight.sum().clamp_min(1.0)


def soft_morphological_boundary(
    probability: torch.Tensor, clamp_for_log: bool = True
) -> torch.Tensor:
    """Differentiable 3x3 morphological gradient for soft pair-overlap maps."""
    dilated = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-probability, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded).clamp(0.0, 1.0)
    return boundary.clamp(1e-4, 1 - 1e-4) if clamp_for_log else boundary


def relation_interface_losses(
    logits: torch.Tensor,
    relation_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    pairs: list[tuple[int, int]],
    radius: int,
) -> dict[str, torch.Tensor]:
    labels, roi, overlap_mask, gap_mask = interface_targets(target, pairs, radius)
    batch, pair_count, _, height, width = relation_logits.shape
    relation_ce = F.cross_entropy(
        relation_logits.float().reshape(batch * pair_count, 3, height, width),
        labels.reshape(batch * pair_count, height, width),
        weight=torch.tensor([1.0, 2.0, 0.25], device=logits.device),
        reduction="none",
    ).reshape(batch, pair_count, height, width)
    relation = masked_mean(relation_ce, roi)

    probability = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    target_binary = target > 0.5
    pair_probability: list[torch.Tensor] = []
    endpoint_probability: list[torch.Tensor] = []
    endpoint_target: list[torch.Tensor] = []
    for first, second in pairs:
        pair_probability.append(probability[:, first] * probability[:, second])
        endpoint_probability.extend([probability[:, first], probability[:, second]])
        endpoint_target.extend([target_binary[:, first], target_binary[:, second]])
    pair_probability_t = torch.stack(pair_probability, dim=1)
    overlap_target = overlap_mask.float()
    # Probability-form BCE is intentionally written out in FP32 because PyTorch
    # rejects torch.binary_cross_entropy inside CUDA autocast regions.
    overlap_bce = -(
        overlap_target * pair_probability_t.log()
        + (1.0 - overlap_target) * (1.0 - pair_probability_t).log()
    ) * torch.where(overlap_mask, 4.0, 1.0)
    overlap = masked_mean(overlap_bce, roi)

    # Match the contour of every pair-specific overlap region, not only its area.
    predicted_boundary = soft_morphological_boundary(pair_probability_t)
    target_boundary = soft_morphological_boundary(overlap_target, clamp_for_log=False).detach()
    roi_float = roi.float()
    axes = (-2, -1)
    boundary_dice = 1.0 - (
        2.0 * (predicted_boundary * target_boundary * roi_float).sum(axes) + 1.0
    ) / (
        (predicted_boundary * roi_float).sum(axes)
        + (target_boundary * roi_float).sum(axes) + 1.0
    )
    roi_mass = roi_float.sum(axes, keepdim=True).clamp_min(1.0)
    positive_mass = (target_boundary * roi_float).sum(axes, keepdim=True)
    positive_weight = (roi_mass - positive_mass) / roi_mass
    negative_weight = positive_mass / roi_mass
    boundary_bce = -(
        positive_weight * target_boundary * predicted_boundary.log()
        + negative_weight * (1.0 - target_boundary) * (1.0 - predicted_boundary).log()
    )
    boundary = boundary_dice.mean() + masked_mean(boundary_bce, roi)

    endpoint_probability_t = torch.stack(endpoint_probability, dim=1).reshape(
        batch, pair_count, 2, height, width
    )
    endpoint_target_t = torch.stack(endpoint_target, dim=1).reshape(
        batch, pair_count, 2, height, width
    )
    gap = masked_mean(endpoint_probability_t, gap_mask[:, :, None].expand_as(endpoint_probability_t))
    support_mask = roi[:, :, None] & endpoint_target_t
    support = masked_mean(-endpoint_probability_t.log(), support_mask)

    # Preserve the mature teacher everywhere outside the union of selected interfaces.
    outside = ~roi.any(dim=1, keepdim=True)
    keep_map = F.binary_cross_entropy_with_logits(
        logits.float(), torch.sigmoid(teacher_logits.float()), reduction="none"
    )
    keep = masked_mean(keep_map, outside.expand_as(keep_map))
    return {
        "relation": relation,
        "overlap": overlap,
        "boundary": boundary,
        "gap": gap,
        "support": support,
        "keep": keep,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pairs: list[tuple[int, int]],
    collect: bool = False,
) -> tuple[dict, dict[str, dict[str, np.ndarray]] | None]:
    model.eval()
    intersection = torch.zeros(14, device=device)
    predicted_mass = torch.zeros(14, device=device)
    target_mass = torch.zeros(14, device=device)
    overlap_intersection = overlap_predicted = overlap_target = 0.0
    pair_intersection = torch.zeros(len(pairs), device=device)
    pair_predicted = torch.zeros(len(pairs), device=device)
    pair_target = torch.zeros(len(pairs), device=device)
    overlap_nsd: list[float] = []
    pair_nsd: list[list[float]] = [[] for _ in pairs]
    predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for batch_data in loader:
        logits = model(batch_data["image"].to(device))[0]
        prediction = torch.sigmoid(logits) >= 0.5
        target = batch_data["mask"].to(device) > 0.5
        intersection += (prediction & target).sum((0, 2, 3))
        predicted_mass += prediction.sum((0, 2, 3))
        target_mass += target.sum((0, 2, 3))
        prediction_overlap = prediction.sum(1) >= 2
        target_overlap_map = target.sum(1) >= 2
        overlap_intersection += float((prediction_overlap & target_overlap_map).sum())
        overlap_predicted += float(prediction_overlap.sum())
        overlap_target += float(target_overlap_map.sum())
        prediction_np = prediction.cpu().numpy()
        target_np = target.cpu().numpy()
        for sample_index, case in enumerate(batch_data["case"]):
            overlap_nsd.append(surface_dice(prediction_np[sample_index].sum(0) >= 2,
                                            target_np[sample_index].sum(0) >= 2))
            if collect:
                predictions[str(case)] = prediction_np[sample_index]
                targets[str(case)] = target_np[sample_index]
        for pair_index, (first, second) in enumerate(pairs):
            pred_pair = prediction[:, first] & prediction[:, second]
            target_pair_map = target[:, first] & target[:, second]
            pair_intersection[pair_index] += (pred_pair & target_pair_map).sum()
            pair_predicted[pair_index] += pred_pair.sum()
            pair_target[pair_index] += target_pair_map.sum()
            for sample_index in range(prediction.shape[0]):
                pair_nsd[pair_index].append(surface_dice(
                    prediction_np[sample_index, first] & prediction_np[sample_index, second],
                    target_np[sample_index, first] & target_np[sample_index, second],
                ))
    class_dice = (2 * intersection + 1) / (predicted_mass + target_mass + 1)
    pair_dice = (2 * pair_intersection + 1) / (pair_predicted + pair_target + 1)
    metrics = {
        "macro_dice": float(class_dice.mean()),
        "class_dice": class_dice.cpu().tolist(),
        "overlap_dice": float((2 * overlap_intersection + 1) / (overlap_predicted + overlap_target + 1)),
        "overlap_nsd_2px": float(np.mean(overlap_nsd)),
        "pair_overlap_dice": pair_dice.cpu().tolist(),
        "pair_overlap_nsd_2px": [float(np.mean(values)) for values in pair_nsd],
    }
    collected = {"predictions": predictions, "targets": targets} if collect else None
    return metrics, collected


@torch.no_grad()
def evaluate_legacy(
    model: NnUNetMultiLabelPrior,
    loader: DataLoader,
    device: torch.device,
    pairs: list[tuple[int, int]],
    collect: bool = False,
) -> tuple[dict, dict[str, dict[str, np.ndarray]] | None]:
    class Adapter(nn.Module):
        def __init__(self, wrapped: NnUNetMultiLabelPrior) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.wrapped(image, False)

    return evaluate(Adapter(model), loader, device, pairs, collect)


def train(
    model: PairRelationInterfaceNnUNet,
    teacher: NnUNetMultiLabelPrior,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output: Path,
    pairs: list[tuple[int, int]],
    baseline_val: dict,
    head_epochs: int,
    finetune_epochs: int,
    radius: int,
    boundary_weight: float,
    selection_metric: str,
) -> tuple[Path, dict]:
    checkpoint = output / "improved_best.pth"
    initial_metrics, _ = evaluate(model, val_loader, device, pairs)
    torch.save({"model": model.state_dict(), "metrics": initial_metrics, "epoch": -1,
                "phase": "baseline_identity", "pairs": pairs}, checkpoint)
    # Macro Dice is a hard eligibility gate; among eligible checkpoints, optimize
    # overlap Dice and use macro Dice only as the tie breaker.
    best_key = (1.0, float(initial_metrics[selection_metric]),
                float(initial_metrics["overlap_dice"]),
                float(initial_metrics["macro_dice"]))
    history = output / "improved_history.jsonl"
    weights = {"relation": 0.03, "overlap": 0.015, "boundary": boundary_weight, "gap": 0.005,
               "support": 0.005, "keep": 0.05}
    epoch_global = 0
    for phase, epochs, learning_rate in (
        ("relation_warmup", head_epochs, 2e-4),
        ("decoder_finetune", finetune_epochs, 2e-5),
    ):
        model.set_trainable_phase(phase)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        bad_epochs = 0
        for phase_epoch in range(epochs):
            model.train()
            teacher.eval()
            sums = {"total": 0.0, "base": 0.0, **{key: 0.0 for key in weights}}
            started = time.time()
            for batch_data in train_loader:
                image = batch_data["image"].to(device, non_blocking=True)
                target = batch_data["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    teacher_logits, _ = teacher(image, False)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits, relation_logits = model(image)
                    base = dice_bce_logits(logits, target)
                    terms = relation_interface_losses(
                        logits, relation_logits, teacher_logits, target, pairs, radius
                    )
                    total = base + sum(weights[key] * terms[key] for key in weights)
                if not torch.isfinite(total):
                    raise RuntimeError({key: float(value.detach().float())
                                        for key, value in {"total": total, "base": base, **terms}.items()})
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                optimizer.step()
                with torch.no_grad():
                    model.gap_gain.clamp_(min=0.0)
                    model.overlap_gain.clamp_(min=0.0)
                sums["total"] += float(total.detach())
                sums["base"] += float(base.detach())
                for key in weights:
                    sums[key] += float(terms[key].detach())
            scheduler.step()
            metrics, _ = evaluate(model, val_loader, device, pairs)
            eligible = float(metrics["macro_dice"]) >= float(baseline_val["macro_dice"]) - 2e-4
            key = ((1.0, float(metrics[selection_metric]), float(metrics["overlap_dice"]),
                    float(metrics["macro_dice"]))
                   if eligible else (0.0, float(metrics["macro_dice"]),
                                     float(metrics["overlap_dice"]), float(metrics[selection_metric])))
            row = {
                "phase": phase, "phase_epoch": phase_epoch, "epoch": epoch_global,
                "seconds": time.time() - started, "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train_{name}": value / len(train_loader) for name, value in sums.items()},
                "val_macro_dice": metrics["macro_dice"], "val_overlap_dice": metrics["overlap_dice"],
                "val_overlap_nsd_2px": metrics["overlap_nsd_2px"],
                "eligible_macro_gate": eligible,
                "gap_gain_mean": float(model.gap_gain.mean().detach()),
                "overlap_gain_mean": float(model.overlap_gain.mean().detach()),
            }
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if key > best_key:
                best_key = key
                bad_epochs = 0
                torch.save({"model": model.state_dict(), "metrics": metrics, "epoch": epoch_global,
                            "phase": phase, "pairs": pairs}, checkpoint)
            else:
                bad_epochs += 1
            epoch_global += 1
            if phase == "decoder_finetune" and phase_epoch >= 8 and bad_epochs >= 7:
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return checkpoint, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\gutou\RAM-W600"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ram_w600/r286_pair_relation_interface_prior"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pair-count", type=int, default=15)
    parser.add_argument("--interface-radius", type=int, default=7)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--boundary-weight", type=float, default=0.0)
    parser.add_argument("--selection-metric", choices=("overlap_dice", "overlap_nsd_2px"),
                        default="overlap_dice")
    parser.add_argument("--experiment-name", default="R286")
    parser.add_argument("--directional-confidence-gate", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--seed", type=int, default=286)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if "TSRS_RSNA-Articular-Surface" in str(args.dataset_root):
        raise RuntimeError("R286 is RAM-W600 only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = top_training_pairs(args.dataset_root, args.pair_count)
    train_set = WristDataset(args.dataset_root, "train", args.size, augment=True)
    val_set = WristDataset(args.dataset_root, "val", args.size, augment=False)
    test_set = WristDataset(args.dataset_root, "test", args.size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    teacher = NnUNetMultiLabelPrior().to(device)
    baseline_payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    if baseline_payload.get("phase") != "baseline":
        raise RuntimeError(f"Checkpoint is not a baseline: {args.baseline_checkpoint}")
    teacher.load_state_dict(baseline_payload["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    model = PairRelationInterfaceNnUNet(
        pairs, directional_confidence_gate=args.directional_confidence_gate
    ).to(device)
    model.load_r285_baseline(args.baseline_checkpoint, device)

    manifest = {
        "experiment": args.experiment_name, "dataset": str(args.dataset_root),
        "splits": {"train": 425, "val": 69, "test": 124},
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "architecture": "nnU-Net v2 PlainConvUNet + pair-specific 3-relation local-interface head",
        "relations": ["gap", "overlap", "uncertain_support"],
        "pair_count": len(pairs), "interface_radius": args.interface_radius,
        "pairs": [{"indices": pair, "names": [BONE_NAMES[pair[0]], BONE_NAMES[pair[1]]]}
                  for pair in pairs],
        "checkpoint_selection": "validation macro Dice gate then overlap Dice; baseline identity included",
        "selection_metric_after_macro_gate": args.selection_metric,
        "boundary_weight": args.boundary_weight,
        "directional_confidence_gate": args.directional_confidence_gate,
        "test_used_for_model_selection": False, "threshold": 0.5,
        "threshold_search_used": False, "clean_test_v2_used": False,
        "test_evaluation_skipped": args.skip_test,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.dry_run:
        batch_data = next(iter(train_loader))
        image = batch_data["image"].to(device)
        target = batch_data["mask"].to(device)
        with torch.no_grad():
            teacher_logits, _ = teacher(image, False)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits, relation_logits = model(image)
            terms = relation_interface_losses(logits, relation_logits, teacher_logits, target,
                                              pairs, args.interface_radius)
            total = dice_bce_logits(logits, target) + sum(terms.values())
        total.backward()
        print(json.dumps({
            "logits": list(logits.shape), "relations": list(relation_logits.shape),
            "losses": {key: float(value.detach()) for key, value in terms.items()},
            "total_finite": bool(torch.isfinite(total)), "device": str(device),
        }), flush=True)
        return

    baseline_val, _ = evaluate_legacy(teacher, val_loader, device, pairs)
    improved_checkpoint, improved_payload = train(
        model, teacher, train_loader, val_loader, device, args.output, pairs, baseline_val,
        args.head_epochs, args.finetune_epochs, args.interface_radius, args.boundary_weight,
        args.selection_metric,
    )
    if args.skip_test:
        baseline_val_final, baseline_predictions = evaluate_legacy(
            teacher, val_loader, device, pairs, collect=True
        )
        improved_val_final, improved_predictions = evaluate(
            model, val_loader, device, pairs, collect=True
        )
        assert baseline_predictions is not None and improved_predictions is not None
        result = {
            "development_only": True,
            "official_test_evaluated": False,
            "baseline_val": baseline_val_final,
            "improved_val": improved_val_final,
            "selected_epoch": improved_payload["epoch"],
            "selected_phase": improved_payload["phase"],
            "delta_val": {
                key: float(improved_val_final[key]) - float(baseline_val_final[key])
                for key in ("macro_dice", "overlap_dice", "overlap_nsd_2px")
            },
            "threshold": 0.5,
            "threshold_search_used": False,
        }
        (args.output / "development_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        overlap_rank = sorted(
            baseline_predictions["targets"],
            key=lambda case: int((baseline_predictions["targets"][case].sum(0) >= 2).sum()),
            reverse=True,
        )[:12]
        render_comparisons(args.dataset_root, args.output / "val_visualizations", overlap_rank,
                           baseline_predictions, improved_predictions, args.size)
        print(json.dumps({"status": "complete_development_only",
                          "official_test_evaluated": False,
                          "delta_val": result["delta_val"]}), flush=True)
        return
    baseline_test, baseline_predictions = evaluate_legacy(
        teacher, test_loader, device, pairs, collect=True
    )
    improved_test, improved_predictions = evaluate(model, test_loader, device, pairs, collect=True)
    assert baseline_predictions is not None and improved_predictions is not None
    result = {
        "baseline_val": baseline_val,
        "improved_val": improved_payload["metrics"],
        "selected_epoch": improved_payload["epoch"],
        "selected_phase": improved_payload["phase"],
        "baseline_test": baseline_test,
        "improved_test": improved_test,
        "delta_test": {
            key: float(improved_test[key]) - float(baseline_test[key])
            for key in ("macro_dice", "overlap_dice", "overlap_nsd_2px")
        },
        "threshold": 0.5, "threshold_search_used": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    overlap_rank = sorted(
        baseline_predictions["targets"],
        key=lambda case: int((baseline_predictions["targets"][case].sum(0) >= 2).sum()),
        reverse=True,
    )[:12]
    render_comparisons(args.dataset_root, args.output / "visualizations", overlap_rank,
                       baseline_predictions, improved_predictions, args.size)
    print(json.dumps({"status": "complete", "checkpoint": str(improved_checkpoint),
                      "delta_test": result["delta_test"]}), flush=True)


if __name__ == "__main__":
    main()
