"""R293-B: explicit dual branches plus non-saturating gap/support supervision."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from nnUNetTrainerR293DualBranch import nnUNetTrainerR293DualBranch


def shift_zero(value: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    out = torch.zeros_like(value)
    h, w = value.shape[-2:]
    yd = slice(max(0, dy), min(h, h + dy))
    xd = slice(max(0, dx), min(w, w + dx))
    ys = slice(max(0, -dy), min(h, h - dy))
    xs = slice(max(0, -dx), min(w, w - dx))
    out[..., yd, xd] = value[..., ys, xs]
    return out


def directional_gap_support(target: torch.Tensor, radius: int = 14,
                            gap_halfwidth: int = 2, support_outer: int = 12,
                            safety_buffer: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Find background pixels bracketed by foreground in opposing directions."""
    foreground = (target == 1).float()
    left = right = up = down = torch.zeros_like(foreground)
    nw = se = ne = sw = torch.zeros_like(foreground)
    for distance in range(1, radius + 1):
        left = torch.maximum(left, shift_zero(foreground, 0, distance))
        right = torch.maximum(right, shift_zero(foreground, 0, -distance))
        up = torch.maximum(up, shift_zero(foreground, distance, 0))
        down = torch.maximum(down, shift_zero(foreground, -distance, 0))
        nw = torch.maximum(nw, shift_zero(foreground, distance, distance))
        se = torch.maximum(se, shift_zero(foreground, -distance, -distance))
        ne = torch.maximum(ne, shift_zero(foreground, distance, -distance))
        sw = torch.maximum(sw, shift_zero(foreground, -distance, distance))
    opposed = (left * right + up * down + nw * se + ne * sw).clamp_max(1)
    gap = opposed * (1.0 - foreground)
    if gap_halfwidth > 0:
        gap = F.max_pool2d(gap, 2 * gap_halfwidth + 1, stride=1,
                           padding=gap_halfwidth) * (1.0 - foreground)
    near = F.max_pool2d(gap, 2 * support_outer + 1, stride=1, padding=support_outer)
    buffer = F.max_pool2d(gap, 2 * safety_buffer + 1, stride=1, padding=safety_buffer)
    support = foreground * near * (1.0 - buffer)
    return gap, support


def weighted_case_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numerator = (value * weight).flatten(1).sum(1)
    denominator = weight.flatten(1).sum(1)
    valid = denominator > 1e-6
    if not valid.any():
        return value.sum() * 0
    return (numerator[valid] / (denominator[valid] + 1e-6)).mean()


class nnUNetTrainerR293DualBranchGap(nnUNetTrainerR293DualBranch):
    """Add wide gap-band, safe support and explicit prior-map objectives."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.gap_alpha = 0.03
        self.support_alpha = 0.008
        self.prior_alpha = 0.015
        self.gap_log_odds_ceiling = math.log(0.20 / 0.80)
        self.support_log_odds_floor = math.log(0.75 / 0.25)
        np.random.seed(294)
        torch.manual_seed(294)
        torch.cuda.manual_seed_all(294)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([item.to(self.device, non_blocking=True) for item in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        full_target = target[0] if isinstance(target, list) else target
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            gap, support = directional_gap_support(full_target.float())
            log_odds = logits[:, 1:2].float() - logits[:, 0:1].float()
            gap_hinge = weighted_case_mean(
                F.relu(log_odds - self.gap_log_odds_ceiling), gap)
            support_hinge = weighted_case_mean(
                F.relu(self.support_log_odds_floor - log_odds), support)
            prior_logits = self.network.last_prior_logits.float()
            prior_bce = weighted_case_mean(
                F.softplus(-prior_logits), gap) + weighted_case_mean(
                F.softplus(prior_logits), support)
            loss = (base_loss + self.gap_alpha * gap_hinge +
                    self.support_alpha * support_hinge + self.prior_alpha * prior_bce)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        trainable = [p for p in self.network.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 12)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "gap_hinge": gap_hinge.detach().cpu().numpy(),
            "support_hinge": support_hinge.detach().cpu().numpy(),
            "prior_bce": prior_bce.detach().cpu().numpy(),
            "gap_fraction": gap.mean().detach().cpu().numpy(),
            "support_fraction": support.mean().detach().cpu().numpy(),
            "gate_mean": self.network.last_local_gate.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs: list[dict]):
        # Call nnU-Net's implementation directly, then write the richer R293-B row.
        super(nnUNetTrainerR293DualBranch, self).on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0]}
        row.update({
            "epoch": int(self.current_epoch),
            "global_residual_absmax": float(self.network.global_branch[-1].weight.abs().max().detach().cpu()),
            "local_residual_absmax": float(self.network.local_residual.weight.abs().max().detach().cpu()),
        })
        with (Path(self.output_folder) / "r293b_loss_dynamics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
