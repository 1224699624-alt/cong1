"""R283: spatially and confidence-gated bone-side support for instance seams."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from nnUNetTrainerR280BalancedInstancePrior import (
    nnUNetTrainerR280BalancedInstancePrior,
    weighted_case_mean,
)


def gated_support_weight(
    prior: torch.Tensor,
    bone: torch.Tensor,
    foreground_probability: torch.Tensor,
    *,
    buffer_radius: int,
    outer_radius: int,
    interior_radius: int,
    confidence_low: float,
    confidence_high: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return support only on confident bone-interior shoulders outside the seam buffer."""
    if not 0 <= buffer_radius < outer_radius:
        raise ValueError((buffer_radius, outer_radius))
    if not 0 <= confidence_low < confidence_high <= 1:
        raise ValueError((confidence_low, confidence_high))

    def dilate(value: torch.Tensor, radius: int) -> torch.Tensor:
        return F.max_pool2d(value[:, None], 2 * radius + 1, stride=1, padding=radius)[:, 0]

    # The normalized high-probability seam core defines a strict no-support buffer.
    seam_core = (prior >= 0.5).to(prior.dtype)
    seam_buffer = dilate(seam_core, buffer_radius)
    outer_band = dilate(prior, outer_radius)
    shoulder = outer_band * (1.0 - seam_buffer)

    # Erode the binary target so support never acts on an uncertain label boundary.
    if interior_radius:
        bone_interior = 1.0 - dilate(1.0 - bone, interior_radius)
    else:
        bone_interior = bone

    # Detached prediction confidence makes support protective, not self-reinforcing.
    confidence = ((foreground_probability.detach() - confidence_low)
                  / (confidence_high - confidence_low)).clamp(0, 1).square()
    weight = shoulder.square() * bone_interior * confidence
    diagnostics = {
        "seam_buffer": seam_buffer,
        "shoulder": shoulder,
        "bone_interior": bone_interior,
        "support_confidence": confidence,
    }
    return weight, diagnostics


class nnUNetTrainerR283GatedSupportInstancePrior(nnUNetTrainerR280BalancedInstancePrior):
    """Protect only confident bone interiors beside, never within, the seam center."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Fixed from R282's converged effective coefficient (0.05 * about 0.097).
        # It is not tuned or dynamically normalized in R283.
        self.support_alpha = 0.00485
        self.support_buffer_radius = 3
        self.support_outer_radius = 8
        self.support_interior_radius = 2
        self.support_confidence_low = 0.70
        self.support_confidence_high = 0.90
        np.random.seed(283)
        torch.manual_seed(283)
        torch.cuda.manual_seed_all(283)

    def initialize(self):
        super().initialize()
        parent = Path(self.output_folder) / "r280_initialization.json"
        manifest = json.loads(parent.read_text(encoding="utf-8"))
        manifest.update({
            "experiment": "R283",
            "controlled_change": "spatial/confidence-gated bone-interior support",
            "support_alpha_fixed_from_r282_equilibrium": self.support_alpha,
            "support_buffer_radius_px": self.support_buffer_radius,
            "support_outer_radius_px": self.support_outer_radius,
            "support_interior_radius_px": self.support_interior_radius,
            "support_confidence_ramp": [
                self.support_confidence_low, self.support_confidence_high],
            "global_support_search": False,
            "dynamic_gradient_normalization": False,
            "clean_test_used": False,
        })
        (Path(self.output_folder) / "r283_initialization.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([item.to(self.device, non_blocking=True) for item in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        high_target = target[0] if isinstance(target, list) else target
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output

            prior = data[:, 1].float()
            low = prior.amin((1, 2), keepdim=True)
            high = prior.amax((1, 2), keepdim=True)
            prior = (prior - low) / (high - low + 1e-6)
            bone = (high_target[:, 0] == 1).float()
            background = 1.0 - bone
            foreground_odds = logits[:, 1].float() - logits[:, 0].float()
            foreground_probability = torch.sigmoid(foreground_odds)

            seam_weight = prior.square() * background
            seam_loss = weighted_case_mean(F.softplus(foreground_odds), seam_weight)
            support_weight, gate = gated_support_weight(
                prior, bone, foreground_probability,
                buffer_radius=self.support_buffer_radius,
                outer_radius=self.support_outer_radius,
                interior_radius=self.support_interior_radius,
                confidence_low=self.support_confidence_low,
                confidence_high=self.support_confidence_high,
            )
            support_loss = weighted_case_mean(F.softplus(-foreground_odds), support_weight)
            loss = (base_loss + self.seam_alpha * seam_loss
                    + self.support_alpha * support_loss)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        buffer_overlap = (support_weight * gate["seam_buffer"]).mean()
        return {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "seam_loss": seam_loss.detach().cpu().numpy(),
            "support_loss": support_loss.detach().cpu().numpy(),
            "seam_mass": seam_weight.mean().detach().cpu().numpy(),
            "support_mass": support_weight.mean().detach().cpu().numpy(),
            "support_shoulder_mass": gate["shoulder"].mean().detach().cpu().numpy(),
            "support_interior_mass": gate["bone_interior"].mean().detach().cpu().numpy(),
            "support_confident_mass": gate["support_confidence"].mean().detach().cpu().numpy(),
            "support_buffer_overlap": buffer_overlap.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs: list[dict]):
        # Keep native nnU-Net logging, but write R283-specific gate dynamics.
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        nnUNetTrainer.on_train_epoch_end(self, outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r283_loss_dynamics.jsonl").open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_epoch_end(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        nnUNetTrainer.on_epoch_end(self)
        score = float(self.logger.get_value("ema_fg_dice", step=-1))
        improved = self._early_best is None or score > self._early_best + self.early_min_delta
        if improved:
            self._early_best = score
            self._early_bad = 0
        else:
            self._early_bad += 1
        if self.current_epoch >= self.early_min_epochs and self._early_bad >= self.early_patience:
            self._early_stop = True
        state = {
            "completed_epochs": int(self.current_epoch),
            "ema_dice": score,
            "early_best": self._early_best,
            "bad_epochs": self._early_bad,
            "stop": self._early_stop,
        }
        (Path(self.output_folder) / "r283_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R283 early stop", json.dumps(state))
