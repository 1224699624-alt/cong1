"""R282: automatically balance seam and bone-support losses by logit gradient norm."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerR280BalancedInstancePrior import (
    nnUNetTrainerR280BalancedInstancePrior,
    weighted_case_mean,
)


class nnUNetTrainerR282GradNormInstancePrior(nnUNetTrainerR280BalancedInstancePrior):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.support_alpha = self.seam_alpha
        self.gradnorm_ema_beta = 0.95
        self.gradnorm_scale_min = 0.05
        self.gradnorm_scale_max = 1.0
        self._support_scale_ema: float | None = None
        np.random.seed(282)
        torch.manual_seed(282)
        torch.cuda.manual_seed_all(282)

    def initialize(self):
        super().initialize()
        parent = Path(self.output_folder) / "r280_initialization.json"
        manifest = json.loads(parent.read_text(encoding="utf-8"))
        manifest.update({
            "experiment": "R282",
            "controlled_change": "support coefficient replaced by logit-gradient normalization",
            "auxiliary_alpha": self.seam_alpha,
            "gradnorm_ema_beta": self.gradnorm_ema_beta,
            "gradnorm_scale_bounds": [self.gradnorm_scale_min, self.gradnorm_scale_max],
        })
        (Path(self.output_folder) / "r282_initialization.json").write_text(
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

            seam_weight = prior.square() * background
            seam_loss = weighted_case_mean(F.softplus(foreground_odds), seam_weight)
            kernel = self.support_radius * 2 + 1
            support_band = F.max_pool2d(prior[:, None], kernel_size=kernel, stride=1,
                                        padding=self.support_radius)[:, 0]
            support_weight = support_band.square() * bone
            support_loss = weighted_case_mean(F.softplus(-foreground_odds), support_weight)

            seam_gradient = torch.autograd.grad(
                seam_loss, logits, retain_graph=True, create_graph=False)[0]
            support_gradient = torch.autograd.grad(
                support_loss, logits, retain_graph=True, create_graph=False)[0]
            seam_grad_norm = seam_gradient.float().square().mean().sqrt().detach()
            support_grad_norm = support_gradient.float().square().mean().sqrt().detach()
            instantaneous = torch.clamp(
                seam_grad_norm / (support_grad_norm + 1e-12),
                self.gradnorm_scale_min, self.gradnorm_scale_max).item()
            if self._support_scale_ema is None:
                self._support_scale_ema = instantaneous
            else:
                self._support_scale_ema = (
                    self.gradnorm_ema_beta * self._support_scale_ema
                    + (1.0 - self.gradnorm_ema_beta) * instantaneous)
            support_scale = logits.new_tensor(self._support_scale_ema)
            loss = base_loss + self.seam_alpha * (seam_loss + support_scale * support_loss)

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
        return {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "seam_loss": seam_loss.detach().cpu().numpy(),
            "support_loss": support_loss.detach().cpu().numpy(),
            "seam_mass": seam_weight.mean().detach().cpu().numpy(),
            "support_mass": support_weight.mean().detach().cpu().numpy(),
            "seam_grad_norm": seam_grad_norm.cpu().numpy(),
            "support_grad_norm": support_grad_norm.cpu().numpy(),
            "support_scale": support_scale.detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs: list[dict]):
        nnUNetTrainer.on_train_epoch_end(self, outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r282_loss_dynamics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_epoch_end(self):
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
            "support_scale_ema": self._support_scale_ema,
            "stop": self._early_stop,
        }
        (Path(self.output_folder) / "r282_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R282 early stop", json.dumps(state))
