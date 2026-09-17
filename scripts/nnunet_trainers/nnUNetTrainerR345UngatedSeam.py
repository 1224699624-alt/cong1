"""R345A: controlled ungated seam continuation from the mature R317 checkpoint.

This is the fair control for R344: same R317 initialization, learning rate,
epoch budget, data and evaluation, but only the original continuous seam loss
is active. No region gate, support term or teacher preservation is applied.
"""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerR345UngatedSeam(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        expected = "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
        if self.plans_manager.dataset_name != expected:
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = float(os.environ.get("R345A_LR", "2e-5"))
        self.num_epochs = int(os.environ.get("R345A_EPOCHS", "18"))
        self.prior_alpha = float(os.environ.get("R345A_ALPHA", ".035"))
        np.random.seed(3451)
        torch.manual_seed(3451)
        torch.cuda.manual_seed_all(3451)

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        path = Path(os.environ["R345A_INIT_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        first = payload["network_weights"]["encoder.stages.0.0.convs.0.conv.weight"]
        if tuple(first.shape) != (32, 2, 3, 3):
            raise RuntimeError(first.shape)
        metadata = {
            "checkpoint": str(path),
            "strict": True,
            "prior_alpha": self.prior_alpha,
            "region_gate": False,
            "support_loss": False,
            "teacher_preservation": False,
        }
        (Path(self.output_folder) / "r345a_initialization.json").write_text(
            json.dumps(metadata, indent=2)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr,
            weight_decay=self.weight_decay, momentum=.99, nesterov=True
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([x.to(self.device, non_blocking=True) for x in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        high = target[0] if isinstance(target, list) else target
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            prior = data[:, 1].float()
            low = prior.amin((1, 2), keepdim=True)
            high_prior = prior.amax((1, 2), keepdim=True)
            prior = ((prior - low) / (high_prior - low + 1e-6)).clamp(0, 1)
            seam = prior.square() * (high[:, 0] == 0).float()
            foreground_odds = logits[:, 1].float() - logits[:, 0].float()
            numerator = (seam * F.softplus(foreground_odds)).flatten(1).sum(1)
            denominator = seam.flatten(1).sum(1)
            valid = denominator > 1e-6
            seam_loss = ((numerator[valid] / (denominator[valid] + 1e-6)).mean()
                         if valid.any() else logits.sum() * 0)
            loss = base_loss + self.prior_alpha * seam_loss
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
            "weighted_seam_loss": (self.prior_alpha * seam_loss).detach().cpu().numpy(),
            "seam_mass": seam.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r345a_ungated_dynamics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def run_training(self):
        self.on_train_start()
        while self.current_epoch < self.num_epochs:
            self.on_epoch_start(); self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train))
                             for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                validation_outputs = [self.validation_step(next(self.dataloader_val))
                                      for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(validation_outputs)
            self.on_epoch_end()
        self.on_train_end()
