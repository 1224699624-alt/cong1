"""R260-faithful joint input/loss training with a frozen instance-derived seam prior."""
from __future__ import annotations

import hashlib
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


class nnUNetTrainerR279InstancePrior(nnUNetTrainer):
    """Change only the R260 prior source: frozen instance-seam probability maps."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        expected = "Dataset279_TSRS_RSNAEpiphysisInstancePrior2D"
        if self.plans_manager.dataset_name != expected:
            raise RuntimeError((self.plans_manager.dataset_name, expected))
        self.initial_lr = 1e-4
        self.num_epochs = 60
        self.prior_alpha = 0.05
        self.early_min_epochs = 15
        self.early_patience = 10
        self.early_min_delta = 1e-4
        self._early_best = None
        self._early_bad = 0
        self._early_stop = False
        np.random.seed(279)
        torch.manual_seed(279)
        torch.cuda.manual_seed_all(279)

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R279 requires nnUNet_compile=false for strict mature loading")
        super().initialize()
        path = Path(os.environ["R279_ADAPTED_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload["network_weights"]
        self.network.load_state_dict(state, strict=True)
        first = state["encoder.stages.0.0.convs.0.conv.weight"]
        if tuple(first.shape) != (32, 2, 3, 3) or int(torch.count_nonzero(first[:, 1])) != 0:
            raise RuntimeError("Prior input channel was not initialized to exact zero")
        heads = [key for key in state if "seg_layers" in key]
        if not heads:
            raise RuntimeError("No inherited segmentation heads")
        manifest = {
            "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "strict_all_network_keys_loaded": True,
            "xray_channel_inherited": True,
            "prior_channel_nonzero": 0,
            "segmentation_head_tensors_loaded": len(heads),
            "clean_test_used": False,
        }
        (Path(self.output_folder) / "r279_initialization.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        self.print_to_log_file("R279 mature initialization", json.dumps(manifest))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

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
            weight = prior.square() * (high_target[:, 0] == 0).float()
            foreground_odds = logits[:, 1].float() - logits[:, 0].float()
            numerator = (weight * F.softplus(foreground_odds)).flatten(1).sum(1)
            denominator = weight.flatten(1).sum(1)
            valid = denominator > 1e-6
            prior_loss = ((numerator[valid] / (denominator[valid] + 1e-6)).mean()
                          if valid.any() else logits.sum() * 0)
            loss = base_loss + self.prior_alpha * prior_loss
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
            "prior_loss": prior_loss.detach().cpu().numpy(),
            "prior_mass": weight.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs: list[dict]):
        super().on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r279_prior_dynamics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_epoch_end(self):
        super().on_epoch_end()
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
        (Path(self.output_folder) / "r279_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R279 early stop", json.dumps(state))

    def run_training(self):
        self.on_train_start()
        while self.current_epoch < self.num_epochs and not self._early_stop:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train))
                             for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [self.validation_step(next(self.dataloader_val))
                               for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()
