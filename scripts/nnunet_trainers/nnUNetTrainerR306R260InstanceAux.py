"""Strict R260 reproduction with optional instance-derived seam auxiliary loss."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch import autocast


def _map_seam_to_background(target):
    if isinstance(target, (list, tuple)):
        return [torch.where(x == 2, torch.zeros_like(x), x) for x in target]
    return torch.where(target == 2, torch.zeros_like(target), target)


class _BinaryTargetLoss(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, output, target):
        return self.loss(output, _map_seam_to_background(target))


class _R306Base(nnUNetTrainer):
    expected_dataset = ""
    use_instance_aux = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != self.expected_dataset:
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = 1e-4
        self.num_epochs = 60
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.prior_alpha = 0.05
        self.instance_alpha = 0.05 if self.use_instance_aux else 0.0
        self.early_min_epochs = 15
        self.early_patience = 10
        self.early_min_delta = 1e-4
        self._early_best = None
        self._early_bad = 0
        self._early_stop = False
        np.random.seed(260)
        torch.manual_seed(260)
        torch.cuda.manual_seed_all(260)

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R306 requires nnUNet_compile=false for strict checkpoint loading")
        super().initialize()
        path = Path(os.environ.get("R306_SOURCE_CHECKPOINT", os.environ.get("R306_ADAPTED_CHECKPOINT", "")))
        if not path.is_file():
            raise RuntimeError(f"missing R306/R307 source checkpoint: {path}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        expected_sha256 = os.environ.get("R306_EXPECTED_SOURCE_SHA256")
        if expected_sha256 and source_sha256 != expected_sha256:
            raise RuntimeError(f"source checkpoint hash mismatch: {source_sha256} != {expected_sha256}")
        source = payload["network_weights"]
        state = self.network.state_dict()
        expanded_aliases = 0
        for key, value in state.items():
            if key not in source:
                raise RuntimeError(f"missing source tensor: {key}")
            src = source[key]
            if src.shape == value.shape:
                state[key] = src
            elif (
                src.ndim == 4
                and value.ndim == 4
                and src.shape[1] == 1
                and value.shape[1] == 2
                and src.shape[0] == value.shape[0]
                and src.shape[2:] == value.shape[2:]
                and (
                    key.endswith("encoder.stages.0.0.convs.0.conv.weight")
                    or key.endswith("encoder.stages.0.0.convs.0.all_modules.0.weight")
                )
            ):
                adapted = value.new_zeros(value.shape)
                adapted[:, 0] = src[:, 0]
                state[key] = adapted
                expanded_aliases += 1
            else:
                raise RuntimeError(f"incompatible source tensor {key}: {tuple(src.shape)} -> {tuple(value.shape)}")
        self.network.load_state_dict(state, strict=True)
        first = state["encoder.stages.0.0.convs.0.conv.weight"]
        if tuple(first.shape) != (32, 2, 3, 3):
            raise RuntimeError("R306/R307 source is not a two-channel R260-compatible checkpoint")
        prior_channel_nonzero = int(torch.count_nonzero(first[:, 1]))
        if self.use_instance_aux:
            self.loss = _BinaryTargetLoss(self.loss)
        audit = {
            "checkpoint": str(path),
            "sha256": source_sha256,
            "strict_all_network_keys_loaded": True,
            "prior_channel_nonzero": prior_channel_nonzero,
            "expanded_single_channel_aliases": expanded_aliases,
            "binary_output_channels": 2,
            "prior_alpha": self.prior_alpha,
            "instance_alpha": self.instance_alpha,
            "instance_aux": self.use_instance_aux,
            "seed": 260,
        }
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        (Path(self.output_folder) / "r306_initialization.json").write_text(json.dumps(audit, indent=2))
        self.print_to_log_file("R306 initialization", json.dumps(audit))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr,
            weight_decay=self.weight_decay, momentum=0.99, nesterov=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, 60)

    @staticmethod
    def _masked_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        num = (values * weight).flatten(1).sum(1)
        den = weight.flatten(1).sum(1)
        valid = den > 1e-6
        return (num[valid] / (den[valid] + 1e-6)).mean() if valid.any() else values.sum() * 0

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [x.to(self.device, non_blocking=True) for x in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
        high = target[0] if isinstance(target, list) else target
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            base = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            prior = data[:, 1].float()
            lo = prior.amin((1, 2), keepdim=True)
            hi = prior.amax((1, 2), keepdim=True)
            prior = (prior - lo) / (hi - lo + 1e-6)
            nonbone = (high[:, 0] != 1).float()
            odds = logits[:, 1].float() - logits[:, 0].float()
            penalty = F.softplus(odds)
            prior_loss = self._masked_mean(penalty, prior.square() * nonbone)
            seam = (high[:, 0] == 2).float()
            instance_loss = self._masked_mean(penalty, seam) if self.use_instance_aux else penalty.sum() * 0
            loss = base + self.prior_alpha * prior_loss + self.instance_alpha * instance_loss
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
            "base_loss": base.detach().cpu().numpy(),
            "prior_loss": prior_loss.detach().cpu().numpy(),
            "instance_loss": instance_loss.detach().cpu().numpy(),
            "prior_mass": (prior.square() * nonbone).mean().detach().cpu().numpy(),
            "seam_mass": seam.mean().detach().cpu().numpy(),
        }

    def validation_step(self, batch):
        if not self.use_instance_aux:
            return super().validation_step(batch)
        mapped = dict(batch)
        mapped["target"] = _map_seam_to_background(batch["target"])
        return super().validation_step(mapped)

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row = {k: float(np.mean([float(np.asarray(x[k])) for x in outputs])) for k in outputs[0] if k != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r306_dynamics.jsonl").open("a") as handle:
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
        state = {
            "completed_epochs": int(self.current_epoch),
            "ema_dice": score,
            "early_best": self._early_best,
            "bad_epochs": self._early_bad,
            "stop": False,
        }
        if self.current_epoch >= self.early_min_epochs and self._early_bad >= self.early_patience:
            self._early_stop = True
            state["stop"] = True
        (Path(self.output_folder) / "r306_early_stop.json").write_text(json.dumps(state, indent=2))
        self.print_to_log_file("R306 early stop", json.dumps(state))

    def run_training(self):
        self.on_train_start()
        while self.current_epoch < self.num_epochs and not self._early_stop:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train = [self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val = [self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val)
            self.on_epoch_end()
        self.on_train_end()


class nnUNetTrainerR306BinaryPrior(_R306Base):
    expected_dataset = "Dataset307_TSRSR260BinaryPrior2D"


class nnUNetTrainerR306InstanceAuxPrior(_R306Base):
    expected_dataset = "Dataset308_TSRSR260InstanceAuxPrior2D"
    use_instance_aux = True


class nnUNetTrainerR307BinaryContinue(_R306Base):
    expected_dataset = "Dataset307_TSRSR260BinaryPrior2D"


class nnUNetTrainerR307InstanceAuxContinue(_R306Base):
    expected_dataset = "Dataset308_TSRSR260InstanceAuxPrior2D"
    use_instance_aux = True


class nnUNetTrainerR308CleanContinue(_R306Base):
    """Exact R260 continuation on the manually retained clean-panel subset."""

    expected_dataset = "Dataset309_TSRSR260CleanPanels2D"
