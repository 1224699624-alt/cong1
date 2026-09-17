"""R291-A: train-only local-interface prior with absolute gap/support hinges."""
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


def normalize_aux(value: torch.Tensor) -> torch.Tensor:
    low = value.amin((1, 2), keepdim=True)
    high = value.amax((1, 2), keepdim=True)
    return ((value - low) / (high - low + 1e-6)).clamp(0, 1)


def weighted_case_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    numerator = (value * weight).flatten(1).sum(1)
    denominator = weight.flatten(1).sum(1)
    valid = denominator > 1e-6
    return ((numerator[valid] / (denominator[valid] + 1e-6)).mean()
            if valid.any() else value.sum() * 0)


class nnUNetTrainerR291AInterfaceHinge(nnUNetTrainer):
    """Keep mature binary nnU-Net deployable while regularizing local interfaces."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        expected = "Dataset291_TSRS_RSNAEpiphysisInterfaceAux2D"
        if self.plans_manager.dataset_name != expected:
            raise RuntimeError((self.plans_manager.dataset_name, expected))
        self.initial_lr = 1e-4
        self.num_epochs = 60
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30
        self.gap_alpha = 0.02
        self.support_alpha = 0.01
        self.gap_probability_ceiling = 0.20
        self.support_probability_floor = 0.75
        self.early_min_epochs = 20
        self.early_patience = 12
        self.early_min_delta = 1e-4
        self._early_best = None
        self._early_bad = 0
        self._early_stop = False
        np.random.seed(291)
        torch.manual_seed(291)
        torch.cuda.manual_seed_all(291)

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R291-A requires nnUNet_compile=false")
        super().initialize()
        path = Path(os.environ["R291_ADAPTED_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload["network_weights"]
        self.network.load_state_dict(state, strict=True)
        first = state["encoder.stages.0.0.convs.0.conv.weight"]
        if tuple(first.shape) != (32, 4, 3, 3) or int(torch.count_nonzero(first[:, 1:])):
            raise RuntimeError("Auxiliary input weights must start at exact zero")
        manifest = {
            "experiment": "R291-A", "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "strict_all_network_keys_loaded": True,
            "deployable_network_input": "channel 0 X-ray only",
            "auxiliary_channels_zeroed_before_forward": True,
            "gap_alpha": self.gap_alpha, "support_alpha": self.support_alpha,
            "gap_probability_ceiling": self.gap_probability_ceiling,
            "support_probability_floor": self.support_probability_floor,
            "ambiguous_background_ignored_by_base_loss": True,
            "clean_test_used": False, "age_or_sex_used": False,
            "fixed_pair_list_used": False, "relation_head_used": False,
        }
        (Path(self.output_folder) / "r291a_initialization.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        self.print_to_log_file("R291-A initialization", json.dumps(manifest))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    @staticmethod
    def xray_only(data: torch.Tensor) -> torch.Tensor:
        deployable = data.clone()
        deployable[:, 1:] = 0
        return deployable

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([item.to(self.device, non_blocking=True) for item in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(self.xray_only(data))
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
            gap_weight = normalize_aux(data[:, 1].float()).square()
            support_weight = normalize_aux(data[:, 2].float()).square()
            gap_loss = weighted_case_mean(
                F.relu(probability - self.gap_probability_ceiling), gap_weight)
            support_loss = weighted_case_mean(
                F.relu(self.support_probability_floor - probability), support_weight)
            loss = base_loss + self.gap_alpha * gap_loss + self.support_alpha * support_loss

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
            "gap_hinge": gap_loss.detach().cpu().numpy(),
            "support_hinge": support_loss.detach().cpu().numpy(),
            "gap_mass": gap_weight.mean().detach().cpu().numpy(),
            "support_mass": support_weight.mean().detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        clean = dict(batch)
        clean["data"] = self.xray_only(batch["data"])
        return super().validation_step(clean)

    def on_train_epoch_end(self, outputs: list[dict]):
        super().on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        first = self.network.state_dict()["encoder.stages.0.0.convs.0.conv.weight"]
        row["aux_input_weight_absmax"] = float(first[:, 1:].abs().max().detach().cpu())
        with (Path(self.output_folder) / "r291a_loss_dynamics.jsonl").open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_epoch_end(self):
        super().on_epoch_end()
        score = float(self.logger.get_value("ema_fg_dice", step=-1))
        improved = self._early_best is None or score > self._early_best + self.early_min_delta
        if improved:
            self._early_best, self._early_bad = score, 0
        else:
            self._early_bad += 1
        if self.current_epoch >= self.early_min_epochs and self._early_bad >= self.early_patience:
            self._early_stop = True
        state = {"completed_epochs": int(self.current_epoch), "ema_dice": score,
                 "early_best": self._early_best, "bad_epochs": self._early_bad,
                 "stop": self._early_stop}
        (Path(self.output_folder) / "r291a_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R291-A early stop", json.dumps(state))

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
