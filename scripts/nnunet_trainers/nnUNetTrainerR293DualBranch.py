"""R293-A: explicit low-resolution global and high-resolution local branches."""
from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def case_normalize(value: torch.Tensor) -> torch.Tensor:
    flat = value.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1).view(-1, 1, 1, 1)
    high = torch.quantile(flat, 0.95, dim=1).view(-1, 1, 1, 1)
    return ((value - low) / (high - low + 1e-6)).clamp_(0, 1)


def image_cues(xray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = F.pad((xray[:, :, :, 1:] - xray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((xray[:, :, 1:, :] - xray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    edge = case_normalize(dx + dy)
    blur = F.avg_pool2d(xray, kernel_size=9, stride=1, padding=4)
    highpass = case_normalize((xray - blur).abs())
    return edge, highpass


class R293DualBranchNetwork(nn.Module):
    """Frozen mature nnU-Net plus trainable global/local residual adapters.

    The global path sees an explicitly downsampled full patch. The local path
    operates at native resolution and exposes a separation-prior logit map.
    Both residual heads start at exact zero, so initialization is exactly R275.
    """

    def __init__(self, base: nn.Module, output_channels: int):
        super().__init__()
        self.base = base
        self.global_branch = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1, bias=True),
            nn.InstanceNorm2d(12, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(12, 12, 3, padding=1, bias=True),
            nn.InstanceNorm2d(12, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(12, output_channels, 1, bias=True),
        )
        self.local_stem = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1, bias=True),
            nn.InstanceNorm2d(12, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(12, 12, 3, padding=1, bias=True),
            nn.InstanceNorm2d(12, affine=True), nn.LeakyReLU(0.01, inplace=True),
        )
        self.local_residual = nn.Conv2d(12, output_channels, 1, bias=True)
        self.separation_prior = nn.Conv2d(12, 1, 1, bias=True)
        nn.init.zeros_(self.global_branch[-1].weight)
        nn.init.zeros_(self.global_branch[-1].bias)
        nn.init.zeros_(self.local_residual.weight)
        nn.init.zeros_(self.local_residual.bias)
        nn.init.zeros_(self.separation_prior.weight)
        nn.init.zeros_(self.separation_prior.bias)
        self.last_prior_logits: torch.Tensor | None = None
        self.last_local_gate: torch.Tensor | None = None

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def encoder(self):
        return self.base.encoder

    def compute_conv_feature_map_size(self, input_size):
        return self.base.compute_conv_feature_map_size(input_size)

    def forward(self, xray: torch.Tensor):
        with torch.no_grad():
            base_output = self.base(xray)
        outputs = list(base_output) if isinstance(base_output, (list, tuple)) else [base_output]
        final_logits = outputs[0]
        low = F.interpolate(xray, scale_factor=0.25, mode="bilinear", align_corners=False,
                            recompute_scale_factor=False)
        global_residual = self.global_branch(low)
        edge, highpass = image_cues(xray)
        local_features = self.local_stem(torch.cat((xray, edge, highpass), dim=1))
        prior_logits = self.separation_prior(local_features)
        foreground_probability = torch.softmax(final_logits.detach().float(), dim=1)[:, 1:2]
        uncertainty = 4.0 * foreground_probability * (1.0 - foreground_probability)
        uncertainty = F.interpolate(uncertainty, size=xray.shape[-2:], mode="bilinear",
                                    align_corners=False)
        local_gate = torch.sigmoid(prior_logits) * (0.15 + 0.85 * edge) * (0.25 + 0.75 * uncertainty)
        local_residual = self.local_residual(local_features) * local_gate
        self.last_prior_logits = prior_logits
        self.last_local_gate = local_gate
        fused = []
        for logits in outputs:
            size = logits.shape[-2:]
            g = F.interpolate(global_residual, size=size, mode="bilinear", align_corners=False)
            l = F.interpolate(local_residual, size=size, mode="bilinear", align_corners=False)
            fused.append(logits + g + l)
        return fused if isinstance(base_output, (list, tuple)) else fused[0]


class nnUNetTrainerR293DualBranch(nnUNetTrainer):
    """Architecture-only R293-A control; no gap/support objective."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        expected = "Dataset293_TSRS_RSNAEpiphysisDualBranch2D"
        if self.plans_manager.dataset_name != expected:
            raise RuntimeError((self.plans_manager.dataset_name, expected))
        self.initial_lr = 2e-4
        self.num_epochs = 35
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30
        self.early_min_epochs = 12
        self.early_patience = 8
        self.early_min_delta = 1e-4
        self._early_best = None
        self._early_bad = 0
        self._early_stop = False
        np.random.seed(293)
        torch.manual_seed(293)
        torch.cuda.manual_seed_all(293)

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if num_input_channels != 1:
            raise RuntimeError(f"R293 expects one deployable X-ray channel, got {num_input_channels}")
        base = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            1, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)
        return R293DualBranchNetwork(base, num_output_channels)

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R293 requires nnUNet_compile=false")
        super().initialize()
        source_value = os.environ.get("R293_SOURCE_CHECKPOINT")
        source = Path(source_value) if source_value else None
        if source is not None:
            payload = torch.load(source, map_location=self.device, weights_only=False)
            state = payload["network_weights"]
            self.network.base.load_state_dict(state, strict=True)
            for parameter in self.network.base.parameters():
                parameter.requires_grad_(False)
            if (int(torch.count_nonzero(self.network.global_branch[-1].weight)) or
                    int(torch.count_nonzero(self.network.local_residual.weight))):
                raise RuntimeError("R293 residual heads must initialize at exact zero")
        manifest = {
            "experiment": self.__class__.__name__, "source": str(source) if source else None,
            "source_sha256": (hashlib.sha256(source.read_bytes()).hexdigest()
                              if source and source.exists() else None),
            "base_frozen": True, "base_strict_loaded": bool(source),
            "global_path": "quarter-resolution full-patch context",
            "local_path": "native-resolution image/edge/high-pass interface residual",
            "inference_input": "X-ray only", "gt_roi_used": False,
            "clean_test_used": False,
        }
        (Path(self.output_folder) / "r293_initialization.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        self.print_to_log_file("R293 initialization", json.dumps(manifest))

    def configure_optimizers(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([item.to(self.device, non_blocking=True) for item in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            loss = self.loss(output, target)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.network.parameters() if p.requires_grad], 12)
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy(),
                "gate_mean": self.network.last_local_gate.mean().detach().cpu().numpy()}

    def on_train_epoch_end(self, outputs: list[dict]):
        super().on_train_epoch_end(outputs)
        row = {
            "epoch": int(self.current_epoch),
            "loss": float(np.mean([float(np.asarray(x["loss"])) for x in outputs])),
            "gate_mean": float(np.mean([float(np.asarray(x["gate_mean"])) for x in outputs])),
            "global_residual_absmax": float(self.network.global_branch[-1].weight.abs().max().detach().cpu()),
            "local_residual_absmax": float(self.network.local_residual.weight.abs().max().detach().cpu()),
        }
        with (Path(self.output_folder) / "r293_loss_dynamics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

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
        (Path(self.output_folder) / "r293_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R293 early stop", json.dumps(state))

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
