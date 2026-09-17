"""R294-A: global context conditions a bounded, sparse local correction only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

from nnUNetTrainerR293DualBranch import (
    R293DualBranchNetwork,
    image_cues,
    nnUNetTrainerR293DualBranch,
)


class R294LocalFusionNetwork(R293DualBranchNetwork):
    """Use low-resolution features only inside a sparse local residual path."""

    def __init__(self, base: nn.Module, output_channels: int):
        super().__init__(base, output_channels)
        self.global_branch = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=True),
            nn.InstanceNorm2d(8, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, bias=True),
            nn.InstanceNorm2d(8, affine=True), nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(8, 8, 1, bias=True),
        )
        self.local_stem = nn.Sequential(
            nn.Conv2d(11, 12, 3, padding=1, bias=True),
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
        self.max_logit_shift = 0.75
        self.last_base_logits: torch.Tensor | None = None
        self.last_candidate_gate: torch.Tensor | None = None

    @staticmethod
    def candidate_gate(edge: torch.Tensor, foreground_probability: torch.Tensor) -> torch.Tensor:
        uncertainty = 4.0 * foreground_probability * (1.0 - foreground_probability)
        edge_gate = ((edge - 0.15) / 0.35).clamp(0, 1)
        uncertainty_gate = ((uncertainty - 0.20) / 0.50).clamp(0, 1)
        return F.max_pool2d(edge_gate * uncertainty_gate, 5, stride=1, padding=2)

    def local_path(self, xray: torch.Tensor, base_logits: torch.Tensor,
                   detach_global: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low = F.interpolate(xray, scale_factor=0.25, mode="bilinear", align_corners=False,
                            recompute_scale_factor=False)
        global_features = self.global_branch(low)
        global_features = F.interpolate(global_features, size=xray.shape[-2:], mode="bilinear",
                                        align_corners=False)
        if detach_global:
            global_features = global_features.detach()
        edge, highpass = image_cues(xray)
        local_features = self.local_stem(torch.cat(
            (xray, edge, highpass, global_features), dim=1))
        prior_logits = self.separation_prior(local_features)
        foreground_probability = torch.softmax(base_logits.detach().float(), dim=1)[:, 1:2]
        candidate = self.candidate_gate(edge, foreground_probability)
        gate = candidate * torch.sigmoid(prior_logits)
        correction = self.max_logit_shift * torch.tanh(self.local_residual(local_features)) * gate
        return correction, prior_logits, gate

    def forward(self, xray: torch.Tensor):
        with torch.no_grad():
            base_output = self.base(xray)
        outputs = list(base_output) if isinstance(base_output, (list, tuple)) else [base_output]
        base_logits = outputs[0]
        correction, prior_logits, gate = self.local_path(xray, base_logits, detach_global=False)
        self.last_base_logits = base_logits.detach()
        self.last_prior_logits = prior_logits
        self.last_local_gate = gate
        self.last_candidate_gate = (gate > 0).float()
        fused = []
        for logits in outputs:
            local = F.interpolate(correction, size=logits.shape[-2:], mode="bilinear",
                                  align_corners=False)
            fused.append(logits + local)
        return fused if isinstance(base_output, (list, tuple)) else fused[0]

    def forward_aux_local(self, xray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Auxiliary route blocks gap/support gradients into global context."""
        if self.last_base_logits is None:
            raise RuntimeError("Main forward must run before forward_aux_local")
        correction, prior_logits, gate = self.local_path(
            xray, self.last_base_logits, detach_global=True)
        return self.last_base_logits + correction, prior_logits, gate


class nnUNetTrainerR294LocalFusion(nnUNetTrainerR293DualBranch):
    """Architecture control for locality-safe global-to-local fusion."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 35
        self.early_min_epochs = 12
        self.early_patience = 8
        np.random.seed(294)
        torch.manual_seed(294)
        torch.cuda.manual_seed_all(294)

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager,
                                   num_input_channels: int, num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        if num_input_channels != 1:
            raise RuntimeError(f"R294 expects one X-ray channel, got {num_input_channels}")
        base = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            1, num_output_channels, allow_init=True,
            deep_supervision=enable_deep_supervision)
        return R294LocalFusionNetwork(base, num_output_channels)

    def on_train_epoch_end(self, outputs: list[dict]):
        # Bypass R293's global-logit-residual naming and retain native logging.
        super(nnUNetTrainerR293DualBranch, self).on_train_epoch_end(outputs)
        row = {
            "epoch": int(self.current_epoch),
            "loss": float(np.mean([float(np.asarray(x["loss"])) for x in outputs])),
            "gate_mean": float(np.mean([float(np.asarray(x["gate_mean"])) for x in outputs])),
            "global_context_weight_absmax": float(
                self.network.global_branch[-1].weight.abs().max().detach().cpu()),
            "local_residual_weight_absmax": float(
                self.network.local_residual.weight.abs().max().detach().cpu()),
            "max_logit_shift": self.network.max_logit_shift,
        }
        with (Path(self.output_folder) / "r294_loss_dynamics.jsonl").open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
