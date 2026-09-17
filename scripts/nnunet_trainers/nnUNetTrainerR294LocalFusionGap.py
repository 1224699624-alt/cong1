"""R294-B: locality-safe dual-scale fusion plus decoupled interface loss."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast

from nnUNetTrainerR293DualBranchGap import directional_gap_support, weighted_case_mean
from nnUNetTrainerR294LocalFusion import nnUNetTrainerR294LocalFusion


class nnUNetTrainerR294LocalFusionGap(nnUNetTrainerR294LocalFusion):
    """Route gap/support gradients through local modules, never global context."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.gap_alpha = 0.03
        self.support_alpha = 0.008
        self.prior_alpha = 0.015
        self.gap_log_odds_ceiling = math.log(0.20 / 0.80)
        self.support_log_odds_floor = math.log(0.75 / 0.25)
        np.random.seed(295)
        torch.manual_seed(295)
        torch.cuda.manual_seed_all(295)

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
            auxiliary_logits, prior_logits, auxiliary_gate = self.network.forward_aux_local(data)
            gap, support = directional_gap_support(full_target.float())
            # Only supervise target pixels that the image/prediction gate permits editing.
            editable = (auxiliary_gate.detach() > 0).float()
            gap_weight = gap * editable
            support_weight = support * editable
            log_odds = auxiliary_logits[:, 1:2].float() - auxiliary_logits[:, 0:1].float()
            gap_hinge = weighted_case_mean(
                F.relu(log_odds - self.gap_log_odds_ceiling), gap_weight)
            support_hinge = weighted_case_mean(
                F.relu(self.support_log_odds_floor - log_odds), support_weight)
            prior_bce = (weighted_case_mean(F.softplus(-prior_logits.float()), gap_weight) +
                         weighted_case_mean(F.softplus(prior_logits.float()), support_weight))
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
            "editable_gap_fraction": gap_weight.mean().detach().cpu().numpy(),
            "support_fraction": support.mean().detach().cpu().numpy(),
            "editable_support_fraction": support_weight.mean().detach().cpu().numpy(),
            "gate_mean": auxiliary_gate.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs: list[dict]):
        super().on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0]}
        row.update({
            "epoch": int(self.current_epoch),
            "global_context_weight_absmax": float(
                self.network.global_branch[-1].weight.abs().max().detach().cpu()),
            "local_residual_weight_absmax": float(
                self.network.local_residual.weight.abs().max().detach().cpu()),
            "auxiliary_global_context_detached": True,
            "max_logit_shift": self.network.max_logit_shift,
        })
        with (Path(self.output_folder) / "r294b_loss_dynamics.jsonl").open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
