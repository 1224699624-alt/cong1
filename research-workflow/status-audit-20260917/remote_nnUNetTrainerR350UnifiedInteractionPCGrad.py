"""R350 TSRS: R317 initialization with unified separation/completion PCGrad."""
from __future__ import annotations

import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import autocast
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients
from r350_unified_interaction_loss import (
    completion_loss,
    distribution_preserve_loss,
    preserve_state,
    separation_loss,
    state_weights,
)


class nnUNetTrainerR350UnifiedInteractionPCGrad(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D":
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = float(os.environ.get("R350_TSRS_LR", "1e-5"))
        self.num_epochs = int(os.environ.get("R350_TSRS_EPOCHS", "30"))
        self.structural_weight = float(os.environ.get("R350_STRUCTURAL_WEIGHT", ".035"))
        self.preserve_weight = float(os.environ.get("R350_PRESERVE_WEIGHT", ".10"))
        self.max_aux_ratio = float(os.environ.get("R350_MAX_AUX_RATIO", ".15"))
        self.seam_threshold = float(os.environ.get("R350_SEAM_THRESHOLD", ".25"))
        self.overlap_confidence = float(os.environ.get("R350_TSRS_OVERLAP_CONFIDENCE", ".75"))
        self.reference_mass = float(os.environ.get("R350_REFERENCE_MASS", ".01"))
        self.ceiling = .10
        self.audit_only = os.environ.get("R350_AUDIT_ONLY", "false").lower() == "true"
        self.teacher = None
        np.random.seed(3502); torch.manual_seed(3502); torch.cuda.manual_seed_all(3502)

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R350 requires nnUNet_compile=false")
        nnUNetTrainer.initialize(self)
        path = Path(os.environ["R350_TSRS_INIT_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        self.teacher = copy.deepcopy(self.network).eval().requires_grad_(False)
        identity = max(float((self.network.state_dict()[key] - self.teacher.state_dict()[key]).abs().max())
                       for key in self.network.state_dict())
        report = {
            "checkpoint": str(path), "strict": True,
            "parameter_max_abs_difference": identity, "passed": identity == 0.0,
            "fixed_seam_prior": "R317_continuous035",
            "overlap_state": "seam_corridor_and_gt_foreground_and_frozen_R317_confidence",
        }
        (Path(self.output_folder) / "identity_audit.json").write_text(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError(report)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start(); self.teacher.eval()

    def _forward_terms(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([value.to(self.device, non_blocking=True) for value in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        high = target[0] if isinstance(target, list) else target
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            base = self.loss(output, target)
            with torch.no_grad():
                teacher_output = self.teacher(data)
                teacher_logits = teacher_output[0] if isinstance(teacher_output, (list, tuple)) else teacher_output

            prior = data[:, 1:2].float()
            low = prior.amin((-2, -1), keepdim=True); high_prior = prior.amax((-2, -1), keepdim=True)
            normalized = ((prior - low) / (high_prior - low + 1e-6)).clamp(0, 1)
            corridor = (normalized >= self.seam_threshold).float()
            foreground = (high[:, :1] > 0).float()
            teacher_distribution = torch.softmax(teacher_logits.float(), dim=1)
            teacher_foreground = teacher_distribution[:, 1:2]

            separation = corridor * (1 - foreground)
            overlap = corridor * foreground * (teacher_foreground >= self.overlap_confidence).float()
            preserve = preserve_state(separation, overlap)

            student_distribution = torch.softmax(logits.float(), dim=1)
            student_foreground = student_distribution[:, 1:2]
            foreground_margin = logits.float()[:, 1:2] - logits.float()[:, 0:1]
            seam = separation_loss(student_foreground, separation, self.ceiling)
            completion = completion_loss(foreground_margin, overlap)
            keep = preserve.expand(-1, student_distribution.shape[1], -1, -1)
            stable = distribution_preserve_loss(student_distribution, teacher_distribution, keep)
            weights = state_weights(separation, overlap, preserve, self.structural_weight,
                                    self.preserve_weight, self.reference_mass)

        states = {
            "separation_mass": float(separation.mean()),
            "overlap_mass": float(overlap.mean()),
            "preserve_mass": float(preserve.mean()),
            "gate_conflict": float(((separation > 0) & (overlap > 0)).float().mean()),
        }
        return logits, teacher_logits, base, seam, completion, stable, weights, states

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        logits, teacher_logits, base, seam, overlap, preserve, weights, states = self._forward_terms(batch)
        ramp = min(1.0, (self.current_epoch + 1) / 5.0)
        gradients, routing = route_parameter_gradients(self.network.parameters(), base, [
            ("separation", seam, weights.separation * ramp),
            ("overlap", overlap, weights.overlap * ramp),
            ("preserve", preserve, weights.preserve * ramp),
        ], max_aux_ratio=self.max_aux_ratio)
        identity_logits = float((logits.detach() - teacher_logits.detach()).abs().max())
        audit = {
            "passed": identity_logits == 0.0 and states["gate_conflict"] == 0.0
                      and states["separation_mass"] > 0.0 and states["overlap_mass"] > 0.0
                      and all(row.cosine_after >= -1e-3
                              and row.applied_norm <= self.max_aux_ratio * row.base_norm + 1e-6
                              for row in routing),
            "prediction_max_abs_difference": identity_logits,
            "states": states,
            "effective_weights": vars(weights),
            "losses": {"base": float(base.detach()), "separation": float(seam.detach()),
                       "overlap": float(overlap.detach()), "preserve": float(preserve.detach())},
            "gradient_routing": [vars(row) for row in routing],
        }
        if self.audit_only:
            (Path(self.output_folder) / "smoke_audit.json").write_text(json.dumps(audit, indent=2))
            if not audit["passed"]:
                raise RuntimeError(audit)
        else:
            assign_gradients(self.network.parameters(), gradients)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        values = {
            "loss": base.detach().cpu().numpy(), "base_loss": base.detach().cpu().numpy(),
            "separation_loss": seam.detach().cpu().numpy(), "overlap_loss": overlap.detach().cpu().numpy(),
            "preserve_loss": preserve.detach().cpu().numpy(),
            **{key: np.asarray(value, dtype=np.float32) for key, value in states.items()},
            "separation_weight": np.asarray(weights.separation, dtype=np.float32),
            "overlap_weight": np.asarray(weights.overlap, dtype=np.float32),
            "preserve_weight": np.asarray(weights.preserve, dtype=np.float32),
            "separation_conflict": np.asarray(routing[0].conflicted, dtype=np.float32),
            "overlap_conflict": np.asarray(routing[1].conflicted, dtype=np.float32),
            "preserve_conflict": np.asarray(routing[2].conflicted, dtype=np.float32),
        }
        return values

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(value[key])) for value in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r350_dynamics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def run_training(self):
        if not self.audit_only:
            return super().run_training()
        self.on_train_start(); self.on_epoch_start(); self.on_train_epoch_start()
        selected = None
        observed = []
        for index, batch in enumerate(self.dataloader_train):
            terms = self._forward_terms(batch)
            observed.append(terms[-1])
            if terms[-1]["separation_mass"] > 0 and terms[-1]["overlap_mass"] > 0:
                selected = batch
                break
            if index >= 7:
                break
        if selected is None:
            raise RuntimeError({"reason": "no audit batch covered both interaction states",
                                "observed": observed})
        self.train_step(selected)
        self.print_to_log_file("R350 TSRS audit complete")
