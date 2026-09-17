"""R349 TSRS: fixed R317 prior with parameter-space conflict-routed losses."""
from __future__ import annotations

import copy
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

from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients


class nnUNetTrainerR349ParameterPCGrad(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D":
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = float(os.environ.get("R349_TSRS_LR", "1e-5"))
        self.num_epochs = int(os.environ.get("R349_TSRS_EPOCHS", "30"))
        self.seam_weight = float(os.environ.get("R349_TSRS_SEAM_WEIGHT", ".035"))
        self.preserve_weight = float(os.environ.get("R349_TSRS_PRESERVE_WEIGHT", ".10"))
        self.max_aux_ratio = float(os.environ.get("R349_MAX_AUX_RATIO", ".15"))
        self.seam_threshold = float(os.environ.get("R349_TSRS_SEAM_THRESHOLD", ".25"))
        self.ceiling = .10
        self.audit_only = os.environ.get("R349_AUDIT_ONLY", "false").lower() == "true"
        self.teacher = None
        np.random.seed(3492); torch.manual_seed(3492); torch.cuda.manual_seed_all(3492)

    def initialize(self):
        if self.was_initialized: return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R349 requires nnUNet_compile=false for exact R317 identity")
        nnUNetTrainer.initialize(self)
        path = Path(os.environ["R349_TSRS_INIT_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        self.teacher = copy.deepcopy(self.network).eval().requires_grad_(False)
        student_state = self.network.state_dict(); teacher_state = self.teacher.state_dict()
        identity = max(float((student_state[key] - teacher_state[key]).abs().max()) for key in student_state)
        report = {"checkpoint": str(path), "strict": True, "parameter_max_abs_difference": identity,
                  "passed": identity == 0.0, "fixed_prior": "R317_continuous035",
                  "seam_threshold": self.seam_threshold}
        (Path(self.output_folder) / "identity_audit.json").write_text(json.dumps(report, indent=2))
        if not report["passed"]: raise RuntimeError(report)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start(); self.teacher.eval()

    def _forward_terms(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [value.to(self.device, non_blocking=True) for value in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
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
            seam_gate = corridor * (1 - foreground)
            probability = torch.softmax(logits.float(), dim=1)[:, 1:2]
            seam = (seam_gate * F.relu(probability - self.ceiling).square()).sum() / seam_gate.sum().clamp_min(1)
            outside = 1 - corridor
            student_distribution = torch.softmax(logits.float(), dim=1)
            teacher_distribution = torch.softmax(teacher_logits.float(), dim=1)
            keep = outside.expand_as(student_distribution)
            preserve = (keep * (student_distribution - teacher_distribution).square()).sum() / keep.sum().clamp_min(1)
        return data, logits, teacher_logits, base, seam, preserve, {
            "corridor_mass": float(corridor.mean()), "seam_mass": float(seam_gate.mean()),
            "preserve_mass": float(outside.mean()), "state_overlap": 0.0}

    def train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        _, logits, teacher_logits, base, seam, preserve, states = self._forward_terms(batch)
        ramp = min(1.0, (self.current_epoch + 1) / 5.0)
        gradients, routing = route_parameter_gradients(self.network.parameters(), base, [
            ("seam", seam, self.seam_weight * ramp),
            ("preserve", preserve, self.preserve_weight * ramp),
        ], max_aux_ratio=self.max_aux_ratio)
        identity_logits = float((logits.detach() - teacher_logits.detach()).abs().max())
        audit = {"passed": identity_logits == 0.0 and states["state_overlap"] == 0.0 and
                           all(row.cosine_after >= -1e-6 and row.applied_norm <= self.max_aux_ratio * row.base_norm + 1e-6 for row in routing),
                 "prediction_max_abs_difference": identity_logits, "states": states,
                 "losses": {"base": float(base.detach()), "seam": float(seam.detach()), "preserve": float(preserve.detach())},
                 "gradient_routing": [vars(row) for row in routing]}
        if self.audit_only:
            (Path(self.output_folder) / "smoke_audit.json").write_text(json.dumps(audit, indent=2))
            if not audit["passed"]: raise RuntimeError(audit)
        else:
            assign_gradients(self.network.parameters(), gradients)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        values = {"loss": base.detach().cpu().numpy(), "base_loss": base.detach().cpu().numpy(),
                  "seam_loss": seam.detach().cpu().numpy(), "preserve_loss": preserve.detach().cpu().numpy(),
                  "corridor_mass": np.asarray(states["corridor_mass"], dtype=np.float32),
                  "seam_conflict": np.asarray(routing[0].conflicted, dtype=np.float32),
                  "preserve_conflict": np.asarray(routing[1].conflicted, dtype=np.float32),
                  "seam_scale": np.asarray(routing[0].scale, dtype=np.float32),
                  "preserve_scale": np.asarray(routing[1].scale, dtype=np.float32)}
        return values

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row = {key: float(np.mean([float(np.asarray(value[key])) for value in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r349_dynamics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def run_training(self):
        if not self.audit_only:
            return super().run_training()
        self.on_train_start(); self.on_epoch_start(); self.on_train_epoch_start()
        self.train_step(next(self.dataloader_train))
        self.print_to_log_file("R349 audit complete")

