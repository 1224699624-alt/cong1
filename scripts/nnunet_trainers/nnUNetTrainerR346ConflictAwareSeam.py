"""R346: R317 seam prior with mutually-exclusive conflict-aware routing."""
from __future__ import annotations
import copy, json, os
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import autocast
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .nnUNetTrainerR260MaturePrior import nnUNetTrainerR260MaturePrior
from conflict_aware_loss import conflict_aware_loss


class nnUNetTrainerR346ConflictAwareSeam(nnUNetTrainerR260MaturePrior):
    """Only the loss is new; R317/R260 network and prior input stay fixed."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = float(os.environ.get("R346_LR", "2e-5"))
        self.num_epochs = int(os.environ.get("R346_EPOCHS", "18"))
        self.seam_weight = float(os.environ.get("R346_SEAM_WEIGHT", ".035"))
        self.preserve_weight = float(os.environ.get("R346_PRESERVE_WEIGHT", ".20"))
        self.seam_threshold = float(os.environ.get("R346_SEAM_THRESHOLD", ".25"))
        # Full scheduled run is required for this comparison; do not inherit
        # R260's validation early-stop rule.
        self.early_min_epochs = 10**9
        self.early_patience = 10**9
        self._early_stop = False
        self.teacher = None
        np.random.seed(3460); torch.manual_seed(3460); torch.cuda.manual_seed_all(3460)

    def initialize(self):
        if self.was_initialized: return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R346 requires nnUNet_compile=false for strict R317 loading")
        nnUNetTrainer.initialize(self)
        checkpoint = Path(os.environ["R346_INIT_CHECKPOINT"])
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        self.teacher = copy.deepcopy(self.network).eval().requires_grad_(False)
        (Path(self.output_folder) / "r346_initialization.json").write_text(json.dumps({
            "checkpoint": str(checkpoint), "fixed_prior": "R317_continuous035", "teacher_frozen": True,
            "state_routing": "explicit_seam_corridor/outside_full_distribution_preserve",
            "seam_threshold": self.seam_threshold, "overlap_state": False}, indent=2))

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                              weight_decay=self.weight_decay, momentum=.99, nesterov=True)
        return opt, PolyLRScheduler(opt, self.initial_lr, self.num_epochs)

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = [x.to(self.device, non_blocking=True) for x in target] if isinstance(target, list) else target.to(self.device, non_blocking=True)
        high = target[0] if isinstance(target, list) else target
        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data); logits = output[0] if isinstance(output, (list, tuple)) else output
            base = self.loss(output, target)
            with torch.no_grad():
                teacher_out = self.teacher(data); teacher_logits = teacher_out[0] if isinstance(teacher_out, (list, tuple)) else teacher_out
            terms = conflict_aware_loss(logits, high, teacher_logits, data[:, 1], base,
                                        mode="tsrs", seam_weight=self.seam_weight,
                                        preserve_weight=self.preserve_weight,
                                        seam_threshold=self.seam_threshold)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(terms.total).backward(); self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer); self.grad_scaler.update()
        else:
            terms.total.backward(); torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12); self.optimizer.step()
        return {"loss": terms.total.detach().cpu().numpy(), "base_loss": terms.base.detach().cpu().numpy(),
                "seam_loss": terms.seam.detach().cpu().numpy(), "overlap_loss": terms.overlap.detach().cpu().numpy(),
                "preserve_loss": terms.preserve.detach().cpu().numpy(), "seam_gate_mass": terms.seam_gate_mass.detach().cpu().numpy(),
                "overlap_gate_mass": terms.overlap_gate_mass.detach().cpu().numpy(), "uncertain_mass": terms.uncertain_mass.detach().cpu().numpy()}
