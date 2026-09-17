"""R345B: strong seam prior with inference-available conflict-only gating.

R345A showed that an ungated seam loss improves merge/component metrics while
R344's broad region budget better protects boundary metrics.  R345B keeps the
strong seam loss by default and attenuates it only for seam regions where the
frozen R317 teacher still predicts foreground (an inference-available conflict
signal).  Outside the seam ROI, a stronger teacher consistency term prevents
remote false-positive islands such as the R344 1518 failure.
"""
from __future__ import annotations

import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import autocast

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerR345BConflictSeam(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D":
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = float(os.environ.get("R345B_LR", "2e-5"))
        self.num_epochs = int(os.environ.get("R345B_EPOCHS", "18"))
        self.alpha = float(os.environ.get("R345B_ALPHA", ".035"))
        self.region_threshold = float(os.environ.get("R345B_REGION_THRESHOLD", ".20"))
        self.min_region_pixels = int(os.environ.get("R345B_MIN_REGION_PIXELS", "20"))
        self.conflict_center = float(os.environ.get("R345B_CONFLICT_CENTER", ".45"))
        self.conflict_temperature = float(os.environ.get("R345B_CONFLICT_TEMP", ".10"))
        self.preserve_weight = float(os.environ.get("R345B_PRESERVE", ".20"))
        np.random.seed(3452)
        torch.manual_seed(3452)
        torch.cuda.manual_seed_all(3452)

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        path = Path(os.environ["R345B_INIT_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        self.teacher = copy.deepcopy(self.network).eval().requires_grad_(False)
        (Path(self.output_folder) / "r345b_initialization.json").write_text(
            json.dumps({
                "checkpoint": str(path),
                "strict": True,
                "teacher_frozen": True,
                "gate": "teacher_foreground_conflict_only",
                "inference_gt_free": True,
                "outside_roi_preserve_weight": self.preserve_weight,
            }, indent=2)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr,
            weight_decay=self.weight_decay, momentum=.99, nesterov=True
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.teacher.eval()

    def _region_ids(self, prior: torch.Tensor) -> torch.Tensor:
        labels = []
        structure = np.ones((3, 3), dtype=np.uint8)
        for sample in prior.detach().float().cpu().numpy():
            lab, count = ndimage.label(sample >= self.region_threshold, structure=structure)
            if count:
                sizes = np.bincount(lab.reshape(-1))
                keep = sizes >= self.min_region_pixels
                keep[0] = False
                lab = np.where(keep[lab], lab, 0)
                unique = np.unique(lab)
                remap = np.zeros(int(lab.max()) + 1, dtype=np.int32)
                if unique.size > 1:
                    remap[unique[1:]] = np.arange(1, unique.size, dtype=np.int32)
                lab = remap[lab]
            labels.append(torch.from_numpy(lab.astype(np.int64, copy=False)))
        return torch.stack(labels).to(prior.device, non_blocking=True)

    @staticmethod
    def _region_sum(values: torch.Tensor, ids: torch.Tensor, count: int) -> torch.Tensor:
        sums = torch.zeros(count + 1, dtype=values.dtype, device=values.device)
        sums.scatter_add_(0, ids.reshape(-1), values.reshape(-1))
        return sums[1:]

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = ([x.to(self.device, non_blocking=True) for x in target]
                  if isinstance(target, list) else target.to(self.device, non_blocking=True))
        high = target[0] if isinstance(target, list) else target
        fg = (high[:, 0] > 0).float()
        self.optimizer.zero_grad(set_to_none=True)
        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with ctx:
            output = self.network(data)
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            with torch.no_grad():
                teacher_output = self.teacher(data)
                teacher_logits = teacher_output[0] if isinstance(teacher_output, (list, tuple)) else teacher_output
                teacher_probability = torch.softmax(teacher_logits.float(), dim=1)

            prior = data[:, 1].float()
            lo = prior.amin((1, 2), keepdim=True)
            hi = prior.amax((1, 2), keepdim=True)
            prior = ((prior - lo) / (hi - lo + 1e-6)).clamp(0, 1)
            logits_float = logits.float()
            probability = torch.softmax(logits_float, dim=1)[:, 1].clamp(1e-5, 1 - 1e-5)
            seam = prior.square() * (1 - fg)
            # Keep the structural-loss scale identical to R345A. Applying
            # softplus to probabilities would inflate the term by an order of
            # magnitude and invalidate the controlled comparison.
            foreground_odds = logits_float[:, 1] - logits_float[:, 0]
            interaction = seam * F.softplus(foreground_odds)
            region_ids = self._region_ids(prior)
            sample_losses, gate_values, conflict_values, region_counts = [], [], [], []

            for b in range(prior.shape[0]):
                ids = region_ids[b]
                count = int(ids.max().item())
                region_counts.append(float(count))
                if count == 0:
                    sample_losses.append(logits[b].sum() * 0)
                    continue
                with torch.no_grad():
                    mask = (ids > 0).float()
                    mass = self._region_sum(seam[b], ids, count).clamp_min(1e-5)
                    region_area = self._region_sum(mask, ids, count).clamp_min(1)
                    teacher_fg = teacher_probability[b, 1]
                    teacher_fg_mean = self._region_sum(teacher_fg * mask, ids, count) / region_area
                    # This signal is available at inference: frozen teacher
                    # foreground evidence inside a candidate seam region.
                    conflict = torch.sigmoid(
                        (teacher_fg_mean - self.conflict_center) / self.conflict_temperature
                    )
                    # Strong seam action is the default; only conflict regions
                    # are attenuated. The floor avoids turning the gate into a
                    # global suppression mechanism.
                    gate = (1.0 - .75 * conflict).clamp(.25, 1.0)
                    gate_values.append(gate)
                    conflict_values.append(conflict)
                region_loss = self._region_sum(interaction[b], ids, count) / mass
                sample_losses.append((gate * region_loss).sum() / max(count, 1))

            seam_loss = torch.stack(sample_losses).mean() * self.alpha
            seam_loss = torch.nan_to_num(seam_loss, nan=0.0, posinf=1.0, neginf=0.0)

            # Strong consistency outside the candidate seam ROI. This is the
            # explicit protection against remote islands, not a GT-derived cut.
            teacher_conf = teacher_probability.max(1).values
            outside = (prior < .10).float() * (teacher_conf > .90).float()
            kl = F.kl_div(torch.log_softmax(logits.float(), dim=1), teacher_probability,
                          reduction="none").sum(1)
            preserve = (kl * outside).flatten(1).sum(1) / outside.flatten(1).sum(1).clamp_min(1)
            support = 1 - F.max_pool2d((1 - fg)[:, None], 5, 1, 2)[:, 0]
            support_error = F.relu(.80 - probability).square()
            support_loss = (support * support_error).flatten(1).sum(1) / support.flatten(1).sum(1).clamp_min(1)
            loss = base_loss + seam_loss + .01 * support_loss.mean() + self.preserve_weight * preserve.mean()

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

        zero = torch.zeros((), device=self.device)
        gates = torch.cat(gate_values) if gate_values else zero[None]
        conflicts = torch.cat(conflict_values) if conflict_values else zero[None]
        return {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "seam_loss": seam_loss.detach().cpu().numpy(),
            "preserve_loss": preserve.mean().detach().cpu().numpy(),
            "support_loss": support_loss.mean().detach().cpu().numpy(),
            "regions_per_sample": np.mean(region_counts) if region_counts else 0.0,
            "gate_mean": gates.mean().detach().cpu().numpy(),
            "gate_std": gates.std(unbiased=False).detach().cpu().numpy(),
            "conflict_mean": conflicts.mean().detach().cpu().numpy(),
            "seam_mass": seam.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row = {k: float(np.mean([float(np.asarray(x[k])) for x in outputs]))
               for k in outputs[0] if k != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r345b_conflict_dynamics.jsonl").open("a") as handle:
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
