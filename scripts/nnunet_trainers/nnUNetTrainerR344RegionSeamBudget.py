"""R344: per-seam-region adaptive loss with a non-cancelling budget.

R343 assigned a weight per pixel but divided the loss by weighted seam mass.
That normalization could cancel the intended small weights. R344 first splits
the prior into disconnected local seam regions, estimates one bounded utility
for each region, and averages weighted region losses over the number of regions.
Thus every seam can be accepted or rejected independently and its weight remains
in the final loss magnitude.
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


class nnUNetTrainerR344RegionSeamBudget(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        expected = "Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
        if self.plans_manager.dataset_name != expected:
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = float(os.environ.get("R344_LR", "2e-5"))
        self.num_epochs = int(os.environ.get("R344_EPOCHS", "18"))
        self.region_threshold = float(os.environ.get("R344_REGION_THRESHOLD", ".20"))
        self.min_region_pixels = int(os.environ.get("R344_MIN_REGION_PIXELS", "20"))
        self.region_budget = float(os.environ.get("R344_REGION_BUDGET", ".10"))
        self.region_weight_cap = float(os.environ.get("R344_REGION_WEIGHT_CAP", ".25"))
        self.loss_budget = float(os.environ.get("R344_LOSS_BUDGET", ".02"))
        self.preserve_weight = float(os.environ.get("R344_PRESERVE", ".10"))
        self.ceiling = float(os.environ.get("R344_CEILING", ".10"))
        self.edit_strength = float(os.environ.get("R344_EDIT", ".25"))
        self.teacher = None
        np.random.seed(3441)
        torch.manual_seed(3441)
        torch.cuda.manual_seed_all(3441)

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        path = Path(os.environ["R344_INIT_CHECKPOINT"])
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        self.teacher = copy.deepcopy(self.network).eval().requires_grad_(False)
        metadata = {
            "checkpoint": str(path),
            "strict": True,
            "teacher_frozen": True,
            "weight_granularity": "connected_seam_region",
            "weighted_mass_normalization": False,
            "region_threshold": self.region_threshold,
            "min_region_pixels": self.min_region_pixels,
        }
        (Path(self.output_folder) / "r344_initialization.json").write_text(
            json.dumps(metadata, indent=2)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=.99,
            nesterov=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.teacher.eval()

    def _region_ids(self, prior: torch.Tensor) -> torch.Tensor:
        """Label disconnected prior islands independently for every image."""
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
        # torch.bincount(weights=...) has no autograd implementation. A
        # scatter-add is mathematically identical here and keeps gradients from
        # each region loss flowing into the segmentation logits.
        sums = torch.zeros(count + 1, dtype=values.dtype, device=values.device)
        sums.scatter_add_(0, ids.reshape(-1), values.reshape(-1))
        return sums[1:]

    def train_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        target = (
            [x.to(self.device, non_blocking=True) for x in target]
            if isinstance(target, list)
            else target.to(self.device, non_blocking=True)
        )
        high = target[0] if isinstance(target, list) else target
        fg = (high[:, 0] > 0).float()
        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()

        with context:
            output = self.network(data)
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            with torch.no_grad():
                teacher_output = self.teacher(data)
                teacher_logits = teacher_output[0] if isinstance(teacher_output, (list, tuple)) else teacher_output

            prior = data[:, 1].float()
            low = prior.amin((1, 2), keepdim=True)
            high_prior = prior.amax((1, 2), keepdim=True)
            prior = ((prior - low) / (high_prior - low + 1e-6)).clamp(0, 1)
            probability = torch.softmax(logits.float(), dim=1)[:, 1].clamp(1e-5, 1 - 1e-5)
            seam = prior.square() * (1 - fg)
            interaction = seam * F.relu(probability - self.ceiling).square()
            primary = -(
                fg * torch.log(probability) + (1 - fg) * torch.log1p(-probability)
            )
            uncertainty = 4 * probability * (1 - probability)

            region_ids = self._region_ids(prior)
            sample_losses = []
            region_weights = []
            region_counts = []
            active_counts = []
            for batch_index in range(prior.shape[0]):
                ids = region_ids[batch_index]
                count = int(ids.max().item())
                region_counts.append(float(count))
                if count == 0:
                    sample_losses.append(interaction[batch_index].sum() * 0.0)
                    active_counts.append(0.0)
                    continue

                with torch.no_grad():
                    mask = (ids > 0).float()
                    mass = self._region_sum(seam[batch_index], ids, count).clamp_min(1e-5)
                    prior_mean = self._region_sum(prior[batch_index] * mask, ids, count) / self._region_sum(mask, ids, count).clamp_min(1)
                    uncertainty_mean = self._region_sum(uncertainty[batch_index] * mask, ids, count) / self._region_sum(mask, ids, count).clamp_min(1)
                    background_fraction = 1 - self._region_sum(fg[batch_index] * mask, ids, count) / self._region_sum(mask, ids, count).clamp_min(1)
                    reliability = prior_mean * (.25 + .75 * uncertainty_mean) * background_fraction.square()
                    interaction_sum_detached = self._region_sum(interaction[batch_index].detach(), ids, count)
                    primary_sum = self._region_sum(primary[batch_index].detach() * mask, ids, count)
                    budget_weight = self.region_budget * primary_sum / (interaction_sum_detached + 1e-5)
                    weight = torch.minimum(reliability, budget_weight)
                    weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
                    weight = weight.clamp(0, self.region_weight_cap)
                    region_weights.append(weight)
                    active_counts.append(float((weight > .01).sum().item()))

                interaction_sum = self._region_sum(interaction[batch_index], ids, count)
                region_loss = interaction_sum / mass
                # Divide by the fixed number of valid regions, not weighted
                # seam mass. The bounded local weights therefore cannot cancel.
                sample_losses.append((weight * region_loss).sum() / max(count, 1))

            seam_loss = torch.stack(sample_losses).mean()
            global_cap = self.loss_budget * base_loss.detach().abs().clamp_min(1e-5)
            scale = torch.minimum(
                torch.ones_like(global_cap), global_cap / seam_loss.detach().clamp_min(1e-5)
            )
            seam_loss = torch.nan_to_num(seam_loss * scale, nan=0.0, posinf=0.0, neginf=0.0)

            support = 1 - F.max_pool2d((1 - fg)[:, None], 5, 1, 2)[:, 0]
            support_error = F.relu(.80 - probability).square()
            support_loss = (support * support_error).flatten(1).sum(1) / support.flatten(1).sum(1).clamp_min(1)

            teacher_probability = torch.softmax(teacher_logits.float(), dim=1)
            confidence = teacher_probability.max(1).values
            preserve_mask = (prior < .05).float() * (confidence > .90).float()
            kl = F.kl_div(
                torch.log_softmax(logits.float(), dim=1),
                teacher_probability,
                reduction="none",
            ).sum(1)
            preserve_loss = (kl * preserve_mask).flatten(1).sum(1) / preserve_mask.flatten(1).sum(1).clamp_min(1)
            loss = base_loss + seam_loss + .01 * support_loss.mean() + self.preserve_weight * preserve_loss.mean()

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

        if region_weights:
            weights = torch.cat(region_weights)
            weight_mean = weights.mean()
            weight_std = weights.std(unbiased=False)
            weight_max = weights.max()
        else:
            zero = torch.zeros((), device=self.device)
            weight_mean = weight_std = weight_max = zero
        return {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "seam_loss": seam_loss.detach().cpu().numpy(),
            "support_loss": support_loss.mean().detach().cpu().numpy(),
            "preserve_loss": preserve_loss.mean().detach().cpu().numpy(),
            "regions_per_sample": np.mean(region_counts),
            "active_regions_per_sample": np.mean(active_counts),
            "region_weight_mean": weight_mean.detach().cpu().numpy(),
            "region_weight_std": weight_std.detach().cpu().numpy(),
            "region_weight_max": weight_max.detach().cpu().numpy(),
            "seam_mass": seam.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, outputs):
        super().on_train_epoch_end(outputs)
        row = {
            key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
            for key in outputs[0]
            if key != "loss"
        }
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r344_region_dynamics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def run_training(self):
        self.on_train_start()
        while self.current_epoch < self.num_epochs:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [
                self.train_step(next(self.dataloader_train))
                for _ in range(self.num_iterations_per_epoch)
            ]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                validation_outputs = [
                    self.validation_step(next(self.dataloader_val))
                    for _ in range(self.num_val_iterations_per_epoch)
                ]
                self.on_validation_epoch_end(validation_outputs)
            self.on_epoch_end()
        self.on_train_end()
