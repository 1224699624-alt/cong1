"""R276: weighted seam CE with an explicit bone-recall retention term."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class WeightedSeamRecallLoss(nn.Module):
    def __init__(self, *, batch_dice: bool, ddp: bool, seam_weight: float = 2.0,
                 recall_weight: float = 0.20, recall_floor: float = 0.65):
        super().__init__()
        self.base = DC_and_CE_loss(
            {'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': ddp},
            {'weight': torch.tensor([0.5, 1.0, seam_weight], dtype=torch.float32)},
            weight_ce=1.0, weight_dice=1.0, ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss)
        self.recall_weight = recall_weight
        self.recall_floor = recall_floor

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.base.ce.weight is not None and self.base.ce.weight.device != logits.device:
            self.base.ce.weight = self.base.ce.weight.to(logits.device)
        loss = self.base(logits, target)
        labels = target[:, 0].long() if target.ndim == logits.ndim else target.long()
        bone = labels == 1
        if bone.any():
            p_bone = torch.softmax(logits, 1)[:, 1]
            retention = torch.relu(self.recall_floor - p_bone[bone]).mean()
            loss = loss + self.recall_weight * retention
        return loss


class nnUNetTrainerR276WeightedSeam(nnUNetTrainer):
    """Fine-tune mature R275 seam weights while favoring sparse seams safely."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 80
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30
        self.initial_lr = 1e-3

    def _build_loss(self):
        loss = WeightedSeamRecallLoss(
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
            seam_weight=2.0,
            recall_weight=0.20,
            recall_floor=0.65)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
