"""R267: relation-agnostic adjacent-structure separation loss.

The input prior channels are organized as repeated generic pair records:
``[gap_0, support_0, gap_1, support_1, ...]``.  A pair means only that two
candidate structures are locally adjacent; it is not assigned an anatomical
label such as CC, CM, or CR.  The same loss can therefore be reused for
metacarpals, carpal bones, teeth, or another instance-segmentation dataset.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerR261ConditionalSeam import (
    nnUNetTrainerR261ConditionalSeam,
)


class nnUNetTrainerR267AdjacentStructure(nnUNetTrainerR261ConditionalSeam):
    """Generic adjacent-pair prior trainer; no relation-type branches."""

    run_prefix = "r267"
    expected_dataset = "Dataset206_TSRS_RSNAEpiphysisHierarchicalCarpal2D"
    checkpoint_env = "R267_ADAPTED_CHECKPOINT"
    pair_slots = 6

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.gradient_ratio = 1.0
        self.alpha_max = 0.10
        self.warmup_epochs = 5
        self.seam_temperature = 0.06
        self.support_temperature = 0.08
        self.support_keep_weight = 0.30
        self.base_gap_upper = 0.10
        self.uncertain_gap_relaxation = 0.12
        self.support_floor = 0.70

    def _relation_loss(self, logits, target, data):
        """Apply the same local separation rule to every valid adjacent pair.

        The confidence-dependent upper bound is deliberately generic:
        uncertain pair proposals receive a slightly wider bound instead of
        being forced into a false anatomical category.  This is useful for
        both immature metacarpal ossification and variable dental eruption.
        """
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
        y = (target[:, 0] > 0).float()
        channels = data.shape[1] - 1
        slots = min(self.pair_slots, channels // 2)
        priors = torch.clamp(data[:, 1:].float() / 255.0, 0.0, 1.0)
        losses, gaps, supports, confidences, excesses = [], [], [], [], []
        valid_per_sample = torch.zeros(len(data), device=logits.device)

        for b in range(len(data)):
            for slot in range(slots):
                seam = priors[b, 2 * slot]
                support = priors[b, 2 * slot + 1]
                confidence = torch.minimum(seam.max(), support.max())
                gap_w = seam * (1.0 - y[b])
                support_w = support * y[b]
                gap_mass = gap_w.sum()
                support_mass = support_w.sum()
                if (float(confidence.detach()) < 0.02 or
                        float(gap_mass.detach()) < 2.0 or
                        float(support_mass.detach()) < 4.0):
                    continue

                pg = (gap_w * probability[b]).sum() / (gap_mass + 1e-6)
                ps = (support_w * probability[b]).sum() / (support_mass + 1e-6)
                # Confidence is the only pair-specific relaxation variable;
                # no anatomy-specific CC/CR/CM constants are used.
                tau = self.base_gap_upper + self.uncertain_gap_relaxation * (1.0 - confidence)
                gap_penalty = F.softplus((pg - tau) / self.seam_temperature)
                support_penalty = F.softplus((self.support_floor - ps) / self.support_temperature)
                losses.append(confidence * (gap_penalty + self.support_keep_weight * support_penalty))
                gaps.append(pg)
                supports.append(ps)
                confidences.append(confidence)
                excesses.append((pg - tau).clamp_min(0.0))
                valid_per_sample[b] += 1

        value = torch.stack(losses).mean() if losses else logits.sum() * 0
        zero = value.detach() * 0
        return value, {
            "valid_slots": valid_per_sample.mean(),
            "abstention_rate": (valid_per_sample == 0).float().mean(),
            "gap_probability": torch.stack(gaps).mean() if gaps else zero,
            "support_probability": torch.stack(supports).mean() if supports else zero,
            "slot_confidence": torch.stack(confidences).mean() if confidences else zero,
            "gap_excess": torch.stack(excesses).mean() if excesses else zero,
            "pair_slots_used": torch.as_tensor(float(slots), device=value.device),
        }


class nnUNetTrainerR267AdjacentStructureSanity(nnUNetTrainerR267AdjacentStructure):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5
        self.early_min_epochs = 99
