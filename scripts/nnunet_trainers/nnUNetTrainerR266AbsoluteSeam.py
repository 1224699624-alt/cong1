"""R266: non-saturating absolute seam constraint on the R265 priors.

The R265 relative hinge compared the foreground probability in a seam with
the probability in its two supporting bones.  Once that relative margin was
met, the relation term became exactly zero.  R266 instead applies an
absolute, soft upper bound to the seam foreground probability and keeps an
independent lower bound on the support probability.

The six channels retain the R265 schema: slots 0--3 are CC, slot 4 is CM,
and slot 5 is CR.  Age/sex conditioning is preserved in the prior maps that
activate and weight each slot; CR intentionally has a permissive upper bound
so legitimate low-age CR configurations are not suppressed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerR265HierarchicalCarpal import (
    nnUNetTrainerR265HierarchicalCarpal,
)


class nnUNetTrainerR266AbsoluteSeam(nnUNetTrainerR265HierarchicalCarpal):
    """R265 architecture/checkpoint with an absolute seam probability loss."""

    run_prefix = "r266"
    # CC is the strongest separation relation, CM is intermediate, and CR is
    # deliberately permissive because low-age CR relations can be anatomically
    # valid before the carpal row is fully ossified.
    slot_upper_bounds = (0.10, 0.10, 0.10, 0.10, 0.15, 0.28)
    slot_support_floors = (0.72, 0.72, 0.72, 0.72, 0.68, 0.65)
    slot_weights = (1.0, 1.0, 1.0, 1.0, 0.45, 0.15)

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # R265's 0.04 ratio made the new absolute term numerically negligible
        # (alpha was ~1e-4 while the relation gradient was ~1e-2).  A ratio of
        # 1.0 brings the weighted relation gradient to the same order as the
        # base segmentation gradient without allowing it to dominate.
        self.gradient_ratio = 1.0
        self.seam_temperature = 0.06
        self.support_temperature = 0.08
        self.support_keep_weight = 0.35

    def _relation_loss(self, logits, target, data):
        probability = torch.softmax(logits.float(), dim=1)[:, 1]
        y = (target[:, 0] > 0).float()
        priors = torch.clamp(data[:, 1:].float() / 255.0, 0.0, 1.0)
        losses = []
        gap_values, support_values, confidences = [], [], []
        type_gap, type_support, type_count = {}, {}, {}
        gap_excesses = []
        valid_per_sample = torch.zeros(len(data), device=logits.device)

        for b in range(len(data)):
            for slot in range(6):
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
                tau = torch.as_tensor(self.slot_upper_bounds[slot], device=pg.device)
                floor = torch.as_tensor(self.slot_support_floors[slot], device=ps.device)

                # softplus((p - tau) / T) is non-zero for every finite p.  It
                # behaves like a smooth absolute upper-bound penalty without
                # the dead zone of ReLU/hinge.  Subtracting softplus(0) keeps
                # the loss near zero when the probability is comfortably below
                # the bound while preserving a small smooth tail.
                gap_penalty = F.softplus((pg - tau) / self.seam_temperature)
                support_penalty = F.softplus((floor - ps) / self.support_temperature)
                relation = self.slot_weights[slot] * confidence * (
                    gap_penalty + self.support_keep_weight * support_penalty
                )
                losses.append(relation)
                gap_values.append(pg)
                support_values.append(ps)
                confidences.append(confidence)
                gap_excesses.append((pg - tau).clamp_min(0))
                valid_per_sample[b] += 1

                kind = "cc" if slot < 4 else ("cm" if slot == 4 else "cr")
                type_gap.setdefault(kind, []).append(pg)
                type_support.setdefault(kind, []).append(ps)
                type_count[kind] = type_count.get(kind, 0) + 1

        value = torch.stack(losses).mean() if losses else logits.sum() * 0
        stats = {
            "valid_slots": valid_per_sample.mean(),
            "abstention_rate": (valid_per_sample == 0).float().mean(),
            "gap_probability": torch.stack(gap_values).mean() if gap_values else value.detach() * 0,
            "support_probability": torch.stack(support_values).mean() if support_values else value.detach() * 0,
            "slot_confidence": torch.stack(confidences).mean() if confidences else value.detach() * 0,
            "gap_excess": torch.stack(gap_excesses).mean() if gap_excesses else value.detach() * 0,
        }
        # Per-relation diagnostics make it possible to verify that CR remains
        # active in younger cases rather than silently being filtered out.
        for kind in ("cc", "cm", "cr"):
            stats[f"{kind}_valid"] = torch.as_tensor(float(type_count.get(kind, 0)), device=value.device)
            stats[f"{kind}_gap_probability"] = (
                torch.stack(type_gap[kind]).mean() if kind in type_gap else value.detach() * 0
            )
            stats[f"{kind}_support_probability"] = (
                torch.stack(type_support[kind]).mean() if kind in type_support else value.detach() * 0
            )
        return value, stats


class nnUNetTrainerR266AbsoluteSeamSanity(nnUNetTrainerR266AbsoluteSeam):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 5
        self.early_min_epochs = 99
