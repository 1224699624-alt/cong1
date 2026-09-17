"""R316C loss-strength ablations on the exact mature R260 trainer."""
from __future__ import annotations

import torch

from .nnUNetTrainerR260MaturePrior import nnUNetTrainerR260MaturePrior


class nnUNetTrainerR316CAlpha020(nnUNetTrainerR260MaturePrior):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.prior_alpha = 0.020


class nnUNetTrainerR316CAlpha035(nnUNetTrainerR260MaturePrior):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.prior_alpha = 0.035
