"""Local matched nnU-Net 2.8.1 trainers for R273/R274."""
from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerR273Local(nnUNetTrainer):
    """Short but non-smoke local baseline for a 6 GB GPU."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 30
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30


class nnUNetTrainerR273LocalSanity(nnUNetTrainerR273Local):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 3


class nnUNetTrainerR274SeamLocal(nnUNetTrainerR273Local):
    """Matched trainer for the ternary bone/background/seam target dataset."""
    pass


class nnUNetTrainerR274SeamLocalSanity(nnUNetTrainerR274SeamLocal):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 3


class nnUNetTrainerR275ControlMature(nnUNetTrainerR273Local):
    """Continuation trainer for the matched mature-control run."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30


class nnUNetTrainerR275SeamMature(nnUNetTrainerR274SeamLocal):
    """Continuation trainer for the matched mature seam-supervision run."""
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30
