"""Strict R305 binary-vs-instance-supervision paired trainers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _R305Base(nnUNetTrainer):
    expected_dataset = ""
    instance_supervision = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != self.expected_dataset:
            raise RuntimeError(self.plans_manager.dataset_name)
        self.num_epochs = 30
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30
        self.initial_lr = 1e-4
        np.random.seed(305)
        torch.manual_seed(305)
        torch.cuda.manual_seed_all(305)

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        source_path = Path(os.environ["R305_SOURCE_CHECKPOINT"])
        source = torch.load(source_path, map_location=self.device, weights_only=False)["network_weights"]
        target = self.network.state_dict()
        copied = expanded = 0
        for key, value in target.items():
            if key not in source:
                raise RuntimeError(f"missing source tensor: {key}")
            src = source[key]
            if src.shape == value.shape:
                target[key] = src
                copied += 1
            elif self.instance_supervision and "seg_layers" in key and src.shape[0] == 2 and value.shape[0] == 3 and src.shape[1:] == value.shape[1:]:
                adapted = value.clone()
                adapted[:2] = src
                adapted[2].zero_()
                if adapted.ndim == 1:
                    adapted[2] = -6.0
                target[key] = adapted
                expanded += 1
            else:
                raise RuntimeError(f"incompatible tensor {key}: source={tuple(src.shape)}, target={tuple(value.shape)}")
        self.network.load_state_dict(target, strict=True)
        audit = {"source": str(source_path), "matching_tensors": copied,
                 "expanded_segmentation_tensors": expanded,
                 "instance_supervision": self.instance_supervision,
                 "fixed_epochs": self.num_epochs, "seed": 305}
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        (Path(self.output_folder) / "r305_initialization.json").write_text(json.dumps(audit, indent=2))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)


class nnUNetTrainerR305Binary(_R305Base):
    expected_dataset = "Dataset305_TSRSBinarySupervision2D"


class nnUNetTrainerR305InstanceSeam(_R305Base):
    expected_dataset = "Dataset306_TSRSInstanceSeamSupervision2D"
    instance_supervision = True
