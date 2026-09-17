"""Mature standard nnU-Net continuation on the clean-panel subset."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerR310CleanPlain(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != "Dataset310_TSRSPlainCleanPanels2D":
            raise RuntimeError(self.plans_manager.dataset_name)
        self.initial_lr = 1e-4
        self.num_epochs = 60
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.early_min_epochs = 15
        self.early_patience = 10
        self.early_min_delta = 1e-4
        self._early_best = None
        self._early_bad = 0
        self._early_stop = False
        np.random.seed(260)
        torch.manual_seed(260)
        torch.cuda.manual_seed_all(260)

    def initialize(self):
        if self.was_initialized:
            return
        super().initialize()
        path = Path(os.environ["R310_SOURCE_CHECKPOINT"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = os.environ.get("R310_EXPECTED_SOURCE_SHA256")
        if expected and digest != expected:
            raise RuntimeError(f"checkpoint hash mismatch: {digest} != {expected}")
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(payload["network_weights"], strict=True)
        first = self.network.state_dict()["encoder.stages.0.0.convs.0.conv.weight"]
        if tuple(first.shape) != (32, 1, 3, 3):
            raise RuntimeError(f"source is not a one-channel mature nnU-Net: {tuple(first.shape)}")
        audit = {
            "checkpoint": str(path), "sha256": digest,
            "strict_all_network_keys_loaded": True,
            "input_channels": 1, "prior_channel": False,
            "prior_loss": False, "seed": 260,
        }
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        (Path(self.output_folder) / "r310_initialization.json").write_text(json.dumps(audit, indent=2))
        self.print_to_log_file("R310 initialization", json.dumps(audit))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(), self.initial_lr,
            weight_decay=self.weight_decay, momentum=0.99, nesterov=True,
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, 60)

    def on_epoch_end(self):
        super().on_epoch_end()
        score = float(self.logger.get_value("ema_fg_dice", step=-1))
        improved = self._early_best is None or score > self._early_best + self.early_min_delta
        if improved:
            self._early_best, self._early_bad = score, 0
        else:
            self._early_bad += 1
        if self.current_epoch >= self.early_min_epochs and self._early_bad >= self.early_patience:
            self._early_stop = True
        state = {
            "completed_epochs": int(self.current_epoch), "ema_dice": score,
            "early_best": self._early_best, "bad_epochs": self._early_bad,
            "stop": self._early_stop,
        }
        (Path(self.output_folder) / "r310_early_stop.json").write_text(json.dumps(state, indent=2))
        self.print_to_log_file("R310 early stop", json.dumps(state))

    def run_training(self):
        self.on_train_start()
        while self.current_epoch < self.num_epochs and not self._early_stop:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train = [self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val = [self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val)
            self.on_epoch_end()
        self.on_train_end()
