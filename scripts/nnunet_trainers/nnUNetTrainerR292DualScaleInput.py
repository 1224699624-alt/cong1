"""R292: mature nnU-Net continuation with native and global-context inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerR292DualScaleInput(nnUNetTrainer):
    """Isolate the architectural dual-scale input effect before adding priors."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        expected = "Dataset292_TSRS_RSNAEpiphysisDualScale2D"
        if self.plans_manager.dataset_name != expected:
            raise RuntimeError((self.plans_manager.dataset_name, expected))
        self.initial_lr = 1e-4
        self.num_epochs = 60
        self.num_iterations_per_epoch = 100
        self.num_val_iterations_per_epoch = 30
        self.early_min_epochs = 20
        self.early_patience = 12
        self.early_min_delta = 1e-4
        self._early_best = None
        self._early_bad = 0
        self._early_stop = False
        np.random.seed(292)
        torch.manual_seed(292)
        torch.cuda.manual_seed_all(292)

    def initialize(self):
        if self.was_initialized:
            return
        if os.environ.get("nnUNet_compile", "").lower() != "false":
            raise RuntimeError("R292 requires nnUNet_compile=false")
        super().initialize()
        path_value = os.environ.get("R292_ADAPTED_CHECKPOINT")
        if path_value:
            path = Path(path_value)
            payload = torch.load(path, map_location=self.device, weights_only=False)
            state = payload["network_weights"]
            self.network.load_state_dict(state, strict=True)
            first = state["encoder.stages.0.0.convs.0.conv.weight"]
            if tuple(first.shape) != (32, 2, 3, 3) or int(torch.count_nonzero(first[:, 1])):
                raise RuntimeError("Global-context channel was not initialized at exact zero")
        else:
            # Prediction from a completed run uses nnU-Net's native checkpoint
            # loaded by the predictor; no mature one-channel adapter is needed.
            path = Path(self.output_folder) / "checkpoint_best.pth"
        manifest = {
            "experiment": "R292", "checkpoint": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
            "strict_all_network_keys_loaded": bool(path_value),
            "channel_0": "native high-resolution X-ray",
            "channel_1": "whole-hand global context downsampled to long-side 512",
            "labels_used_for_input": False, "prior_or_gap_loss": False,
            "clean_test_used": False,
        }
        (Path(self.output_folder) / "r292_initialization.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        self.print_to_log_file("R292 initialization", json.dumps(manifest))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr,
                                    weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def on_train_epoch_end(self, outputs: list[dict]):
        super().on_train_epoch_end(outputs)
        first = self.network.state_dict()["encoder.stages.0.0.convs.0.conv.weight"]
        row = {"epoch": int(self.current_epoch),
               "loss": float(np.mean([float(np.asarray(x["loss"])) for x in outputs])),
               "aux_input_weight_absmax": float(first[:, 1].abs().max().detach().cpu())}
        with (Path(self.output_folder) / "r292_loss_dynamics.jsonl").open(
                "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

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
        state = {"completed_epochs": int(self.current_epoch), "ema_dice": score,
                 "early_best": self._early_best, "bad_epochs": self._early_bad,
                 "stop": self._early_stop}
        (Path(self.output_folder) / "r292_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R292 early stop", json.dumps(state))

    def run_training(self):
        self.on_train_start()
        while self.current_epoch < self.num_epochs and not self._early_stop:
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train))
                             for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [self.validation_step(next(self.dataloader_val))
                               for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()
