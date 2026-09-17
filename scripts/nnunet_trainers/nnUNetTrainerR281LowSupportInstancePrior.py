"""R281: controlled R280 follow-up with support weight reduced from 0.05 to 0.015."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnUNetTrainerR280BalancedInstancePrior import nnUNetTrainerR280BalancedInstancePrior


class nnUNetTrainerR281LowSupportInstancePrior(nnUNetTrainerR280BalancedInstancePrior):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.support_alpha = 0.015
        np.random.seed(281)
        torch.manual_seed(281)
        torch.cuda.manual_seed_all(281)

    def initialize(self):
        super().initialize()
        parent = Path(self.output_folder) / "r280_initialization.json"
        manifest = json.loads(parent.read_text(encoding="utf-8"))
        manifest.update({
            "experiment": "R281",
            "controlled_change": "support_alpha 0.05 -> 0.015",
            "support_alpha": self.support_alpha,
        })
        (Path(self.output_folder) / "r281_initialization.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    def on_train_epoch_end(self, outputs: list[dict]):
        nnUNetTrainer.on_train_epoch_end(self, outputs)
        row = {key: float(np.mean([float(np.asarray(item[key])) for item in outputs]))
               for key in outputs[0] if key != "loss"}
        row["epoch"] = int(self.current_epoch)
        with (Path(self.output_folder) / "r281_loss_dynamics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

    def on_epoch_end(self):
        nnUNetTrainer.on_epoch_end(self)
        score = float(self.logger.get_value("ema_fg_dice", step=-1))
        improved = self._early_best is None or score > self._early_best + self.early_min_delta
        if improved:
            self._early_best = score
            self._early_bad = 0
        else:
            self._early_bad += 1
        if self.current_epoch >= self.early_min_epochs and self._early_bad >= self.early_patience:
            self._early_stop = True
        state = {
            "completed_epochs": int(self.current_epoch),
            "ema_dice": score,
            "early_best": self._early_best,
            "bad_epochs": self._early_bad,
            "stop": self._early_stop,
        }
        (Path(self.output_folder) / "r281_early_stop.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")
        self.print_to_log_file("R281 early stop", json.dumps(state))
