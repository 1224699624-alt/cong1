"""R317 continuous-prior trainer with periodic checkpoints for R201 selection."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .nnUNetTrainerR260MaturePrior import nnUNetTrainerR260MaturePrior


class nnUNetTrainerR317Continuous035(nnUNetTrainerR260MaturePrior):
    snapshot_period = 3

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.prior_alpha = 0.035

    def on_epoch_end(self):
        super().on_epoch_end()
        completed = int(self.current_epoch)
        if completed > 0 and completed % self.snapshot_period == 0:
            path = Path(self.output_folder) / f"checkpoint_epoch_{completed:03d}.pth"
            self.save_checkpoint(path)
            manifest = Path(self.output_folder) / "r317_periodic_checkpoints.json"
            existing = json.loads(manifest.read_text()) if manifest.is_file() else {"checkpoints": []}
            row = {"completed_epochs": completed, "path": str(path),
                   "ema_dice": float(self.logger.get_value("ema_fg_dice", step=-1))}
            existing["checkpoints"] = [x for x in existing["checkpoints"]
                                       if int(x["completed_epochs"]) != completed] + [row]
            existing["checkpoints"].sort(key=lambda x: int(x["completed_epochs"]))
            manifest.write_text(json.dumps(existing, indent=2))
