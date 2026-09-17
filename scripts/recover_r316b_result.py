#!/usr/bin/env python3
"""Recover an R316B result JSON from a completed checkpoint without retraining."""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

import train_r316_safe_relation_selector as r316
import train_r258b_prediction_relation_prior as r258


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()

    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    run = Namespace(**config)
    train_meta = r258.read_metadata(Path(run.train_metadata))
    val_meta = r258.read_metadata(Path(run.val_metadata))
    if (len(train_meta), len(val_meta)) != (875, 96):
        raise RuntimeError("Recovery expected the full 875/96 protocol")
    train, train_audit = r316.build_samples(Path(run.variant_root), "train", train_meta, Path(run.train_proposals), run)
    val, val_audit = r316.build_samples(Path(run.variant_root), "val", val_meta, Path(run.val_proposals), run)
    train_dataset = r316.SafePairDataset(train, run, True)
    val_dataset = r316.SafePairDataset(val, run, False)
    result = {
        "experiment": "R316B_FULL_QUANTILE_HARD_NEGATIVE",
        "recovered_from_completed_checkpoint": True,
        "config": config,
        "train_audit": train_audit,
        "val_audit": val_audit,
        "hard_negative_train_count": train_dataset.hard_negative_count,
        "hard_negative_val_count": val_dataset.hard_negative_count,
        "history": payload["history"],
        "checkpoint": str(a.checkpoint),
        "clean_test_used": False,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "output": str(a.output),
        "epochs": len(payload["history"]),
        "hard_negative_train_count": train_dataset.hard_negative_count,
        "hard_negative_val_count": val_dataset.hard_negative_count,
    }))


if __name__ == "__main__":
    main()
