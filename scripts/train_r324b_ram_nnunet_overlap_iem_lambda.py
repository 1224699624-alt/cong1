#!/usr/bin/env python3
"""R324b: prior-only lambda increase using the locked R324 plain arm."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, WristDataset
from train_r324_ram_nnunet_explicit_overlap_iem import seed_all, train_arm, validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--plain-result", type=Path, required=True)
    parser.add_argument("--relation-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-weight", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=3241)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = json.loads(args.relation_graph.read_text(encoding="utf-8"))
    pairs = [tuple(item["indices"]) for item in graph["pairs"]]
    locked = json.loads(args.plain_result.read_text(encoding="utf-8"))
    plain_val = locked["plain_val"]
    plain_best = locked["plain_best"]

    train_set = WristDataset(args.dataset_root, "train", args.size, True)
    val_set = WristDataset(args.dataset_root, "val", args.size, False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    initial = NnUNetMultiLabelPrior().to(device)
    initial.load_state_dict(payload["model"], strict=True)
    initial_state = copy.deepcopy(initial.state_dict())
    del initial
    torch.cuda.empty_cache()

    arm = f"explicit_overlap_iem_lambda_{args.prior_weight:g}"
    prior_ckpt, prior_best = train_arm(
        arm, initial_state, train_set, val_loader, device, args.output,
        pairs, args.epochs, args.learning_rate, args.prior_weight,
        args.min_epochs, args.patience, args.seed,
    )
    model = NnUNetMultiLabelPrior().to(device)
    model.load_state_dict(torch.load(prior_ckpt, map_location=device, weights_only=False)["model"], strict=True)
    prior_val = validate(model, val_loader, device, pairs)
    numeric = ["macro_dsc", "macro_iou", "voe", "macro_sensitivity", "macro_specificity",
               "overlap_dsc", "overlap_iou", "overlap_voe", "overlap_nsd_2px",
               "overlap_msd_px", "overlap_msd_fail_rate"]
    delta_plain = {key: prior_val[key] - plain_val[key] for key in numeric}
    previous = locked["explicit_overlap_iem_val"]
    delta_previous = {key: prior_val[key] - previous[key] for key in numeric}
    lambda_tag = f"{args.prior_weight:g}".replace(".", "P")
    result = {
        "experiment": f"R324_RAM_NNUNET_EXPLICIT_OVERLAP_IEM_LAMBDA_{lambda_tag}",
        "split": "validation", "test_used": False, "threshold": 0.5,
        "threshold_search": False, "prior_weight": args.prior_weight,
        "gradient_ratio_estimate": 0.16 * args.prior_weight / 0.001,
        "initialization": {"source": str(args.baseline_checkpoint),
                           "sha256": sha256(args.baseline_checkpoint), "strict_load": True},
        "plain_reused_from": str(args.plain_result),
        "relation_graph_reused_from": str(args.relation_graph),
        "plain_best": plain_best, "prior_best": prior_best,
        "plain_val": plain_val, "prior_val": prior_val,
        "delta_prior_minus_plain": delta_plain,
        "delta_prior_minus_r324_lambda_0p001": delta_previous,
        "checkpoint": str(prior_ckpt),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
