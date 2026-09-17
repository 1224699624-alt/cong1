#!/usr/bin/env python3
"""R320 full-data paired U-Net test for the frozen R317 prior.

The control and prior arms share the exact same two-channel initialization.
The control receives an all-zero second channel and native BCE+Dice. The prior
arm receives the frozen R317 image-centers-only probability map and the exact
R317 background-only continuous-prior penalty. Only train/original-val are
supported; test and clean-test-v2 are intentionally unavailable.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_r312_clean_paired_unet_prior import (
    InstanceSeparationUNet,
    infer_arm,
    make_loaders,
    native_loss,
    seed_all,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r317_image_centers_only"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/experiments/r320_multibackbone/unet"))
    parser.add_argument("--experiment", default="R320_UNET_R317_PRIOR")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-alpha", type=float, default=0.035)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    return parser.parse_args()


def stems_with_complete_triplets(args: argparse.Namespace, split: str) -> list[str]:
    label_dir = args.raw_root / f"{split}_labels"
    prior_dir = args.prior_root / split
    stems = []
    for label_path in sorted(label_dir.glob("*.png")):
        stem = label_path.stem
        if not (prior_dir / f"{stem}.png").is_file():
            raise FileNotFoundError(f"missing frozen prior: {prior_dir / (stem + '.png')}")
        stems.append(stem)
    limit = args.limit_train if split == "train" else args.limit_val
    return stems[:limit] if limit > 0 else stems


def r317_continuous_prior_penalty(
    logits: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
) -> torch.Tensor:
    """Exact one-logit equivalent of R317's two-class foreground-odds loss."""
    lo = prior.amin(dim=(2, 3), keepdim=True)
    hi = prior.amax(dim=(2, 3), keepdim=True)
    normalized = (prior - lo) / (hi - lo + 1e-6)
    weight = normalized.square() * (target < 0.5).float()
    numerator = (weight * F.softplus(logits.float())).flatten(1).sum(1)
    denominator = weight.flatten(1).sum(1)
    valid = denominator > 1e-6
    return (
        (numerator[valid] / (denominator[valid] + 1e-6)).mean()
        if valid.any()
        else logits.sum() * 0.0
    )


def train_arm_r320(
    args: argparse.Namespace,
    stems: dict[str, list[str]],
    arm: str,
    initial_state: dict[str, torch.Tensor],
    use_prior: bool,
) -> dict[str, object]:
    seed_all(args.seed)
    model = InstanceSeparationUNet(args.base_channels, in_channels=2).to(args.device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    train_loader, val_loader = make_loaders(args, stems, use_prior)
    arm_dir = args.output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, prior_losses = [], []
        for batch in train_loader:
            image = batch["image"].to(args.device)
            target = batch["mask"].to(args.device)
            prior = batch["prior"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(image)["mask"]
                base = native_loss(logits, target)
                auxiliary = (
                    r317_continuous_prior_penalty(logits, target, prior)
                    if use_prior
                    else logits.new_zeros(())
                )
                loss = base + args.prior_alpha * auxiliary
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            prior_losses.append(float(auxiliary.detach().cpu()))
        dice = validate(model, val_loader, args.device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "prior_loss": float(np.mean(prior_losses)),
            "val_dice": dice,
        }
        history.append(row)
        print(json.dumps({"arm": arm, **row}), flush=True)
        if dice > best + 1e-4:
            best, best_epoch, bad = dice, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": dice, "arm": arm}, arm_dir / "best.pt")
        else:
            bad += 1
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    (arm_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"arm": arm, "best_epoch": best_epoch, "best_val_dice": best, "completed_epochs": len(history)}


def main() -> None:
    args = parse_args()
    stems = {split: stems_with_complete_triplets(args, split) for split in ("train", "val")}
    expected = {"train": 875, "val": 96}
    if args.limit_train <= 0 and args.limit_val <= 0 and {key: len(value) for key, value in stems.items()} != expected:
        raise RuntimeError(f"unexpected full-data counts: { {key: len(value) for key, value in stems.items()} }")

    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    initial_state = InstanceSeparationUNet(args.base_channels, in_channels=2).state_dict()
    first = "enc1.net.0.weight"
    initial_state[first][:, 1].zero_()
    (args.output_root / "protocol.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "dataset": "TSRS_RSNA-Epiphysis",
                "splits": {key: len(value) for key, value in stems.items()},
                "prior": str(args.prior_root),
                "prior_alpha": args.prior_alpha,
                "shared_initialization": True,
                "prior_input_weight_initialized_to_zero": True,
                "test_used": False,
                "clean_test_v2_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save({"model": initial_state, "seed": args.seed}, args.output_root / "shared_initialization.pt")
    results = []
    for arm, use_prior in (("plain", False), ("prior", True)):
        results.append(train_arm_r320(args, stems, arm, copy.deepcopy(initial_state), use_prior))
        infer_arm(args, stems, arm, use_prior)
    result = {
        "experiment": args.experiment,
        "dataset": "TSRS_RSNA-Epiphysis",
        "train": len(stems["train"]),
        "val": len(stems["val"]),
        "prior": str(args.prior_root),
        "prior_alpha": args.prior_alpha,
        "shared_initialization": True,
        "clean_test_used": False,
        "arms": results,
    }
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
