#!/usr/bin/env python3
"""Paired R320 training for official TransUNet and Swin-UMamba backbones."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from r320_prior_plugin import (
    ZeroInitializedPriorAdapter,
    binary_dice_bce_loss,
    continuous_background_prior_loss,
    foreground_logits,
)
from run_r312_clean_paired_unet_prior import CleanPriorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=("transunet", "swin_umamba"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r317_image_centers_only"))
    parser.add_argument("--external-root", type=Path, default=Path("/root/autodl-tmp/external"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-alpha", type=float, default=0.035)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--transunet-npz", type=Path)
    parser.add_argument("--swin-pretrained", type=Path)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AdaptedBackbone(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.input_adapter = ZeroInitializedPriorAdapter()
        self.backbone = backbone

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.input_adapter(data))


def build_transunet(args: argparse.Namespace) -> nn.Module:
    repo = args.external_root / "TransUNet"
    sys.path.insert(0, str(repo))
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer

    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = 1
    config.n_skip = 3
    config.patches.grid = (args.img_size // 16, args.img_size // 16)
    model = VisionTransformer(config, img_size=args.img_size, num_classes=1)
    if args.transunet_npz is not None:
        if not args.transunet_npz.is_file():
            raise FileNotFoundError(args.transunet_npz)
        model.load_from(weights=np.load(args.transunet_npz))
    return AdaptedBackbone(model)


def build_swin_umamba(args: argparse.Namespace) -> nn.Module:
    package_root = args.external_root / "Swin-UMamba" / "swin_umamba"
    sys.path.insert(0, str(package_root))
    from nnunetv2.nets.SwinUMamba import SwinUMamba, load_pretrained_ckpt

    model = SwinUMamba(in_chans=1, out_chans=1, deep_supervision=False)
    if args.swin_pretrained is not None:
        if not args.swin_pretrained.is_file():
            raise FileNotFoundError(args.swin_pretrained)
        model = load_pretrained_ckpt(model, str(args.swin_pretrained))
    return AdaptedBackbone(model)


def build_model(args: argparse.Namespace) -> nn.Module:
    return build_transunet(args) if args.backbone == "transunet" else build_swin_umamba(args)


def complete_stems(args: argparse.Namespace, split: str) -> list[str]:
    stems = []
    for label in sorted((args.raw_root / f"{split}_labels").glob("*.png")):
        prior = args.prior_root / split / label.name
        if not prior.is_file():
            raise FileNotFoundError(prior)
        stems.append(label.stem)
    limit = args.limit_train if split == "train" else args.limit_val
    return stems[:limit] if limit > 0 else stems


def make_loaders(args: argparse.Namespace, stems: dict[str, list[str]], use_prior: bool) -> tuple[DataLoader, DataLoader]:
    train = CleanPriorDataset(args, "train", stems["train"], True, use_prior)
    val = CleanPriorDataset(args, "val", stems["val"], False, use_prior)
    generator = torch.Generator().manual_seed(args.seed)
    return (
        DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, generator=generator),
        DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
    )


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, args: argparse.Namespace) -> float:
    model.eval()
    intersection = prediction_sum = target_sum = 0.0
    for batch in loader:
        probability = torch.sigmoid(foreground_logits(model(batch["image"].to(args.device))))
        prediction = probability >= args.threshold
        target = batch["mask"].to(args.device) > 0.5
        intersection += float((prediction & target).sum())
        prediction_sum += float(prediction.sum())
        target_sum += float(target.sum())
    return (2.0 * intersection + 1.0) / (prediction_sum + target_sum + 1.0)


def train_arm(
    args: argparse.Namespace,
    stems: dict[str, list[str]],
    arm: str,
    initial_state: dict[str, torch.Tensor],
    use_prior: bool,
) -> dict[str, object]:
    seed_all(args.seed)
    model = build_model(args).to(args.device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    train_loader, val_loader = make_loaders(args, stems, use_prior)
    arm_dir = args.output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, base_losses, prior_losses = [], [], []
        for batch in train_loader:
            image = batch["image"].to(args.device)
            target = batch["mask"].to(args.device)
            prior = batch["prior"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                output = model(image)
                base = binary_dice_bce_loss(output, target)
                auxiliary = continuous_background_prior_loss(output, target, prior) if use_prior else base.new_zeros(())
                loss = base + args.prior_alpha * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            base_losses.append(float(base.detach().cpu()))
            prior_losses.append(float(auxiliary.detach().cpu()))
        dice = validate(model, val_loader, args)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "base_loss": float(np.mean(base_losses)),
            "prior_loss": float(np.mean(prior_losses)),
            "weighted_prior_to_base": float(args.prior_alpha * np.mean(prior_losses) / max(np.mean(base_losses), 1e-8)),
            "val_dice": dice,
        }
        history.append(row)
        print(json.dumps({"backbone": args.backbone, "arm": arm, **row}), flush=True)
        if dice > best + 1e-4:
            best, best_epoch, bad = dice, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": dice, "arm": arm}, arm_dir / "best.pt")
        else:
            bad += 1
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    (arm_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"arm": arm, "best_epoch": best_epoch, "best_val_dice": best, "completed_epochs": len(history)}


@torch.no_grad()
def infer_arm(args: argparse.Namespace, stems: dict[str, list[str]], arm: str, use_prior: bool) -> None:
    checkpoint = torch.load(args.output_root / arm / "best.pt", map_location=args.device, weights_only=False)
    model = build_model(args).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = CleanPriorDataset(args, "val", stems["val"], False, use_prior)
    output_dir = args.output_root / arm / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in dataset:
        probability = torch.sigmoid(foreground_logits(model(item["image"].unsqueeze(0).to(args.device))))[0, 0].cpu().numpy()
        height, width = map(int, item["shape"].tolist())
        prediction = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR) >= args.threshold
        Image.fromarray(prediction.astype(np.uint8) * 255).save(output_dir / f"{item['stem']}.png")


def main() -> None:
    args = parse_args()
    stems = {split: complete_stems(args, split) for split in ("train", "val")}
    if args.limit_train <= 0 and args.limit_val <= 0 and {key: len(value) for key, value in stems.items()} != {"train": 875, "val": 96}:
        raise RuntimeError("unexpected full-data counts")
    args.output_root.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    initial_model = build_model(args)
    initial_state = copy.deepcopy(initial_model.state_dict())
    adapter_audit = initial_model.input_adapter.audit()
    if adapter_audit != {"xray_weight": 1.0, "prior_weight": 0.0}:
        raise RuntimeError(f"invalid neutral adapter initialization: {adapter_audit}")
    torch.save({"model": initial_state, "seed": args.seed}, args.output_root / "shared_initialization.pt")
    protocol = {
        "experiment": "R320_MULTIBACKBONE_R317_PRIOR",
        "backbone": args.backbone,
        "splits": {key: len(value) for key, value in stems.items()},
        "prior": str(args.prior_root),
        "prior_alpha": args.prior_alpha,
        "adapter_initialization": adapter_audit,
        "shared_initialization": True,
        "test_used": False,
        "clean_test_v2_used": False,
    }
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for arm, use_prior in (("plain", False), ("prior", True)):
        results.append(train_arm(args, stems, arm, copy.deepcopy(initial_state), use_prior))
        infer_arm(args, stems, arm, use_prior)
    protocol["arms"] = results
    (args.output_root / "result.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
