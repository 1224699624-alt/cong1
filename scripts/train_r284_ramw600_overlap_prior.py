#!/usr/bin/env python3
"""R284 local pilot: multi-label wrist U-Net with learned overlap prior and pair loss."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


BONE_NAMES = [
    "Capitate", "DistalRadius", "DistalUlna", "Hamate", "Lunate", "Tri",
    "Scaphoid", "Trapezium", "Trapezoid", "Metacarpal1", "Metacarpal2",
    "Metacarpal3", "Metacarpal4", "Metacarpal5",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WristDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, augment: bool) -> None:
        self.root = root
        self.split = split
        self.size = size
        self.augment = augment
        self.mask_files = sorted((root / "BoneSegmentation" / "masks" / split).glob("*.npy"))
        if not self.mask_files:
            raise RuntimeError(f"No masks for split {split}: {root}")

    def __len__(self) -> int:
        return len(self.mask_files)

    def __getitem__(self, index: int) -> dict[str, object]:
        mask_path = self.mask_files[index]
        image_path = self.root / "BoneSegmentation" / "images" / f"{mask_path.stem}.bmp"
        image = Image.open(image_path).convert("L")
        mask = torch.from_numpy((np.load(mask_path) > 0).astype(np.float32))
        image_t = TF.pil_to_tensor(image).float() / 255.0

        # Canonicalize right wrists. Channel anatomy is invariant to this flip.
        if mask_path.stem.endswith("_R"):
            image_t = torch.flip(image_t, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        image_t = TF.resize(image_t, [self.size, self.size], antialias=True)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            angle = random.uniform(-7.0, 7.0)
            translate = [random.randint(-10, 10), random.randint(-10, 10)]
            scale = random.uniform(0.94, 1.06)
            image_t = TF.affine(image_t, angle, translate, scale, 0.0,
                                interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(mask, angle, translate, scale, 0.0,
                             interpolation=InterpolationMode.NEAREST, fill=0.0)
            image_t = TF.adjust_gamma(image_t.clamp(0, 1), random.uniform(0.85, 1.15))
            image_t = (image_t * random.uniform(0.90, 1.10)).clamp(0, 1)

        image_t = (image_t - image_t.mean()) / (image_t.std() + 1e-6)
        return {"image": image_t, "mask": mask, "case": mask_path.stem}


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels), nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(input_channels, output_channels, 1)
        self.conv = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([self.reduce(value), skip], dim=1))


class MultiLabelPriorUNet(nn.Module):
    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.up4 = UpBlock(base * 16, base * 8, base * 8)
        self.up3 = UpBlock(base * 8, base * 4, base * 4)
        self.up2 = UpBlock(base * 4, base * 2, base * 2)
        self.up1 = UpBlock(base * 2, base, base)
        self.seg_head = nn.Conv2d(base, 14, 1)
        self.prior_head = nn.Conv2d(base, 1, 1)
        self.prior_residual = nn.Conv2d(base, 14, 1, bias=False)
        nn.init.zeros_(self.prior_residual.weight)

    def forward(self, image: torch.Tensor, use_prior: bool) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        value = self.bottleneck(self.pool(e4))
        value = self.up4(value, e4)
        value = self.up3(value, e3)
        value = self.up2(value, e2)
        feature = self.up1(value, e1)
        prior_logit = self.prior_head(feature)
        logits = self.seg_head(feature)
        if use_prior:
            logits = logits + self.prior_residual(feature * torch.sigmoid(prior_logit))
        return logits, prior_logit


class NnUNetMultiLabelPrior(nn.Module):
    """nnU-Net v2 PlainConvUNet with a multi-label head and overlap-prior residual."""

    def __init__(self) -> None:
        super().__init__()
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He

        self.backbone = PlainConvUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3]] * 6,
            strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            n_conv_per_stage=[2] * 6,
            num_classes=15,
            n_conv_per_stage_decoder=[2] * 5,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )
        self.backbone.apply(InitWeights_He(1e-2))
        self.prior_residual = nn.Conv2d(1, 14, 1, bias=False)
        nn.init.zeros_(self.prior_residual.weight)

    def forward(self, image: torch.Tensor, use_prior: bool) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.backbone(image)
        logits = raw[:, :14]
        prior_logit = raw[:, 14:15]
        if use_prior:
            logits = logits + self.prior_residual(torch.sigmoid(prior_logit))
        return logits, prior_logit


def dice_bce_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    axes = tuple(range(2, target.ndim))
    dice = 1.0 - ((2 * (probability * target).sum(axes) + 1.0)
                  / (probability.sum(axes) + target.sum(axes) + 1.0)).mean()
    return dice + F.binary_cross_entropy_with_logits(logits, target)


def balanced_probability_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # FP16 rounds values close to one back to exactly one, making log(1-p)=-inf
    # and producing 0*inf NaNs. Keep this sparse overlap loss explicitly in FP32.
    probability = probability.float().clamp(1e-4, 1 - 1e-4)
    target = target.float()
    axes = tuple(range(2, target.ndim))
    dice = 1.0 - ((2 * (probability * target).sum(axes) + 1.0)
                  / (probability.sum(axes) + target.sum(axes) + 1.0)).mean()
    positive = target.sum(axes, keepdim=True)
    negative = (1 - target).sum(axes, keepdim=True)
    positive_weight = negative / (positive + negative + 1e-6)
    negative_weight = positive / (positive + negative + 1e-6)
    bce = -(positive_weight * target * probability.log()
            + negative_weight * (1 - target) * (1 - probability).log()).mean()
    return dice + bce


def auxiliary_losses(logits: torch.Tensor, prior_logit: torch.Tensor,
                     target: torch.Tensor, pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    overlap_target = (target.sum(1, keepdim=True) >= 2).float()
    prior_loss = dice_bce_logits(prior_logit, overlap_target)
    probability = torch.sigmoid(logits)
    predicted_pairs = torch.stack([probability[:, i] * probability[:, j] for i, j in pairs], dim=1)
    target_pairs = torch.stack([target[:, i] * target[:, j] for i, j in pairs], dim=1)
    pair_loss = balanced_probability_loss(predicted_pairs, target_pairs)
    return prior_loss, pair_loss


def top_training_pairs(root: Path, count: int) -> list[tuple[int, int]]:
    pixels = np.zeros((14, 14), dtype=np.int64)
    for path in sorted((root / "BoneSegmentation" / "masks" / "train").glob("*.npy")):
        mask = np.load(path, mmap_mode="r") > 0
        for i in range(14):
            for j in range(i + 1, 14):
                pixels[i, j] += int(np.logical_and(mask[i], mask[j]).sum())
    pairs = [(i, j) for i in range(14) for j in range(i + 1, 14) if pixels[i, j] > 0]
    return sorted(pairs, key=lambda pair: int(pixels[pair]), reverse=True)[:count]


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             use_prior: bool, pairs: list[tuple[int, int]]) -> dict[str, float | list[float]]:
    model.eval()
    intersection = torch.zeros(14, device=device)
    predicted_mass = torch.zeros(14, device=device)
    target_mass = torch.zeros(14, device=device)
    overlap_intersection = overlap_predicted = overlap_target = 0.0
    pair_intersection = torch.zeros(len(pairs), device=device)
    pair_predicted = torch.zeros(len(pairs), device=device)
    pair_target = torch.zeros(len(pairs), device=device)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        logits, _ = model(image, use_prior)
        prediction = torch.sigmoid(logits) >= 0.5
        target_b = target > 0.5
        intersection += (prediction & target_b).sum((0, 2, 3))
        predicted_mass += prediction.sum((0, 2, 3))
        target_mass += target_b.sum((0, 2, 3))
        prediction_overlap = prediction.sum(1) >= 2
        target_overlap_map = target_b.sum(1) >= 2
        overlap_intersection += float((prediction_overlap & target_overlap_map).sum())
        overlap_predicted += float(prediction_overlap.sum())
        overlap_target += float(target_overlap_map.sum())
        for k, (i, j) in enumerate(pairs):
            pp = prediction[:, i] & prediction[:, j]
            tt = target_b[:, i] & target_b[:, j]
            pair_intersection[k] += (pp & tt).sum()
            pair_predicted[k] += pp.sum()
            pair_target[k] += tt.sum()
    class_dice = (2 * intersection + 1) / (predicted_mass + target_mass + 1)
    pair_dice = (2 * pair_intersection + 1) / (pair_predicted + pair_target + 1)
    return {
        "macro_dice": float(class_dice.mean()),
        "class_dice": class_dice.cpu().tolist(),
        "overlap_dice": float((2 * overlap_intersection + 1) / (overlap_predicted + overlap_target + 1)),
        "pair_overlap_dice": pair_dice.cpu().tolist(),
    }


def surface_dice(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    pred_surface = prediction ^ binary_erosion(prediction)
    target_surface = target ^ binary_erosion(target)
    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_pred = distance_transform_edt(~pred_surface)
    numerator = (distance_to_target[pred_surface] <= tolerance).sum()
    numerator += (distance_to_pred[target_surface] <= tolerance).sum()
    denominator = pred_surface.sum() + target_surface.sum()
    return float(numerator / max(denominator, 1))


@torch.no_grad()
def test_metrics_and_predictions(model: nn.Module, loader: DataLoader, device: torch.device,
                                 use_prior: bool, pairs: list[tuple[int, int]]) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    predictions: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    overlap_nsd: list[float] = []
    pair_nsd: list[list[float]] = [[] for _ in pairs]
    for batch in loader:
        logits, _ = model(batch["image"].to(device), use_prior)
        pred = (torch.sigmoid(logits) >= 0.5).cpu().numpy()
        target = (batch["mask"].numpy() > 0.5)
        for b, case in enumerate(batch["case"]):
            predictions[str(case)] = pred[b]
            targets[str(case)] = target[b]
            overlap_nsd.append(surface_dice(pred[b].sum(0) >= 2, target[b].sum(0) >= 2))
            for k, (i, j) in enumerate(pairs):
                pair_nsd[k].append(surface_dice(pred[b, i] & pred[b, j], target[b, i] & target[b, j]))
    base = validate(model, loader, device, use_prior, pairs)
    base["overlap_nsd_2px"] = float(np.mean(overlap_nsd))
    base["pair_overlap_nsd_2px"] = [float(np.mean(values)) for values in pair_nsd]
    return base, {"predictions": predictions, "targets": targets}


def train_phase(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, output: Path, phase: str, use_prior: bool,
                pairs: list[tuple[int, int]], epochs: int, learning_rate: float,
                baseline_val_dice: float | None = None) -> tuple[Path, dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoint = output / f"{phase}_best.pth"
    history_path = output / f"{phase}_history.jsonl"
    best_key: tuple[float, float] | None = None
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        sums = {"loss": 0.0, "base": 0.0, "prior": 0.0, "pair": 0.0}
        started = time.time()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, prior_logit = model(image, use_prior)
                base_loss = dice_bce_logits(logits, target)
                if use_prior:
                    prior_loss, pair_loss = auxiliary_losses(logits, prior_logit, target, pairs)
                    loss = base_loss + 0.10 * prior_loss + 0.05 * pair_loss
                else:
                    prior_loss = pair_loss = base_loss.new_zeros(())
                    loss = base_loss
            if not torch.isfinite(loss):
                raise RuntimeError({
                    "phase": phase, "epoch": epoch,
                    "base_loss": float(base_loss.detach().float()),
                    "prior_loss": float(prior_loss.detach().float()),
                    "pair_loss": float(pair_loss.detach().float()),
                })
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            for key, value in (("loss", loss), ("base", base_loss),
                               ("prior", prior_loss), ("pair", pair_loss)):
                sums[key] += float(value.detach())
        scheduler.step()
        metrics = validate(model, val_loader, device, use_prior, pairs)
        row = {
            "phase": phase, "epoch": epoch, "seconds": time.time() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{key}": value / len(train_loader) for key, value in sums.items()},
            "val_macro_dice": metrics["macro_dice"],
            "val_overlap_dice": metrics["overlap_dice"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        if use_prior and baseline_val_dice is not None:
            eligible = float(metrics["macro_dice"]) >= baseline_val_dice - 0.002
            key = (1.0 if eligible else 0.0,
                   float(metrics["overlap_dice"]) if eligible else float(metrics["macro_dice"]))
        else:
            key = (float(metrics["macro_dice"]), float(metrics["overlap_dice"]))
        if best_key is None or key > best_key:
            best_key = key
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "metrics": metrics, "epoch": epoch,
                        "phase": phase, "pairs": pairs}, checkpoint)
        else:
            bad_epochs += 1
        if epoch >= 8 and bad_epochs >= 7:
            break
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return checkpoint, payload


def render_comparisons(root: Path, output: Path, cases: list[str],
                       baseline: dict[str, np.ndarray], improved: dict[str, np.ndarray], size: int) -> None:
    palette = np.asarray([
        [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
        [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
        [170, 110, 40], [255, 250, 200],
    ], dtype=np.uint8)

    def colored(mask: np.ndarray) -> np.ndarray:
        result = np.zeros((size, size, 3), dtype=np.uint8)
        for channel in range(14):
            result[mask[channel]] = palette[channel]
        return result

    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        image = Image.open(root / "BoneSegmentation" / "images" / f"{case}.bmp").convert("L")
        array = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))
        if case.endswith("_R"):
            array = np.fliplr(array)
        rgb = np.repeat(array[..., None], 3, axis=2)
        gt = baseline["targets"][case]
        base = baseline["predictions"][case]
        imp = improved["predictions"][case]
        error = np.repeat(array[..., None], 3, axis=2)
        gt_overlap = gt.sum(0) >= 2
        pred_overlap = imp.sum(0) >= 2
        error[gt_overlap & pred_overlap] = [0, 220, 0]
        error[gt_overlap & ~pred_overlap] = [255, 0, 0]
        error[~gt_overlap & pred_overlap] = [255, 220, 0]
        panel = np.concatenate([rgb, colored(gt), colored(base), colored(imp), error], axis=1)
        Image.fromarray(panel).save(output / f"{case}_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\gutou\RAM-W600"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ram_w600/r284_overlap_prior"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--baseline-epochs", type=int, default=30)
    parser.add_argument("--improved-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=284)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--architecture", choices=("compact", "nnunet"), default="compact")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if "TSRS_RSNA-Articular-Surface" in str(args.dataset_root):
        raise RuntimeError("R284 is RAM-W600 only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = top_training_pairs(args.dataset_root, 15)
    manifest = {
        "experiment": "R284", "dataset": str(args.dataset_root),
        "splits": {"train": 425, "val": 69, "test": 124},
        "image_size": args.size, "batch_size": args.batch_size,
        "baseline_epochs": args.baseline_epochs, "improved_epochs": args.improved_epochs,
        "device": str(device), "bone_names": BONE_NAMES,
        "pairs": [{"indices": pair, "names": [BONE_NAMES[pair[0]], BONE_NAMES[pair[1]]]}
                  for pair in pairs],
        "loss": "multilabel Dice+BCE + 0.10 overlap-prior + 0.05 pair-overlap",
        "test_used_for_model_selection": False,
        "threshold_search_used": False,
        "baseline_checkpoint_source": (str(args.baseline_checkpoint)
                                       if args.baseline_checkpoint else None),
        "architecture": args.architecture,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_set = WristDataset(args.dataset_root, "train", args.size, augment=True)
    val_set = WristDataset(args.dataset_root, "val", args.size, augment=False)
    test_set = WristDataset(args.dataset_root, "test", args.size, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = (NnUNetMultiLabelPrior() if args.architecture == "nnunet"
             else MultiLabelPriorUNet()).to(device)

    if args.dry_run:
        batch = next(iter(train_loader))
        logits, prior = model(batch["image"].to(device), True)
        target = batch["mask"].to(device)
        base_loss = dice_bce_logits(logits, target)
        prior_loss, pair_loss = auxiliary_losses(logits, prior, target, pairs)
        total = base_loss + 0.10 * prior_loss + 0.05 * pair_loss
        total.backward()
        result = {"logits": list(logits.shape), "prior": list(prior.shape),
                  "loss": float(total.detach()), "finite": bool(torch.isfinite(total))}
        print(json.dumps(result), flush=True)
        return

    if args.baseline_checkpoint is not None:
        baseline_checkpoint = args.baseline_checkpoint
        baseline_payload = torch.load(baseline_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(baseline_payload["model"])
        if baseline_payload.get("phase") != "baseline":
            raise RuntimeError(f"Not a baseline checkpoint: {baseline_checkpoint}")
    else:
        baseline_checkpoint, baseline_payload = train_phase(
            model, train_loader, val_loader, device, args.output, "baseline", False,
            pairs, args.baseline_epochs, 3e-4)
    baseline_val_dice = float(baseline_payload["metrics"]["macro_dice"])

    # Exact baseline initialization; prior_residual remains zero before fine-tuning.
    improved_checkpoint, improved_payload = train_phase(
        model, train_loader, val_loader, device, args.output, "improved", True,
        pairs, args.improved_epochs, 5e-5, baseline_val_dice)

    model.load_state_dict(torch.load(baseline_checkpoint, map_location=device, weights_only=False)["model"])
    baseline_test, baseline_predictions = test_metrics_and_predictions(
        model, test_loader, device, False, pairs)
    model.load_state_dict(torch.load(improved_checkpoint, map_location=device, weights_only=False)["model"])
    improved_test, improved_predictions = test_metrics_and_predictions(
        model, test_loader, device, True, pairs)
    result = {
        "baseline_val": baseline_payload["metrics"],
        "improved_val": improved_payload["metrics"],
        "baseline_test": baseline_test,
        "improved_test": improved_test,
        "delta_test": {
            "macro_dice": improved_test["macro_dice"] - baseline_test["macro_dice"],
            "overlap_dice": improved_test["overlap_dice"] - baseline_test["overlap_dice"],
            "overlap_nsd_2px": improved_test["overlap_nsd_2px"] - baseline_test["overlap_nsd_2px"],
        },
        "threshold": 0.5, "threshold_search_used": False,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    overlap_rank = sorted(baseline_predictions["targets"],
                          key=lambda case: int((baseline_predictions["targets"][case].sum(0) >= 2).sum()),
                          reverse=True)[:12]
    render_comparisons(args.dataset_root, args.output / "visualizations", overlap_rank,
                       baseline_predictions, improved_predictions, args.size)
    print(json.dumps({"status": "complete", "result": result["delta_test"]}), flush=True)


if __name__ == "__main__":
    main()
