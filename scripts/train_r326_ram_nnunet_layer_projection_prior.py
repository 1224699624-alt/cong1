#!/usr/bin/env python3
"""R326: BLS-GAN/LSN-inspired inference-time layer prior for RAM-W600.

The mature R285 multi-label nnU-Net is frozen. A compact post-logit adapter
receives the radiograph, the 14 base probabilities, and an expected-overlap
map. It predicts (a) bounded per-bone logit corrections and (b) per-bone
radiographic layers. The layers are constrained by the transmission-style
reconstruction used by BLS-GAN/LSN and by label-derived pseudo layers that
split optical attenuation equally among all bones occupying a pixel.

This is a paired validation experiment:
  1. frozen mature baseline;
  2. capacity-matched segmentation-only adapter;
  3. identical adapter with the layer/projection prior.

The official RAM test split is never loaded. No threshold search is performed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, dice_bce_logits


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WristLayerDataset(Dataset):
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
        raw = TF.pil_to_tensor(Image.open(image_path).convert("L")).float() / 255.0
        mask = torch.from_numpy((np.load(mask_path) > 0).astype(np.float32))
        if mask_path.stem.endswith("_R"):
            raw = torch.flip(raw, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))
        raw = TF.resize(raw, [self.size, self.size], antialias=True)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        if self.augment:
            angle = random.uniform(-7.0, 7.0)
            translate = [random.randint(-10, 10), random.randint(-10, 10)]
            scale = random.uniform(0.94, 1.06)
            raw = TF.affine(raw, angle, translate, scale, 0.0,
                            interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(mask, angle, translate, scale, 0.0,
                             interpolation=InterpolationMode.NEAREST, fill=0.0)
            raw = TF.adjust_gamma(raw.clamp(0, 1), random.uniform(0.85, 1.15))
            raw = (raw * random.uniform(0.90, 1.10)).clamp(0, 1)
        image = (raw - raw.mean()) / (raw.std() + 1e-6)
        return {"image": image, "raw": raw, "mask": mask, "case": mask_path.stem}


class ConvNormAct(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(8, cout), nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LayerProjectionAdapter(nn.Module):
    """Small inference-time adapter; no GT is used by forward()."""

    def __init__(self, width: int = 32, residual_limit: float = 0.35) -> None:
        super().__init__()
        self.residual_limit = residual_limit
        self.context = nn.Sequential(
            ConvNormAct(16, width),
            ConvNormAct(width, width),
        )
        self.residual_head = nn.Conv2d(width, 14, 1)
        self.layer_head = nn.Conv2d(width, 14, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, image: torch.Tensor, base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base_logits = base_logits.detach()
        base_probability = torch.sigmoid(base_logits)
        expected_overlap = (base_probability.sum(1, keepdim=True) - 1.0).clamp(0.0, 1.0)
        feature = self.context(torch.cat((image, base_probability, expected_overlap), dim=1))
        residual = self.residual_limit * torch.tanh(self.residual_head(feature))
        refined_logits = base_logits + residual
        refined_probability = torch.sigmoid(refined_logits)
        layers = torch.sigmoid(self.layer_head(feature)) * refined_probability
        return refined_logits, layers


def transmission_reconstruction(layers: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """R = 1 - product_i(1-L_i) * (1-soft), with a non-learned soft layer."""
    union_probability = 1.0 - torch.prod(1.0 - layers.clamp(0, 1), dim=1, keepdim=True)
    soft_layer = raw * (1.0 - union_probability.detach())
    return 1.0 - torch.prod(1.0 - layers.clamp(0, 1), dim=1, keepdim=True) * (1.0 - soft_layer)


def pseudo_layer_target(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal optical-density split; used only as training supervision."""
    count = target.sum(1, keepdim=True).clamp_min(1.0)
    single_layer = 1.0 - torch.pow((1.0 - raw).clamp_min(1e-4), 1.0 / count)
    return single_layer * target


def image_gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = value[..., :, 1:] - value[..., :, :-1]
    gy = value[..., 1:, :] - value[..., :-1, :]
    return gx, gy


def projection_prior_loss(layers: torch.Tensor, raw: torch.Tensor,
                          target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    target = target.float()
    count = target.sum(1, keepdim=True)
    union = (count > 0).float()
    overlap = (count >= 2).float()
    pixel_weight = union + 2.0 * overlap
    pseudo = pseudo_layer_target(raw.float(), target)
    instance_weight = target * (1.0 + 2.0 * overlap)
    layer_l1 = (torch.abs(layers.float() - pseudo) * instance_weight).sum() / instance_weight.sum().clamp_min(1.0)

    reconstruction = transmission_reconstruction(layers.float(), raw.float())
    recon_l1 = (torch.abs(reconstruction - raw.float()) * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    rgx, rgy = image_gradient(reconstruction)
    tgx, tgy = image_gradient(raw.float())
    wx = torch.maximum(pixel_weight[..., :, 1:], pixel_weight[..., :, :-1])
    wy = torch.maximum(pixel_weight[..., 1:, :], pixel_weight[..., :-1, :])
    grad = ((torch.abs(rgx - tgx) * wx).sum() / wx.sum().clamp_min(1.0)
            + (torch.abs(rgy - tgy) * wy).sum() / wy.sum().clamp_min(1.0)) * 0.5
    loss = 0.5 * layer_l1 + 0.3 * recon_l1 + 0.2 * grad
    return loss, {"layer_l1": float(layer_l1.detach()), "recon_l1": float(recon_l1.detach()),
                  "gradient_l1": float(grad.detach()), "overlap_fraction": float(overlap.mean())}


def surfaces(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> tuple[float, float, bool]:
    if not target.any():
        return (1.0 if not pred.any() else 0.0), (0.0 if not pred.any() else float("nan")), bool(pred.any())
    if not pred.any():
        return 0.0, float("nan"), True
    ps = pred ^ binary_erosion(pred)
    ts = target ^ binary_erosion(target)
    dt = distance_transform_edt(~ts)
    dp = distance_transform_edt(~ps)
    denom = max(int(ps.sum() + ts.sum()), 1)
    nsd = float(((dt[ps] <= tolerance).sum() + (dp[ts] <= tolerance).sum()) / denom)
    distances = np.concatenate([dt[ps], dp[ts]])
    return nsd, float(distances.mean()), False


@torch.no_grad()
def validate(baseline: nn.Module, adapter: nn.Module | None, loader: DataLoader,
             device: torch.device) -> dict[str, float | list[float]]:
    baseline.eval()
    if adapter is not None:
        adapter.eval()
    inter = torch.zeros(14, device=device)
    pred_mass = torch.zeros(14, device=device)
    true_mass = torch.zeros(14, device=device)
    ov_inter = ov_pred = ov_true = 0.0
    overlap_nsd: list[float] = []
    overlap_msd: list[float] = []
    failures = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True) > 0.5
        base_logits, _ = baseline(image, False)
        logits = base_logits if adapter is None else adapter(image, base_logits)[0]
        prediction = torch.sigmoid(logits) >= 0.5
        inter += (prediction & target).sum((0, 2, 3))
        pred_mass += prediction.sum((0, 2, 3))
        true_mass += target.sum((0, 2, 3))
        po = prediction.sum(1) >= 2
        to = target.sum(1) >= 2
        ov_inter += float((po & to).sum())
        ov_pred += float(po.sum())
        ov_true += float(to.sum())
        for b in range(po.shape[0]):
            nsd, msd, failed = surfaces(po[b].cpu().numpy(), to[b].cpu().numpy())
            overlap_nsd.append(nsd)
            if np.isfinite(msd):
                overlap_msd.append(msd)
            failures += int(failed)
    class_dice = (2 * inter + 1) / (pred_mass + true_mass + 1)
    class_iou = (inter + 1) / (pred_mass + true_mass - inter + 1)
    overlap_dice = (2 * ov_inter + 1) / (ov_pred + ov_true + 1)
    overlap_iou = (ov_inter + 1) / (ov_pred + ov_true - ov_inter + 1)
    return {
        "macro_dsc": float(class_dice.mean()),
        "macro_iou": float(class_iou.mean()),
        "overlap_dsc": float(overlap_dice),
        "overlap_iou": float(overlap_iou),
        "overlap_voe": float(1.0 - overlap_iou),
        "overlap_nsd_2px": float(np.mean(overlap_nsd)),
        "overlap_msd_px": float(np.mean(overlap_msd)) if overlap_msd else float("nan"),
        "overlap_msd_fail_rate": float(failures / max(len(overlap_nsd), 1)),
        "class_dsc": class_dice.cpu().tolist(),
    }


def train_adapter(name: str, baseline: nn.Module, initial_adapter: dict, train_set: Dataset,
                  val_loader: DataLoader, device: torch.device, output: Path, epochs: int,
                  learning_rate: float, projection_weight: float, min_epochs: int,
                  patience: int, seed: int, batch_size: int, workers: int) -> tuple[Path, dict]:
    seed_all(seed)
    adapter = LayerProjectionAdapter().to(device)
    adapter.load_state_dict(copy.deepcopy(initial_adapter), strict=True)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    checkpoint = output / f"{name}_adapter_best.pth"
    history = output / f"{name}_history.jsonl"
    best: tuple[tuple[float, float, float], dict] | None = None
    bad = 0
    for epoch in range(1, epochs + 1):
        adapter.train()
        totals = {"loss": 0.0, "native": 0.0, "projection": 0.0, "layer_l1": 0.0,
                  "recon_l1": 0.0, "gradient_l1": 0.0, "overlap_fraction": 0.0}
        started = time.time()
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            raw = batch["raw"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.no_grad():
                base_logits, _ = baseline(image, False)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, layers = adapter(image, base_logits)
                native = dice_bce_logits(logits, target)
            prior, stats = projection_prior_loss(layers, raw, target)
            loss = native.float() + projection_weight * prior
            if not torch.isfinite(loss):
                raise RuntimeError({"arm": name, "epoch": epoch, "native": float(native), "prior": float(prior)})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["native"] += float(native.detach())
            totals["projection"] += float(prior.detach())
            for key, value in stats.items():
                totals[key] += value
        scheduler.step()
        metrics = validate(baseline, adapter, val_loader, device)
        row = {"arm": name, "epoch": epoch, "seconds": time.time() - started,
               "learning_rate": optimizer.param_groups[0]["lr"], "projection_weight": projection_weight,
               **{f"train_{k}": v / len(loader) for k, v in totals.items()}, **metrics}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = (metrics["macro_dsc"], metrics["overlap_nsd_2px"], metrics["overlap_dsc"])
        if best is None or key > best[0]:
            best = (key, row)
            bad = 0
            torch.save({"adapter": adapter.state_dict(), "row": row, "arm": name,
                        "projection_weight": projection_weight}, checkpoint)
        else:
            bad += 1
        if epoch >= min_epochs and bad >= patience:
            break
    assert best is not None
    return checkpoint, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ram_w600/r326_layer_projection_prior"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    # One RAM training-batch audit gave ||g_projection||/||g_native||=1263.
    # 2.5e-4 therefore starts at about 0.32x the native adapter gradient.
    parser.add_argument("--projection-weight", type=float, default=0.00025)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3261)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = WristLayerDataset(args.dataset_root, "train", args.size, True)
    val_set = WristLayerDataset(args.dataset_root, "val", args.size, False)
    if args.limit_train:
        train_set.mask_files = train_set.mask_files[:args.limit_train]
    if args.limit_val:
        val_set.mask_files = val_set.mask_files[:args.limit_val]
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(0, min(args.workers, 2)), pin_memory=True,
                            persistent_workers=args.workers > 0)

    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline = NnUNetMultiLabelPrior().to(device)
    baseline.load_state_dict(payload["model"], strict=True)
    baseline.eval()
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    adapter = LayerProjectionAdapter().to(device)
    initial_adapter = copy.deepcopy(adapter.state_dict())
    initialization = {"source": str(args.baseline_checkpoint), "sha256": sha256(args.baseline_checkpoint),
                      "strict_load": True, "baseline_frozen": True, "same_adapter_state": True}
    (args.output / "initialization_audit.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")

    if args.smoke:
        batch = next(iter(DataLoader(train_set, batch_size=1)))
        image = batch["image"].to(device)
        raw = batch["raw"].to(device)
        target = batch["mask"].to(device)
        with torch.no_grad():
            base_logits, _ = baseline(image, False)
        logits, layers = adapter(image, base_logits)
        native = dice_bce_logits(logits, target)
        prior, stats = projection_prior_loss(layers, raw, target)
        native_grad = torch.autograd.grad(native, adapter.parameters(), retain_graph=True, allow_unused=True)
        prior_grad = torch.autograd.grad(prior, adapter.parameters(), allow_unused=True)
        def norm(values: tuple[torch.Tensor | None, ...]) -> float:
            return float(torch.sqrt(sum((v.float() ** 2).sum() for v in values if v is not None)))
        ng, pg = norm(native_grad), norm(prior_grad)
        print(json.dumps({"smoke": True, "device": str(device), "native": float(native),
                          "projection": float(prior), "native_grad_norm": ng,
                          "projection_grad_norm": pg, "weighted_prior_to_native_grad":
                          args.projection_weight * pg / max(ng, 1e-12), **stats}, indent=2))
        return

    del adapter
    baseline_metrics = validate(baseline, None, val_loader, device)
    seg_ckpt, seg_best = train_adapter("seg_only", baseline, initial_adapter, train_set, val_loader,
                                       device, args.output, args.epochs, args.learning_rate, 0.0,
                                       args.min_epochs, args.patience, args.seed,
                                       args.batch_size, args.workers)
    projection_ckpt, projection_best = train_adapter(
        "layer_projection", baseline, initial_adapter, train_set, val_loader, device, args.output,
        args.epochs, args.learning_rate, args.projection_weight, args.min_epochs, args.patience,
        args.seed, args.batch_size, args.workers)

    def eval_adapter(path: Path) -> dict:
        module = LayerProjectionAdapter().to(device)
        module.load_state_dict(torch.load(path, map_location=device, weights_only=False)["adapter"], strict=True)
        return validate(baseline, module, val_loader, device)

    seg_metrics = eval_adapter(seg_ckpt)
    projection_metrics = eval_adapter(projection_ckpt)
    keys = ["macro_dsc", "macro_iou", "overlap_dsc", "overlap_iou", "overlap_voe",
            "overlap_nsd_2px", "overlap_msd_px", "overlap_msd_fail_rate"]
    result = {
        "experiment": "R326_RAM_NNUNET_LAYER_PROJECTION_PRIOR",
        "split": "validation", "test_used": False, "threshold": 0.5, "threshold_search": False,
        "paper_anchors": ["BLS-GAN (AAAI 2025)", "Layer Separation Networks (ACM MM 2025)"],
        "prior_position": "inference-time post-logit adapter",
        "pseudo_layer_rule": "equal optical-density split among GT memberships, train only",
        "inference_gt_dependency": False,
        "initialization": initialization,
        "baseline_val": baseline_metrics,
        "seg_only_val": seg_metrics,
        "layer_projection_val": projection_metrics,
        "delta_projection_minus_seg_only": {k: projection_metrics[k] - seg_metrics[k] for k in keys},
        "delta_projection_minus_baseline": {k: projection_metrics[k] - baseline_metrics[k] for k in keys},
        "best_training_rows": {"seg_only": seg_best, "layer_projection": projection_best},
        "checkpoints": {"seg_only_adapter": str(seg_ckpt), "layer_projection_adapter": str(projection_ckpt)},
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
