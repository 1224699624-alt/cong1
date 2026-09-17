#!/usr/bin/env python3
"""Paired RAM-W600 nnU-Net experiment for the frozen R317 seam prior.

Both arms start from the exact same mature R285 checkpoint. RAM remains a
14-channel sigmoid task; validation only is used for checkpoint selection.
"""
from __future__ import annotations

import argparse
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

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RamPriorDataset(Dataset):
    def __init__(self, root: Path, prior_root: Path, split: str, size: int,
                 augment: bool, limit: int = 0) -> None:
        self.root, self.prior_root, self.split, self.size, self.augment = root, prior_root, split, size, augment
        self.paths = sorted((root / "BoneSegmentation" / "masks" / split).glob("*.npy"))[:limit or None]
        if not self.paths:
            raise RuntimeError(f"No RAM masks: {split}")
        missing = [p.stem for p in self.paths if not (prior_root / split / f"{p.stem}.npy").is_file()]
        if missing:
            raise RuntimeError(f"Missing frozen priors for {split}: {missing[:5]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        mask_path = self.paths[index]
        image = Image.open(self.root / "BoneSegmentation" / "images" / f"{mask_path.stem}.bmp").convert("L")
        image_t = TF.pil_to_tensor(image).float() / 255.
        mask = torch.from_numpy((np.load(mask_path) > 0).astype(np.float32))
        prior = torch.from_numpy(np.load(self.prior_root / self.split / f"{mask_path.stem}.npy").astype(np.float32))[None]
        if mask_path.stem.endswith("_R"):
            image_t, mask = torch.flip(image_t, (-1,)), torch.flip(mask, (-1,))
        image_t = TF.resize(image_t, [self.size, self.size], antialias=True)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        if tuple(prior.shape[-2:]) != (self.size, self.size):
            prior = TF.resize(prior, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        if self.augment:
            angle, translate, scale = random.uniform(-7., 7.), [random.randint(-10, 10), random.randint(-10, 10)], random.uniform(.94, 1.06)
            image_t = TF.affine(image_t, angle, translate, scale, 0., InterpolationMode.BILINEAR, fill=0.)
            prior = TF.affine(prior, angle, translate, scale, 0., InterpolationMode.BILINEAR, fill=0.)
            mask = TF.affine(mask, angle, translate, scale, 0., InterpolationMode.NEAREST, fill=0.)
            image_t = TF.adjust_gamma(image_t.clamp(0, 1), random.uniform(.85, 1.15))
            image_t = (image_t * random.uniform(.90, 1.10)).clamp(0, 1)
        image_t = (image_t - image_t.mean()) / (image_t.std() + 1e-6)
        return {"data": torch.cat([image_t, prior], 0), "target": mask, "case": mask_path.stem}


class RamNnUNetTwoChannel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        self.backbone = PlainConvUNet(
            input_channels=2, n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv2d, kernel_sizes=[[3, 3]] * 6,
            strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
            n_conv_per_stage=[2] * 6, num_classes=14, n_conv_per_stage_decoder=[2] * 5,
            conv_bias=True, norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
            dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True}, deep_supervision=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.backbone(value)


def initialize_from_r285(model: RamNnUNetTwoChannel, checkpoint: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    source = {k: v for k, v in payload["model"].items() if k.startswith("backbone.")}
    target = model.state_dict()
    expanded_keys, truncated_keys = [], []
    for key, target_value in target.items():
        source_value = source[key]
        if source_value.shape == target_value.shape:
            continue
        if (source_value.ndim == 4 and target_value.ndim == 4
                and source_value.shape[1] == 1 and target_value.shape[1] == 2
                and source_value.shape[0] == target_value.shape[0]
                and source_value.shape[2:] == target_value.shape[2:]):
            expanded = torch.zeros_like(target_value)
            expanded[:, :1] = source_value
            source[key] = expanded
            expanded_keys.append(key)
        elif source_value.shape[0] == 15 and target_value.shape[0] == 14 and source_value.shape[1:] == target_value.shape[1:]:
            source[key] = source_value[:14].clone()
            truncated_keys.append(key)
        else:
            raise RuntimeError(f"Unsupported R285 adaptation {key}: {source_value.shape} -> {target_value.shape}")
    model.load_state_dict(source, strict=True)
    canonical_first = "backbone.encoder.stages.0.0.convs.0.conv.weight"
    return {"source": str(checkpoint), "sha256": sha256(checkpoint),
            "expanded_input_keys": expanded_keys, "truncated_output_keys": truncated_keys,
            "prior_channel_nonzero": int(torch.count_nonzero(model.state_dict()[canonical_first][:, 1]).item())}


def base_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dice = 1 - ((2 * (probability * target).sum((2, 3)) + 1)
                / (probability.sum((2, 3)) + target.sum((2, 3)) + 1)).mean()
    return dice + F.binary_cross_entropy_with_logits(logits, target)


def continuous_prior_loss(logits: torch.Tensor, target: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    prior = prior.float()
    low, high = prior.amin((2, 3), keepdim=True), prior.amax((2, 3), keepdim=True)
    weight = ((prior - low) / (high - low + 1e-6)).square() * (target == 0).float()
    numerator = (weight * F.softplus(logits.float())).sum((1, 2, 3))
    denominator = weight.sum((1, 2, 3))
    valid = denominator > 1e-6
    return (numerator[valid] / (denominator[valid] + 1e-6)).mean() if valid.any() else logits.sum() * 0


def surface_distances(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    if not pred.any() and not target.any(): return 1., 0., 0.
    if not pred.any() or not target.any(): return 0., float("nan"), float("nan")
    ps, ts = pred ^ binary_erosion(pred), target ^ binary_erosion(target)
    dt, dp = distance_transform_edt(~ts), distance_transform_edt(~ps)
    distances = np.concatenate([dt[ps], dp[ts]])
    nsd = float(((dt[ps] <= 2).sum() + (dp[ts] <= 2).sum()) / max(ps.sum() + ts.sum(), 1))
    return nsd, float(np.mean(distances)), float(np.percentile(distances, 95))


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, use_prior: bool,
             save_predictions: bool = False) -> tuple[dict, dict]:
    model.eval(); inter = torch.zeros(14, device=device); pm = torch.zeros(14, device=device); tm = torch.zeros(14, device=device)
    nsd, assd, hd95, prior_fp = [], [], [], []; predictions = {}
    for batch in loader:
        data, target = batch["data"].to(device), batch["target"].to(device)
        diagnostic_prior = data[:, 1].clone()
        if not use_prior: data = data.clone(); data[:, 1] = 0
        pred = torch.sigmoid(model(data)) >= .5; truth = target > .5
        inter += (pred & truth).sum((0, 2, 3)); pm += pred.sum((0, 2, 3)); tm += truth.sum((0, 2, 3))
        for b, case in enumerate(batch["case"]):
            p, t = pred[b].cpu().numpy(), truth[b].cpu().numpy()
            s = surface_distances(p.any(0), t.any(0)); nsd.append(s[0]); assd.append(s[1]); hd95.append(s[2])
            # The plain arm does not receive the prior, but both arms must be
            # evaluated on the same frozen high-confidence prior region.
            region = diagnostic_prior[b].cpu().numpy() >= .5
            prior_fp.append(float(np.logical_and(p.any(0), np.logical_and(region, ~t.any(0))).sum() / max(region.sum(), 1)))
            if save_predictions: predictions[str(case)] = p
    dice = (2 * inter + 1) / (pm + tm + 1); iou = (inter + 1) / (pm + tm - inter + 1)
    finite = lambda x: float(np.nanmean(np.asarray(x)))
    return {"macro_dice": float(dice.mean()), "macro_iou": float(iou.mean()),
            "union_nsd_2px": finite(nsd), "union_assd_px": finite(assd), "union_hd95_px": finite(hd95),
            "prior_region_fp_rate": finite(prior_fp), "class_dice": dice.cpu().tolist()}, predictions


def train_arm(name: str, model: RamNnUNetTwoChannel, train_set: Dataset, val_loader: DataLoader,
              device: torch.device, output: Path, epochs: int, alpha: float, use_prior: bool,
              seed: int, min_epochs: int, patience: int) -> tuple[Path, dict]:
    seed_all(seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history = output / f"{name}_history.jsonl"; best = None; checkpoint = output / f"{name}_best.pth"; bad = 0
    for epoch in range(1, epochs + 1):
        model.train(); totals = np.zeros(3); started = time.time()
        for batch in train_loader:
            data, target = batch["data"].to(device), batch["target"].to(device)
            if not use_prior: data = data.clone(); data[:, 1] = 0
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(data); native = base_loss(logits, target)
                prior = continuous_prior_loss(logits, target, data[:, 1:2]) if use_prior else native * 0
                loss = native + alpha * prior
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
            scaler.step(optimizer); scaler.update(); totals += [float(loss.detach()), float(native.detach()), float(prior.detach())]
        scheduler.step(); metrics, _ = validate(model, val_loader, device, use_prior)
        row = {"arm": name, "epoch": epoch, "seconds": time.time()-started, "loss": totals[0]/len(train_loader),
               "native_loss": totals[1]/len(train_loader), "prior_loss": totals[2]/len(train_loader), **metrics}
        with history.open("a", encoding="utf-8") as f: f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = (metrics["macro_dice"], metrics["macro_iou"], metrics["union_nsd_2px"], -metrics["union_assd_px"])
        if best is None or key > best[0]:
            best = (key, row); bad = 0; torch.save({"model": model.state_dict(), "row": row, "arm": name}, checkpoint)
        else: bad += 1
        if epoch >= min_epochs and bad >= patience: break
    return checkpoint, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\gutou\RAM-W600"))
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ram_w600/r322_nnunet_r317_seam"))
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--alpha", type=float, default=.035)
    parser.add_argument("--min-epochs", type=int, default=6); parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--experiment-id", default="R322_RAM_NNUNET_R317_SEAM")
    parser.add_argument("--size", type=int, default=384); parser.add_argument("--seed", type=int, default=322)
    parser.add_argument("--limit-train", type=int, default=0); parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True); seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = RamPriorDataset(args.dataset_root, args.prior_root, "train", args.size, True, args.limit_train)
    val_set = RamPriorDataset(args.dataset_root, args.prior_root, "val", args.size, False, args.limit_val)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    initial = RamNnUNetTwoChannel().to(device); init_audit = initialize_from_r285(initial, args.baseline_checkpoint, device)
    # Exact functional equality at initialization: old 1ch model vs new 2ch model with zero prior.
    old = NnUNetMultiLabelPrior().to(device); old.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)["model"]); old.eval(); initial.eval()
    sample = next(iter(val_loader))["data"].to(device)
    with torch.no_grad(): old_logits, _ = old(sample[:, :1], False); new_logits = initial(torch.cat([sample[:, :1], torch.zeros_like(sample[:, 1:2])], 1))
    init_audit["max_abs_logit_difference"] = float((old_logits-new_logits).abs().max())
    if init_audit["max_abs_logit_difference"] > 1e-5 or init_audit["prior_channel_nonzero"] != 0: raise RuntimeError(init_audit)
    (args.output / "initialization_audit.json").write_text(json.dumps(init_audit, indent=2), encoding="utf-8")
    if args.smoke:
        batch = next(iter(DataLoader(train_set, batch_size=1)))
        logits = initial(batch["data"].to(device)); loss = base_loss(logits, batch["target"].to(device)) + args.alpha * continuous_prior_loss(logits, batch["target"].to(device), batch["data"][:, 1:2].to(device)); loss.backward()
        print(json.dumps({"smoke": True, "loss": float(loss), "finite": bool(torch.isfinite(loss)), **init_audit})); return
    initial_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
    plain = RamNnUNetTwoChannel().to(device); plain.load_state_dict(initial_state)
    plain_ckpt, plain_best = train_arm("plain", plain, train_set, val_loader, device, args.output, args.epochs, 0., False, args.seed, args.min_epochs, args.patience)
    del plain; torch.cuda.empty_cache()
    prior = RamNnUNetTwoChannel().to(device); prior.load_state_dict(initial_state)
    prior_ckpt, prior_best = train_arm("r317_prior", prior, train_set, val_loader, device, args.output, args.epochs, args.alpha, True, args.seed, args.min_epochs, args.patience)
    prior.load_state_dict(torch.load(prior_ckpt, map_location=device, weights_only=False)["model"]); prior_metrics, _ = validate(prior, val_loader, device, True)
    plain = RamNnUNetTwoChannel().to(device); plain.load_state_dict(torch.load(plain_ckpt, map_location=device, weights_only=False)["model"]); plain_metrics, _ = validate(plain, val_loader, device, False)
    delta = {k: prior_metrics[k] - plain_metrics[k] for k in ("macro_dice", "macro_iou", "union_nsd_2px", "union_assd_px", "union_hd95_px", "prior_region_fp_rate")}
    result = {"experiment": args.experiment_id, "split": "validation", "test_used": False,
              "threshold": .5, "threshold_search": False, "alpha": args.alpha, "initialization": init_audit,
              "plain_best": plain_best, "prior_best": prior_best, "plain_val": plain_metrics,
              "r317_prior_val": prior_metrics, "delta_prior_minus_plain": delta}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
