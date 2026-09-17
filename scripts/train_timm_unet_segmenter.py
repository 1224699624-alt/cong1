#!/usr/bin/env python3
"""Train a timm-encoder U-Net/FPN segmenter for epiphysis masks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train timm encoder segmentation model.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--encoder", default="convnext_tiny")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.08)
    parser.add_argument("--distance-loss-weight", type=float, default=0.0)
    parser.add_argument("--distance-sigma", type=float, default=3.0)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--min-component-areas", default="0")
    parser.add_argument("--strong-xray-aug", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output-exp", default="r097_timm_unet_convnext_tiny")
    parser.add_argument("--checkpoint", default="outputs/timm_unet/r097_timm_unet_convnext_tiny/best.pt")
    parser.add_argument("--history-json", default="outputs/timm_unet/r097_timm_unet_convnext_tiny/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r097_timm_unet_convnext_tiny_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r097_timm_unet_convnext_tiny_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_path(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found: {root}/{dataset}/{split}/{stem}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.float32)


def resize_float(arr: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=interpolation)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(mask.shape, dtype=bool)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == idx
    return keep


def soft_signed_distance_target(mask: np.ndarray, sigma: float) -> np.ndarray:
    mask_bool = mask > 0.5
    inside = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 5)
    signed = inside - outside
    signed = np.clip(signed / max(float(sigma), 1e-3), -20.0, 20.0)
    return (1.0 / (1.0 + np.exp(-signed))).astype(np.float32)


class SegDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dataset: str,
        split: str,
        img_size: int,
        limit: int = 0,
        augment: bool = False,
        strong_xray_aug: bool = False,
        distance_sigma: float = 3.0,
    ) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.strong_xray_aug = strong_xray_aug
        self.distance_sigma = distance_sigma
        self.labels = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
        if limit > 0:
            self.labels = self.labels[:limit]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        label_path = self.labels[idx]
        image = read_gray(find_image_path(self.raw_root, self.dataset, self.split, label_path.stem))
        mask = read_mask(label_path)
        original_shape = mask.shape
        image = resize_float(image, self.img_size, cv2.INTER_AREA)
        mask = resize_float(mask, self.img_size, cv2.INTER_NEAREST)
        distance_target = soft_signed_distance_target(mask, self.distance_sigma)
        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))
                distance_target = np.ascontiguousarray(np.fliplr(distance_target))
            if random.random() < 0.25:
                image = np.ascontiguousarray(np.flipud(image))
                mask = np.ascontiguousarray(np.flipud(mask))
                distance_target = np.ascontiguousarray(np.flipud(distance_target))
            if random.random() < 0.35:
                gamma = random.uniform(0.80, 1.25)
                image = np.clip(image**gamma, 0.0, 1.0)
            if random.random() < 0.35:
                image = np.clip(image + np.random.normal(0.0, 0.018, size=image.shape).astype(np.float32), 0.0, 1.0)
            if self.strong_xray_aug:
                if random.random() < 0.35:
                    angle = random.uniform(-7.0, 7.0)
                    scale = random.uniform(0.94, 1.06)
                    tx = random.uniform(-0.035, 0.035) * self.img_size
                    ty = random.uniform(-0.035, 0.035) * self.img_size
                    center = (self.img_size / 2.0, self.img_size / 2.0)
                    mat = cv2.getRotationMatrix2D(center, angle, scale)
                    mat[:, 2] += (tx, ty)
                    image = cv2.warpAffine(
                        image,
                        mat,
                        (self.img_size, self.img_size),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT_101,
                    )
                    mask = cv2.warpAffine(
                        mask,
                        mat,
                        (self.img_size, self.img_size),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    distance_target = soft_signed_distance_target(mask, self.distance_sigma)
                if random.random() < 0.25:
                    clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 3.0), tileGridSize=(8, 8))
                    image = clahe.apply((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)).astype(np.float32) / 255.0
                if random.random() < 0.20:
                    image = cv2.GaussianBlur(image, (3, 3), random.uniform(0.15, 0.65))
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
            "distance_target": torch.from_numpy(distance_target[None].astype(np.float32)),
            "name": label_path.name,
            "shape": original_shape,
        }


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimmUNet(nn.Module):
    def __init__(self, encoder: str, pretrained: bool, decoder_channels: int) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            encoder,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=1,
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(ch, decoder_channels, 1) for ch in channels])
        self.smooth = nn.ModuleList([ConvBNAct(decoder_channels, decoder_channels) for _ in channels])
        self.head = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )

    @staticmethod
    def _as_nchw(feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feat.ndim == 4 and feat.shape[1] == expected_channels:
            return feat
        if feat.ndim == 4 and feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats = [
            self._as_nchw(feat, expected_channels)
            for feat, expected_channels in zip(self.encoder(x), self.encoder.feature_info.channels(), strict=True)
        ]
        y = self.lateral[-1](feats[-1])
        y = self.smooth[-1](y)
        for idx in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[idx].shape[-2:], mode="bilinear", align_corners=False)
            y = y + self.lateral[idx](feats[idx])
            y = self.smooth[idx](y)
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        return self.head(y)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def boundary_tensor(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float,
    distance_target: torch.Tensor | None = None,
    distance_weight: float = 0.0,
) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=((1.0 - pos) / pos))
    loss = bce + dice_loss(logits, target)
    prob = torch.sigmoid(logits)
    if boundary_weight > 0:
        loss = loss + boundary_weight * F.l1_loss(boundary_tensor(prob), boundary_tensor(target))
    if distance_weight > 0 and distance_target is not None:
        boundary_focus = (4.0 * distance_target * (1.0 - distance_target)).detach()
        loss = loss + distance_weight * (F.smooth_l1_loss(prob, distance_target, reduction="none") * (0.25 + boundary_focus)).mean()
    return loss


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0)


def add_structure_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        import copy

        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = model.state_dict()
        for key, value in self.ema.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(source[key].detach(), alpha=1.0 - self.decay)
            else:
                value.copy_(source[key])


@torch.no_grad()
def predict_prob(model: nn.Module, image: torch.Tensor, tta_flips: bool) -> torch.Tensor:
    probs = torch.sigmoid(model(image))
    if not tta_flips:
        return probs
    probs = probs + torch.flip(torch.sigmoid(model(torch.flip(image, dims=[-1]))), dims=[-1])
    probs = probs + torch.flip(torch.sigmoid(model(torch.flip(image, dims=[-2]))), dims=[-2])
    flipped = torch.flip(image, dims=[-2, -1])
    probs = probs + torch.flip(torch.sigmoid(model(flipped)), dims=[-2, -1])
    return probs / 4.0


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
    min_component_area: int = 0,
    tta_flips: bool = False,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        probs = predict_prob(model, images, tta_flips).cpu().numpy()
        for i in range(probs.shape[0]):
            pred = remove_small_components(probs[i, 0] >= threshold, min_component_area)
            gt = masks[i, 0] > 0.5
            rec = compute_metrics(pred, gt, boundary_kernel=3)
            rec = add_structure_metrics(rec, pred, gt)
            rec["image"] = str(batch["name"][i])
            records.append(rec)
    return mean_metrics(records)


@torch.no_grad()
def infer_and_evaluate(
    model: nn.Module,
    raw_root: Path,
    dataset: str,
    split: str,
    img_size: int,
    threshold: float,
    device: str,
    output_mask_dir: Path,
    metrics_json: Path,
    min_component_area: int = 0,
    tta_flips: bool = False,
    limit: int = 0,
) -> dict[str, object]:
    ds = SegDataset(raw_root, dataset, split, img_size, limit=limit, augment=False)
    records: list[dict[str, float]] = []
    model.eval()
    for item in tqdm(ds, desc=f"infer/{dataset}/{split}"):
        prob = predict_prob(model, item["image"].unsqueeze(0).to(device), tta_flips).cpu().numpy()[0, 0]
        shape = item["shape"]
        pred = remove_small_components(
            cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= threshold,
            min_component_area,
        )
        gt = read_mask(raw_root / dataset / f"{split}_labels" / str(item["name"])) > 0.5
        write_mask(output_mask_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec = add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    summary: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "threshold": threshold,
        "min_component_area": min_component_area,
        "tta_flips": tta_flips,
        "num_evaluated": len(records),
        "mean": mean_metrics(records),
        "per_image": records,
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_ds = SegDataset(
        Path(args.raw_root),
        args.dataset,
        args.train_split,
        args.img_size,
        args.limit_train,
        augment=True,
        strong_xray_aug=args.strong_xray_aug,
        distance_sigma=args.distance_sigma,
    )
    val_ds = SegDataset(
        Path(args.raw_root),
        args.dataset,
        args.val_split,
        args.img_size,
        args.limit_val,
        augment=False,
        distance_sigma=args.distance_sigma,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = TimmUNet(args.encoder, args.pretrained, args.decoder_channels).to(args.device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    min_component_areas = [int(float(x)) for x in args.min_component_areas.split(",") if x.strip()]
    history: list[dict[str, float | int | str | bool]] = []
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_min_component_area = 0
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"train/epoch{epoch}"), start=1):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(images)
                distance_targets = batch["distance_target"].to(args.device)
                loss = loss_fn(
                    logits,
                    masks,
                    args.boundary_loss_weight,
                    distance_targets,
                    args.distance_loss_weight,
                )
                loss = loss / max(args.grad_accum_steps, 1)
            scaler.scale(loss).backward()
            if step % max(args.grad_accum_steps, 1) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            losses.append(float(loss.item() * max(args.grad_accum_steps, 1)))
        if len(train_loader) % max(args.grad_accum_steps, 1) != 0:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        eval_model = ema.ema if ema is not None else model
        val_by_config = {
            (thr, area): evaluate_loader(eval_model, val_loader, args.device, thr, area, args.tta_flips)
            for thr in thresholds
            for area in min_component_areas
        }
        (chosen_thr, chosen_area), chosen_metrics = max(val_by_config.items(), key=lambda kv: kv[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(chosen_metrics["dice"]),
            "val_precision": float(chosen_metrics["precision"]),
            "val_recall": float(chosen_metrics["recall"]),
            "val_boundary_iou": float(chosen_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(chosen_metrics["false_bridge_flag"]),
            "threshold": float(chosen_thr),
            "min_component_area": int(chosen_area),
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
            "ema_decay": float(args.ema_decay),
            "strong_xray_aug": bool(args.strong_xray_aug),
            "tta_flips": bool(args.tta_flips),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_threshold = float(chosen_thr)
            best_min_component_area = int(chosen_area)
            torch.save({
                "model": eval_model.state_dict(),
                "args": vars(args),
                "threshold": best_threshold,
                "min_component_area": best_min_component_area,
                "epoch": epoch,
            }, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best_threshold = float(state["threshold"])
    best_min_component_area = int(state.get("min_component_area", 0))
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        args.device,
        Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
        best_min_component_area,
        args.tta_flips,
        args.limit_eval,
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        args.device,
        Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
        best_min_component_area,
        args.tta_flips,
        args.limit_eval,
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "min_component_area": best_min_component_area,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
