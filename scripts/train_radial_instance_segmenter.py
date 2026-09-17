#!/usr/bin/env python3
"""Train a StarDist-inspired radial instance segmenter for epiphysis masks."""

from __future__ import annotations

import argparse
import json
import math
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
    parser = argparse.ArgumentParser(description="Train radial instance segmenter.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--min-epochs", type=int, default=14)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=28)
    parser.add_argument("--num-rays", type=int, default=16)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--center-loss-weight", type=float, default=0.45)
    parser.add_argument("--radial-loss-weight", type=float, default=0.35)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.06)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--center-thresholds", default="0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--shape-weights", default="0.25,0.40,0.55,0.70")
    parser.add_argument("--min-center-distance", type=int, default=13)
    parser.add_argument("--max-instances", type=int, default=32)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-exp", default="r071_radial_instance_segmenter")
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r071_radial_instance_segmenter/best.pt")
    parser.add_argument("--history-json", default="outputs/context_segmenter/r071_radial_instance_segmenter/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r071_radial_instance_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r071_radial_instance_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260626)
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


def read_instance_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def resize_float(arr: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=interpolation)


def resize_instance(arr: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return (dilated - eroded).astype(np.float32)


def ray_distances(component: np.ndarray, cy: float, cx: float, num_rays: int, max_radius: float) -> np.ndarray:
    distances = np.zeros((num_rays,), dtype=np.float32)
    height, width = component.shape
    for ray_idx in range(num_rays):
        theta = 2.0 * math.pi * float(ray_idx) / float(num_rays)
        dy = math.sin(theta)
        dx = math.cos(theta)
        last = 0.0
        for step in np.arange(0.5, max_radius + 1.0, 0.5, dtype=np.float32):
            yy = int(round(cy + float(dy * step)))
            xx = int(round(cx + float(dx * step)))
            if yy < 0 or yy >= height or xx < 0 or xx >= width or not component[yy, xx]:
                break
            last = float(step)
        distances[ray_idx] = last / max_radius
    return distances


def radial_targets(instance: np.ndarray, num_rays: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binary = (instance > 0).astype(np.float32)
    center = np.zeros_like(binary, dtype=np.float32)
    radial = np.zeros((num_rays, *binary.shape), dtype=np.float32)
    radial_weight = np.zeros_like(binary, dtype=np.float32)
    max_radius = float(max(instance.shape))

    for value in [int(v) for v in np.unique(instance) if int(v) > 0]:
        component = instance == value
        if int(component.sum()) < 4:
            continue
        ys, xs = np.where(component)
        cy = float(ys.mean())
        cx = float(xs.mean())
        sigma = max(2.0, 0.12 * math.sqrt(float(component.sum())))
        y0 = np.arange(instance.shape[0], dtype=np.float32)[:, None]
        x0 = np.arange(instance.shape[1], dtype=np.float32)[None, :]
        heat = np.exp(-((y0 - cy) ** 2 + (x0 - cx) ** 2) / (2.0 * sigma * sigma)).astype(np.float32)
        center = np.maximum(center, heat * component.astype(np.float32))

        distances = ray_distances(component, cy, cx, num_rays, max_radius)
        dist_transform = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        core = component & (dist_transform >= max(1.0, 0.45 * float(dist_transform.max())))
        if int(core.sum()) == 0:
            core = component
        radial[:, core] = distances[:, None]
        radial_weight[core] = 1.0

    return binary, center, radial, radial_weight


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


class RadialDataset(Dataset):
    def __init__(self, raw_root: Path, dataset: str, split: str, img_size: int, num_rays: int, limit: int = 0, augment: bool = False) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.num_rays = num_rays
        self.augment = augment
        self.labels = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
        if limit > 0:
            self.labels = self.labels[:limit]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        label_path = self.labels[idx]
        image = read_gray(find_image_path(self.raw_root, self.dataset, self.split, label_path.stem))
        instance = read_instance_mask(label_path)
        original_shape = instance.shape
        image = resize_float(image, self.img_size, cv2.INTER_AREA)
        instance = resize_instance(instance, self.img_size)
        binary, center, radial, radial_weight = radial_targets(instance, self.num_rays)
        boundary = boundary_map(instance)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                binary = np.ascontiguousarray(np.fliplr(binary))
                center = np.ascontiguousarray(np.fliplr(center))
                boundary = np.ascontiguousarray(np.fliplr(boundary))
                radial_weight = np.ascontiguousarray(np.fliplr(radial_weight))
                radial = np.ascontiguousarray(np.flip(radial, axis=2))
                radial = np.roll(radial, shift=self.num_rays // 2, axis=0)
            if random.random() < 0.25:
                gamma = random.uniform(0.85, 1.20)
                image = np.clip(image ** gamma, 0.0, 1.0)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0.0, 0.015, size=image.shape).astype(np.float32), 0.0, 1.0)

        target = np.concatenate([
            binary[None],
            center[None],
            radial,
            radial_weight[None],
            boundary[None],
        ], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "target": torch.from_numpy(target),
            "name": label_path.name,
            "shape": original_shape,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContextBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.row = nn.Conv1d(channels, channels, 9, padding=4, groups=channels, bias=False)
        self.col = nn.Conv1d(channels, channels, 9, padding=4, groups=channels, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(x)
        row_ctx = self.row(y.mean(dim=2)).unsqueeze(2).expand_as(y)
        col_ctx = self.col(y.mean(dim=3)).unsqueeze(3).expand_as(y)
        return x + self.proj(y + row_ctx + col_ctx)


class RadialUNet(nn.Module):
    def __init__(self, base: int = 28, num_rays: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.context = nn.Sequential(ContextBlock(base * 8), ContextBlock(base * 8))
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.mask_head = nn.Conv2d(base, 1, 1)
        self.center_head = nn.Conv2d(base, 1, 1)
        self.radial_head = nn.Conv2d(base, num_rays, 1)
        self.boundary_head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.context(e4)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {
            "mask": self.mask_head(d1),
            "center": self.center_head(d1),
            "radial": self.radial_head(d1),
            "boundary": self.boundary_head(d1),
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def weighted_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=((1.0 - pos) / pos))


def train_loss(outputs: dict[str, torch.Tensor], target: torch.Tensor, num_rays: int, args: argparse.Namespace) -> torch.Tensor:
    binary = target[:, 0:1]
    center = target[:, 1:2]
    radial = target[:, 2:2 + num_rays]
    radial_weight = target[:, 2 + num_rays:3 + num_rays]
    boundary = target[:, 3 + num_rays:4 + num_rays]
    loss = weighted_bce(outputs["mask"], binary) + dice_loss(outputs["mask"], binary)
    loss = loss + args.center_loss_weight * F.mse_loss(torch.sigmoid(outputs["center"]), center)
    loss = loss + args.boundary_loss_weight * (weighted_bce(outputs["boundary"], boundary) + dice_loss(outputs["boundary"], boundary))
    pred_radial = torch.sigmoid(outputs["radial"])
    weight = radial_weight.expand_as(pred_radial)
    denom = weight.sum().clamp_min(1.0)
    loss = loss + args.radial_loss_weight * (F.smooth_l1_loss(pred_radial * weight, radial * weight, reduction="sum") / denom)
    return loss


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


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def select_centers(center_prob: np.ndarray, mask_prob: np.ndarray, center_threshold: float, min_distance: int, max_instances: int) -> list[tuple[int, int]]:
    pooled = cv2.dilate(center_prob, np.ones((max(3, min_distance), max(3, min_distance)), np.uint8))
    peaks = (center_prob >= pooled - 1e-6) & (center_prob >= center_threshold) & (mask_prob >= 0.20)
    ys, xs = np.where(peaks)
    order = np.argsort(center_prob[ys, xs])[::-1]
    centers: list[tuple[int, int]] = []
    min_dist_sq = float(min_distance * min_distance)
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        if all((y - py) ** 2 + (x - px) ** 2 >= min_dist_sq for py, px in centers):
            centers.append((y, x))
        if len(centers) >= max_instances:
            break
    return centers


def radial_shape_mask(center_prob: np.ndarray, radial_prob: np.ndarray, mask_prob: np.ndarray, center_threshold: float, min_distance: int, max_instances: int) -> np.ndarray:
    height, width = center_prob.shape
    max_radius = float(max(height, width))
    output = np.zeros((height, width), dtype=np.uint8)
    centers = select_centers(center_prob, mask_prob, center_threshold, min_distance, max_instances)
    num_rays = radial_prob.shape[0]
    for y, x in centers:
        radii = np.clip(radial_prob[:, y, x] * max_radius, 1.0, max_radius)
        pts = []
        for ray_idx, radius in enumerate(radii):
            theta = 2.0 * math.pi * float(ray_idx) / float(num_rays)
            yy = int(round(y + math.sin(theta) * float(radius)))
            xx = int(round(x + math.cos(theta) * float(radius)))
            pts.append([np.clip(xx, 0, width - 1), np.clip(yy, 0, height - 1)])
        cv2.fillPoly(output, [np.asarray(pts, dtype=np.int32)], 1)
    return output.astype(bool)


def decode_prediction(
    outputs: dict[str, torch.Tensor],
    threshold: float,
    center_threshold: float,
    shape_weight: float,
    min_distance: int,
    max_instances: int,
) -> np.ndarray:
    mask_prob = torch.sigmoid(outputs["mask"]).cpu().numpy()[0, 0]
    center_prob = torch.sigmoid(outputs["center"]).cpu().numpy()[0, 0]
    radial_prob = torch.sigmoid(outputs["radial"]).cpu().numpy()[0]
    shape = radial_shape_mask(center_prob, radial_prob, mask_prob, center_threshold, min_distance, max_instances)
    mixed = ((1.0 - shape_weight) * mask_prob) + (shape_weight * shape.astype(np.float32))
    return mixed >= threshold


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, args: argparse.Namespace, threshold: float, center_threshold: float, shape_weight: float) -> dict[str, float]:
    records: list[dict[str, float]] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(args.device)
        targets = batch["target"].numpy()
        outputs = model(images)
        for i in range(images.shape[0]):
            one = {key: value[i:i + 1] for key, value in outputs.items()}
            pred = decode_prediction(one, threshold, center_threshold, shape_weight, args.min_center_distance, args.max_instances)
            gt = targets[i, 0] > 0.5
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
    args: argparse.Namespace,
    threshold: float,
    center_threshold: float,
    shape_weight: float,
    output_mask_dir: Path,
    metrics_json: Path,
) -> dict[str, object]:
    ds = RadialDataset(raw_root, dataset, split, args.img_size, args.num_rays, augment=False)
    records: list[dict[str, float]] = []
    model.eval()
    for item in tqdm(ds, desc=f"infer/{dataset}/{split}"):
        image = item["image"].unsqueeze(0).to(args.device)
        outputs = model(image)
        pred_small = decode_prediction(outputs, threshold, center_threshold, shape_weight, args.min_center_distance, args.max_instances)
        shape = item["shape"]
        pred = cv2.resize(pred_small.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        gt = read_instance_mask(raw_root / dataset / f"{split}_labels" / str(item["name"])) > 0
        write_mask(output_mask_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec = add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    summary: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "threshold": threshold,
        "center_threshold": center_threshold,
        "shape_weight": shape_weight,
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
    train_ds = RadialDataset(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.num_rays, args.limit_train, True)
    val_ds = RadialDataset(Path(args.raw_root), args.dataset, args.val_split, args.img_size, args.num_rays, args.limit_val, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = RadialUNet(args.base_channels, args.num_rays).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    center_thresholds = [float(x) for x in args.center_thresholds.split(",") if x.strip()]
    shape_weights = [float(x) for x in args.shape_weights.split(",") if x.strip()]
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    best_cfg = (0.5, 0.3, 0.4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            images = batch["image"].to(args.device)
            targets = batch["target"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                outputs = model(images)
                loss = train_loss(outputs, targets, args.num_rays, args)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        val_grid = {}
        for thr in thresholds:
            for cthr in center_thresholds:
                for sw in shape_weights:
                    val_grid[(thr, cthr, sw)] = evaluate_loader(model, val_loader, args, thr, cthr, sw)
        best_grid_cfg, chosen_metrics = max(val_grid.items(), key=lambda kv: kv[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(chosen_metrics["dice"]),
            "val_precision": float(chosen_metrics["precision"]),
            "val_recall": float(chosen_metrics["recall"]),
            "val_boundary_iou": float(chosen_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(chosen_metrics["false_bridge_flag"]),
            "threshold": float(best_grid_cfg[0]),
            "center_threshold": float(best_grid_cfg[1]),
            "shape_weight": float(best_grid_cfg[2]),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_cfg = best_grid_cfg
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "cfg": best_cfg}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    threshold, center_threshold, shape_weight = [float(x) for x in state["cfg"]]
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args,
        threshold,
        center_threshold,
        shape_weight,
        Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args,
        threshold,
        center_threshold,
        shape_weight,
        Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": threshold,
        "center_threshold": center_threshold,
        "shape_weight": shape_weight,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
