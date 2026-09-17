#!/usr/bin/env python3
"""Train a compact context-aware direct segmenter for epiphysis masks."""

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
    parser = argparse.ArgumentParser(description="Train compact context segmenter.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.10)
    parser.add_argument("--gap-loss-weight", type=float, default=0.0)
    parser.add_argument("--gap-band-kernel", type=int, default=9)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-exp", default="r065_context_segmenter")
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r065_context_segmenter/best.pt")
    parser.add_argument("--history-json", default="outputs/context_segmenter/r065_context_segmenter/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r065_context_segmenter_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r065_context_segmenter_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260625)
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


class SegDataset(Dataset):
    def __init__(self, raw_root: Path, dataset: str, split: str, img_size: int, limit: int = 0, augment: bool = False) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.augment = augment
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
        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                mask = np.ascontiguousarray(np.fliplr(mask))
            if random.random() < 0.25:
                gamma = random.uniform(0.85, 1.20)
                image = np.clip(image ** gamma, 0.0, 1.0)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0.0, 0.015, size=image.shape).astype(np.float32), 0.0, 1.0)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "mask": torch.from_numpy(mask[None].astype(np.float32)),
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
    """Dependency-free long-range context block inspired by axial/SSM mixing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.row = nn.Conv1d(channels, channels, 9, padding=4, groups=channels, bias=False)
        self.col = nn.Conv1d(channels, channels, 9, padding=4, groups=channels, bias=False)
        hidden = max(8, channels // 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.local(x)
        row_ctx = y.mean(dim=2)
        col_ctx = y.mean(dim=3)
        row_ctx = self.row(row_ctx).unsqueeze(2).expand_as(y)
        col_ctx = self.col(col_ctx).unsqueeze(3).expand_as(y)
        mixed = y + row_ctx + col_ctx
        return x + self.proj(mixed * self.gate(mixed))


class ContextUNet(nn.Module):
    def __init__(self, base: int = 32) -> None:
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
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.context(e4)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def boundary_tensor(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def gap_band_tensor(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    k = max(3, int(kernel_size) | 1)
    pad = k // 2
    near_fg = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    return (near_fg * (1.0 - mask)).clamp(0.0, 1.0)


def loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float,
    gap_weight: float,
    gap_kernel: int,
) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=((1.0 - pos) / pos))
    loss = bce + dice_loss(logits, target)
    prob = torch.sigmoid(logits)
    if boundary_weight > 0:
        pred_boundary = boundary_tensor(prob)
        gt_boundary = boundary_tensor(target)
        loss = loss + boundary_weight * F.l1_loss(pred_boundary, gt_boundary)
    if gap_weight > 0:
        gap_band = gap_band_tensor(target, gap_kernel)
        if float(gap_band.sum().detach().cpu()) > 0:
            loss = loss + gap_weight * ((prob * gap_band).sum() / gap_band.sum().clamp_min(1.0))
    return loss


@torch.no_grad()
def predict(model: nn.Module, image: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    return torch.sigmoid(model(image.to(device))).cpu()


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count = 0
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) > 0:
            count += 1
    return count


def add_structure_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, threshold: float) -> dict[str, float]:
    records: list[dict[str, float]] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        probs = torch.sigmoid(model(images)).cpu().numpy()
        for i in range(probs.shape[0]):
            pred = probs[i, 0] >= threshold
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
) -> dict[str, object]:
    ds = SegDataset(raw_root, dataset, split, img_size, augment=False)
    records: list[dict[str, float]] = []
    for item in tqdm(ds, desc=f"infer/{dataset}/{split}"):
        image = item["image"].unsqueeze(0)
        prob = predict(model, image, device).numpy()[0, 0]
        shape = item["shape"]
        pred = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= threshold
        gt = read_mask(raw_root / dataset / f"{split}_labels" / str(item["name"]))
        write_mask(output_mask_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt > 0.5, boundary_kernel=3)
        rec = add_structure_metrics(rec, pred, gt > 0.5)
        rec["image"] = str(item["name"])
        records.append(rec)
    summary: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "threshold": threshold,
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
    train_ds = SegDataset(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.limit_train, augment=True)
    val_ds = SegDataset(Path(args.raw_root), args.dataset, args.val_split, args.img_size, args.limit_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ContextUNet(args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(images)
                loss = loss_fn(logits, masks, args.boundary_loss_weight, args.gap_loss_weight, args.gap_band_kernel)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        val_by_threshold = {thr: evaluate_loader(model, val_loader, args.device, thr) for thr in thresholds}
        chosen_thr, chosen_metrics = max(val_by_threshold.items(), key=lambda kv: kv[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(chosen_metrics["dice"]),
            "val_precision": float(chosen_metrics["precision"]),
            "val_recall": float(chosen_metrics["recall"]),
            "val_boundary_iou": float(chosen_metrics["boundary_iou"]),
            "threshold": float(chosen_thr),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_threshold = float(chosen_thr)
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best_threshold, "epoch": epoch}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best_threshold = float(state["threshold"])
    eval_mask_dir = Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks"
    control_mask_dir = Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks"
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        args.device,
        eval_mask_dir,
        Path(args.metrics_json),
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        args.device,
        control_mask_dir,
        Path(args.control_metrics_json),
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
