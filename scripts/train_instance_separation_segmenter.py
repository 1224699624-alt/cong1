#!/usr/bin/env python3
"""Train a direct segmenter with instance-separation supervision.

The model predicts a binary epiphysis mask plus auxiliary separation and
HoVer-style offset maps derived from instance-valued labels. At inference,
the separation head suppresses predicted foreground in between adjacent
instances, targeting the false-bridge failure mode seen in R065.
"""

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
    parser = argparse.ArgumentParser(description="Train instance-separation direct segmenter.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.08)
    parser.add_argument("--sep-loss-weight", type=float, default=0.35)
    parser.add_argument("--hover-loss-weight", type=float, default=0.20)
    parser.add_argument("--sep-band-kernel", type=int, default=11)
    parser.add_argument("--sep-suppress-weights", default="0.00,0.20,0.35,0.50")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-exp", default="r068_instance_separation_segmenter")
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r068_instance_separation_segmenter/best.pt")
    parser.add_argument("--history-json", default="outputs/context_segmenter/r068_instance_separation_segmenter/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r068_instance_separation_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r068_instance_separation_original_test_metrics.json")
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


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def boundary_map(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    dilated = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return (dilated - eroded).astype(np.float32)


def instance_targets(instance: np.ndarray, sep_kernel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    binary = (instance > 0).astype(np.float32)
    core = np.zeros_like(binary, dtype=np.float32)
    sep = np.zeros_like(binary, dtype=np.float32)
    hover_x = np.zeros_like(binary, dtype=np.float32)
    hover_y = np.zeros_like(binary, dtype=np.float32)
    dilated_sum = np.zeros_like(binary, dtype=np.float32)
    kernel = np.ones((max(3, int(sep_kernel) | 1), max(3, int(sep_kernel) | 1)), np.uint8)

    for value in [int(v) for v in np.unique(instance) if int(v) > 0]:
        component = instance == value
        if int(component.sum()) == 0:
            continue
        ys, xs = np.where(component)
        cy = float(ys.mean())
        cx = float(xs.mean())
        h = max(1.0, float(ys.max() - ys.min() + 1))
        w = max(1.0, float(xs.max() - xs.min() + 1))
        hover_x[component] = ((xs.astype(np.float32) - cx) / (0.5 * w)).clip(-1.0, 1.0)
        hover_y[component] = ((ys.astype(np.float32) - cy) / (0.5 * h)).clip(-1.0, 1.0)

        dist = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
        if float(dist.max()) > 0:
            core[(dist >= max(1.5, 0.35 * float(dist.max()))) & component] = 1.0
        dilated_sum += cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(np.float32)

    sep[(dilated_sum >= 2.0) & (binary <= 0.5)] = 1.0
    return binary, core, sep, hover_x.astype(np.float32), hover_y.astype(np.float32)


class InstanceSegDataset(Dataset):
    def __init__(self, raw_root: Path, dataset: str, split: str, img_size: int, limit: int = 0, augment: bool = False, sep_kernel: int = 11) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.sep_kernel = sep_kernel
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
        binary, core, sep, hover_x, hover_y = instance_targets(instance, self.sep_kernel)
        boundary = boundary_map(instance)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                binary = np.ascontiguousarray(np.fliplr(binary))
                core = np.ascontiguousarray(np.fliplr(core))
                sep = np.ascontiguousarray(np.fliplr(sep))
                boundary = np.ascontiguousarray(np.fliplr(boundary))
                hover_x = np.ascontiguousarray(-np.fliplr(hover_x))
                hover_y = np.ascontiguousarray(np.fliplr(hover_y))
            if random.random() < 0.25:
                gamma = random.uniform(0.85, 1.20)
                image = np.clip(image ** gamma, 0.0, 1.0)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0.0, 0.015, size=image.shape).astype(np.float32), 0.0, 1.0)

        targets = np.stack([binary, core, sep, hover_x, hover_y, boundary], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "target": torch.from_numpy(targets),
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
        row_ctx = self.row(y.mean(dim=2)).unsqueeze(2).expand_as(y)
        col_ctx = self.col(y.mean(dim=3)).unsqueeze(3).expand_as(y)
        mixed = y + row_ctx + col_ctx
        return x + self.proj(mixed * self.gate(mixed))


class InstanceSeparationUNet(nn.Module):
    def __init__(self, base: int = 24, in_channels: int = 1) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
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
        self.core_head = nn.Conv2d(base, 1, 1)
        self.sep_head = nn.Conv2d(base, 1, 1)
        self.hover_head = nn.Conv2d(base, 2, 1)
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
            "core": self.core_head(d1),
            "sep": self.sep_head(d1),
            "hover": self.hover_head(d1),
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


def train_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    boundary_weight: float,
    sep_weight: float,
    hover_weight: float,
) -> torch.Tensor:
    binary = target[:, 0:1]
    core = target[:, 1:2]
    sep = target[:, 2:3]
    hover = target[:, 3:5]
    boundary = target[:, 5:6]
    loss = weighted_bce(outputs["mask"], binary) + dice_loss(outputs["mask"], binary)
    loss = loss + 0.25 * (weighted_bce(outputs["core"], core) + dice_loss(outputs["core"], core))
    if boundary_weight > 0:
        loss = loss + boundary_weight * (weighted_bce(outputs["boundary"], boundary) + dice_loss(outputs["boundary"], boundary))
    if sep_weight > 0:
        loss = loss + sep_weight * (weighted_bce(outputs["sep"], sep) + dice_loss(outputs["sep"], sep))
    if hover_weight > 0:
        fg = binary.expand_as(hover)
        denom = fg.sum().clamp_min(1.0)
        loss = loss + hover_weight * (F.smooth_l1_loss(torch.tanh(outputs["hover"]) * fg, hover * fg, reduction="sum") / denom)
    return loss


def apply_sep_suppression(outputs: dict[str, torch.Tensor], suppress_weight: float) -> torch.Tensor:
    mask_prob = torch.sigmoid(outputs["mask"])
    sep_prob = torch.sigmoid(outputs["sep"])
    core_prob = torch.sigmoid(outputs["core"])
    suppressed = mask_prob * (1.0 - float(suppress_weight) * sep_prob)
    return torch.maximum(suppressed, 0.75 * core_prob)


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


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, threshold: float, suppress_weight: float) -> dict[str, float]:
    records: list[dict[str, float]] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        binary = batch["target"][:, 0:1].numpy()
        outputs = model(images)
        probs = apply_sep_suppression(outputs, suppress_weight).cpu().numpy()
        for i in range(probs.shape[0]):
            pred = probs[i, 0] >= threshold
            gt = binary[i, 0] > 0.5
            rec = compute_metrics(pred, gt, boundary_kernel=3)
            rec = add_structure_metrics(rec, pred, gt)
            rec["image"] = str(batch["name"][i])
            records.append(rec)
    return mean_metrics(records)


@torch.no_grad()
def predict_prob(model: nn.Module, image: torch.Tensor, device: str, suppress_weight: float) -> torch.Tensor:
    model.eval()
    return apply_sep_suppression(model(image.to(device)), suppress_weight).cpu()


@torch.no_grad()
def infer_and_evaluate(
    model: nn.Module,
    raw_root: Path,
    dataset: str,
    split: str,
    img_size: int,
    threshold: float,
    suppress_weight: float,
    device: str,
    output_mask_dir: Path,
    metrics_json: Path,
    sep_kernel: int,
) -> dict[str, object]:
    ds = InstanceSegDataset(raw_root, dataset, split, img_size, augment=False, sep_kernel=sep_kernel)
    records: list[dict[str, float]] = []
    for item in tqdm(ds, desc=f"infer/{dataset}/{split}"):
        image = item["image"].unsqueeze(0)
        prob = predict_prob(model, image, device, suppress_weight).numpy()[0, 0]
        shape = item["shape"]
        pred = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= threshold
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
        "sep_suppress_weight": suppress_weight,
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
    train_ds = InstanceSegDataset(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.limit_train, True, args.sep_band_kernel)
    val_ds = InstanceSegDataset(Path(args.raw_root), args.dataset, args.val_split, args.img_size, args.limit_val, False, args.sep_band_kernel)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = InstanceSeparationUNet(args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    suppress_weights = [float(x) for x in args.sep_suppress_weights.split(",") if x.strip()]
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_suppress = 0.0
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            images = batch["image"].to(args.device)
            targets = batch["target"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                outputs = model(images)
                loss = train_loss(outputs, targets, args.boundary_loss_weight, args.sep_loss_weight, args.hover_loss_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        val_grid = {}
        for thr in thresholds:
            for sup in suppress_weights:
                val_grid[(thr, sup)] = evaluate_loader(model, val_loader, args.device, thr, sup)
        (chosen_thr, chosen_sup), chosen_metrics = max(val_grid.items(), key=lambda kv: kv[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(chosen_metrics["dice"]),
            "val_precision": float(chosen_metrics["precision"]),
            "val_recall": float(chosen_metrics["recall"]),
            "val_boundary_iou": float(chosen_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(chosen_metrics["false_bridge_flag"]),
            "threshold": float(chosen_thr),
            "sep_suppress_weight": float(chosen_sup),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_threshold = float(chosen_thr)
            best_suppress = float(chosen_sup)
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best_threshold, "sep_suppress_weight": best_suppress, "epoch": epoch}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best_threshold = float(state["threshold"])
    best_suppress = float(state.get("sep_suppress_weight", 0.0))
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        best_suppress,
        args.device,
        Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
        args.sep_band_kernel,
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        best_suppress,
        args.device,
        Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
        args.sep_band_kernel,
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "sep_suppress_weight": best_suppress,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
