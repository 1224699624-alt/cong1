#!/usr/bin/env python3
"""Train an ROI instance segmenter from proposal boxes.

R093-style candidate generator: learn to redraw an instance/ROI from image +
proposal context, then apply it to connected components from a strong anchor
mask. This creates new full-contour candidates instead of editing only narrow
candidate-disagreement pixels.
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
from train_anchor_pixel_residual import find_mask, image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ROI proposal instance segmenter.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--proposal-exp", required=True, help="Proposal mask experiment for train/val.")
    parser.add_argument("--apply-proposal-exp", default=None, help="Proposal mask experiment for clean-test-v2.")
    parser.add_argument("--control-proposal-exp", default=None, help="Proposal mask experiment for original test control.")
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--control-metrics-json", required=True)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--box-padding", type=float, default=0.18)
    parser.add_argument("--box-jitter", type=float, default=0.08)
    parser.add_argument("--min-proposal-area", type=int, default=12)
    parser.add_argument("--max-proposals-per-image", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.12)
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--blend-modes", default="roi,union,intersect", help="roi, union, intersect.")
    parser.add_argument("--limit-train-images", type=int, default=0)
    parser.add_argument("--limit-tune-images", type=int, default=0)
    parser.add_argument("--limit-apply-images", type=int, default=0)
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260693)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def instance_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


def boundary_tensor(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, 3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor, boundary_weight: float) -> torch.Tensor:
    pos = target.mean().clamp(1e-4, 0.5)
    pos_weight = ((1.0 - pos) / pos).detach()
    bce_map = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none")
    weight = 1.0 + float(boundary_weight) * boundary_tensor(target)
    bce = (bce_map * weight).sum() / weight.sum().clamp_min(1.0)
    loss = bce + dice_loss(logits, target)
    return loss


def expand_box(box: tuple[int, int, int, int], shape: tuple[int, int], pad: float, jitter: float = 0.0) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    if jitter > 0:
        x0 += random.uniform(-bw * jitter, bw * jitter)
        x1 += random.uniform(-bw * jitter, bw * jitter)
        y0 += random.uniform(-bh * jitter, bh * jitter)
        y1 += random.uniform(-bh * jitter, bh * jitter)
    return (
        int(np.clip(np.floor(x0), 0, w - 1)),
        int(np.clip(np.floor(y0), 0, h - 1)),
        int(np.clip(np.ceil(x1), 0, w - 1)),
        int(np.clip(np.ceil(y1), 0, h - 1)),
    )


def component_boxes(mask: np.ndarray, min_area: int, max_count: int) -> list[tuple[int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w - 1, y + h - 1))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes[:max_count] if max_count > 0 else boxes


def instance_boxes(inst: np.ndarray, min_area: int, max_count: int) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    for value in [int(v) for v in np.unique(inst) if int(v) > 0]:
        ys, xs = np.where(inst == value)
        if xs.size < min_area:
            continue
        out.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), value))
    out.sort(key=lambda b: (b[1], b[0]))
    return out[:max_count] if max_count > 0 else out


def crop_resize(arr: np.ndarray, box: tuple[int, int, int, int], size: int, interpolation: int) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = arr[y0 : y1 + 1, x0 : x1 + 1]
    return cv2.resize(crop.astype(np.float32), (size, size), interpolation=interpolation).astype(np.float32)


def make_coord_channels(size: int) -> tuple[np.ndarray, np.ndarray]:
    vals = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(vals, vals, indexing="ij")
    return xx, yy


class RoiProposalDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, limit_images: int = 0, augment: bool = False) -> None:
        self.args = args
        self.split = split
        self.augment = augment
        self.roots = [Path(p) for p in args.ablations_roots]
        self.raw_root = Path(args.train_raw_root)
        self.items: list[tuple[str, tuple[int, int, int, int], int | None]] = []
        split_names = names(self.raw_root, args.train_dataset, split)
        if limit_images > 0:
            split_names = split_names[:limit_images]
        for name in split_names:
            inst = instance_mask(self.raw_root / args.train_dataset / f"{split}_labels" / name)
            boxes = instance_boxes(inst, args.min_proposal_area, args.max_proposals_per_image)
            for x0, y0, x1, y1, value in boxes:
                self.items.append((name, (x0, y0, x1, y1), value))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        name, base_box, value = self.items[index]
        image = read_gray(image_path(self.raw_root, self.args.train_dataset, self.split, name))
        inst = instance_mask(self.raw_root / self.args.train_dataset / f"{self.split}_labels" / name)
        target = (inst == int(value)).astype(np.float32)
        box = expand_box(base_box, target.shape, self.args.box_padding, self.args.box_jitter if self.augment else 0.0)
        image_crop = crop_resize(image, box, self.args.roi_size, cv2.INTER_AREA)
        target_crop = crop_resize(target, box, self.args.roi_size, cv2.INTER_NEAREST)
        box_prior = np.ones_like(image_crop, dtype=np.float32)
        xx, yy = make_coord_channels(self.args.roi_size)
        if self.augment and random.random() < 0.5:
            image_crop = np.ascontiguousarray(np.fliplr(image_crop))
            target_crop = np.ascontiguousarray(np.fliplr(target_crop))
            xx = np.ascontiguousarray(np.fliplr(xx))
            yy = np.ascontiguousarray(np.fliplr(yy))
        if self.augment and random.random() < 0.25:
            image_crop = np.clip(image_crop ** random.uniform(0.85, 1.20), 0.0, 1.0)
        x = np.stack([image_crop, box_prior, xx, yy], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(target_crop[None].astype(np.float32))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RoiUNet(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.enc1 = ConvBlock(4, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(
            ConvBlock(base * 8, base * 8),
            nn.Conv2d(base * 8, base * 8, 3, padding=2, dilation=2, groups=base * 8, bias=False),
            nn.SiLU(inplace=True),
        )
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
        b = self.bridge(e4)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


@torch.no_grad()
def predict_roi(model: nn.Module, image: np.ndarray, box: tuple[int, int, int, int], roi_size: int, device: str) -> np.ndarray:
    image_crop = crop_resize(image, box, roi_size, cv2.INTER_AREA)
    xx, yy = make_coord_channels(roi_size)
    x = np.stack([image_crop, np.ones_like(image_crop, dtype=np.float32), xx, yy], axis=0)
    logits = model(torch.from_numpy(x[None].astype(np.float32)).to(device))
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    x0, y0, x1, y1 = box
    return cv2.resize(prob.astype(np.float32), (x1 - x0 + 1, y1 - y0 + 1), interpolation=cv2.INTER_LINEAR)


def proposal_mask_for(
    roots: list[Path],
    exp: str,
    raw_root: Path,
    dataset: str,
    split: str,
    name: str,
    source_dataset: str | None,
) -> np.ndarray:
    gt_shape = read_mask(raw_root / dataset / f"{split}_labels" / name).shape
    return resize_like(read_mask(find_mask(roots, exp, dataset, split, name, source_dataset)), gt_shape)


@torch.no_grad()
def infer_dataset(
    model: nn.Module,
    args: argparse.Namespace,
    raw_root: Path,
    dataset: str,
    split: str,
    proposal_exp: str,
    source_dataset: str | None,
    threshold: float,
    blend_mode: str,
    write_dir: Path | None = None,
    limit_images: int = 0,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    roots = [Path(p) for p in args.ablations_roots]
    records: list[dict[str, float]] = []
    split_names = names(raw_root, dataset, split)
    if limit_images > 0:
        split_names = split_names[:limit_images]
    for name in tqdm(split_names, desc=f"infer/{dataset}/{split}/t{threshold:.2f}/{blend_mode}", leave=False):
        gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
        image = read_gray(image_path(raw_root, dataset, split, name))
        anchor = proposal_mask_for(roots, proposal_exp, raw_root, dataset, split, name, source_dataset)
        pred = np.zeros_like(anchor, dtype=bool)
        for base_box in component_boxes(anchor, args.min_proposal_area, args.max_proposals_per_image):
            box = expand_box(base_box, anchor.shape, args.box_padding, 0.0)
            prob = predict_roi(model, image, box, args.roi_size, args.device)
            x0, y0, x1, y1 = box
            pred[y0 : y1 + 1, x0 : x1 + 1] |= prob >= threshold
        if blend_mode == "union":
            pred = pred | anchor
        elif blend_mode == "intersect":
            pred = pred & anchor
        elif blend_mode != "roi":
            raise ValueError(f"unknown blend mode: {blend_mode}")
        if write_dir is not None:
            write_mask(write_dir / name, pred)
        rec = compute_metrics(pred, gt, args.boundary_kernel)
        rec["image"] = name
        records.append(rec)
    return mean_metrics(records), records


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_ds = RoiProposalDataset(args, args.train_split, args.limit_train_images, augment=True)
    if len(train_ds) == 0:
        raise RuntimeError("empty ROI training dataset")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    model = RoiUNet(args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = parse_float_list(args.thresholds)
    blend_modes = parse_list(args.blend_modes)
    best: dict[str, object] | None = None
    history: list[dict[str, object]] = []
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(xb)
                loss = loss_fn(logits, yb, args.boundary_loss_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        epoch_best = None
        for threshold in thresholds:
            for blend_mode in blend_modes:
                mean, _ = infer_dataset(
                    model,
                    args,
                    Path(args.train_raw_root),
                    args.train_dataset,
                    args.tune_split,
                    args.proposal_exp,
                    None,
                    threshold,
                    blend_mode,
                    None,
                    args.limit_tune_images,
                )
                item = {"threshold": threshold, "blend_mode": blend_mode, "mean": mean}
                if epoch_best is None or mean["dice"] > epoch_best["mean"]["dice"]:
                    epoch_best = item
        assert epoch_best is not None
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_best": epoch_best,
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if best is None or epoch_best["mean"]["dice"] > best["mean"]["dice"]:
            best = {"epoch": epoch, **epoch_best}
            torch.save({"model": model.state_dict(), "args": vars(args), "best": best}, checkpoint)
        if epoch >= args.min_epochs and best is not None and epoch - int(best["epoch"]) >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best["epoch"]}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best = state["best"]
    apply_proposal_exp = args.apply_proposal_exp or args.proposal_exp
    out_dir = Path(args.pred_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = infer_dataset(
        model,
        args,
        Path(args.apply_raw_root),
        args.apply_dataset,
        args.apply_split,
        apply_proposal_exp,
        args.source_apply_dataset,
        float(best["threshold"]),
        str(best["blend_mode"]),
        out_dir,
        args.limit_apply_images,
    )
    summary = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "best": best,
        "output_exp": args.output_exp,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    control_out_dir = Path(args.pred_root) / args.output_exp / args.train_dataset / args.apply_split / "masks"
    control_proposal_exp = args.control_proposal_exp or args.proposal_exp
    control_mean, control_records = infer_dataset(
        model,
        args,
        Path(args.train_raw_root),
        args.train_dataset,
        args.apply_split,
        control_proposal_exp,
        None,
        float(best["threshold"]),
        str(best["blend_mode"]),
        control_out_dir,
        0,
    )
    control_summary = {
        "dataset": args.train_dataset,
        "split": args.apply_split,
        "num_evaluated": len(control_records),
        "mean": control_mean,
        "per_image": control_records,
        "best": best,
        "output_exp": args.output_exp,
    }
    Path(args.control_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.control_metrics_json).write_text(json.dumps(control_summary, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "clean_test_v2": mean, "original_test": control_mean}, indent=2), flush=True)


if __name__ == "__main__":
    main()
