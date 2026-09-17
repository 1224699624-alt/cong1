#!/usr/bin/env python3
"""Train a permutation-invariant set/query instance segmenter.

This is a variable-slot alternative to the fixed anatomical slot model. It
predicts a fixed number of query masks, but matches them to instance-valued
labels with Hungarian assignment during training, so label ID/order is not
assumed to be stable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_instance_separation_segmenter import (
    ContextBlock,
    ConvBlock,
    add_structure_metrics,
    find_image_path,
    mean_metrics,
    read_gray,
    read_instance_mask,
    resize_float,
    resize_instance,
    write_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train set-matching variable-slot instance segmenter.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--binary-loss-weight", type=float, default=0.55)
    parser.add_argument("--empty-loss-weight", type=float, default=0.02)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.03)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--slot-blends", default="0.75,0.90,1.00")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output-exp", default="r129_set_instance_segmenter")
    parser.add_argument("--checkpoint", default="outputs/set_instance/r129_set_instance_segmenter/best.pt")
    parser.add_argument("--history-json", default="outputs/set_instance/r129_set_instance_segmenter/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r129_set_instance_segmenter_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r129_set_instance_segmenter_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(-2, -1))
    den = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def weighted_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=(1.0 - pos) / pos)


def instance_slots(instance: np.ndarray, num_queries: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary = (instance > 0).astype(np.float32)
    slots = np.zeros((num_queries, instance.shape[0], instance.shape[1]), dtype=np.float32)
    valid = np.zeros((num_queries,), dtype=np.float32)
    components = []
    for value in [int(v) for v in np.unique(instance) if int(v) > 0]:
        mask = instance == value
        area = int(mask.sum())
        if area <= 0:
            continue
        components.append((area, mask))
    components.sort(key=lambda item: -item[0])
    for idx, (_, mask) in enumerate(components[:num_queries]):
        slots[idx, mask] = 1.0
        valid[idx] = 1.0
    return binary, slots, valid


class SetInstanceDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dataset: str,
        split: str,
        img_size: int,
        num_queries: int,
        limit: int = 0,
        augment: bool = False,
    ) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.num_queries = num_queries
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
        binary, slots, valid = instance_slots(instance, self.num_queries)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                binary = np.ascontiguousarray(np.fliplr(binary))
                slots = np.ascontiguousarray(np.flip(slots, axis=2))
            if random.random() < 0.25:
                image = np.clip(image ** random.uniform(0.85, 1.20), 0.0, 1.0)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0.0, 0.015, size=image.shape).astype(np.float32), 0.0, 1.0)

        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "binary": torch.from_numpy(binary[None].astype(np.float32)),
            "slots": torch.from_numpy(slots.astype(np.float32)),
            "valid": torch.from_numpy(valid.astype(np.float32)),
            "name": label_path.name,
            "shape": original_shape,
        }


class SetInstanceUNet(nn.Module):
    def __init__(self, num_queries: int, base: int = 24) -> None:
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
        self.query_head = nn.Conv2d(base, num_queries, 1)
        self.binary_head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.context(e4)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return {"queries": self.query_head(d1), "binary": self.binary_head(d1)}


def hungarian_instance_loss(
    query_logits: torch.Tensor,
    target_slots: torch.Tensor,
    valid: torch.Tensor,
    empty_weight: float,
) -> torch.Tensor:
    batch_losses = []
    probs = torch.sigmoid(query_logits)
    for b in range(query_logits.shape[0]):
        gt_idx = torch.where(valid[b] > 0.5)[0]
        if gt_idx.numel() == 0:
            batch_losses.append(empty_weight * probs[b].mean())
            continue
        pred_flat = probs[b].flatten(1)
        gt_flat = target_slots[b, gt_idx].flatten(1)
        inter = pred_flat @ gt_flat.t()
        den = pred_flat.sum(dim=1, keepdim=True) + gt_flat.sum(dim=1).unsqueeze(0)
        dice_cost = 1.0 - (2.0 * inter + 1.0) / (den + 1.0)
        row_ind, col_ind = linear_sum_assignment(dice_cost.detach().cpu().numpy())
        pred_ids = torch.as_tensor(row_ind, device=query_logits.device, dtype=torch.long)
        gt_ids = gt_idx[torch.as_tensor(col_ind, device=query_logits.device, dtype=torch.long)]
        matched_logits = query_logits[b, pred_ids]
        matched_targets = target_slots[b, gt_ids]
        loss = weighted_bce(matched_logits, matched_targets) + dice_loss(matched_logits, matched_targets)
        unmatched = torch.ones(query_logits.shape[1], dtype=torch.bool, device=query_logits.device)
        unmatched[pred_ids] = False
        if unmatched.any() and empty_weight > 0:
            loss = loss + empty_weight * probs[b, unmatched].mean()
        batch_losses.append(loss)
    return torch.stack(batch_losses).mean()


def train_loss(outputs: dict[str, torch.Tensor], slots: torch.Tensor, valid: torch.Tensor, binary: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    query_logits = outputs["queries"]
    binary_logits = outputs["binary"]
    loss = hungarian_instance_loss(query_logits, slots, valid, args.empty_loss_weight)
    if args.binary_loss_weight > 0:
        loss = loss + args.binary_loss_weight * (weighted_bce(binary_logits, binary) + dice_loss(binary_logits, binary))
    if args.overlap_loss_weight > 0:
        qprob = torch.sigmoid(query_logits)
        overlap = F.relu(qprob.sum(dim=1, keepdim=True) - 1.0)
        loss = loss + args.overlap_loss_weight * overlap.mean()
    return loss


def blended_prob(outputs: dict[str, torch.Tensor], slot_blend: float) -> torch.Tensor:
    query_prob = torch.sigmoid(outputs["queries"]).amax(dim=1, keepdim=True)
    binary_prob = torch.sigmoid(outputs["binary"])
    return float(slot_blend) * query_prob + (1.0 - float(slot_blend)) * binary_prob


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, threshold: float, slot_blend: float) -> dict[str, float]:
    records: list[dict[str, float]] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        binary = batch["binary"].numpy()
        probs = blended_prob(model(images), slot_blend).cpu().numpy()
        for i in range(probs.shape[0]):
            pred = probs[i, 0] >= threshold
            gt = binary[i, 0] > 0.5
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
    num_queries: int,
    threshold: float,
    slot_blend: float,
    device: str,
    output_mask_dir: Path,
    metrics_json: Path,
    limit_eval: int = 0,
) -> dict[str, Any]:
    ds = SetInstanceDataset(raw_root, dataset, split, img_size, num_queries, limit=limit_eval, augment=False)
    if len(ds) == 0:
        raise FileNotFoundError(f"no labels found for evaluation: {raw_root / dataset / f'{split}_labels'}")
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    records: list[dict[str, float]] = []
    model.eval()
    for batch in tqdm(loader, desc=f"infer/{dataset}/{split}"):
        image = batch["image"].to(device)
        probs = blended_prob(model(image), slot_blend).cpu().numpy()[0, 0]
        pred_small = probs >= threshold
        shape = tuple(int(v) for v in batch["shape"])
        pred = cv2.resize(pred_small.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        name = str(batch["name"][0])
        write_mask(output_mask_dir / name, pred)
        gt = read_instance_mask(raw_root / dataset / f"{split}_labels" / name) > 0
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec = add_structure_metrics(rec, pred, gt)
        rec["image"] = name
        records.append(rec)
    summary = {
        "dataset": dataset,
        "split": split,
        "threshold": threshold,
        "slot_blend": slot_blend,
        "num_evaluated": len(records),
        "mean": mean_metrics(records),
        "per_image": records,
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def select_best(model: nn.Module, loader: DataLoader, device: str, thresholds: list[float], slot_blends: list[float]) -> tuple[float, float, dict[str, float]]:
    best = (-1.0, thresholds[0], slot_blends[0], {})
    for blend in slot_blends:
        for thr in thresholds:
            metrics = evaluate_loader(model, loader, device, thr, blend)
            dice = float(metrics.get("dice", 0.0))
            if dice > best[0]:
                best = (dice, thr, blend, metrics)
    return float(best[1]), float(best[2]), best[3]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    raw_root = Path(args.raw_root)
    device = args.device
    thresholds = [float(x) for x in args.thresholds.split(",") if x]
    slot_blends = [float(x) for x in args.slot_blends.split(",") if x]

    train_ds = SetInstanceDataset(raw_root, args.dataset, args.train_split, args.img_size, args.num_queries, args.limit_train, augment=True)
    val_ds = SetInstanceDataset(raw_root, args.dataset, args.val_split, args.img_size, args.num_queries, args.limit_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))

    model = SetInstanceUNet(args.num_queries, args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    history = []
    best_dice = -1.0
    best_epoch = 0
    best_thr = thresholds[0]
    best_blend = slot_blends[0]
    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            images = batch["image"].to(device)
            slots = batch["slots"].to(device)
            valid = batch["valid"].to(device)
            binary = batch["binary"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                loss = train_loss(model(images), slots, valid, binary, args)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        thr, blend, val_metrics = select_best(model, val_loader, device, thresholds, slot_blends)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_dice": float(val_metrics.get("dice", 0.0)),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_boundary_iou": float(val_metrics.get("boundary_iou", 0.0)),
            "val_false_bridge_flag": float(val_metrics.get("false_bridge_flag", 0.0)),
            "threshold": thr,
            "slot_blend": blend,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            best_epoch = epoch
            best_thr = thr
            best_blend = blend
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best_thr, "slot_blend": best_blend}, ckpt)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            break

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    best_thr = float(state.get("threshold", best_thr))
    best_blend = float(state.get("slot_blend", best_blend))

    pred_root = Path(args.pred_root) / args.output_exp
    infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        args.num_queries,
        best_thr,
        best_blend,
        device,
        pred_root / args.eval_dataset / args.eval_split,
        Path(args.metrics_json),
        args.limit_eval,
    )
    infer_and_evaluate(
        model,
        raw_root,
        args.control_dataset,
        args.eval_split,
        args.img_size,
        args.num_queries,
        best_thr,
        best_blend,
        device,
        pred_root / args.control_dataset / args.eval_split,
        Path(args.control_metrics_json),
        args.limit_eval,
    )


if __name__ == "__main__":
    main()
