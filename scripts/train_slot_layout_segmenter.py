#!/usr/bin/env python3
"""Train a slot-layout instance segmenter with explicit anatomical slots."""

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
    parser = argparse.ArgumentParser(description="Train slot-layout anatomical segmenter.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-slots", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--binary-loss-weight", type=float, default=0.5)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.03)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--slot-blends", default="0.70,0.85,1.00")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-exp", default="r078_slot_layout_segmenter")
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r078_slot_layout_segmenter/best.pt")
    parser.add_argument("--history-json", default="outputs/context_segmenter/r078_slot_layout_segmenter/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r078_slot_layout_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r078_slot_layout_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def weighted_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pos = target.mean().detach().clamp(1e-4, 0.5)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=(1.0 - pos) / pos)


def slot_targets(instance: np.ndarray, num_slots: int) -> tuple[np.ndarray, np.ndarray]:
    binary = (instance > 0).astype(np.float32)
    targets = np.zeros((num_slots, instance.shape[0], instance.shape[1]), dtype=np.float32)
    components = []
    for value in [int(v) for v in np.unique(instance) if int(v) > 0]:
        mask = instance == value
        if int(mask.sum()) == 0:
            continue
        ys, xs = np.where(mask)
        components.append((float(np.median(ys)), float(np.median(xs)), int(mask.sum()), mask))
    components.sort(key=lambda item: (item[0], item[1], -item[2]))
    for slot, (_, _, _, mask) in enumerate(components[:num_slots]):
        targets[slot, mask] = 1.0
    return binary, targets


class SlotDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dataset: str,
        split: str,
        img_size: int,
        num_slots: int,
        limit: int = 0,
        augment: bool = False,
    ) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.num_slots = num_slots
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
        binary, slots = slot_targets(instance, self.num_slots)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                binary = np.ascontiguousarray(np.fliplr(binary))
                slots = np.ascontiguousarray(np.flip(slots, axis=2))
            if random.random() < 0.25:
                gamma = random.uniform(0.85, 1.20)
                image = np.clip(image ** gamma, 0.0, 1.0)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0.0, 0.015, size=image.shape).astype(np.float32), 0.0, 1.0)

        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "binary": torch.from_numpy(binary[None].astype(np.float32)),
            "slots": torch.from_numpy(slots.astype(np.float32)),
            "name": label_path.name,
            "shape": original_shape,
        }


class SlotLayoutUNet(nn.Module):
    def __init__(self, num_slots: int, base: int = 24) -> None:
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
        self.slot_head = nn.Conv2d(base, num_slots, 1)
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
        return {"slots": self.slot_head(d1), "binary": self.binary_head(d1)}


def train_loss(outputs: dict[str, torch.Tensor], slots: torch.Tensor, binary: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    slot_logits = outputs["slots"]
    binary_logits = outputs["binary"]
    loss = weighted_bce(slot_logits, slots) + dice_loss(slot_logits, slots)
    if args.binary_loss_weight > 0:
        loss = loss + args.binary_loss_weight * (weighted_bce(binary_logits, binary) + dice_loss(binary_logits, binary))
    if args.overlap_loss_weight > 0:
        prob = torch.sigmoid(slot_logits)
        overlap = F.relu(prob.sum(dim=1, keepdim=True) - 1.0)
        loss = loss + args.overlap_loss_weight * overlap.mean()
    return loss


def blended_prob(outputs: dict[str, torch.Tensor], slot_blend: float) -> torch.Tensor:
    slot_prob = torch.sigmoid(outputs["slots"]).amax(dim=1, keepdim=True)
    binary_prob = torch.sigmoid(outputs["binary"])
    return float(slot_blend) * slot_prob + (1.0 - float(slot_blend)) * binary_prob


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
    num_slots: int,
    threshold: float,
    slot_blend: float,
    device: str,
    output_mask_dir: Path,
    metrics_json: Path,
) -> dict[str, object]:
    ds = SlotDataset(raw_root, dataset, split, img_size, num_slots, augment=False)
    records: list[dict[str, float]] = []
    model.eval()
    for item in tqdm(ds, desc=f"infer/{dataset}/{split}"):
        image = item["image"].unsqueeze(0).to(device)
        prob = blended_prob(model(image), slot_blend).cpu().numpy()[0, 0]
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
        "slot_blend": slot_blend,
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
    train_ds = SlotDataset(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.num_slots, args.limit_train, True)
    val_ds = SlotDataset(Path(args.raw_root), args.dataset, args.val_split, args.img_size, args.num_slots, args.limit_val, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = SlotLayoutUNet(args.num_slots, args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    slot_blends = [float(x) for x in args.slot_blends.split(",") if x.strip()]
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_blend = 1.0
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            images = batch["image"].to(args.device)
            slots = batch["slots"].to(args.device)
            binary = batch["binary"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                outputs = model(images)
                loss = train_loss(outputs, slots, binary, args)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        val_grid = {}
        for thr in thresholds:
            for blend in slot_blends:
                val_grid[(thr, blend)] = evaluate_loader(model, val_loader, args.device, thr, blend)
        (chosen_thr, chosen_blend), chosen_metrics = max(val_grid.items(), key=lambda kv: kv[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(chosen_metrics["dice"]),
            "val_precision": float(chosen_metrics["precision"]),
            "val_recall": float(chosen_metrics["recall"]),
            "val_boundary_iou": float(chosen_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(chosen_metrics["false_bridge_flag"]),
            "threshold": float(chosen_thr),
            "slot_blend": float(chosen_blend),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_threshold = float(chosen_thr)
            best_blend = float(chosen_blend)
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best_threshold, "slot_blend": best_blend, "epoch": epoch}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best_threshold = float(state["threshold"])
    best_blend = float(state.get("slot_blend", 1.0))
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        args.num_slots,
        best_threshold,
        best_blend,
        args.device,
        Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        args.num_slots,
        best_threshold,
        best_blend,
        args.device,
        Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "slot_blend": best_blend,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
