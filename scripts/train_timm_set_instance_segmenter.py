#!/usr/bin/env python3
"""Train a timm-backed set/query instance segmenter with no-object calibration."""

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
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_instance_separation_segmenter import (
    ConvBlock,
    add_structure_metrics,
    mean_metrics,
    read_instance_mask,
    write_mask,
)
from train_set_instance_segmenter import SetInstanceDataset, dice_loss, weighted_bce
from train_timm_instance_separation_segmenter import ConvBNAct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train timm/FPN query instance segmenter.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--encoder", default="convnext_tiny.dinov3_lvd1689m")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--binary-loss-weight", type=float, default=0.45)
    parser.add_argument("--object-loss-weight", type=float, default=0.35)
    parser.add_argument("--no-object-loss-weight", type=float, default=0.08)
    parser.add_argument("--empty-mask-loss-weight", type=float, default=0.02)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.04)
    parser.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--slot-blends", default="0.70,0.85,1.00")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output-exp", default="r132_dinov3_query_noobject")
    parser.add_argument("--checkpoint", default="outputs/timm_set_instance/r132_dinov3_query_noobject/best.pt")
    parser.add_argument("--history-json", default="outputs/timm_set_instance/r132_dinov3_query_noobject/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r132_dinov3_query_noobject_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r132_dinov3_query_noobject_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260732)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TimmSetInstanceUNet(nn.Module):
    def __init__(self, encoder: str, pretrained: bool, decoder_channels: int, num_queries: int) -> None:
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
        self.trunk = nn.Sequential(
            ConvBNAct(decoder_channels, decoder_channels // 2),
            ConvBlock(decoder_channels // 2, decoder_channels // 2),
        )
        head_channels = decoder_channels // 2
        self.query_head = nn.Conv2d(head_channels, num_queries, 1)
        self.binary_head = nn.Conv2d(head_channels, 1, 1)
        self.object_head = nn.Linear(head_channels, num_queries)

    @staticmethod
    def _as_nchw(feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feat.ndim == 4 and feat.shape[1] == expected_channels:
            return feat
        if feat.ndim == 4 and feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        feats = [
            self._as_nchw(feat, expected_channels)
            for feat, expected_channels in zip(self.encoder(x), self.encoder.feature_info.channels(), strict=True)
        ]
        y = self.smooth[-1](self.lateral[-1](feats[-1]))
        for idx in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[idx].shape[-2:], mode="bilinear", align_corners=False)
            y = self.smooth[idx](y + self.lateral[idx](feats[idx]))
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        y = self.trunk(y)
        pooled = F.adaptive_avg_pool2d(y, 1).flatten(1)
        return {
            "queries": self.query_head(y),
            "binary": self.binary_head(y),
            "object_logits": self.object_head(pooled),
        }


def hungarian_query_loss(
    outputs: dict[str, torch.Tensor],
    target_slots: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    query_logits = outputs["queries"]
    object_logits = outputs["object_logits"]
    probs = torch.sigmoid(query_logits)
    batch_losses = []
    for b in range(query_logits.shape[0]):
        gt_idx = torch.where(valid[b] > 0.5)[0]
        object_target = torch.zeros_like(object_logits[b])
        if gt_idx.numel() == 0:
            obj_loss = F.binary_cross_entropy_with_logits(object_logits[b], object_target)
            empty_loss = args.empty_mask_loss_weight * probs[b].mean()
            batch_losses.append(args.object_loss_weight * obj_loss + empty_loss)
            continue

        pred_flat = probs[b].flatten(1)
        gt_flat = target_slots[b, gt_idx].flatten(1)
        inter = pred_flat @ gt_flat.t()
        den = pred_flat.sum(dim=1, keepdim=True) + gt_flat.sum(dim=1).unsqueeze(0)
        dice_cost = 1.0 - (2.0 * inter + 1.0) / (den + 1.0)
        obj_prob = torch.sigmoid(object_logits[b]).unsqueeze(1)
        cost = dice_cost - 0.15 * obj_prob
        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        pred_ids = torch.as_tensor(row_ind, device=query_logits.device, dtype=torch.long)
        gt_ids = gt_idx[torch.as_tensor(col_ind, device=query_logits.device, dtype=torch.long)]

        object_target[pred_ids] = 1.0
        obj_weight = torch.ones_like(object_target)
        obj_weight[object_target < 0.5] = args.no_object_loss_weight
        obj_loss = F.binary_cross_entropy_with_logits(object_logits[b], object_target, weight=obj_weight)
        mask_loss = weighted_bce(query_logits[b, pred_ids], target_slots[b, gt_ids]) + dice_loss(query_logits[b, pred_ids], target_slots[b, gt_ids])

        unmatched = torch.ones(query_logits.shape[1], dtype=torch.bool, device=query_logits.device)
        unmatched[pred_ids] = False
        empty_loss = query_logits.new_tensor(0.0)
        if unmatched.any() and args.empty_mask_loss_weight > 0:
            empty_loss = args.empty_mask_loss_weight * probs[b, unmatched].mean()
        batch_losses.append(mask_loss + args.object_loss_weight * obj_loss + empty_loss)
    return torch.stack(batch_losses).mean()


def train_loss(outputs: dict[str, torch.Tensor], slots: torch.Tensor, valid: torch.Tensor, binary: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    loss = hungarian_query_loss(outputs, slots, valid, args)
    if args.binary_loss_weight > 0:
        loss = loss + args.binary_loss_weight * (weighted_bce(outputs["binary"], binary) + dice_loss(outputs["binary"], binary))
    if args.overlap_loss_weight > 0:
        object_prob = torch.sigmoid(outputs["object_logits"]).unsqueeze(-1).unsqueeze(-1)
        qprob = torch.sigmoid(outputs["queries"]) * object_prob
        overlap = F.relu(qprob.sum(dim=1, keepdim=True) - 1.0)
        loss = loss + args.overlap_loss_weight * overlap.mean()
    return loss


def blended_prob(outputs: dict[str, torch.Tensor], slot_blend: float) -> torch.Tensor:
    object_prob = torch.sigmoid(outputs["object_logits"]).unsqueeze(-1).unsqueeze(-1)
    query_prob = (torch.sigmoid(outputs["queries"]) * object_prob).amax(dim=1, keepdim=True)
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
    device = args.device
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    slot_blends = [float(x) for x in args.slot_blends.split(",") if x.strip()]

    train_ds = SetInstanceDataset(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.num_queries, args.limit_train, augment=True)
    val_ds = SetInstanceDataset(Path(args.raw_root), args.dataset, args.val_split, args.img_size, args.num_queries, args.limit_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))

    model = TimmSetInstanceUNet(args.encoder, args.pretrained, args.decoder_channels, args.num_queries).to(device)
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
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
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
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best_thr, "slot_blend": best_blend, "epoch": epoch}, ckpt)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    best_thr = float(state.get("threshold", best_thr))
    best_blend = float(state.get("slot_blend", best_blend))
    pred_root = Path(args.pred_root) / args.output_exp
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        args.num_queries,
        best_thr,
        best_blend,
        device,
        pred_root / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
        args.limit_eval,
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        args.num_queries,
        best_thr,
        best_blend,
        device,
        pred_root / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
        args.limit_eval,
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_thr,
        "slot_blend": best_blend,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(ckpt),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
