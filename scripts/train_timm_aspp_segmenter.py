#!/usr/bin/env python3
"""Train a timm encoder with an ASPP/context decoder for epiphysis masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_timm_unet_segmenter import (
    ModelEMA,
    SegDataset,
    add_structure_metrics,
    compute_metrics,
    infer_and_evaluate,
    loss_fn,
    mean_metrics,
    remove_small_components,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train timm ASPP/context segmentation model.")
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
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--aspp-rates", default="1,3,6,9")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.12)
    parser.add_argument("--distance-loss-weight", type=float, default=0.08)
    parser.add_argument("--distance-sigma", type=float, default=3.0)
    parser.add_argument("--thresholds", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--min-component-areas", default="0,8,16")
    parser.add_argument("--strong-xray-aug", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output-exp", default="r155_aspp_context_segmenter")
    parser.add_argument("--checkpoint", default="outputs/timm_aspp/r155_aspp_context_segmenter/best.pt")
    parser.add_argument("--history-json", default="outputs/timm_aspp/r155_aspp_context_segmenter/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r155_aspp_context_segmenter_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r155_aspp_context_segmenter_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel // 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates: list[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            ConvBNAct(in_ch, out_ch, kernel=1, dilation=1) if rate == 1 else ConvBNAct(in_ch, out_ch, kernel=3, dilation=rate)
            for rate in rates
        ])
        self.project = ConvBNAct(out_ch * len(rates), out_ch, kernel=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class TimmASPPNet(nn.Module):
    def __init__(self, encoder: str, pretrained: bool, decoder_channels: int, aspp_rates: list[int]) -> None:
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
        self.aspp = ASPP(channels[-1], decoder_channels, aspp_rates)
        self.low_proj = ConvBNAct(channels[0], decoder_channels // 2, kernel=1)
        self.mid_proj = ConvBNAct(channels[1], decoder_channels // 2, kernel=1)
        self.refine = nn.Sequential(
            ConvBNAct(decoder_channels * 2, decoder_channels),
            ConvBNAct(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, 1, 1),
        )

    @staticmethod
    def _as_nchw(feat: torch.Tensor, channels: int) -> torch.Tensor:
        if feat.ndim == 4 and feat.shape[1] == channels:
            return feat
        if feat.ndim == 4 and feat.shape[-1] == channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        feats = [
            self._as_nchw(feat, channels)
            for feat, channels in zip(self.encoder(x), self.encoder.feature_info.channels(), strict=True)
        ]
        deep = self.aspp(feats[-1])
        mid = self.mid_proj(feats[1])
        low = self.low_proj(feats[0])
        deep = F.interpolate(deep, size=low.shape[-2:], mode="bilinear", align_corners=False)
        mid = F.interpolate(mid, size=low.shape[-2:], mode="bilinear", align_corners=False)
        out = self.refine(torch.cat([deep, mid, low], dim=1))
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, threshold: float, min_component_area: int) -> dict[str, float]:
    records = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].numpy()
        probs = torch.sigmoid(model(images)).cpu().numpy()
        for idx in range(probs.shape[0]):
            pred = remove_small_components(probs[idx, 0] >= threshold, min_component_area)
            gt = masks[idx, 0] > 0.5
            rec = compute_metrics(pred, gt, boundary_kernel=3)
            rec = add_structure_metrics(rec, pred, gt)
            rec["image"] = str(batch["name"][idx])
            records.append(rec)
    return mean_metrics(records)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    aspp_rates = [int(x) for x in args.aspp_rates.split(",") if x.strip()]
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
    model = TimmASPPNet(args.encoder, args.pretrained, args.decoder_channels, aspp_rates).to(args.device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    areas = [int(float(x)) for x in args.min_component_areas.split(",") if x.strip()]
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"r155/epoch{epoch}"), start=1):
            images = batch["image"].to(args.device)
            masks = batch["mask"].to(args.device)
            distance_targets = batch["distance_target"].to(args.device)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(images)
                loss = loss_fn(
                    logits,
                    masks,
                    args.boundary_loss_weight,
                    distance_targets,
                    args.distance_loss_weight,
                ) / max(args.grad_accum_steps, 1)
            scaler.scale(loss).backward()
            if step % max(args.grad_accum_steps, 1) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            losses.append(float(loss.item() * max(args.grad_accum_steps, 1)))

        eval_model = ema.ema if ema is not None else model
        grid = {
            (thr, area): evaluate_loader(eval_model, val_loader, args.device, thr, area)
            for thr in thresholds
            for area in areas
        }
        (best_thr, best_area), best_metrics = max(grid.items(), key=lambda item: item[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(best_metrics["dice"]),
            "val_precision": float(best_metrics["precision"]),
            "val_recall": float(best_metrics["recall"]),
            "val_boundary_iou": float(best_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(best_metrics["false_bridge_flag"]),
            "threshold": float(best_thr),
            "min_component_area": int(best_area),
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
            "aspp_rates": aspp_rates,
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "threshold": best_thr, "min_component_area": best_area}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    threshold = float(state["threshold"])
    min_component_area = int(state.get("min_component_area", 0))
    clean = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        threshold,
        args.device,
        Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
        min_component_area,
        args.tta_flips,
        args.limit_eval,
    )
    control = infer_and_evaluate(
        model,
        Path(args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        threshold,
        args.device,
        Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
        min_component_area,
        args.tta_flips,
        args.limit_eval,
    )
    print(json.dumps({"best_epoch": best_epoch, "threshold": threshold, "clean_test_v2": clean["mean"], "original_test": control["mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
