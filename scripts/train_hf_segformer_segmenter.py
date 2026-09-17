#!/usr/bin/env python3
"""Train a Hugging Face semantic-segmentation binary segmenter for epiphysis masks."""

from __future__ import annotations

import argparse
import json
import sys
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
    parser = argparse.ArgumentParser(description="Train Hugging Face semantic segmentation models for binary epiphysis segmentation.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--model-name", default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--hf-arch", choices=["segformer", "upernet", "auto"], default="segformer")
    parser.add_argument("--extra-pythonpath", default=".tmp/r144_hf_deps")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.08)
    parser.add_argument("--distance-loss-weight", type=float, default=0.04)
    parser.add_argument("--distance-sigma", type=float, default=3.0)
    parser.add_argument("--thresholds", default="0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--min-component-areas", default="0,8,16")
    parser.add_argument("--strong-xray-aug", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output-exp", default="r157_hf_segformer_smoke")
    parser.add_argument("--checkpoint", default="outputs/hf_segformer/r157_hf_segformer_smoke/best.pt")
    parser.add_argument("--history-json", default="outputs/hf_segformer/r157_hf_segformer_smoke/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r157_hf_segformer_smoke_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r157_hf_segformer_smoke_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


class HFSemanticBinary(nn.Module):
    def __init__(self, model_name: str, hf_arch: str, extra_pythonpath: str = "") -> None:
        super().__init__()
        if extra_pythonpath:
            extra = Path(extra_pythonpath)
            if extra.exists():
                sys.path.insert(0, str(extra.resolve()))
        if hf_arch == "segformer":
            from transformers import SegformerForSemanticSegmentation

            model_cls = SegformerForSemanticSegmentation
        elif hf_arch == "upernet":
            from transformers import UperNetForSemanticSegmentation

            model_cls = UperNetForSemanticSegmentation
        else:
            from transformers import AutoModelForSemanticSegmentation

            model_cls = AutoModelForSemanticSegmentation

        self.model = model_cls.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "background", 1: "epiphysis"},
            label2id={"background": 0, "epiphysis": 1},
            ignore_mismatched_sizes=True,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = x.repeat(1, 3, 1, 1)
        rgb = (rgb - self.mean) / self.std
        out = self.model(pixel_values=rgb).logits
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out[:, 1:2] - out[:, 0:1]


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float,
    min_component_area: int = 0,
) -> dict[str, float]:
    records: list[dict[str, float]] = []
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
    model = HFSemanticBinary(args.model_name, args.hf_arch, args.extra_pythonpath).to(args.device)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    min_component_areas = [int(float(x)) for x in args.min_component_areas.split(",") if x.strip()]
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int | str]] = []
    best_score = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"r157/epoch{epoch}"), start=1):
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
            for area in min_component_areas
        }
        (best_thr, best_area), best_metrics = max(grid.items(), key=lambda item: item[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(best_metrics["dice"]),
            "val_precision": float(best_metrics["precision"]),
            "val_recall": float(best_metrics["recall"]),
            "val_specificity": float(best_metrics["specificity"]),
            "val_boundary_iou": float(best_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(best_metrics["false_bridge_flag"]),
            "threshold": float(best_thr),
            "min_component_area": int(best_area),
            "model_name": args.model_name,
            "hf_arch": args.hf_arch,
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            torch.save(
                {
                    "model": eval_model.state_dict(),
                    "args": vars(args),
                    "threshold": best_thr,
                    "min_component_area": best_area,
                },
                checkpoint,
            )
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if args.skip_final_eval:
        print(json.dumps({"best_epoch": best_epoch, "best_val_dice": best_score, "skip_final_eval": True}), flush=True)
        return

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
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "threshold": threshold,
                "min_component_area": min_component_area,
                "clean_test_v2": clean["mean"],
                "original_test": control["mean"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
