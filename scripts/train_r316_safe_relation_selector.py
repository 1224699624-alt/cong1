#!/usr/bin/env python3
"""R316: relation selector with hard-negative focus and two-sided bone support.

The selector predicts whether a candidate pair has a reliable separating
interface, its local heatmap, and two eroded bone-core support maps. The
support target is auxiliary supervision only; the binary segmentation target
remains the primary task. Validation reports a threshold grid rather than
silently choosing a cut threshold.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from train_r258b_prediction_relation_prior import (
    Block,
    build_samples,
    find_image,
    gaussian,
    make_heat_target,
    read_gray,
    read_label,
    RelationPriorNet,
    set_seed,
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant-root", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--train-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_train.csv"))
    p.add_argument("--val-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_val.csv"))
    p.add_argument("--train-proposals", type=Path, default=Path("outputs/analysis/r258_oof_center_basin_full_proposals.csv"))
    p.add_argument("--val-proposals", type=Path, default=Path("outputs/analysis/r257b_scale_repair_fullval_proposals.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pair_prior/r316_safe_relation_selector"))
    p.add_argument("--result-json", type=Path, default=Path("outputs/analysis/r316_safe_relation_selector.json"))
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=3161)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--crop-margin-scales", type=float, default=1.35)
    p.add_argument("--match-tolerance-scale", type=float, default=.60)
    p.add_argument("--relative-pair-limit", type=float, default=.80)
    p.add_argument("--proposal-distance-limit", type=float, default=4.0)
    p.add_argument("--relative-close-threshold", type=float, default=.20)
    p.add_argument("--max-pairs-per-case", type=int, default=32)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--hard-negative-quantile", type=float, default=0.0,
                   help="If >0, define hard negatives as the closest q fraction of negative pairs; 0 preserves the historical fixed 0.35 rule.")
    p.add_argument("--require-full-data", action="store_true")
    p.add_argument("--r316c", action="store_true", help="Use the isolated R316C 256/128 dual-scale pair representation")
    p.add_argument("--global-size", type=int, default=128)
    p.add_argument("--center-sigma-rel", type=float, default=.15)
    p.add_argument("--center-sigma-min", type=float, default=3.)
    p.add_argument("--center-sigma-max", type=float, default=12.)
    p.add_argument("--center-jitter-rel", type=float, default=.03)
    return p.parse_args()


class SafePairDataset(Dataset):
    def __init__(self, samples: list[dict], a: argparse.Namespace, augment: bool):
        self.samples, self.a, self.augment = samples, a, augment
        negative_distances = np.asarray(
            [float(s["proposal_relative_distance"]) for s in samples if not s["positive"]],
            dtype=np.float64,
        )
        if not 0.0 <= a.hard_negative_quantile < 1.0:
            raise ValueError("hard-negative-quantile must be in [0, 1)")
        self.hard_negative_threshold = (
            float(np.quantile(negative_distances, a.hard_negative_quantile))
            if a.hard_negative_quantile > 0 and negative_distances.size
            else 0.35
        )
        self.hard_negative_count = int(sum(
            (not s["positive"])
            and float(s["proposal_relative_distance"]) <= self.hard_negative_threshold
            for s in samples
        ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        s = self.samples[index]
        image = read_gray(s["image"])
        instance = read_label(s["label"])
        pa, pb = np.asarray(s["pa"], np.float64), np.asarray(s["pb"], np.float64)
        psa, psb = float(s["psa"]), float(s["psb"])
        center = .5 * (pa + pb)
        side = int(math.ceil(max(32.0, np.linalg.norm(pb - pa) + self.a.crop_margin_scales * (psa + psb))))
        x0, y0 = int(round(center[0] - side / 2)), int(round(center[1] - side / 2))

        def crop(array: np.ndarray) -> np.ndarray:
            out = np.zeros((side, side), dtype=array.dtype)
            sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(array.shape[1], x0 + side), min(array.shape[0], y0 + side)
            if sx1 > sx0 and sy1 > sy0:
                out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = array[sy0:sy1, sx0:sx1]
            return out

        size = self.a.crop_size
        scale_xy = size / side
        ca, cb = (pa - [x0, y0]) * scale_xy, (pb - [x0, y0]) * scale_xy
        image_crop = cv2.resize(crop(image), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if self.a.r316c:
            from r316c_pair_inputs import pair_channels
            channels, _, _, _ = pair_channels(
                crop(image), ca, cb, .5 * (psa + psb) * scale_xy, size,
                self.a.global_size, self.a.center_sigma_rel,
                self.a.center_sigma_min, self.a.center_sigma_max,
                self.a.center_jitter_rel if self.augment else 0.0,
            )
        else:
            channels = np.stack([image_crop, gaussian(size, *ca), gaussian(size, *cb)]).astype(np.float32)
        seam = np.zeros((size, size), np.float32)
        support = np.zeros((2, size, size), np.float32)
        valid = False
        if s["positive"]:
            ma, mb, fg = crop(instance == int(s["a"])), crop(instance == int(s["b"])), crop(instance > 0)
            seam, valid = make_heat_target(ma, mb, fg, size, ca, cb, .5 * (psa + psb) * scale_xy)
            # Erosion makes support supervision robust to inaccurate colored boundaries.
            erode = max(1, int(round(.03 * size)))
            structure = np.ones((erode, erode), dtype=bool)
            support[0] = cv2.resize(ndimage.binary_erosion(ma, structure=structure).astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
            support[1] = cv2.resize(ndimage.binary_erosion(mb, structure=structure).astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < .5:
                channels, seam, support = channels[:, :, ::-1].copy(), seam[:, ::-1].copy(), support[:, :, ::-1].copy()
            if random.random() < .5:
                channels, seam, support = channels[:, ::-1, :].copy(), seam[::-1, :].copy(), support[:, ::-1, :].copy()
            turns = random.randrange(4)
            if turns:
                channels, seam, support = np.rot90(channels, turns, axes=(1, 2)).copy(), np.rot90(seam, turns).copy(), np.rot90(support, turns, axes=(1, 2)).copy()
        features = np.asarray([float(s["age"]) / 240.0, float(s["male"])], np.float32)
        return {
            "image": torch.from_numpy(channels), "features": torch.from_numpy(features),
            "label": torch.tensor(float(s["positive"])), "seam": torch.from_numpy(seam[None]),
            "support": torch.from_numpy(support), "valid": torch.tensor(float(valid)),
            "hard_negative": torch.tensor(float(
                (not s["positive"])
                and float(s["proposal_relative_distance"]) <= self.hard_negative_threshold
            )),
            "close": torch.tensor(float(s["positive"] and s["relative_gap"] <= self.a.relative_close_threshold)),
        }


class SafeRelationNet(nn.Module):
    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1, self.e2, self.e3, self.e4 = Block(input_channels, 16), Block(16, 32), Block(32, 64), Block(64, 128)
        self.meta = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 128))
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
        self.u3, self.d3 = nn.ConvTranspose2d(128, 64, 2, 2), Block(128, 64)
        self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), Block(64, 32)
        self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), Block(32, 16)
        self.seam, self.support = nn.Conv2d(16, 1, 1), nn.Conv2d(16, 2, 1)

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.e1(image); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        meta = self.meta(features); conditioned = e4 + meta[:, :, None, None]
        logit = self.classifier(torch.cat([F.adaptive_avg_pool2d(e4, 1).flatten(1), meta], 1)).squeeze(1)
        d3 = self.d3(torch.cat([self.u3(conditioned), e3], 1)); d2 = self.d2(torch.cat([self.u2(d3), e2], 1)); d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return logit, self.seam(d1), self.support(d1)


def focal_bce(logit: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = torch.exp(-bce)
    return (weight * (1 - pt).pow(2.0) * bce).mean()


@torch.no_grad()
def evaluate(model: SafeRelationNet, loader: DataLoader, device: str) -> dict[str, object]:
    model.eval(); labels, scores, close, hard, seam_dice = [], [], [], [], []
    for batch in loader:
        logit, seam, support = model(batch["image"].to(device), batch["features"].to(device))
        labels.extend(batch["label"].numpy()); scores.extend(torch.sigmoid(logit).cpu().numpy()); close.extend(batch["close"].numpy()); hard.extend(batch["hard_negative"].numpy())
        prob, target = torch.sigmoid(seam), batch["seam"].to(device)
        dice = (2 * (prob * target).sum((1, 2, 3)) + 1) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
        seam_dice.extend(dice.cpu().numpy())
    y, p = np.asarray(labels), np.asarray(scores); close, hard = np.asarray(close) > .5, np.asarray(hard) > .5
    thresholds = []
    for threshold in np.arange(.30, .81, .05):
        pred = p >= threshold
        thresholds.append({"threshold": float(threshold), "positive_recall": float(pred[y > .5].mean()), "close_recall": float(pred[close].mean()) if close.any() else 0.0, "negative_specificity": float((~pred[y < .5]).mean()), "hard_negative_specificity": float((~pred[hard]).mean()) if hard.any() else 0.0})
    return {
        "num_pairs": int(len(y)), "threshold_grid": thresholds,
        "mean_seam_soft_dice": float(np.mean(seam_dice)),
        "hard_negative_count": int(hard.sum()),
        "hard_negative_distance_threshold": float(loader.dataset.hard_negative_threshold),
        "hard_negative_quantile": float(loader.dataset.a.hard_negative_quantile),
    }


def main() -> None:
    a = args(); set_seed(a.seed); a.device = a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu"
    if a.r316c and a.require_full_data:
        fixed = {"crop_size": 256, "epochs": 8, "batch_size": 16, "workers": 6,
                 "lr": 3e-4, "seed": 3161, "hard_negative_quantile": .20,
                 "global_size": 128, "center_sigma_rel": .15,
                 "center_sigma_min": 3., "center_sigma_max": 12.,
                 "center_jitter_rel": .03}
        if any(abs(float(getattr(a, key)) - float(value)) > 1e-12 for key, value in fixed.items()):
            raise RuntimeError("Full R316C safe-selector protocol mismatch")
    metadata_train = __import__("train_r258b_prediction_relation_prior").read_metadata(a.train_metadata)
    metadata_val = __import__("train_r258b_prediction_relation_prior").read_metadata(a.val_metadata)
    if a.require_full_data and (len(metadata_train), len(metadata_val)) != (875, 96):
        raise RuntimeError(f"R316 full-data protocol requires 875/96 metadata rows, got {len(metadata_train)}/{len(metadata_val)}")
    train, train_audit = build_samples(a.variant_root, "train", metadata_train, a.train_proposals, a)
    val, val_audit = build_samples(a.variant_root, "val", metadata_val, a.val_proposals, a)
    train_dataset, val_dataset = SafePairDataset(train, a, True), SafePairDataset(val, a, False)
    if a.hard_negative_quantile > 0 and (train_dataset.hard_negative_count == 0 or val_dataset.hard_negative_count == 0):
        raise RuntimeError("Quantile hard-negative construction produced an empty train or validation subset")
    train_loader = DataLoader(train_dataset, a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
    model = SafeRelationNet(4 if a.r316c else 3).to(a.device); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=a.device.startswith("cuda")); history = []
    for epoch in range(1, a.epochs + 1):
        model.train(); losses = []
        for batch in train_loader:
            image, features = batch["image"].to(a.device), batch["features"].to(a.device); label, seam_t = batch["label"].to(a.device), batch["seam"].to(a.device); support_t, valid = batch["support"].to(a.device), batch["valid"].to(a.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=a.device.startswith("cuda")):
                logit, seam, support = model(image, features)
                pair_weight = 1.0 + 1.5 * batch["hard_negative"].to(a.device)
                loss = focal_bce(logit, label, pair_weight)
                valid_positive = (label > .5) & (valid > .5)
                if valid_positive.any():
                    seam_prob = torch.sigmoid(seam); seam_loss = F.binary_cross_entropy_with_logits(seam, seam_t) + (1 - (2 * (seam_prob * seam_t).sum((1, 2, 3)) + 1) / (seam_prob.sum((1, 2, 3)) + seam_t.sum((1, 2, 3)) + 1))[valid_positive].mean()
                    support_loss = F.binary_cross_entropy_with_logits(support, support_t)
                    loss = loss + .35 * seam_loss + .15 * support_loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, a.device); row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}; history.append(row); print(json.dumps(row), flush=True)
    a.output_dir.mkdir(parents=True, exist_ok=True); checkpoint = a.output_dir / "safe_relation_seed3161_final.pt"; torch.save({"model": model.state_dict(), "config": vars(a), "history": history}, checkpoint)
    a.result_json.parent.mkdir(parents=True, exist_ok=True); a.result_json.write_text(json.dumps({
        "experiment": "R316B_FULL_QUANTILE_HARD_NEGATIVE" if a.hard_negative_quantile > 0 else "R316",
        "config": vars(a), "train_audit": train_audit, "val_audit": val_audit,
        "hard_negative_train_count": train_dataset.hard_negative_count,
        "hard_negative_val_count": val_dataset.hard_negative_count,
        "history": history, "checkpoint": str(checkpoint), "clean_test_used": False,
    }, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
