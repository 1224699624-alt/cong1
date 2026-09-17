#!/usr/bin/env python3
"""R324: output-side explicit overlap-interaction prior on RAM-W600.

This is a paired validation experiment initialized from the same mature R285
nnU-Net checkpoint.  The prior arm adds no input channel and no inference-time
network block.  During training, an IEM-style module examines the 14 sigmoid
instance outputs and creates a stop-gradient violation map for:

1. spurious assignment of bone i inside a related bone j outside their true
   overlap; and
2. missing membership of either bone inside a genuine multi-label overlap.

The relation graph is estimated only from RAM training masks.  Validation GT
is used only for validation metrics, and the test split is never loaded.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from torch.utils.data import DataLoader

from train_r284_ramw600_overlap_prior import (
    BONE_NAMES,
    NnUNetMultiLabelPrior,
    WristDataset,
    dice_bce_logits,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_overlap_graph(root: Path, min_cases: int, max_pairs: int) -> tuple[list[tuple[int, int]], dict]:
    pixel_counts = np.zeros((14, 14), dtype=np.int64)
    case_counts = np.zeros((14, 14), dtype=np.int64)
    paths = sorted((root / "BoneSegmentation" / "masks" / "train").glob("*.npy"))
    for path in paths:
        # One 14x14 product replaces 91 full-resolution logical-and scans.
        # RAM masks are well below the int32 per-image pixel-count limit.
        flat = (np.load(path, mmap_mode="r") > 0).reshape(14, -1).astype(np.int32, copy=False)
        overlap = flat @ flat.T
        pixel_counts += overlap.astype(np.int64, copy=False)
        case_counts += (overlap > 0).astype(np.int64, copy=False)
    candidates = [(i, j) for i in range(14) for j in range(i + 1, 14)
                  if case_counts[i, j] >= min_cases and pixel_counts[i, j] > 0]
    pairs = sorted(candidates, key=lambda p: (int(case_counts[p]), int(pixel_counts[p])), reverse=True)[:max_pairs]
    audit = {
        "source_split": "train",
        "num_train_masks": len(paths),
        "min_overlap_cases": min_cases,
        "max_pairs": max_pairs,
        "pairs": [{"indices": [i, j], "bones": [BONE_NAMES[i], BONE_NAMES[j]],
                   "overlap_cases": int(case_counts[i, j]), "overlap_pixels": int(pixel_counts[i, j])}
                  for i, j in pairs],
    }
    if not pairs:
        raise RuntimeError("RAM training masks produced no valid overlap relation pairs")
    return pairs, audit


def explicit_overlap_iem_loss(logits: torch.Tensor, target: torch.Tensor,
                              pairs: list[tuple[int, int]]) -> tuple[torch.Tensor, dict[str, float]]:
    """Differentiable local loss on a hard, detached IEM-style violation map."""
    probability = torch.sigmoid(logits.float()).clamp(1e-5, 1 - 1e-5)
    truth = target.float()
    hard = probability.detach() >= 0.5
    critical = torch.zeros_like(truth, dtype=torch.bool)
    spurious = torch.zeros_like(critical)
    missing = torch.zeros_like(critical)

    for i, j in pairs:
        ti, tj = truth[:, i] > 0.5, truth[:, j] > 0.5
        genuine_overlap = ti & tj
        fp_i = hard[:, i] & tj & ~ti
        fp_j = hard[:, j] & ti & ~tj
        fn_i = ~hard[:, i] & genuine_overlap
        fn_j = ~hard[:, j] & genuine_overlap
        spurious[:, i] |= fp_i
        spurious[:, j] |= fp_j
        missing[:, i] |= fn_i
        missing[:, j] |= fn_j
    critical |= spurious | missing
    weight = critical.float()
    count = weight.sum()
    if count <= 0:
        zero = logits.sum() * 0.0
        return zero, {"critical_fraction": 0.0, "spurious_fraction": 0.0, "missing_fraction": 0.0}

    bce = F.binary_cross_entropy_with_logits(logits.float(), truth, reduction="none")
    local_bce = (bce * weight).sum() / count
    axes = (0, 2, 3)
    intersection = (probability * truth * weight).sum(axes)
    denominator = (probability * weight).sum(axes) + (truth * weight).sum(axes)
    valid = weight.sum(axes) > 0
    local_dice = 1.0 - ((2.0 * intersection[valid] + 1.0) / (denominator[valid] + 1.0)).mean()
    loss = 0.5 * local_bce + 0.5 * local_dice
    total = float(weight.numel())
    stats = {
        "critical_fraction": float(count.detach()) / total,
        "spurious_fraction": float(spurious.sum().detach()) / total,
        "missing_fraction": float(missing.sum().detach()) / total,
    }
    return loss, stats


def surfaces(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> tuple[float, float, bool]:
    if not target.any():
        return (1.0 if not pred.any() else 0.0), (0.0 if not pred.any() else float("nan")), bool(pred.any())
    if not pred.any():
        return 0.0, float("nan"), True
    ps = pred ^ binary_erosion(pred)
    ts = target ^ binary_erosion(target)
    dt = distance_transform_edt(~ts)
    dp = distance_transform_edt(~ps)
    denom = max(int(ps.sum() + ts.sum()), 1)
    nsd = float(((dt[ps] <= tolerance).sum() + (dp[ts] <= tolerance).sum()) / denom)
    distances = np.concatenate([dt[ps], dp[ts]])
    return nsd, float(distances.mean()), False


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             pairs: list[tuple[int, int]]) -> dict:
    model.eval()
    inter = torch.zeros(14, device=device)
    pred_mass = torch.zeros(14, device=device)
    true_mass = torch.zeros(14, device=device)
    tn = torch.zeros(14, device=device)
    neg = torch.zeros(14, device=device)
    ov_inter = ov_pred = ov_true = 0.0
    overlap_nsd, overlap_msd = [], []
    overlap_failures = 0
    pair_inter = torch.zeros(len(pairs), device=device)
    pair_pred = torch.zeros(len(pairs), device=device)
    pair_true = torch.zeros(len(pairs), device=device)
    for batch in loader:
        target = batch["mask"].to(device, non_blocking=True) > 0.5
        logits, _ = model(batch["image"].to(device, non_blocking=True), False)
        prediction = torch.sigmoid(logits) >= 0.5
        inter += (prediction & target).sum((0, 2, 3))
        pred_mass += prediction.sum((0, 2, 3))
        true_mass += target.sum((0, 2, 3))
        tn += (~prediction & ~target).sum((0, 2, 3))
        neg += (~target).sum((0, 2, 3))
        po = prediction.sum(1) >= 2
        to = target.sum(1) >= 2
        ov_inter += float((po & to).sum())
        ov_pred += float(po.sum())
        ov_true += float(to.sum())
        for b in range(po.shape[0]):
            n, m, failed = surfaces(po[b].cpu().numpy(), to[b].cpu().numpy())
            overlap_nsd.append(n)
            if np.isfinite(m): overlap_msd.append(m)
            overlap_failures += int(failed)
        for k, (i, j) in enumerate(pairs):
            pp, tt = prediction[:, i] & prediction[:, j], target[:, i] & target[:, j]
            pair_inter[k] += (pp & tt).sum()
            pair_pred[k] += pp.sum()
            pair_true[k] += tt.sum()
    class_dice = (2 * inter + 1) / (pred_mass + true_mass + 1)
    class_iou = (inter + 1) / (pred_mass + true_mass - inter + 1)
    sensitivity = (inter + 1) / (true_mass + 1)
    specificity = (tn + 1) / (neg + 1)
    overlap_dice = (2 * ov_inter + 1) / (ov_pred + ov_true + 1)
    overlap_iou = (ov_inter + 1) / (ov_pred + ov_true - ov_inter + 1)
    pair_dice = (2 * pair_inter + 1) / (pair_pred + pair_true + 1)
    return {
        "macro_dsc": float(class_dice.mean()),
        "macro_iou": float(class_iou.mean()),
        "voe": float(1.0 - class_iou.mean()),
        "macro_sensitivity": float(sensitivity.mean()),
        "macro_specificity": float(specificity.mean()),
        "overlap_dsc": float(overlap_dice),
        "overlap_iou": float(overlap_iou),
        "overlap_voe": float(1.0 - overlap_iou),
        "overlap_nsd_2px": float(np.mean(overlap_nsd)),
        "overlap_msd_px": float(np.mean(overlap_msd)) if overlap_msd else float("nan"),
        "overlap_msd_fail_rate": float(overlap_failures / max(len(overlap_nsd), 1)),
        "pair_overlap_dsc": pair_dice.cpu().tolist(),
        "class_dsc": class_dice.cpu().tolist(),
    }


def train_arm(name: str, initial_state: dict, train_set: WristDataset, val_loader: DataLoader,
              device: torch.device, output: Path, pairs: list[tuple[int, int]], epochs: int,
              learning_rate: float, prior_weight: float, min_epochs: int, patience: int,
              seed: int) -> tuple[Path, dict]:
    seed_all(seed)
    model = NnUNetMultiLabelPrior().to(device)
    model.load_state_dict(initial_state, strict=True)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    history_path = output / f"{name}_history.jsonl"
    checkpoint = output / f"{name}_best.pth"
    best = None
    bad = 0
    for epoch in range(1, epochs + 1):
        model.train()
        totals = {"loss": 0.0, "native": 0.0, "prior": 0.0, "critical": 0.0,
                  "spurious": 0.0, "missing": 0.0}
        started = time.time()
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, _ = model(image, False)
                native = dice_bce_logits(logits, target)
                if prior_weight > 0:
                    prior, stats = explicit_overlap_iem_loss(logits, target, pairs)
                else:
                    prior = native * 0.0
                    stats = {"critical_fraction": 0.0, "spurious_fraction": 0.0, "missing_fraction": 0.0}
                loss = native + prior_weight * prior
            if not torch.isfinite(loss):
                raise RuntimeError({"arm": name, "epoch": epoch, "native": float(native), "prior": float(prior)})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["native"] += float(native.detach())
            totals["prior"] += float(prior.detach())
            totals["critical"] += stats["critical_fraction"]
            totals["spurious"] += stats["spurious_fraction"]
            totals["missing"] += stats["missing_fraction"]
        scheduler.step()
        metrics = validate(model, val_loader, device, pairs)
        row = {"arm": name, "epoch": epoch, "seconds": time.time() - started,
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v / len(train_loader) for k, v in totals.items()}, **metrics}
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = (metrics["macro_dsc"], metrics["overlap_nsd_2px"], metrics["overlap_dsc"])
        if best is None or key > best[0]:
            best = (key, row)
            bad = 0
            torch.save({"model": model.state_dict(), "row": row, "arm": name,
                        "pairs": pairs, "prior_weight": prior_weight}, checkpoint)
        else:
            bad += 1
        if epoch >= min_epochs and bad >= patience:
            break
    return checkpoint, best[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/ram_w600/r324_nnunet_explicit_overlap_iem"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    # The MICCAI rib paper reports lambda=0.3, but its loss/network scale is not
    # transferable to a mature R285 checkpoint.  Eight RAM training batches
    # gave median ||g_native||/||g_prior||=0.006262; 0.001 therefore limits the
    # initial prior gradient to about 16% of the native gradient.
    parser.add_argument("--prior-weight", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=3241)
    parser.add_argument("--min-overlap-cases", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs, graph_audit = build_overlap_graph(args.dataset_root, args.min_overlap_cases, args.max_pairs)
    (args.output / "overlap_relation_graph.json").write_text(json.dumps(graph_audit, indent=2), encoding="utf-8")

    train_set = WristDataset(args.dataset_root, "train", args.size, True)
    val_set = WristDataset(args.dataset_root, "val", args.size, False)
    if args.limit_train: train_set.mask_files = train_set.mask_files[:args.limit_train]
    if args.limit_val: val_set.mask_files = val_set.mask_files[:args.limit_val]
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    initial = NnUNetMultiLabelPrior().to(device)
    initial.load_state_dict(payload["model"], strict=True)
    initial_state = copy.deepcopy(initial.state_dict())
    initialization = {"source": str(args.baseline_checkpoint), "sha256": sha256(args.baseline_checkpoint),
                      "strict_load": True, "same_state_for_both_arms": True}
    (args.output / "initialization_audit.json").write_text(json.dumps(initialization, indent=2), encoding="utf-8")

    if args.smoke:
        batch = next(iter(DataLoader(train_set, batch_size=1)))
        logits, _ = initial(batch["image"].to(device), False)
        target = batch["mask"].to(device)
        native = dice_bce_logits(logits, target)
        prior, stats = explicit_overlap_iem_loss(logits, target, pairs)
        loss = native + args.prior_weight * prior
        loss.backward()
        print(json.dumps({"smoke": True, "device": str(device), "native": float(native),
                          "prior": float(prior), "loss": float(loss), "finite": bool(torch.isfinite(loss)),
                          "pairs": len(pairs), **stats}, indent=2))
        return

    del initial
    torch.cuda.empty_cache()
    plain_ckpt, plain_best = train_arm("plain", initial_state, train_set, val_loader, device, args.output,
                                      pairs, args.epochs, args.learning_rate, 0.0,
                                      args.min_epochs, args.patience, args.seed)
    torch.cuda.empty_cache()
    prior_ckpt, prior_best = train_arm("explicit_overlap_iem", initial_state, train_set, val_loader, device,
                                      args.output, pairs, args.epochs, args.learning_rate, args.prior_weight,
                                      args.min_epochs, args.patience, args.seed)

    def evaluate_checkpoint(path: Path) -> dict:
        model = NnUNetMultiLabelPrior().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
        return validate(model, val_loader, device, pairs)

    plain_val = evaluate_checkpoint(plain_ckpt)
    prior_val = evaluate_checkpoint(prior_ckpt)
    numeric = ["macro_dsc", "macro_iou", "voe", "macro_sensitivity", "macro_specificity",
               "overlap_dsc", "overlap_iou", "overlap_voe", "overlap_nsd_2px",
               "overlap_msd_px", "overlap_msd_fail_rate"]
    delta = {k: prior_val[k] - plain_val[k] for k in numeric}
    result = {
        "experiment": "R324_RAM_NNUNET_EXPLICIT_OVERLAP_IEM",
        "split": "validation",
        "test_used": False,
        "threshold": 0.5,
        "threshold_search": False,
        "prior_position": "after multi-label logits, before training loss; absent at inference",
        "prior_weight": args.prior_weight,
        "initialization": initialization,
        "relation_graph": graph_audit,
        "plain_best": plain_best,
        "prior_best": prior_best,
        "plain_val": plain_val,
        "explicit_overlap_iem_val": prior_val,
        "delta_prior_minus_plain": delta,
        "checkpoints": {"plain": str(plain_ckpt), "prior": str(prior_ckpt)},
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
