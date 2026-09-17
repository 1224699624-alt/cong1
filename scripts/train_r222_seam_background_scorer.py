#!/usr/bin/env python3
"""R222 GT-free seam/background scorer.

Train on original train with GT-derived R221 oracle cut targets, then apply on
original val using only image + R110 anchor-derived features. This is a
train/val development experiment only. It must not use clean-test-v2 for
threshold search or model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from audit_r221_instance_preserving_gap_oracle import (
    adjacent_instance_count,
    build_fast_gt_cache,
    component_match_metrics,
    fast_metrics,
)
from run_r201_unified_eval import read_binary_mask
from run_r209_component_preserving_feasibility_audit import read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/apply R222 seam-background scorer on original train/val.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--output-exp", default="r222_seam_background_scorer")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/r222_seam_background_scorer.pt"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/analysis/r222_seam_background_scorer_val_summary.json"))
    parser.add_argument("--per-image-csv", type=Path, default=Path("outputs/analysis/r222_seam_background_scorer_val_per_image.csv"))
    parser.add_argument("--diagnostic-json", type=Path, default=Path(""))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--positive-oversample", type=int, default=2048)
    parser.add_argument("--oracle-gap-kernel", type=int, default=13)
    parser.add_argument("--oracle-max-cut-frac", type=float, default=0.006)
    parser.add_argument("--eligible-dist-percentile", type=float, default=20.0)
    parser.add_argument("--eligible-band-radius", type=int, default=5)
    parser.add_argument("--min-adjacent-instances", type=int, default=2)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--max-cut-fracs", default="0.0015,0.003,0.006")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=202607222)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume-checkpoint", default="")
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list[Any]:
    return [cast(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, split: str, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / split / "masks" / name


def normalized_distance(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist_in = ndimage.distance_transform_edt(mask).astype(np.float32)
    dist_out = ndimage.distance_transform_edt(~mask).astype(np.float32)
    signed = dist_in - dist_out
    denom = max(1.0, float(np.percentile(np.abs(signed), 95)))
    return (
        np.clip(dist_in / max(1.0, float(np.percentile(dist_in[mask], 95)) if mask.any() else 1.0), 0.0, 1.0),
        np.clip(dist_out / max(1.0, float(np.percentile(dist_out[~mask], 95)) if (~mask).any() else 1.0), 0.0, 1.0),
        np.clip(signed / denom, -1.0, 1.0),
    )


def feature_stack(image: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    anchor_f = anchor.astype(np.float32)
    gy, gx = np.gradient(image)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    if float(grad.max()) > 0:
        grad /= float(grad.max())
    dist_in, dist_out, signed = normalized_distance(anchor)
    boundary = ndimage.binary_dilation(anchor, structure=np.ones((3, 3), dtype=bool)) ^ ndimage.binary_erosion(
        anchor, structure=np.ones((3, 3), dtype=bool)
    )
    local = []
    for size in (3, 7, 15):
        img_mean = ndimage.uniform_filter(image, size=size, mode="nearest").astype(np.float32)
        anc_mean = ndimage.uniform_filter(anchor_f, size=size, mode="nearest").astype(np.float32)
        local.extend([img_mean, anc_mean])
    yy, xx = np.indices(anchor.shape, dtype=np.float32)
    yy = yy / max(1.0, float(anchor.shape[0] - 1))
    xx = xx / max(1.0, float(anchor.shape[1] - 1))
    return np.stack(
        [
            image,
            anchor_f,
            grad,
            dist_in.astype(np.float32),
            dist_out.astype(np.float32),
            signed.astype(np.float32),
            boundary.astype(np.float32),
            yy,
            xx,
            *local,
        ],
        axis=-1,
    ).astype(np.float32)


def seam_eligible_region(anchor: np.ndarray, image: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if not anchor.any():
        return np.zeros_like(anchor, dtype=bool)
    dist_in = ndimage.distance_transform_edt(anchor)
    vals = dist_in[anchor]
    shallow_thr = max(1.0, float(np.percentile(vals, args.eligible_dist_percentile))) if vals.size else 1.0
    shallow = anchor & (dist_in <= shallow_thr)
    boundary_band = ndimage.binary_dilation(anchor, structure=np.ones((2 * args.eligible_band_radius + 1, 2 * args.eligible_band_radius + 1), dtype=bool)) ^ ndimage.binary_erosion(
        anchor, structure=np.ones((2 * args.eligible_band_radius + 1, 2 * args.eligible_band_radius + 1), dtype=bool)
    )
    bg_contact = ndimage.binary_dilation(~anchor, structure=np.ones((5, 5), dtype=bool))
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    grad_support = np.ones_like(anchor, dtype=bool)
    if np.logical_and(anchor, bg_contact).any():
        gvals = grad[np.logical_and(anchor, bg_contact)]
        if gvals.size:
            grad_support = grad >= float(np.percentile(gvals, 35))
    return anchor & shallow & boundary_band & bg_contact & grad_support


def oracle_cut(anchor: np.ndarray, instance: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    gt = instance > 0
    structure = np.ones((args.oracle_gap_kernel, args.oracle_gap_kernel), dtype=bool)
    gap = np.logical_and(ndimage.binary_dilation(gt, structure=structure), ~gt)
    base_cut = np.logical_and(anchor, gap)
    labels, n_labels = ndimage.label(base_cut, structure=np.ones((3, 3), dtype=np.uint8))
    between = np.zeros_like(base_cut, dtype=bool)
    for component_id in range(1, n_labels + 1):
        comp = labels == component_id
        if int(comp.sum()) < args.min_cut_area:
            continue
        if adjacent_instance_count(comp, instance) >= args.min_adjacent_instances:
            between |= comp
    max_pixels = max(args.min_cut_area, int(round(max(1, int(anchor.sum())) * args.oracle_max_cut_frac)))
    guarded = np.zeros_like(between, dtype=bool)
    labels2, n_labels2 = ndimage.label(between, structure=np.ones((3, 3), dtype=np.uint8))
    comps: list[tuple[int, int, np.ndarray]] = []
    for component_id in range(1, n_labels2 + 1):
        comp = labels2 == component_id
        area = int(comp.sum())
        if area >= args.min_cut_area:
            comps.append((adjacent_instance_count(comp, instance), area, comp))
    comps.sort(key=lambda item: (item[0], item[1]), reverse=True)
    used = 0
    for _adj, area, comp in comps:
        if used + area > max_pixels:
            continue
        guarded |= comp
        used += area
    if np.logical_and(guarded, gt).any():
        raise AssertionError("R222 oracle cut intersects GT foreground")
    return guarded


def sample_training(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    missing: list[str] = []
    stats = {"positive_pixels": 0, "sampled_positive": 0, "sampled_negative": 0}
    split = args.train_split
    train_names = names(args.raw_root, args.dataset, split)[: args.limit_train or None]
    for name in tqdm(train_names, desc="r222/sample/train"):
        mask_path = anchor_path(args, split, name)
        if not mask_path.exists():
            missing.append(name)
            continue
        instance = read_instance(args.raw_root / args.dataset / f"{split}_labels" / name)
        gt_shape = instance.shape
        anchor = resize_like(read_binary_mask(mask_path), gt_shape)
        image = read_gray(image_path(args.raw_root, args.dataset, split, name))
        feat = feature_stack(image, anchor)
        eligible = seam_eligible_region(anchor, image, args)
        cut = np.logical_and(oracle_cut(anchor, instance, args), eligible)
        pos = np.flatnonzero(cut.ravel())
        neg = np.flatnonzero(np.logical_and(eligible, ~cut).ravel())
        bg = np.flatnonzero((~eligible).ravel())
        stats["positive_pixels"] += int(pos.size)
        pick_parts = []
        if pos.size:
            n_pos = min(args.positive_oversample, max(args.min_cut_area, pos.size * 4))
            pick_parts.append(rng.choice(pos, size=n_pos, replace=pos.size < n_pos))
            stats["sampled_positive"] += int(n_pos)
        n_neg = max(0, args.samples_per_image - sum(len(part) for part in pick_parts))
        neg_pool = np.concatenate([neg, bg]) if bg.size else neg
        if n_neg > 0 and neg_pool.size:
            pick_parts.append(rng.choice(neg_pool, size=n_neg, replace=neg_pool.size < n_neg))
            stats["sampled_negative"] += int(n_neg)
        if not pick_parts:
            continue
        idx = np.concatenate(pick_parts)
        xs.append(feat.reshape(-1, feat.shape[-1])[idx])
        ys.append(cut.reshape(-1)[idx].astype(np.float32))
    if not xs:
        raise RuntimeError("No R222 training samples were collected.")
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32), {"missing_train_anchors": missing, **stats}


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> MLP:
    torch.manual_seed(args.seed)
    model = MLP(x.shape[1], args.hidden).to(args.device)
    pos = max(float(y.mean()), 1e-5)
    pos_weight = torch.tensor([(1.0 - pos) / pos], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y))
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = xt[idx].to(args.device)
            yb = yt[idx].to(args.device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "pos_rate": pos}), flush=True)
    return model


def load_checkpoint(path: Path, args: argparse.Namespace) -> MLP:
    state = torch.load(path, map_location=args.device)
    model = MLP(int(state["in_dim"]), int(state["hidden"])).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_prob(model: MLP, feat: np.ndarray, device: str, batch: int = 262144) -> np.ndarray:
    flat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
    outs = []
    model.eval()
    for start in range(0, len(flat), batch):
        logits = model(torch.from_numpy(flat[start : start + batch]).to(device))
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs).reshape(feat.shape[:2])


def apply_cut(anchor: np.ndarray, image: np.ndarray, prob: np.ndarray, threshold: float, max_cut_frac: float, min_cut_area: int, args: argparse.Namespace) -> np.ndarray:
    eligible_region = seam_eligible_region(anchor, image, args)
    eligible = eligible_region & (prob >= threshold)
    max_pixels = max(min_cut_area, int(round(max(1, int(anchor.sum())) * max_cut_frac)))
    if int(eligible.sum()) <= max_pixels:
        cut = eligible
    else:
        ys, xs = np.where(eligible)
        vals = prob[ys, xs]
        order = np.argsort(vals)[::-1][:max_pixels]
        cut = np.zeros_like(anchor, dtype=bool)
        cut[ys[order], xs[order]] = True
    cut = ndimage.binary_opening(cut, structure=np.ones((2, 2), dtype=bool))
    if int(cut.sum()) < min_cut_area:
        return anchor.copy()
    return np.logical_and(anchor, ~cut)


def mean_record(records: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({key for row in records for key in row if key != "image"})
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [row.get(key) for row in records]
        finite = [float(value) for value in vals if isinstance(value, (float, int)) and np.isfinite(float(value))]
        out[key] = float(np.mean(finite)) if finite else None
    return out


def diagnostic_records(args: argparse.Namespace, cache: list[dict[str, Any]], thresholds: list[float]) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    open_structure = np.ones((2, 2), dtype=bool)
    for item in tqdm(cache, desc="r222/diagnostic", leave=False):
        name = str(item["name"])
        instance = item["instance"]
        anchor = item["anchor"]
        image = item["image"]
        prob = item["prob"]
        eligible = seam_eligible_region(anchor, image, args)
        oracle = np.logical_and(oracle_cut(anchor, instance, args), eligible)
        eligible_prob = prob[eligible]
        row: dict[str, Any] = {
            "image": name,
            "anchor_pixels": float(anchor.sum()),
            "eligible_pixels": float(eligible.sum()),
            "eligible_frac_anchor": float(eligible.sum() / max(1, int(anchor.sum()))),
            "oracle_cut_pixels_after_eligible": float(oracle.sum()),
            "oracle_cut_frac_eligible": float(oracle.sum() / max(1, int(eligible.sum()))),
        }
        if eligible_prob.size:
            for pct in (50, 75, 90, 95, 99, 99.5, 99.9, 100):
                row[f"eligible_prob_p{str(pct).replace('.', '_')}"] = float(np.percentile(eligible_prob, pct))
        else:
            for pct in (50, 75, 90, 95, 99, 99.5, 99.9, 100):
                row[f"eligible_prob_p{str(pct).replace('.', '_')}"] = None
        for threshold in thresholds:
            raw = eligible & (prob >= threshold)
            opened = ndimage.binary_opening(raw, structure=open_structure)
            row[f"raw_ge_{threshold:g}"] = float(raw.sum())
            row[f"opened_ge_{threshold:g}"] = float(opened.sum())
        rows.append(row)
    return mean_record(rows), rows


def evaluate_config(
    args: argparse.Namespace,
    model: MLP,
    threshold: float,
    max_cut_frac: float,
    gt_cache: dict[str, Any],
    write_dir: Path | None = None,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    split = args.val_split
    val_names = names(args.raw_root, args.dataset, split)[: args.limit_val or None]
    records: list[dict[str, Any]] = []
    for name in tqdm(val_names, desc=f"r222/eval/t{threshold:g}/f{max_cut_frac:g}", leave=False):
        mask_path = anchor_path(args, split, name)
        if not mask_path.exists():
            continue
        instance = read_instance(args.raw_root / args.dataset / f"{split}_labels" / name)
        gt = instance > 0
        anchor = resize_like(read_binary_mask(mask_path), gt.shape)
        image = read_gray(image_path(args.raw_root, args.dataset, split, name))
        feat = feature_stack(image, anchor)
        prob = predict_prob(model, feat, args.device)
        pred = apply_cut(anchor, image, prob, threshold, max_cut_frac, args.min_cut_area, args)
        if write_dir is not None:
            write_mask(write_dir / name, pred)
        anchor_metrics = fast_metrics(anchor, gt, gt_cache[name], args.boundary_kernel)
        pred_metrics = fast_metrics(pred, gt, gt_cache[name], args.boundary_kernel)
        anchor_match = component_match_metrics(anchor, instance, 0.05)
        pred_match = component_match_metrics(pred, instance, 0.05)
        row: dict[str, Any] = {
            "image": name,
            "threshold": threshold,
            "max_cut_frac": max_cut_frac,
            "cut_pixels": float(np.logical_and(anchor, ~pred).sum()),
        }
        for key, value in anchor_metrics.items():
            row[f"anchor_{key}"] = value
        for key, value in pred_metrics.items():
            row[f"r222_{key}"] = value
            row[f"delta_{key}"] = None if anchor_metrics[key] is None or value is None else float(value) - float(anchor_metrics[key])
        for key, value in anchor_match.items():
            row[f"anchor_{key}"] = value
        for key, value in pred_match.items():
            row[f"r222_{key}"] = value
            row[f"delta_{key}"] = float(value) - float(anchor_match[key])
        records.append(row)
    return mean_record(records), records


def collect_val_cache(args: argparse.Namespace, model: MLP, gt_cache: dict[str, Any]) -> list[dict[str, Any]]:
    split = args.val_split
    val_names = names(args.raw_root, args.dataset, split)[: args.limit_val or None]
    cache: list[dict[str, Any]] = []
    for name in tqdm(val_names, desc="r222/cache/val"):
        mask_path = anchor_path(args, split, name)
        if not mask_path.exists():
            continue
        instance = read_instance(args.raw_root / args.dataset / f"{split}_labels" / name)
        gt = instance > 0
        anchor = resize_like(read_binary_mask(mask_path), gt.shape)
        image = read_gray(image_path(args.raw_root, args.dataset, split, name))
        feat = feature_stack(image, anchor)
        prob = predict_prob(model, feat, args.device)
        anchor_metrics = fast_metrics(anchor, gt, gt_cache[name], args.boundary_kernel)
        anchor_match = component_match_metrics(anchor, instance, 0.05)
        cache.append(
            {
                "name": name,
                "instance": instance,
                "gt": gt,
                "anchor": anchor,
                "image": image,
                "prob": prob,
                "gt_cache": gt_cache[name],
                "anchor_metrics": anchor_metrics,
                "anchor_match": anchor_match,
            }
        )
    return cache


def evaluate_cached_config(
    args: argparse.Namespace,
    cache: list[dict[str, Any]],
    threshold: float,
    max_cut_frac: float,
    write_dir: Path | None = None,
) -> tuple[dict[str, float | None], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for item in tqdm(cache, desc=f"r222/eval-cache/t{threshold:g}/f{max_cut_frac:g}", leave=False):
        name = str(item["name"])
        instance = item["instance"]
        gt = item["gt"]
        anchor = item["anchor"]
        image = item["image"]
        prob = item["prob"]
        gt_cache = item["gt_cache"]
        anchor_metrics = item["anchor_metrics"]
        anchor_match = item["anchor_match"]
        pred = apply_cut(anchor, image, prob, threshold, max_cut_frac, args.min_cut_area, args)
        if write_dir is not None:
            write_mask(write_dir / name, pred)
        pred_metrics = fast_metrics(pred, gt, gt_cache, args.boundary_kernel)
        pred_match = component_match_metrics(pred, instance, 0.05)
        row: dict[str, Any] = {
            "image": name,
            "threshold": threshold,
            "max_cut_frac": max_cut_frac,
            "cut_pixels": float(np.logical_and(anchor, ~pred).sum()),
        }
        for key, value in anchor_metrics.items():
            row[f"anchor_{key}"] = value
        for key, value in pred_metrics.items():
            row[f"r222_{key}"] = value
            row[f"delta_{key}"] = None if anchor_metrics[key] is None or value is None else float(value) - float(anchor_metrics[key])
        for key, value in anchor_match.items():
            row[f"anchor_{key}"] = value
        for key, value in pred_match.items():
            row[f"r222_{key}"] = value
            row[f"delta_{key}"] = float(value) - float(anchor_match[key])
        records.append(row)
    return mean_record(records), records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gate_pass(mean: dict[str, float | None]) -> bool:
    return bool(
        float(mean.get("delta_recall") or 0.0) >= -1e-4
        and float(mean.get("delta_dice") or 0.0) >= -5e-4
        and float(mean.get("delta_iou") or 0.0) >= -8e-4
        and float(mean.get("delta_boundary_iou") or 0.0) > 0.0
        and float(mean.get("delta_boundary_f1") or 0.0) > 0.0
        and float(mean.get("delta_gap_region_fp_rate") or 0.0) < 0.0
        and float(mean.get("delta_component_count_mae") or 0.0) <= 0.0
    )


def main() -> None:
    args = parse_args()
    resume_checkpoint = Path(args.resume_checkpoint) if str(args.resume_checkpoint).strip() else None
    if resume_checkpoint is not None and resume_checkpoint.exists():
        model = load_checkpoint(resume_checkpoint, args)
        train_info = {"resume_checkpoint": str(resume_checkpoint)}
    else:
        x, y, train_info = sample_training(args)
        model = train_model(args, x, y)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "in_dim": x.shape[1], "hidden": args.hidden, "args": vars(args)}, args.checkpoint)
        train_info.update({"num_samples": int(len(y)), "positive_sample_rate": float(y.mean()), "checkpoint": str(args.checkpoint)})

    gt_dir = args.raw_root / args.dataset / f"{args.val_split}_labels"
    gt_cache = build_fast_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)
    val_cache = collect_val_cache(args, model, gt_cache)
    thresholds = parse_nums(args.thresholds, float)
    diag_mean, diag_rows = diagnostic_records(args, val_cache, thresholds)
    grid = []
    best: dict[str, Any] | None = None
    best_records: list[dict[str, Any]] = []
    for threshold in thresholds:
        for max_cut_frac in parse_nums(args.max_cut_fracs, float):
            mean, records = evaluate_cached_config(args, val_cache, threshold, max_cut_frac)
            item = {"threshold": threshold, "max_cut_frac": max_cut_frac, "mean": mean, "gate_pass": gate_pass(mean)}
            grid.append(item)
            score = (
                int(item["gate_pass"]),
                mean.get("delta_boundary_iou") or -999.0,
                -(mean.get("delta_gap_region_fp_rate") or 999.0),
                -(mean.get("delta_component_count_mae") or 999.0),
                mean.get("delta_dice") or -999.0,
            )
            if best is None or score > best["_score"]:
                best = {**item, "_score": score}
                best_records = records
    assert best is not None
    best.pop("_score", None)
    output_dir = args.ablations_root / args.output_exp / args.dataset / args.val_split / "masks"
    _mean, best_records = evaluate_cached_config(args, val_cache, best["threshold"], best["max_cut_frac"], output_dir)
    write_csv(args.per_image_csv, best_records)
    summary = {
        "run_id": "R222-seam-background-scorer",
        "dataset": args.dataset,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "anchor_exp": args.anchor_exp,
        "output_exp": args.output_exp,
        "clean_test_v2_used": False,
        "evidence_level": "train_val_mask_level_development",
        "metric_mode": "fast_r201_overlap_boundary_gap_component; surface metrics deferred",
        "writes_masks": True,
        "train_info": train_info,
        "seam_eligible": {
            "eligible_dist_percentile": args.eligible_dist_percentile,
            "eligible_band_radius": args.eligible_band_radius,
            "oracle_gap_kernel": args.oracle_gap_kernel,
            "oracle_max_cut_frac": args.oracle_max_cut_frac,
        },
        "best": best,
        "grid": grid,
        "diagnostic_mean": diag_mean,
        "num_val_records": len(best_records),
        "output_mask_dir": str(output_dir),
        "warning": "Original train/val development only; do not treat as clean-test-v2 R201 evidence.",
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if str(args.diagnostic_json):
        args.diagnostic_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostic_json.write_text(json.dumps({"mean": diag_mean, "per_image": diag_rows}, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "summary_json": str(args.summary_json), "per_image_csv": str(args.per_image_csv)}, indent=2))


if __name__ == "__main__":
    main()
