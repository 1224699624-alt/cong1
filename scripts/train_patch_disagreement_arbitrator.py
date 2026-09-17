#!/usr/bin/env python3
"""Train a lightweight CNN to arbitrate anchor/candidate disagreement edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_anchor_pixel_residual import (
    candidate_exps_for,
    edit_zone,
    feature_stack,
    find_mask,
    image_path,
    names,
    read_gray,
    read_mask,
    resize_like,
    write_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch-level CNN disagreement arbitrator.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--apply-anchor-exp", default=None)
    parser.add_argument("--candidate-exps", nargs="+", required=True)
    parser.add_argument("--apply-candidate-exps", nargs="+", default=None)
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--feature-mode", default="basic", choices=["basic", "local_stats"])
    parser.add_argument("--zone-mode", default="anchor_candidate_disagreement", choices=["candidate_disagreement", "anchor_candidate_disagreement"])
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-tune-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--train-zone-radius", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--crops-per-image", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-scale", type=float, default=0.65)
    parser.add_argument("--zone-loss-weight", type=float, default=4.0)
    parser.add_argument("--boundary-radii", default="0,1,2")
    parser.add_argument("--prob-thresholds", default="0.50,0.55,0.60,0.65")
    parser.add_argument("--edit-margins", default="0.05,0.10,0.15")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260690)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def limited_names(raw_root: Path, dataset: str, split: str, limit: int) -> list[str]:
    out = names(raw_root, dataset, split)
    return out[:limit] if limit and limit > 0 else out


def load_item(
    roots: list[Path],
    raw_root: Path,
    dataset: str,
    split: str,
    name: str,
    anchor_exp: str,
    candidate_exps: list[str],
    source_dataset: str | None = None,
):
    gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
    image = read_gray(image_path(raw_root, dataset, split, name))
    anchor = resize_like(read_mask(find_mask(roots, anchor_exp, dataset, split, name, source_dataset)), gt.shape)
    candidates = [resize_like(read_mask(find_mask(roots, exp, dataset, split, name, source_dataset)), gt.shape) for exp in candidate_exps]
    return image, gt, anchor, candidates


class TinyPatchNet(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=2, dilation=2),
            nn.GroupNorm(4, base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=4, dilation=4),
            nn.GroupNorm(4, base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(base // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def crop_bounds(y: int, x: int, h: int, w: int, size: int) -> tuple[int, int, int, int]:
    half = size // 2
    y0 = min(max(0, y - half), max(0, h - size))
    x0 = min(max(0, x - half), max(0, w - size))
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    return y0, y1, x0, x1


def pad_crop(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int, size: int) -> np.ndarray:
    if arr.ndim == 3:
        crop = arr[:, y0:y1, x0:x1]
        if crop.shape[-2:] == (size, size):
            return crop
        out = np.zeros((crop.shape[0], size, size), dtype=crop.dtype)
        out[:, : crop.shape[1], : crop.shape[2]] = crop
    else:
        crop = arr[y0:y1, x0:x1]
        if crop.shape[-2:] == (size, size):
            return crop
        out = np.zeros((size, size), dtype=crop.dtype)
        out[: crop.shape[0], : crop.shape[1]] = crop
    return out


class CropDataset(Dataset):
    def __init__(self, args: argparse.Namespace, roots: list[Path]) -> None:
        self.args = args
        self.roots = roots
        self.raw_root = Path(args.train_raw_root)
        self.names = limited_names(self.raw_root, args.train_dataset, args.train_split, args.max_train_images)
        self.total = len(self.names) * args.crops_per_image

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        image_idx = index // self.args.crops_per_image
        crop_idx = index % self.args.crops_per_image
        name = self.names[image_idx]
        image, gt, anchor, candidates = load_item(
            self.roots,
            self.raw_root,
            self.args.train_dataset,
            self.args.train_split,
            name,
            self.args.anchor_exp,
            self.args.candidate_exps,
        )
        feat = feature_stack(image, anchor, candidates, self.args.feature_mode).transpose(2, 0, 1).astype(np.float32)
        zone = edit_zone(anchor, candidates, self.args.train_zone_radius, self.args.zone_mode)
        rng = np.random.default_rng(self.args.seed + index)
        idx_zone = np.flatnonzero(zone.ravel())
        h, w = gt.shape
        use_zone = len(idx_zone) > 0 and crop_idx < max(1, int(self.args.crops_per_image * 0.8))
        center_idx = int(rng.choice(idx_zone)) if use_zone else int(rng.integers(0, gt.size))
        y, x = divmod(center_idx, w)
        y0, y1, x0, x1 = crop_bounds(y, x, h, w, self.args.crop_size)
        x_crop = pad_crop(feat, y0, y1, x0, x1, self.args.crop_size)
        y_crop = pad_crop(gt.astype(np.float32), y0, y1, x0, x1, self.args.crop_size)
        weight = np.ones((self.args.crop_size, self.args.crop_size), dtype=np.float32)
        zc = pad_crop(zone.astype(np.float32), y0, y1, x0, x1, self.args.crop_size)
        weight += zc * float(self.args.zone_loss_weight)
        return torch.from_numpy(x_crop), torch.from_numpy(y_crop), torch.from_numpy(weight)


def estimate_positive_rate(dataset: CropDataset, sample_count: int = 256) -> float:
    n = min(len(dataset), sample_count)
    if n <= 0:
        return 0.03
    vals = []
    step = max(1, len(dataset) // n)
    for idx in range(0, len(dataset), step):
        _, y, _ = dataset[idx]
        vals.append(float(y.mean()))
        if len(vals) >= n:
            break
    return float(np.mean(vals)) if vals else 0.03


def train_model(args: argparse.Namespace, dataset: CropDataset, in_ch: int) -> TinyPatchNet:
    torch.manual_seed(args.seed)
    model = TinyPatchNet(in_ch, args.base_channels).to(args.device)
    pos = estimate_positive_rate(dataset)
    pos_weight = torch.tensor(args.pos_weight_scale * (1.0 - pos) / max(pos, 1e-4), device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=args.device.startswith("cuda"))
    for epoch in range(1, args.epochs + 1):
        losses = []
        model.train()
        for xb, yb, wb in tqdm(loader, desc=f"train/epoch{epoch}", leave=False):
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            wb = wb.to(args.device)
            logits = model(xb)
            loss_map = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            loss = (loss_map * wb).sum() / wb.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}), flush=True)
    return model


@torch.no_grad()
def predict_prob(model: TinyPatchNet, feat: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(feat.transpose(2, 0, 1)[None].astype(np.float32)).to(device)
    logits = model(x)
    return torch.sigmoid(logits)[0].cpu().numpy()


def collect_eval_cache(
    model: TinyPatchNet,
    args: argparse.Namespace,
    roots: list[Path],
    dataset: str,
    raw_root: Path,
    split: str,
    source_dataset: str | None,
    limit: int = 0,
):
    cache = []
    anchor_exp = args.apply_anchor_exp if dataset == args.apply_dataset and split == args.apply_split and args.apply_anchor_exp else args.anchor_exp
    candidate_exps = candidate_exps_for(args, dataset, split)
    for name in tqdm(limited_names(raw_root, dataset, split, limit), desc=f"cache/{dataset}/{split}"):
        image, gt, anchor, candidates = load_item(roots, raw_root, dataset, split, name, anchor_exp, candidate_exps, source_dataset)
        feat = feature_stack(image, anchor, candidates, args.feature_mode)
        prob = predict_prob(model, feat, args.device)
        cache.append({"name": name, "gt": gt, "anchor": anchor, "candidates": candidates, "prob": prob})
    return cache


def eval_cached(cache, threshold: float, margin: float, radius: int, boundary_kernel: int, zone_mode: str, write_dir: Path | None = None):
    records = []
    for item in tqdm(cache, desc=f"eval/cache/r{radius}/t{threshold:.2f}/m{margin:.2f}", leave=False):
        gt = item["gt"]
        anchor = item["anchor"]
        candidates = item["candidates"]
        prob = item["prob"]
        zones = item.setdefault("zones", {})
        if radius not in zones:
            zones[radius] = edit_zone(anchor, candidates, radius=radius, mode=zone_mode)
        zone = zones[radius]
        pred = anchor.copy()
        pred[zone & (prob >= threshold + margin)] = True
        pred[zone & (prob <= threshold - margin)] = False
        if write_dir is not None:
            write_mask(write_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel)
        rec["image"] = str(item["name"])
        records.append(rec)
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}, records


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    train_dataset = CropDataset(args, roots)
    if len(train_dataset) == 0:
        raise RuntimeError("empty training dataset")
    sample_x, _, _ = train_dataset[0]
    model = train_model(args, train_dataset, int(sample_x.shape[0]))
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "in_ch": int(sample_x.shape[0]), "args": vars(args)}, args.checkpoint)

    tune_cache = collect_eval_cache(
        model, args, roots, args.train_dataset, Path(args.train_raw_root), args.tune_split, None, args.max_tune_images
    )
    best = None
    for radius in parse_nums(args.boundary_radii, int):
        for threshold in parse_nums(args.prob_thresholds, float):
            for margin in parse_nums(args.edit_margins, float):
                mean, _ = eval_cached(tune_cache, threshold, margin, radius, args.boundary_kernel, args.zone_mode)
                item = {"radius": radius, "threshold": threshold, "margin": margin, "mean": mean}
                if best is None or mean["dice"] > best["mean"]["dice"]:
                    best = item
    assert best is not None

    out_dir = Path(args.ablations_roots[0]) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    apply_cache = collect_eval_cache(
        model,
        args,
        roots,
        args.apply_dataset,
        Path(args.apply_raw_root),
        args.apply_split,
        args.source_apply_dataset,
        args.max_apply_images,
    )
    mean, records = eval_cached(apply_cache, best["threshold"], best["margin"], best["radius"], args.boundary_kernel, args.zone_mode, out_dir)
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
    print(json.dumps({"mean": mean, "best": best}, indent=2))


if __name__ == "__main__":
    main()
