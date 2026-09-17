#!/usr/bin/env python3
"""Train R177 local bridge/boundary separation arbitrator.

The model is deliberately small and anchored: it edits only local risk zones around
an existing mask. Train/validation use original train/val labels. clean-test-v2 is
used only after validation selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R177 separation arbitrator.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", default="r100_like_r025b_r097_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--manifest-csv", default="outputs/analysis/r177_bridge_risk_manifest/r177_bridge_risk_manifest.csv")
    parser.add_argument("--output-exp", default="r177_boundary_preserving_separation_arbitrator")
    parser.add_argument("--checkpoint", default="outputs/r177_separation_arbitrator/r177_boundary_preserving_separation_arbitrator.pt")
    parser.add_argument("--metrics-json", default="outputs/analysis/r177_boundary_preserving_separation_arbitrator_clean_test_v2_metrics.json")
    parser.add_argument("--val-summary-json", default="outputs/analysis/r177_boundary_preserving_separation_arbitrator_val_summary.json")
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--samples-per-risk-image", type=int, default=8192)
    parser.add_argument("--samples-per-normal-image", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--risk-radius", type=int, default=9)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--zone-radii", default="3,5,7")
    parser.add_argument("--prob-thresholds", default="0.45,0.50,0.55,0.60")
    parser.add_argument("--edit-margins", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=202607177)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def find_anchor_mask(root: Path, exp: str, dataset: str, split: str, name: str) -> Path:
    path = root / exp / dataset / split / "masks" / name
    if not path.exists():
        raise FileNotFoundError(f"Anchor mask not found: {path}")
    return path


def component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask)
    return int(n)


def boundary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def separation_band(gt_or_anchor: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((radius, radius), dtype=bool)
    return np.logical_and(ndimage.binary_dilation(gt_or_anchor, structure=structure), ~gt_or_anchor)


def risk_zone(anchor: np.ndarray, radius: int, boundary_radius: int) -> np.ndarray:
    sep = separation_band(anchor, radius)
    bnd = boundary(anchor, boundary_radius)
    return np.logical_or(sep, bnd)


def feature_stack(image: np.ndarray, anchor: np.ndarray, radius: int, boundary_radius: int) -> np.ndarray:
    anchor_f = anchor.astype(np.float32)
    dist_in = ndimage.distance_transform_edt(anchor)
    dist_out = ndimage.distance_transform_edt(~anchor)
    signed = dist_in - dist_out
    denom = max(1.0, float(np.percentile(np.abs(signed), 95)))
    signed = np.clip(signed / denom, -1.0, 1.0).astype(np.float32)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0:
        grad = grad / float(grad.max())
    sep = separation_band(anchor, radius).astype(np.float32)
    bnd = boundary(anchor, boundary_radius).astype(np.float32)
    local_anchor_9 = ndimage.uniform_filter(anchor_f, size=9, mode="nearest")
    local_img_9 = ndimage.uniform_filter(image.astype(np.float32), size=9, mode="nearest")
    local_img_21 = ndimage.uniform_filter(image.astype(np.float32), size=21, mode="nearest")
    return np.stack(
        [
            image.astype(np.float32),
            anchor_f,
            signed,
            grad.astype(np.float32),
            sep,
            bnd,
            local_anchor_9.astype(np.float32),
            local_img_9.astype(np.float32),
            local_img_21.astype(np.float32),
        ],
        axis=-1,
    )


def read_manifest(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["split"], {})[row["image"]] = row
    return out


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_names(raw_root: Path, dataset: str, split: str, limit: int) -> list[str]:
    out = names(raw_root, dataset, split)
    return out[:limit] if limit and limit > 0 else out


def load_case(args: argparse.Namespace, dataset: str, raw_root: Path, split: str, name: str, exp: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
    image = read_gray(image_path(raw_root, dataset, split, name))
    anchor = resize_like(read_mask(find_anchor_mask(Path(args.ablations_root), exp, dataset, split, name)), gt.shape)
    return image, gt, anchor


def sample_training(args: argparse.Namespace, manifest: dict[str, dict[str, dict[str, str]]]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    xs, ys = [], []
    train_root = Path(args.train_raw_root)
    train_manifest = manifest.get(args.train_split, {})
    for name in tqdm(split_names(train_root, args.train_dataset, args.train_split, args.max_train_images), desc="sample/r177"):
        image, gt, anchor = load_case(args, args.train_dataset, train_root, args.train_split, name, args.anchor_exp)
        feat = feature_stack(image, anchor, args.risk_radius, args.boundary_radius)
        zone = risk_zone(anchor, args.risk_radius, args.boundary_radius)
        is_risk = int(train_manifest.get(name, {}).get("bridge_risk", "0") or 0) > 0
        samples = args.samples_per_risk_image if is_risk else args.samples_per_normal_image
        idx_zone = np.flatnonzero(zone.ravel())
        idx_error = np.flatnonzero((anchor != gt).ravel())
        idx_all = np.arange(gt.size)
        picks = []
        n_error = min(len(idx_error), samples // 2)
        if n_error > 0:
            picks.append(rng.choice(idx_error, size=n_error, replace=len(idx_error) < n_error))
        n_zone = min(len(idx_zone), samples - sum(len(p) for p in picks))
        if n_zone > 0:
            picks.append(rng.choice(idx_zone, size=n_zone, replace=len(idx_zone) < n_zone))
        n_rest = samples - sum(len(p) for p in picks)
        if n_rest > 0:
            picks.append(rng.choice(idx_all, size=n_rest, replace=gt.size < n_rest))
        idx = np.concatenate(picks)
        xs.append(feat.reshape(-1, feat.shape[-1])[idx])
        ys.append(gt.reshape(-1)[idx].astype(np.float32))
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def train_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> MLP:
    torch.manual_seed(args.seed)
    model = MLP(x.shape[1], args.hidden).to(args.device)
    pos = float(y.mean())
    pos_weight = torch.tensor([args.pos_weight_scale * (1.0 - pos) / max(pos, 1e-4)], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(args.seed + 17)
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


@torch.no_grad()
def predict_prob(model: MLP, feat: np.ndarray, device: str, batch: int = 262144) -> np.ndarray:
    flat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
    outs = []
    model.eval()
    for start in range(0, len(flat), batch):
        logits = model(torch.from_numpy(flat[start : start + batch]).to(device))
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs).reshape(feat.shape[:2])


def add_structure_metrics(rec: dict[str, float], pred: np.ndarray, gt: np.ndarray, radius: int) -> dict[str, float]:
    pred_components = component_count(pred)
    gt_components = component_count(gt)
    sep = separation_band(gt, radius)
    sep_fp = float(np.logical_and(pred, sep).sum())
    sep_pixels = float(sep.sum())
    rec["pred_component_count"] = float(pred_components)
    rec["gt_component_count"] = float(gt_components)
    rec["component_count_error"] = float(abs(pred_components - gt_components))
    rec["false_bridge_flag"] = float(pred_components < gt_components)
    rec["sep_fp_rate"] = float(sep_fp / sep_pixels) if sep_pixels > 0 else 0.0
    return rec


def eval_cache(
    model: MLP,
    args: argparse.Namespace,
    dataset: str,
    raw_root: Path,
    split: str,
    exp: str,
    limit: int,
) -> list[dict[str, object]]:
    cache = []
    for name in tqdm(split_names(raw_root, dataset, split, limit), desc=f"cache/{dataset}/{split}"):
        image, gt, anchor = load_case(args, dataset, raw_root, split, name, exp)
        feat = feature_stack(image, anchor, args.risk_radius, args.boundary_radius)
        prob = predict_prob(model, feat, args.device)
        cache.append({"name": name, "gt": gt, "anchor": anchor, "prob": prob})
    return cache


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([float(r[k]) for r in records])) for k in keys}


def eval_cached(
    cache: list[dict[str, object]],
    threshold: float,
    margin: float,
    zone_radius: int,
    boundary_radius: int,
    boundary_kernel: int,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for item in tqdm(cache, desc=f"eval/t{threshold:.2f}/m{margin:.2f}/z{zone_radius}", leave=False):
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        prob = item["prob"]  # type: ignore[assignment]
        zone_key = f"zone_{zone_radius}"
        if zone_key not in item:
            item[zone_key] = risk_zone(anchor, zone_radius, boundary_radius)
        zone = item[zone_key]  # type: ignore[assignment]
        pred = anchor.copy()
        pred[zone & (prob >= threshold + margin)] = True
        pred[zone & (prob <= threshold - margin)] = False
        rec = compute_metrics(pred, gt, boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt, zone_radius)
        rec["image"] = str(item["name"])
        records.append(rec)
        if write_dir is not None:
            write_mask(write_dir / str(item["name"]), pred)
    return mean_records(records), records


def main() -> None:
    args = parse_args()
    manifest = read_manifest(Path(args.manifest_csv))
    x, y = sample_training(args, manifest)
    model = train_model(args, x, y)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "in_dim": x.shape[1], "args": vars(args)}, args.checkpoint)

    val_cache = eval_cache(model, args, args.train_dataset, Path(args.train_raw_root), args.tune_split, args.anchor_exp, args.max_val_images)
    best = None
    val_grid = []
    for zone_radius in parse_nums(args.zone_radii, int):
        for threshold in parse_nums(args.prob_thresholds, float):
            for margin in parse_nums(args.edit_margins, float):
                mean, _ = eval_cached(val_cache, threshold, margin, zone_radius, args.boundary_radius, args.boundary_kernel)
                item = {"zone_radius": zone_radius, "threshold": threshold, "margin": margin, "mean": mean}
                val_grid.append(item)
                if best is None:
                    best = item
                    continue
                # Primary guard: Dice. Tie-break with structure and boundary quality.
                score = (mean["dice"], -mean["component_count_error"], -mean["false_bridge_flag"], mean["boundary_iou"])
                best_mean = best["mean"]
                best_score = (
                    best_mean["dice"],
                    -best_mean["component_count_error"],
                    -best_mean["false_bridge_flag"],
                    best_mean["boundary_iou"],
                )
                if score > best_score:
                    best = item
    assert best is not None
    val_summary = {
        "dataset": args.train_dataset,
        "split": args.tune_split,
        "num_evaluated": len(val_cache),
        "best": best,
        "grid": val_grid,
    }
    Path(args.val_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_summary_json).write_text(json.dumps(val_summary, indent=2), encoding="utf-8")

    apply_cache = eval_cache(model, args, args.apply_dataset, Path(args.apply_raw_root), args.apply_split, args.apply_anchor_exp, args.max_apply_images)
    out_dir = Path(args.ablations_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = eval_cached(
        apply_cache,
        float(best["threshold"]),
        float(best["margin"]),
        int(best["zone_radius"]),
        args.boundary_radius,
        args.boundary_kernel,
        out_dir,
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
    print(json.dumps({"val_best": best, "apply_mean": mean}, indent=2))


if __name__ == "__main__":
    main()
