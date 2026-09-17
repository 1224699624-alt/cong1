#!/usr/bin/env python3
"""Train a tiny pixel MLP to edit an anchor mask only in local uncertain regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor-local pixel residual classifier.")
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
    parser.add_argument(
        "--apply-anchor-exp",
        default=None,
        help="Anchor mask experiment used for apply-dataset evaluation. Defaults to --anchor-exp.",
    )
    parser.add_argument("--candidate-exps", nargs="+", required=True)
    parser.add_argument(
        "--apply-candidate-exps",
        nargs="+",
        default=None,
        help="Candidate mask experiments used on apply-dataset. Defaults to --candidate-exps.",
    )
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume-checkpoint", default=None, help="Load a trained pixel residual checkpoint and skip training.")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--boundary-radii", default="2,4,6,8")
    parser.add_argument("--prob-thresholds", default="0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--edit-margins", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument(
        "--feature-mode",
        default="basic",
        choices=["basic", "local_stats"],
        help="Feature set for pixel residual learning.",
    )
    parser.add_argument(
        "--zone-mode",
        default="candidate_boundary",
        choices=["candidate_boundary", "candidate_disagreement", "anchor_candidate_disagreement"],
        help="Pixels eligible for sampling/editing.",
    )
    parser.add_argument("--train-zone-radius", type=int, default=6)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list:
    return [cast(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def names(raw_root: Path, dataset: str, split: str) -> list[str]:
    return sorted(p.name for p in (raw_root / dataset / f"{split}_labels").glob("*.png"))


def image_path(raw_root: Path, dataset: str, split: str, name: str) -> Path:
    stem = Path(name).stem
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        p = raw_root / dataset / split / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"image not found for {name}")


def find_mask(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None = None) -> Path:
    ds_list = [dataset]
    if source_dataset and source_dataset not in ds_list:
        ds_list.append(source_dataset)
    for root in roots:
        for ds in ds_list:
            p = root / exp / ds / split / "masks" / name
            if p.exists():
                return p
    raise FileNotFoundError(f"mask not found: {exp} {dataset}/{split}/{name}")


def resize_like(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(target_shape[::-1], Image.NEAREST)) > 0


def local_mean(arr: np.ndarray, size: int) -> np.ndarray:
    return ndimage.uniform_filter(arr.astype(np.float32), size=size, mode="nearest")


def feature_stack(image: np.ndarray, anchor: np.ndarray, candidates: list[np.ndarray], mode: str = "basic") -> np.ndarray:
    cand = np.stack(candidates, axis=-1).astype(np.float32)
    vote = cand.mean(axis=-1, keepdims=True)
    union = cand.max(axis=-1, keepdims=True)
    inter = cand.min(axis=-1, keepdims=True)
    disagreement = union - inter
    anchor_f = anchor.astype(np.float32)[..., None]
    missing = ((union[..., 0] > 0.5) & ~anchor).astype(np.float32)[..., None]
    excess = (anchor & (vote[..., 0] <= 0.34)).astype(np.float32)[..., None]
    dist_in = ndimage.distance_transform_edt(anchor)
    dist_out = ndimage.distance_transform_edt(~anchor)
    signed = dist_in - dist_out
    denom = max(1.0, float(np.percentile(np.abs(signed), 95)))
    signed = np.clip(signed / denom, -1.0, 1.0)[..., None].astype(np.float32)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0:
        grad = grad / float(grad.max())
    features = [
        image[..., None].astype(np.float32),
        anchor_f,
        cand,
        vote,
        union,
        inter,
        disagreement,
        missing,
        excess,
        signed,
        grad[..., None].astype(np.float32),
    ]
    if mode == "local_stats":
        image_f = image.astype(np.float32)
        anchor_f2 = anchor.astype(np.float32)
        vote2 = vote[..., 0].astype(np.float32)
        disagreement2 = disagreement[..., 0].astype(np.float32)
        for size in (3, 7, 15):
            img_mean = local_mean(image_f, size)
            img_sq_mean = local_mean(image_f * image_f, size)
            img_std = np.sqrt(np.maximum(img_sq_mean - img_mean * img_mean, 0.0))
            features.extend(
                [
                    img_mean[..., None].astype(np.float32),
                    img_std[..., None].astype(np.float32),
                    local_mean(anchor_f2, size)[..., None].astype(np.float32),
                    local_mean(vote2, size)[..., None].astype(np.float32),
                    local_mean(disagreement2, size)[..., None].astype(np.float32),
                ]
            )
    return np.concatenate(features, axis=-1)


def edit_zone(anchor: np.ndarray, candidates: list[np.ndarray], radius: int, mode: str = "candidate_boundary") -> np.ndarray:
    cand = np.stack(candidates, axis=0)
    union = cand.max(axis=0).astype(bool)
    inter = cand.min(axis=0).astype(bool)
    if mode == "candidate_disagreement":
        zone = union ^ inter
    elif mode == "anchor_candidate_disagreement":
        zone = np.any(cand.astype(bool) != anchor[None, ...], axis=0)
    else:
        zone = union ^ inter
    if radius > 0 and mode == "candidate_boundary":
        k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        boundary = ndimage.binary_dilation(anchor, structure=k) ^ ndimage.binary_erosion(anchor, structure=k)
        zone |= boundary
    elif radius > 0 and mode in {"candidate_disagreement", "anchor_candidate_disagreement"}:
        k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        zone = ndimage.binary_dilation(zone, structure=k)
    return zone


def load_item(roots: list[Path], raw_root: Path, dataset: str, split: str, name: str, anchor_exp: str, candidate_exps: list[str], source_dataset: str | None = None):
    gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
    image = read_gray(image_path(raw_root, dataset, split, name))
    anchor = resize_like(read_mask(find_mask(roots, anchor_exp, dataset, split, name, source_dataset)), gt.shape)
    candidates = [resize_like(read_mask(find_mask(roots, exp, dataset, split, name, source_dataset)), gt.shape) for exp in candidate_exps]
    return image, gt, anchor, candidates


def candidate_exps_for(args: argparse.Namespace, dataset: str, split: str) -> list[str]:
    if dataset == args.apply_dataset and split == args.apply_split and args.apply_candidate_exps:
        if len(args.apply_candidate_exps) != len(args.candidate_exps):
            raise ValueError("--apply-candidate-exps must have the same length as --candidate-exps")
        return args.apply_candidate_exps
    return args.candidate_exps


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def sample_training(args: argparse.Namespace, roots: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    xs, ys = [], []
    for name in tqdm(names(Path(args.train_raw_root), args.train_dataset, args.train_split), desc="sample/train"):
        image, gt, anchor, candidates = load_item(roots, Path(args.train_raw_root), args.train_dataset, args.train_split, name, args.anchor_exp, args.candidate_exps)
        feat = feature_stack(image, anchor, candidates, args.feature_mode)
        zone = edit_zone(anchor, candidates, radius=args.train_zone_radius, mode=args.zone_mode)
        idx_zone = np.flatnonzero(zone.ravel())
        idx_all = np.arange(gt.size)
        n_zone = min(len(idx_zone), int(args.samples_per_image * 0.75))
        pick = []
        if n_zone > 0:
            pick.append(rng.choice(idx_zone, size=n_zone, replace=len(idx_zone) < n_zone))
        n_rest = args.samples_per_image - sum(len(p) for p in pick)
        if n_rest > 0:
            pick.append(rng.choice(idx_all, size=n_rest, replace=gt.size < n_rest))
        idx = np.concatenate(pick)
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
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses))}), flush=True)
    return model


def load_model_checkpoint(path: Path, args: argparse.Namespace) -> MLP:
    state = torch.load(path, map_location=args.device)
    model = MLP(int(state["in_dim"]), int(state.get("args", {}).get("hidden", args.hidden))).to(args.device)
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


def eval_config(model: MLP, args: argparse.Namespace, roots: list[Path], dataset: str, raw_root: Path, split: str, source_dataset: str | None, threshold: float, margin: float, radius: int, write_dir: Path | None = None):
    records = []
    anchor_exp = args.anchor_exp
    if dataset == args.apply_dataset and split == args.apply_split and args.apply_anchor_exp:
        anchor_exp = args.apply_anchor_exp
    for name in tqdm(names(raw_root, dataset, split), desc=f"eval/{dataset}/{split}"):
        candidate_exps = candidate_exps_for(args, dataset, split)
        image, gt, anchor, candidates = load_item(roots, raw_root, dataset, split, name, anchor_exp, candidate_exps, source_dataset)
        feat = feature_stack(image, anchor, candidates, args.feature_mode)
        prob = predict_prob(model, feat, args.device)
        zone = edit_zone(anchor, candidates, radius=radius, mode=args.zone_mode)
        pred = anchor.copy()
        pred[zone & (prob >= threshold + margin)] = True
        pred[zone & (prob <= threshold - margin)] = False
        if write_dir is not None:
            write_mask(write_dir / name, pred)
        item = compute_metrics(pred, gt, args.boundary_kernel)
        item["image"] = name
        records.append(item)
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}, records


def collect_eval_cache(model: MLP, args: argparse.Namespace, roots: list[Path], dataset: str, raw_root: Path, split: str, source_dataset: str | None):
    cache = []
    anchor_exp = args.anchor_exp
    if dataset == args.apply_dataset and split == args.apply_split and args.apply_anchor_exp:
        anchor_exp = args.apply_anchor_exp
    candidate_exps = candidate_exps_for(args, dataset, split)
    for name in tqdm(names(raw_root, dataset, split), desc=f"cache/{dataset}/{split}"):
        image, gt, anchor, candidates = load_item(roots, raw_root, dataset, split, name, anchor_exp, candidate_exps, source_dataset)
        feat = feature_stack(image, anchor, candidates, args.feature_mode)
        prob = predict_prob(model, feat, args.device)
        cache.append({"name": name, "gt": gt, "anchor": anchor, "candidates": candidates, "prob": prob})
    return cache


def eval_cached(
    cache: list[dict[str, object]],
    threshold: float,
    margin: float,
    radius: int,
    boundary_kernel: int,
    zone_mode: str,
    write_dir: Path | None = None,
):
    records = []
    for item in tqdm(cache, desc=f"eval/cache/r{radius}/t{threshold:.2f}/m{margin:.2f}", leave=False):
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        candidates = item["candidates"]  # type: ignore[assignment]
        prob = item["prob"]  # type: ignore[assignment]
        zones = item.setdefault("zones", {})  # type: ignore[assignment]
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
    if args.resume_checkpoint:
        model = load_model_checkpoint(Path(args.resume_checkpoint), args)
    else:
        x, y = sample_training(args, roots)
        model = train_model(args, x, y)
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "in_dim": x.shape[1], "args": vars(args)}, args.checkpoint)

    best = None
    tune_cache = collect_eval_cache(model, args, roots, args.train_dataset, Path(args.train_raw_root), args.tune_split, None)
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
    )
    mean, records = eval_cached(
        apply_cache,
        best["threshold"],
        best["margin"],
        best["radius"],
        args.boundary_kernel,
        args.zone_mode,
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
    print(json.dumps({"mean": mean, "best": best}, indent=2))


if __name__ == "__main__":
    main()
