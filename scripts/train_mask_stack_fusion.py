#!/usr/bin/env python3
"""Train a lightweight CNN to fuse multiple candidate masks."""

from __future__ import annotations

import argparse
import json
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mask-stack fusion CNN.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument(
        "--apply-candidates",
        nargs="+",
        default=None,
        help="Candidate masks used for apply-dataset inference. Defaults to --candidates and must have the same length.",
    )
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def find_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(image_dir.glob(ext))
    return sorted(files)


def read_gray(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return arr


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return (arr > 0).astype(np.float32)


def resize(arr: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=interpolation)


def find_mask_path(
    roots: list[Path],
    exp: str,
    dataset: str,
    split: str,
    name: str,
    source_dataset: str | None = None,
) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Mask not found: exp={exp}, dataset={dataset}, split={split}, name={name}")


def make_features(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(masks, axis=0).astype(np.float32)
    vote = stack.mean(axis=0, keepdims=True)
    union = stack.max(axis=0, keepdims=True)
    inter = stack.min(axis=0, keepdims=True)
    disagreement = (union - inter).astype(np.float32)
    grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    if float(grad.max()) > 0:
        grad = grad / float(grad.max())
    return np.concatenate([image[None], stack, vote, union, inter, disagreement, grad[None]], axis=0).astype(np.float32)


class StackDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        roots: list[Path],
        dataset: str,
        split: str,
        candidates: list[str],
        img_size: int,
        source_dataset: str | None = None,
        need_gt: bool = True,
    ) -> None:
        self.raw_root = raw_root
        self.roots = roots
        self.dataset = dataset
        self.split = split
        self.candidates = candidates
        self.img_size = img_size
        self.source_dataset = source_dataset
        self.need_gt = need_gt
        self.image_dir = raw_root / dataset / split
        self.label_dir = raw_root / dataset / f"{split}_labels"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.image_dir}")
        if need_gt and not self.label_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")
        self.image_files = find_image_files(self.image_dir)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        image_path = self.image_files[index]
        name = f"{image_path.stem}.png"
        image_orig = read_gray(image_path)
        h, w = image_orig.shape
        image = resize(image_orig, self.img_size, cv2.INTER_LINEAR)
        masks = []
        for exp in self.candidates:
            mask_path = find_mask_path(self.roots, exp, self.dataset, self.split, name, self.source_dataset)
            masks.append(resize(read_mask(mask_path), self.img_size, cv2.INTER_NEAREST))
        item: dict[str, torch.Tensor | str | tuple[int, int]] = {
            "image": torch.from_numpy(make_features(image, masks)),
            "name": name,
            "shape": (h, w),
        }
        if self.need_gt:
            gt = resize(read_mask(self.label_dir / name), self.img_size, cv2.INTER_NEAREST)
            item["target"] = torch.from_numpy(gt[None].astype(np.float32))
        return item


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int, base: int) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.b = ConvBlock(base * 4, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = ConvBlock(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = ConvBlock(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.b(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def batch_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= 0.5
    gt = target >= 0.5
    tp = (pred & gt).float().sum(dim=(1, 2, 3))
    fp = (pred & ~gt).float().sum(dim=(1, 2, 3))
    fn = (~pred & gt).float().sum(dim=(1, 2, 3))
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return {"dice": float(dice.mean().item())}


def run_epoch(model: nn.Module, loader: DataLoader, device: str, opt: torch.optim.Optimizer | None) -> dict[str, float]:
    model.train(opt is not None)
    loss_sum = 0.0
    dice_sum = 0.0
    n = 0
    with torch.set_grad_enabled(opt is not None):
        for batch in tqdm(loader, desc="train" if opt is not None else "val"):
            x = batch["image"].to(device=device, dtype=torch.float32)
            y = batch["target"].to(device=device, dtype=torch.float32)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            m = batch_metrics(logits.detach(), y)
            loss_sum += float(loss.item())
            dice_sum += m["dice"]
            n += 1
    return {"loss": loss_sum / max(1, n), "dice": dice_sum / max(1, n)}


def parse_thresholds(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str) -> list[tuple[str, np.ndarray, tuple[int, int]]]:
    model.eval()
    out = []
    for batch in tqdm(loader, desc="predict"):
        logits = model(batch["image"].to(device=device, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy()
        for idx, name in enumerate(batch["name"]):
            shape = (int(batch["shape"][0][idx]), int(batch["shape"][1][idx]))
            out.append((str(name), probs[idx, 0], shape))
    return out


def eval_probs(probs: list[tuple[str, np.ndarray, tuple[int, int]]], label_dir: Path, threshold: float) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for name, prob, shape in probs:
        gt = read_mask(label_dir / name).astype(bool)
        pred = cv2.resize((prob >= threshold).astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        metrics = compute_metrics(pred, gt, 3)
        metrics["image"] = name
        records.append(metrics)
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}, records


def main() -> None:
    args = parse_args()
    apply_candidates = args.apply_candidates or args.candidates
    if len(apply_candidates) != len(args.candidates):
        raise ValueError("--apply-candidates must have the same number of entries as --candidates.")
    roots = [Path(p) for p in args.ablations_roots]
    train_ds = StackDataset(Path(args.raw_root), roots, args.dataset, "train", args.candidates, args.img_size)
    val_ds = StackDataset(Path(args.raw_root), roots, args.dataset, "val", args.candidates, args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    in_ch = 1 + len(args.candidates) + 5
    model = TinyUNet(in_ch, args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dice = -1.0
    best_epoch = 0
    stale = 0
    history = []
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, args.device, opt)
        val_metrics = run_epoch(model, val_loader, args.device, None)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            stale = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "in_ch": in_ch}, ckpt_path)
        else:
            stale += 1
        if epoch >= args.min_epochs and stale >= args.patience:
            break

    checkpoint = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    val_probs = predict_probs(model, val_loader, args.device)
    thresholds = []
    for threshold in parse_thresholds(args.thresholds):
        mean, _ = eval_probs(val_probs, Path(args.raw_root) / args.dataset / "val_labels", threshold)
        thresholds.append({"threshold": threshold, "mean": mean})
    best_threshold = max(thresholds, key=lambda x: x["mean"]["dice"])["threshold"]

    apply_ds = StackDataset(
        Path(args.apply_raw_root),
        roots,
        args.apply_dataset,
        "test",
        apply_candidates,
        args.img_size,
        source_dataset=args.source_apply_dataset,
        need_gt=True,
    )
    apply_loader = DataLoader(apply_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    apply_probs = predict_probs(model, apply_loader, args.device)
    out_dir = Path(args.ablations_roots[0]) / args.output_exp / args.apply_dataset / "test" / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, prob, shape in apply_probs:
        pred = cv2.resize((prob >= best_threshold).astype(np.uint8) * 255, shape[::-1], interpolation=cv2.INTER_NEAREST)
        Image.fromarray(pred).save(out_dir / name)
    mean, records = eval_probs(apply_probs, Path(args.apply_raw_root) / args.apply_dataset / "test_labels", best_threshold)
    summary = {
        "dataset": args.apply_dataset,
        "split": "test",
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "thresholds": thresholds,
        "selected_threshold": best_threshold,
        "output_exp": args.output_exp,
        "train_candidates": args.candidates,
        "apply_candidates": apply_candidates,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"mean": mean, "selected_threshold": best_threshold, "best_epoch": best_epoch}, indent=2))


if __name__ == "__main__":
    main()
