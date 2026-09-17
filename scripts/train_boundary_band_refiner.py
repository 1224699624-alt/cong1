#!/usr/bin/env python3
"""Train a boundary-band refiner that edits a strong anchor mask only near its contour."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anchor boundary-band refiner.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--apply-anchor-exp", default=None)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-band-radius", type=int, default=12)
    parser.add_argument("--band-loss-weight", type=float, default=4.0)
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--edit-radii", default="2,4,6,8,10,12")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--output-exp", default="r080_boundary_band_refiner")
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r080_boundary_band_refiner/best.pt")
    parser.add_argument("--history-json", default="outputs/context_segmenter/r080_boundary_band_refiner/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r080_boundary_band_refiner_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r080_boundary_band_refiner_original_test_metrics.json")
    parser.add_argument("--skip-control", action="store_true", help="Skip original-test control when the chosen anchor has no control masks.")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"image not found/readable: {path}")
    return image.astype(np.float32) / 255.0


def find_image(raw_root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = raw_root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found: {raw_root}/{dataset}/{split}/{stem}")


def find_anchor(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None = None) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"anchor not found: exp={exp} dataset={dataset} split={split} name={name}")


def resize_float(arr: np.ndarray, size: int, interpolation: int) -> np.ndarray:
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=interpolation)


def resize_bool(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr.astype(bool)
    return cv2.resize(arr.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def signed_distance(anchor: np.ndarray) -> np.ndarray:
    dist_in = ndimage.distance_transform_edt(anchor)
    dist_out = ndimage.distance_transform_edt(~anchor)
    signed = dist_in - dist_out
    denom = max(1.0, float(np.percentile(np.abs(signed), 95)))
    return np.clip(signed / denom, -1.0, 1.0).astype(np.float32)


class BandDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dataset: str,
        split: str,
        roots: list[Path],
        anchor_exp: str,
        img_size: int,
        limit: int = 0,
        augment: bool = False,
        source_dataset: str | None = None,
    ) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.roots = roots
        self.anchor_exp = anchor_exp
        self.img_size = img_size
        self.augment = augment
        self.source_dataset = source_dataset
        self.labels = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
        if limit > 0:
            self.labels = self.labels[:limit]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        label_path = self.labels[idx]
        image = read_gray(find_image(self.raw_root, self.dataset, self.split, label_path.stem))
        gt = read_mask(label_path)
        anchor = resize_bool(read_mask(find_anchor(self.roots, self.anchor_exp, self.dataset, self.split, label_path.name, self.source_dataset)), gt.shape)
        original_shape = gt.shape

        image = resize_float(image, self.img_size, cv2.INTER_AREA)
        gt_f = resize_float(gt.astype(np.float32), self.img_size, cv2.INTER_NEAREST)
        anchor_f = resize_float(anchor.astype(np.float32), self.img_size, cv2.INTER_NEAREST)
        signed = resize_float(signed_distance(anchor), self.img_size, cv2.INTER_LINEAR)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                gt_f = np.ascontiguousarray(np.fliplr(gt_f))
                anchor_f = np.ascontiguousarray(np.fliplr(anchor_f))
                signed = np.ascontiguousarray(np.fliplr(signed))
            if random.random() < 0.25:
                image = np.clip(image ** random.uniform(0.85, 1.20), 0.0, 1.0)
            if random.random() < 0.25:
                image = np.clip(image + np.random.normal(0.0, 0.012, size=image.shape).astype(np.float32), 0.0, 1.0)

        x = np.stack([image, anchor_f, signed], axis=0).astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(gt_f[None].astype(np.float32)),
            "anchor": torch.from_numpy(anchor_f[None].astype(np.float32)),
            "name": label_path.name,
            "shape": original_shape,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BandUNet(nn.Module):
    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.mid(e4)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def dice_loss(prob: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    inter = (prob * target * weight).sum(dim=(1, 2, 3))
    den = (prob * weight).sum(dim=(1, 2, 3)) + (target * weight).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (den + 1.0)).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor, radius: int, band_weight: float) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    with torch.no_grad():
        pooled_max = F.max_pool2d(anchor, kernel_size=2 * radius + 1, stride=1, padding=radius)
        pooled_min = -F.max_pool2d(-anchor, kernel_size=2 * radius + 1, stride=1, padding=radius)
        band = (pooled_max - pooled_min).clamp(0.0, 1.0)
        weight = 1.0 + band_weight * band
    bce = F.binary_cross_entropy_with_logits(logits, target, weight=weight)
    return bce + dice_loss(prob, target, weight)


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in records[0] if key != "image"] if records else []
    return {key: float(np.mean([float(record[key]) for record in records])) for key in keys}


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0)


def add_structure_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


@torch.no_grad()
def predict_prob(model: nn.Module, x: torch.Tensor, device: str) -> torch.Tensor:
    model.eval()
    return torch.sigmoid(model(x.to(device))).cpu()


def apply_band(anchor: np.ndarray, prob: np.ndarray, threshold: float, radius: int) -> np.ndarray:
    band = boundary_band(anchor, radius)
    pred = anchor.copy()
    pred[band] = prob[band] >= threshold
    return pred


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, threshold: float, radius: int, boundary_kernel: int) -> dict[str, float]:
    records: list[dict[str, float]] = []
    for batch in loader:
        probs = predict_prob(model, batch["x"], device).numpy()
        masks = batch["mask"].numpy()
        anchors = batch["anchor"].numpy()
        for i in range(probs.shape[0]):
            pred = apply_band(anchors[i, 0] > 0.5, probs[i, 0], threshold, radius)
            gt = masks[i, 0] > 0.5
            rec = compute_metrics(pred, gt, boundary_kernel)
            add_structure_metrics(rec, pred, gt)
            rec["image"] = str(batch["name"][i])
            records.append(rec)
    return mean_metrics(records)


@torch.no_grad()
def infer_and_evaluate(
    model: nn.Module,
    ds: BandDataset,
    device: str,
    threshold: float,
    radius: int,
    boundary_kernel: int,
    output_mask_dir: Path,
    metrics_json: Path,
) -> dict[str, object]:
    records: list[dict[str, float]] = []
    for item in tqdm(ds, desc=f"infer/{ds.dataset}/{ds.split}"):
        prob = predict_prob(model, item["x"].unsqueeze(0), device).numpy()[0, 0]
        shape = item["shape"]
        prob_full = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        gt = read_mask(ds.raw_root / ds.dataset / f"{ds.split}_labels" / str(item["name"]))
        anchor = resize_bool(read_mask(find_anchor(ds.roots, ds.anchor_exp, ds.dataset, ds.split, str(item["name"]), ds.source_dataset)), gt.shape)
        pred = apply_band(anchor, prob_full, threshold, radius)
        write_mask(output_mask_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel)
        add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    summary: dict[str, object] = {
        "dataset": ds.dataset,
        "split": ds.split,
        "threshold": threshold,
        "edit_radius": radius,
        "num_evaluated": len(records),
        "mean": mean_metrics(records),
        "per_image": records,
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    roots = [Path(root) for root in args.ablations_roots]
    train_ds = BandDataset(Path(args.train_raw_root), args.train_dataset, args.train_split, roots, args.anchor_exp, args.img_size, args.limit_train, True)
    val_ds = BandDataset(Path(args.train_raw_root), args.train_dataset, args.val_split, roots, args.anchor_exp, args.img_size, args.limit_val, False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = BandUNet(args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = parse_floats(args.thresholds)
    radii = parse_ints(args.edit_radii)
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_radius = radii[0]
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            x = batch["x"].to(args.device)
            target = batch["mask"].to(args.device)
            anchor = batch["anchor"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(x)
                loss = loss_fn(logits, target, anchor, args.train_band_radius, args.band_loss_weight)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        best_val = None
        for radius in radii:
            for threshold in thresholds:
                metrics = evaluate_loader(model, val_loader, args.device, threshold, radius, args.boundary_kernel)
                item = {"threshold": threshold, "radius": radius, "metrics": metrics}
                if best_val is None or metrics["dice"] > best_val["metrics"]["dice"]:
                    best_val = item
        assert best_val is not None
        metrics = best_val["metrics"]
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(metrics["dice"]),
            "val_precision": float(metrics["precision"]),
            "val_recall": float(metrics["recall"]),
            "val_boundary_iou": float(metrics["boundary_iou"]),
            "val_false_bridge_flag": float(metrics["false_bridge_flag"]),
            "threshold": float(best_val["threshold"]),
            "edit_radius": int(best_val["radius"]),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_threshold = float(row["threshold"])
            best_radius = int(row["edit_radius"])
            torch.save({"model": model.state_dict(), "args": vars(args), "threshold": best_threshold, "edit_radius": best_radius, "epoch": epoch}, checkpoint)
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best_threshold = float(state["threshold"])
    best_radius = int(state["edit_radius"])

    apply_anchor = args.apply_anchor_exp or args.anchor_exp
    eval_ds = BandDataset(Path(args.apply_raw_root), args.apply_dataset, args.apply_split, roots, apply_anchor, args.img_size, 0, False, args.source_apply_dataset)
    eval_mask_dir = Path(args.pred_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    eval_summary = infer_and_evaluate(model, eval_ds, args.device, best_threshold, best_radius, args.boundary_kernel, eval_mask_dir, Path(args.metrics_json))
    control_summary = None
    if not args.skip_control:
        control_ds = BandDataset(Path(args.train_raw_root), args.control_dataset, args.apply_split, roots, args.anchor_exp, args.img_size, 0, False)
        control_mask_dir = Path(args.pred_root) / args.output_exp / args.control_dataset / args.apply_split / "masks"
        control_summary = infer_and_evaluate(model, control_ds, args.device, best_threshold, best_radius, args.boundary_kernel, control_mask_dir, Path(args.control_metrics_json))
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "edit_radius": best_radius,
        "clean_test_v2": eval_summary["mean"],
        "original_test": None if control_summary is None else control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
