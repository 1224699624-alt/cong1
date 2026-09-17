"""R272 local seam-prior + separation-loss pilot.

This experiment uses indexed instance labels only to build train-time seam
targets prepared by R271.  It trains a two-head U-Net: semantic foreground and
soft inter-instance seam.  At inference no GT target is read; the predicted
seam head is used only as a confidence-weighted suppression term.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_instance_separation_segmenter import (
    InstanceSeparationUNet,
    add_structure_metrics,
    dice_loss,
    mean_metrics,
    weighted_bce,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    p.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    p.add_argument("--target-root", type=Path, default=Path("outputs/targets/r271_instance_seam_full"))
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--min-epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seam-loss-weight", type=float, default=0.35)
    p.add_argument("--gap-loss-weight", type=float, default=0.20)
    p.add_argument("--support-loss-weight", type=float, default=0.20)
    p.add_argument("--seam-suppress-weight", type=float, default=0.35)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--output-root", type=Path, default=Path("outputs/experiments/r272_seam_prior_loss_local"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=20260720)
    return p.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def find_image(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        path = root / dataset / split / f"{stem}{ext}"
        if path.exists() and path.stat().st_size > 0:
            try:
                Image.open(path).verify()
                return path
            except Exception:
                continue
    raise FileNotFoundError(stem)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(path)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return image.astype(np.float32) / 255.0


def read_u8(path: Path, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if image.shape != shape:
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def contour(binary: np.ndarray) -> np.ndarray:
    b = (binary > 0.5).astype(np.uint8)
    return (cv2.dilate(b, np.ones((3, 3), np.uint8)) - cv2.erode(b, np.ones((3, 3), np.uint8))).astype(np.float32)


class SeamDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, limit: int, augment: bool):
        self.args, self.split, self.augment = args, split, augment
        self.labels = sorted((args.raw_root / args.dataset / f"{split}_labels").glob("*.png"))
        if limit > 0: self.labels = self.labels[:limit]
        self.seam_dir = args.target_root / split / "seam"
        self.semantic_dir = args.target_root / split / "semantic"
        if not self.seam_dir.exists() or not self.semantic_dir.exists():
            raise FileNotFoundError(f"R271 targets missing: {self.seam_dir}")

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]; stem = label.stem
        image_path = find_image(self.args.raw_root, self.args.dataset, self.split, stem)
        image = read_gray(image_path)
        h, w = image.shape
        image = cv2.resize(image, (self.args.img_size, self.args.img_size), interpolation=cv2.INTER_AREA)
        semantic = read_u8(self.semantic_dir / label.name, (h, w), cv2.INTER_NEAREST)
        seam = read_u8(self.seam_dir / label.name, (h, w), cv2.INTER_LINEAR)
        semantic = cv2.resize(semantic, (self.args.img_size, self.args.img_size), interpolation=cv2.INTER_NEAREST)
        seam = cv2.resize(seam, (self.args.img_size, self.args.img_size), interpolation=cv2.INTER_LINEAR)
        # Support is the bone immediately around a predicted/target seam. It
        # prevents gap loss from reducing both sides of a narrow bone.
        seam_band = cv2.dilate((seam > 0.25).astype(np.uint8), np.ones((9, 9), np.uint8))
        support = semantic * seam_band.astype(np.float32)
        boundary = contour(semantic)
        if self.augment and random.random() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image)); semantic = np.ascontiguousarray(np.fliplr(semantic))
            seam = np.ascontiguousarray(np.fliplr(seam)); support = np.ascontiguousarray(np.fliplr(support)); boundary = np.ascontiguousarray(np.fliplr(boundary))
        if self.augment and random.random() < 0.25:
            image = np.clip(image ** random.uniform(0.85, 1.20), 0.0, 1.0)
        target = np.stack([semantic, support, seam, np.zeros_like(seam), np.zeros_like(seam), boundary], axis=0).astype(np.float32)
        return {"image": torch.from_numpy(image[None]), "target": torch.from_numpy(target), "name": label.name}


def r272_loss(outputs, target, args):
    binary, support, seam, boundary = target[:, 0:1], target[:, 1:2], target[:, 2:3], target[:, 5:6]
    loss = weighted_bce(outputs["mask"], binary) + dice_loss(outputs["mask"], binary)
    loss = loss + 0.25 * (weighted_bce(outputs["core"], support) + dice_loss(outputs["core"], support))
    loss = loss + args.seam_loss_weight * (weighted_bce(outputs["sep"], seam) + dice_loss(outputs["sep"], seam))
    loss = loss + 0.10 * (weighted_bce(outputs["boundary"], boundary) + dice_loss(outputs["boundary"], boundary))
    mask_prob, seam_prob = torch.sigmoid(outputs["mask"]), torch.sigmoid(outputs["sep"])
    gap_mass = seam.sum().clamp_min(1.0)
    support_mass = support.sum().clamp_min(1.0)
    # Absolute upper-bound style gap constraint and two-side support retention.
    loss = loss + args.gap_loss_weight * (mask_prob * seam).sum() / gap_mass
    loss = loss + args.support_loss_weight * F.relu(0.80 - mask_prob).mul(support).sum() / support_mass
    return loss


def suppressed(outputs, weight):
    mask = torch.sigmoid(outputs["mask"]); seam = torch.sigmoid(outputs["sep"]); core = torch.sigmoid(outputs["core"])
    return torch.maximum(mask * (1.0 - weight * seam), 0.75 * core)


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval(); records = []
    for batch in loader:
        out = model(batch["image"].to(args.device)); prob = suppressed(out, args.seam_suppress_weight).cpu().numpy()
        gt = batch["target"][:, 0].numpy() > 0.5
        seam_target = batch["target"][:, 2].numpy() > 0.25
        seam_pred = torch.sigmoid(out["sep"]).cpu().numpy()[:, 0] > 0.5
        for i in range(len(gt)):
            pred = prob[i, 0] >= args.threshold
            rec = compute_metrics(pred, gt[i], boundary_kernel=3); rec = add_structure_metrics(rec, pred, gt[i])
            rec["seam_iou"] = float(((seam_pred[i] & seam_target[i]).sum()) / max(1, (seam_pred[i] | seam_target[i]).sum()))
            records.append(rec)
    return mean_metrics(records)


def main():
    args = parse_args(); seed_all(args.seed); args.output_root.mkdir(parents=True, exist_ok=True)
    if "clean-test" in " ".join(map(str, vars(args).values())).lower(): raise RuntimeError("clean-test forbidden")
    train = SeamDataset(args, "train", args.limit_train, True); val = SeamDataset(args, "val", args.limit_val, False)
    tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    vl = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = InstanceSeparationUNet(args.base_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=str(args.device).startswith("cuda"))
    best = -1.0e9; best_epoch = 0; bad = 0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch in tqdm(tl, desc=f"R272 train {epoch}"):
            images, target = batch["image"].to(args.device), batch["target"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=str(args.device).startswith("cuda")):
                loss = r272_loss(model(images), target, args)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, vl, args)
        score = (metrics.get("dice", 0.0) + 0.50 * metrics.get("boundary_iou", 0.0)
                 - 0.01 * metrics.get("component_count_error", 0.0))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}; history.append(row)
        print(json.dumps(row), flush=True)
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "score": score}, args.output_root / "best.pt")
        else: bad += 1
        if epoch >= args.min_epochs and bad >= args.patience: break
    result = {"run_id": "R272_seam_prior_loss_local", "device": str(args.device), "train_images": len(train), "val_images": len(val), "best_epoch": best_epoch, "best_score": best, "history": history, "clean_test_used": False}
    (args.output_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("run_id", "device", "train_images", "val_images", "best_epoch", "best_score")}, indent=2))


if __name__ == "__main__": main()
