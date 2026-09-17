#!/usr/bin/env python3
"""Train a timm encoder segmenter with instance-separation supervision."""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_instance_separation_segmenter import (
    add_structure_metrics,
    apply_sep_suppression,
    find_image_path,
    instance_targets,
    mean_metrics,
    read_gray,
    read_instance_mask,
    resize_float,
    resize_instance,
    train_loss,
    write_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train timm encoder with instance-separation heads.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--control-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--control-raw-root", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--encoder", default="convnext_tiny.dinov3_lvd1689m")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--min-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.10)
    parser.add_argument("--sep-loss-weight", type=float, default=0.45)
    parser.add_argument("--hover-loss-weight", type=float, default=0.20)
    parser.add_argument("--bridge-gap-loss-weight", type=float, default=0.0)
    parser.add_argument("--background-loss-weight", type=float, default=0.0)
    parser.add_argument("--cldice-loss-weight", type=float, default=0.0)
    parser.add_argument("--skeleton-iters", type=int, default=8)
    parser.add_argument("--sep-band-kernel", type=int, default=11)
    parser.add_argument("--sep-suppress-weights", default="0.00,0.15,0.30,0.45,0.60")
    parser.add_argument("--thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--hard-case-manifest", default=None)
    parser.add_argument("--hard-case-weight", type=float, default=1.0)
    parser.add_argument("--hard-case-epoch-multiplier", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--output-exp", default="r117_dinov3_instance_sep")
    parser.add_argument("--checkpoint", default="outputs/timm_instance_sep/r117_dinov3_instance_sep/best.pt")
    parser.add_argument("--history-json", default="outputs/timm_instance_sep/r117_dinov3_instance_sep/history.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r117_dinov3_instance_sep_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r117_dinov3_instance_sep_original_test_metrics.json")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class InstanceSegDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dataset: str,
        split: str,
        img_size: int,
        limit: int = 0,
        augment: bool = False,
        sep_kernel: int = 11,
    ) -> None:
        self.raw_root = raw_root
        self.dataset = dataset
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.sep_kernel = sep_kernel
        self.labels = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
        if limit > 0:
            self.labels = self.labels[:limit]
        self.name_to_index = {path.name: idx for idx, path in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | tuple[int, int]]:
        label_path = self.labels[idx]
        image = read_gray(find_image_path(self.raw_root, self.dataset, self.split, label_path.stem))
        instance = read_instance_mask(label_path)
        original_shape = instance.shape
        image = resize_float(image, self.img_size, cv2.INTER_AREA)
        instance = resize_instance(instance, self.img_size)
        binary, core, sep, hover_x, hover_y = instance_targets(instance, self.sep_kernel)
        boundary = ((cv2.dilate(binary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) -
                     cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)) > 0).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(np.fliplr(image))
                binary = np.ascontiguousarray(np.fliplr(binary))
                core = np.ascontiguousarray(np.fliplr(core))
                sep = np.ascontiguousarray(np.fliplr(sep))
                hover_x = np.ascontiguousarray(-np.fliplr(hover_x))
                hover_y = np.ascontiguousarray(np.fliplr(hover_y))
                boundary = np.ascontiguousarray(np.fliplr(boundary))
            if random.random() < 0.25:
                image = np.ascontiguousarray(np.flipud(image))
                binary = np.ascontiguousarray(np.flipud(binary))
                core = np.ascontiguousarray(np.flipud(core))
                sep = np.ascontiguousarray(np.flipud(sep))
                hover_x = np.ascontiguousarray(np.flipud(hover_x))
                hover_y = np.ascontiguousarray(-np.flipud(hover_y))
                boundary = np.ascontiguousarray(np.flipud(boundary))
            if random.random() < 0.35:
                image = np.clip(image ** random.uniform(0.80, 1.25), 0.0, 1.0)
            if random.random() < 0.35:
                image = np.clip(image + np.random.normal(0.0, 0.018, size=image.shape).astype(np.float32), 0.0, 1.0)

        target = np.stack([binary, core, sep, hover_x, hover_y, boundary], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "target": torch.from_numpy(target),
            "name": label_path.name,
            "shape": original_shape,
        }


def load_hard_case_train_names(path: str | None) -> set[str]:
    if not path:
        return set()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    names = set()
    for row in payload.get("hard_train", []):
        image = str(row.get("image", ""))
        if image:
            names.add(image)
    return names


def bridge_precision_loss(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    bridge_gap_weight: float,
    background_weight: float,
) -> torch.Tensor:
    mask_prob = torch.sigmoid(outputs["mask"])
    binary = target[:, 0:1]
    sep = target[:, 2:3]
    loss = mask_prob.new_tensor(0.0)
    if bridge_gap_weight > 0:
        sep_denom = sep.sum().clamp_min(1.0)
        loss = loss + bridge_gap_weight * ((mask_prob * sep).sum() / sep_denom)
    if background_weight > 0:
        background = 1.0 - binary
        bg_denom = background.sum().clamp_min(1.0)
        loss = loss + background_weight * ((mask_prob * background).sum() / bg_denom)
    return loss


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(x))


def soft_skeleton(x: torch.Tensor, iters: int) -> torch.Tensor:
    skel = F.relu(x - soft_open(x))
    for _ in range(max(0, iters - 1)):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def cldice_loss(logits: torch.Tensor, target: torch.Tensor, iters: int) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    pred_skel = soft_skeleton(pred, iters)
    target_skel = soft_skeleton(target, iters)
    eps = 1e-6
    tprec = (pred_skel * target).sum(dim=(1, 2, 3)) / (pred_skel.sum(dim=(1, 2, 3)) + eps)
    tsens = (target_skel * pred).sum(dim=(1, 2, 3)) / (target_skel.sum(dim=(1, 2, 3)) + eps)
    cldice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cldice).mean()


def make_train_sampler(
    dataset: InstanceSegDataset,
    hard_names: set[str],
    hard_weight: float,
    epoch_multiplier: float,
) -> WeightedRandomSampler | None:
    if not hard_names or hard_weight <= 1.0:
        return None
    weights = []
    matched = 0
    for label_path in dataset.labels:
        if label_path.name in hard_names:
            weights.append(float(hard_weight))
            matched += 1
        else:
            weights.append(1.0)
    if matched == 0:
        raise ValueError("Hard-case manifest did not match any training labels.")
    num_samples = max(len(dataset), int(round(len(dataset) * max(epoch_multiplier, 1.0))))
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimmInstanceSepUNet(nn.Module):
    def __init__(self, encoder: str, pretrained: bool, decoder_channels: int) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            encoder,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=1,
        )
        channels = list(self.encoder.feature_info.channels())
        self.lateral = nn.ModuleList([nn.Conv2d(ch, decoder_channels, 1) for ch in channels])
        self.smooth = nn.ModuleList([ConvBNAct(decoder_channels, decoder_channels) for _ in channels])
        self.trunk = ConvBNAct(decoder_channels, decoder_channels // 2)
        head_channels = decoder_channels // 2
        self.mask_head = nn.Conv2d(head_channels, 1, 1)
        self.core_head = nn.Conv2d(head_channels, 1, 1)
        self.sep_head = nn.Conv2d(head_channels, 1, 1)
        self.hover_head = nn.Conv2d(head_channels, 2, 1)
        self.boundary_head = nn.Conv2d(head_channels, 1, 1)

    @staticmethod
    def _as_nchw(feat: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if feat.ndim == 4 and feat.shape[1] == expected_channels:
            return feat
        if feat.ndim == 4 and feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        feats = [
            self._as_nchw(feat, expected_channels)
            for feat, expected_channels in zip(self.encoder(x), self.encoder.feature_info.channels(), strict=True)
        ]
        y = self.smooth[-1](self.lateral[-1](feats[-1]))
        for idx in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[idx].shape[-2:], mode="bilinear", align_corners=False)
            y = self.smooth[idx](y + self.lateral[idx](feats[idx]))
        y = F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)
        y = self.trunk(y)
        return {
            "mask": self.mask_head(y),
            "core": self.core_head(y),
            "sep": self.sep_head(y),
            "hover": self.hover_head(y),
            "boundary": self.boundary_head(y),
        }


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: str, threshold: float, suppress_weight: float) -> dict[str, float]:
    records: list[dict[str, float]] = []
    model.eval()
    for batch in loader:
        images = batch["image"].to(device)
        binary = batch["target"][:, 0:1].numpy()
        probs = apply_sep_suppression(model(images), suppress_weight).cpu().numpy()
        for i in range(probs.shape[0]):
            pred = probs[i, 0] >= threshold
            gt = binary[i, 0] > 0.5
            rec = compute_metrics(pred, gt, boundary_kernel=3)
            rec = add_structure_metrics(rec, pred, gt)
            rec["image"] = str(batch["name"][i])
            records.append(rec)
    return mean_metrics(records)


@torch.no_grad()
def infer_and_evaluate(
    model: nn.Module,
    raw_root: Path,
    dataset: str,
    split: str,
    img_size: int,
    threshold: float,
    suppress_weight: float,
    device: str,
    output_mask_dir: Path,
    metrics_json: Path,
    sep_kernel: int,
    limit: int = 0,
) -> dict[str, object]:
    ds = InstanceSegDataset(raw_root, dataset, split, img_size, limit=limit, augment=False, sep_kernel=sep_kernel)
    records: list[dict[str, float]] = []
    model.eval()
    for item in tqdm(ds, desc=f"infer/{dataset}/{split}"):
        image = item["image"].unsqueeze(0).to(device)
        prob = apply_sep_suppression(model(image), suppress_weight).cpu().numpy()[0, 0]
        shape = item["shape"]
        pred = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= threshold
        gt = read_instance_mask(raw_root / dataset / f"{split}_labels" / str(item["name"])) > 0
        write_mask(output_mask_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec = add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    summary: dict[str, object] = {
        "dataset": dataset,
        "split": split,
        "threshold": threshold,
        "sep_suppress_weight": suppress_weight,
        "num_evaluated": len(records),
        "mean": mean_metrics(records),
        "per_image": records,
    }
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train_ds = InstanceSegDataset(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.limit_train, True, args.sep_band_kernel)
    val_ds = InstanceSegDataset(Path(args.raw_root), args.dataset, args.val_split, args.img_size, args.limit_val, False, args.sep_band_kernel)
    hard_names = load_hard_case_train_names(args.hard_case_manifest)
    matched_hard_names = sorted(name for name in hard_names if name in train_ds.name_to_index)
    train_sampler = make_train_sampler(train_ds, hard_names, args.hard_case_weight, args.hard_case_epoch_multiplier)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(json.dumps({
        "hard_case_manifest": args.hard_case_manifest,
        "hard_case_weight": args.hard_case_weight,
        "hard_case_epoch_multiplier": args.hard_case_epoch_multiplier,
        "hard_case_manifest_names": len(hard_names),
        "hard_case_matched_train_names": len(matched_hard_names),
        "hard_case_sampler_enabled": train_sampler is not None,
    }), flush=True)

    model = TimmInstanceSepUNet(args.encoder, args.pretrained, args.decoder_channels).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    suppress_weights = [float(x) for x in args.sep_suppress_weights.split(",") if x.strip()]
    history: list[dict[str, float | int | str | bool]] = []
    best_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_suppress = 0.0
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"train/epoch{epoch}"):
            images = batch["image"].to(args.device)
            targets = batch["target"].to(args.device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                outputs = model(images)
                loss = train_loss(outputs, targets, args.boundary_loss_weight, args.sep_loss_weight, args.hover_loss_weight)
                loss = loss + bridge_precision_loss(
                    outputs,
                    targets,
                    args.bridge_gap_loss_weight,
                    args.background_loss_weight,
                )
                if args.cldice_loss_weight > 0:
                    loss = loss + args.cldice_loss_weight * cldice_loss(
                        outputs["mask"],
                        targets[:, 0:1],
                        args.skeleton_iters,
                    )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        val_grid = {}
        for thr in thresholds:
            for sup in suppress_weights:
                val_grid[(thr, sup)] = evaluate_loader(model, val_loader, args.device, thr, sup)
        (chosen_thr, chosen_sup), chosen_metrics = max(val_grid.items(), key=lambda kv: kv[1]["dice"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_dice": float(chosen_metrics["dice"]),
            "val_precision": float(chosen_metrics["precision"]),
            "val_recall": float(chosen_metrics["recall"]),
            "val_boundary_iou": float(chosen_metrics["boundary_iou"]),
            "val_false_bridge_flag": float(chosen_metrics["false_bridge_flag"]),
            "threshold": float(chosen_thr),
            "sep_suppress_weight": float(chosen_sup),
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
            "hard_case_matched_train_names": len(matched_hard_names),
            "hard_case_weight": float(args.hard_case_weight),
            "hard_case_epoch_multiplier": float(args.hard_case_epoch_multiplier),
            "bridge_gap_loss_weight": float(args.bridge_gap_loss_weight),
            "background_loss_weight": float(args.background_loss_weight),
            "cldice_loss_weight": float(args.cldice_loss_weight),
            "skeleton_iters": int(args.skeleton_iters),
        }
        history.append(row)
        Path(args.history_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.history_json).write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        if row["val_dice"] > best_score:
            best_score = float(row["val_dice"])
            best_epoch = epoch
            best_threshold = float(chosen_thr)
            best_suppress = float(chosen_sup)
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "threshold": best_threshold,
                    "sep_suppress_weight": best_suppress,
                    "epoch": epoch,
                },
                checkpoint,
            )
        if epoch >= args.min_epochs and epoch - best_epoch >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    best_threshold = float(state["threshold"])
    best_suppress = float(state.get("sep_suppress_weight", 0.0))
    eval_summary = infer_and_evaluate(
        model,
        Path(args.eval_raw_root),
        args.eval_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        best_suppress,
        args.device,
        Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks",
        Path(args.metrics_json),
        args.sep_band_kernel,
        args.limit_eval,
    )
    control_summary = infer_and_evaluate(
        model,
        Path(args.control_raw_root or args.raw_root),
        args.control_dataset,
        args.eval_split,
        args.img_size,
        best_threshold,
        best_suppress,
        args.device,
        Path(args.pred_root) / args.output_exp / args.control_dataset / args.eval_split / "masks",
        Path(args.control_metrics_json),
        args.sep_band_kernel,
        args.limit_eval,
    )
    print(json.dumps({
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "sep_suppress_weight": best_suppress,
        "clean_test_v2": eval_summary["mean"],
        "original_test": control_summary["mean"],
        "checkpoint": str(checkpoint),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
