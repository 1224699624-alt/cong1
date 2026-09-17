#!/usr/bin/env python3
"""R289: frozen-nnU-Net pair-specific gap relation-head pilot on TSRS.

This gate trains no segmentation weights and never uses clean-test-v2. Instance labels
define local pair-interface targets only during training/evaluation; inference uses X-ray
features alone.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image(folder: Path, stem: str) -> Path:
    # This raw split contains zero-byte extension placeholders beside the real image.
    # Select the first non-empty file using the project's canonical extension order.
    for extension in (".jpg", ".jpeg", ".png", ".bmp"):
        path = folder / f"{stem}{extension}"
        if path.exists() and path.stat().st_size > 0:
            return path
    raise RuntimeError({"stem": stem, "nonempty_image_not_found": str(folder)})


def pair_adjacency(
    label: np.ndarray,
    radius: int,
) -> list[tuple[int, int, int]]:
    labels = [int(value) for value in np.unique(label) if value > 0]
    masks = {value: (label == value).astype(np.uint8) for value in labels}
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dilated = {value: cv2.dilate(mask, kernel) > 0 for value, mask in masks.items()}
    background = label == 0
    pairs: list[tuple[int, int, int]] = []
    for first_index, first in enumerate(labels):
        for second in labels[first_index + 1:]:
            gap = dilated[first] & dilated[second] & background
            pixels = int(gap.sum())
            if pixels:
                pairs.append((first, second, pixels))
    return pairs


def audit_training_pairs(
    root: Path,
    output: Path,
    pair_count: int,
    audit_size: int,
    radius: int,
) -> tuple[list[tuple[int, int]], dict[str, list[tuple[int, int]]]]:
    case_count: dict[tuple[int, int], int] = defaultdict(int)
    pixel_count: dict[tuple[int, int], int] = defaultdict(int)
    all_case_pairs: dict[str, list[tuple[int, int]]] = {}
    for path in sorted((root / "train_labels").glob("*.png")):
        raw = np.asarray(Image.open(path))
        label = cv2.resize(raw, (audit_size, audit_size), interpolation=cv2.INTER_NEAREST)
        records = pair_adjacency(label, radius)
        case_pairs: list[tuple[int, int]] = []
        for first, second, pixels in records:
            pair = (first, second)
            case_count[pair] += 1
            pixel_count[pair] += pixels
            case_pairs.append(pair)
        all_case_pairs[path.stem] = case_pairs
    ranked = sorted(case_count, key=lambda pair: (case_count[pair], pixel_count[pair]), reverse=True)
    pairs = ranked[:pair_count]
    selected = set(pairs)
    case_pairs = {
        case: [pair for pair in records if pair in selected]
        for case, records in all_case_pairs.items()
    }
    case_pairs = {case: records for case, records in case_pairs.items() if records}
    payload = {
        "source": "TSRS_RSNA-Epiphysis/train_labels only",
        "audit_size": audit_size,
        "audit_radius": radius,
        "pair_count": len(pairs),
        "pairs": [
            {"labels": pair, "case_count": case_count[pair], "gap_pixels": pixel_count[pair]}
            for pair in pairs
        ],
        "train_cases_with_selected_pairs": len(case_pairs),
    }
    (output / "pair_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return pairs, case_pairs


class PairCropDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        pairs: list[tuple[int, int]],
        crop_size: int,
        train_case_pairs: dict[str, list[tuple[int, int]]] | None = None,
        max_pairs_per_val_case: int = 0,
    ) -> None:
        self.root = root
        self.split = split
        self.pairs = pairs
        self.pair_to_index = {pair: index for index, pair in enumerate(pairs)}
        self.crop_size = crop_size
        self.training = split == "train"
        self.records: list[tuple[Path, tuple[int, int]]] = []
        label_folder = root / f"{split}_labels"
        if self.training:
            assert train_case_pairs is not None
            for path in sorted(label_folder.glob("*.png")):
                for pair in train_case_pairs.get(path.stem, []):
                    self.records.append((path, pair))
        else:
            for path in sorted(label_folder.glob("*.png")):
                raw = np.asarray(Image.open(path))
                small = cv2.resize(raw, (192, 192), interpolation=cv2.INTER_NEAREST)
                adjacent = {(first, second) for first, second, _ in pair_adjacency(small, 3)}
                selected = [pair for pair in pairs if pair in adjacent]
                if max_pairs_per_val_case > 0:
                    selected = selected[:max_pairs_per_val_case]
                self.records.extend((path, pair) for pair in selected)
        if not self.records:
            raise RuntimeError(f"No pair crops for {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        label_path, focus_pair = self.records[index]
        label = np.asarray(Image.open(label_path), dtype=np.uint8)
        image = np.asarray(
            Image.open(find_image(self.root / self.split, label_path.stem)).convert("L"),
            dtype=np.float32,
        )
        points = []
        for value in focus_pair:
            yy, xx = np.where(label == value)
            if not len(xx):
                raise RuntimeError((label_path, focus_pair))
            points.append((float(xx.mean()), float(yy.mean())))
        center_x = int(round((points[0][0] + points[1][0]) / 2))
        center_y = int(round((points[0][1] + points[1][1]) / 2))
        if self.training:
            center_x += random.randint(-48, 48)
            center_y += random.randint(-48, 48)
        half = self.crop_size // 2
        pad_left = max(0, half - center_x)
        pad_top = max(0, half - center_y)
        pad_right = max(0, center_x + half - image.shape[1])
        pad_bottom = max(0, center_y + half - image.shape[0])
        if pad_left or pad_top or pad_right or pad_bottom:
            image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            label = np.pad(label, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
            center_x += pad_left
            center_y += pad_top
        x0, y0 = center_x - half, center_y - half
        image = image[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        label = label[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
        if image.shape != (self.crop_size, self.crop_size):
            raise RuntimeError((label_path, image.shape))
        image = (image - image.mean()) / (image.std() + 1e-6)
        return {
            "image": torch.from_numpy(image[None].astype(np.float32)),
            "instance": torch.from_numpy(label.astype(np.int64)),
            "focus_pair": torch.tensor(self.pair_to_index[focus_pair], dtype=torch.long),
            "case": label_path.stem,
        }


class FrozenNnUNetPairHead(nn.Module):
    def __init__(self, checkpoint: Path, pair_count: int, device: torch.device) -> None:
        super().__init__()
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        self.backbone = PlainConvUNet(
            input_channels=1,
            n_stages=9,
            features_per_stage=[32, 64, 128, 256, 512, 512, 512, 512, 512],
            conv_op=nn.Conv2d,
            kernel_sizes=[[3, 3]] * 9,
            strides=[[1, 1]] + [[2, 2]] * 8,
            n_conv_per_stage=[2] * 9,
            num_classes=2,
            n_conv_per_stage_decoder=[2] * 8,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.backbone.load_state_dict(payload["network_weights"], strict=True)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.features: dict[str, torch.Tensor] = {}
        self.backbone.decoder.stages[-2].register_forward_hook(self._capture("half"))
        self.backbone.decoder.stages[-1].register_forward_hook(self._capture("full"))
        self.half_projection = nn.Conv2d(64, 32, 1, bias=False)
        self.relation_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(32, pair_count * 3, 1),
        )
        self.pair_count = pair_count

    def _capture(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.features[name] = output
        return hook

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.features.clear()
        with torch.no_grad():
            _ = self.backbone(image)
        half = F.interpolate(self.half_projection(self.features["half"].detach()),
                             size=self.features["full"].shape[-2:], mode="bilinear",
                             align_corners=False)
        fused = torch.cat([half, self.features["full"].detach()], dim=1)
        return self.relation_head(fused).reshape(
            image.shape[0], self.pair_count, 3, *image.shape[-2:]
        )


def relation_targets(
    instance: torch.Tensor,
    pairs: list[tuple[int, int]],
    radius: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels, rois, gaps = [], [], []
    for first, second in pairs:
        first_mask = instance == first
        second_mask = instance == second
        first_d = F.max_pool2d(first_mask[:, None].float(), 2 * radius + 1, 1, radius)[:, 0] > 0.5
        second_d = F.max_pool2d(second_mask[:, None].float(), 2 * radius + 1, 1, radius)[:, 0] > 0.5
        roi = first_d & second_d
        gap = roi & (instance == 0)
        overlap = first_mask & second_mask
        target = torch.full_like(instance, 2, dtype=torch.long)
        target[gap] = 0
        target[overlap] = 1
        labels.append(target)
        rois.append(roi)
        gaps.append(gap)
    return torch.stack(labels, 1), torch.stack(rois, 1), torch.stack(gaps, 1)


def relation_loss(logits: torch.Tensor, labels: torch.Tensor, roi: torch.Tensor) -> torch.Tensor:
    batch, pairs, _, height, width = logits.shape
    loss = F.cross_entropy(
        logits.float().reshape(batch * pairs, 3, height, width),
        labels.reshape(batch * pairs, height, width),
        weight=torch.tensor([1.5, 1.0, 0.25], device=logits.device),
        reduction="none",
    ).reshape(batch, pairs, height, width)
    return (loss * roi.float()).sum() / roi.sum().clamp_min(1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pairs: list[tuple[int, int]],
    radius: int,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0.0
    soft_intersection = soft_sum = target_sum = 0.0
    cases: set[str] = set()
    pair_crops = 0
    for batch in loader:
        image = batch["image"].to(device)
        instance = batch["instance"].to(device)
        logits = model(image)
        _, roi, gap = relation_targets(instance, pairs, radius)
        probability = torch.softmax(logits.float(), dim=2)[:, :, 0]
        prediction = (probability >= 0.5) & roi
        target = gap & roi
        tp += float((prediction & target).sum())
        fp += float((prediction & ~target & roi).sum())
        fn += float((~prediction & target).sum())
        soft_intersection += float((probability * target.float()).sum())
        soft_sum += float((probability * roi.float()).sum())
        target_sum += float(target.sum())
        cases.update(map(str, batch["case"]))
        pair_crops += int(image.shape[0])
    return {
        "cases": float(len(cases)),
        "pair_crops": float(pair_crops),
        "gap_precision": tp / max(tp + fp, 1.0),
        "gap_recall": tp / max(tp + fn, 1.0),
        "gap_dice": 2 * tp / max(2 * tp + fp + fn, 1.0),
        "gap_soft_dice": (2 * soft_intersection + 1) / max(soft_sum + target_sum + 1, 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/tsrs/r289_pair_gap_head_dev"))
    parser.add_argument("--pair-count", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--gap-radius", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=289)
    parser.add_argument("--experiment-name", default="R289")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    joined = " ".join(map(str, (args.dataset_root, args.output))).lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R289 is restricted to TSRS_RSNA-Epiphysis train/original-val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs, train_case_pairs = audit_training_pairs(
        args.dataset_root, args.output, args.pair_count, audit_size=192, radius=3
    )
    train_dataset = PairCropDataset(args.dataset_root, "train", pairs, args.crop_size,
                                    train_case_pairs=train_case_pairs)
    val_dataset = PairCropDataset(args.dataset_root, "val", pairs, args.crop_size)
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=args.samples_per_epoch
    )
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=0,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    model = FrozenNnUNetPairHead(args.checkpoint, len(pairs), device).to(device)
    manifest = {
        "experiment": args.experiment_name, "dataset": str(args.dataset_root),
        "train_cases": 875, "original_val_cases": 96,
        "train_pair_crops": len(train_dataset), "val_pair_crops": len(val_dataset),
        "sampled_train_pair_crops_per_epoch": args.samples_per_epoch,
        "pair_count": len(pairs), "pairs": pairs, "gap_radius": args.gap_radius,
        "backbone": "frozen mature R275 nnU-Net", "checkpoint": str(args.checkpoint),
        "relation_classes": ["gap", "overlap", "uncertain_support"],
        "instance_gt_used_at_inference": False, "mask_edited": False,
        "clean_test_v2_used": False, "articular_surface_used": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        batch = next(iter(train_loader))
        image = batch["image"].to(device)
        instance = batch["instance"].to(device)
        logits = model(image)
        labels, roi, _ = relation_targets(instance, pairs, args.gap_radius)
        loss = relation_loss(logits, labels, roi)
        loss.backward()
        print(json.dumps({"relations": list(logits.shape), "loss": float(loss.detach()),
                          "finite": bool(torch.isfinite(loss)), "device": str(device),
                          "train_pair_crops": len(train_dataset),
                          "val_pair_crops": len(val_dataset)}), flush=True)
        return
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_key: tuple[float, float] | None = None
    checkpoint = args.output / "relation_head_best.pth"
    history = args.output / "history.jsonl"
    for epoch in range(args.epochs):
        model.train()
        started = time.time()
        total_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            instance = batch["instance"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(image)
                labels, roi, _ = relation_targets(instance, pairs, args.gap_radius)
                loss = relation_loss(logits, labels, roi)
            if not torch.isfinite(loss):
                raise RuntimeError({"epoch": epoch, "loss": float(loss.detach())})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            total_loss += float(loss.detach())
        scheduler.step()
        metrics = evaluate(model, val_loader, device, pairs, args.gap_radius)
        row = {"epoch": epoch, "seconds": time.time() - started,
               "train_relation_loss": total_loss / len(train_loader), **metrics}
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        key = (metrics["gap_dice"], metrics["gap_recall"])
        if best_key is None or key > best_key:
            best_key = key
            torch.save({"relation_head": model.relation_head.state_dict(),
                        "half_projection": model.half_projection.state_dict(),
                        "pairs": pairs, "epoch": epoch, "metrics": metrics}, checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    result = {
        "status": "complete_relation_gate",
        "selected_epoch": payload["epoch"], "validation": payload["metrics"],
        "gate": {
            "gap_dice_ge_0_30": payload["metrics"]["gap_dice"] >= 0.30,
            "gap_recall_ge_0_50": payload["metrics"]["gap_recall"] >= 0.50,
            "gap_precision_ge_0_20": payload["metrics"]["gap_precision"] >= 0.20,
        },
        "decision_if_pass": "allow_r290_frozen_segmentation_residual",
        "decision_if_fail": "no_go_pair_relation_head_not_localizing_gap",
        "official_r201_not_run_because_masks_unchanged": True,
    }
    result["gate_pass"] = all(result["gate"].values())
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
