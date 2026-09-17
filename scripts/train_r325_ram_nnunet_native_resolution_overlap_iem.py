#!/usr/bin/env python3
"""R325: RAM-W600 native-resolution nnU-Net vs explicit overlap IEM.

The experiment changes only spatial preprocessing relative to R324: 600x600 and
720x720 source images are kept at native resolution and padded (never resized)
to a multiple of the nnU-Net stride. Train/validation split, initialization,
loss, threshold, relation graph and checkpoint rule remain aligned with R324.
The test split is not loaded or evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
from train_r324_ram_nnunet_explicit_overlap_iem import (
    build_overlap_graph,
    seed_all,
    train_arm,
    validate,
)


class NativeWristDataset(Dataset):
    """Load original RAM pixels and pad only; no spatial resampling."""

    def __init__(self, root: Path, split: str, augment: bool, divisor: int = 32) -> None:
        self.root = root
        self.split = split
        self.augment = augment
        self.divisor = divisor
        self.mask_files = sorted((root / "BoneSegmentation" / "masks" / split).glob("*.npy"))
        if not self.mask_files:
            raise RuntimeError(f"No masks for split {split}: {root}")

    def __len__(self) -> int:
        return len(self.mask_files)

    def __getitem__(self, index: int) -> dict[str, object]:
        mask_path = self.mask_files[index]
        image_path = self.root / "BoneSegmentation" / "images" / f"{mask_path.stem}.bmp"
        image = Image.open(image_path).convert("L")
        mask_array = (np.load(mask_path) > 0).astype(np.float32)
        if mask_array.shape[-2:] != (image.height, image.width):
            raise RuntimeError({"case": mask_path.stem, "image": image.size, "mask": mask_array.shape})

        image_t = TF.pil_to_tensor(image).float() / 255.0
        mask = torch.from_numpy(mask_array)
        if mask_path.stem.endswith("_R"):
            image_t = torch.flip(image_t, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if self.augment:
            angle = random.uniform(-7.0, 7.0)
            translate = [random.randint(-10, 10), random.randint(-10, 10)]
            scale = random.uniform(0.94, 1.06)
            image_t = TF.affine(image_t, angle, translate, scale, 0.0,
                                interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(mask, angle, translate, scale, 0.0,
                             interpolation=InterpolationMode.NEAREST, fill=0.0)
            image_t = TF.adjust_gamma(image_t.clamp(0, 1), random.uniform(0.85, 1.15))
            image_t = (image_t * random.uniform(0.90, 1.10)).clamp(0, 1)

        image_t = (image_t - image_t.mean()) / (image_t.std() + 1e-6)
        height, width = image_t.shape[-2:]
        padded_h = ((height + self.divisor - 1) // self.divisor) * self.divisor
        padded_w = ((width + self.divisor - 1) // self.divisor) * self.divisor
        padding = (0, padded_w - width, 0, padded_h - height)
        image_t = F.pad(image_t, padding, value=0.0)
        mask = F.pad(mask, padding, value=0.0)
        valid = torch.zeros((1, padded_h, padded_w), dtype=torch.bool)
        valid[:, :height, :width] = True
        return {
            "image": image_t,
            "mask": mask,
            "valid": valid,
            "original_hw": torch.tensor([height, width], dtype=torch.int64),
            "case": mask_path.stem,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spatial_audit(dataset: NativeWristDataset) -> dict:
    original, padded = {}, {}
    for path in dataset.mask_files:
        shape = np.load(path, mmap_mode="r").shape[-2:]
        h, w = int(shape[0]), int(shape[1])
        ph = ((h + dataset.divisor - 1) // dataset.divisor) * dataset.divisor
        pw = ((w + dataset.divisor - 1) // dataset.divisor) * dataset.divisor
        original[f"{h}x{w}"] = original.get(f"{h}x{w}", 0) + 1
        padded[f"{ph}x{pw}"] = padded.get(f"{ph}x{pw}", 0) + 1
    return {
        "split": dataset.split,
        "num_images": len(dataset),
        "original_hw_counts": original,
        "padded_hw_counts": padded,
        "resize_used": False,
        "interpolation_used_for_spatial_size": False,
        "padding": "right/bottom zero padding to a multiple of 32",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/remote_variants/RAM-W600"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem"))
    parser.add_argument("--prior-weight", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=3241)
    parser.add_argument("--min-overlap-cases", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=24)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("R325 requires CUDA for native-resolution full-image training")
    device = torch.device("cuda")

    train_set = NativeWristDataset(args.dataset_root, "train", augment=True)
    val_set = NativeWristDataset(args.dataset_root, "val", augment=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    audit = {"train": spatial_audit(train_set), "val": spatial_audit(val_set)}
    (args.output / "spatial_preprocessing_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    pairs, graph_audit = build_overlap_graph(args.dataset_root, args.min_overlap_cases, args.max_pairs)
    (args.output / "overlap_relation_graph.json").write_text(
        json.dumps(graph_audit, indent=2), encoding="utf-8")

    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    initial_state = payload["model"]
    probe = NnUNetMultiLabelPrior().to(device)
    probe.load_state_dict(initial_state, strict=True)
    with torch.no_grad():
        sample = val_set[0]
        logits, _ = probe(sample["image"].unsqueeze(0).to(device), False)
        if logits.shape[-2:] != sample["mask"].shape[-2:]:
            raise RuntimeError({"probe_logits": logits.shape, "probe_mask": sample["mask"].shape})
    del probe
    torch.cuda.empty_cache()

    plain_ckpt, plain_best = train_arm(
        "plain_native", initial_state, train_set, val_loader, device, args.output, pairs,
        args.epochs, args.learning_rate, 0.0, args.min_epochs, args.patience, args.seed)
    prior_ckpt, prior_best = train_arm(
        f"overlap_iem_native_lambda_{args.prior_weight:g}", initial_state, train_set, val_loader,
        device, args.output, pairs, args.epochs, args.learning_rate, args.prior_weight,
        args.min_epochs, args.patience, args.seed)

    def evaluate(path: Path) -> dict:
        model = NnUNetMultiLabelPrior().to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
        result = validate(model, val_loader, device, pairs)
        del model
        torch.cuda.empty_cache()
        return result

    plain_val = evaluate(plain_ckpt)
    prior_val = evaluate(prior_ckpt)
    numeric = [
        "macro_dsc", "macro_iou", "voe", "macro_sensitivity", "macro_specificity",
        "overlap_dsc", "overlap_iou", "overlap_voe", "overlap_nsd_2px",
        "overlap_msd_px", "overlap_msd_fail_rate",
    ]
    result = {
        "experiment": "R325_RAM_NNUNET_NATIVE_RESOLUTION_OVERLAP_IEM",
        "split": "validation",
        "test_used": False,
        "threshold": 0.5,
        "threshold_search": False,
        "spatial_preprocessing": audit,
        "prior_weight": args.prior_weight,
        "initialization": {
            "source": str(args.baseline_checkpoint),
            "sha256": sha256(args.baseline_checkpoint),
            "strict_load": True,
        },
        "relation_graph": graph_audit,
        "plain_best": plain_best,
        "prior_best": prior_best,
        "plain_val": plain_val,
        "prior_val": prior_val,
        "delta_prior_minus_plain": {k: prior_val[k] - plain_val[k] for k in numeric},
        "checkpoints": {"plain": str(plain_ckpt), "prior": str(prior_ckpt)},
        "config": vars(args),
    }
    result["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in result["config"].items()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "complete", "output": str(args.output),
        "plain": plain_val, "prior": prior_val,
        "delta": result["delta_prior_minus_plain"],
    }), flush=True)


if __name__ == "__main__":
    main()
