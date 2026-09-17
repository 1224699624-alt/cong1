#!/usr/bin/env python3
"""R320 paired prompt-free MedSAM training with the frozen R317 prior.

No point, box, or mask prompt is used at training or inference. Frozen MedSAM
image embeddings are augmented by a zero-initialized prior-to-embedding adapter;
the mask decoder and adapter are trained with native Dice+BCE plus the exact
R317 continuous background penalty.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from r320_prior_plugin import binary_dice_bce_loss, continuous_background_prior_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r317_image_centers_only"))
    parser.add_argument("--medsam-repo", type=Path, default=Path("/root/autodl-tmp/external/MedSAM"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/cache/r320_medsam_embeddings"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-alpha", type=float, default=0.035)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--build-cache-only", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image(raw_root: Path, split: str, stem: str) -> Path:
    for extension in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = raw_root / split / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image missing for {split}/{stem}")


def read_binary(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return (array > 0).astype(np.float32)


def load_medsam(args: argparse.Namespace) -> nn.Module:
    sys.path.insert(0, str(args.medsam_repo))
    from segment_anything import sam_model_registry

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    return sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(args.device)


def case_stems(args: argparse.Namespace, split: str) -> list[str]:
    stems = []
    for label in sorted((args.raw_root / f"{split}_labels").glob("*.png")):
        prior = args.prior_root / split / label.name
        if not prior.is_file():
            raise FileNotFoundError(prior)
        find_image(args.raw_root, split, label.stem)
        stems.append(label.stem)
    limit = args.limit_train if split == "train" else args.limit_val
    return stems[:limit] if limit > 0 else stems


def build_embedding_cache(args: argparse.Namespace, sam: nn.Module, stems: dict[str, list[str]]) -> None:
    sys.path.insert(0, str(args.medsam_repo))
    from segment_anything.utils.transforms import ResizeLongestSide

    transform = ResizeLongestSide(sam.image_encoder.img_size)
    sam.image_encoder.eval()
    for split, split_stems in stems.items():
        output_dir = args.cache_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, stem in enumerate(split_stems, start=1):
            output = output_dir / f"{stem}.pt"
            if output.is_file():
                continue
            gray = cv2.imread(str(find_image(args.raw_root, split, stem)), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(stem)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            resized = transform.apply_image(rgb)
            tensor = torch.as_tensor(resized, dtype=torch.float32, device=args.device).permute(2, 0, 1)[None]
            with torch.inference_mode():
                embedding = sam.image_encoder(sam.preprocess(tensor)).cpu().half()
            torch.save(
                {
                    "embedding": embedding,
                    "input_size": list(resized.shape[:2]),
                    "original_size": list(gray.shape[:2]),
                },
                output,
            )
            if index % 50 == 0:
                print(json.dumps({"cache_split": split, "completed": index, "total": len(split_stems)}), flush=True)


class CachedMedSAMDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, stems: list[str], use_prior: bool) -> None:
        self.args, self.split, self.stems, self.use_prior = args, split, stems, use_prior

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, object]:
        stem = self.stems[index]
        cache = torch.load(self.args.cache_root / self.split / f"{stem}.pt", map_location="cpu", weights_only=False)
        target = read_binary(self.args.raw_root / f"{self.split}_labels" / f"{stem}.png")
        prior = np.asarray(Image.open(self.args.prior_root / self.split / f"{stem}.png").convert("L"), dtype=np.float32) / 255.0
        if prior.shape != target.shape:
            raise RuntimeError(f"unaligned prior: {stem}")
        if not self.use_prior:
            prior = np.zeros_like(prior)
        target_256 = cv2.resize(target, (256, 256), interpolation=cv2.INTER_NEAREST)
        prior_64 = cv2.resize(prior, (64, 64), interpolation=cv2.INTER_LINEAR)
        return {
            "embedding": cache["embedding"][0].float(),
            "target": torch.from_numpy(target_256[None].astype(np.float32)),
            "prior_64": torch.from_numpy(prior_64[None].astype(np.float32)),
            "input_size": torch.tensor(cache["input_size"], dtype=torch.long),
            "original_size": torch.tensor(cache["original_size"], dtype=torch.long),
            "stem": stem,
        }


class PromptFreeMedSAM(nn.Module):
    def __init__(self, sam: nn.Module) -> None:
        super().__init__()
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        for parameter in self.prompt_encoder.parameters():
            parameter.requires_grad_(False)
        self.prior_adapter = nn.Conv2d(1, 256, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.prior_adapter.weight)

    def forward(self, embedding: torch.Tensor, prior_64: torch.Tensor) -> torch.Tensor:
        enriched = embedding + self.prior_adapter(prior_64)
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(points=None, boxes=None, masks=None)
        logits, _ = self.mask_decoder(
            image_embeddings=enriched,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return logits


def make_loader(args: argparse.Namespace, split: str, stems: list[str], use_prior: bool, shuffle: bool) -> DataLoader:
    return DataLoader(
        CachedMedSAMDataset(args, split, stems, use_prior),
        batch_size=1,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def validate(model: PromptFreeMedSAM, loader: DataLoader, args: argparse.Namespace) -> float:
    model.eval()
    intersection = prediction_sum = target_sum = 0.0
    for batch in loader:
        logits = model(batch["embedding"].to(args.device), batch["prior_64"].to(args.device))
        prediction = torch.sigmoid(logits) >= args.threshold
        target = batch["target"].to(args.device) > 0.5
        intersection += float((prediction & target).sum())
        prediction_sum += float(prediction.sum())
        target_sum += float(target.sum())
    return (2.0 * intersection + 1.0) / (prediction_sum + target_sum + 1.0)


def train_arm(
    args: argparse.Namespace,
    sam: nn.Module,
    stems: dict[str, list[str]],
    initial_state: dict[str, torch.Tensor],
    arm: str,
    use_prior: bool,
) -> dict[str, object]:
    seed_all(args.seed)
    model = PromptFreeMedSAM(sam).to(args.device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    train_loader = make_loader(args, "train", stems["train"], use_prior, True)
    val_loader = make_loader(args, "val", stems["val"], use_prior, False)
    arm_dir = args.output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, base_losses, prior_losses = [], [], []
        for batch in train_loader:
            embedding = batch["embedding"].to(args.device)
            target = batch["target"].to(args.device)
            prior_64 = batch["prior_64"].to(args.device)
            prior_256 = F.interpolate(prior_64, size=(256, 256), mode="bilinear", align_corners=False)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
                logits = model(embedding, prior_64)
                base = binary_dice_bce_loss(logits, target)
                auxiliary = continuous_background_prior_loss(logits, target, prior_256) if use_prior else base.new_zeros(())
                loss = base + args.prior_alpha * auxiliary
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 12.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            base_losses.append(float(base.detach().cpu()))
            prior_losses.append(float(auxiliary.detach().cpu()))
        dice = validate(model, val_loader, args)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "base_loss": float(np.mean(base_losses)),
            "prior_loss": float(np.mean(prior_losses)),
            "weighted_prior_to_base": float(args.prior_alpha * np.mean(prior_losses) / max(np.mean(base_losses), 1e-8)),
            "val_dice": dice,
        }
        history.append(row)
        print(json.dumps({"backbone": "promptfree_medsam", "arm": arm, **row}), flush=True)
        if dice > best + 1e-4:
            best, best_epoch, bad = dice, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": dice}, arm_dir / "best.pt")
        else:
            bad += 1
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    (arm_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"arm": arm, "best_epoch": best_epoch, "best_val_dice": best, "completed_epochs": len(history)}


@torch.no_grad()
def infer_arm(args: argparse.Namespace, sam: nn.Module, stems: dict[str, list[str]], arm: str, use_prior: bool) -> None:
    checkpoint = torch.load(args.output_root / arm / "best.pt", map_location=args.device, weights_only=False)
    model = PromptFreeMedSAM(sam).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = CachedMedSAMDataset(args, "val", stems["val"], use_prior)
    output_dir = args.output_root / arm / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in dataset:
        low_resolution = model(item["embedding"].unsqueeze(0).to(args.device), item["prior_64"].unsqueeze(0).to(args.device))
        input_size = tuple(map(int, item["input_size"].tolist()))
        original_size = tuple(map(int, item["original_size"].tolist()))
        logits = sam.postprocess_masks(low_resolution, input_size, original_size)
        prediction = torch.sigmoid(logits)[0, 0].cpu().numpy() >= args.threshold
        Image.fromarray(prediction.astype(np.uint8) * 255).save(output_dir / f"{item['stem']}.png")


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    stems = {split: case_stems(args, split) for split in ("train", "val")}
    if args.limit_train <= 0 and args.limit_val <= 0 and {key: len(value) for key, value in stems.items()} != {"train": 875, "val": 96}:
        raise RuntimeError("unexpected full-data counts")
    sam = load_medsam(args)
    build_embedding_cache(args, sam, stems)
    if args.build_cache_only:
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    initial = PromptFreeMedSAM(sam)
    if int(torch.count_nonzero(initial.prior_adapter.weight)) != 0:
        raise RuntimeError("prior adapter must start neutral")
    initial_state = copy.deepcopy(initial.state_dict())
    torch.save({"model": initial_state, "seed": args.seed}, args.output_root / "shared_initialization.pt")
    protocol = {
        "experiment": "R320_PROMPTFREE_MEDSAM_R317_PRIOR",
        "prompt_free": True,
        "points": None,
        "boxes": None,
        "mask_prompts": None,
        "splits": {key: len(value) for key, value in stems.items()},
        "prior_alpha": args.prior_alpha,
        "shared_initialization": True,
        "test_used": False,
        "clean_test_v2_used": False,
    }
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for arm, use_prior in (("plain", False), ("prior", True)):
        results.append(train_arm(args, sam, stems, copy.deepcopy(initial_state), arm, use_prior))
        infer_arm(args, sam, stems, arm, use_prior)
    protocol["arms"] = results
    (args.output_root / "result.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
