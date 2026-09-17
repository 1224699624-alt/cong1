#!/usr/bin/env python3
"""R321 rib-paper-style prompt-free MedSAM adapter fine-tuning.

Both paired arms use the Space-Adapter and parallel MLP-Adapter layout from
Medical-SAM-Adapter inside every frozen SAM ViT block and train the mask decoder
without point, box, or mask prompts. The prior arm additionally injects the
frozen R317 map at embedding resolution and applies the exact continuous
background-prior loss.

Unlike R320, targets and priors are transformed into SAM's padded square
coordinate system before supervision. No cached image embeddings are used,
because gradients must pass through the image-encoder adapters.
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
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset

from r320_prior_plugin import binary_dice_bce_loss, continuous_background_prior_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r317_image_centers_only"))
    parser.add_argument("--medsam-repo", type=Path, default=Path("/root/autodl-tmp/external/MedSAM"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--adapter-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-alpha", type=float, default=0.035)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--disable-checkpointing", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image(root: Path, split: str, stem: str) -> Path:
    for extension in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = root / split / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image missing: {split}/{stem}")


def read_binary(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[..., 0]
    return (array > 0).astype(np.float32)


def preprocess_shape(height: int, width: int, long_side: int) -> tuple[int, int]:
    scale = float(long_side) / float(max(height, width))
    return int(height * scale + 0.5), int(width * scale + 0.5)


def resize_and_pad(array: np.ndarray, size: int, interpolation: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[:2]
    new_height, new_width = preprocess_shape(height, width, size)
    resized = cv2.resize(array, (new_width, new_height), interpolation=interpolation)
    output_shape = (size, size) if array.ndim == 2 else (size, size, array.shape[2])
    padded = np.zeros(output_shape, dtype=resized.dtype)
    padded[:new_height, :new_width, ...] = resized
    return padded, (new_height, new_width)


def complete_stems(args: argparse.Namespace, split: str) -> list[str]:
    stems: list[str] = []
    for label in sorted((args.raw_root / f"{split}_labels").glob("*.png")):
        if not (args.prior_root / split / label.name).is_file():
            raise FileNotFoundError(args.prior_root / split / label.name)
        find_image(args.raw_root, split, label.stem)
        stems.append(label.stem)
    limit = args.limit_train if split == "train" else args.limit_val
    return stems[:limit] if limit > 0 else stems


class SAMAlignedDataset(Dataset):
    def __init__(self, args: argparse.Namespace, split: str, stems: list[str], augment: bool, use_prior: bool) -> None:
        self.args = args
        self.split = split
        self.stems = stems
        self.augment = augment
        self.use_prior = use_prior

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, object]:
        stem = self.stems[index]
        gray = cv2.imread(str(find_image(self.args.raw_root, self.split, stem)), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(stem)
        target = read_binary(self.args.raw_root / f"{self.split}_labels" / f"{stem}.png")
        prior = np.asarray(
            Image.open(self.args.prior_root / self.split / f"{stem}.png").convert("L"), dtype=np.float32
        ) / 255.0
        if gray.shape != target.shape or prior.shape != target.shape:
            raise RuntimeError(f"unaligned inputs: {stem} {gray.shape} {target.shape} {prior.shape}")
        if self.augment and random.random() < 0.5:
            gray = np.ascontiguousarray(np.fliplr(gray))
            target = np.ascontiguousarray(np.fliplr(target))
            prior = np.ascontiguousarray(np.fliplr(prior))
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if self.augment and random.random() < 0.25:
            gamma = random.uniform(0.85, 1.20)
            gray = np.clip((gray.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        if not self.use_prior:
            prior = np.zeros_like(prior)

        rgb = np.repeat(gray[..., None], 3, axis=2)
        padded_rgb, input_size = resize_and_pad(rgb, self.args.image_size, cv2.INTER_LINEAR)
        padded_target, target_input_size = resize_and_pad(target, self.args.image_size, cv2.INTER_NEAREST)
        padded_prior, prior_input_size = resize_and_pad(prior, self.args.image_size, cv2.INTER_LINEAR)
        if input_size != target_input_size or input_size != prior_input_size:
            raise RuntimeError(f"SAM transform mismatch: {stem}")

        embedding_size = self.args.image_size // 16
        mask_size = self.args.image_size // 4
        target_mask = cv2.resize(padded_target, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
        prior_embedding = cv2.resize(
            padded_prior, (embedding_size, embedding_size), interpolation=cv2.INTER_LINEAR
        )
        prior_mask = cv2.resize(padded_prior, (mask_size, mask_size), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(padded_rgb.astype(np.float32)).permute(2, 0, 1)
        return {
            "image": image,
            "target": torch.from_numpy(target_mask[None].astype(np.float32)),
            "prior_embedding": torch.from_numpy(prior_embedding[None].astype(np.float32)),
            "prior_mask": torch.from_numpy(prior_mask[None].astype(np.float32)),
            "input_size": torch.tensor(input_size, dtype=torch.long),
            "original_size": torch.tensor(target.shape, dtype=torch.long),
            "stem": stem,
        }


class MedicalSAMAdapter(nn.Module):
    """Official Medical-SAM-Adapter MLP layout."""

    def __init__(self, embed_dim: int, ratio: float, skip_connect: bool) -> None:
        super().__init__()
        hidden_dim = max(1, int(embed_dim * ratio))
        self.down = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden_dim, embed_dim)
        self.skip_connect = skip_connect

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        adapted = self.up(self.activation(self.down(features)))
        return features + adapted if self.skip_connect else adapted


def window_partition(features: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    batch, height, width, channels = features.shape
    pad_height = (window_size - height % window_size) % window_size
    pad_width = (window_size - width % window_size) % window_size
    if pad_height or pad_width:
        features = F.pad(features, (0, 0, 0, pad_width, 0, pad_height))
    padded_height, padded_width = height + pad_height, width + pad_width
    windows = features.view(
        batch,
        padded_height // window_size,
        window_size,
        padded_width // window_size,
        window_size,
        channels,
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, channels)
    return windows, (padded_height, padded_width)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    padded_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> torch.Tensor:
    padded_height, padded_width = padded_shape
    height, width = original_shape
    batch = windows.shape[0] // (padded_height * padded_width // window_size // window_size)
    features = windows.view(
        batch,
        padded_height // window_size,
        padded_width // window_size,
        window_size,
        window_size,
        -1,
    )
    features = features.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch, padded_height, padded_width, -1)
    return features[:, :height, :width, :].contiguous()


class MedicalSAMAdapterBlock(nn.Module):
    """Wrap a pretrained SAM block with the official 2-D adapter paths."""

    def __init__(self, block: nn.Module, ratio: float) -> None:
        super().__init__()
        self.block = block
        embed_dim = int(block.norm1.normalized_shape[0])
        self.Space_Adapter = MedicalSAMAdapter(embed_dim, ratio, skip_connect=True)
        self.MLP_Adapter = MedicalSAMAdapter(embed_dim, ratio, skip_connect=False)
        self.scale = 0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        shortcut = features
        if self.block.window_size > 0:
            height, width = features.shape[1:3]
            features, padded_shape = window_partition(features, self.block.window_size)
        features = self.block.norm1(features)
        features = self.Space_Adapter(self.block.attn(features))
        if self.block.window_size > 0:
            features = window_unpartition(
                features, self.block.window_size, padded_shape, (height, width)
            )
        features = shortcut + features
        normalized = self.block.norm2(features)
        return features + self.block.mlp(normalized) + self.scale * self.MLP_Adapter(normalized)


class AdapterImageEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, adapter_ratio: float, use_checkpointing: bool) -> None:
        super().__init__()
        self.patch_embed = encoder.patch_embed
        self.pos_embed = encoder.pos_embed
        self.blocks = nn.ModuleList(MedicalSAMAdapterBlock(block, adapter_ratio) for block in encoder.blocks)
        self.neck = encoder.neck
        embed_dim = int(self.pos_embed.shape[-1])
        self.use_checkpointing = use_checkpointing

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(image)
        if self.pos_embed is not None:
            position = F.interpolate(
                self.pos_embed.permute(0, 3, 1, 2),
                size=features.shape[1:3],
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            features = features + position
        for block in self.blocks:
            if self.training and self.use_checkpointing:
                features = checkpoint(
                    block,
                    features,
                    use_reentrant=False,
                )
            else:
                features = block(features)
        return self.neck(features.permute(0, 3, 1, 2))


class PromptFreeAdapterMedSAM(nn.Module):
    def __init__(
        self, sam: nn.Module, adapter_ratio: float, use_checkpointing: bool, image_size: int
    ) -> None:
        super().__init__()
        for parameter in sam.parameters():
            parameter.requires_grad_(False)
        self.image_encoder = AdapterImageEncoder(sam.image_encoder, adapter_ratio, use_checkpointing)
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        for parameter in self.mask_decoder.parameters():
            parameter.requires_grad_(True)
        self.prior_adapter = nn.Conv2d(1, 256, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.prior_adapter.weight)
        self.register_buffer("pixel_mean", sam.pixel_mean.detach().clone(), persistent=False)
        self.register_buffer("pixel_std", sam.pixel_std.detach().clone(), persistent=False)
        self.image_size = image_size

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise RuntimeError(f"expected padded SAM image, got {tuple(image.shape)}")
        return (image - self.pixel_mean) / self.pixel_std

    def forward(self, image: torch.Tensor, prior_embedding: torch.Tensor) -> torch.Tensor:
        embedding = self.image_encoder(self.preprocess(image))
        embedding = embedding + self.prior_adapter(prior_embedding)
        with torch.no_grad():
            sparse, dense = self.prompt_encoder(points=None, boxes=None, masks=None)
        batch_size = embedding.shape[0]
        sparse = sparse.expand(batch_size, -1, -1)
        dense = F.interpolate(dense, embedding.shape[-2:], mode="bilinear", align_corners=False)
        dense = dense.expand(batch_size, -1, -1, -1)
        dense_pe = F.interpolate(
            self.prompt_encoder.get_dense_pe(), embedding.shape[-2:], mode="bilinear", align_corners=False
        )
        logits, _ = self.mask_decoder(
            image_embeddings=embedding,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        return logits

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: tuple[int, int], original_size: tuple[int, int]
    ) -> torch.Tensor:
        masks = F.interpolate(masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        masks = masks[..., : input_size[0], : input_size[1]]
        return F.interpolate(masks, original_size, mode="bilinear", align_corners=False)


def load_model(args: argparse.Namespace) -> PromptFreeAdapterMedSAM:
    sys.path.insert(0, str(args.medsam_repo))
    from segment_anything import sam_model_registry

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    model = PromptFreeAdapterMedSAM(
        sam, args.adapter_ratio, not args.disable_checkpointing, args.image_size
    )
    return model.to(args.device)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name in trainable}


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    missing = sorted(set(state) - set(current))
    if missing:
        raise RuntimeError(f"missing trainable keys: {missing[:5]}")
    current.update(state)
    model.load_state_dict(current, strict=True)


def make_loader(
    args: argparse.Namespace, split: str, stems: list[str], use_prior: bool, augment: bool, shuffle: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        SAMAlignedDataset(args, split, stems, augment, use_prior),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def validate(model: PromptFreeAdapterMedSAM, loader: DataLoader, args: argparse.Namespace) -> float:
    model.eval()
    intersection = prediction_sum = target_sum = 0.0
    for batch in loader:
        logits = model(batch["image"].to(args.device), batch["prior_embedding"].to(args.device))
        prediction = torch.sigmoid(logits) >= args.threshold
        target = batch["target"].to(args.device) > 0.5
        intersection += float((prediction & target).sum())
        prediction_sum += float(prediction.sum())
        target_sum += float(target.sum())
    return (2.0 * intersection + 1.0) / (prediction_sum + target_sum + 1.0)


def train_arm(
    args: argparse.Namespace,
    stems: dict[str, list[str]],
    arm: str,
    use_prior: bool,
    initial_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    seed_all(args.seed)
    model = load_model(args)
    load_trainable_state(model, initial_state)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_loader = make_loader(args, "train", stems["train"], use_prior, True, True)
    val_loader = make_loader(args, "val", stems["val"], use_prior, False, False)
    arm_dir = args.output_root / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, base_losses, prior_losses = [], [], []
        for batch in train_loader:
            image = batch["image"].to(args.device)
            target = batch["target"].to(args.device)
            prior_embedding = batch["prior_embedding"].to(args.device)
            prior_mask = batch["prior_mask"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=args.device.startswith("cuda"),
            ):
                logits = model(image, prior_embedding)
                base = binary_dice_bce_loss(logits, target)
                auxiliary = continuous_background_prior_loss(logits, target, prior_mask) if use_prior else base.new_zeros(())
                loss = base + args.prior_alpha * auxiliary
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 12.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient in {arm} epoch {epoch}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            base_losses.append(float(base.detach().cpu()))
            prior_losses.append(float(auxiliary.detach().cpu()))
        scheduler.step()
        dice = validate(model, val_loader, args)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(losses)),
            "base_loss": float(np.mean(base_losses)),
            "prior_loss": float(np.mean(prior_losses)),
            "weighted_prior_to_base": float(args.prior_alpha * np.mean(prior_losses) / max(np.mean(base_losses), 1e-8)),
            "val_dice": dice,
            "max_memory_gib": float(torch.cuda.max_memory_allocated() / (1024**3)) if args.device.startswith("cuda") else 0.0,
        }
        history.append(row)
        print(json.dumps({"backbone": "promptfree_medsam_adapter", "arm": arm, **row}), flush=True)
        if dice > best + 1e-4:
            best, best_epoch, bad = dice, epoch, 0
            torch.save({"trainable": trainable_state(model), "epoch": epoch, "val_dice": dice}, arm_dir / "best.pt")
        else:
            bad += 1
        if epoch >= args.min_epochs and bad >= args.patience:
            break
    (arm_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    del model
    torch.cuda.empty_cache()
    return {"arm": arm, "best_epoch": best_epoch, "best_val_dice": best, "completed_epochs": len(history)}


@torch.no_grad()
def infer_arm(args: argparse.Namespace, stems: dict[str, list[str]], arm: str, use_prior: bool) -> None:
    model = load_model(args)
    saved = torch.load(args.output_root / arm / "best.pt", map_location="cpu", weights_only=False)
    load_trainable_state(model, saved["trainable"])
    model.eval()
    dataset = SAMAlignedDataset(args, "val", stems["val"], False, use_prior)
    output_dir = args.output_root / arm / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in dataset:
        logits = model(
            item["image"].unsqueeze(0).to(args.device),
            item["prior_embedding"].unsqueeze(0).to(args.device),
        )
        input_size = tuple(map(int, item["input_size"].tolist()))
        original_size = tuple(map(int, item["original_size"].tolist()))
        restored = model.postprocess_masks(logits, input_size, original_size)
        prediction = torch.sigmoid(restored)[0, 0].cpu().numpy() >= args.threshold
        Image.fromarray(prediction.astype(np.uint8) * 255).save(output_dir / f"{item['stem']}.png")
    del model
    torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    stems = {split: complete_stems(args, split) for split in ("train", "val")}
    if args.limit_train <= 0 and args.limit_val <= 0 and {key: len(value) for key, value in stems.items()} != {"train": 875, "val": 96}:
        raise RuntimeError("unexpected full-data counts")
    args.output_root.mkdir(parents=True, exist_ok=True)
    initial_model = load_model(args)
    initial_state = trainable_state(initial_model)
    trainable_parameters = sum(parameter.numel() for parameter in initial_model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in initial_model.parameters())
    if int(torch.count_nonzero(initial_model.prior_adapter.weight)) != 0:
        raise RuntimeError("prior adapter must start neutral")
    torch.save({"trainable": initial_state, "seed": args.seed}, args.output_root / "shared_initialization.pt")
    del initial_model
    torch.cuda.empty_cache()
    protocol: dict[str, object] = {
        "experiment": "R321_PROMPTFREE_MEDSAM_ADAPTER_R317_PRIOR",
        "prompt_free": True,
        "points": None,
        "boxes": None,
        "mask_prompts": None,
        "image_encoder": "frozen SAM ViT-B blocks with trainable Medical-SAM-Adapter paths",
        "mask_decoder_trainable": True,
        "sam_aligned_target": True,
        "image_size": args.image_size,
        "embedding_size": args.image_size // 16,
        "mask_supervision_size": args.image_size // 4,
        "adapter_layout": "Medical-SAM-Adapter Space_Adapter + 0.5 * MLP_Adapter",
        "adapter_ratio": args.adapter_ratio,
        "adapter_hidden_dim": int(768 * args.adapter_ratio),
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "precision": "bfloat16 autocast",
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "splits": {key: len(value) for key, value in stems.items()},
        "prior": str(args.prior_root),
        "prior_alpha": args.prior_alpha,
        "shared_initialization": True,
        "test_used": False,
        "clean_test_v2_used": False,
    }
    (args.output_root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    results = []
    for arm, use_prior in (("plain", False), ("prior", True)):
        results.append(train_arm(args, stems, arm, use_prior, copy.deepcopy(initial_state)))
        infer_arm(args, stems, arm, use_prior)
    protocol["arms"] = results
    (args.output_root / "result.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == "__main__":
    main()
