#!/usr/bin/env python3
"""Fine-tune MedSAM mask decoder with box prompts from instance masks."""

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
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from medsam_gbc import GBCAdapter


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedSAM mask decoder on hand bone masks.")
    parser.add_argument("--dataset", required=True, help="Dataset folder under data/raw.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--medsam-checkpoint", default="checkpoints/medsam_vit_b.pth")
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output", default=None, help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep 1 unless you customize the collate path.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--box-padding-ratio", type=float, default=0.08)
    parser.add_argument("--box-jitter-ratio", type=float, default=0.03)
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--max-instances", type=int, default=64, help="Randomly keep at most this many instances per image.")
    parser.add_argument("--use-gbc", action="store_true", help="Train a SCSegamba-style GBC adapter after the image encoder.")
    parser.add_argument("--gbc-reduction", type=int, default=8)
    parser.add_argument("--train-prompt-encoder", action="store_true", help="Also fine-tune the SAM prompt encoder for box prompts.")
    parser.add_argument(
        "--train-image-encoder-last-n",
        type=int,
        default=0,
        help="Fine-tune the last N image encoder blocks. Requires disabling --cache-embeddings.",
    )
    parser.add_argument("--prompt-lr", type=float, default=None, help="Learning rate for prompt encoder; defaults to --lr.")
    parser.add_argument("--image-encoder-lr", type=float, default=None, help="Learning rate for unfrozen image encoder blocks; defaults to --lr * 0.1.")
    parser.add_argument("--cache-embeddings", action="store_true", help="Cache frozen MedSAM image embeddings before training.")
    parser.add_argument("--cache-dir", default="outputs/medsam_embedding_cache")
    parser.add_argument("--cache-dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def find_image_files(image_dir: Path) -> list[Path]:
    files: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        files.extend(image_dir.glob(f"*{extension}"))
    return sorted(files)


def read_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def expand_and_jitter_box(
    box: np.ndarray,
    image_shape: tuple[int, int],
    padding_ratio: float,
    jitter_ratio: float,
    training: bool,
) -> np.ndarray:
    height, width = image_shape
    x1, y1, x2, y2 = box.astype(np.float32).tolist()
    box_w = max(1.0, x2 - x1 + 1.0)
    box_h = max(1.0, y2 - y1 + 1.0)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    if training and jitter_ratio > 0:
        jitter_x = box_w * jitter_ratio
        jitter_y = box_h * jitter_ratio
        x1 += random.uniform(-jitter_x, jitter_x)
        x2 += random.uniform(-jitter_x, jitter_x)
        y1 += random.uniform(-jitter_y, jitter_y)
        y2 += random.uniform(-jitter_y, jitter_y)

    return np.array(
        [
            np.clip(x1, 0, width - 1),
            np.clip(y1, 0, height - 1),
            np.clip(x2, 0, width - 1),
            np.clip(y2, 0, height - 1),
        ],
        dtype=np.float32,
    )


def instance_masks_and_boxes(
    instance_mask: np.ndarray,
    min_area: int,
    padding_ratio: float,
    jitter_ratio: float,
    training: bool,
    max_instances: int,
) -> tuple[np.ndarray, np.ndarray]:
    masks = []
    boxes = []
    instance_ids = [int(value) for value in np.unique(instance_mask) if int(value) > 0]
    if training:
        random.shuffle(instance_ids)
    for instance_id in instance_ids:
        binary = instance_mask == instance_id
        ys, xs = np.where(binary)
        if xs.size < min_area:
            continue
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
        box = expand_and_jitter_box(box, instance_mask.shape[:2], padding_ratio, jitter_ratio, training)
        masks.append(binary.astype(np.float32))
        boxes.append(box)
        if len(masks) >= max_instances:
            break

    if not masks:
        return np.zeros((0, *instance_mask.shape[:2]), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    return np.stack(masks, axis=0), np.stack(boxes, axis=0)


def pad_masks_to_square(masks: np.ndarray, target_size: int) -> np.ndarray:
    """Pad resized masks to the square canvas used by SAM preprocess."""
    _, height, width = masks.shape
    if height > target_size or width > target_size:
        raise ValueError(f"Mask size {(height, width)} exceeds target square size {target_size}.")
    pad_h = target_size - height
    pad_w = target_size - width
    if pad_h == 0 and pad_w == 0:
        return masks
    return np.pad(masks, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


class MedSAMBoxDataset(Dataset):
    def __init__(
        self,
        raw_root: Path,
        dataset_name: str,
        split: str,
        image_size: int,
        padding_ratio: float,
        jitter_ratio: float,
        min_area: int,
        max_instances: int,
        cache_dir: Path | None = None,
    ) -> None:
        self.image_dir = raw_root / dataset_name / split
        self.mask_dir = raw_root / dataset_name / f"{split}_labels"
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image split not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask split not found: {self.mask_dir}")
        self.image_paths = [
            image_path for image_path in find_image_files(self.image_dir)
            if (self.mask_dir / f"{image_path.stem}.png").exists()
        ]
        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size
        self.padding_ratio = padding_ratio
        self.jitter_ratio = jitter_ratio
        self.min_area = min_area
        self.max_instances = max_instances
        self.training = split == "train"
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    def cache_path(self, image_path: Path) -> Path:
        if self.cache_dir is None:
            raise ValueError("cache_dir is not configured")
        return self.cache_dir / self.image_dir.parent.name / self.image_dir.name / f"{image_path.stem}.pt"

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / f"{image_path.stem}.png"
        instance_mask = read_mask(mask_path)
        original_size = instance_mask.shape[:2]

        cached = None
        if self.cache_dir is not None:
            cache_path = self.cache_path(image_path)
            if cache_path.exists():
                cached = torch.load(cache_path, map_location="cpu")

        masks, boxes = instance_masks_and_boxes(
            instance_mask,
            min_area=self.min_area,
            padding_ratio=self.padding_ratio,
            jitter_ratio=self.jitter_ratio,
            training=self.training,
            max_instances=self.max_instances,
        )
        if masks.shape[0] == 0:
            raise ValueError(f"No valid instances in {mask_path}")

        if cached is None:
            image = read_rgb(image_path)
            resized_image = self.transform.apply_image(image)
            input_size = resized_image.shape[:2]
        else:
            resized_image = None
            input_size = tuple(int(value) for value in cached["input_size"])

        resized_masks = np.stack(
            [
                cv2.resize(mask, input_size[::-1], interpolation=cv2.INTER_NEAREST)
                for mask in masks
            ],
            axis=0,
        )
        resized_masks = pad_masks_to_square(resized_masks, self.image_size)
        resized_boxes = self.transform.apply_boxes(boxes, original_size)

        mask_tensor = torch.as_tensor(resized_masks[:, None, :, :], dtype=torch.float32)
        box_tensor = torch.as_tensor(resized_boxes, dtype=torch.float32)
        sample = {
            "masks": mask_tensor,
            "boxes": box_tensor,
            "original_size": torch.as_tensor(original_size, dtype=torch.long),
            "input_size": torch.as_tensor(input_size, dtype=torch.long),
            "name": image_path.name,
        }
        if cached is None:
            sample["image"] = torch.as_tensor(resized_image, dtype=torch.float32).permute(2, 0, 1)
        else:
            sample["image_embeddings"] = cached["image_embeddings"]
        return sample


def collate_single_image(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("This trainer currently expects --batch-size 1 because each image has variable boxes.")
    return batch[0]


class DiceBCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice = 1.0 - ((2.0 * intersection + 1.0) / (union + 1.0)).mean()
        return bce + dice


def load_sam(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = Path(args.medsam_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=args.device)
    return sam


def build_embedding_cache(
    sam: torch.nn.Module,
    dataset: MedSAMBoxDataset,
    device: str,
    cache_dtype: str,
) -> None:
    if dataset.cache_dir is None:
        return
    output_dtype = torch.float16 if cache_dtype == "fp16" else torch.float32
    sam.image_encoder.eval()
    iterator = tqdm(dataset.image_paths, desc=f"cache/{dataset.image_dir.parent.name}/{dataset.image_dir.name}")
    with torch.inference_mode():
        for image_path in iterator:
            cache_path = dataset.cache_path(image_path)
            if cache_path.exists():
                continue
            image = read_rgb(image_path)
            resized_image = dataset.transform.apply_image(image)
            image_tensor = torch.as_tensor(resized_image, dtype=torch.float32).permute(2, 0, 1).to(device)
            padded_image = sam.preprocess(image_tensor[None, :, :, :])
            image_embeddings = sam.image_encoder(padded_image).detach().cpu().to(output_dtype)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "image_embeddings": image_embeddings,
                    "input_size": list(resized_image.shape[:2]),
                    "original_size": list(image.shape[:2]),
                },
                cache_path,
            )


def set_trainable_modules(
    sam: torch.nn.Module,
    gbc_adapter: nn.Module | None = None,
    train_prompt_encoder: bool = False,
    train_image_encoder_last_n: int = 0,
) -> None:
    for parameter in sam.parameters():
        parameter.requires_grad = False
    for parameter in sam.mask_decoder.parameters():
        parameter.requires_grad = True

    if train_prompt_encoder:
        for parameter in sam.prompt_encoder.parameters():
            parameter.requires_grad = True

    if train_image_encoder_last_n > 0:
        blocks = getattr(sam.image_encoder, "blocks", None)
        if blocks is None:
            raise ValueError("This SAM image encoder does not expose a .blocks module list.")
        if train_image_encoder_last_n > len(blocks):
            raise ValueError(f"--train-image-encoder-last-n={train_image_encoder_last_n} exceeds {len(blocks)} blocks.")
        for block in blocks[-train_image_encoder_last_n:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    sam.image_encoder.train(train_image_encoder_last_n > 0)
    sam.prompt_encoder.train(train_prompt_encoder)
    sam.mask_decoder.train()
    if gbc_adapter is not None:
        gbc_adapter.train()
        for parameter in gbc_adapter.parameters():
            parameter.requires_grad = True


def forward_sam_batch(
    sam: torch.nn.Module,
    sample: dict,
    device: str,
    gbc_adapter: nn.Module | None = None,
    train_prompt_encoder: bool = False,
    train_image_encoder: bool = False,
) -> torch.Tensor:
    masks = sample["masks"].to(device)
    boxes = sample["boxes"].to(device)
    input_size = tuple(int(value) for value in sample["input_size"].tolist())

    if "image_embeddings" in sample:
        image_embeddings = sample["image_embeddings"].to(device=device, dtype=torch.float32)
    else:
        image = sample["image"].to(device)
        padded_image = sam.preprocess(image[None, :, :, :])
        if train_image_encoder:
            image_embeddings = sam.image_encoder(padded_image)
        else:
            with torch.no_grad():
                image_embeddings = sam.image_encoder(padded_image)
    if gbc_adapter is not None:
        image_embeddings = gbc_adapter(image_embeddings)
    if train_prompt_encoder:
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=None,
            boxes=boxes,
            masks=None,
        )
    else:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    target_masks = F.interpolate(masks, size=low_res_masks.shape[-2:], mode="nearest")
    return low_res_masks, target_masks, input_size


def run_epoch(
    sam: torch.nn.Module,
    gbc_adapter: nn.Module | None,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    epoch: int,
    train: bool,
    train_prompt_encoder: bool = False,
    train_image_encoder: bool = False,
) -> float:
    total_loss = 0.0
    iterator = tqdm(data_loader, desc=f"{'train' if train else 'val'} epoch {epoch}")
    sam.mask_decoder.train(train)
    sam.prompt_encoder.train(train and train_prompt_encoder)
    sam.image_encoder.train(train and train_image_encoder)
    if gbc_adapter is not None:
        gbc_adapter.train(train)
    for sample in iterator:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits, targets, _ = forward_sam_batch(
            sam,
            sample,
            device,
            gbc_adapter=gbc_adapter,
            train_prompt_encoder=train and train_prompt_encoder,
            train_image_encoder=train and train_image_encoder,
        )
        loss = criterion(logits, targets)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item())
        iterator.set_postfix(loss=f"{loss.item():.4f}", instances=int(targets.shape[0]))
    return total_loss / max(1, len(data_loader))


def save_checkpoint(
    path: Path,
    sam: torch.nn.Module,
    gbc_adapter: nn.Module | None,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "sam_model_type": args.sam_model_type,
            "mask_decoder": sam.mask_decoder.state_dict(),
            "prompt_encoder": sam.prompt_encoder.state_dict() if args.train_prompt_encoder else None,
            "image_encoder": sam.image_encoder.state_dict() if args.train_image_encoder_last_n > 0 else None,
            "gbc_adapter": gbc_adapter.state_dict() if gbc_adapter is not None else None,
            "use_gbc": gbc_adapter is not None,
            "train_prompt_encoder": args.train_prompt_encoder,
            "train_image_encoder_last_n": args.train_image_encoder_last_n,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.batch_size != 1:
        raise ValueError("Please use --batch-size 1 for variable-instance MedSAM fine-tuning.")
    if args.cache_embeddings and args.train_image_encoder_last_n > 0:
        raise ValueError("--cache-embeddings cannot be used when fine-tuning image encoder blocks.")

    output = Path(args.output) if args.output else Path("outputs") / "medsam_finetune" / args.dataset / "best_mask_decoder.pt"
    train_dataset = MedSAMBoxDataset(
        Path(args.raw_root),
        args.dataset,
        "train",
        args.image_size,
        args.box_padding_ratio,
        args.box_jitter_ratio,
        args.min_area,
        args.max_instances,
        cache_dir=Path(args.cache_dir) if args.cache_embeddings else None,
    )
    val_dataset = MedSAMBoxDataset(
        Path(args.raw_root),
        args.dataset,
        "val",
        args.image_size,
        args.box_padding_ratio,
        0.0,
        args.min_area,
        args.max_instances,
        cache_dir=Path(args.cache_dir) if args.cache_embeddings else None,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single_image,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_single_image,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    sam = load_sam(args)
    if args.cache_embeddings:
        build_embedding_cache(sam, train_dataset, args.device, args.cache_dtype)
        build_embedding_cache(sam, val_dataset, args.device, args.cache_dtype)
    gbc_adapter = GBCAdapter(in_channels=256, reduction=args.gbc_reduction).to(args.device) if args.use_gbc else None
    set_trainable_modules(
        sam,
        gbc_adapter=gbc_adapter,
        train_prompt_encoder=args.train_prompt_encoder,
        train_image_encoder_last_n=args.train_image_encoder_last_n,
    )
    criterion = DiceBCELoss()
    trainable_params = [{"params": sam.mask_decoder.parameters(), "lr": args.lr}]
    if gbc_adapter is not None:
        trainable_params.append({"params": gbc_adapter.parameters(), "lr": args.lr})
    if args.train_prompt_encoder:
        trainable_params.append({"params": sam.prompt_encoder.parameters(), "lr": args.prompt_lr or args.lr})
    if args.train_image_encoder_last_n > 0:
        image_lr = args.image_encoder_lr if args.image_encoder_lr is not None else args.lr * 0.1
        image_params = [parameter for parameter in sam.image_encoder.parameters() if parameter.requires_grad]
        trainable_params.append({"params": image_params, "lr": image_lr})
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=args.weight_decay)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            sam,
            gbc_adapter,
            train_loader,
            criterion,
            optimizer,
            args.device,
            epoch,
            train=True,
            train_prompt_encoder=args.train_prompt_encoder,
            train_image_encoder=args.train_image_encoder_last_n > 0,
        )
        val_loss = None
        if epoch % args.val_every == 0:
            with torch.no_grad():
                val_loss = run_epoch(
                    sam,
                    gbc_adapter,
                    val_loader,
                    criterion,
                    None,
                    args.device,
                    epoch,
                    train=False,
                    train_prompt_encoder=args.train_prompt_encoder,
                    train_image_encoder=args.train_image_encoder_last_n > 0,
                )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(output, sam, gbc_adapter, args, epoch, val_loss)
                print(f"Saved best checkpoint: {output} val_loss={val_loss:.5f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: {output}")


if __name__ == "__main__":
    main()
