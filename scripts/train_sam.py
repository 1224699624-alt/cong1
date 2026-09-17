#!/usr/bin/env python3
"""Fine-tune SAM mask decoder with box prompts from instance masks."""

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
    parser = argparse.ArgumentParser(description="Fine-tune SAM mask decoder on hand bone masks.")
    parser.add_argument("--dataset", required=True, help="Dataset folder under data/raw.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--medsam-checkpoint", dest="sam_checkpoint", help=argparse.SUPPRESS)
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
    parser.add_argument("--prompt-mode", default="box", choices=["box", "box+neg", "box+fgbg"], help="Use only box prompts or augment them with structured point prompts.")
    parser.add_argument("--num-positive-points", type=int, default=1)
    parser.add_argument("--num-negative-points", type=int, default=8)
    parser.add_argument("--negative-point-offset-ratio", type=float, default=0.08)
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
    parser.add_argument("--cache-embeddings", action="store_true", help="Cache frozen SAM image embeddings before training.")
    parser.add_argument("--cache-dir", default="outputs/sam_embedding_cache")
    parser.add_argument("--cache-dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--prompt-loss-weight", type=float, default=0.25, help="Extra supervision weight applied at prompt locations when point prompts are enabled.")
    parser.add_argument("--prompt-heatmap-sigma", type=float, default=4.0)
    parser.add_argument("--prompt-heatmap-loss-weight", type=float, default=0.15, help="Heatmap-style local supervision around structured prompt points.")
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.05, help="Boundary-aware foreground/background contrastive regularization weight.")
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--boundary-kernel-size", type=int, default=5)
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


def sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    center = points.mean(axis=0, keepdims=True)
    angles = np.arctan2(points[:, 1] - center[0, 1], points[:, 0] - center[0, 0])
    order = np.argsort(angles)
    return points[order]


def uniform_sample_contour(mask: np.ndarray, num_points: int) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((num_points, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    contour = sort_points_clockwise(contour)
    if contour.shape[0] == 1:
        return np.repeat(contour, num_points, axis=0)

    closed = np.vstack([contour, contour[:1]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    if total_length <= 1e-6:
        return np.repeat(contour[:1], num_points, axis=0)

    target_lengths = np.linspace(0.0, total_length, num_points, endpoint=False, dtype=np.float32)
    sampled_points: list[np.ndarray] = []
    seg_idx = 0
    for target_length in target_lengths:
        while seg_idx < len(seg_lengths) - 1 and cumulative[seg_idx + 1] < target_length:
            seg_idx += 1
        pt1 = closed[seg_idx]
        pt2 = closed[seg_idx + 1]
        denom = max(seg_lengths[seg_idx], 1e-6)
        ratio = (target_length - cumulative[seg_idx]) / denom
        sampled_points.append(pt1 + ratio * (pt2 - pt1))
    return np.asarray(sampled_points, dtype=np.float32)


def gaussian_heatmap(image_shape: tuple[int, int], point_xy: np.ndarray, sigma: float) -> np.ndarray:
    height, width = image_shape
    x = np.arange(width, dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    px, py = float(point_xy[0]), float(point_xy[1])
    heatmap = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma))
    return heatmap.astype(np.float32)


def sample_structured_mask_points(
    binary_mask: np.ndarray,
    num_positive_points: int,
    num_negative_points: int,
    negative_point_offset_ratio: float,
    prompt_mode: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if prompt_mode == "box":
        return None, None

    mask = binary_mask.astype(bool)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None, None

    x_min = float(xs.min())
    x_max = float(xs.max())
    y_min = float(ys.min())
    y_max = float(ys.max())
    box_w = max(1.0, x_max - x_min + 1.0)
    box_h = max(1.0, y_max - y_min + 1.0)
    offset = max(1, int(round(max(box_w, box_h) * negative_point_offset_ratio)))
    height, width = mask.shape
    contour_points = uniform_sample_contour(mask.astype(np.uint8), max(num_negative_points, 1))
    contour_center = contour_points.mean(axis=0, keepdims=True)

    coords: list[list[float]] = []
    labels: list[int] = []

    if prompt_mode == "box+fgbg":
        dist_map = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_index = np.unravel_index(np.argmax(dist_map), dist_map.shape)
        coords.append([float(max_index[1]), float(max_index[0])])
        labels.append(1)
        if num_positive_points > 1:
            eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
            inner_contour = uniform_sample_contour(eroded if eroded.sum() > 0 else mask.astype(np.uint8), num_positive_points - 1)
            for point in inner_contour:
                coords.append([float(point[0]), float(point[1])])
                labels.append(1)

    contour_points = uniform_sample_contour(mask.astype(np.uint8), num_negative_points)
    for point in contour_points:
        direction = point - contour_center[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm
        nx = int(np.clip(round(point[0] + direction[0] * offset), 0, width - 1))
        ny = int(np.clip(round(point[1] + direction[1] * offset), 0, height - 1))
        guard = 0
        while mask[ny, nx] and guard < max(offset * 2, 4):
            nx = int(np.clip(round(nx + direction[0]), 0, width - 1))
            ny = int(np.clip(round(ny + direction[1]), 0, height - 1))
            guard += 1
        coords.append([float(nx), float(ny)])
        labels.append(0)

    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int64)


class SAMBoxDataset(Dataset):
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
        prompt_mode: str,
        num_positive_points: int,
        num_negative_points: int,
        negative_point_offset_ratio: float,
        prompt_heatmap_sigma: float,
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
        self.prompt_mode = prompt_mode
        self.num_positive_points = num_positive_points
        self.num_negative_points = num_negative_points
        self.negative_point_offset_ratio = negative_point_offset_ratio
        self.prompt_heatmap_sigma = prompt_heatmap_sigma
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
        point_coords_list: list[np.ndarray] = []
        point_labels_list: list[np.ndarray] = []
        prompt_target_maps: list[np.ndarray] = []
        prompt_weight_maps: list[np.ndarray] = []
        if self.prompt_mode != "box":
            for mask_idx in range(masks.shape[0]):
                point_coords, point_labels = sample_structured_mask_points(
                    masks[mask_idx] > 0,
                    num_positive_points=self.num_positive_points,
                    num_negative_points=self.num_negative_points,
                    negative_point_offset_ratio=self.negative_point_offset_ratio,
                    prompt_mode=self.prompt_mode,
                )
                if point_coords is None or point_labels is None:
                    raise ValueError(f"Failed to sample structured prompts for {mask_path}")
                point_coords_resized = self.transform.apply_coords(point_coords, original_size).astype(np.float32)
                point_coords_list.append(point_coords_resized)
                point_labels_list.append(point_labels.astype(np.int64))
                target_map = np.zeros(input_size, dtype=np.float32)
                weight_map = np.zeros(input_size, dtype=np.float32)
                for coord, label in zip(point_coords_resized, point_labels):
                    heatmap = gaussian_heatmap(input_size, coord, sigma=self.prompt_heatmap_sigma)
                    if int(label) == 1:
                        target_map = np.maximum(target_map, heatmap)
                    else:
                        weight_map = np.maximum(weight_map, heatmap)
                prompt_target_maps.append(target_map)
                prompt_weight_maps.append(weight_map)

        mask_tensor = torch.as_tensor(resized_masks[:, None, :, :], dtype=torch.float32)
        box_tensor = torch.as_tensor(resized_boxes, dtype=torch.float32)
        sample = {
            "masks": mask_tensor,
            "boxes": box_tensor,
            "original_size": torch.as_tensor(original_size, dtype=torch.long),
            "input_size": torch.as_tensor(input_size, dtype=torch.long),
            "name": image_path.name,
        }
        if point_coords_list:
            sample["point_coords"] = torch.as_tensor(np.stack(point_coords_list, axis=0), dtype=torch.float32)
            sample["point_labels"] = torch.as_tensor(np.stack(point_labels_list, axis=0), dtype=torch.int64)
            prompt_target_tensor = pad_masks_to_square(np.stack(prompt_target_maps, axis=0), self.image_size)
            prompt_weight_tensor = pad_masks_to_square(np.stack(prompt_weight_maps, axis=0), self.image_size)
            sample["prompt_target_maps"] = torch.as_tensor(prompt_target_tensor[:, None, :, :], dtype=torch.float32)
            sample["prompt_weight_maps"] = torch.as_tensor(prompt_weight_tensor[:, None, :, :], dtype=torch.float32)
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
    checkpoint = Path(args.sam_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=args.device)
    return sam


def build_embedding_cache(
    sam: torch.nn.Module,
    dataset: SAMBoxDataset,
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
    point_coords = sample.get("point_coords")
    point_labels = sample.get("point_labels")
    if point_coords is not None:
        point_coords = point_coords.to(device)
    if point_labels is not None:
        point_labels = point_labels.to(device)
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
    prompt_points = None
    if point_coords is not None and point_labels is not None:
        prompt_points = (point_coords, point_labels)
    if train_prompt_encoder:
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=prompt_points,
            boxes=boxes,
            masks=None,
        )
    else:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=prompt_points,
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
    aux = {
        "image_embeddings": image_embeddings,
        "full_res_masks": masks,
    }
    return low_res_masks, target_masks, input_size, aux


def compute_prompt_loss(logits: torch.Tensor, sample: dict, device: str) -> torch.Tensor | None:
    point_coords = sample.get("point_coords")
    point_labels = sample.get("point_labels")
    if point_coords is None or point_labels is None:
        return None

    point_coords = point_coords.to(device=device, dtype=torch.float32)
    point_labels = point_labels.to(device=device, dtype=torch.float32)
    padded_height = max(1, sample["masks"].shape[-2] - 1)
    padded_width = max(1, sample["masks"].shape[-1] - 1)

    grid_x = (point_coords[..., 0] / padded_width) * 2.0 - 1.0
    grid_y = (point_coords[..., 1] / padded_height) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
    sampled_logits = F.grid_sample(logits, grid, mode="bilinear", align_corners=True).squeeze(1).squeeze(-1)
    return F.binary_cross_entropy_with_logits(sampled_logits, point_labels)


def compute_prompt_heatmap_loss(logits: torch.Tensor, sample: dict, device: str) -> torch.Tensor | None:
    prompt_target_maps = sample.get("prompt_target_maps")
    prompt_weight_maps = sample.get("prompt_weight_maps")
    if prompt_target_maps is None or prompt_weight_maps is None:
        return None

    prompt_target_maps = prompt_target_maps.to(device=device, dtype=torch.float32)
    prompt_weight_maps = prompt_weight_maps.to(device=device, dtype=torch.float32)
    prompt_target_maps = F.interpolate(prompt_target_maps, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    prompt_weight_maps = F.interpolate(prompt_weight_maps, size=logits.shape[-2:], mode="bilinear", align_corners=False)

    pos_weight = prompt_target_maps.clamp(0.0, 1.0)
    neg_weight = prompt_weight_maps.clamp(0.0, 1.0)
    weight_sum = pos_weight.sum() + neg_weight.sum()
    if float(weight_sum.item()) <= 0:
        return None

    probs = torch.sigmoid(logits)
    pos_loss = -(pos_weight * torch.log(probs.clamp_min(1e-6))).sum()
    neg_loss = -(neg_weight * torch.log((1.0 - probs).clamp_min(1e-6))).sum()
    return (pos_loss + neg_loss) / weight_sum


def compute_boundary_contrastive_loss(
    image_embeddings: torch.Tensor,
    sample: dict,
    device: str,
    boundary_kernel_size: int,
    temperature: float,
) -> torch.Tensor | None:
    masks = sample.get("masks")
    if masks is None:
        return None

    union_mask = masks.to(device=device, dtype=torch.float32).amax(dim=0, keepdim=True)
    union_mask = F.interpolate(union_mask, size=image_embeddings.shape[-2:], mode="nearest")
    fg_mask = (union_mask > 0.5).float()

    kernel_size = max(3, int(boundary_kernel_size) | 1)
    dilated = F.max_pool2d(fg_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = 1.0 - F.max_pool2d(1.0 - fg_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    bg_ring = (dilated - fg_mask).clamp_min(0.0)
    fg_core = eroded.clamp(0.0, 1.0)
    if float(fg_core.sum().item()) < 1:
        fg_core = fg_mask
    if float(bg_ring.sum().item()) < 1 or float(fg_mask.sum().item()) < 1:
        return None

    feats = image_embeddings.squeeze(0).permute(1, 2, 0).reshape(-1, image_embeddings.shape[1])
    fg_core_flat = fg_core.reshape(-1) > 0.5
    bg_ring_flat = bg_ring.reshape(-1) > 0.5
    if int(fg_core_flat.sum().item()) == 0 or int(bg_ring_flat.sum().item()) == 0:
        return None

    fg_feats = F.normalize(feats[fg_core_flat], dim=-1)
    bg_feats = F.normalize(feats[bg_ring_flat], dim=-1)
    fg_proto = F.normalize(fg_feats.mean(dim=0, keepdim=True), dim=-1)
    pos_sim = torch.matmul(fg_feats, fg_proto.t()).mean()
    neg_sim = torch.matmul(bg_feats, fg_proto.t()).mean()
    logits_ = torch.stack([pos_sim / temperature, neg_sim / temperature], dim=0).unsqueeze(0)
    labels_ = torch.zeros((1,), dtype=torch.long, device=device)
    return F.cross_entropy(logits_, labels_)


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
    prompt_loss_weight: float = 0.0,
    prompt_heatmap_loss_weight: float = 0.0,
    contrastive_loss_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    boundary_kernel_size: int = 5,
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
        logits, targets, _, aux = forward_sam_batch(
            sam,
            sample,
            device,
            gbc_adapter=gbc_adapter,
            train_prompt_encoder=train and train_prompt_encoder,
            train_image_encoder=train and train_image_encoder,
        )
        loss = criterion(logits, targets)
        if prompt_loss_weight > 0:
            prompt_loss = compute_prompt_loss(logits, sample, device)
            if prompt_loss is not None:
                loss = loss + prompt_loss_weight * prompt_loss
        if prompt_heatmap_loss_weight > 0:
            prompt_heatmap_loss = compute_prompt_heatmap_loss(logits, sample, device)
            if prompt_heatmap_loss is not None:
                loss = loss + prompt_heatmap_loss_weight * prompt_heatmap_loss
        if contrastive_loss_weight > 0:
            contrastive_loss = compute_boundary_contrastive_loss(
                aux["image_embeddings"],
                sample,
                device,
                boundary_kernel_size=boundary_kernel_size,
                temperature=contrastive_temperature,
            )
            if contrastive_loss is not None:
                loss = loss + contrastive_loss_weight * contrastive_loss
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
        raise ValueError("Please use --batch-size 1 for variable-instance SAM fine-tuning.")
    if args.cache_embeddings and args.train_image_encoder_last_n > 0:
        raise ValueError("--cache-embeddings cannot be used when fine-tuning image encoder blocks.")

    output = Path(args.output) if args.output else Path("outputs") / "sam_finetune" / args.dataset / "best_mask_decoder.pt"
    train_dataset = SAMBoxDataset(
        Path(args.raw_root),
        args.dataset,
        "train",
        args.image_size,
        args.box_padding_ratio,
        args.box_jitter_ratio,
        args.min_area,
        args.max_instances,
        args.prompt_mode,
        args.num_positive_points,
        args.num_negative_points,
        args.negative_point_offset_ratio,
        args.prompt_heatmap_sigma,
        cache_dir=Path(args.cache_dir) if args.cache_embeddings else None,
    )
    val_dataset = SAMBoxDataset(
        Path(args.raw_root),
        args.dataset,
        "val",
        args.image_size,
        args.box_padding_ratio,
        0.0,
        args.min_area,
        args.max_instances,
        args.prompt_mode,
        args.num_positive_points,
        args.num_negative_points,
        args.negative_point_offset_ratio,
        args.prompt_heatmap_sigma,
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
            prompt_loss_weight=args.prompt_loss_weight if args.prompt_mode != "box" else 0.0,
            prompt_heatmap_loss_weight=args.prompt_heatmap_loss_weight if args.prompt_mode != "box" else 0.0,
            contrastive_loss_weight=args.contrastive_loss_weight,
            contrastive_temperature=args.contrastive_temperature,
            boundary_kernel_size=args.boundary_kernel_size,
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
                    prompt_loss_weight=args.prompt_loss_weight if args.prompt_mode != "box" else 0.0,
                    prompt_heatmap_loss_weight=args.prompt_heatmap_loss_weight if args.prompt_mode != "box" else 0.0,
                    contrastive_loss_weight=args.contrastive_loss_weight,
                    contrastive_temperature=args.contrastive_temperature,
                    boundary_kernel_size=args.boundary_kernel_size,
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
