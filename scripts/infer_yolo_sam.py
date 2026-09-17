#!/usr/bin/env python3
"""Run YOLO detection and SAM box-prompt segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from medsam_gbc import GBCAdapter

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "segment-anything is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO boxes + SAM segmentation inference.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/raw.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-root", default="outputs/predictions")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument(
        "--sam-backend",
        default="sam",
        choices=["sam", "hq_sam"],
        help="Use the original SAM predictor or switch to HQ-SAM while keeping the same prompt flow.",
    )
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--medsam-checkpoint", dest="sam_checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--finetuned-checkpoint", default=None, help="Optional fine-tuned mask decoder checkpoint. Only supported with --sam-backend sam.")
    parser.add_argument("--sam-model-type", default="vit_b", help="SAM backbone type, e.g. vit_b/vit_l/vit_h. HQ-SAM also supports vit_tiny if installed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--box-padding-ratio", type=float, default=0.05)
    parser.add_argument("--prompt-mode", default="box", choices=["box", "box+neg", "box+fgbg"], help="Use only box prompts or augment them with structured point prompts.")
    parser.add_argument("--num-positive-points", type=int, default=1)
    parser.add_argument("--num-negative-points", type=int, default=8)
    parser.add_argument("--negative-point-offset-ratio", type=float, default=0.08, help="Relative offset used to place structured background points outside each box.")
    parser.add_argument("--dynamic-correction", action="store_true", help="Try multiple box padding ratios and select a mask with box-mask consistency rules.")
    parser.add_argument("--dynamic-padding-ratios", default="0.00,0.03,0.05,0.08,0.10,0.15", help="Comma-separated padding ratios used when --dynamic-correction is enabled.")
    parser.add_argument("--dynamic-base-padding-ratio", type=float, default=None, help="Baseline padding ratio used as fallback. Defaults to --box-padding-ratio.")
    parser.add_argument("--dynamic-switch-margin", type=float, default=0.02, help="Only switch away from the baseline candidate if the dynamic score improves by this margin.")
    parser.add_argument("--dynamic-target-mask-box-ratio", type=float, default=0.45, help="Preferred mask-area / box-union-area ratio for dynamic correction.")
    parser.add_argument("--dynamic-max-outside-box-ratio", type=float, default=0.08, help="Soft upper bound for mask pixels outside the candidate box union.")
    parser.add_argument("--mask-to-prompt-refine", action="store_true", help="Run a second SAM pass using prompts generated from the first-pass mask.")
    parser.add_argument("--refine-box-padding-ratio", type=float, default=0.03, help="Padding ratio for boxes regenerated from first-pass masks.")
    parser.add_argument("--refine-num-positive-points", type=int, default=1, help="Foreground points sampled from first-pass masks in the refinement pass.")
    parser.add_argument("--refine-num-negative-points", type=int, default=4, help="Background points sampled around first-pass masks in the refinement pass.")
    parser.add_argument("--refine-negative-dilate-kernel", type=int, default=9, help="Kernel size for sampling near-boundary negative points.")
    parser.add_argument("--refine-iters", type=int, default=2, help="Number of SAM passes used for mask-to-prompt refinement.")
    parser.add_argument(
        "--refine-prompt-scheme",
        default="box+point+mask",
        choices=["box", "box+point", "box+mask", "box+point+mask"],
        help="Prompt combination used during iterative mask refinement.",
    )
    parser.add_argument("--disable-refine-mask-input", action="store_true", help="Do not pass first-pass low-res logits as SAM mask_input during refinement.")
    parser.add_argument("--box-source", default="yolo", choices=["yolo", "gt"], help="Use YOLO predicted boxes or GT boxes from masks.")
    parser.add_argument("--min-mask-area", type=int, default=20)
    parser.add_argument("--save-overlays", action="store_true")
    return parser.parse_args()


def find_image_files(image_dir: Path) -> list[Path]:
    image_files: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        image_files.extend(image_dir.glob(f"*{extension}"))
    return sorted(image_files)


def read_rgb(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def expand_boxes(boxes: np.ndarray, image_shape: tuple[int, int], padding_ratio: float) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4).astype(np.float32)

    height, width = image_shape
    expanded = boxes.astype(np.float32).copy()
    box_widths = expanded[:, 2] - expanded[:, 0] + 1.0
    box_heights = expanded[:, 3] - expanded[:, 1] + 1.0
    expanded[:, 0] -= box_widths * padding_ratio
    expanded[:, 2] += box_widths * padding_ratio
    expanded[:, 1] -= box_heights * padding_ratio
    expanded[:, 3] += box_heights * padding_ratio
    expanded[:, [0, 2]] = np.clip(expanded[:, [0, 2]], 0, width - 1)
    expanded[:, [1, 3]] = np.clip(expanded[:, [1, 3]], 0, height - 1)
    return expanded


def parse_float_list(value: str) -> list[float]:
    ratios = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ratios.append(float(item))
    if not ratios:
        raise ValueError("At least one dynamic padding ratio is required.")
    return sorted(set(ratios))


def boxes_to_union_mask(boxes: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    union = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in boxes.astype(int):
        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        if x2 >= x1 and y2 >= y1:
            union[y1 : y2 + 1, x1 : x2 + 1] = True
    return union


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def connected_component_penalty(mask: np.ndarray, min_area: int = 20) -> float:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    small_components = int((areas < min_area).sum())
    component_excess = max(0, num_labels - 2)
    return min(1.0, 0.05 * component_excess + 0.08 * small_components)


def score_dynamic_candidate(
    mask: np.ndarray,
    boxes: np.ndarray,
    all_masks: list[np.ndarray],
    image_shape: tuple[int, int],
    target_mask_box_ratio: float,
    max_outside_box_ratio: float,
) -> dict[str, float]:
    mask_bool = mask.astype(bool)
    mask_area = float(mask_bool.sum())
    box_union = boxes_to_union_mask(boxes, image_shape)
    box_area = float(box_union.sum())
    mask_box_ratio = mask_area / box_area if box_area > 0 else 0.0
    outside_ratio = float(np.logical_and(mask_bool, ~box_union).sum() / mask_area) if mask_area > 0 else 0.0
    stability = float(np.mean([mask_iou(mask_bool, other) for other in all_masks])) if all_masks else 1.0
    area_penalty = min(1.0, abs(mask_box_ratio - target_mask_box_ratio) / max(target_mask_box_ratio, 1e-6))
    outside_penalty = max(0.0, outside_ratio - max_outside_box_ratio) / max(1.0 - max_outside_box_ratio, 1e-6)
    component_penalty = connected_component_penalty(mask_bool)
    score = stability - 0.35 * area_penalty - 0.30 * outside_penalty - 0.20 * component_penalty
    return {
        "score": float(score),
        "stability": stability,
        "mask_box_ratio": float(mask_box_ratio),
        "outside_box_ratio": outside_ratio,
        "component_penalty": component_penalty,
    }


def boxes_from_gt_mask(mask_path: Path) -> np.ndarray:
    if not mask_path.exists():
        raise FileNotFoundError(f"GT mask not found: {mask_path}")
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    boxes = []
    for instance_id in sorted(int(value) for value in np.unique(mask) if int(value) > 0):
        ys, xs = np.where(mask == instance_id)
        if xs.size == 0:
            continue
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    if not boxes:
        return np.empty((0, 4), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def sample_structured_box_points(
    boxes_xyxy: np.ndarray,
    image_shape: tuple[int, int],
    prompt_mode: str,
    num_positive_points: int,
    num_negative_points: int,
    negative_point_offset_ratio: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if prompt_mode == "box" or boxes_xyxy.size == 0:
        return None, None

    height, width = image_shape
    prompt_coords: list[np.ndarray] = []
    prompt_labels: list[np.ndarray] = []

    for x1, y1, x2, y2 in boxes_xyxy.astype(np.float32):
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        half_w = box_w * 0.5
        half_h = box_h * 0.5
        offset = max(1.0, max(box_w, box_h) * negative_point_offset_ratio)

        coords_for_box: list[list[float]] = []
        labels_for_box: list[int] = []

        if prompt_mode == "box+fgbg":
            coords_for_box.append([cx, cy])
            labels_for_box.append(1)
            if num_positive_points > 1:
                pos_angles = np.linspace(0.0, 2.0 * np.pi, num_positive_points - 1, endpoint=False, dtype=np.float32)
                for angle in pos_angles:
                    px = cx + np.cos(angle) * half_w * 0.35
                    py = cy + np.sin(angle) * half_h * 0.35
                    coords_for_box.append([float(np.clip(px, 0, width - 1)), float(np.clip(py, 0, height - 1))])
                    labels_for_box.append(1)

        neg_angles = np.linspace(0.0, 2.0 * np.pi, num_negative_points, endpoint=False, dtype=np.float32)
        for angle in neg_angles:
            nx = cx + np.cos(angle) * (half_w + offset)
            ny = cy + np.sin(angle) * (half_h + offset)
            coords_for_box.append([float(np.clip(nx, 0, width - 1)), float(np.clip(ny, 0, height - 1))])
            labels_for_box.append(0)

        prompt_coords.append(np.asarray(coords_for_box, dtype=np.float32))
        prompt_labels.append(np.asarray(labels_for_box, dtype=np.int64))

    return np.stack(prompt_coords, axis=0), np.stack(prompt_labels, axis=0)


def resolve_sam_backend(
    backend: str,
) -> tuple[type[Any], dict[str, Any]]:
    if backend == "sam":
        return SamPredictor, sam_model_registry
    if backend == "hq_sam":
        try:
            from segment_anything_hq import SamPredictor as HQSamPredictor
            from segment_anything_hq import sam_model_registry as hq_sam_model_registry
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "HQ-SAM backend requested, but segment-anything-hq is not installed. "
                "Install it with: pip install segment-anything-hq"
            ) from exc
        return HQSamPredictor, hq_sam_model_registry
    raise ValueError(f"Unsupported SAM backend: {backend}")


def load_sam(
    checkpoint: Path,
    model_type: str,
    device: str,
    backend: str = "sam",
    finetuned_checkpoint: Path | None = None,
) -> Any:
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    predictor_cls, model_registry = resolve_sam_backend(backend)
    if model_type not in model_registry:
        available_models = ", ".join(sorted(model_registry.keys()))
        raise ValueError(f"Model type '{model_type}' is not available for backend '{backend}'. Available: {available_models}")
    sam_model = model_registry[model_type](checkpoint=str(checkpoint))
    if finetuned_checkpoint is not None:
        if backend != "sam":
            raise ValueError("Fine-tuned SAM decoder checkpoints are only supported with --sam-backend sam.")
        if not finetuned_checkpoint.exists():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {finetuned_checkpoint}")
        payload = torch.load(finetuned_checkpoint, map_location=device)
        state_dict = payload.get("mask_decoder", payload) if isinstance(payload, dict) else payload
        missing, unexpected = sam_model.mask_decoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing mask_decoder keys: {missing[:5]}")
        if unexpected:
            print(f"Warning: unexpected mask_decoder keys: {unexpected[:5]}")
        prompt_state = payload.get("prompt_encoder") if isinstance(payload, dict) else None
        if prompt_state is not None:
            missing, unexpected = sam_model.prompt_encoder.load_state_dict(prompt_state, strict=False)
            if missing:
                print(f"Warning: missing prompt_encoder keys: {missing[:5]}")
            if unexpected:
                print(f"Warning: unexpected prompt_encoder keys: {unexpected[:5]}")
            print("Loaded fine-tuned prompt encoder.")
        image_state = payload.get("image_encoder") if isinstance(payload, dict) else None
        if image_state is not None:
            missing, unexpected = sam_model.image_encoder.load_state_dict(image_state, strict=False)
            if missing:
                print(f"Warning: missing image_encoder keys: {missing[:5]}")
            if unexpected:
                print(f"Warning: unexpected image_encoder keys: {unexpected[:5]}")
            print("Loaded fine-tuned image encoder.")
        gbc_state = payload.get("gbc_adapter") if isinstance(payload, dict) else None
        if gbc_state is not None:
            gbc_adapter = GBCAdapter(in_channels=256)
            gbc_adapter.load_state_dict(gbc_state, strict=True)
            gbc_adapter.to(device=device)
            gbc_adapter.eval()
            sam_model.gbc_adapter = gbc_adapter
            print("Loaded GBC adapter from fine-tuned checkpoint.")
        print(f"Loaded fine-tuned mask decoder: {finetuned_checkpoint}")
    sam_model.to(device=device)
    sam_model.eval()
    return predictor_cls(sam_model)


def predict_sam_masks(
    predictor: Any,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    device: str,
    prompt_mode: str = "box",
    num_positive_points: int = 1,
    num_negative_points: int = 8,
    negative_point_offset_ratio: float = 0.08,
) -> np.ndarray:
    masks, _ = predict_sam_instance_masks(
        predictor,
        image_rgb,
        boxes_xyxy,
        device,
        prompt_mode=prompt_mode,
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        negative_point_offset_ratio=negative_point_offset_ratio,
    )
    if masks.size == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    return masks.any(axis=0).astype(np.uint8)


def predict_sam_instance_masks(
    predictor: Any,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    device: str,
    prompt_mode: str = "box",
    num_positive_points: int = 1,
    num_negative_points: int = 8,
    negative_point_offset_ratio: float = 0.08,
    point_coords_np: np.ndarray | None = None,
    point_labels_np: np.ndarray | None = None,
    mask_input: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    height, width = image_rgb.shape[:2]
    if boxes_xyxy.size == 0:
        return np.zeros((0, height, width), dtype=np.uint8), None

    predictor.set_image(image_rgb)
    gbc_adapter = getattr(predictor.model, "gbc_adapter", None)
    if gbc_adapter is not None:
        with torch.no_grad():
            predictor.features = gbc_adapter(predictor.features)
    boxes_torch = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_torch, image_rgb.shape[:2])
    prompt_coords_np = point_coords_np
    prompt_labels_np = point_labels_np
    if point_coords_np is None or point_labels_np is None:
        prompt_coords_np, prompt_labels_np = sample_structured_box_points(
            boxes_xyxy,
            image_rgb.shape[:2],
            prompt_mode=prompt_mode,
            num_positive_points=num_positive_points,
            num_negative_points=num_negative_points,
            negative_point_offset_ratio=negative_point_offset_ratio,
        )
    point_coords_torch = None
    point_labels_torch = None
    if prompt_coords_np is not None and prompt_labels_np is not None:
        point_coords_torch = torch.as_tensor(prompt_coords_np, dtype=torch.float32, device=device)
        point_coords_torch = predictor.transform.apply_coords_torch(point_coords_torch, image_rgb.shape[:2])
        point_labels_torch = torch.as_tensor(prompt_labels_np, dtype=torch.int64, device=device)
    mask_input_torch = mask_input.to(device=device) if mask_input is not None else None
    with torch.no_grad():
        masks, _, low_res_masks = predictor.predict_torch(
            point_coords=point_coords_torch,
            point_labels=point_labels_torch,
            boxes=transformed_boxes,
            mask_input=mask_input_torch,
            multimask_output=False,
        )
    instance_masks = masks[:, 0].detach().cpu().numpy().astype(np.uint8)
    return instance_masks, low_res_masks[:, 0:1].detach()


def mask_to_box(mask: np.ndarray, fallback_box: np.ndarray, image_shape: tuple[int, int], padding_ratio: float) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return fallback_box.astype(np.float32)
    box = np.asarray([[xs.min(), ys.min(), xs.max(), ys.max()]], dtype=np.float32)
    return expand_boxes(box, image_shape, padding_ratio)[0]


def sample_mask_refine_points(
    mask: np.ndarray,
    fallback_box: np.ndarray,
    image_shape: tuple[int, int],
    num_positive_points: int,
    num_negative_points: int,
    negative_dilate_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    mask_u8 = (mask > 0).astype(np.uint8)
    coords: list[list[float]] = []
    labels: list[int] = []

    if mask_u8.sum() > 0 and num_positive_points > 0:
        distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        flat_indices = np.argsort(distance.ravel())[::-1]
        used: list[tuple[int, int]] = []
        min_gap = max(2.0, np.sqrt(float(mask_u8.sum())) * 0.15)
        for flat_index in flat_indices:
            if len(used) >= num_positive_points:
                break
            y, x = np.unravel_index(int(flat_index), distance.shape)
            if distance[y, x] <= 0:
                break
            if all((x - ux) ** 2 + (y - uy) ** 2 >= min_gap**2 for ux, uy in used):
                coords.append([float(x), float(y)])
                labels.append(1)
                used.append((int(x), int(y)))

    if not coords and num_positive_points > 0:
        x1, y1, x2, y2 = fallback_box.astype(np.float32)
        coords.append([float(np.clip((x1 + x2) * 0.5, 0, width - 1)), float(np.clip((y1 + y2) * 0.5, 0, height - 1))])
        labels.append(1)
    while labels.count(1) < num_positive_points:
        coords.append(coords[0])
        labels.append(1)

    if num_negative_points > 0:
        kernel_size = max(3, int(negative_dilate_kernel) | 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        ring = cv2.dilate(mask_u8, kernel, iterations=1).astype(bool) & ~mask_u8.astype(bool)
        ys, xs = np.where(ring)
        if xs.size == 0:
            x1, y1, x2, y2 = fallback_box.astype(np.float32)
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            half_w = max(1.0, (x2 - x1) * 0.5)
            half_h = max(1.0, (y2 - y1) * 0.5)
            angles = np.linspace(0, 2 * np.pi, num_negative_points, endpoint=False, dtype=np.float32)
            for angle in angles:
                coords.append([
                    float(np.clip(cx + np.cos(angle) * half_w * 1.15, 0, width - 1)),
                    float(np.clip(cy + np.sin(angle) * half_h * 1.15, 0, height - 1)),
                ])
                labels.append(0)
        else:
            center_x = float(xs.mean())
            center_y = float(ys.mean())
            angles = np.arctan2(ys - center_y, xs - center_x)
            order = np.argsort(angles)
            sample_positions = np.linspace(0, len(order) - 1, num_negative_points, dtype=np.int64)
            for pos in sample_positions:
                idx = order[int(pos)]
                coords.append([float(xs[idx]), float(ys[idx])])
                labels.append(0)

    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def predict_mask_to_prompt_refined_masks(
    predictor: Any,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    device: str,
    prompt_mode: str = "box",
    num_positive_points: int = 1,
    num_negative_points: int = 8,
    negative_point_offset_ratio: float = 0.08,
    refine_box_padding_ratio: float = 0.03,
    refine_num_positive_points: int = 1,
    refine_num_negative_points: int = 4,
    refine_negative_dilate_kernel: int = 9,
    refine_iters: int = 2,
    refine_prompt_scheme: str = "box+point+mask",
    use_mask_input: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image_rgb.shape[:2]
    if boxes_xyxy.size == 0:
        return np.zeros((height, width), dtype=np.uint8), boxes_xyxy.reshape(0, 4), {"refined_instances": 0}

    refine_iters = max(1, int(refine_iters))
    use_refine_box = "box" in refine_prompt_scheme
    use_refine_point = "point" in refine_prompt_scheme
    use_refine_mask = "mask" in refine_prompt_scheme and use_mask_input
    current_boxes = boxes_xyxy.astype(np.float32)
    current_masks, low_res_masks = predict_sam_instance_masks(
        predictor,
        image_rgb,
        current_boxes,
        device,
        prompt_mode=prompt_mode,
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        negative_point_offset_ratio=negative_point_offset_ratio,
    )
    last_refined_boxes = current_boxes
    for _ in range(1, refine_iters):
        refined_boxes = []
        point_coords = []
        point_labels = []
        for mask, fallback_box in zip(current_masks, current_boxes, strict=False):
            refined_box = mask_to_box(mask, fallback_box, image_rgb.shape[:2], refine_box_padding_ratio)
            coords = None
            labels = None
            if use_refine_point:
                coords, labels = sample_mask_refine_points(
                    mask,
                    refined_box,
                    image_rgb.shape[:2],
                    num_positive_points=refine_num_positive_points,
                    num_negative_points=refine_num_negative_points,
                    negative_dilate_kernel=refine_negative_dilate_kernel,
                )
                point_coords.append(coords)
                point_labels.append(labels)
            refined_boxes.append(refined_box if use_refine_box else fallback_box.astype(np.float32))

        last_refined_boxes = np.asarray(refined_boxes, dtype=np.float32)
        point_coords_np = np.stack(point_coords, axis=0) if use_refine_point and point_coords else None
        point_labels_np = np.stack(point_labels, axis=0) if use_refine_point and point_labels else None
        current_masks, low_res_masks = predict_sam_instance_masks(
            predictor,
            image_rgb,
            last_refined_boxes,
            device,
            prompt_mode="box",
            point_coords_np=point_coords_np,
            point_labels_np=point_labels_np,
            mask_input=low_res_masks if use_refine_mask else None,
        )
        current_boxes = last_refined_boxes

    merged = current_masks.any(axis=0).astype(np.uint8) if current_masks.size else np.zeros((height, width), dtype=np.uint8)
    debug = {
        "refined_instances": int(len(last_refined_boxes)),
        "refine_prompt_scheme": refine_prompt_scheme,
        "refine_used_box_prompt": bool(use_refine_box),
        "refine_used_point_prompt": bool(use_refine_point),
        "refine_used_mask_input": bool(use_refine_mask),
        "refine_box_padding_ratio": float(refine_box_padding_ratio),
        "refine_iters": int(refine_iters),
    }
    return merged, last_refined_boxes, debug


def predict_dynamic_sam_masks(
    predictor: Any,
    image_rgb: np.ndarray,
    raw_boxes_xyxy: np.ndarray,
    device: str,
    padding_ratios: list[float],
    base_padding_ratio: float,
    switch_margin: float,
    target_mask_box_ratio: float,
    max_outside_box_ratio: float,
    prompt_mode: str = "box",
    num_positive_points: int = 1,
    num_negative_points: int = 8,
    negative_point_offset_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    height, width = image_rgb.shape[:2]
    if raw_boxes_xyxy.size == 0:
        return np.zeros((height, width), dtype=np.uint8), raw_boxes_xyxy.reshape(0, 4), {
            "selected_padding_ratio": None,
            "dynamic_candidates": [],
        }

    if base_padding_ratio not in padding_ratios:
        padding_ratios = sorted(set([*padding_ratios, base_padding_ratio]))

    candidates: list[dict[str, object]] = []
    masks: list[np.ndarray] = []
    for ratio in padding_ratios:
        boxes = expand_boxes(raw_boxes_xyxy, image_rgb.shape[:2], ratio)
        mask = predict_sam_masks(
            predictor,
            image_rgb,
            boxes,
            device,
            prompt_mode=prompt_mode,
            num_positive_points=num_positive_points,
            num_negative_points=num_negative_points,
            negative_point_offset_ratio=negative_point_offset_ratio,
        )
        candidates.append({"padding_ratio": ratio, "boxes": boxes, "mask": mask})
        masks.append(mask)

    scored_candidates: list[dict[str, object]] = []
    for candidate in candidates:
        other_masks = [other for other in masks if other is not candidate["mask"]]
        stats = score_dynamic_candidate(
            candidate["mask"],
            candidate["boxes"],
            other_masks,
            image_rgb.shape[:2],
            target_mask_box_ratio=target_mask_box_ratio,
            max_outside_box_ratio=max_outside_box_ratio,
        )
        scored = {**candidate, **stats}
        scored_candidates.append(scored)

    base_candidate = min(scored_candidates, key=lambda item: abs(float(item["padding_ratio"]) - base_padding_ratio))
    best_candidate = max(scored_candidates, key=lambda item: float(item["score"]))
    selected = best_candidate if float(best_candidate["score"]) > float(base_candidate["score"]) + switch_margin else base_candidate
    debug = {
        "selected_padding_ratio": float(selected["padding_ratio"]),
        "base_padding_ratio": float(base_candidate["padding_ratio"]),
        "selected_score": float(selected["score"]),
        "base_score": float(base_candidate["score"]),
        "dynamic_candidates": [
            {
                "padding_ratio": float(candidate["padding_ratio"]),
                "score": float(candidate["score"]),
                "stability": float(candidate["stability"]),
                "mask_box_ratio": float(candidate["mask_box_ratio"]),
                "outside_box_ratio": float(candidate["outside_box_ratio"]),
                "component_penalty": float(candidate["component_penalty"]),
                "mask_area": int(np.asarray(candidate["mask"]).sum()),
            }
            for candidate in scored_candidates
        ],
    }
    return np.asarray(selected["mask"], dtype=np.uint8), np.asarray(selected["boxes"], dtype=np.float32), debug


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, boxes: np.ndarray, output_path: Path) -> None:
    overlay = image_rgb.copy()
    overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([255, 40, 40])).astype(np.uint8)
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 220, 60), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.raw_root) / args.dataset / args.split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split not found: {image_dir}")

    out_dir = Path(args.out_root) / args.dataset / args.split
    mask_dir = out_dir / "masks"
    overlay_dir = out_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.yolo_weights) if args.box_source == "yolo" else None
    finetuned_checkpoint = Path(args.finetuned_checkpoint) if args.finetuned_checkpoint else None
    predictor = load_sam(
        Path(args.sam_checkpoint),
        args.sam_model_type,
        args.device,
        backend=args.sam_backend,
        finetuned_checkpoint=finetuned_checkpoint,
    )

    records = []
    dynamic_padding_ratios = parse_float_list(args.dynamic_padding_ratios)
    dynamic_base_padding_ratio = args.dynamic_base_padding_ratio
    if dynamic_base_padding_ratio is None:
        dynamic_base_padding_ratio = args.box_padding_ratio
    image_files = find_image_files(image_dir)
    for image_path in tqdm(image_files, desc=f"infer/{args.dataset}/{args.split}"):
        image_rgb = read_rgb(image_path)
        if args.box_source == "gt":
            boxes = boxes_from_gt_mask(Path(args.raw_root) / args.dataset / f"{args.split}_labels" / f"{image_path.stem}.png")
            scores = np.ones((len(boxes),), dtype=np.float32)
        else:
            result = yolo.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                verbose=False,
                device=args.device,
            )[0]
            boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.empty((0, 4))
            scores = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.empty((0,))
        raw_boxes = boxes.astype(np.float32).reshape(-1, 4)
        dynamic_info: dict[str, object] = {}
        if args.mask_to_prompt_refine:
            boxes = expand_boxes(raw_boxes, image_rgb.shape[:2], args.box_padding_ratio)
            mask, boxes, refine_info = predict_mask_to_prompt_refined_masks(
                predictor,
                image_rgb,
                boxes,
                args.device,
                prompt_mode=args.prompt_mode,
                num_positive_points=args.num_positive_points,
                num_negative_points=args.num_negative_points,
                negative_point_offset_ratio=args.negative_point_offset_ratio,
                refine_box_padding_ratio=args.refine_box_padding_ratio,
                refine_num_positive_points=args.refine_num_positive_points,
                refine_num_negative_points=args.refine_num_negative_points,
                refine_negative_dilate_kernel=args.refine_negative_dilate_kernel,
                refine_iters=args.refine_iters,
                refine_prompt_scheme=args.refine_prompt_scheme,
                use_mask_input=not args.disable_refine_mask_input,
            )
            dynamic_info.update(refine_info)
        elif args.dynamic_correction:
            mask, boxes, dynamic_info = predict_dynamic_sam_masks(
                predictor,
                image_rgb,
                raw_boxes,
                args.device,
                padding_ratios=dynamic_padding_ratios,
                base_padding_ratio=dynamic_base_padding_ratio,
                switch_margin=args.dynamic_switch_margin,
                target_mask_box_ratio=args.dynamic_target_mask_box_ratio,
                max_outside_box_ratio=args.dynamic_max_outside_box_ratio,
                prompt_mode=args.prompt_mode,
                num_positive_points=args.num_positive_points,
                num_negative_points=args.num_negative_points,
                negative_point_offset_ratio=args.negative_point_offset_ratio,
            )
        else:
            boxes = expand_boxes(raw_boxes, image_rgb.shape[:2], args.box_padding_ratio)
            mask = predict_sam_masks(
                predictor,
                image_rgb,
                boxes,
                args.device,
                prompt_mode=args.prompt_mode,
                num_positive_points=args.num_positive_points,
                num_negative_points=args.num_negative_points,
                negative_point_offset_ratio=args.negative_point_offset_ratio,
            )
        if int(mask.sum()) < args.min_mask_area:
            mask = np.zeros_like(mask, dtype=np.uint8)

        output_mask = (mask * 255).astype(np.uint8)
        Image.fromarray(output_mask).save(mask_dir / f"{image_path.stem}.png")
        if args.save_overlays:
            save_overlay(image_rgb, mask, boxes, overlay_dir / f"{image_path.stem}.jpg")

        records.append(
            {
                "image": image_path.name,
                "num_boxes": int(len(boxes)),
                "scores": [float(score) for score in scores],
                "mask_area": int(mask.sum()),
                "sam_backend": args.sam_backend,
                "sam_model_type": args.sam_model_type,
                **dynamic_info,
            }
        )

    (out_dir / "inference_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved masks to: {mask_dir}")


if __name__ == "__main__":
    main()
