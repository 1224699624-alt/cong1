#!/usr/bin/env python3
"""Run YOLO detection and MedSAM box-prompt segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="YOLO boxes + MedSAM segmentation inference.")
    parser.add_argument("--dataset", required=True, help="Dataset folder name under data/raw.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-root", default="outputs/predictions")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--medsam-checkpoint", default="checkpoints/medsam_vit_b.pth")
    parser.add_argument("--finetuned-checkpoint", default=None, help="Optional fine-tuned mask decoder checkpoint.")
    parser.add_argument("--sam-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=128)
    parser.add_argument("--box-padding-ratio", type=float, default=0.05)
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


def load_medsam(
    checkpoint: Path,
    model_type: str,
    device: str,
    finetuned_checkpoint: Path | None = None,
) -> SamPredictor:
    if not checkpoint.exists():
        raise FileNotFoundError(f"MedSAM checkpoint not found: {checkpoint}")
    sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    if finetuned_checkpoint is not None:
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
    return SamPredictor(sam_model)


def predict_medsam_masks(
    predictor: SamPredictor,
    image_rgb: np.ndarray,
    boxes_xyxy: np.ndarray,
    device: str,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if boxes_xyxy.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    predictor.set_image(image_rgb)
    gbc_adapter = getattr(predictor.model, "gbc_adapter", None)
    if gbc_adapter is not None:
        with torch.no_grad():
            predictor.features = gbc_adapter(predictor.features)
    boxes_torch = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_torch, image_rgb.shape[:2])
    with torch.no_grad():
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    merged = masks[:, 0].detach().cpu().numpy().any(axis=0).astype(np.uint8)
    return merged


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
    predictor = load_medsam(
        Path(args.medsam_checkpoint),
        args.sam_model_type,
        args.device,
        finetuned_checkpoint=finetuned_checkpoint,
    )

    records = []
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
        boxes = expand_boxes(boxes, image_rgb.shape[:2], args.box_padding_ratio)
        mask = predict_medsam_masks(predictor, image_rgb, boxes, args.device)
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
            }
        )

    (out_dir / "inference_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved masks to: {mask_dir}")


if __name__ == "__main__":
    main()
