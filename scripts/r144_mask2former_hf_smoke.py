#!/usr/bin/env python3
"""R144 Mask2Former feasibility smoke using an isolated HuggingFace dependency path."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny Mask2Former train/infer smoke.")
    parser.add_argument("--hf-deps", default=".tmp/r144_hf_deps")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--limit-train", type=int, default=2)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--encoder-layers", type=int, default=1)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--readout-mode", choices=["class_gated", "mask_only"], default="class_gated")
    parser.add_argument("--output-dir", default="outputs/r144_mask2former_hf_smoke")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--metrics-json", default="outputs/analysis/r144_mask2former_hf_smoke_clean_test_v2_metrics.json")
    parser.add_argument("--train-metrics-json", default="")
    parser.add_argument("--val-metrics-json", default="")
    parser.add_argument("--manifest-json", default="outputs/analysis/r144_mask2former_hf_smoke_manifest.json")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def add_hf_deps(path: str) -> None:
    deps = Path(path)
    if deps.exists():
        sys.path.insert(0, str(deps.resolve()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_image_path(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found for {stem} under {root / dataset / split}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def read_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(image.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def resize_label(label: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(label.astype(np.int32), (size, size), interpolation=cv2.INTER_NEAREST)


def instance_targets(label: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    masks = []
    for value in sorted(int(v) for v in np.unique(label) if int(v) > 0):
        mask = label == value
        if mask.sum() > 0:
            masks.append(torch.from_numpy(mask.astype(np.float32)))
    if not masks:
        masks.append(torch.zeros(label.shape, dtype=torch.float32))
    mask_tensor = torch.stack(masks, dim=0)
    class_labels = torch.zeros(len(masks), dtype=torch.long)
    return mask_tensor, class_labels


def load_sample(raw_root: Path, dataset: str, split: str, label_path: Path, img_size: int) -> dict[str, object]:
    image = resize_image(read_gray(find_image_path(raw_root, dataset, split, label_path.stem)), img_size)
    label_orig = read_label(label_path)
    label = resize_label(label_orig, img_size)
    masks, class_labels = instance_targets(label)
    return {
        "pixel_values": torch.from_numpy(np.repeat(image[None], 3, axis=0).astype(np.float32)),
        "mask_labels": masks,
        "class_labels": class_labels,
        "name": label_path.name,
        "original_shape": label_orig.shape,
        "gt_original": label_orig > 0,
    }


def load_split(raw_root: Path, dataset: str, split: str, img_size: int, limit: int) -> list[dict[str, object]]:
    labels = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
    if limit > 0:
        labels = labels[:limit]
    return [load_sample(raw_root, dataset, split, path, img_size) for path in labels]


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = [key for key in records[0] if key != "image"]
    return {key: float(np.mean([record[key] for record in records])) for key in keys}


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


@torch.no_grad()
def predict_union(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    out_shape: tuple[int, int],
    threshold: float,
    readout_mode: str,
) -> np.ndarray:
    output = model(pixel_values=pixel_values)
    logits = output.masks_queries_logits[0]
    class_logits = output.class_queries_logits[0]
    probs = torch.sigmoid(torch.nn.functional.interpolate(
        logits[:, None],
        size=out_shape,
        mode="bilinear",
        align_corners=False,
    )[:, 0])
    foreground_scores = torch.softmax(class_logits, dim=-1)[:, 0]
    keep = foreground_scores > 0.05 if readout_mode == "class_gated" else torch.ones_like(foreground_scores, dtype=torch.bool)
    if keep.any():
        union_prob = probs[keep].max(dim=0).values
    else:
        union_prob = probs.max(dim=0).values
    return (union_prob.detach().cpu().numpy() >= threshold)


@torch.no_grad()
def evaluate_samples(
    model: torch.nn.Module,
    samples: list[dict[str, object]],
    dataset: str,
    split: str,
    output_dir: Path,
    threshold: float,
    readout_mode: str,
    device: str,
) -> dict[str, object]:
    pred_dir = output_dir / dataset / split / "masks"
    records = []
    for sample in samples:
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        pred = predict_union(model, pixel_values, sample["original_shape"], threshold, readout_mode)
        gt = sample["gt_original"]
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec["image"] = str(sample["name"])
        records.append(rec)
        write_mask(pred_dir / str(sample["name"]), pred)
    return {
        "dataset": dataset,
        "split": split,
        "num_evaluated": len(records),
        "threshold": threshold,
        "readout_mode": readout_mode,
        "mean": mean_metrics(records),
        "per_image": records,
        "prediction_dir": str(pred_dir),
    }


def main() -> None:
    args = parse_args()
    add_hf_deps(args.hf_deps)
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    set_seed(args.seed)
    train_samples = load_split(Path(args.raw_root), args.dataset, args.train_split, args.img_size, args.limit_train)
    val_samples = load_split(Path(args.raw_root), args.dataset, "val", args.img_size, args.limit_val) if args.limit_val > 0 else []
    eval_samples = load_split(Path(args.eval_raw_root), args.eval_dataset, args.eval_split, args.img_size, args.limit_eval)

    config = Mask2FormerConfig(
        num_labels=1,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feature_size=args.hidden_dim,
        mask_feature_size=args.hidden_dim,
        fpn_feature_size=args.hidden_dim,
        use_auxiliary_loss=True,
    )
    model = Mask2FormerForUniversalSegmentation(config).to(args.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_trace = []
    for step in range(1, args.steps + 1):
        sample = train_samples[(step - 1) % len(train_samples)]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(args.device)
        mask_labels = [sample["mask_labels"].to(args.device)]
        class_labels = [sample["class_labels"].to(args.device)]
        opt.zero_grad(set_to_none=True)
        output = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        loss = output.loss
        loss.backward()
        opt.step()
        train_trace.append({"step": step, "image": str(sample["name"]), "loss": float(loss.detach().cpu())})
        print(json.dumps(train_trace[-1]), flush=True)

    model.eval()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "train_trace": train_trace}, checkpoint)

    output_dir = Path(args.output_dir)
    train_eval = evaluate_samples(model, train_samples, args.dataset, args.train_split, output_dir, args.threshold, args.readout_mode, args.device)
    val_eval = evaluate_samples(model, val_samples, args.dataset, "val", output_dir, args.threshold, args.readout_mode, args.device) if val_samples else None
    clean_eval = evaluate_samples(model, eval_samples, args.eval_dataset, args.eval_split, output_dir, args.threshold, args.readout_mode, args.device)

    summary = {
        "status": "smoke_complete",
        "architecture": "hf_mask2former",
        "dataset": args.eval_dataset,
        "split": args.eval_split,
        "num_train": len(train_samples),
        "num_evaluated": clean_eval["num_evaluated"],
        "threshold": args.threshold,
        "readout_mode": args.readout_mode,
        "train_trace": train_trace,
        "train_eval": train_eval,
        "val_eval": val_eval,
        "mean": clean_eval["mean"],
        "per_image": clean_eval["per_image"],
        "clean_test_policy": "clean-test-v2 used only for smoke/final inference, not training or tuning",
    }
    metrics_json = Path(args.metrics_json)
    metrics_json.parent.mkdir(parents=True, exist_ok=True)
    metrics_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.train_metrics_json:
        Path(args.train_metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.train_metrics_json).write_text(json.dumps(train_eval, indent=2), encoding="utf-8")
    if args.val_metrics_json and val_eval:
        Path(args.val_metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.val_metrics_json).write_text(json.dumps(val_eval, indent=2), encoding="utf-8")

    manifest = {
        "status": "ok",
        "hf_deps": args.hf_deps,
        "model_class": "Mask2FormerForUniversalSegmentation",
        "config": {
            "num_queries": args.num_queries,
            "hidden_dim": args.hidden_dim,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
        },
        "metrics_json": str(metrics_json),
        "prediction_dir": str(output_dir / args.eval_dataset / args.eval_split / "masks"),
        "checkpoint": args.checkpoint,
        "train_images": [str(s["name"]) for s in train_samples],
        "val_images": [str(s["name"]) for s in val_samples],
        "eval_images": [str(s["name"]) for s in eval_samples],
    }
    Path(args.manifest_json).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metrics": summary["mean"], "manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
