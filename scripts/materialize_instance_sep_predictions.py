#!/usr/bin/env python3
"""Materialize R068 predictions for arbitrary dataset splits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize instance-separation segmenter masks.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r068_instance_separation_segmenter/best.pt")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--output-exp", default="r068_instance_separation_segmenter")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sep-suppress-weight", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_r068_module():
    script = Path(__file__).with_name("train_instance_separation_segmenter.py")
    spec = importlib.util.spec_from_file_location("train_instance_separation_segmenter", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_image_path(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found for {stem}: {root}/{dataset}/{split}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def main() -> None:
    args = parse_args()
    module = load_r068_module()
    state = torch.load(args.checkpoint, map_location=args.device)
    ckpt_args = state.get("args", {})
    base = int(ckpt_args.get("base_channels", args.base_channels))
    threshold = float(args.threshold if args.threshold is not None else state.get("threshold", 0.7))
    suppress = float(args.sep_suppress_weight if args.sep_suppress_weight is not None else state.get("sep_suppress_weight", 0.2))
    model = module.InstanceSeparationUNet(base).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    counts = {}
    with torch.no_grad():
        for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
            label_dir = Path(args.raw_root) / args.dataset / f"{split}_labels"
            out_dir = Path(args.pred_root) / args.output_exp / args.dataset / split / "masks"
            out_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for label_path in tqdm(sorted(label_dir.glob("*.png")), desc=f"materialize/{args.dataset}/{split}"):
                image = read_gray(find_image_path(Path(args.raw_root), args.dataset, split, label_path.stem))
                shape = image.shape
                resized = cv2.resize(image, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
                tensor = torch.from_numpy(resized[None, None].astype(np.float32)).to(args.device)
                prob = module.apply_sep_suppression(model(tensor), suppress).cpu().numpy()[0, 0]
                pred = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) >= threshold
                write_mask(out_dir / label_path.name, pred)
                n += 1
            counts[split] = n
    print(json.dumps({"output_exp": args.output_exp, "dataset": args.dataset, "threshold": threshold, "sep_suppress_weight": suppress, "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
