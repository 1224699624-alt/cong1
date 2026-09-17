#!/usr/bin/env python3
"""Create side-by-side comparison figures for key samples across experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create side-by-side comparison panels for selected samples.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return np.asarray(mask) > 0


def overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float = 0.35) -> Image.Image:
    image_np = np.asarray(image).copy()
    image_np[mask] = ((1.0 - alpha) * image_np[mask] + alpha * np.array([255, 40, 40])).astype(np.uint8)
    return Image.fromarray(image_np)


def add_title(image: Image.Image, title: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 30), (255, 255, 255))
    canvas.paste(image, (0, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(0, 0, 0))
    return canvas


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path.cwd()
    sample_names = ["1667", "2982", "3830", "1468", "1784", "3040"]
    experiments = [
        ("Original", None),
        ("Old Refine", root / "outputs/ablations/zero_shot_mask_to_prompt_refine_pad003_neg2_k5/TSRS_RSNA-Epiphysis/test/masks"),
        ("Iter3 Refine", root / "outputs/ablations/zero_shot_mask_to_prompt_refine_iter3_pad003_neg2_k5/TSRS_RSNA-Epiphysis/test/masks"),
        ("Old Error", root / "outputs/ablations/error_refiner_unet512_b32/TSRS_RSNA-Epiphysis/test/masks"),
        ("Iter3 Error", root / "outputs/ablations/error_refiner_iter3_unet512_b32/TSRS_RSNA-Epiphysis/test/masks"),
    ]

    image_dir = root / args.raw_root / args.dataset / args.split
    for stem in sample_names:
        image_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            continue

        original = load_image(image_path)
        panels = [add_title(original, "Original")]
        for title, mask_dir in experiments[1:]:
            mask_path = mask_dir / f"{stem}.png"
            if mask_path.exists():
                mask = load_mask(mask_path, original.size)
                panel = add_title(overlay_mask(original, mask), title)
            else:
                panel = add_title(original, f"{title} (missing)")
            panels.append(panel)

        total_width = sum(panel.width for panel in panels)
        max_height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (total_width, max_height), (255, 255, 255))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        canvas.save(output_dir / f"{stem}_comparison.jpg", quality=95)


if __name__ == "__main__":
    main()
