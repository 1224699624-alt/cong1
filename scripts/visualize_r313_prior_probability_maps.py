#!/usr/bin/env python3
"""Render deterministic R313 prior probability-map audit panels.

This script only visualizes stored arrays and source images/labels. It does not
generate, synthesize, or alter medical image content.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--prior-root", type=Path, default=Path("outputs/priors/r313_clean_relation"))
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path("outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/r313_clean_relation_prior_heatmaps_val94"),
    )
    parser.add_argument("--num-cases", type=int, default=12)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--panel-height", type=int, default=620)
    return parser.parse_args()


def find_image(folder: Path, stem: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = folder / f"{stem}{extension}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            try:
                with Image.open(candidate) as image:
                    image.verify()
                return candidate
            except (OSError, ValueError):
                continue
    raise FileNotFoundError(f"No readable image for {folder / stem}")


def fit(image: Image.Image, width: int, height: int, *, nearest: bool = False) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize(size, method)


def titled(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    header = 64
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((9, 8), title, fill="black", font=font)
    draw.text((9, 33), subtitle, fill=(60, 60, 60), font=font)
    return canvas


def center(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (238, 240, 242))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def normalized_gray(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    lo, hi = np.percentile(array, (1.0, 99.0))
    if hi <= lo:
        return np.uint8(np.clip(array, 0, 255))
    return np.uint8(np.clip((array - lo) * 255.0 / (hi - lo), 0, 255))


def colorize(prior: np.ndarray) -> np.ndarray:
    raw = np.uint8(np.clip(prior, 0.0, 1.0) * 255.0)
    return cv2.cvtColor(cv2.applyColorMap(raw, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def add_probability_legend(rgb: np.ndarray) -> np.ndarray:
    result = rgb.copy()
    height, width = result.shape[:2]
    bar_width = max(12, round(width * 0.025))
    bar_height = max(80, round(height * 0.28))
    x1, y1 = width - bar_width - 14, 14
    gradient = np.linspace(1.0, 0.0, bar_height, dtype=np.float32)[:, None]
    colored = colorize(np.repeat(gradient, bar_width, axis=1))
    result[y1 : y1 + bar_height, x1 : x1 + bar_width] = colored
    cv2.rectangle(result, (x1 - 1, y1 - 1), (x1 + bar_width, y1 + bar_height), (255, 255, 255), 1)
    cv2.putText(result, "1.0", (x1 - 4, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    cv2.putText(result, "0.0", (x1 - 4, y1 + bar_height - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    return result


def heat_overlay(gray: np.ndarray, prior: np.ndarray) -> np.ndarray:
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    heat = colorize(prior).astype(np.float32)
    alpha = (0.08 + 0.70 * np.clip(prior, 0.0, 1.0))[..., None]
    return np.uint8(np.clip((1.0 - alpha) * base + alpha * heat, 0, 255))


def compatibility_overlay(gray: np.ndarray, prior: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    strength = np.clip(prior, 0.0, 1.0)[..., None]
    colors = np.zeros_like(base)
    colors[~foreground] = (40, 220, 90)  # prior in labelled background
    colors[foreground] = (255, 50, 210)  # prior over labelled bone, audit risk
    alpha = 0.72 * strength
    return np.uint8(np.clip((1.0 - alpha) * base + alpha * colors, 0, 255))


def read_min_gap(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.is_file():
        return result
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["image"]).stem
            gap = float(row["gap_px"])
            result[stem] = min(gap, result.get(stem, float("inf")))
    return result


def audit_case(args: argparse.Namespace, stem: str, min_gap: float | None) -> dict[str, object]:
    prior = np.load(args.prior_root / "val" / f"{stem}.npy").astype(np.float32)
    if prior.ndim != 2 or not np.isfinite(prior).all():
        raise ValueError(f"Invalid prior map: {stem}, shape={prior.shape}")
    image_path = find_image(args.dataset_root / "val", stem)
    label_path = args.dataset_root / "val_labels" / f"{stem}.png"
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    image = Image.open(image_path).convert("L")
    label = Image.open(label_path)
    ids = np.asarray(label)
    if ids.ndim != 2 or ids.shape != prior.shape or image.size != (prior.shape[1], prior.shape[0]):
        raise ValueError(f"Unaligned image/label/prior for {stem}")
    foreground = ids > 0
    total_mass = float(prior.sum())
    foreground_mass_fraction = float(prior[foreground].sum() / max(total_mass, 1e-8))
    stats: dict[str, object] = {
        "stem": stem,
        "image": str(image_path),
        "label": str(label_path),
        "prior": str(args.prior_root / "val" / f"{stem}.npy"),
        "min_gap_px": min_gap,
        "prior_mean": float(prior.mean()),
        "prior_p95": float(np.quantile(prior, 0.95)),
        "prior_p99": float(np.quantile(prior, 0.99)),
        "prior_max": float(prior.max()),
        "prior_mass_on_labelled_foreground": foreground_mass_fraction,
    }
    gray = normalized_gray(image)
    heat = add_probability_legend(colorize(prior))
    overlay = heat_overlay(gray, prior)
    compatibility = compatibility_overlay(gray, prior, foreground)
    label_rgb = np.asarray(label.convert("RGB"))
    subtitle = f"p99={stats['prior_p99']:.3f}; max={stats['prior_max']:.3f}"
    risk_subtitle = f"magenta=on labelled bone; mass={foreground_mass_fraction:.3f}"
    tiles = [
        titled(fit(Image.fromarray(gray), args.panel_width, args.panel_height), "Original X-ray", f"case={stem}"),
        titled(fit(Image.fromarray(label_rgb), args.panel_width, args.panel_height, nearest=True), "Instance label", "colors are original, not corrected"),
        titled(fit(Image.fromarray(heat), args.panel_width, args.panel_height), "Raw prior probability", subtitle),
        titled(fit(Image.fromarray(overlay), args.panel_width, args.panel_height), "X-ray + prior heatmap", "blue=low, red=high; raw 0-1 scale"),
        titled(fit(Image.fromarray(compatibility), args.panel_width, args.panel_height), "Prior/label compatibility", risk_subtitle),
    ]
    cell_width = max(tile.width for tile in tiles)
    cell_height = max(tile.height for tile in tiles)
    canvas = Image.new("RGB", (cell_width * len(tiles), cell_height), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(center(tile, cell_width, cell_height), (index * cell_width, 0))
    panel_path = args.output_dir / f"{stem}_prior_heatmap_panel.jpg"
    canvas.save(panel_path, quality=93, subsampling=0)
    stats["panel"] = str(panel_path)
    return stats


def main() -> None:
    args = parse_args()
    if args.num_cases < 1:
        raise ValueError("--num-cases must be positive")
    prior_dir = args.prior_root / "val"
    stems = sorted(path.stem for path in prior_dir.glob("*.npy"))
    if len(stems) != 94:
        raise RuntimeError(f"Expected exactly 94 val prior arrays, found {len(stems)}")
    min_gaps = read_min_gap(args.selection_csv)
    rows: list[dict[str, object]] = []
    for stem in stems:
        prior = np.load(prior_dir / f"{stem}.npy").astype(np.float32)
        ids = np.asarray(Image.open(args.dataset_root / "val_labels" / f"{stem}.png"))
        mass = float(prior.sum())
        rows.append(
            {
                "stem": stem,
                "risk": float(prior[ids > 0].sum() / max(mass, 1e-8)),
                "p99": float(np.quantile(prior, 0.99)),
                "min_gap": min_gaps.get(stem),
            }
        )
    gap_ranked = sorted((row for row in rows if row["min_gap"] is not None), key=lambda row: (row["min_gap"], row["stem"]))
    risk_ranked = sorted(rows, key=lambda row: (-row["risk"], -row["p99"], row["stem"]))
    selected: list[str] = []
    for pool in (gap_ranked[: max(1, args.num_cases // 2)], risk_ranked, sorted(rows, key=lambda row: (-row["p99"], row["stem"]))):
        for row in pool:
            stem = str(row["stem"])
            if stem not in selected:
                selected.append(stem)
            if len(selected) >= args.num_cases:
                break
        if len(selected) >= args.num_cases:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = [audit_case(args, stem, min_gaps.get(stem)) for stem in selected]
    manifest = {
        "experiment": "R313",
        "dataset": "TSRS_RSNA-Epiphysis",
        "split": "original-val filtered to 94 retained cases",
        "selection": "half minimum-gap cases, then highest prior mass on labelled foreground",
        "legend": {
            "heatmap": "TURBO: blue low probability, red high probability, raw scale 0-1",
            "compatibility": "green prior lies on labelled background; magenta prior lies on labelled bone and is an audit risk, not automatically an error",
        },
        "clean_test_used": False,
        "cases": audit,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (args.output_dir / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    print(json.dumps({"output_dir": str(args.output_dir), "panels": len(audit), "clean_test_used": False}, indent=2))


if __name__ == "__main__":
    main()
