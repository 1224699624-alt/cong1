#!/usr/bin/env python3
"""Build an isolated manual label-cleaning package for TSRS epiphysis.

The raw dataset is read-only.  Each review panel contains the original X-ray,
the original indexed/palette instance label, and a color overlay.  A copied
working label is provided for manual correction, preserving the PNG palette.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/manual_review/tsrs_epiphysis_label_cleaning_v1"),
    )
    parser.add_argument("--panel-width", type=int, default=600)
    parser.add_argument("--panel-height", type=int, default=900)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--limit", type=int, default=0, help="per-split limit; 0 means all")
    return parser.parse_args()


def find_valid_image(folder: Path, stem: str) -> Path:
    candidates = [folder / f"{stem}{ext}" for ext in IMAGE_EXTENSIONS]
    candidates += [p for p in folder.glob(f"{stem}.*") if p not in candidates]
    failures: list[str] = []
    for path in candidates:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            return path
        except (OSError, ValueError) as exc:
            failures.append(f"{path.name}: {exc}")
    raise FileNotFoundError(f"no readable image for {folder / stem}; failures={failures}")


def fit(image: Image.Image, max_width: int, max_height: int, *, nearest: bool = False) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return image.resize(size, method) if size != image.size else image.copy()


def titled(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    header = 58
    canvas = Image.new("RGB", (image.width, image.height + header), "white")
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 8), title, fill="black", font=font)
    draw.text((10, 31), subtitle, fill=(65, 65, 65), font=font)
    return canvas


def center_on_canvas(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (238, 240, 242))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def label_ids(label_path: Path) -> tuple[Image.Image, np.ndarray]:
    label = Image.open(label_path)
    ids = np.asarray(label)
    if ids.ndim != 2:
        raise ValueError(f"expected indexed 2-D PNG label, got mode={label.mode}, shape={ids.shape}: {label_path}")
    return label, ids.astype(np.int32)


def overlay(image: Image.Image, label_rgb: Image.Image, ids: np.ndarray) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = np.asarray(label_rgb.resize(image.size, Image.Resampling.NEAREST), dtype=np.float32)
    mask = np.asarray(Image.fromarray((ids > 0).astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)) > 0
    result = base.copy()
    result[mask] = 0.58 * base[mask] + 0.42 * colors[mask]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def connected_components(mask: np.ndarray) -> int:
    """Count 8-connected components."""
    return int(ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))[1])


def label_diagnostics(ids: np.ndarray) -> dict[str, int | float | bool | str]:
    positive_ids = [int(value) for value in np.unique(ids) if value > 0]
    expected = list(range(1, max(positive_ids, default=0) + 1))
    disconnected = 0
    tiny = 0
    min_area = max(20, round(ids.size * 0.00002))
    for instance_id in positive_ids:
        instance = ids == instance_id
        tiny += int(instance.sum() < min_area)
        disconnected += int(connected_components(instance) > 1)
    flags = []
    if positive_ids != expected:
        flags.append("noncontiguous_ids")
    if disconnected:
        flags.append("disconnected_same_id")
    if tiny:
        flags.append("tiny_instance")
    return {
        "num_instances": len(positive_ids),
        "max_instance_id": max(positive_ids, default=0),
        "ids_contiguous": positive_ids == expected,
        "disconnected_instance_count": disconnected,
        "tiny_instance_count": tiny,
        "foreground_fraction": float((ids > 0).mean()),
        "automatic_flags": ";".join(flags),
    }


def make_panel(image: Image.Image, label: Image.Image, ids: np.ndarray, width: int, height: int,
               case_id: str) -> Image.Image:
    image_rgb = image.convert("RGB")
    label_rgb = label.convert("RGB")
    overlay_rgb = overlay(image_rgb, label_rgb, ids)
    panels = [
        titled(fit(image_rgb, width, height), "Original X-ray", f"case={case_id}; size={image.size}"),
        titled(fit(label_rgb, width, height, nearest=True), "Indexed instance label", "different colors = different instance IDs"),
        titled(fit(overlay_rgb, width, height), "X-ray + label overlay", "check missed bone, leakage, merge and boundary offset"),
    ]
    cell_width = max(panel.width for panel in panels)
    cell_height = max(panel.height for panel in panels)
    cells = [center_on_canvas(panel, cell_width, cell_height) for panel in panels]
    canvas = Image.new("RGB", (cell_width * len(cells), cell_height), "white")
    for index, cell in enumerate(cells):
        canvas.paste(cell, (index * cell_width, 0))
    return canvas


def write_readme(path: Path, counts: dict[str, int]) -> None:
    path.write_text(
        "\n".join([
            "# TSRS_RSNA-Epiphysis manual label-cleaning package",
            "",
            "The raw dataset was not modified. Review each JPEG under `panels/<split>/`.",
            "Edit only the copied indexed PNG under `working_labels/<split>/`; preserve image size,",
            "instance IDs and palette. Record the decision in `review_worklist.csv`.",
            "",
            "Decision suggestions: `ok`, `corrected`, `exclude`, `uncertain_second_review`.",
            "Common checks: missing ossification center, adjacent bones sharing one ID, one bone split",
            "across multiple IDs, foreground leakage across a seam, boundary offset, and tiny artifacts.",
            "",
            "Test labels are review-only. Do not use test review decisions for model selection or tuning.",
            "Any cleaned training variant must be created later in a separate `data/raw_variants/` folder.",
            "",
            f"Counts: train={counts.get('train', 0)}, val={counts.get('val', 0)}, test={counts.get('test', 0)}.",
        ]) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis":
        raise RuntimeError("this package is restricted to TSRS_RSNA-Epiphysis")
    if "articular" in str(args.dataset_root).lower() or "clean-test" in str(args.dataset_root).lower():
        raise RuntimeError("Articular-Surface and clean-test variants are excluded")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for split in SPLITS:
        label_paths = sorted((args.dataset_root / f"{split}_labels").glob("*.png"))
        if args.limit:
            label_paths = label_paths[:args.limit]
        panel_dir = args.output_root / "panels" / split
        working_dir = args.output_root / "working_labels" / split
        panel_dir.mkdir(parents=True, exist_ok=True)
        working_dir.mkdir(parents=True, exist_ok=True)
        for label_path in label_paths:
            image_path = find_valid_image(args.dataset_root / split, label_path.stem)
            with Image.open(image_path) as source_image:
                image = source_image.copy()
            label, ids = label_ids(label_path)
            diagnostics = label_diagnostics(ids)
            panel = make_panel(image, label, ids, args.panel_width, args.panel_height, label_path.stem)
            panel_path = panel_dir / f"{label_path.stem}.jpg"
            panel.save(panel_path, quality=args.jpeg_quality, subsampling=0, optimize=True)
            working_label = working_dir / label_path.name
            shutil.copy2(label_path, working_label)
            rows.append({
                "split": split,
                "case_id": label_path.stem,
                "original_image": str(image_path.resolve()),
                "original_label": str(label_path.resolve()),
                "working_label": str(working_label.resolve()),
                "panel": str(panel_path.resolve()),
                "image_width": image.width,
                "image_height": image.height,
                "label_width": label.width,
                "label_height": label.height,
                "size_match": image.size == label.size,
                **diagnostics,
                "allowed_use": "review_only_no_tuning" if split == "test" else "manual_cleaning_candidate",
                "review_decision": "",
                "review_notes": "",
            })
            label.close()
        counts[split] = len(label_paths)
    fieldnames = list(rows[0]) if rows else []
    with (args.output_root / "review_worklist.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "dataset": "TSRS_RSNA-Epiphysis",
        "source_dataset_unchanged": True,
        "articular_surface_used": False,
        "clean_test_v2_used": False,
        "test_review_only": True,
        "counts": counts,
        "total_cases": len(rows),
        "panel_layout": ["original_xray", "indexed_instance_label", "color_overlay"],
        "working_labels_are_copies": True,
        "output_root": str(args.output_root.resolve()),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(args.output_root / "README.md", counts)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
