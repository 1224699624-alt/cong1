#!/usr/bin/env python3
"""Build an isolated train/val X-ray contrast-normalized Epiphysis variant."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import exposure


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def find_valid_image(folder: Path, stem: str) -> Path:
    for extension in EXTENSIONS:
        path = folder / f"{stem}{extension}"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            return path
        except OSError:
            continue
    raise FileNotFoundError(f"No valid image for {stem} in {folder}")


def image_stats(array: np.ndarray) -> dict[str, float]:
    values = array.astype(np.float32)
    roi = values[values > 1]
    if roi.size == 0:
        roi = values.ravel()
    gradient_y, gradient_x = np.gradient(values)
    gradient = np.hypot(gradient_x, gradient_y)
    return {
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "gradient_p90": float(np.percentile(gradient, 90)),
        "roi_p05": float(np.percentile(roi, 5)),
        "roi_p50": float(np.percentile(roi, 50)),
        "roi_p95": float(np.percentile(roi, 95)),
        "roi_std": float(roi.std()),
    }


def enhance_xray(
    array: np.ndarray,
    target_low: float = 96.0,
    target_mid: float = 120.0,
    target_high: float = 225.0,
    clahe_weight: float = 0.20,
    unsharp_weight: float = 0.15,
) -> np.ndarray:
    """Robust windowing plus restrained local contrast enhancement.

    The transform uses image intensities only and is identical at train and
    inference time. It never reads labels, bone age, sex, or another split.
    """
    source = array.astype(np.float32)
    fit_values = source[source > 1]
    if fit_values.size < max(256, int(source.size * 0.05)):
        fit_values = source.ravel()
    q_low, q_mid, q_high = np.percentile(fit_values, (1.0, 50.0, 99.0))
    if q_high - q_low < 4.0:
        return array.copy()
    q_mid = float(np.clip(q_mid, q_low + 1e-3, q_high - 1e-3))
    mapped = np.interp(
        source,
        [0.0, q_low, q_mid, q_high, 255.0],
        [0.0, target_low, target_mid, target_high, 255.0],
    ).astype(np.float32)
    height, width = mapped.shape
    kernel = (max(32, height // 8), max(32, width // 8))
    local = exposure.equalize_adapthist(mapped.astype(np.uint8), kernel_size=kernel, clip_limit=0.01, nbins=256)
    local = local.astype(np.float32) * 255.0
    combined = (1.0 - clahe_weight) * mapped + clahe_weight * local
    if unsharp_weight > 0:
        blurred = gaussian_filter(combined, sigma=1.2)
        combined = combined + unsharp_weight * (combined - blurred)
    # Preserve truly black acquisition borders instead of turning them gray.
    combined[source <= 1] = 0
    return np.clip(np.rint(combined), 0, 255).astype(np.uint8)


def process_one(task: tuple[str, str, str, str, dict[str, float]]) -> dict:
    split, stem, input_path, output_path, config = task
    source = np.asarray(Image.open(input_path).convert("L"))
    enhanced = enhance_xray(source, **config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(enhanced, mode="L").save(output)
    return {
        "split": split,
        "image": f"{stem}.png",
        "source": input_path,
        "output": str(output),
        "before": image_stats(source),
        "after": image_stats(enhanced),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--output-root", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--splits", default="train,val")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--target-low", type=float, default=96.0)
    p.add_argument("--target-mid", type=float, default=120.0)
    p.add_argument("--target-high", type=float, default=225.0)
    p.add_argument("--clahe-weight", type=float, default=0.20)
    p.add_argument("--unsharp-weight", type=float, default=0.15)
    p.add_argument("--demo-input", type=Path)
    p.add_argument("--demo-output", type=Path)
    args = p.parse_args()
    config = {
        "target_low": args.target_low,
        "target_mid": args.target_mid,
        "target_high": args.target_high,
        "clahe_weight": args.clahe_weight,
        "unsharp_weight": args.unsharp_weight,
    }
    if args.demo_input:
        if not args.demo_output:
            raise ValueError("--demo-output is required with --demo-input")
        source = np.asarray(Image.open(args.demo_input).convert("L"))
        enhanced = enhance_xray(source, **config)
        args.demo_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(enhanced, mode="L").save(args.demo_output)
        print(json.dumps({"before": image_stats(source), "after": image_stats(enhanced), "config": config}, indent=2))
        return
    if args.source_root.name != "TSRS_RSNA-Epiphysis" or "Articular" in str(args.source_root):
        raise RuntimeError("This builder permits TSRS_RSNA-Epiphysis only")
    if "clean-test" in str(args.output_root).lower():
        raise RuntimeError("Development contrast variants cannot target clean-test-v2")
    tasks: list[tuple[str, str, str, str, dict[str, float]]] = []
    expected = {"train": 875, "val": 96}
    for split in [value.strip() for value in args.splits.split(",") if value.strip()]:
        if split not in expected:
            raise ValueError(f"Only original train/val are allowed, got {split}")
        label_dir = args.source_root / f"{split}_labels"
        labels = sorted(label_dir.glob("*.png"))
        if len(labels) != expected[split]:
            raise RuntimeError(f"Expected {expected[split]} {split} labels, found {len(labels)}")
        output_labels = args.output_root / f"{split}_labels"
        output_labels.mkdir(parents=True, exist_ok=True)
        for label in labels:
            image = find_valid_image(args.source_root / split, label.stem)
            tasks.append((split, label.stem, str(image), str(args.output_root / split / f"{label.stem}.png"), config))
            shutil.copy2(label, output_labels / label.name)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rows = list(executor.map(process_one, tasks))
    summary = {
        "variant": "TSRS_RSNA-Epiphysis_contrast_v1",
        "scope": "original train/val images only; labels copied unchanged",
        "clean_test_used": False,
        "config": config,
        "num_images": len(rows),
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in expected},
        "mean_before": {key: float(np.mean([row["before"][key] for row in rows])) for key in rows[0]["before"]},
        "mean_after": {key: float(np.mean([row["after"][key] for row in rows])) for key in rows[0]["after"]},
        "per_image": rows,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "contrast_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("variant", "scope", "num_images", "split_counts", "mean_before", "mean_after")}, indent=2))


if __name__ == "__main__":
    main()
