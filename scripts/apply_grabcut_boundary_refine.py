#!/usr/bin/env python3
"""Apply image-driven GrabCut refinement inside an anchor boundary band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boundary-limited GrabCut refinement for anchor masks.")
    parser.add_argument("--tune-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--tune-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--apply-anchor-exp", default=None)
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--band-radii", default="2,4,6")
    parser.add_argument("--sure-fg-erode", default="1,2,3")
    parser.add_argument("--sure-bg-dilate", default="4,8,12")
    parser.add_argument("--max-edit-fracs", default="0.001,0.0025,0.005,0.01")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--output-exp", default="r084_grabcut_boundary_refine")
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def find_image(raw_root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = raw_root / dataset / split / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"image not found: {raw_root}/{dataset}/{split}/{stem}")


def read_image(path: Path) -> np.ndarray:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def find_anchor(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None = None) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"anchor not found: exp={exp} dataset={dataset} split={split} name={name}")


def resize_like(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if mask.shape == gt.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0)


def add_structure_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {k: float(np.mean([float(r[k]) for r in records])) for k in keys}


def limit_edits(anchor: np.ndarray, pred: np.ndarray, score: np.ndarray, max_edit_frac: float) -> np.ndarray:
    changed = pred != anchor
    max_edits = int(round(max_edit_frac * float(anchor.size)))
    if max_edits <= 0 or int(changed.sum()) <= max_edits:
        return pred
    idx = np.flatnonzero(changed.ravel())
    keep = idx[np.argsort(score.ravel()[idx])[-max_edits:]]
    out = anchor.copy().ravel()
    out[keep] = pred.ravel()[keep]
    return out.reshape(anchor.shape)


def refine_one(image: np.ndarray, anchor: np.ndarray, band_radius: int, erode: int, dilate: int, max_edit_frac: float, iterations: int) -> np.ndarray:
    if anchor.sum() == 0:
        return anchor
    gc_mask = np.full(anchor.shape, cv2.GC_BGD, dtype=np.uint8)
    fg_kernel = np.ones((2 * erode + 1, 2 * erode + 1), dtype=np.uint8)
    bg_kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=np.uint8)
    sure_fg = cv2.erode(anchor.astype(np.uint8), fg_kernel, iterations=1).astype(bool) if erode > 0 else anchor.copy()
    expanded = cv2.dilate(anchor.astype(np.uint8), bg_kernel, iterations=1).astype(bool)
    band = boundary_band(anchor, band_radius)
    gc_mask[expanded] = cv2.GC_PR_BGD
    gc_mask[anchor | band] = cv2.GC_PR_FGD
    gc_mask[sure_fg] = cv2.GC_FGD
    gc_mask[~expanded] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, gc_mask, None, bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return anchor
    raw = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    pred = anchor.copy()
    pred[band] = raw[band]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    score = cv2.GaussianBlur(cv2.magnitude(grad_x, grad_y), (0, 0), 1.0)
    return limit_edits(anchor, pred, score, max_edit_frac)


def collect(raw_root: Path, dataset: str, split: str, roots: list[Path], anchor_exp: str, source_dataset: str | None) -> list[dict[str, object]]:
    label_dir = raw_root / dataset / f"{split}_labels"
    items = []
    for label_path in sorted(label_dir.glob("*.png")):
        gt = read_mask(label_path)
        image = read_image(find_image(raw_root, dataset, split, label_path.stem))
        anchor = resize_like(read_mask(find_anchor(roots, anchor_exp, dataset, split, label_path.name, source_dataset)), gt)
        items.append({"name": label_path.name, "image": image, "gt": gt, "anchor": anchor})
    return items


def evaluate_items(items: list[dict[str, object]], cfg: dict[str, float], boundary_kernel: int, write_dir: Path | None = None) -> tuple[dict[str, float], list[dict[str, float]]]:
    records = []
    for item in tqdm(items, desc=f"eval/grabcut/r{int(cfg['band_radius'])}", leave=False):
        pred = refine_one(
            item["image"],  # type: ignore[arg-type]
            item["anchor"],  # type: ignore[arg-type]
            int(cfg["band_radius"]),
            int(cfg["erode"]),
            int(cfg["dilate"]),
            float(cfg["max_edit_frac"]),
            int(cfg["iterations"]),
        )
        gt = item["gt"]  # type: ignore[assignment]
        if write_dir is not None:
            write_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred.astype(np.uint8) * 255).save(write_dir / str(item["name"]))
        rec = compute_metrics(pred, gt, boundary_kernel)
        add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
    return mean_metrics(records), records


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    tune_items = collect(Path(args.tune_raw_root), args.tune_dataset, args.tune_split, roots, args.anchor_exp, None)
    grid = []
    for band_radius in parse_ints(args.band_radii):
        for erode in parse_ints(args.sure_fg_erode):
            for dilate in parse_ints(args.sure_bg_dilate):
                for max_edit_frac in parse_floats(args.max_edit_fracs):
                    cfg = {
                        "band_radius": float(band_radius),
                        "erode": float(erode),
                        "dilate": float(dilate),
                        "max_edit_frac": float(max_edit_frac),
                        "iterations": float(args.iterations),
                    }
                    mean, _ = evaluate_items(tune_items, cfg, args.boundary_kernel)
                    grid.append({"cfg": cfg, "mean": mean})
    best = max(grid, key=lambda row: row["mean"]["dice"])

    apply_anchor = args.apply_anchor_exp or args.anchor_exp
    apply_items = collect(Path(args.apply_raw_root), args.apply_dataset, args.apply_split, roots, apply_anchor, args.source_apply_dataset)
    write_dir = Path(args.pred_root) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    apply_mean, apply_records = evaluate_items(apply_items, best["cfg"], args.boundary_kernel, write_dir)
    output = {
        "tune_dataset": args.tune_dataset,
        "apply_dataset": args.apply_dataset,
        "anchor_exp": args.anchor_exp,
        "apply_anchor_exp": apply_anchor,
        "best": best,
        "apply_mean": apply_mean,
        "apply_per_image": apply_records,
        "grid": grid,
        "output_exp": args.output_exp,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "apply_mean": apply_mean, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
