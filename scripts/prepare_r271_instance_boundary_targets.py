"""Prepare isolated instance-aware targets from the palette labels.

The raw TSRS_RSNA-Epiphysis labels are indexed PNGs (mode ``P``), not binary
masks.  Pixel values are per-image instance IDs and the visible colors come
from the PNG palette.  This utility preserves the raw dataset and writes an
isolated target variant containing semantic foreground, inter-instance
boundary, normalized per-instance distance, center heatmap, and the original
instance IDs.  These are training targets; they are not inference probability
maps and never use clean-test-v2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def read_ids(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.asarray(image)
    if image.mode == "P" and array.ndim == 2:
        return array.astype(np.int32)
    if array.ndim == 2:
        return array.astype(np.int32)
    raise ValueError(f"expected indexed/grayscale label, got mode={image.mode}, shape={array.shape}: {path}")


def instance_boundary(ids: np.ndarray, radius: int) -> np.ndarray:
    """Return a band around boundaries between two positive instance IDs."""
    boundary = np.zeros(ids.shape, dtype=bool)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        shifted = np.zeros_like(ids)
        ys = slice(max(0, dy), min(ids.shape[0], ids.shape[0] + dy))
        xs = slice(max(0, dx), min(ids.shape[1], ids.shape[1] + dx))
        src_y = slice(max(0, -dy), min(ids.shape[0], ids.shape[0] - dy))
        src_x = slice(max(0, -dx), min(ids.shape[1], ids.shape[1] - dx))
        shifted[ys, xs] = ids[src_y, src_x]
        boundary |= (ids > 0) & (shifted > 0) & (ids != shifted)
    if radius > 0:
        boundary = ndimage.binary_dilation(boundary, iterations=radius)
    return boundary


def instance_voronoi_seam(ids: np.ndarray, max_gap_radius: int) -> np.ndarray:
    """Centerlines in background where the nearest positive instance changes."""
    background = ids == 0
    distance, nearest_index = ndimage.distance_transform_edt(background, return_indices=True)
    nearest_id = ids[nearest_index[0], nearest_index[1]]
    seam = np.zeros(ids.shape, dtype=bool)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        shifted = np.zeros_like(nearest_id)
        ys = slice(max(0, dy), min(ids.shape[0], ids.shape[0] + dy))
        xs = slice(max(0, dx), min(ids.shape[1], ids.shape[1] + dx))
        src_y = slice(max(0, -dy), min(ids.shape[0], ids.shape[0] - dy))
        src_x = slice(max(0, -dx), min(ids.shape[1], ids.shape[1] - dx))
        shifted[ys, xs] = nearest_id[src_y, src_x]
        seam |= background & (nearest_id > 0) & (shifted > 0) & (nearest_id != shifted)
    seam &= distance <= float(max_gap_radius)
    return ndimage.binary_dilation(seam, iterations=1)


def soft_seam_target(seam: np.ndarray, sigma: float) -> np.ndarray:
    if not seam.any():
        return np.zeros(seam.shape, dtype=np.float32)
    d = ndimage.distance_transform_edt(~seam)
    return np.exp(-(d * d) / (2.0 * sigma * sigma)).astype(np.float32)


def distance_and_center(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.zeros(ids.shape, dtype=np.float32)
    center = np.zeros(ids.shape, dtype=np.float32)
    for label_id in np.unique(ids):
        if label_id <= 0:
            continue
        component = ids == label_id
        if int(component.sum()) < 2:
            continue
        dist = ndimage.distance_transform_edt(component).astype(np.float32)
        scale = max(float(dist.max()), 1.0)
        distance[component] = np.maximum(distance[component], dist[component] / scale)
        ys, xs = np.where(component)
        cy, cx = float(ys.mean()), float(xs.mean())
        sigma = max(2.0, 0.18 * np.sqrt(float(component.sum())))
        pad = int(np.ceil(3.0 * sigma))
        y0, y1 = max(0, int(ys.min()) - pad), min(ids.shape[0], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(ids.shape[1], int(xs.max()) + pad + 1)
        gy, gx = np.ogrid[y0:y1, x0:x1]
        heat = np.exp(-((gy - cy) ** 2 + (gx - cx) ** 2) / (2.0 * sigma * sigma))
        local = component[y0:y1, x0:x1]
        center[y0:y1, x0:x1] = np.maximum(center[y0:y1, x0:x1], (heat * local).astype(np.float32))
    return distance, center


def write_u8(array: np.ndarray, path: Path) -> None:
    Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255)).save(path)


def process_split(dataset_root: Path, split: str, output_root: Path, radius: int,
                  gap_radius: int, seam_sigma: float, limit: int,
                  include_distance_center: bool) -> dict:
    label_dir = dataset_root / f"{split}_labels"
    out = output_root / split
    for sub in ("instance_id", "semantic", "boundary", "seam", "seam_probability", "distance", "center"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    paths = sorted(label_dir.glob("*.png"))
    if limit:
        paths = paths[:limit]
    rows = []
    for label_path in paths:
        ids = read_ids(label_path)
        semantic = ids > 0
        boundary = instance_boundary(ids, radius)
        seam = instance_voronoi_seam(ids, gap_radius) | boundary
        seam_probability = soft_seam_target(seam, seam_sigma)
        distance, center = distance_and_center(ids) if include_distance_center else (None, None)
        # Keep the exact integer IDs in a lossless numpy file; a PNG palette
        # copy is also written for convenient visual inspection.
        np.save(out / "instance_id" / f"{label_path.stem}.npy", ids.astype(np.int32))
        Image.fromarray(ids.astype(np.uint8), mode="L").save(out / "instance_id" / label_path.name)
        write_u8(semantic.astype(np.float32), out / "semantic" / label_path.name)
        write_u8(boundary.astype(np.float32), out / "boundary" / label_path.name)
        write_u8(seam.astype(np.float32), out / "seam" / label_path.name)
        write_u8(seam_probability, out / "seam_probability" / label_path.name)
        if include_distance_center:
            np.save(out / "distance" / f"{label_path.stem}.npy", distance)
            np.save(out / "center" / f"{label_path.stem}.npy", center)
        rows.append({"image": label_path.name, "height": int(ids.shape[0]), "width": int(ids.shape[1]),
                     "num_instances": int((np.unique(ids) > 0).sum()),
                     "foreground_pixels": int(semantic.sum()), "boundary_pixels": int(boundary.sum()),
                     "seam_pixels": int(seam.sum()), "boundary_radius": radius,
                     "max_gap_radius": gap_radius})
    return {"split": split, "images": len(rows), "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/targets/r271_instance_boundary"))
    p.add_argument("--boundary-radius", type=int, default=1)
    p.add_argument("--max-gap-radius", type=int, default=12)
    p.add_argument("--seam-sigma", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--include-distance-center", action="store_true",
                   help="also write per-instance distance/center arrays; slower and larger")
    args = p.parse_args()
    joined = " ".join(map(str, vars(args).values())).lower()
    if "articular" in joined or "clean-test" in joined:
        raise RuntimeError("R271 is restricted to original TSRS_RSNA-Epiphysis train/val")
    args.output_root.mkdir(parents=True, exist_ok=True)
    train = process_split(args.dataset_root, "train", args.output_root, args.boundary_radius,
                          args.max_gap_radius, args.seam_sigma, args.limit, args.include_distance_center)
    val = process_split(args.dataset_root, "val", args.output_root, args.boundary_radius,
                        args.max_gap_radius, args.seam_sigma, args.limit, args.include_distance_center)
    result = {"run_id": "R271_instance_boundary_targets", "dataset": "TSRS_RSNA-Epiphysis",
              "source_dataset_unchanged": True, "clean_test_used": False,
              "label_encoding": "indexed_palette_png_instance_ids", "boundary_radius": args.boundary_radius,
              "max_gap_radius": args.max_gap_radius, "seam_sigma": args.seam_sigma,
              "distance_center_written": args.include_distance_center,
              "train": {"images": train["images"], "mean_instances": float(np.mean([r["num_instances"] for r in train["rows"]])) if train["rows"] else 0.0},
              "val": {"images": val["images"], "mean_instances": float(np.mean([r["num_instances"] for r in val["rows"]])) if val["rows"] else 0.0}}
    (args.output_root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
