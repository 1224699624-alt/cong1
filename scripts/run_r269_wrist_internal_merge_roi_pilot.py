"""R269 wrist-only internal merge-risk ROI pilot.

Candidate generation is GT-free: it uses an R110 anchor mask and the X-ray.
Original-val instance labels are opened only after proposal generation for
coverage/risk auditing.  The script never writes edited masks and refuses
clean-test/articular inputs.

Unlike R225's distance-peak Voronoi heuristic, this pilot combines four local
signals inside a single connected component: two-sided distance peaks,
distance-transform saddle depth, a narrow neck, and an X-ray intensity valley.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def read_instance(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.int32)


def read_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if image.shape != shape:
        image = np.asarray(Image.fromarray(image).resize(shape[::-1], Image.Resampling.BILINEAR))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)
    return clahe.astype(np.float32) / 255.0


def find_readable_image(directory: Path, stem: str, shape: tuple[int, int]) -> tuple[Path, np.ndarray] | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        path = directory / f"{stem}{ext}"
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            return path, read_image(path, shape)
        except (OSError, ValueError):
            continue
    return None


def bbox(mask: np.ndarray, pad: int = 0) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    return (slice(max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)),
            slice(max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)))


def greedy_peaks(dist: np.ndarray, component: np.ndarray, min_spacing: int,
                 max_peaks: int) -> list[tuple[int, int, float]]:
    if not component.any():
        return []
    values = dist[component]
    threshold = max(2.0, float(np.percentile(values, 65)))
    local = (dist == ndimage.maximum_filter(dist, size=max(3, 2 * min_spacing + 1)))
    ys, xs = np.where(component & local & (dist >= threshold))
    order = np.argsort(dist[ys, xs])[::-1]
    peaks: list[tuple[int, int, float]] = []
    for idx in order:
        y, x, score = int(ys[idx]), int(xs[idx]), float(dist[ys[idx], xs[idx]])
        if all((y - py) ** 2 + (x - px) ** 2 >= min_spacing ** 2 for py, px, _ in peaks):
            peaks.append((y, x, score))
        if len(peaks) >= max_peaks:
            break
    return peaks


def watershed_basins(component: np.ndarray, dist: np.ndarray, image: np.ndarray,
                     peaks: list[tuple[int, int, float]]) -> np.ndarray:
    markers = np.zeros(component.shape, dtype=np.int32)
    for idx, (y, x, _score) in enumerate(peaks, 1):
        markers[y, x] = idx
    # Low relief near thick distance peaks, high relief at thin/dark bridges.
    dnorm = dist / max(float(dist.max()), 1e-6)
    local_mean = ndimage.gaussian_filter(image, sigma=5.0)
    darkness = np.clip(local_mean - image, 0.0, 1.0)
    relief = np.clip(0.78 * (1.0 - dnorm) + 0.22 * darkness, 0.0, 1.0)
    labels = ndimage.watershed_ift(np.rint(relief * 255).astype(np.uint8), markers)
    labels[~component] = 0
    return labels


def pair_boundary(labels: np.ndarray, a: int, b: int, component: np.ndarray) -> np.ndarray:
    aa, bb = labels == a, labels == b
    boundary = (ndimage.binary_dilation(aa, iterations=2) &
                ndimage.binary_dilation(bb, iterations=2) & component)
    return ndimage.binary_dilation(boundary, iterations=1) & component


def robust_mean(values: np.ndarray, q: float = 0.5) -> float:
    if values.size == 0:
        return 0.0
    cutoff = np.quantile(values, q)
    selected = values[values <= cutoff]
    return float(selected.mean() if selected.size else values.mean())


def candidate_from_pair(component: np.ndarray, labels: np.ndarray, dist: np.ndarray,
                        image: np.ndarray, peaks: list[tuple[int, int, float]],
                        a: int, b: int) -> dict | None:
    seam = pair_boundary(labels, a, b, component)
    if seam.sum() < 3:
        return None
    pa, pb = peaks[a - 1], peaks[b - 1]
    peak_min = max(1e-6, min(pa[2], pb[2]))
    saddle_level = robust_mean(dist[seam], 0.5)
    saddle_score = float(np.clip((peak_min - saddle_level) / peak_min, 0.0, 1.0))
    neck_score = float(np.clip(1.0 - saddle_level / peak_min, 0.0, 1.0))

    support_radius = max(2, int(round(0.45 * peak_min)))
    yy, xx = np.ogrid[:component.shape[0], :component.shape[1]]
    support_a = (yy - pa[0]) ** 2 + (xx - pa[1]) ** 2 <= support_radius ** 2
    support_b = (yy - pb[0]) ** 2 + (xx - pb[1]) ** 2 <= support_radius ** 2
    side_level = min(float(image[support_a].mean()), float(image[support_b].mean()))
    seam_level = robust_mean(image[seam], 0.6)
    image_scale = float(np.percentile(image, 99) - np.percentile(image, 1) + 1e-6)
    gray_valley = float(np.clip((side_level - seam_level) / image_scale, 0.0, 1.0))

    area_a, area_b = int((labels == a).sum()), int((labels == b).sum())
    balance = float(min(area_a, area_b) / max(area_a, area_b, 1))
    two_side_support = float(np.clip(min(pa[2], pb[2]) / 8.0, 0.0, 1.0) * np.sqrt(balance))
    score = float(0.30 * neck_score + 0.30 * saddle_score +
                  0.25 * gray_valley + 0.15 * two_side_support)
    roi = ndimage.binary_dilation(seam, iterations=max(4, int(round(0.35 * peak_min))))
    roi &= ndimage.binary_dilation(component, iterations=3)
    box = bbox(roi)
    if box is None:
        return None
    sy, sx = box
    return {
        "pair_a": a, "pair_b": b, "peak_a_y": pa[0], "peak_a_x": pa[1],
        "peak_b_y": pb[0], "peak_b_x": pb[1], "peak_min": peak_min,
        "neck_score": neck_score, "saddle_score": saddle_score,
        "gray_valley_score": gray_valley, "two_side_support": two_side_support,
        "basin_balance": balance, "score": score, "seam": seam, "roi": roi,
        "roi_xyxy": [sx.start, sy.start, sx.stop, sy.stop],
    }


def propose(image: np.ndarray, anchor: np.ndarray, v_min: float, v_max: float,
            max_peaks: int, max_candidates: int) -> tuple[list[dict], dict]:
    h, w = anchor.shape
    band = np.zeros_like(anchor)
    band[int(round(v_min * h)):min(h, int(round(v_max * h)))] = True
    labels, n = ndimage.label(anchor, structure=np.ones((3, 3), dtype=np.uint8))
    candidates, inspected = [], 0
    for component_id in range(1, n + 1):
        full = labels == component_id
        wrist_area = int((full & band).sum())
        if wrist_area < 80:
            continue
        local_component = full & ndimage.binary_dilation(band, iterations=max(3, h // 100))
        section = bbox(local_component, pad=max(8, int(round(0.015 * max(h, w)))))
        if section is None:
            continue
        comp = local_component[section]
        dist = ndimage.distance_transform_edt(comp)
        spacing = max(6, int(round(0.025 * min(h, w))))
        peaks = greedy_peaks(dist, comp, spacing, max_peaks)
        if len(peaks) < 2:
            continue
        inspected += 1
        basins = watershed_basins(comp, dist, image[section], peaks)
        for a in range(1, len(peaks) + 1):
            for b in range(a + 1, len(peaks) + 1):
                item = candidate_from_pair(comp, basins, dist, image[section], peaks, a, b)
                if item is None:
                    continue
                # Lift local masks and coordinates back to the full image.
                full_seam = np.zeros_like(anchor); full_roi = np.zeros_like(anchor)
                full_seam[section] = item.pop("seam"); full_roi[section] = item.pop("roi")
                yoff, xoff = section[0].start, section[1].start
                item["peak_a_y"] += yoff; item["peak_b_y"] += yoff
                item["peak_a_x"] += xoff; item["peak_b_x"] += xoff
                x0, y0, x1, y1 = item["roi_xyxy"]
                item["roi_xyxy"] = [x0 + xoff, y0 + yoff, x1 + xoff, y1 + yoff]
                item["seam"], item["roi"] = full_seam, full_roi
                item["component_id"] = component_id
                candidates.append(item)
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:max_candidates], {"components_inspected": inspected, "anchor_components": n}


def pairwise_gt_gap(instance: np.ndarray, radius: int = 4) -> np.ndarray:
    count = np.zeros(instance.shape, dtype=np.uint8)
    for label_id in np.unique(instance):
        if label_id <= 0:
            continue
        count += ndimage.binary_dilation(instance == label_id, iterations=radius)
    return (count >= 2) & (instance == 0)


def pairwise_gt_boundary(instance: np.ndarray, radius: int = 2) -> np.ndarray:
    """Pixels near two distinct instance IDs, including touching-label borders."""
    count = np.zeros(instance.shape, dtype=np.uint8)
    for label_id in np.unique(instance):
        if label_id <= 0:
            continue
        count += ndimage.binary_dilation(instance == label_id, iterations=radius)
    return count >= 2


def audit(candidates: list[dict], instance: np.ndarray) -> dict:
    gt = instance > 0
    gap = pairwise_gt_gap(instance)
    inter_boundary = pairwise_gt_boundary(instance)
    union = np.zeros_like(gt)
    for item in candidates:
        seam, roi = item["seam"], item["roi"]
        union |= roi
        item["seam_gt_fg_fraction"] = float((seam & gt).sum() / max(1, seam.sum()))
        item["seam_gt_gap_fraction"] = float((seam & gap).sum() / max(1, seam.sum()))
        item["seam_gt_interinstance_fraction"] = float((seam & inter_boundary).sum() / max(1, seam.sum()))
        item["seam_destructive_fg_fraction"] = float((seam & gt & ~inter_boundary).sum() / max(1, seam.sum()))
        item["roi_gt_gap_coverage"] = float((roi & gap).sum() / max(1, gap.sum()))
    return {
        "gt_gap_pixels": int(gap.sum()),
        "union_gt_gap_coverage": float((union & gap).sum() / max(1, gap.sum())),
        "best_seam_gt_fg_fraction": min((x["seam_gt_fg_fraction"] for x in candidates), default=0.0),
        "top1_seam_gt_fg_fraction": candidates[0]["seam_gt_fg_fraction"] if candidates else 0.0,
        "top1_seam_gt_gap_fraction": candidates[0]["seam_gt_gap_fraction"] if candidates else 0.0,
        "top1_seam_gt_interinstance_fraction": candidates[0]["seam_gt_interinstance_fraction"] if candidates else 0.0,
        "top1_seam_destructive_fg_fraction": candidates[0]["seam_destructive_fg_fraction"] if candidates else 0.0,
    }


def render(image: np.ndarray, anchor: np.ndarray, candidates: list[dict], path: Path) -> None:
    canvas = Image.fromarray(np.uint8(np.clip(image * 255, 0, 255))).convert("RGB")
    overlay = np.asarray(canvas).copy()
    boundary = anchor ^ ndimage.binary_erosion(anchor)
    overlay[boundary] = (40, 210, 255)
    canvas = Image.blend(canvas, Image.fromarray(overlay), 0.65)
    draw = ImageDraw.Draw(canvas)
    for rank, item in enumerate(candidates):
        color = (255, 40, 40) if rank == 0 else (255, 160, 40)
        ys, xs = np.where(item["seam"])
        for x, y in zip(xs[::max(1, len(xs) // 250)], ys[::max(1, len(ys) // 250)]):
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
        for key in ("a", "b"):
            x, y = item[f"peak_{key}_x"], item[f"peak_{key}_y"]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(30, 255, 90))
        x0, y0, x1, y1 = item["roi_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), f"{rank + 1}:{item['score']:.2f}", fill=color)
    canvas.save(path, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val"))
    p.add_argument("--anchor-dir", type=Path, default=Path("outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks"))
    p.add_argument("--gt-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val_labels"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r269_wrist_internal_merge_roi_pilot"))
    p.add_argument("--v-min", type=float, default=0.68)
    p.add_argument("--v-max", type=float, default=0.93)
    p.add_argument("--max-peaks", type=int, default=6)
    p.add_argument("--max-candidates", type=int, default=6)
    p.add_argument("--visual-count", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    joined = " ".join(map(str, vars(args).values())).lower()
    if "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R269 is restricted to TSRS_RSNA-Epiphysis original-val")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(x.name for x in args.anchor_dir.glob("*.png"))
    if args.limit:
        names = names[:args.limit]
    rows, cases = [], []
    for name in names:
        stem = Path(name).stem
        anchor = read_mask(args.anchor_dir / name)
        loaded = find_readable_image(args.image_dir, stem, anchor.shape)
        if loaded is None:
            continue
        _image_path, image = loaded
        candidates, diag = propose(image, anchor, args.v_min, args.v_max,
                                   args.max_peaks, args.max_candidates)
        gt_path = args.gt_dir / name
        audit_data = audit(candidates, read_instance(gt_path)) if gt_path.exists() else {}
        for rank, item in enumerate(candidates, 1):
            serial = {k: v for k, v in item.items() if k not in {"seam", "roi"}}
            rows.append({"image": name, "rank": rank, **serial})
        cases.append({"image": name, "num_candidates": len(candidates), **diag, **audit_data})
        if len(cases) <= args.visual_count:
            render(image, anchor, candidates, args.output_dir / f"{stem}_merge_roi_overlay.png")
    if rows:
        with (args.output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    else:
        (args.output_dir / "candidates.csv").write_text("", encoding="utf-8")
    top = [r for r in rows if r["rank"] == 1]
    summary = {
        "run_id": "R269_wrist_internal_merge_roi_pilot",
        "dataset": "TSRS_RSNA-Epiphysis", "split": "original-val",
        "clean_test_used": False, "writes_masks": False,
        "images": len(cases), "candidate_rows": len(rows),
        "images_with_candidates": sum(x["num_candidates"] > 0 for x in cases),
        "mean_candidates_per_image": float(np.mean([x["num_candidates"] for x in cases])) if cases else 0.0,
        "mean_union_gt_gap_coverage": float(np.mean([x.get("union_gt_gap_coverage", 0.0) for x in cases])) if cases else 0.0,
        "top1_mean_gt_fg_fraction": float(np.mean([x.get("seam_gt_fg_fraction", 0.0) for x in top])) if top else 0.0,
        "top1_mean_gt_gap_fraction": float(np.mean([x.get("seam_gt_gap_fraction", 0.0) for x in top])) if top else 0.0,
        "top1_mean_gt_interinstance_fraction": float(np.mean([x.get("seam_gt_interinstance_fraction", 0.0) for x in top])) if top else 0.0,
        "top1_mean_destructive_fg_fraction": float(np.mean([x.get("seam_destructive_fg_fraction", 0.0) for x in top])) if top else 0.0,
        "cases": cases,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
