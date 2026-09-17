"""CPU pilot for generic adjacent-structure pair ROIs.

The proposal stage uses only the anchor mask and X-ray.  Original-val labels
are opened only by the audit stage to measure ROI coverage; they never select
or modify predictions.  This is a coverage/risk pilot, not a final metric
claim and not a clean-test experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def read_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if image.shape != shape:
        image = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize(shape[::-1], Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return image


def components(mask: np.ndarray, min_area: int = 8):
    labels, n = ndimage.label(mask)
    result = []
    for idx in range(1, n + 1):
        ys, xs = np.where(labels == idx)
        if len(xs) < min_area:
            continue
        result.append({
            "label": idx,
            "area": int(len(xs)),
            "center": (float(xs.mean()), float(ys.mean())),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
        })
    return result


def pair_roi(image: np.ndarray, a: dict, b: dict) -> dict:
    h, w = image.shape
    ax, ay = a["center"]
    bx, by = b["center"]
    distance = float(np.hypot(ax - bx, ay - by))
    x0 = min(a["bbox"][0], b["bbox"][0])
    y0 = min(a["bbox"][1], b["bbox"][1])
    x1 = max(a["bbox"][2], b["bbox"][2])
    y1 = max(a["bbox"][3], b["bbox"][3])
    scale = max(x1 - x0, y1 - y0, 8)
    pad = int(round(0.35 * scale + 0.15 * distance))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    # Sample the center line and compare its intensity with two short endpoint
    # neighborhoods. A darker center valley is useful evidence, but never
    # becomes a hard cut by itself.
    steps = max(int(distance), 2)
    xs = np.clip(np.rint(np.linspace(ax, bx, steps)).astype(int), 0, w - 1)
    ys = np.clip(np.rint(np.linspace(ay, by, steps)).astype(int), 0, h - 1)
    profile = image[ys, xs]
    k = max(1, len(profile) // 5)
    endpoint = float(np.mean(np.r_[profile[:k], profile[-k:]]))
    center = float(np.mean(profile[k:-k] if len(profile) > 2 * k else profile))
    valley = float(np.clip(endpoint - center, -1.0, 1.0))
    return {"i": a["label"], "j": b["label"], "distance": distance,
            "roi_xyxy": [x0, y0, x1, y1], "valley_score": valley,
            "roi_area_fraction": float((x1 - x0) * (y1 - y0) / (h * w))}


def audit_roi(roi: dict, gt: np.ndarray) -> tuple[float, int]:
    x0, y0, x1, y1 = roi["roi_xyxy"]
    gap = ndimage.binary_dilation(gt, iterations=4) & ~gt
    region = np.zeros_like(gt)
    region[y0:y1, x0:x1] = True
    coverage = float((gap & region).sum() / max(1, gap.sum()))
    return coverage, int((gap & region).sum())


def audit_union(rois: list[dict], gt: np.ndarray) -> float:
    gap = ndimage.binary_dilation(gt, iterations=4) & ~gt
    region = np.zeros_like(gt)
    for roi in rois:
        x0, y0, x1, y1 = roi["roi_xyxy"]
        region[y0:y1, x0:x1] = True
    return float((gap & region).sum() / max(1, gap.sum()))


def render(image: np.ndarray, mask: np.ndarray, rois: list[dict], output: Path) -> None:
    canvas = Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for roi in rois:
        x0, y0, x1, y1 = roi["roi_xyxy"]
        color = (255, 80, 40) if roi["valley_score"] > 0.03 else (60, 180, 255)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 2, y0 + 2), f"{roi['valley_score']:.2f}", fill=color)
    canvas.save(output, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val"))
    p.add_argument("--anchor-dir", type=Path, default=Path("outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks"))
    p.add_argument("--gt-dir", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis/val_labels"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/analysis/r268_carpal_pair_roi_pilot"))
    p.add_argument("--visual-count", type=int, default=12)
    p.add_argument("--v-min", type=float, default=0.68,
                   help="normalized vertical lower bound for carpal ossification centers")
    p.add_argument("--v-max", type=float, default=0.93,
                   help="normalized vertical upper bound for carpal ossification centers")
    args = p.parse_args()
    joined = " ".join(str(x) for x in vars(args).values()).lower()
    if "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R268 pilot is original Epiphysis val only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    names = sorted(x.name for x in args.anchor_dir.glob("*.png"))
    for name in names:
        stem = Path(name).stem
        anchor = read_mask(args.anchor_dir / name)
        image_path = next((args.image_dir / f"{stem}{ext}" for ext in (".png", ".jpg", ".jpeg", ".bmp")
                           if (args.image_dir / f"{stem}{ext}").exists()
                           and (args.image_dir / f"{stem}{ext}").stat().st_size > 0), None)
        if image_path is None:
            continue
        image = read_image(image_path, anchor.shape)
        all_comps = components(anchor)
        # R265's validated anatomy gate places carpal ossification centers in
        # this lower wrist band.  Without this gate, nearest-neighbor pairs on
        # the fingers dominate the ROI list and are irrelevant to the target
        # failure mode.
        h = anchor.shape[0]
        wrist = [c for c in all_comps if args.v_min <= c["center"][1] / h <= args.v_max]
        wrist_scales = [max(c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1]) for c in wrist]
        carpal_scale = float(np.median(wrist_scales)) if wrist_scales else 1.0
        # Exclude the much larger radius/ulna and metacarpal-base components;
        # this is the same scale-relative wrist-center gate used by R265.
        comps = [c for c, scale in zip(wrist, wrist_scales)
                 if scale <= 1.55 * carpal_scale]
        candidates = []
        for i, a in enumerate(comps):
            nearest = sorted((pair_roi(image, a, b) for j, b in enumerate(comps) if i != j), key=lambda x: x["distance"])
            if nearest:
                candidates.append(nearest[0])
        candidates = sorted(candidates, key=lambda x: (x["distance"], -x["valley_score"]))[:8]
        gt_path = args.gt_dir / name
        coverage = 0.0
        union_coverage = 0.0
        covered = 0
        if gt_path.exists():
            gt = read_mask(gt_path)
            union_coverage = audit_union(candidates, gt)
            for roi in candidates:
                roi["gt_gap_coverage"], n = audit_roi(roi, gt)
                coverage = max(coverage, roi["gt_gap_coverage"])
                covered += n
        for rank, roi in enumerate(candidates):
            rows.append({"image": name, "rank": rank, **roi})
        summaries.append({"image": name, "num_components": len(comps), "num_wrist_components": len(wrist), "num_all_components": len(all_comps), "carpal_scale": carpal_scale, "num_rois": len(candidates), "max_gt_gap_coverage": coverage, "union_gt_gap_coverage": union_coverage, "covered_gap_pixels": covered, "region_v_min": args.v_min, "region_v_max": args.v_max})
        if len(summaries) <= args.visual_count:
            render(image, anchor, candidates, args.output_dir / f"{stem}_roi_overlay.png")
    (args.output_dir / "pair_candidates.csv").write_text("", encoding="utf-8")
    if rows:
        with (args.output_dir / "pair_candidates.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {"run_id": "R268_carpal_pair_roi_pilot", "split": "original-val", "region": {"v_min": args.v_min, "v_max": args.v_max}, "clean_test_used": False,
               "images": len(summaries), "mean_rois": float(np.mean([x["num_rois"] for x in summaries])) if summaries else 0.0,
               "mean_max_gt_gap_coverage": float(np.mean([x["max_gt_gap_coverage"] for x in summaries])) if summaries else 0.0,
               "mean_union_gt_gap_coverage": float(np.mean([x["union_gt_gap_coverage"] for x in summaries])) if summaries else 0.0,
               "summaries": summaries}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("run_id", "images", "mean_rois", "mean_max_gt_gap_coverage", "mean_union_gt_gap_coverage", "clean_test_used")}, indent=2))


if __name__ == "__main__":
    main()
