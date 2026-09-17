#!/usr/bin/env python3
"""Retrieve train/val samples that resemble audited clean-test hard cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


FEATURE_NAMES = [
    "fg_frac",
    "component_count",
    "component_area_median",
    "component_area_iqr",
    "component_area_max",
    "component_area_min",
    "component_area_cv",
    "layout_width",
    "layout_height",
    "layout_cy_median",
    "layout_cy_span",
    "nearest_center_gap",
    "nearest_bbox_gap",
    "perimeter_area_median",
    "compactness_median",
    "aspect_median",
    "thin_component_frac",
    "boundary_fg_ratio",
    "image_mean",
    "image_std",
    "fg_mean",
    "bg_mean",
    "fg_bg_contrast",
    "fg_std",
    "roi_std",
]


TAG_FEATURE_WEIGHTS = {
    "under_separated_bridge": {
        "component_count": 2.0,
        "nearest_center_gap": 1.8,
        "nearest_bbox_gap": 1.8,
        "layout_width": 1.4,
        "thin_component_frac": 1.3,
    },
    "candidate_shared_failure": {
        "image_std": 1.6,
        "fg_bg_contrast": 1.8,
        "roi_std": 1.4,
        "nearest_bbox_gap": 1.3,
    },
    "overmask_low_precision": {
        "fg_frac": 1.8,
        "component_area_max": 1.5,
        "layout_height": 1.3,
    },
    "severe_boundary_shift": {
        "perimeter_area_median": 1.6,
        "compactness_median": 1.6,
        "boundary_fg_ratio": 1.4,
        "thin_component_frac": 1.3,
    },
    "underreach_high_precision": {
        "component_area_min": 1.6,
        "component_area_median": 1.4,
        "thin_component_frac": 1.4,
        "fg_frac": 1.2,
    },
    "candidate_fixable": {
        "nearest_bbox_gap": 1.5,
        "perimeter_area_median": 1.4,
        "fg_bg_contrast": 1.3,
        "component_area_iqr": 1.2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve train/val hard-case candidates.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--audit-taxonomy", required=True)
    parser.add_argument("--target-raw-root", default=None)
    parser.add_argument("--target-dataset", default=None)
    parser.add_argument("--target-split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--panel-k", type=int, default=32)
    parser.add_argument("--panel-max-width", type=int, default=240)
    parser.add_argument("--panel-fill-alpha", type=float, default=0.10)
    parser.add_argument("--panel-outline-width", type=int, default=2)
    parser.add_argument("--min-component-area", type=int, default=4)
    return parser.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def read_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def bbox_gap(a: dict[str, float], b: dict[str, float], width: int, height: int) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = b["x"] + b["w"], b["y"] + b["h"]
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return math.sqrt(dx * dx + dy * dy) / float(max(width, height))


def component_stats(mask: np.ndarray, min_area: int) -> list[dict[str, float]]:
    height, width = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps: list[dict[str, float]] = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = float(stats[idx, cv2.CC_STAT_LEFT])
        y = float(stats[idx, cv2.CC_STAT_TOP])
        w = float(stats[idx, cv2.CC_STAT_WIDTH])
        h = float(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[idx]
        bbox_perimeter = 2.0 * (w + h)
        comps.append({
            "area": float(area),
            "area_frac": safe_div(float(area), float(mask.size)),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "bbox_w_frac": safe_div(w, float(width)),
            "bbox_h_frac": safe_div(h, float(height)),
            "aspect": safe_div(w, max(h, 1.0)),
            "cx": safe_div(float(cx), float(width)),
            "cy": safe_div(float(cy), float(height)),
            "perimeter_area": safe_div(bbox_perimeter, float(area)),
            "compactness": safe_div(bbox_perimeter * bbox_perimeter, float(area)),
            "thinness": safe_div(min(w, h), float(max(width, height))),
        })
    return comps


def nearest_center_gap(comps: list[dict[str, float]]) -> float:
    if len(comps) < 2:
        return 0.0
    pts = np.asarray([[c["cx"], c["cy"]] for c in comps], dtype=np.float32)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    d[d == 0] = np.inf
    return float(np.min(d))


def nearest_box_gap(mask: np.ndarray, comps: list[dict[str, float]]) -> float:
    if len(comps) < 2:
        return 0.0
    height, width = mask.shape
    gaps = []
    for i, comp_a in enumerate(comps):
        for comp_b in comps[i + 1 :]:
            gaps.append(bbox_gap(comp_a, comp_b, width, height))
    return float(min(gaps)) if gaps else 0.0


def image_features(image_path: Path, label_path: Path, min_area: int) -> dict[str, float]:
    image = read_gray(image_path)
    mask = read_mask(label_path)
    if image.shape != mask.shape:
        image = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize(mask.shape[::-1], Image.BILINEAR), dtype=np.float32) / 255.0
    comps = component_stats(mask, min_area)
    areas = np.asarray([c["area_frac"] for c in comps], dtype=np.float64)
    perim = np.asarray([c["perimeter_area"] for c in comps], dtype=np.float64)
    compact = np.asarray([c["compactness"] for c in comps], dtype=np.float64)
    aspects = np.asarray([c["aspect"] for c in comps], dtype=np.float64)
    cxs = np.asarray([c["cx"] for c in comps], dtype=np.float64)
    cys = np.asarray([c["cy"] for c in comps], dtype=np.float64)
    thin = np.asarray([c["thinness"] for c in comps], dtype=np.float64)
    contour = mask ^ ndimage.binary_erosion(mask)
    ys, xs = np.where(mask)
    if len(xs):
        pad = 16
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1])
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0])
        roi = image[y0:y1, x0:x1]
    else:
        roi = image
    fg = image[mask]
    bg = image[~mask]
    feature = {
        "fg_frac": float(mask.mean()),
        "component_count": float(len(comps)),
        "component_area_median": float(np.median(areas)) if areas.size else 0.0,
        "component_area_iqr": float(np.percentile(areas, 75) - np.percentile(areas, 25)) if areas.size else 0.0,
        "component_area_max": float(areas.max()) if areas.size else 0.0,
        "component_area_min": float(areas.min()) if areas.size else 0.0,
        "component_area_cv": safe_div(float(areas.std()), float(areas.mean())) if areas.size else 0.0,
        "layout_width": float(cxs.max() - cxs.min()) if cxs.size else 0.0,
        "layout_height": float(cys.max() - cys.min()) if cys.size else 0.0,
        "layout_cy_median": float(np.median(cys)) if cys.size else 0.0,
        "layout_cy_span": float(cys.max() - cys.min()) if cys.size else 0.0,
        "nearest_center_gap": nearest_center_gap(comps),
        "nearest_bbox_gap": nearest_box_gap(mask, comps),
        "perimeter_area_median": float(np.median(perim)) if perim.size else 0.0,
        "compactness_median": float(np.median(compact)) if compact.size else 0.0,
        "aspect_median": float(np.median(aspects)) if aspects.size else 0.0,
        "thin_component_frac": float((thin < 0.018).mean()) if thin.size else 0.0,
        "boundary_fg_ratio": safe_div(float(contour.sum()), float(mask.sum())),
        "image_mean": float(image.mean()),
        "image_std": float(image.std()),
        "fg_mean": float(fg.mean()) if fg.size else 0.0,
        "bg_mean": float(bg.mean()) if bg.size else 0.0,
        "fg_bg_contrast": float(abs((fg.mean() if fg.size else 0.0) - (bg.mean() if bg.size else 0.0))),
        "fg_std": float(fg.std()) if fg.size else 0.0,
        "roi_std": float(roi.std()) if roi.size else 0.0,
    }
    return {name: float(feature[name]) for name in FEATURE_NAMES}


def collect_split(raw_root: Path, dataset: str, split: str, min_area: int) -> list[dict[str, Any]]:
    image_dir = raw_root / dataset / split
    label_dir = raw_root / dataset / f"{split}_labels"
    records = []
    for label_path in sorted(label_dir.glob("*.png")):
        stem = label_path.stem
        image_path = find_image(image_dir, stem)
        if image_path is None:
            continue
        records.append({
            "image": label_path.name,
            "split": split,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "features": image_features(image_path, label_path, min_area),
        })
    return records


def collect_targets(args: argparse.Namespace, taxonomy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_raw_root = Path(args.target_raw_root or args.raw_root)
    target_dataset = args.target_dataset or args.dataset
    image_dir = target_raw_root / target_dataset / args.target_split
    label_dir = target_raw_root / target_dataset / f"{args.target_split}_labels"
    targets = []
    for row in taxonomy:
        image_name = str(row["image"])
        stem = Path(image_name).stem
        image_path = find_image(image_dir, stem)
        label_path = label_dir / image_name
        if image_path is None or not label_path.exists():
            continue
        targets.append({
            "image": image_name,
            "split": args.target_split,
            "primary_tag": row.get("primary_tag", ""),
            "auto_tags": row.get("auto_tags", []),
            "dice": row.get("dice", None),
            "panel": row.get("panel", ""),
            "features": image_features(image_path, label_path, args.min_component_area),
        })
    return targets


def robust_scale(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats = {}
    for name in FEATURE_NAMES:
        arr = np.asarray([float(rec["features"][name]) for rec in records], dtype=np.float64)
        med = float(np.median(arr)) if arr.size else 0.0
        mad = float(np.median(np.abs(arr - med))) if arr.size else 1.0
        std = float(arr.std()) if arr.size else 1.0
        stats[name] = {"median": med, "mad": max(mad, 1e-6), "std": max(std, 1e-6)}
    return stats


def z_features(features: dict[str, float], stats: dict[str, dict[str, float]]) -> dict[str, float]:
    return {name: float(0.6745 * (float(features[name]) - stats[name]["median"]) / stats[name]["mad"]) for name in FEATURE_NAMES}


def prototype(records: list[dict[str, Any]]) -> dict[str, float]:
    return {
        name: float(np.median([float(rec["z_features"][name]) for rec in records]))
        for name in FEATURE_NAMES
    }


def distance(
    z: dict[str, float],
    proto: dict[str, float],
    weights: dict[str, float] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    weights = weights or {}
    total = 0.0
    denom = 0.0
    contrib = []
    for name in FEATURE_NAMES:
        weight = float(weights.get(name, 1.0))
        diff = float(z[name] - proto[name])
        value = weight * diff * diff
        total += value
        denom += weight
        contrib.append({"feature": name, "abs_z_delta": abs(diff), "weighted_sq": value})
    return math.sqrt(total / max(denom, 1e-6)), sorted(contrib, key=lambda item: item["weighted_sq"], reverse=True)[:6]


def build_prototypes(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["all_hard"] = targets
    for target in targets:
        for tag in target.get("auto_tags", []):
            grouped[str(tag)].append(target)
    protos = {}
    for tag, rows in grouped.items():
        if len(rows) < 2 and tag != "all_hard":
            continue
        protos[tag] = {
            "count": len(rows),
            "prototype": prototype(rows),
            "weights": TAG_FEATURE_WEIGHTS.get(tag, {}),
        }
    return protos


def score_candidates(
    candidates: list[dict[str, Any]],
    protos: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scored = []
    for rec in candidates:
        proto_scores = []
        for name, proto_info in protos.items():
            score, contrib = distance(rec["z_features"], proto_info["prototype"], proto_info.get("weights", {}))
            proto_scores.append({
                "prototype": name,
                "score": score,
                "count": proto_info["count"],
                "top_contributors": contrib,
            })
        proto_scores = sorted(proto_scores, key=lambda item: item["score"])
        best = proto_scores[0]
        scored.append({
            "image": rec["image"],
            "split": rec["split"],
            "image_path": rec["image_path"],
            "label_path": rec["label_path"],
            "hardcase_similarity_score": best["score"],
            "nearest_prototype": best["prototype"],
            "nearest_prototype_count": best["count"],
            "top_contributors": best["top_contributors"],
            "prototype_scores": proto_scores,
            "features": rec["features"],
            "z_features": rec["z_features"],
        })
    return sorted(scored, key=lambda item: item["hardcase_similarity_score"])


def overlay_panel(
    image_path: Path,
    label_path: Path,
    title: str,
    max_width: int,
    fill_alpha: float,
    outline_width: int,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if image.width > max_width:
        scale = max_width / float(image.width)
        image = image.resize((max_width, max(1, int(image.height * scale))), Image.BILINEAR)
    mask = np.asarray(Image.open(label_path).convert("L").resize(image.size, Image.NEAREST)) > 0
    arr = np.asarray(image).copy()
    fill_alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    arr[mask] = (((1.0 - fill_alpha) * arr[mask]) + fill_alpha * np.array([40, 220, 90])).astype(np.uint8)
    if mask.any() and outline_width > 0:
        structure = np.ones((3, 3), dtype=bool)
        outer = ndimage.binary_dilation(mask, structure=structure, iterations=outline_width)
        inner = ndimage.binary_erosion(mask, structure=structure, iterations=max(1, outline_width - 1))
        boundary = outer ^ inner
        arr[boundary] = np.array([255, 230, 35], dtype=np.uint8)
    panel = Image.fromarray(arr)
    title_h = 36
    canvas = Image.new("RGB", (panel.width, panel.height + title_h), (255, 255, 255))
    canvas.paste(panel, (0, title_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title[:42], fill=(0, 0, 0))
    return canvas


def write_panels(
    rows: list[dict[str, Any]],
    output_dir: Path,
    panel_k: int,
    panel_max_width: int,
    fill_alpha: float,
    outline_width: int,
) -> None:
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(rows[:panel_k], start=1):
        title = f"{rank:02d} {row['split']} {row['image']} {row['nearest_prototype']} {row['hardcase_similarity_score']:.2f}"
        panel = overlay_panel(
            Path(row["image_path"]),
            Path(row["label_path"]),
            title,
            panel_max_width,
            fill_alpha,
            outline_width,
        )
        stem = Path(str(row["image"])).stem
        out = panel_dir / f"{rank:02d}_{row['split']}_{stem}_candidate.jpg"
        panel.save(out, quality=93)
        row["panel"] = str(out.relative_to(output_dir)).replace("\\", "/")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "rank",
        "split",
        "image",
        "hardcase_similarity_score",
        "nearest_prototype",
        "nearest_prototype_count",
        "component_count",
        "fg_frac",
        "nearest_bbox_gap",
        "nearest_center_gap",
        "fg_bg_contrast",
        "image_std",
        "roi_std",
        "top_contributors",
        "panel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            features = row["features"]
            writer.writerow({
                "rank": rank,
                "split": row["split"],
                "image": row["image"],
                "hardcase_similarity_score": f"{row['hardcase_similarity_score']:.6f}",
                "nearest_prototype": row["nearest_prototype"],
                "nearest_prototype_count": row["nearest_prototype_count"],
                "component_count": f"{features['component_count']:.0f}",
                "fg_frac": f"{features['fg_frac']:.6f}",
                "nearest_bbox_gap": f"{features['nearest_bbox_gap']:.6f}",
                "nearest_center_gap": f"{features['nearest_center_gap']:.6f}",
                "fg_bg_contrast": f"{features['fg_bg_contrast']:.6f}",
                "image_std": f"{features['image_std']:.6f}",
                "roi_std": f"{features['roi_std']:.6f}",
                "top_contributors": ";".join(item["feature"] for item in row["top_contributors"][:4]),
                "panel": row.get("panel", ""),
            })


def write_markdown(
    output_dir: Path,
    rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    protos: dict[str, dict[str, Any]],
    top_k: int,
) -> None:
    top = rows[:top_k]
    split_counts = Counter(row["split"] for row in top)
    proto_counts = Counter(row["nearest_prototype"] for row in top)
    proto_text = ", ".join(f"{name}({info['count']})" for name, info in protos.items())
    md = [
        "# R124 Hard-Case Train/Val Retrieval",
        "",
        "This is a non-GPU data audit. It ranks original train/val labels by similarity to the R122/R123 hard-case audit set.",
        "",
        "## Summary",
        "",
        f"- Target hard cases with local image/label features: `{len(targets)}`",
        f"- Prototypes: `{proto_text}`",
        f"- Top `{top_k}` candidate split counts: `{dict(split_counts)}`",
        f"- Top `{top_k}` nearest prototype counts: `{dict(proto_counts)}`",
        "",
        "## Interpretation",
        "",
        "- Use these candidates for manual visual review and isolated data-variant design.",
        "- Do not treat this as a new model result; it does not use clean-test-v2 labels for training or threshold selection.",
        "- If the top candidates visibly match bridge/low-contrast hard cases, the next justified step is a hard-case curation manifest rather than another blind architecture run.",
        "",
        "## Top Candidates",
        "",
        "| Rank | Split | Image | Score | Prototype | Key Feature Deltas | Panel |",
        "| ---: | --- | --- | ---: | --- | --- | --- |",
    ]
    for rank, row in enumerate(top[:40], start=1):
        contrib = ", ".join(item["feature"] for item in row["top_contributors"][:3])
        panel = row.get("panel", "")
        panel_link = f"[panel]({panel})" if panel else ""
        md.append(
            f"| {rank} | {row['split']} | {row['image']} | {row['hardcase_similarity_score']:.3f} | "
            f"`{row['nearest_prototype']}` | {contrib} | {panel_link} |"
        )
    (output_dir / "R124_HARD_CASE_RETRIEVAL.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    taxonomy = json.loads(Path(args.audit_taxonomy).read_text(encoding="utf-8"))
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = collect_split(raw_root, args.dataset, "train", args.min_component_area)
    val = collect_split(raw_root, args.dataset, "val", args.min_component_area)
    candidates = train + val
    targets = collect_targets(args, taxonomy)
    if not candidates:
        raise FileNotFoundError(f"No train/val candidates found under {raw_root / args.dataset}")
    if not targets:
        raise FileNotFoundError("No target audit cases could be matched to local images/labels")

    stats = robust_scale(candidates)
    for rec in candidates + targets:
        rec["z_features"] = z_features(rec["features"], stats)
    protos = build_prototypes(targets)
    scored = score_candidates(candidates, protos)
    write_panels(scored, output_dir, args.panel_k, args.panel_max_width, args.panel_fill_alpha, args.panel_outline_width)
    write_csv(scored, output_dir / "r124_hard_case_trainval_retrieval.csv")
    write_markdown(output_dir, scored, targets, protos, args.top_k)

    compact_rows = scored[: args.top_k]
    json_output = {
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "audit_taxonomy": args.audit_taxonomy,
        "num_train": len(train),
        "num_val": len(val),
        "num_targets": len(targets),
        "feature_names": FEATURE_NAMES,
        "robust_stats_from_trainval": stats,
        "prototypes": protos,
        "top_k": args.top_k,
        "panel_max_width": args.panel_max_width,
        "panel_fill_alpha": args.panel_fill_alpha,
        "panel_outline_width": args.panel_outline_width,
        "top_candidates": compact_rows,
    }
    json_path = output_dir / "r124_hard_case_trainval_retrieval.json"
    json_path.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "json": str(json_path),
        "csv": str(output_dir / "r124_hard_case_trainval_retrieval.csv"),
        "markdown": str(output_dir / "R124_HARD_CASE_RETRIEVAL.md"),
        "num_train": len(train),
        "num_val": len(val),
        "num_targets": len(targets),
        "top_candidates": [
            {
                "rank": idx + 1,
                "split": row["split"],
                "image": row["image"],
                "score": row["hardcase_similarity_score"],
                "prototype": row["nearest_prototype"],
            }
            for idx, row in enumerate(compact_rows[:10])
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
