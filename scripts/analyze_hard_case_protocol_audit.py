#!/usr/bin/env python3
"""Audit whether a hard-case curation pool supports a label/protocol intervention."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


FEATURE_NAMES = [
    "fg_frac",
    "component_count",
    "component_area_median",
    "component_area_min",
    "component_area_max",
    "component_area_cv",
    "layout_width",
    "layout_height",
    "layout_cy_median",
    "nearest_center_gap",
    "nearest_bbox_gap",
    "perimeter_area_median",
    "compactness_median",
    "thin_component_frac",
    "boundary_fg_ratio",
    "image_std",
    "fg_bg_contrast",
    "roi_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit hard-case label/protocol intervention evidence.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--hardcase-manifest", required=True)
    parser.add_argument("--r122-audit-rows", required=True)
    parser.add_argument("--r126-clean-metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-component-area", type=int, default=4)
    parser.add_argument("--outlier-z", type=float, default=3.5)
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
        comps.append(
            {
                "area_frac": safe_div(float(area), float(mask.size)),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "cx": safe_div(float(cx), float(width)),
                "cy": safe_div(float(cy), float(height)),
                "thinness": safe_div(min(w, h), float(max(width, height))),
                "perimeter_area": safe_div(bbox_perimeter, float(area)),
                "compactness": safe_div(bbox_perimeter * bbox_perimeter, float(area)),
            }
        )
    return comps


def nearest_center_gap(comps: list[dict[str, float]]) -> float:
    if len(comps) < 2:
        return 0.0
    pts = np.asarray([[c["cx"], c["cy"]] for c in comps], dtype=np.float32)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    d[d == 0] = np.inf
    return float(np.min(d))


def bbox_gap(a: dict[str, float], b: dict[str, float], width: int, height: int) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = a["x"] + a["w"], a["y"] + a["h"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = b["x"] + b["w"], b["y"] + b["h"]
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    return (dx * dx + dy * dy) ** 0.5 / float(max(width, height))


def nearest_bbox_gap(mask: np.ndarray, comps: list[dict[str, float]]) -> float:
    if len(comps) < 2:
        return 0.0
    height, width = mask.shape
    gaps = [bbox_gap(a, b, width, height) for i, a in enumerate(comps) for b in comps[i + 1 :]]
    return float(min(gaps)) if gaps else 0.0


def image_features(image_path: Path, label_path: Path, min_area: int) -> dict[str, float]:
    image = read_gray(image_path)
    mask = read_mask(label_path)
    if image.shape != mask.shape:
        image = np.asarray(
            Image.fromarray((image * 255).astype(np.uint8)).resize(mask.shape[::-1], Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    comps = component_stats(mask, min_area)
    areas = np.asarray([c["area_frac"] for c in comps], dtype=np.float64)
    cxs = np.asarray([c["cx"] for c in comps], dtype=np.float64)
    cys = np.asarray([c["cy"] for c in comps], dtype=np.float64)
    thin = np.asarray([c["thinness"] for c in comps], dtype=np.float64)
    perim = np.asarray([c["perimeter_area"] for c in comps], dtype=np.float64)
    compact = np.asarray([c["compactness"] for c in comps], dtype=np.float64)
    contour = mask ^ ndimage.binary_erosion(mask)
    ys, xs = np.where(mask)
    if xs.size:
        pad = 16
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, mask.shape[1])
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, mask.shape[0])
        roi = image[y0:y1, x0:x1]
    else:
        roi = image
    fg = image[mask]
    bg = image[~mask]
    return {
        "fg_frac": float(mask.mean()),
        "component_count": float(len(comps)),
        "component_area_median": float(np.median(areas)) if areas.size else 0.0,
        "component_area_min": float(areas.min()) if areas.size else 0.0,
        "component_area_max": float(areas.max()) if areas.size else 0.0,
        "component_area_cv": safe_div(float(areas.std()), float(areas.mean())) if areas.size else 0.0,
        "layout_width": float(cxs.max() - cxs.min()) if cxs.size else 0.0,
        "layout_height": float(cys.max() - cys.min()) if cys.size else 0.0,
        "layout_cy_median": float(np.median(cys)) if cys.size else 0.0,
        "nearest_center_gap": nearest_center_gap(comps),
        "nearest_bbox_gap": nearest_bbox_gap(mask, comps),
        "perimeter_area_median": float(np.median(perim)) if perim.size else 0.0,
        "compactness_median": float(np.median(compact)) if compact.size else 0.0,
        "thin_component_frac": float((thin < 0.018).mean()) if thin.size else 0.0,
        "boundary_fg_ratio": safe_div(float(contour.sum()), float(mask.sum())),
        "image_std": float(image.std()),
        "fg_bg_contrast": float(abs((fg.mean() if fg.size else 0.0) - (bg.mean() if bg.size else 0.0))),
        "roi_std": float(roi.std()) if roi.size else 0.0,
    }


def collect_split(raw_root: Path, dataset: str, split: str, min_area: int) -> list[dict[str, Any]]:
    image_dir = raw_root / dataset / split
    label_dir = raw_root / dataset / f"{split}_labels"
    records = []
    for label_path in sorted(label_dir.glob("*.png")):
        image_path = find_image(image_dir, label_path.stem)
        if image_path is None:
            continue
        records.append(
            {
                "image": label_path.name,
                "split": split,
                "features": image_features(image_path, label_path, min_area),
            }
        )
    return records


def robust_stats(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    stats = {}
    for name in FEATURE_NAMES:
        arr = np.asarray([float(r["features"][name]) for r in records], dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        stats[name] = {"median": median, "mad": max(mad, 1e-6), "mean": float(arr.mean()), "std": max(float(arr.std()), 1e-6)}
    return stats


def score_against(rec: dict[str, Any], stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    z = {}
    for name in FEATURE_NAMES:
        s = stats[name]
        z[name] = float(0.6745 * (float(rec["features"][name]) - s["median"]) / s["mad"])
    top = [{"feature": k, "robust_z": v} for k, v in sorted(z.items(), key=lambda item: abs(item[1]), reverse=True)[:6]]
    return {
        "max_abs_z": float(max(abs(v) for v in z.values())) if z else 0.0,
        "mean_top3_abs_z": float(np.mean(sorted([abs(v) for v in z.values()], reverse=True)[:3])) if z else 0.0,
        "top_features": top,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "q10": 0.0, "q90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "q10": float(np.percentile(arr, 10)),
        "q90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "features": {name: summarize([float(r["features"][name]) for r in records]) for name in FEATURE_NAMES},
        "outlier_score": summarize([float(r["audit"]["mean_top3_abs_z"]) for r in records]) if records else summarize([]),
    }


def load_hard_names(manifest: dict[str, Any], split: str) -> set[str]:
    key = f"hard_{split}"
    return {str(item["image"]) for item in manifest.get(key, [])}


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.hardcase_manifest).read_text(encoding="utf-8"))
    r122_rows = json.loads(Path(args.r122_audit_rows).read_text(encoding="utf-8"))
    r126_clean = json.loads(Path(args.r126_clean_metrics).read_text(encoding="utf-8"))

    train_records = collect_split(raw_root, args.dataset, "train", args.min_component_area)
    val_records = collect_split(raw_root, args.dataset, "val", args.min_component_area)
    train_stats = robust_stats(train_records)

    hard_train_names = load_hard_names(manifest, "train")
    hard_val_names = load_hard_names(manifest, "val_audit")
    hard_train = []
    hard_val = []
    all_train = []
    all_val = []
    for rec in train_records:
        rec["audit"] = score_against(rec, train_stats)
        rec["is_r125_hard"] = rec["image"] in hard_train_names
        all_train.append(rec)
        if rec["is_r125_hard"]:
            hard_train.append(rec)
    for rec in val_records:
        rec["audit"] = score_against(rec, train_stats)
        rec["is_r125_hard"] = rec["image"] in hard_val_names
        all_val.append(rec)
        if rec["is_r125_hard"]:
            hard_val.append(rec)

    proto_counts = Counter(item.get("nearest_prototype", "unknown") for item in manifest.get("hard_train", []))
    severe_outliers = [
        {
            "image": rec["image"],
            "split": rec["split"],
            "mean_top3_abs_z": rec["audit"]["mean_top3_abs_z"],
            "max_abs_z": rec["audit"]["max_abs_z"],
            "top_features": rec["audit"]["top_features"],
        }
        for rec in hard_train + hard_val
        if float(rec["audit"]["mean_top3_abs_z"]) >= args.outlier_z
    ]

    r126_by_image = {item["image"]: item for item in r126_clean.get("per_image", [])}
    r122_by_tag = defaultdict(list)
    for row in r122_rows:
        for tag in row.get("auto_tags", []):
            r122_by_tag[tag].append(row)
    r126_hard_overlap = [
        {
            "image": image,
            "dice": r126_by_image[image]["dice"],
            "boundary_iou": r126_by_image[image]["boundary_iou"],
            "component_count_error": r126_by_image[image]["component_count_error"],
            "false_bridge_flag": r126_by_image[image]["false_bridge_flag"],
        }
        for image in sorted(r126_by_image)
        if image in {row["image"] for row in r122_rows}
    ]

    decision = {
        "simple_resampling_supported": False,
        "broad_label_filter_supported": False,
        "manual_label_protocol_review_supported": True,
        "next_gpu_supported_without_new_labels": False,
        "rationale": [
            "R126 hard-case sampler underperformed R117/R110, so the current hard pool is not sufficient as weighting signal.",
            "R125 hard-train prototypes are dominated by shared-failure, bridge, and underreach-like cases rather than candidate-fixable cases.",
            "Use severe feature outliers and hard-case tags to prioritize manual label/protocol review before another training run.",
        ],
    }
    if len(severe_outliers) >= max(8, int(0.15 * max(1, len(hard_train)))):
        decision["broad_label_filter_supported"] = True
        decision["rationale"].append("A large fraction of hard-train cases are robust feature outliers; consider a small verified manifest.")
    if proto_counts.get("candidate_fixable", 0) >= max(12, int(0.25 * max(1, len(hard_train)))):
        decision["next_gpu_supported_without_new_labels"] = True
        decision["rationale"].append("Candidate-fixable hard-train share is high enough to justify a learner.")

    output = {
        "run_id": "R127",
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "inputs": {
            "hardcase_manifest": args.hardcase_manifest,
            "r122_audit_rows": args.r122_audit_rows,
            "r126_clean_metrics": args.r126_clean_metrics,
        },
        "counts": {
            "train_total": len(all_train),
            "val_total": len(all_val),
            "hard_train": len(hard_train),
            "hard_val_audit": len(hard_val),
            "hard_train_prototypes": dict(proto_counts),
            "severe_outlier_threshold": args.outlier_z,
            "severe_outliers_in_hard_pool": len(severe_outliers),
        },
        "summaries": {
            "train_all": summarize_group(all_train),
            "val_all": summarize_group(all_val),
            "r125_hard_train": summarize_group(hard_train),
            "r125_hard_val_audit": summarize_group(hard_val),
        },
        "r122_tag_counts": {tag: len(rows) for tag, rows in sorted(r122_by_tag.items())},
        "r126_on_r122_hard_cases": {
            "count": len(r126_hard_overlap),
            "dice": summarize([float(r["dice"]) for r in r126_hard_overlap]),
            "boundary_iou": summarize([float(r["boundary_iou"]) for r in r126_hard_overlap]),
            "component_count_error": summarize([float(r["component_count_error"]) for r in r126_hard_overlap]),
            "false_bridge_mean": float(np.mean([float(r["false_bridge_flag"]) for r in r126_hard_overlap])) if r126_hard_overlap else 0.0,
            "per_image": r126_hard_overlap,
        },
        "severe_outliers": sorted(severe_outliers, key=lambda r: float(r["mean_top3_abs_z"]), reverse=True),
        "decision": decision,
    }

    json_path = output_dir / "r127_hard_case_protocol_audit.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    md_lines = [
        "# R127 Hard-Case Protocol Audit",
        "",
        "## Raw Numbers",
        "",
        f"- train total: `{len(all_train)}`",
        f"- val total: `{len(all_val)}`",
        f"- R125 hard-train: `{len(hard_train)}`",
        f"- R125 hard-val-audit: `{len(hard_val)}`",
        f"- severe hard-pool robust outliers (top3 z >= {args.outlier_z}): `{len(severe_outliers)}`",
        "",
        "## R125 Prototype Counts",
        "",
    ]
    for key, value in proto_counts.most_common():
        md_lines.append(f"- `{key}`: `{value}`")
    md_lines.extend(
        [
            "",
            "## R126 On R122 Hard Cases",
            "",
            f"- overlap count: `{len(r126_hard_overlap)}`",
            f"- mean Dice: `{output['r126_on_r122_hard_cases']['dice']['mean']:.6f}`",
            f"- mean Boundary IoU: `{output['r126_on_r122_hard_cases']['boundary_iou']['mean']:.6f}`",
            f"- mean component-count error: `{output['r126_on_r122_hard_cases']['component_count_error']['mean']:.6f}`",
            f"- false-bridge mean: `{output['r126_on_r122_hard_cases']['false_bridge_mean']:.6f}`",
            "",
            "## Decision",
            "",
            f"- simple resampling supported: `{decision['simple_resampling_supported']}`",
            f"- broad label filtering supported: `{decision['broad_label_filter_supported']}`",
            f"- manual label/protocol review supported: `{decision['manual_label_protocol_review_supported']}`",
            f"- next GPU without new labels supported: `{decision['next_gpu_supported_without_new_labels']}`",
            "",
            "## Rationale",
            "",
        ]
    )
    for item in decision["rationale"]:
        md_lines.append(f"- {item}")
    md_lines.extend(["", "## Top Severe Outliers", ""])
    for rec in output["severe_outliers"][:20]:
        top = ", ".join(f"{f['feature']}={f['robust_z']:.2f}" for f in rec["top_features"][:3])
        md_lines.append(f"- `{rec['split']}/{rec['image']}` top3-z `{rec['mean_top3_abs_z']:.3f}`: {top}")
    md_path = output_dir / "R127_HARD_CASE_PROTOCOL_AUDIT.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
