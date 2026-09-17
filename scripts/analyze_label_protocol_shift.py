#!/usr/bin/env python3
"""Audit label protocol and morphology shifts across dataset splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare label morphology/protocol statistics across splits.")
    parser.add_argument("--case", action="append", required=True, help="label,raw_root,dataset,split")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-component-area", type=int, default=4)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def component_stats(mask: np.ndarray, min_area: int) -> list[dict[str, float]]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps: list[dict[str, float]] = []
    for idx in range(1, n):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == idx
        contour = comp ^ ndimage.binary_erosion(comp)
        x = float(stats[idx, cv2.CC_STAT_LEFT])
        y = float(stats[idx, cv2.CC_STAT_TOP])
        w = float(stats[idx, cv2.CC_STAT_WIDTH])
        h = float(stats[idx, cv2.CC_STAT_HEIGHT])
        comps.append(
            {
                "area": float(area),
                "area_frac": safe_div(float(area), float(mask.size)),
                "bbox_w_frac": safe_div(w, float(mask.shape[1])),
                "bbox_h_frac": safe_div(h, float(mask.shape[0])),
                "aspect": safe_div(w, max(h, 1.0)),
                "perimeter_area": safe_div(float(contour.sum()), float(area)),
                "compactness": safe_div(float(contour.sum() ** 2), float(area)),
                "cx": safe_div(x + 0.5 * w, float(mask.shape[1])),
                "cy": safe_div(y + 0.5 * h, float(mask.shape[0])),
            }
        )
    return comps


def nearest_gap(comps: list[dict[str, float]]) -> float:
    if len(comps) < 2:
        return 0.0
    pts = np.asarray([[c["cx"], c["cy"]] for c in comps], dtype=np.float32)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    d[d == 0] = np.inf
    return float(np.min(d))


def image_features(mask: np.ndarray, min_area: int) -> dict[str, float]:
    comps = component_stats(mask, min_area)
    areas = np.asarray([c["area_frac"] for c in comps], dtype=np.float64)
    perim = np.asarray([c["perimeter_area"] for c in comps], dtype=np.float64)
    compact = np.asarray([c["compactness"] for c in comps], dtype=np.float64)
    aspects = np.asarray([c["aspect"] for c in comps], dtype=np.float64)
    cxs = np.asarray([c["cx"] for c in comps], dtype=np.float64)
    cys = np.asarray([c["cy"] for c in comps], dtype=np.float64)
    contour = mask ^ ndimage.binary_erosion(mask)
    return {
        "fg_frac": float(mask.mean()),
        "component_count": float(len(comps)),
        "component_area_median": float(np.median(areas)) if areas.size else 0.0,
        "component_area_iqr": float(np.percentile(areas, 75) - np.percentile(areas, 25)) if areas.size else 0.0,
        "component_area_max": float(areas.max()) if areas.size else 0.0,
        "perimeter_area_median": float(np.median(perim)) if perim.size else 0.0,
        "compactness_median": float(np.median(compact)) if compact.size else 0.0,
        "aspect_median": float(np.median(aspects)) if aspects.size else 0.0,
        "layout_width": float(cxs.max() - cxs.min()) if cxs.size else 0.0,
        "layout_height": float(cys.max() - cys.min()) if cys.size else 0.0,
        "nearest_center_gap": nearest_gap(comps),
        "global_boundary_frac": float(contour.mean()),
        "global_boundary_fg_ratio": safe_div(float(contour.sum()), float(mask.sum())),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q10": 0.0, "q90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q10": float(np.percentile(arr, 10)),
        "q90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def parse_case(text: str) -> tuple[str, Path, str, str]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("--case must be label,raw_root,dataset,split")
    return parts[0], Path(parts[1]), parts[2], parts[3]


def collect_case(case_text: str, min_area: int) -> dict[str, object]:
    label, raw_root, dataset, split = parse_case(case_text)
    label_dir = raw_root / dataset / f"{split}_labels"
    records = []
    for path in sorted(label_dir.glob("*.png")):
        mask = read_mask(path)
        rec = image_features(mask, min_area)
        rec["image"] = path.name
        records.append(rec)
    if not records:
        raise FileNotFoundError(f"no labels: {label_dir}")
    feature_names = [k for k in records[0] if k != "image"]
    return {
        "label": label,
        "raw_root": str(raw_root),
        "dataset": dataset,
        "split": split,
        "num_images": len(records),
        "summary": {name: summarize([float(r[name]) for r in records]) for name in feature_names},
        "per_image": records,
    }


def effect_vs_reference(case: dict[str, object], ref: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    summary = case["summary"]  # type: ignore[assignment]
    ref_summary = ref["summary"]  # type: ignore[assignment]
    for name, stats in summary.items():
        ref_stats = ref_summary[name]
        denom = max(float(ref_stats["std"]), 1e-6)
        out[name] = float((float(stats["mean"]) - float(ref_stats["mean"])) / denom)
    return out


def main() -> None:
    args = parse_args()
    cases = [collect_case(case, args.min_component_area) for case in args.case]
    ref = cases[0]
    for case in cases:
        case["effect_vs_first_case"] = effect_vs_reference(case, ref)
        effects = case["effect_vs_first_case"]  # type: ignore[assignment]
        case["top_shift_features"] = [
            {"feature": k, "std_effect": v}
            for k, v in sorted(effects.items(), key=lambda item: abs(item[1]), reverse=True)[:8]
        ]
    output = {
        "min_component_area": args.min_component_area,
        "reference_case": ref["label"],
        "cases": cases,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {
        case["label"]: {
            "num_images": case["num_images"],
            "fg_frac_mean": case["summary"]["fg_frac"]["mean"],  # type: ignore[index]
            "component_count_mean": case["summary"]["component_count"]["mean"],  # type: ignore[index]
            "perimeter_area_median": case["summary"]["perimeter_area_median"]["mean"],  # type: ignore[index]
            "top_shift_features": case["top_shift_features"],
        }
        for case in cases
    }
    print(json.dumps({"output": str(out), "compact": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
