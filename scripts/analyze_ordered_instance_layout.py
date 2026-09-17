#!/usr/bin/env python3
"""Audit ordered instance-layout structure in epiphysis labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether instance-valued labels support ordered-layout modeling.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-area", type=int, default=4)
    return parser.parse_args()


def read_instance(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def robust_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "q10": 0.0, "q90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "q10": float(np.percentile(arr, 10)),
        "q90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def component_rows(instance: np.ndarray, min_area: int) -> list[dict[str, float]]:
    h, w = instance.shape
    rows: list[dict[str, float]] = []
    labels = instance.astype(np.int32, copy=False)
    max_id = int(labels.max()) if labels.size else 0
    if max_id <= 0:
        return rows
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=max_id + 1).astype(np.float64)
    yy, xx = np.indices(labels.shape, dtype=np.float64)
    sum_x = np.bincount(flat, weights=xx.ravel(), minlength=max_id + 1)
    sum_y = np.bincount(flat, weights=yy.ravel(), minlength=max_id + 1)
    for value in range(1, max_id + 1):
        area = counts[value]
        if area < min_area:
            continue
        component = (labels == value).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(component, connectivity=8)
        if len(stats) <= 1:
            continue
        x0 = int(stats[1, cv2.CC_STAT_LEFT])
        y0 = int(stats[1, cv2.CC_STAT_TOP])
        bw = int(stats[1, cv2.CC_STAT_WIDTH])
        bh = int(stats[1, cv2.CC_STAT_HEIGHT])
        x1 = x0 + bw - 1
        y1 = y0 + bh - 1
        rows.append(
            {
                "id": float(value),
                "area": float(area),
                "cx": float((sum_x[value] / area) / max(w - 1, 1)),
                "cy": float((sum_y[value] / area) / max(h - 1, 1)),
                "x0": float(x0 / max(w - 1, 1)),
                "x1": float(x1 / max(w - 1, 1)),
                "y0": float(y0 / max(h - 1, 1)),
                "y1": float(y1 / max(h - 1, 1)),
                "width": float((x1 - x0 + 1) / max(w, 1)),
                "height": float((y1 - y0 + 1) / max(h, 1)),
            }
        )
    return rows


def inversion_fraction(values: list[float]) -> float:
    total = 0
    inv = 0
    for i, vi in enumerate(values):
        for vj in values[i + 1 :]:
            total += 1
            if vi > vj:
                inv += 1
    return float(inv / total) if total else 0.0


def adjacent_gaps(rows: list[dict[str, float]], order_key: str) -> list[float]:
    ordered = sorted(rows, key=lambda r: r[order_key])
    gaps = []
    for a, b in zip(ordered, ordered[1:]):
        dx = float(a["cx"] - b["cx"])
        dy = float(a["cy"] - b["cy"])
        gaps.append(float((dx * dx + dy * dy) ** 0.5))
    return gaps


def bbox_overlap_pairs(rows: list[dict[str, float]]) -> int:
    overlaps = 0
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            x_overlap = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            y_overlap = min(a["y1"], b["y1"]) - max(a["y0"], b["y0"])
            if x_overlap > 0 and y_overlap > 0:
                overlaps += 1
    return overlaps


def longest_axis_score(rows: list[dict[str, float]]) -> dict[str, float]:
    if len(rows) < 2:
        return {"pc1_explained": 0.0, "pc1_angle_deg": 0.0, "pc1_order_inversion": 0.0}
    pts = np.asarray([[r["cx"], r["cy"]] for r in rows], dtype=np.float64)
    pts = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vec = vecs[:, order[0]]
    explained = float(vals[0] / max(vals.sum(), 1e-9))
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    projections = (pts @ vec).tolist()
    by_id = [p for _, p in sorted(zip([r["id"] for r in rows], projections), key=lambda item: item[0])]
    inv = min(inversion_fraction(by_id), inversion_fraction([-p for p in by_id]))
    return {"pc1_explained": explained, "pc1_angle_deg": angle, "pc1_order_inversion": float(inv)}


def audit_label(path: Path, min_area: int) -> dict[str, Any]:
    instance = read_instance(path)
    rows = component_rows(instance, min_area)
    ids = sorted(int(r["id"]) for r in rows)
    id_is_contiguous = ids == list(range(1, len(ids) + 1))
    by_id = sorted(rows, key=lambda r: r["id"])
    x_by_id = [float(r["cx"]) for r in by_id]
    y_by_id = [float(r["cy"]) for r in by_id]
    x_inv = min(inversion_fraction(x_by_id), inversion_fraction([-v for v in x_by_id]))
    y_inv = min(inversion_fraction(y_by_id), inversion_fraction([-v for v in y_by_id]))
    pc1 = longest_axis_score(rows)
    gaps_id = adjacent_gaps(rows, "id")
    gaps_x = adjacent_gaps(rows, "cx")
    return {
        "image": path.name,
        "component_count": len(rows),
        "max_id": max(ids) if ids else 0,
        "id_is_contiguous": bool(id_is_contiguous),
        "id_x_order_inversion": float(x_inv),
        "id_y_order_inversion": float(y_inv),
        "id_pc1_order_inversion": float(pc1["pc1_order_inversion"]),
        "pc1_explained": float(pc1["pc1_explained"]),
        "pc1_angle_deg": float(pc1["pc1_angle_deg"]),
        "median_adjacent_gap_by_id": float(np.median(gaps_id)) if gaps_id else 0.0,
        "median_adjacent_gap_by_x": float(np.median(gaps_x)) if gaps_x else 0.0,
        "bbox_overlap_pairs": bbox_overlap_pairs(rows),
        "components": rows,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count_hist = Counter(int(r["component_count"]) for r in records)
    return {
        "num_images": len(records),
        "component_count_hist": dict(sorted(count_hist.items())),
        "component_count": robust_summary([float(r["component_count"]) for r in records]),
        "contiguous_id_fraction": float(np.mean([float(r["id_is_contiguous"]) for r in records])) if records else 0.0,
        "id_x_order_inversion": robust_summary([float(r["id_x_order_inversion"]) for r in records]),
        "id_y_order_inversion": robust_summary([float(r["id_y_order_inversion"]) for r in records]),
        "id_pc1_order_inversion": robust_summary([float(r["id_pc1_order_inversion"]) for r in records]),
        "pc1_explained": robust_summary([float(r["pc1_explained"]) for r in records]),
        "median_adjacent_gap_by_id": robust_summary([float(r["median_adjacent_gap_by_id"]) for r in records]),
        "median_adjacent_gap_by_x": robust_summary([float(r["median_adjacent_gap_by_x"]) for r in records]),
        "bbox_overlap_pairs": robust_summary([float(r["bbox_overlap_pairs"]) for r in records]),
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        label_dir = raw_root / args.dataset / f"{split}_labels"
        records = [audit_label(path, args.min_area) | {"split": split} for path in sorted(label_dir.glob("*.png"))]
        by_split[split] = records
        all_records.extend(records)

    summary_by_split = {split: summarize_records(records) for split, records in by_split.items()}
    summary_all = summarize_records(all_records)

    decision = {
        "ordered_instance_signal_strong": summary_all["contiguous_id_fraction"] >= 0.95
        and summary_all["id_pc1_order_inversion"]["median"] <= 0.08
        and summary_all["pc1_explained"]["median"] >= 0.65,
        "fixed_slot_count_safe": summary_all["component_count"]["q10"] == summary_all["component_count"]["q90"],
        "single_axis_order_safe": summary_all["id_pc1_order_inversion"]["q90"] <= 0.12,
        "needs_variable_slots": summary_all["component_count"]["min"] != summary_all["component_count"]["max"],
    }
    if decision["ordered_instance_signal_strong"]:
        recommendation = "Ordered instance ids are stable enough to justify a variable-slot/order-aware architecture."
    else:
        recommendation = "Instance ids/layout are not stable enough for a naive fixed-slot model; use variable slots or manual protocol review first."
    if not decision["single_axis_order_safe"]:
        recommendation += " A one-dimensional left-right or top-bottom order alone is risky."

    output = {
        "run_id": "R128",
        "dataset": args.dataset,
        "raw_root": str(raw_root),
        "splits": list(by_split),
        "summary_all": summary_all,
        "summary_by_split": summary_by_split,
        "decision": decision,
        "recommendation": recommendation,
        "records": all_records,
    }
    json_path = output_dir / "r128_ordered_instance_layout_audit.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    md_lines = [
        "# R128 Ordered Instance Layout Audit",
        "",
        "## Raw Numbers",
        "",
        f"- images audited: `{summary_all['num_images']}`",
        f"- contiguous id fraction: `{summary_all['contiguous_id_fraction']:.6f}`",
        f"- component-count median: `{summary_all['component_count']['median']:.3f}`",
        f"- component-count q10/q90: `{summary_all['component_count']['q10']:.3f}` / `{summary_all['component_count']['q90']:.3f}`",
        f"- id-vs-PC1 inversion median/q90: `{summary_all['id_pc1_order_inversion']['median']:.6f}` / `{summary_all['id_pc1_order_inversion']['q90']:.6f}`",
        f"- PC1 explained median/q10: `{summary_all['pc1_explained']['median']:.6f}` / `{summary_all['pc1_explained']['q10']:.6f}`",
        f"- bbox-overlap pairs median/q90: `{summary_all['bbox_overlap_pairs']['median']:.3f}` / `{summary_all['bbox_overlap_pairs']['q90']:.3f}`",
        "",
        "## Decision",
        "",
    ]
    for key, value in decision.items():
        md_lines.append(f"- `{key}`: `{value}`")
    md_lines.extend(["", "## Recommendation", "", recommendation, "", "## Component Count Histogram", ""])
    for key, value in summary_all["component_count_hist"].items():
        md_lines.append(f"- `{key}`: `{value}`")
    md_path = output_dir / "R128_ORDERED_INSTANCE_LAYOUT_AUDIT.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
