#!/usr/bin/env python3
"""Frozen original-val diagnostics for the R255 2+2 epoch sanity gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from audit_r255_close_gap_gate_a import capsule_coords, instance_records, nearest_boundary


def read_image(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path))
    return value[..., 0] if value.ndim == 3 else value


def read_metadata(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {str(row["id"]): float(row["boneage"]) for row in csv.DictReader(f)}


def resize_binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask > 0
    return np.asarray(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(shape[::-1], Image.Resampling.NEAREST)) > 0


def candidate_pairs(label: np.ndarray, min_area: int, neighbors: int) -> list[tuple[dict, dict, np.ndarray, np.ndarray, float, float]]:
    records = instance_records(label, min_area)
    if len(records) < 2:
        return []
    values = sorted(records)
    coords = np.asarray([[records[v]["centroid_y"], records[v]["centroid_x"]] for v in values])
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    raw: set[tuple[int, int]] = set()
    for idx, value in enumerate(values):
        nearest = [int(j) for j in np.argsort(distances[idx]) if int(j) != idx and np.isfinite(distances[idx, int(j)])]
        for other_idx in nearest[:neighbors]:
            raw.add(tuple(sorted((value, values[other_idx]))))
    fy, _ = np.nonzero(label > 0)
    y0, y1 = float(fy.min()), float(fy.max())
    result = []
    for i, j in sorted(raw):
        a, b = records[i], records[j]
        p0, p1, distance = nearest_boundary(a, b)
        gap = max(0.0, distance - 1.0)
        width = min(float(a["width"]), float(b["width"]))
        relative_gap = gap / max(width, 1e-6)
        midpoint_v = (((a["centroid_y"] + b["centroid_y"]) * 0.5) - y0) / max(1.0, y1 - y0)
        if midpoint_v < 0.45 or gap > 12.0 or relative_gap > 0.40:
            continue
        radius = max(1, int(round(0.12 * width)))
        cy, cx = capsule_coords(p0, p1, radius, label.shape)
        values_in_capsule = label[cy, cx]
        if np.any((values_in_capsule > 0) & (values_in_capsule != i) & (values_in_capsule != j)):
            continue
        legal = values_in_capsule == 0
        if legal.any():
            result.append((a, b, cy[legal], cx[legal], gap, relative_gap))
    return result


def gap_bin(gap: float) -> str:
    if 1.0 <= gap < 3.0:
        return "1-2"
    if 3.0 <= gap < 5.0:
        return "3-4"
    if 5.0 <= gap < 9.0:
        return "5-8"
    return ">8" if gap >= 9.0 else "<1"


def pair_merged(pred_labels: np.ndarray, label: np.ndarray, a: dict, b: dict, threshold: float) -> bool:
    shared = None
    for record in (a, b):
        mask = label == int(record["value"])
        ids, counts = np.unique(pred_labels[mask], return_counts=True)
        meaningful = {int(i) for i, count in zip(ids, counts) if i > 0 and count / max(1, int(mask.sum())) >= threshold}
        shared = meaningful if shared is None else shared & meaningful
    return bool(shared)


def image_diagnostics(pred: np.ndarray, label: np.ndarray, threshold: float) -> dict[str, float]:
    pred_labels, _ = ndimage.label(pred, structure=np.ones((3, 3), dtype=np.uint8))
    recalls, small_recalls, splits = [], [], []
    for value in [int(v) for v in np.unique(label) if int(v) > 0]:
        mask = label == value
        area = int(mask.sum())
        recalls.append(float(pred[mask].mean()))
        if area <= 512:
            small_recalls.append(float(pred[mask].mean()))
        ids, counts = np.unique(pred_labels[mask], return_counts=True)
        meaningful = sum(int(i > 0 and count / max(1, area) >= threshold) for i, count in zip(ids, counts))
        splits.append(float(meaningful >= 2))
    return {
        "instance_recall": float(np.mean(recalls)) if recalls else 0.0,
        "small_instance_recall_le512": float(np.mean(small_recalls)) if small_recalls else 0.0,
        "false_split_rate": float(np.mean(splits)) if splits else 0.0,
        "foreground_area_ratio": float(pred.sum() / max(1, (label > 0).sum())),
    }


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["gap_bin"])].append(row)
        groups["all"].append(row)
        if row["gap_bin"] in {"1-2", "3-4"}:
            groups["1-4"].append(row)
    result = {}
    for name, values in groups.items():
        result[name] = {
            "num_pairs": len(values),
            "baseline_gap_fp_rate": float(np.mean([r["baseline_gap_fp_rate"] for r in values])),
            "r255_gap_fp_rate": float(np.mean([r["r255_gap_fp_rate"] for r in values])),
            "delta_gap_fp_rate": float(np.mean([r["r255_gap_fp_rate"] - r["baseline_gap_fp_rate"] for r in values])),
            "baseline_pair_merge_rate": float(np.mean([r["baseline_pair_merged"] for r in values])),
            "r255_pair_merge_rate": float(np.mean([r["r255_pair_merged"] for r in values])),
            "delta_pair_merge_rate": float(np.mean([r["r255_pair_merged"] - r["baseline_pair_merged"] for r in values])),
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--r255-dir", type=Path, required=True)
    p.add_argument("--metadata-csv", type=Path)
    p.add_argument("--gate-json", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--expected-count", type=int, default=96)
    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--neighbors", type=int, default=4)
    p.add_argument("--min-overlap-frac", type=float, default=0.05)
    args = p.parse_args()
    joined = " ".join(map(str, (args.gt_dir, args.baseline_dir, args.r255_dir, args.output_json))).lower()
    if "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R255 sanity evaluation permits Epiphysis original-val only")
    metadata = read_metadata(args.metadata_csv)
    gate = json.loads(args.gate_json.read_text(encoding="utf-8"))
    age_low, age_high = map(float, gate["age_quartiles_months"])
    rows, image_rows = [], []
    paths = sorted(args.gt_dir.glob("*.png"))
    if len(paths) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} GT images, found {len(paths)}")
    for gt_path in paths:
        label = read_image(gt_path).astype(np.int32)
        baseline = resize_binary(read_image(args.baseline_dir / gt_path.name), label.shape)
        ours = resize_binary(read_image(args.r255_dir / gt_path.name), label.shape)
        pred_baseline, _ = ndimage.label(baseline, structure=np.ones((3, 3), dtype=np.uint8))
        pred_ours, _ = ndimage.label(ours, structure=np.ones((3, 3), dtype=np.uint8))
        age = metadata.get(gt_path.stem)
        age_group = "unknown" if age is None else "low" if age <= age_low else "high" if age >= age_high else "middle"
        for a, b, gy, gx, gap, relative_gap in candidate_pairs(label, args.min_area, args.neighbors):
            rows.append({
                "image": gt_path.name, "boneage": age, "age_group": age_group,
                "instance_i": int(a["value"]), "instance_j": int(b["value"]),
                "gap_px": gap, "relative_gap": relative_gap, "gap_bin": gap_bin(gap),
                "legal_gap_pixels": len(gx),
                "baseline_gap_fp_rate": float(baseline[gy, gx].mean()),
                "r255_gap_fp_rate": float(ours[gy, gx].mean()),
                "baseline_pair_merged": float(pair_merged(pred_baseline, label, a, b, args.min_overlap_frac)),
                "r255_pair_merged": float(pair_merged(pred_ours, label, a, b, args.min_overlap_frac)),
            })
        base_image = image_diagnostics(baseline, label, args.min_overlap_frac)
        ours_image = image_diagnostics(ours, label, args.min_overlap_frac)
        image_rows.append({"image": gt_path.name, "age_group": age_group, **{f"baseline_{k}": v for k, v in base_image.items()}, **{f"r255_{k}": v for k, v in ours_image.items()}})
    if not rows:
        raise RuntimeError("No legal original-val close-gap pairs found")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    image_summary = {}
    for key in [k for k in image_rows[0] if k not in {"image", "age_group"}]:
        image_summary[key] = float(np.mean([r[key] for r in image_rows]))
    payload = {
        "run_id": "R255_GATE_B_ORIGINAL_VAL",
        "scope": "TSRS_RSNA-Epiphysis original-val only",
        "clean_test_used": False,
        "frozen_gate_a_config": gate["config"],
        "num_images": len(paths), "num_pairs": len(rows),
        "pair_summary": summarize(rows), "image_summary": image_summary,
        "per_image": image_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"num_pairs": len(rows), "pair_summary": payload["pair_summary"], "image_summary": image_summary}, indent=2))


if __name__ == "__main__":
    main()
