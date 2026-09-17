#!/usr/bin/env python3
"""Train-only Gate A for R255 development-conditioned close-gap supervision.

This audit never reads validation/test/clean-test-v2. It tests whether the
instance-valued train labels and X-rays provide enough stable, legal y=0 close
gap anchors to justify implementing the training loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="data/raw")
    p.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    p.add_argument("--metadata-csv", default="G:/gutou/filtered_train.csv")
    p.add_argument("--output-json", default="outputs/analysis/r255_gate_a/r255_close_gap_gate_a.json")
    p.add_argument("--output-csv", default="outputs/analysis/r255_gate_a/r255_close_gap_pairs.csv")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=255)
    p.add_argument("--min-area", type=int, default=24)
    p.add_argument("--neighbors", type=int, default=4)
    p.add_argument("--roi-v-min", type=float, default=0.45)
    p.add_argument("--max-gap-px", type=float, default=12.0)
    p.add_argument("--max-relative-gap", type=float, default=0.40)
    p.add_argument("--capsule-rho", type=float, default=0.12)
    p.add_argument("--xray-floor", type=float, default=0.15)
    p.add_argument("--rotate-deg", type=float, default=2.0)
    return p.parse_args()


def read_metadata(path: Path) -> dict[str, tuple[float, bool]]:
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = (
                float(row["boneage"]),
                str(row["male"]).strip().lower() in {"true", "1", "yes"},
            )
    return rows


def find_image(root: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(stem)


def local_instance_record(
    label: np.ndarray,
    value: int,
    component_slice: tuple[slice, slice],
    min_area: int,
) -> dict[str, Any] | None:
    y0, y1 = int(component_slice[0].start), int(component_slice[0].stop)
    x0, x1 = int(component_slice[1].start), int(component_slice[1].stop)
    local = label[y0:y1, x0:x1] == value
    local_y, local_x = np.nonzero(local)
    if len(local_x) < min_area:
        return None
    yy, xx = local_y + y0, local_x + x0
    components, component_count = ndimage.label(local)
    component_areas = np.bincount(components.ravel())[1:]
    recoverable = bool(component_count == 1 or (len(component_areas) and component_areas.max() / len(xx) >= 0.98))
    boundary = local & ~ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
    by, bx = np.nonzero(boundary)
    by, bx = by + y0, bx + x0
    if len(bx) > 512:
        take = np.linspace(0, len(bx) - 1, 512).astype(int)
        by, bx = by[take], bx[take]
    coords = np.stack([yy, xx], axis=1).astype(np.float64)
    centered = coords - coords.mean(0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(coords) - 1)
    eigvals = np.maximum(np.linalg.eigvalsh(covariance), 1e-6)
    local_width = float(max(2.0, 4.0 * math.sqrt(float(eigvals.min()))))
    eroded = ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
    ey, ex = np.nonzero(eroded if eroded.any() else local)
    return {
        "value": int(value),
        "area": int(len(xx)),
        "centroid_y": float(yy.mean()),
        "centroid_x": float(xx.mean()),
        "boundary": np.stack([by, bx], axis=1),
        "support_y": ey + y0,
        "support_x": ex + x0,
        "width": local_width,
        "recoverable": recoverable,
    }


def instance_records(label: np.ndarray, min_area: int) -> dict[int, dict[str, Any]]:
    result = {}
    objects = ndimage.find_objects(label)
    for value, component_slice in enumerate(objects, start=1):
        if component_slice is None:
            continue
        record = local_instance_record(label, int(value), component_slice, min_area)
        if record is not None:
            result[int(value)] = record
    return result


def nearest_boundary(a: dict[str, Any], b: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    pa, pb = a["boundary"], b["boundary"]
    tree = cKDTree(pb)
    distances, indices = tree.query(pa, k=1)
    idx = int(np.argmin(distances))
    return pa[idx], pb[int(indices[idx])], float(distances[idx])


def capsule_coords(p0: np.ndarray, p1: np.ndarray, radius: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    yy = np.rint(np.linspace(p0[0], p1[0], steps)).astype(int)
    xx = np.rint(np.linspace(p0[1], p1[1], steps)).astype(int)
    y0, y1 = max(0, yy.min() - radius), min(shape[0], yy.max() + radius + 1)
    x0, x1 = max(0, xx.min() - radius), min(shape[1], xx.max() + radius + 1)
    local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    local[yy - y0, xx - x0] = True
    local = ndimage.binary_dilation(local, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))
    cy, cx = np.nonzero(local)
    return cy + y0, cx + x0


def xray_score(image: np.ndarray, label: np.ndarray, a: dict[str, Any], b: dict[str, Any], ay: np.ndarray, ax: np.ndarray) -> dict[str, float]:
    gap = label[ay, ax] == 0
    if not gap.any():
        return {"q_xray": 0.0, "q_valley": 0.0, "q_double": 0.0}
    gap_values = image[ay[gap], ax[gap]].astype(np.float64)
    ia = image[a["support_y"], a["support_x"]].astype(np.float64)
    ib = image[b["support_y"], b["support_x"]].astype(np.float64)
    bone_a = float(np.median(ia))
    bone_b = float(np.median(ib))
    gap_i = float(np.median(gap_values))
    local_values = np.concatenate([ia, ib, gap_values])
    sigma = float(np.median(np.abs(local_values - np.median(local_values))) * 1.4826 + 1.0)
    da, db = bone_a - gap_i, bone_b - gap_i
    same_direction = float(da * db > 0)
    valley = same_direction * float(np.clip(min(abs(da), abs(db)) / (abs(bone_a - bone_b) + sigma), 0.0, 1.0))
    double = same_direction * float(np.clip(math.sqrt(abs(da * db)) / sigma, 0.0, 1.0))
    return {"q_xray": valley * double, "q_valley": valley, "q_double": double}


def rotate_label(label: np.ndarray, degrees: float) -> np.ndarray:
    return np.asarray(
        Image.fromarray(label.astype(np.uint8), mode="L").rotate(
            degrees,
            resample=Image.Resampling.NEAREST,
            expand=False,
            fillcolor=0,
        )
    )


def pair_gap(records: dict[int, dict[str, Any]], i: int, j: int) -> float | None:
    if i not in records or j not in records:
        return None
    return max(0.0, nearest_boundary(records[i], records[j])[2] - 1.0)


def effective_sample_size(ages: np.ndarray, males: np.ndarray, age: float, male: bool, bandwidth: float = 24.0) -> float:
    weights = np.exp(-0.5 * ((ages - age) / bandwidth) ** 2) * np.where(males == male, 1.0, 0.25)
    return float(weights.sum() ** 2 / (np.square(weights).sum() + 1e-9))


def main() -> None:
    args = parse_args()
    if args.dataset != "TSRS_RSNA-Epiphysis" or "Articular" in args.dataset:
        raise RuntimeError("R255 Gate A only permits TSRS_RSNA-Epiphysis")
    root = Path(args.raw_root) / args.dataset
    label_dir, image_dir = root / "train_labels", root / "train"
    metadata = read_metadata(Path(args.metadata_csv))
    label_paths = sorted(label_dir.glob("*.png"))
    if args.max_cases > 0:
        label_paths = label_paths[: args.max_cases]
    rows: list[dict[str, Any]] = []
    case_rows = []
    total_instances = recoverable_instances = 0
    missing_metadata = []
    for case_index, label_path in enumerate(label_paths, start=1):
        case = label_path.stem
        if case not in metadata:
            missing_metadata.append(case)
            continue
        label = np.asarray(Image.open(label_path))
        if label.ndim == 3:
            label = label[..., 0]
        image = np.asarray(Image.open(find_image(image_dir, case)).convert("L"))
        if image.shape != label.shape:
            image = np.asarray(Image.fromarray(image).resize(label.shape[::-1], Image.Resampling.BILINEAR))
        records = instance_records(label, args.min_area)
        rotated_minus = instance_records(rotate_label(label, -args.rotate_deg), args.min_area)
        rotated_plus = instance_records(rotate_label(label, args.rotate_deg), args.min_area)
        total_instances += len(records)
        recoverable_instances += sum(int(r["recoverable"]) for r in records.values())
        if len(records) < 2:
            continue
        values = sorted(records)
        coords = np.asarray([[records[v]["centroid_y"], records[v]["centroid_x"]] for v in values])
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        candidate_pairs = set()
        for idx, value in enumerate(values):
            nearest = [
                int(other_idx)
                for other_idx in np.argsort(distances[idx])
                if int(other_idx) != idx and np.isfinite(distances[idx, int(other_idx)])
            ]
            for other_idx in nearest[: args.neighbors]:
                candidate_pairs.add(tuple(sorted((value, values[int(other_idx)]))))
        fg_y, fg_x = np.nonzero(label > 0)
        bbox_y0, bbox_y1 = float(fg_y.min()), float(fg_y.max())
        age, male = metadata[case]
        accepted = 0
        for i, j in sorted(candidate_pairs):
            a, b = records[i], records[j]
            if not a["recoverable"] or not b["recoverable"]:
                continue
            p0, p1, boundary_distance = nearest_boundary(a, b)
            gap_px = max(0.0, boundary_distance - 1.0)
            local_width = min(a["width"], b["width"])
            relative_gap = gap_px / max(local_width, 1e-6)
            midpoint_v = (((a["centroid_y"] + b["centroid_y"]) * 0.5) - bbox_y0) / max(1.0, bbox_y1 - bbox_y0)
            if midpoint_v < args.roi_v_min or gap_px > args.max_gap_px or relative_gap > args.max_relative_gap:
                continue
            radius = max(1, int(round(args.capsule_rho * local_width)))
            cy, cx = capsule_coords(p0, p1, radius, label.shape)
            capsule_values = label[cy, cx]
            third_instance = bool(np.any((capsule_values > 0) & (capsule_values != i) & (capsule_values != j)))
            legal_anchor_count = int(np.sum(capsule_values == 0))
            if third_instance or legal_anchor_count == 0:
                continue
            aug_gap_minus = pair_gap(rotated_minus, i, j)
            aug_gap_plus = pair_gap(rotated_plus, i, j)
            augmentation_survival = float(
                aug_gap_minus is not None
                and aug_gap_plus is not None
                and aug_gap_minus >= 1.0
                and aug_gap_plus >= 1.0
            )
            xray = xray_score(image, label, a, b, cy, cx)
            row = {
                "case": case,
                "boneage": age,
                "male": male,
                "instance_i": i,
                "instance_j": j,
                "midpoint_v": midpoint_v,
                "gap_px": gap_px,
                "relative_gap": relative_gap,
                "local_width": local_width,
                "capsule_radius": radius,
                "legal_anchor_count": legal_anchor_count,
                "third_instance_crossing": third_instance,
                "augmentation_survival": augmentation_survival,
                "aug_gap_minus": aug_gap_minus,
                "aug_gap_plus": aug_gap_plus,
                **xray,
                "xray_pass": float(xray["q_xray"] >= args.xray_floor),
            }
            rows.append(row)
            accepted += 1
        case_rows.append({"case": case, "boneage": age, "male": male, "instances": len(records), "accepted_pairs": accepted})
        if case_index % 50 == 0:
            print(f"[R255 Gate A] {case_index}/{len(label_paths)} cases pairs={len(rows)}", flush=True)

    if not rows:
        raise RuntimeError("No Gate A pair rows generated")
    ages = np.asarray([metadata[p.stem][0] for p in label_paths if p.stem in metadata])
    males = np.asarray([metadata[p.stem][1] for p in label_paths if p.stem in metadata], dtype=bool)
    for row in rows:
        row["support_ess"] = effective_sample_size(ages, males, float(row["boneage"]), bool(row["male"]))
        gap = float(row["gap_px"])
        row["gap_bin"] = "1-2" if 1 <= gap <= 2.999 else "3-4" if gap <= 4.999 else "5-8" if gap <= 8.999 else ">8" if gap > 8.999 else "<1"

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def mean_where(key: str, predicate) -> float:
        values = [float(r[key]) for r in rows if predicate(r)]
        return float(np.mean(values)) if values else 0.0

    close_rows = [r for r in rows if 1 <= float(r["gap_px"]) <= 4.999]
    narrow_rows = [r for r in rows if 1 <= float(r["gap_px"]) <= 2.999]
    age_q = np.quantile(ages, [0.25, 0.75])
    low = [r for r in rows if float(r["boneage"]) <= age_q[0]]
    high = [r for r in rows if float(r["boneage"]) >= age_q[1]]
    gate_checks = {
        "metadata_complete": len(missing_metadata) == 0,
        "instance_recoverability_ge_095": recoverable_instances / max(total_instances, 1) >= 0.95,
        "close_gap_pairs_ge_100": len(close_rows) >= 100,
        "close_gap_cases_ge_50": len({r["case"] for r in close_rows}) >= 50,
        "narrow_aug_survival_ge_060": mean_where("augmentation_survival", lambda r: r in narrow_rows) >= 0.60,
        "close_xray_pass_ge_030": mean_where("xray_pass", lambda r: r in close_rows) >= 0.30,
        "third_instance_crossing_zero": not any(bool(r["third_instance_crossing"]) for r in rows),
        "minimum_support_ess_ge_30": min(float(r["support_ess"]) for r in rows) >= 30.0,
        "high_age_has_close_pairs": sum(1 <= float(r["gap_px"]) <= 4.999 for r in high) > 0,
    }
    payload = {
        "run_id": "R255_GATE_A",
        "scope": "TSRS_RSNA-Epiphysis original train only",
        "clean_test_used": False,
        "num_cases_requested": len(label_paths),
        "num_cases_with_rows": len({r["case"] for r in rows}),
        "num_pairs": len(rows),
        "num_close_gap_pairs_1_4px": len(close_rows),
        "num_close_gap_cases_1_4px": len({r["case"] for r in close_rows}),
        "num_narrow_pairs_1_2px": len(narrow_rows),
        "instance_recoverability": recoverable_instances / max(total_instances, 1),
        "narrow_augmentation_survival": mean_where("augmentation_survival", lambda r: r in narrow_rows),
        "close_xray_pass_rate": mean_where("xray_pass", lambda r: r in close_rows),
        "close_q_xray_mean": mean_where("q_xray", lambda r: r in close_rows),
        "minimum_support_ess": min(float(r["support_ess"]) for r in rows),
        "gap_bin_counts": dict(Counter(str(r["gap_bin"]) for r in rows)),
        "age_quartiles_months": age_q.tolist(),
        "low_age_pair_count": len(low),
        "high_age_pair_count": len(high),
        "low_age_close_pair_count": sum(1 <= float(r["gap_px"]) <= 4.999 for r in low),
        "high_age_close_pair_count": sum(1 <= float(r["gap_px"]) <= 4.999 for r in high),
        "gate_checks": gate_checks,
        "gate_pass": all(gate_checks.values()),
        "config": vars(args),
        "notes": "All thresholds were pre-registered before reading Gate A results. Rejected third-instance corridors are not emitted as usable rows.",
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
