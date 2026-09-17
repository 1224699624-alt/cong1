#!/usr/bin/env python3
"""Build the train-only demographic relation prior used by R254.

The artifact contains low-dimensional pair measurements, not masks. Validation
and test labels are deliberately rejected so that the prior cannot absorb the
evaluation split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--label-dir", default="data/raw/TSRS_RSNA-Epiphysis/train_labels")
    p.add_argument("--metadata-csv", default="G:/gutou/filtered_train.csv")
    p.add_argument("--output", default="outputs/analysis/r254_hapdsp_train_prior.json")
    p.add_argument("--neighbors-per-component", type=int, default=2)
    p.add_argument("--min-area", type=int, default=24)
    return p.parse_args()


def read_metadata(path: Path) -> dict[str, dict[str, float | bool]]:
    rows: dict[str, dict[str, float | bool]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows[str(row["id"])] = {
                "boneage": float(row["boneage"]),
                "male": str(row["male"]).strip().lower() in {"1", "true", "yes"},
            }
    return rows


def instances(label: np.ndarray, min_area: int) -> list[dict[str, float]]:
    result = []
    for value in np.unique(label):
        if value == 0:
            continue
        yy, xx = np.nonzero(label == value)
        if len(xx) < min_area:
            continue
        result.append({"x": float(xx.mean()), "y": float(yy.mean()), "area": float(len(xx))})
    return result


def pair_features(a: dict[str, float], b: dict[str, float], height: int, width: int) -> list[float]:
    diag = math.hypot(height, width)
    dx = abs(a["x"] - b["x"])
    dy = abs(a["y"] - b["y"])
    distance = math.hypot(dx, dy)
    radius_a = math.sqrt(a["area"] / math.pi)
    radius_b = math.sqrt(b["area"] / math.pi)
    approximate_gap = max(0.0, distance - radius_a - radius_b)
    return [
        distance / diag,
        dx / diag,
        dy / diag,
        abs(math.log((a["area"] + 1.0) / (b["area"] + 1.0))),
        approximate_gap / diag,
        ((a["x"] + b["x"]) * 0.5) / width,
        ((a["y"] + b["y"]) * 0.5) / height,
    ]


def main() -> None:
    args = parse_args()
    label_dir = Path(args.label_dir)
    if "train_labels" not in label_dir.as_posix() or "clean-test" in label_dir.as_posix().lower():
        raise RuntimeError("R254 prior may only be fitted from the original train_labels directory")
    metadata = read_metadata(Path(args.metadata_csv))
    samples: list[dict[str, object]] = []
    used_cases: set[str] = set()
    missing_metadata: list[str] = []
    for path in sorted(label_dir.glob("*.png")):
        case = path.stem
        if case not in metadata:
            missing_metadata.append(case)
            continue
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        comps = instances(arr, args.min_area)
        if len(comps) < 2:
            continue
        coords = np.asarray([[c["x"], c["y"]] for c in comps], dtype=np.float64)
        distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        pairs: set[tuple[int, int]] = set()
        for i in range(len(comps)):
            for j in np.argsort(distances[i])[: args.neighbors_per_component]:
                pairs.add(tuple(sorted((i, int(j)))))
        for i, j in sorted(pairs):
            samples.append(
                {
                    "case": case,
                    "boneage": metadata[case]["boneage"],
                    "male": metadata[case]["male"],
                    "features": pair_features(comps[i], comps[j], arr.shape[0], arr.shape[1]),
                }
            )
        used_cases.add(case)

    if not samples:
        raise RuntimeError("No train-only relation samples were generated")
    features = np.asarray([s["features"] for s in samples], dtype=np.float64)
    payload = {
        "run_id": "R254",
        "method": "HA-PDSP train-only demographic relation prior",
        "feature_names": [
            "distance_diag",
            "abs_dx_diag",
            "abs_dy_diag",
            "abs_log_area_ratio",
            "approx_gap_diag",
            "midpoint_x",
            "midpoint_y",
        ],
        "source_label_dir": str(label_dir),
        "source_metadata_csv": str(args.metadata_csv),
        "num_cases": len(used_cases),
        "num_pair_samples": len(samples),
        "missing_metadata_count": len(missing_metadata),
        "missing_metadata": missing_metadata,
        "global_feature_mean": features.mean(0).tolist(),
        "global_feature_std": np.maximum(features.std(0), 1e-4).tolist(),
        "samples": samples,
        "leakage_guard": "Only original train labels and filtered_train.csv are used; no val/test/clean-test labels.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("run_id", "num_cases", "num_pair_samples", "missing_metadata_count")}, indent=2))


if __name__ == "__main__":
    main()
