#!/usr/bin/env python3
"""Audit paired label shifts between original and reannotated Epiphysis train/val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original and reannotated same-name labels.")
    parser.add_argument("--original", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--reannotated", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1"))
    parser.add_argument("--clean-test", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_clean_test_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/r163_reannotated_pair_shift.json"))
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) >= 4)


def mask_features(mask: np.ndarray) -> dict[str, float]:
    return {
        "fg_frac": float(mask.mean()),
        "component_count": float(component_count(mask)),
    }


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a & b).sum())
    den = float(a.sum() + b.sum())
    return 1.0 if den == 0 else 2.0 * inter / den


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


def label_stems(root: Path, split: str) -> set[str]:
    return {p.stem for p in (root / f"{split}_labels").glob("*.png")}


def paired_split(original: Path, reannotated: Path, split: str) -> dict[str, object]:
    orig_names = label_stems(original, split)
    re_names = label_stems(reannotated, split)
    shared = sorted(orig_names & re_names)
    extras = sorted(re_names - orig_names)
    missing = sorted(orig_names - re_names)
    records = []
    for stem in shared:
        orig = read_mask(original / f"{split}_labels" / f"{stem}.png")
        reann = read_mask(reannotated / f"{split}_labels" / f"{stem}.png")
        orig_feat = mask_features(orig)
        re_feat = mask_features(reann)
        rec = {
            "image": f"{stem}.png",
            "label_dice": dice(orig, reann),
            "orig_fg_frac": orig_feat["fg_frac"],
            "reann_fg_frac": re_feat["fg_frac"],
            "fg_frac_delta": re_feat["fg_frac"] - orig_feat["fg_frac"],
            "fg_frac_ratio": (re_feat["fg_frac"] / orig_feat["fg_frac"]) if orig_feat["fg_frac"] else 0.0,
            "orig_component_count": orig_feat["component_count"],
            "reann_component_count": re_feat["component_count"],
            "component_count_delta": re_feat["component_count"] - orig_feat["component_count"],
        }
        records.append(rec)
    summary = {
        "shared": len(shared),
        "extras_in_reannotated": len(extras),
        "missing_from_reannotated": len(missing),
        "label_dice": summarize([float(r["label_dice"]) for r in records]),
        "fg_frac_delta": summarize([float(r["fg_frac_delta"]) for r in records]),
        "fg_frac_ratio": summarize([float(r["fg_frac_ratio"]) for r in records]),
        "component_count_delta": summarize([float(r["component_count_delta"]) for r in records]),
        "low_agreement": [r for r in sorted(records, key=lambda row: float(row["label_dice"]))[:20]],
        "largest_area_increase": [r for r in sorted(records, key=lambda row: float(row["fg_frac_delta"]), reverse=True)[:20]],
        "largest_area_decrease": [r for r in sorted(records, key=lambda row: float(row["fg_frac_delta"]))[:20]],
        "largest_component_increase": [r for r in sorted(records, key=lambda row: float(row["component_count_delta"]), reverse=True)[:20]],
        "extras_sample": [f"{stem}.png" for stem in extras[:50]],
        "missing_sample": [f"{stem}.png" for stem in missing[:50]],
    }
    return {"summary": summary, "per_image": records}


def split_distribution(root: Path, split: str) -> dict[str, object]:
    records = []
    for path in sorted((root / f"{split}_labels").glob("*.png")):
        mask = read_mask(path)
        rec = {"image": path.name, **mask_features(mask)}
        records.append(rec)
    return {
        "count": len(records),
        "fg_frac": summarize([float(r["fg_frac"]) for r in records]),
        "component_count": summarize([float(r["component_count"]) for r in records]),
    }


def main() -> None:
    args = parse_args()
    report = {
        "status": "complete",
        "original": str(args.original),
        "reannotated": str(args.reannotated),
        "clean_test": str(args.clean_test),
        "paired": {
            "train": paired_split(args.original, args.reannotated, "train"),
            "val": paired_split(args.original, args.reannotated, "val"),
        },
        "distributions": {
            "original_train": split_distribution(args.original, "train"),
            "original_val": split_distribution(args.original, "val"),
            "reannotated_train": split_distribution(args.reannotated, "train"),
            "reannotated_val": split_distribution(args.reannotated, "val"),
            "clean_test_v2": split_distribution(args.clean_test, "test"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        split: {
            "shared": payload["summary"]["shared"],
            "extras": payload["summary"]["extras_in_reannotated"],
            "label_dice_mean": payload["summary"]["label_dice"]["mean"],
            "fg_frac_ratio_mean": payload["summary"]["fg_frac_ratio"]["mean"],
            "component_delta_mean": payload["summary"]["component_count_delta"]["mean"],
        }
        for split, payload in report["paired"].items()
    }
    print(json.dumps({"output": str(args.output), "compact": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
