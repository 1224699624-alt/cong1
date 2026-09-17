#!/usr/bin/env python3
"""Analyze whether candidate-mask uncertainty aligns with anchor errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose uncertainty/error alignment for mask candidates.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--candidate-exps", nargs="+", required=True)
    parser.add_argument("--uncertainty-thresholds", default="0.25,0.50,0.75")
    parser.add_argument("--band-radii", default="1,2,4,8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def find_mask(
    roots: list[Path],
    exp: str,
    dataset: str,
    split: str,
    name: str,
    source_dataset: str | None,
) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"mask not found: exp={exp} dataset={dataset} split={split} name={name}")


def resize_like(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if mask.shape == gt.shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def fast_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return safe_div(2.0 * tp, 2.0 * tp + fp + fn)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.zeros_like(mask, dtype=bool)
    k = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=k) ^ ndimage.binary_erosion(mask, structure=k)


def analyze_one(name: str, gt: np.ndarray, anchor: np.ndarray, candidates: list[np.ndarray], thresholds: list[float], radii: list[int], boundary_kernel: int) -> dict[str, object]:
    stack = np.stack(candidates, axis=0).astype(np.float32)
    vote = stack.mean(axis=0)
    uncertainty = 1.0 - np.abs(2.0 * vote - 1.0)
    error = anchor ^ gt
    fp = anchor & ~gt
    fn = ~anchor & gt
    union = stack.max(axis=0).astype(bool)
    inter = stack.min(axis=0).astype(bool)
    disagreement = union ^ inter
    can_fix = (fp & ~inter) | (fn & union)

    rec: dict[str, object] = {
        "image": name,
        "anchor_dice": fast_dice(anchor, gt),
        "error_frac": float(error.mean()),
        "fp_error_frac": float(fp.mean()),
        "fn_error_frac": float(fn.mean()),
        "disagreement_frac": float(disagreement.mean()),
        "error_in_disagreement_share": safe_div(float((error & disagreement).sum()), float(error.sum())),
        "disagreement_precision_for_error": safe_div(float((error & disagreement).sum()), float(disagreement.sum())),
        "candidate_can_fix_error_share": safe_div(float(can_fix.sum()), float(error.sum())),
        "candidate_fix_precision": safe_div(float(can_fix.sum()), float(disagreement.sum())),
    }

    for threshold in thresholds:
        region = uncertainty >= threshold
        rec[f"unc_ge_{threshold:.2f}_frac"] = float(region.mean())
        rec[f"unc_ge_{threshold:.2f}_error_capture"] = safe_div(float((error & region).sum()), float(error.sum()))
        rec[f"unc_ge_{threshold:.2f}_error_precision"] = safe_div(float((error & region).sum()), float(region.sum()))
        rec[f"unc_ge_{threshold:.2f}_fixable_capture"] = safe_div(float((can_fix & region).sum()), float(error.sum()))

    for radius in radii:
        band = boundary_band(anchor, radius)
        region = band & disagreement
        rec[f"r{radius}_band_error_capture"] = safe_div(float((error & band).sum()), float(error.sum()))
        rec[f"r{radius}_band_disagree_frac"] = float(region.mean())
        rec[f"r{radius}_band_disagree_error_capture"] = safe_div(float((error & region).sum()), float(error.sum()))
        rec[f"r{radius}_band_disagree_error_precision"] = safe_div(float((error & region).sum()), float(region.sum()))

    # Diagnostic oracle: only modify anchor at candidate-disagreement pixels using GT.
    oracle = anchor.copy()
    oracle[disagreement] = gt[disagreement]
    rec["disagreement_pixel_oracle_dice"] = fast_dice(oracle, gt)
    rec["anchor_metrics"] = compute_metrics(anchor, gt, boundary_kernel)
    return rec


def mean_numeric(records: list[dict[str, object]]) -> dict[str, float]:
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float, np.integer, np.floating))]
    return {k: float(np.mean([float(r[k]) for r in records])) for k in keys}


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    thresholds = parse_floats(args.uncertainty_thresholds)
    radii = parse_ints(args.band_radii)
    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    names = [p.name for p in sorted(gt_dir.glob("*.png"))]
    records: list[dict[str, object]] = []
    candidate_metrics: dict[str, list[float]] = {exp: [] for exp in args.candidate_exps}

    for name in tqdm(names, desc="uncertainty/error"):
        gt = read_mask(gt_dir / name)
        anchor = resize_like(read_mask(find_mask(roots, args.anchor_exp, args.dataset, args.split, name, args.source_dataset)), gt)
        candidates = []
        for exp in args.candidate_exps:
            mask = resize_like(read_mask(find_mask(roots, exp, args.dataset, args.split, name, args.source_dataset)), gt)
            candidates.append(mask)
            candidate_metrics[exp].append(fast_dice(mask, gt))
        records.append(analyze_one(name, gt, anchor, candidates, thresholds, radii, args.boundary_kernel))

    summary = mean_numeric(records)
    output = {
        "dataset": args.dataset,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "candidate_exps": args.candidate_exps,
        "num_images": len(records),
        "summary": summary,
        "candidate_dice": {exp: summarize(vals) for exp, vals in candidate_metrics.items()},
        "per_image": records,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {
        "anchor_dice": summary.get("anchor_dice"),
        "disagreement_frac": summary.get("disagreement_frac"),
        "error_in_disagreement_share": summary.get("error_in_disagreement_share"),
        "disagreement_precision_for_error": summary.get("disagreement_precision_for_error"),
        "candidate_can_fix_error_share": summary.get("candidate_can_fix_error_share"),
        "disagreement_pixel_oracle_dice": summary.get("disagreement_pixel_oracle_dice"),
    }
    print(json.dumps({"output": str(out), "compact": compact, "candidate_dice": output["candidate_dice"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
