#!/usr/bin/env python3
"""Compare local correction distributions for different anchor mask domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze anchor error/correction distribution mismatch.")
    parser.add_argument("--cases-json", required=True, help="JSON list of cases with label, dataset, split, raw_root, anchor_exp.")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--radii", default="2,4,8,12,16")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def resize_like(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if mask.shape == gt.shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0


def find_anchor(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"anchor not found: exp={exp} dataset={dataset} split={split} name={name}")


def component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask)
    return int(n)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def summarize(records: list[dict[str, float]]) -> dict[str, object]:
    if not records:
        return {}
    keys = [k for k, v in records[0].items() if isinstance(v, (int, float, np.integer, np.floating))]
    return {key: mean_std([float(r[key]) for r in records]) for key in keys}


def analyze_case(case: dict[str, str], roots: list[Path], radii: list[int], source_dataset: str | None, boundary_kernel: int) -> dict[str, object]:
    raw_root = Path(case.get("raw_root", "data/raw_variants"))
    dataset = case["dataset"]
    split = case.get("split", "test")
    anchor_exp = case["anchor_exp"]
    label = case["label"]
    gt_dir = raw_root / dataset / f"{split}_labels"
    names = [p.name for p in sorted(gt_dir.glob("*.png"))]
    records: list[dict[str, float]] = []

    for name in tqdm(names, desc=label):
        gt = read_mask(gt_dir / name)
        anchor = resize_like(read_mask(find_anchor(roots, anchor_exp, dataset, split, name, source_dataset)), gt)
        metrics = compute_metrics(anchor, gt, boundary_kernel)
        fp = np.logical_and(anchor, ~gt)
        fn = np.logical_and(~anchor, gt)
        error = fp | fn
        rec: dict[str, float] = {
            **metrics,
            "anchor_area_frac": float(anchor.mean()),
            "gt_area_frac": float(gt.mean()),
            "fp_frac": float(fp.mean()),
            "fn_frac": float(fn.mean()),
            "error_frac": float(error.mean()),
            "fp_share_of_error": safe_div(float(fp.sum()), float(error.sum())),
            "fn_share_of_error": safe_div(float(fn.sum()), float(error.sum())),
            "pred_component_count": float(component_count(anchor)),
            "gt_component_count": float(component_count(gt)),
            "component_count_error": float(abs(component_count(anchor) - component_count(gt))),
            "false_bridge_flag": float(component_count(anchor) < component_count(gt) and anchor.sum() >= gt.sum() * 0.90),
        }
        for radius in radii:
            band = boundary_band(anchor, radius)
            band_area = float(band.sum())
            rec[f"r{radius}_band_frac"] = float(band.mean())
            rec[f"r{radius}_error_capture"] = safe_div(float(np.logical_and(error, band).sum()), float(error.sum()))
            rec[f"r{radius}_correction_prevalence"] = safe_div(float(np.logical_and(error, band).sum()), band_area)
            rec[f"r{radius}_fp_share_in_band_error"] = safe_div(float(np.logical_and(fp, band).sum()), float(np.logical_and(error, band).sum()))
            rec[f"r{radius}_fn_share_in_band_error"] = safe_div(float(np.logical_and(fn, band).sum()), float(np.logical_and(error, band).sum()))
        rec["image"] = name  # type: ignore[assignment]
        records.append(rec)

    return {
        "label": label,
        "dataset": dataset,
        "split": split,
        "anchor_exp": anchor_exp,
        "num_images": len(records),
        "summary": summarize(records),
        "per_image": records,
    }


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    radii = [int(x) for x in args.radii.split(",") if x.strip()]
    cases = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
    results = [analyze_case(case, roots, radii, args.source_dataset, args.boundary_kernel) for case in cases]
    output = {"radii": radii, "cases": results}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {
        case["label"]: {
            "dice": case["summary"]["dice"]["mean"],
            "precision": case["summary"]["precision"]["mean"],
            "recall": case["summary"]["recall"]["mean"],
            "boundary_iou": case["summary"]["boundary_iou"]["mean"],
            "fp_share_of_error": case["summary"]["fp_share_of_error"]["mean"],
            "fn_share_of_error": case["summary"]["fn_share_of_error"]["mean"],
            "r4_correction_prevalence": case["summary"].get("r4_correction_prevalence", {}).get("mean"),
            "r4_fp_share_in_band_error": case["summary"].get("r4_fp_share_in_band_error", {}).get("mean"),
            "r4_fn_share_in_band_error": case["summary"].get("r4_fn_share_in_band_error", {}).get("mean"),
        }
        for case in results
    }
    print(json.dumps(compact, indent=2))
    print(f"Saved anchor-domain mismatch analysis: {out}")


if __name__ == "__main__":
    main()
