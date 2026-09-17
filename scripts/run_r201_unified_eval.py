#!/usr/bin/env python3
"""R201 unified evaluation protocol for epiphysis segmentation.

This script freezes the paper-facing metric protocol for clean-test-v2:
standard overlap metrics, boundary metrics, surface-distance metrics, and
task-specific anatomy consistency diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage


DATASET = "TSRS_RSNA-Epiphysis_clean_test_v2"
SPLIT = "test"
EXPECTED_N = 81

R200_ANCHORS = {
    "r110_r100_r108_patch_basic": {
        "dice": 0.9177231563529792,
        "iou": 0.8485699269403064,
        "boundary_iou": 0.25189601044085763,
    },
    "araa_danet_epoch98": {
        "dice": 0.9117660066557425,
        "iou": 0.8384360930797206,
        "boundary_iou": 0.2352651559145258,
    },
}

MASK_EXPERIMENT_LABELS = {
    "araa_danet_epoch98": "ARAA DANet epoch98",
    "r090_online_patch_disagreement_arbitrator": "R090 patch arbitrator",
    "r100_r095b_r097_patch_basic": "R100 patch complement",
    "r110_r100_r108_patch_basic": "R110 current best",
    "r130_dinov3_bridge_suppressed_instance_sep": "R130 bridge-suppressed source",
    "r143_highres_medical_recipe_unet": "R143 high-res medical U-Net recipe",
    "r165_filtered_reannotated_dinov3_instance_sep": "R165 filtered reannotation source",
    "r202_nnunet2d": "R202 nnU-Net 2D baseline",
}

METRICS_ONLY = {
    "r177_boundary_preserving_separation_arbitrator": "R177 boundary-preserving arbitrator",
    "r179_boundary_gap_constrained_dinov3_instance_sep": "R179 boundary-gap constrained source",
    "r182_topology_cldice_smoke": "R182 topology clDice smoke",
    "r184_r110_guarded_arbitrator_smoke": "R184 R110 guarded arbitrator smoke",
    "r186_fast_neck_gate_smoke": "R186 fast-neck gate smoke",
}

METRIC_DIRECTIONS = {
    "dsc": "higher",
    "nsd_2px": "higher",
    "voe": "lower",
    "msd_px": "lower",
    "ravd": "lower",
    "dice": "higher",
    "iou": "higher",
    "precision": "higher",
    "recall": "higher",
    "specificity": "higher",
    "boundary_iou": "higher",
    "boundary_f1": "higher",
    "surface_dice_2px": "higher",
    "surface_dice_5px": "higher",
    "hd95_px": "lower",
    "assd_px": "lower",
    "gap_region_fp_rate": "lower",
    "gap_region_precision": "higher",
    "component_merge_rate": "lower",
    "component_count_mae": "lower",
    "component_delta_mean": "closer_to_zero",
}


@dataclass
class Experiment:
    slug: str
    label: str
    pred_dir: Path | None
    metrics_json: Path | None
    evidence_level: str


@dataclass
class GtCache:
    image: str
    mask: np.ndarray
    boundary: np.ndarray
    surface: np.ndarray
    surface_distance: np.ndarray | None
    gap_region: np.ndarray
    component_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R201 unified clean-test-v2 evaluation.")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--variants-root", default="outputs/ablations_variants")
    parser.add_argument("--analysis-dir", default="outputs/analysis")
    parser.add_argument("--report-dir", default="research-workflow/refine-logs")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tolerance", type=float, default=2.0)
    parser.add_argument("--surface-tolerance-extra", type=float, default=5.0)
    parser.add_argument("--anchor-tolerance", type=float, default=1e-9)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def read_binary_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def boundary(mask: np.ndarray, kernel: int = 3) -> np.ndarray:
    pil_mask = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(pil_mask.filter(ImageFilter.MaxFilter(kernel))) > 0
    eroded = np.asarray(pil_mask.filter(ImageFilter.MinFilter(kernel))) > 0
    return np.logical_xor(dilated, eroded)


def safe_divide(numer: float, denom: float, default: float = 1.0) -> float:
    return float(numer / denom) if denom > 0 else default


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


def surface_distances(src_surface: np.ndarray, dst_surface: np.ndarray) -> np.ndarray:
    if not src_surface.any() or not dst_surface.any():
        return np.array([], dtype=np.float64)
    dist = ndimage.distance_transform_edt(~dst_surface)
    return dist[src_surface].astype(np.float64)


def surface_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    tol_a: float,
    tol_b: float,
    gt_surface_cached: np.ndarray | None = None,
    gt_surface_distance_cached: np.ndarray | None = None,
) -> dict[str, float | None]:
    pred_surface = surface(pred)
    gt_surface = gt_surface_cached if gt_surface_cached is not None else surface(gt)

    if not pred_surface.any() and not gt_surface.any():
        return {
            "surface_dice_2px": 1.0,
            "surface_dice_5px": 1.0,
            "hd95_px": 0.0,
            "assd_px": 0.0,
        }
    if not pred_surface.any() or not gt_surface.any():
        return {
            "surface_dice_2px": 0.0,
            "surface_dice_5px": 0.0,
            "hd95_px": None,
            "assd_px": None,
        }

    if gt_surface_distance_cached is not None and gt_surface.any():
        pred_to_gt = gt_surface_distance_cached[pred_surface].astype(np.float64)
    else:
        pred_to_gt = surface_distances(pred_surface, gt_surface)
    gt_to_pred = surface_distances(gt_surface, pred_surface)
    both = np.concatenate([pred_to_gt, gt_to_pred])

    def surface_dice(tol: float) -> float:
        numer = float((pred_to_gt <= tol).sum() + (gt_to_pred <= tol).sum())
        denom = float(len(pred_to_gt) + len(gt_to_pred))
        return safe_divide(numer, denom, default=1.0)

    return {
        "surface_dice_2px": surface_dice(tol_a),
        "surface_dice_5px": surface_dice(tol_b),
        "hd95_px": float(np.percentile(both, 95)),
        "assd_px": float(np.mean(both)),
    }


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    boundary_kernel: int,
    gap_kernel: int,
    surface_tol: float,
    surface_tol_extra: float,
    gt_cache: GtCache | None = None,
) -> dict[str, float | None]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())

    pred_boundary = boundary(pred, boundary_kernel)
    gt_boundary = gt_cache.boundary if gt_cache is not None else boundary(gt, boundary_kernel)
    b_tp = float(np.logical_and(pred_boundary, gt_boundary).sum())
    b_fp = float(np.logical_and(~gt_boundary, pred_boundary).sum())
    b_fn = float(np.logical_and(gt_boundary, ~pred_boundary).sum())
    boundary_union = float(np.logical_or(pred_boundary, gt_boundary).sum())

    gt_components, _ = ndimage.label(gt)
    pred_components, _ = ndimage.label(pred)
    gt_n = gt_cache.component_count if gt_cache is not None else int(gt_components.max())
    pred_n = int(pred_components.max())

    if gt_cache is not None:
        gap_region = gt_cache.gap_region
    else:
        gap_structure = np.ones((gap_kernel, gap_kernel), dtype=bool)
        gap_region = np.logical_and(ndimage.binary_dilation(gt, structure=gap_structure), ~gt)
    gap_pixels = float(gap_region.sum())
    gap_fp = float(np.logical_and(pred, gap_region).sum())
    gap_region_fp_rate = safe_divide(gap_fp, gap_pixels, default=0.0)

    row: dict[str, float | None] = {
        "dice": safe_divide(2.0 * tp, 2.0 * tp + fp + fn),
        "iou": safe_divide(tp, tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "boundary_iou": safe_divide(b_tp, boundary_union),
        "boundary_f1": safe_divide(2.0 * b_tp, 2.0 * b_tp + b_fp + b_fn, default=0.0),
        "gap_region_fp_rate": gap_region_fp_rate,
        "gap_region_precision": 1.0 - gap_region_fp_rate,
        "component_merge_rate": float(pred_n < gt_n),
        "component_count_mae": float(abs(pred_n - gt_n)),
        "component_delta_mean": float(pred_n - gt_n),
    }
    row.update(
        surface_metrics(
            pred,
            gt,
            surface_tol,
            surface_tol_extra,
            gt_surface_cached=gt_cache.surface if gt_cache is not None else None,
            gt_surface_distance_cached=gt_cache.surface_distance if gt_cache is not None else None,
        )
    )
    # Paper-aligned names are additive aliases; frozen R201 legacy fields remain intact.
    row["dsc"] = row["dice"]
    row["nsd_2px"] = row["surface_dice_2px"]
    row["voe"] = None if row["iou"] is None else 1.0 - float(row["iou"])
    row["msd_px"] = row["assd_px"]
    gt_area = float(gt.sum())
    row["ravd"] = 0.0 if gt_area == 0 else abs(float(pred.sum()) - gt_area) / (gt_area + 1e-8)
    return row


def build_gt_cache(gt_dir: Path, boundary_kernel: int, gap_kernel: int) -> dict[str, GtCache]:
    cache: dict[str, GtCache] = {}
    gap_structure = np.ones((gap_kernel, gap_kernel), dtype=bool)
    for gt_path in sorted(gt_dir.glob("*.png")):
        gt = read_binary_mask(gt_path)
        gt_boundary = boundary(gt, boundary_kernel)
        gt_surface = surface(gt)
        gt_surface_distance = ndimage.distance_transform_edt(~gt_surface) if gt_surface.any() else None
        gt_gap = np.logical_and(ndimage.binary_dilation(gt, structure=gap_structure), ~gt)
        gt_components, _ = ndimage.label(gt)
        cache[gt_path.name] = GtCache(
            image=gt_path.name,
            mask=gt,
            boundary=gt_boundary,
            surface=gt_surface,
            surface_distance=gt_surface_distance,
            gap_region=gt_gap,
            component_count=int(gt_components.max()),
        )
    return cache


def mean_metrics(records: list[dict[str, Any]]) -> tuple[dict[str, float | None], dict[str, int]]:
    metrics = sorted({key for record in records for key in record if key != "image"})
    out: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for metric in metrics:
        values = [record.get(metric) for record in records]
        valid = [float(value) for value in values if value is not None]
        counts[metric] = len(valid)
        out[metric] = float(mean(valid)) if valid else None
    return out, counts


def load_legacy_metrics(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": data.get("dataset"),
        "split": data.get("split"),
        "num_evaluated": int(data.get("num_evaluated", len(data.get("per_image", [])))),
        "num_missing": int(data.get("num_missing", 0)),
        "missing": data.get("missing", []),
        "mean": data.get("mean", {}),
        "per_image": data.get("per_image", []),
    }


def evaluate_mask_experiment(exp: Experiment, args: argparse.Namespace, gt_cache: dict[str, GtCache]) -> dict[str, Any]:
    assert exp.pred_dir is not None
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for image, cached in sorted(gt_cache.items()):
        pred_path = exp.pred_dir / image
        if not pred_path.exists():
            missing.append(image)
            continue
        gt = cached.mask
        pred = read_binary_mask(pred_path)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0
        row = compute_metrics(
            pred=pred,
            gt=gt,
            boundary_kernel=args.boundary_kernel,
            gap_kernel=args.gap_kernel,
            surface_tol=args.surface_tolerance,
            surface_tol_extra=args.surface_tolerance_extra,
            gt_cache=cached,
        )
        row["image"] = image
        records.append(row)
    means, valid_counts = mean_metrics(records)
    return {
        "run_id": "R201",
        "experiment_slug": exp.slug,
        "experiment_label": exp.label,
        "dataset": args.dataset,
        "split": args.split,
        "evidence_level": exp.evidence_level,
        "protocol": protocol_metadata(args),
        "num_evaluated": len(records),
        "num_missing": len(missing),
        "missing": missing,
        "mean": means,
        "mean_valid_counts": valid_counts,
        "per_image": records,
    }


def protocol_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": "R201",
        "raw_root": args.raw_root,
        "boundary_kernel": args.boundary_kernel,
        "gap_kernel": args.gap_kernel,
        "surface_tolerance_px": args.surface_tolerance,
        "surface_tolerance_extra_px": args.surface_tolerance_extra,
        "pixel_spacing": "not_available; all surface distances are reported in pixels",
        "clean_test_v2_rule": "final evaluation/diagnostic only; no threshold tuning",
        "metric_directions": METRIC_DIRECTIONS,
    }


def discover_experiments(args: argparse.Namespace) -> list[Experiment]:
    variants_root = Path(args.variants_root)
    analysis_dir = Path(args.analysis_dir)
    experiments: list[Experiment] = []
    for slug, label in MASK_EXPERIMENT_LABELS.items():
        pred_dir = variants_root / slug / args.dataset / args.split / "masks"
        metrics_json = analysis_dir / f"{slug}_clean_test_v2_metrics.json"
        experiments.append(
            Experiment(
                slug=slug,
                label=label,
                pred_dir=pred_dir if pred_dir.exists() else None,
                metrics_json=metrics_json if metrics_json.exists() else None,
                evidence_level="full_mask_recomputed" if pred_dir.exists() else "legacy_metrics_only",
            )
        )
    for slug, label in METRICS_ONLY.items():
        metrics_json = analysis_dir / f"{slug}_clean_test_v2_metrics.json"
        experiments.append(
            Experiment(
                slug=slug,
                label=label,
                pred_dir=None,
                metrics_json=metrics_json if metrics_json.exists() else None,
                evidence_level="legacy_metrics_only",
            )
        )
    return experiments


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def row_for_table(result: dict[str, Any]) -> dict[str, Any]:
    mean_values = result.get("mean", {})
    row = {
        "model_id": result["experiment_slug"],
        "experiment_slug": result["experiment_slug"],
        "experiment_label": result["experiment_label"],
        "evidence_level": result["evidence_level"],
        "num_evaluated": result.get("num_evaluated"),
        "num_missing": result.get("num_missing"),
    }
    for key in METRIC_DIRECTIONS:
        row[key] = mean_values.get(key)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "experiment_slug",
        "experiment_label",
        "evidence_level",
        "num_evaluated",
        "num_missing",
        *METRIC_DIRECTIONS.keys(),
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def regression_checks(results: dict[str, dict[str, Any]], tolerance: float) -> dict[str, Any]:
    checks = []
    ok = True
    for slug, expected_metrics in R200_ANCHORS.items():
        actual = results.get(slug, {}).get("mean", {})
        for metric, expected in expected_metrics.items():
            value = actual.get(metric)
            delta = None if value is None else abs(float(value) - expected)
            passed = delta is not None and delta <= tolerance
            ok = ok and passed
            checks.append(
                {
                    "experiment_slug": slug,
                    "metric": metric,
                    "expected": expected,
                    "actual": value,
                    "abs_delta": delta,
                    "tolerance": tolerance,
                    "passed": passed,
                }
            )
    r110 = results.get("r110_r100_r108_patch_basic", {}).get("mean", {})
    araa = results.get("araa_danet_epoch98", {}).get("mean", {})
    ranking_checks = {
        "r110_dice_gt_araa": r110.get("dice") is not None and araa.get("dice") is not None and r110["dice"] > araa["dice"],
        "r110_iou_gt_araa": r110.get("iou") is not None and araa.get("iou") is not None and r110["iou"] > araa["iou"],
        "r110_boundary_iou_gt_araa": r110.get("boundary_iou") is not None
        and araa.get("boundary_iou") is not None
        and r110["boundary_iou"] > araa["boundary_iou"],
        "r110_component_merge_worse_than_araa": r110.get("component_merge_rate") is not None
        and araa.get("component_merge_rate") is not None
        and r110["component_merge_rate"] > araa["component_merge_rate"],
    }
    ok = ok and all(ranking_checks.values())
    return {"passed": ok, "metric_checks": checks, "ranking_checks": ranking_checks}


def collect_hard_cases(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    r110 = {row["image"]: row for row in results.get("r110_r100_r108_patch_basic", {}).get("per_image", [])}
    araa = {row["image"]: row for row in results.get("araa_danet_epoch98", {}).get("per_image", [])}
    rows = []
    for image in sorted(set(r110) & set(araa)):
        r110_row = r110[image]
        araa_row = araa[image]
        score = 0.0
        score += float(r110_row.get("component_merge_rate", 0.0)) - float(araa_row.get("component_merge_rate", 0.0))
        score += 0.25 * (float(r110_row.get("component_count_mae", 0.0)) - float(araa_row.get("component_count_mae", 0.0)))
        score += 2.0 * (float(r110_row.get("gap_region_fp_rate", 0.0)) - float(araa_row.get("gap_region_fp_rate", 0.0)))
        if score > 0:
            rows.append(
                {
                    "image": image,
                    "hard_case_score": score,
                    "r110_dice": r110_row.get("dice"),
                    "araa_dice": araa_row.get("dice"),
                    "r110_component_merge_rate": r110_row.get("component_merge_rate"),
                    "araa_component_merge_rate": araa_row.get("component_merge_rate"),
                    "r110_component_count_mae": r110_row.get("component_count_mae"),
                    "araa_component_count_mae": araa_row.get("component_count_mae"),
                    "r110_gap_region_fp_rate": r110_row.get("gap_region_fp_rate"),
                    "araa_gap_region_fp_rate": araa_row.get("gap_region_fp_rate"),
                }
            )
    return sorted(rows, key=lambda row: float(row["hard_case_score"]), reverse=True)


def write_markdown_table(rows: list[dict[str, Any]], metrics: list[str], limit: int = 20) -> str:
    header = ["System", "Evidence", "N", *metrics]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[:limit]:
        cells = [row["experiment_label"], row["evidence_level"], str(row["num_evaluated"])]
        for metric in metrics:
            value = row.get(metric)
            cells.append("" if value is None else f"{float(value):.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_reports(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    regression: dict[str, Any],
    hard_cases: list[dict[str, Any]],
    output_paths: dict[str, str],
) -> None:
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    full_rows = [row for row in rows if row["evidence_level"] == "full_mask_recomputed"]
    full_rows_by_dice = sorted(full_rows, key=lambda row: float(row.get("dice") or -1), reverse=True)
    full_rows_by_component = sorted(full_rows, key=lambda row: float(row.get("component_merge_rate") or 1e9))

    protocol = f"""# R201 Unified Evaluation Protocol Freeze

## Purpose

Freeze the paper-facing evaluation protocol for `TSRS_RSNA-Epiphysis_clean_test_v2/test`.
The target claim is not generic Dice chasing: the method must preserve Dice/IoU while improving boundary quality and bone-gap/component separation diagnostics.

## Frozen Data

- Dataset: `{args.dataset}`
- Split: `{args.split}`
- Ground truth: `{args.raw_root}/{args.dataset}/{args.split}_labels`
- Expected images: `{EXPECTED_N}`
- Rule: clean-test-v2 is final evaluation/diagnostic only; no threshold search, model selection, or tuning is allowed on this split.

## Metric Layers

- Main table standard metrics: Dice, IoU/Jaccard, Precision, Recall.
- Boundary metrics: Boundary IoU, Boundary F1, Surface Dice at `2 px`, Surface Dice at `5 px`, HD95 in pixels, ASSD in pixels.
- Task-specific diagnostics: gap-region FP rate, component merge rate, component count MAE.

## Implementation Details

- Binary mask rule: any pixel value `> 0` is foreground.
- Resize rule: prediction masks with mismatched shape are resized to GT shape with nearest-neighbor interpolation.
- Boundary IoU and Boundary F1 use the existing project-compatible XOR boundary with kernel `{args.boundary_kernel}`.
- Gap region is `dilation(GT, {args.gap_kernel}x{args.gap_kernel}) - GT`.
- Surface metrics use pixel spacing because reliable physical spacing is unavailable.
- Empty-surface convention: if both prediction and GT are empty, Surface Dice/HD95/ASSD are perfect; if only one is empty, Surface Dice is `0` and HD95/ASSD are `null`.

## Metric Direction

```json
{json.dumps(METRIC_DIRECTIONS, indent=2)}
```

## Regression Anchor

R200 R110-vs-ARAA is the fixed regression anchor. The R201 script must reproduce R110/ARAA Dice, IoU, and Boundary IoU before its comparisons are trusted.

- Regression passed: `{regression["passed"]}`
- Unified table: `{output_paths["comparison_csv"]}`
- Unified JSON: `{output_paths["comparison_json"]}`
"""
    (report_dir / "R201_UNIFIED_EVAL_PROTOCOL_FREEZE.md").write_text(protocol, encoding="utf-8")

    report = f"""# R201 Unified Evaluation Report

## Regression Status

- R200 anchor regression passed: `{regression["passed"]}`
- R110 remains above ARAA on Dice/IoU/Boundary IoU, while its component/gap diagnostics remain worse. This preserves the earlier interpretation: R110 is a strong overlap/boundary anchor but has not solved bone-gap adhesion.

## Main Table Candidates

{write_markdown_table(full_rows_by_dice, ["dice", "iou", "precision", "recall", "boundary_iou", "boundary_f1", "surface_dice_2px", "hd95_px", "assd_px"])}

## Anatomy Diagnostic Ranking

{write_markdown_table(full_rows_by_component, ["component_merge_rate", "component_count_mae", "gap_region_fp_rate", "dice", "iou"])}

## Hard-Case Visualization Queue

Top R110-vs-ARAA disagreement cases for qualitative panels are saved in `{output_paths["hard_cases_csv"]}`. These are the first cases to inspect when showing adhesion/merge failures.

## Current Claim Audit

- Supported now: R110 improves overlap and boundary metrics over ARAA.
- Not supported yet: R110 solves bone-gap adhesion/component merging over ARAA.
- Required next evidence: a method or postprocessor that keeps Dice/IoU at least at ARAA and ideally R110 level while lowering component merge rate, component count MAE, and gap-region FP rate.

## Strong Baseline Queue

- First priority: nnU-Net trained only on train/val, with clean-test-v2 inference after checkpoint/threshold lock.
- Secondary baselines after nnU-Net: Swin UNETR or MedNeXt if compute and conversion costs are acceptable; MaskDINO binary-union remains experimental because earlier train/val gates were not yet a clean-test-v2-ready baseline.
"""
    (report_dir / "R201_UNIFIED_EVALUATION_REPORT.md").write_text(report, encoding="utf-8")

    citation = """# R201 Metric Citation And Reviewer-Risk Note

## Standard Metrics

- Dice, IoU/Jaccard, Precision, and Recall are conventional segmentation/classification overlap metrics and are acceptable for the main table.
- Boundary IoU is a boundary-focused segmentation metric from Cheng et al., CVPR 2021.
- Surface Dice was introduced for clinically oriented medical image segmentation evaluation by Nikolov et al. and is appropriate for boundary/surface tolerance claims.
- HD95 and ASSD are common medical image segmentation surface-distance metrics; report them as pixel distances unless reliable physical spacing is available.
- Metrics Reloaded recommends problem-aware metric choice in biomedical image analysis, supporting the use of overlap plus boundary/surface plus task-specific diagnostics.

## Task-Specific Diagnostics

- `gap_region_fp_rate`, `component_merge_rate`, and `component_count_mae` must be described as failure-mode diagnostics for bone-gap adhesion and anatomical consistency.
- Do not present these diagnostics as universal benchmark metrics.
- In the paper, define the exact formula, kernel radius, foreground rule, and low/high direction next to the table or in the appendix.

## Verified Sources

- Boundary IoU: Cheng et al., "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation", CVPR 2021, https://arxiv.org/abs/2103.16562
- Surface Dice: Nikolov et al., "Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy", https://arxiv.org/abs/1809.04430
- Metrics Reloaded: Maier-Hein et al., "Metrics reloaded: recommendations for image analysis validation", Nature Methods, https://www.nature.com/articles/s41592-023-02151-z
- nnU-Net baseline: Isensee et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation", Nature Methods, https://www.nature.com/articles/s41592-020-01008-z
"""
    (report_dir / "R201_METRIC_CITATION_AND_REVIEWER_RISK.md").write_text(citation, encoding="utf-8")

    main_table = f"""# R201 Paper Main-Table Recommendation

## Recommended Main Table

Use Dice, IoU, Precision, Recall, Boundary IoU, Boundary F1, Surface Dice 2px, HD95, and ASSD.

{write_markdown_table(full_rows_by_dice, ["dice", "iou", "precision", "recall", "boundary_iou", "boundary_f1", "surface_dice_2px", "hd95_px", "assd_px"], limit=12)}

## Recommended Diagnostic Table

Keep this separate from the main table and label it as anatomical consistency / failure-mode diagnostics.

{write_markdown_table(full_rows_by_component, ["gap_region_fp_rate", "component_merge_rate", "component_count_mae"], limit=12)}
"""
    (report_dir / "R201_PAPER_TABLE_RECOMMENDATION.md").write_text(main_table, encoding="utf-8")

    manifest = report_dir / "MANIFEST.md"
    manifest_lines = [
        "# Refine Logs Manifest\n\n" if not manifest.exists() else "",
        "- R201_UNIFIED_EVAL_PROTOCOL_FREEZE.md: frozen evaluation protocol.\n",
        "- R201_UNIFIED_EVALUATION_REPORT.md: unified evaluation summary and claim audit.\n",
        "- R201_METRIC_CITATION_AND_REVIEWER_RISK.md: metric citation and reviewer-risk note.\n",
        "- R201_PAPER_TABLE_RECOMMENDATION.md: paper table design recommendation.\n",
    ]
    with manifest.open("a", encoding="utf-8") as handle:
        handle.writelines(manifest_lines)


def write_hard_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "hard_case_score",
        "r110_dice",
        "araa_dice",
        "r110_component_merge_rate",
        "araa_component_merge_rate",
        "r110_component_count_mae",
        "araa_component_count_mae",
        "r110_gap_region_fp_rate",
        "araa_gap_region_fp_rate",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def self_test() -> None:
    gt = np.zeros((16, 16), dtype=bool)
    gt[4:12, 4:12] = True
    perfect = gt.copy()
    empty = np.zeros_like(gt)
    full = np.ones_like(gt)
    shifted = np.zeros_like(gt)
    shifted[6:14, 6:14] = True
    cases = {
        "perfect": perfect,
        "empty": empty,
        "full": full,
        "shifted": shifted,
    }
    for name, pred in cases.items():
        metrics = compute_metrics(pred, gt, 3, 9, 2.0, 5.0)
        if name == "perfect":
            assert abs(float(metrics["dice"]) - 1.0) < 1e-12
            assert abs(float(metrics["iou"]) - 1.0) < 1e-12
            assert abs(float(metrics["surface_dice_2px"]) - 1.0) < 1e-12
            assert abs(float(metrics["hd95_px"]) - 0.0) < 1e-12
        if name == "empty":
            assert metrics["dice"] == 0.0
            assert metrics["surface_dice_2px"] == 0.0
            assert metrics["hd95_px"] is None
        if name == "full":
            assert float(metrics["precision"]) < 1.0
        if name == "shifted":
            assert 0.0 < float(metrics["dice"]) < 1.0
    print("R201 synthetic metric self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return

    gt_dir = Path(args.raw_root) / args.dataset / f"{args.split}_labels"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth label dir not found: {gt_dir}")
    gt_cache = build_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)

    analysis_dir = Path(args.analysis_dir)
    r201_dir = analysis_dir / "r201_unified_eval"
    experiments = discover_experiments(args)
    results_by_slug: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []

    for exp in experiments:
        if exp.pred_dir is not None:
            result = evaluate_mask_experiment(exp, args, gt_cache)
        elif exp.metrics_json is not None:
            legacy = load_legacy_metrics(exp.metrics_json)
            result = {
                "run_id": "R201",
                "experiment_slug": exp.slug,
                "experiment_label": exp.label,
                "dataset": legacy.get("dataset") or args.dataset,
                "split": legacy.get("split") or args.split,
                "evidence_level": exp.evidence_level,
                "protocol": protocol_metadata(args),
                "num_evaluated": legacy["num_evaluated"],
                "num_missing": legacy["num_missing"],
                "missing": legacy["missing"],
                "mean": legacy["mean"],
                "mean_valid_counts": {},
                "per_image": legacy["per_image"],
            }
        else:
            result = {
                "run_id": "R201",
                "experiment_slug": exp.slug,
                "experiment_label": exp.label,
                "dataset": args.dataset,
                "split": args.split,
                "evidence_level": "missing",
                "protocol": protocol_metadata(args),
                "num_evaluated": 0,
                "num_missing": EXPECTED_N,
                "missing": [],
                "mean": {},
                "mean_valid_counts": {},
                "per_image": [],
            }
        results_by_slug[exp.slug] = result
        table_rows.append(row_for_table(result))
        write_json(r201_dir / f"{exp.slug}_unified_metrics.json", result)

    comparison = {
        "run_id": "R201",
        "dataset": args.dataset,
        "split": args.split,
        "protocol": protocol_metadata(args),
        "rows": table_rows,
    }
    comparison_json = r201_dir / "r201_unified_comparison_table.json"
    comparison_csv = r201_dir / "r201_unified_comparison_table.csv"
    write_json(comparison_json, comparison)
    write_csv(comparison_csv, table_rows)

    regression = regression_checks(results_by_slug, args.anchor_tolerance)
    regression_json = r201_dir / "r201_r200_regression_check.json"
    write_json(regression_json, regression)

    hard_cases = collect_hard_cases(results_by_slug)
    hard_cases_csv = r201_dir / "r201_r110_vs_araa_hard_case_queue.csv"
    hard_cases_json = r201_dir / "r201_r110_vs_araa_hard_case_queue.json"
    write_hard_cases_csv(hard_cases_csv, hard_cases)
    write_json(hard_cases_json, hard_cases)

    output_paths = {
        "comparison_json": str(comparison_json),
        "comparison_csv": str(comparison_csv),
        "regression_json": str(regression_json),
        "hard_cases_csv": str(hard_cases_csv),
        "hard_cases_json": str(hard_cases_json),
    }
    write_reports(args, table_rows, regression, hard_cases, output_paths)
    write_json(r201_dir / "r201_output_manifest.json", output_paths)

    print(json.dumps({"regression_passed": regression["passed"], "outputs": output_paths}, indent=2, ensure_ascii=False))
    if not regression["passed"]:
        raise SystemExit("R201 regression check failed; do not trust comparison tables until fixed.")


if __name__ == "__main__":
    main()
