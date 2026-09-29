#!/usr/bin/env python3
"""Targeted, read-only audit of the RAM-native R323 S/seam prior.

The audit deliberately does not run a model or touch RAM test data.  It
recovers provenance from the checked-in R323 scripts and existing artifacts,
then compares twelve deterministic train cases' saved maps with a full-image
GT gap proxy.  The proxy follows the R323 ``make_heat_target`` geometry but
uses GT member centroids/scales because saved full-image maps do not contain
pair provenance.  It is a source/quality diagnostic, not a segmentation
evaluation and not a claim of independent S value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path(r"G:\gutou\RAM-W600")
DEFAULT_MAP_ROOT = ROOT / "outputs" / "priors" / "r323_ram_native_r317_single_seed"
DEFAULT_OUT = ROOT / "outputs" / "ram_so_heatmap_20260929"
MAP_SIZE = 384
MAP_THRESHOLD = 0.5
GT_EPS = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_info(path: Path, root: Path | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if root is not None:
        try:
            value["relative_to_root"] = str(path.relative_to(root))
        except ValueError:
            pass
    if path.is_file():
        stat = path.stat()
        value.update({"bytes": int(stat.st_size), "sha256": sha256(path)})
    return value


def git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def canonical_masks(mask_path: Path, size: int = MAP_SIZE) -> np.ndarray:
    masks = (np.load(mask_path) > 0).astype(np.uint8)
    if masks.ndim != 3 or masks.shape[0] != 14:
        raise ValueError(f"Expected [14,H,W] RAM mask, got {masks.shape} at {mask_path}")
    # R323 canonical_image flips right-side cases before resizing.  The mask
    # tensors must follow the same transform for map/GT comparisons.
    if mask_path.stem.endswith("_R"):
        masks = np.flip(masks, axis=-1)
    resized = np.stack(
        [cv2.resize(channel, (size, size), interpolation=cv2.INTER_NEAREST) for channel in masks]
    )
    return np.ascontiguousarray(resized.astype(bool))


def boundary_gap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    boundary_a = mask_a.astype(np.uint8) - cv2.erode(mask_a.astype(np.uint8), np.ones((3, 3), np.uint8))
    boundary_b = mask_b.astype(np.uint8) - cv2.erode(mask_b.astype(np.uint8), np.ones((3, 3), np.uint8))
    if not boundary_a.any() or not boundary_b.any():
        return float("inf")
    points_a = np.argwhere(boundary_a > 0)
    points_b = np.argwhere(boundary_b > 0)
    distance, _ = cKDTree(points_b).query(points_a, k=1)
    return max(0.0, float(distance.min()) - 1.0)


def point_line_distance(size: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    ab = b - a
    denom = max(float(ab @ ab), 1e-6)
    t = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / denom, 0.0, 1.0)
    return np.sqrt((xx - (a[0] + t * ab[0])) ** 2 + (yy - (a[1] + t * ab[1])) ** 2)


def member_geometry(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.zeros(2, dtype=np.float64), 0.0, 0.0
    area = float(len(xx))
    center = np.asarray([float(xx.mean()), float(yy.mean())], dtype=np.float64)
    scale = max(3.0, math.sqrt(area / math.pi))
    return center, scale, area


def full_image_gt_gap_target(
    masks: np.ndarray,
    size: int = MAP_SIZE,
    pair_limit: float = 2.5,
    seam_relative_limit: float = 0.8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a conservative union of eligible local GT separation targets.

    R323 supervises a pair heatmap only for non-overlapping pairs with
    relative boundary gap <= 0.8.  At map audit time pair provenance is not
    stored, so this computes the same soft target in full-image coordinates
    using GT pair geometry, and reports the eligible-pair count.  It never
    enters inference.
    """
    foreground = masks.any(axis=0)
    target = np.zeros((size, size), dtype=np.float32)
    eligible: list[dict[str, Any]] = []
    overlap_pairs = 0
    near_pairs = 0
    for i in range(masks.shape[0]):
        center_a, scale_a, area_a = member_geometry(masks[i])
        if area_a <= 0:
            continue
        for j in range(i + 1, masks.shape[0]):
            center_b, scale_b, area_b = member_geometry(masks[j])
            if area_b <= 0:
                continue
            if np.logical_and(masks[i], masks[j]).any():
                overlap_pairs += 1
                continue
            distance = float(np.linalg.norm(center_a - center_b))
            relative_center = distance / max(scale_a + scale_b, 1e-6)
            if relative_center > pair_limit:
                continue
            near_pairs += 1
            gap = boundary_gap(masks[i], masks[j])
            relative_gap = gap / max(math.sqrt(0.5 * (area_a + area_b)), 1.0)
            if relative_gap > seam_relative_limit:
                continue

            distance_a = cv2.distanceTransform((~masks[i]).astype(np.uint8), cv2.DIST_L2, 5)
            distance_b = cv2.distanceTransform((~masks[j]).astype(np.uint8), cv2.DIST_L2, 5)
            radius = max(3.0, 0.55 * math.sqrt(0.5 * (area_a + area_b)))
            raw = np.exp(
                -np.abs(distance_a - distance_b) / (0.35 * radius + 1e-6)
                - (distance_a + distance_b) / (2.0 * radius + 1e-6)
            ).astype(np.float32)
            raw *= (~(masks[i] | masks[j])) & (distance_a <= radius) & (distance_b <= radius)
            proposal_scale = 0.5 * (scale_a + scale_b)
            corridor = point_line_distance(size, center_a, center_b) <= max(3.0, 0.45 * proposal_scale)
            raw *= (~foreground) & corridor
            raw[raw < GT_EPS] = 0.0
            target = np.maximum(target, raw)
            eligible.append(
                {
                    "pair": [i, j],
                    "center_distance_px": distance,
                    "relative_center": relative_center,
                    "boundary_gap_px": gap,
                    "relative_gap": relative_gap,
                    "target_pixels": int((raw > GT_EPS).sum()),
                    "target_mass": float(raw.sum()),
                }
            )
    return target, {
        "eligible_pairs": len(eligible),
        "near_nonoverlap_pairs": near_pairs,
        "overlap_pairs": overlap_pairs,
        "pairs": eligible,
        "target_pixels": int((target > GT_EPS).sum()),
        "target_mass": float(target.sum()),
    }


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    if positives == 0 or labels.size == positives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.float64)
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, len(sorted_labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def category_stats(score: np.ndarray, category: np.ndarray, thresholds: tuple[float, ...]) -> dict[str, Any]:
    category = np.asarray(category, dtype=bool)
    values = score[category]
    result: dict[str, Any] = {
        "pixels": int(category.sum()),
        "mean": safe_float(float(values.mean())) if values.size else None,
        "median": safe_float(float(np.median(values))) if values.size else None,
        "max": safe_float(float(values.max())) if values.size else None,
        "mass": safe_float(float(values.sum())) if values.size else None,
    }
    for threshold in thresholds:
        result[f"support_ge_{threshold:g}"] = int((values >= threshold).sum()) if values.size else 0
        result[f"rate_ge_{threshold:g}"] = (
            safe_float(float((values >= threshold).mean())) if values.size else None
        )
    return result


def map_metrics(score: np.ndarray, masks: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    foreground_count = masks.sum(axis=0)
    foreground = foreground_count > 0
    overlap = foreground_count >= 2
    ordinary_foreground = foreground & ~overlap
    gap = target > GT_EPS
    non_gap_background = ~foreground & ~gap
    thresholds = (0.2, 0.5, 0.8)
    background_labels = gap[~foreground]
    background_scores = score[~foreground]
    result: dict[str, Any] = {
        "shape": list(score.shape),
        "finite": bool(np.isfinite(score).all()),
        "range": [safe_float(float(score.min())), safe_float(float(score.max()))],
        "mean": safe_float(float(score.mean())),
        "quantiles": {
            str(q): safe_float(float(np.quantile(score, q))) for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
        "foreground_pixels": int(foreground.sum()),
        "ordinary_foreground_pixels": int(ordinary_foreground.sum()),
        "overlap_pixels": int(overlap.sum()),
        "background_pixels": int((~foreground).sum()),
        "gap_pixels": int(gap.sum()),
        "soft_gap_dice": safe_float(
            float((2.0 * (score * target).sum() + 1.0) / (score.sum() + target.sum() + 1.0))
        ),
        "background_average_precision": average_precision_binary(background_labels, background_scores),
        "categories": {
            "gap_gt": category_stats(score, gap, thresholds),
            "ordinary_foreground": category_stats(score, ordinary_foreground, thresholds),
            "overlap_foreground": category_stats(score, overlap, thresholds),
            "non_gap_background": category_stats(score, non_gap_background, thresholds),
            "all_background": category_stats(score, ~foreground, thresholds),
        },
        "threshold_diagnostics": {},
    }
    for threshold in thresholds:
        predicted = score >= threshold
        gap_tp = int((predicted & gap).sum())
        predicted_count = int(predicted.sum())
        gap_count = int(gap.sum())
        result["threshold_diagnostics"][str(threshold)] = {
            "predicted_pixels": predicted_count,
            "gap_precision": safe_float(gap_tp / max(predicted_count, 1)),
            "gap_recall": safe_float(gap_tp / max(gap_count, 1)),
            "non_gap_background_fp_rate": safe_float(float((predicted & non_gap_background).sum()) / max(int(non_gap_background.sum()), 1)),
            "ordinary_foreground_response_rate": safe_float(float((predicted & ordinary_foreground).sum()) / max(int(ordinary_foreground.sum()), 1)),
            "overlap_response_rate": safe_float(float((predicted & overlap).sum()) / max(int(overlap.sum()), 1)),
        }
    return result


def select_cases(map_paths: list[Path], count: int) -> list[Path]:
    if len(map_paths) <= count:
        return map_paths
    # Fixed, broad coverage over lexicographically sorted train stems.  The
    # exact stems are recorded in JSON/Markdown so the sample cannot drift.
    indices = np.rint(np.linspace(0, len(map_paths) - 1, count)).astype(int).tolist()
    return [map_paths[i] for i in indices]


def map_inventory(map_root: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for split in ("train", "val"):
        split_root = map_root / split
        npy_paths = sorted(split_root.glob("*.npy"))
        png_paths = sorted(split_root.glob("*.png"))
        shape_counts: dict[str, int] = {}
        bad: list[str] = []
        for path in npy_paths:
            try:
                shape = tuple(np.load(path, mmap_mode="r").shape)
            except Exception:
                shape = ("unreadable",)
            key = str(list(shape))
            shape_counts[key] = shape_counts.get(key, 0) + 1
            if shape != (MAP_SIZE, MAP_SIZE):
                bad.append(path.stem)
        inventory[split] = {
            "npy_count": len(npy_paths),
            "png_count": len(png_paths),
            "shape_counts": shape_counts,
            "bad_shapes": bad,
        }
    return inventory


def selected_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Macro aggregate for the declared audit sample, separate from val claims."""
    if not rows:
        return {}
    metric_keys = {
        "background_average_precision": lambda row: row["map_metrics"]["background_average_precision"],
        "soft_gap_dice": lambda row: row["map_metrics"]["soft_gap_dice"],
        "gap_precision_at_0p5": lambda row: row["map_metrics"]["threshold_diagnostics"]["0.5"]["gap_precision"],
        "gap_recall_at_0p5": lambda row: row["map_metrics"]["threshold_diagnostics"]["0.5"]["gap_recall"],
        "non_gap_background_fp_at_0p5": lambda row: row["map_metrics"]["threshold_diagnostics"]["0.5"]["non_gap_background_fp_rate"],
        "ordinary_foreground_response_at_0p5": lambda row: row["map_metrics"]["threshold_diagnostics"]["0.5"]["ordinary_foreground_response_rate"],
        "overlap_response_at_0p5": lambda row: row["map_metrics"]["threshold_diagnostics"]["0.5"]["overlap_response_rate"],
    }
    result: dict[str, Any] = {"cases": len(rows)}
    for key, getter in metric_keys.items():
        values = [getter(row) for row in rows]
        values = [float(value) for value in values if value is not None and np.isfinite(value)]
        result[key] = safe_float(float(np.mean(values))) if values else None
    result["total_gt_gap_pixels"] = int(sum(row["map_metrics"]["gap_pixels"] for row in rows))
    return result


def source_provenance(root: Path, dataset_root: Path, map_root: Path) -> dict[str, Any]:
    script_names = [
        "run_r323_ram_native_r317_remote.sh",
        "scripts/build_r323_ram_native_prior_maps.py",
        "scripts/build_r322_ram_r317_seam_prior_maps.py",
        "scripts/build_r259_frozen_prior_maps.py",
        "scripts/r316c_pair_inputs.py",
        "scripts/train_r323_ram_r317_prior.py",
        "scripts/train_r258b_prediction_relation_prior.py",
        "scripts/train_r256_scale_invariant_pair_prior.py",
        "scripts/train_r284_ramw600_overlap_prior.py",
    ]
    scripts = [file_info(root / name, root) for name in script_names]
    assets = [
        file_info(root / "outputs/priors/r323_ram_native_r317_single_seed/manifest.json", root),
        file_info(root / "outputs/analysis/r323_ram_native_r317_single_seed.json", root),
        file_info(root / "outputs/pair_prior/r323_ram_native_r317_single_seed/ram_image_centers_seed3231_selected.pt", root),
        file_info(root / "outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth", root),
        file_info(root / "outputs/ram_w600/r323_nnunet_ram_native_r317/result.json", root),
        file_info(root / "outputs/ram_w600/r323_nnunet_ram_native_r317/initialization_audit.json", root),
    ]
    manifest_path = map_root / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = root / "outputs/analysis/r323_ram_native_r317_single_seed.json"
    result: dict[str, Any] = {}
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    return {
        "project_root": str(root),
        "project_git_head": git_head(root),
        "dataset_root": str(dataset_root),
        "dataset_git_head": git_head(dataset_root),
        "scripts": scripts,
        "assets": assets,
        "map_manifest": manifest,
        "r323_result_summary": {
            key: result.get(key)
            for key in ("dataset", "test_used", "single_seed", "epochs_completed", "selected_epoch", "checkpoint_sha256", "config")
            if key in result
        },
        "r323_selected_metrics": result.get("selected"),
        "inference_semantics": {
            "status": "GT_FREE_IMAGE_PLUS_FROZEN_BASELINE_PROPOSALS",
            "model_input": "canonical grayscale image -> frozen R285 14-channel prediction -> predicted component centers/scales -> R323 relation prior",
            "gt_at_inference": False,
            "detector_or_segmenter_dependency": True,
            "raw_mask_files_used_only_for_case_enumeration": True,
            "explanation": "build_r323 reads mask filenames to enumerate cases but never loads mask values; model inputs are image and frozen baseline predictions. The relation prior is therefore GT-free at inference but is not raw-image-only because it depends on the frozen baseline proposal stage.",
        },
        "training_target_semantics": {
            "status": "GT_SUPERVISED_TRAINING_ONLY",
            "pair_label": "non-overlap pair with boundary gap/local scale <= 0.8",
            "heat_target": "make_heat_target uses pair masks, full foreground mask, distance transforms, corridor, and clears GT foreground after resize",
            "target_not_used_in_inference": True,
        },
        "normalization_and_geometry": {
            "image": "PIL grayscale, 600x600->384x384 bilinear, horizontal flip for *_R, then /255 and per-image mean/std normalization for frozen R285",
            "proposal": "sigmoid(R285 logits), threshold 0.5, largest connected component >=16 pixels per channel, value-weighted centroid, scale=sqrt(area/pi), nearest 3 pairs with distance/(scale_a+scale_b)<=2.5",
            "pair_input": "zero-padded crop; R316C high detail 256x256 /255, low context 128 then linear upsample to 256 /255, two Gaussian center maps with sigma=clip(0.15*mean_scale,3,12), features=zeros(2)",
            "map": "sigmoid(relation_logit)*sigmoid(heat_logit), bilinear resize to full 384 crop, max-paste; .npy stores float32 score and .png stores uint8 clip(score*255)",
            "audit_gt_proxy": "same target geometry in full 384 coordinates using GT centroids/scales, pair_limit=2.5 and seam_relative_limit=0.8; proxy is diagnostic because map files omit pair IDs/proposals",
        },
        "disk": disk_info(dataset_root),
    }


def disk_info(path: Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(path.anchor or path)
    except OSError:
        return {}
    return {"mount": path.anchor, "total_bytes": int(usage.total), "free_bytes": int(usage.free), "used_bytes": int(usage.used)}


def run_audit(dataset_root: Path, map_root: Path, out_root: Path, count: int) -> dict[str, Any]:
    train_mask_dir = dataset_root / "BoneSegmentation" / "masks" / "train"
    train_image_dir = dataset_root / "BoneSegmentation" / "images"
    map_dir = map_root / "train"
    train_masks = {path.stem: path for path in sorted(train_mask_dir.glob("*.npy"))}
    map_paths = sorted(map_dir.glob("*.npy"))
    if not train_masks:
        raise FileNotFoundError(f"No RAM train masks at {train_mask_dir}")
    if not map_paths:
        raise FileNotFoundError(f"No saved R323 maps at {map_dir}")
    selected = select_cases(map_paths, count)
    rows: list[dict[str, Any]] = []
    map_shape_counts: dict[str, int] = {}
    map_range_violations: list[str] = []
    png_missing: list[str] = []
    png_quantization_violations: list[str] = []
    for map_path in selected:
        stem = map_path.stem
        mask_path = train_masks.get(stem)
        if mask_path is None:
            raise FileNotFoundError(f"Map has no RAM train mask: {stem}")
        image_path = train_image_dir / f"{stem}.bmp"
        if not image_path.is_file():
            raise FileNotFoundError(f"Map has no RAM train image: {image_path}")
        score = np.asarray(np.load(map_path), dtype=np.float32)
        map_shape_counts[str(list(score.shape))] = map_shape_counts.get(str(list(score.shape)), 0) + 1
        if score.shape != (MAP_SIZE, MAP_SIZE) or not np.isfinite(score).all() or score.min() < -1e-6 or score.max() > 1.000001:
            map_range_violations.append(stem)
        png_path = map_path.with_suffix(".png")
        if not png_path.is_file():
            png_missing.append(stem)
        else:
            png = np.asarray(Image.open(png_path).convert("L"))
            expected = np.clip(score * 255.0, 0, 255).astype(np.uint8)
            if png.shape != expected.shape or not np.array_equal(png, expected):
                png_quantization_violations.append(stem)
        masks = canonical_masks(mask_path, MAP_SIZE)
        target, pair_audit = full_image_gt_gap_target(masks, MAP_SIZE)
        row = {
            "case": stem,
            "split": "train",
            "map": file_info(map_path, ROOT),
            "image": file_info(image_path, dataset_root),
            "mask": file_info(mask_path, dataset_root),
            "mask_original_shape": list(np.load(mask_path, mmap_mode="r").shape),
            "gt_geometry": {
                "member_count_nonempty": int(masks.any(axis=(1, 2)).sum()),
                "foreground_pixels": int(masks.any(axis=0).sum()),
                "ordinary_foreground_pixels": int((masks.sum(axis=0) == 1).sum()),
                "overlap_pixels": int((masks.sum(axis=0) >= 2).sum()),
            },
            "pair_target": pair_audit,
            "map_metrics": map_metrics(score, masks, target),
        }
        rows.append(row)

    all_train_maps = sorted(map_dir.glob("*.npy"))
    all_train_pngs = sorted(map_dir.glob("*.png"))
    inventory = map_inventory(map_root)
    summary = {
        "status": "RAM_S_SOURCE_RECOVERED_MAP_AUDITED",
        "s_value_status": "CONDITIONAL_VALUE_PENDING",
        "segmentation_gain_status": "NOT_RUN",
        "scope": {
            "dataset": "RAM-W600",
            "splits_read": ["train"],
            "test_used": False,
            "train_cases_available": len(train_masks),
            "saved_train_maps": len(all_train_maps),
            "saved_train_pngs": len(all_train_pngs),
            "selected_case_count": len(rows),
            "selection": "lexicographically sorted train map stems sampled at 12 evenly spaced indices; exact stems recorded below",
            "selected_cases": [row["case"] for row in rows],
        },
        "artifact_integrity": {
            "map_shape_counts_selected": map_shape_counts,
            "map_range_violations_selected": map_range_violations,
            "png_missing_selected": png_missing,
            "png_quantization_violations_selected": png_quantization_violations,
            "map_inventory_train_val": inventory,
            "all_map_npy_count": len(all_train_maps),
            "all_map_png_count": len(all_train_pngs),
            "manifest": file_info(map_root / "manifest.json", ROOT),
        },
        "source": source_provenance(ROOT, dataset_root, map_root),
        "selected_aggregate": selected_aggregate(rows),
        "cases": rows,
        "limitations": [
            "Saved full-image maps contain no pair IDs or detector proposals; the GT target is a declared full-image proxy using GT geometry, so per-pair reconstruction is unavailable.",
            "The twelve train cases are a source/quality audit sample, not a full train or validation metric and cannot establish segmentation improvement.",
            "Map score is a sigmoid product/max-paste score, not a calibrated probability; no per-image min-max normalization was applied.",
            "No TSRS checkpoint, TSRS score, test image, or test mask was used by this audit.",
        ],
    }
    return summary


def markdown_report(summary: dict[str, Any]) -> str:
    scope = summary["scope"]
    source = summary["source"]
    selected = source.get("r323_selected_metrics") or {}
    lines = [
        "# RAM R323 S source audit (2026-09-29)",
        "",
        "This is a read-only source and map-quality audit. It does not claim an independent S segmentation gain.",
        "",
        f"- Source status: **{summary['status']}**",
        f"- S value status: **{summary['s_value_status']}**",
        f"- Segmentation gain: **{summary['segmentation_gain_status']}**",
        f"- Scope: RAM-W600 train only; {scope['train_cases_available']} train masks and {scope['saved_train_maps']} saved maps; {scope['selected_case_count']} fixed audit cases.",
        f"- Selected cases: `{', '.join(scope['selected_cases'])}`",
        "- Test used: **no**; TSRS used: **no**; new training: **no**; GPU inference: **no**.",
        "",
        "## Recovered source and inference semantics",
        "",
        "R323 S is the four-channel `RelationPriorNet(4)` saved at `outputs/pair_prior/r323_ram_native_r317_single_seed/ram_image_centers_seed3231_selected.pt` (seed 3231, selected epoch 58). Its frozen proposal stage is the R285 RAM nnU-Net checkpoint `outputs/ram_w600/r285_nnunet_overlap_prior/baseline_best.pth`. The saved full-image maps are under `outputs/priors/r323_ram_native_r317_single_seed/{train,val}` and are 384×384 float32 `.npy` maps plus uint8 `.png` views.",
        "",
        "At inference, S is GT-free but proposal-dependent: the image is canonicalized and normalized, R285 produces 14-channel predictions, largest connected components provide centers/scales, and the R323 prior scores nearest pairs. The map builder enumerates mask filenames to select case stems but never loads GT mask values. Thus it is image plus a frozen baseline proposal stage, rather than a raw-image-only module. GT masks enter R323 training targets only.",
        "",
        f"Existing R323 prior training selected epoch {selected.get('epoch', 'NA')} (single seed 3231): pair AUROC {selected.get('auroc', float('nan')):.4f}, pair AP {selected.get('average_precision', float('nan')):.4f}, positive recall {selected.get('positive_recall', float('nan')):.4f}, close-pair recall {selected.get('close_pair_recall', float('nan')):.4f}, contact-pair recall {selected.get('contact_pair_recall', float('nan')):.4f}, negative specificity {selected.get('negative_specificity', float('nan')):.4f}, and positive heat soft Dice {selected.get('positive_heatmap_soft_dice', float('nan')):.4f}. These are prior-training diagnostics, not segmentation gain.",
        "",
        "The training target is not `1 - overlap`. For eligible non-overlap pairs it uses distance transforms, a radius from pair area, a center-line corridor, and explicit foreground clearing. The audit proxy applies that geometry in 384-pixel coordinates with GT centroids/scales, `pair_limit=2.5`, and `seam_relative_limit=0.8`; pair IDs are absent from saved maps, so this proxy is reported as such.",
        "",
        "## Exact assets and hashes",
        "",
        "| asset | exists | bytes | SHA-256 |",
        "|---|---:|---:|---|",
    ]
    for item in source["assets"]:
        lines.append(f"| `{item.get('relative_to_root', item['path'])}` | {item['exists']} | {item.get('bytes', '')} | `{item.get('sha256', '')}` |")
    lines.extend(["", "Source implementation files:", ""])
    for item in source["scripts"]:
        lines.append(f"- `{item.get('relative_to_root', item['path'])}` — `{item.get('sha256', '')}`")
    lines.extend([
        "",
        f"Project git head: `{source.get('project_git_head')}`; RAM-W600 git head: `{source.get('dataset_git_head')}`.",
        "",
        "## Normalization and dependency chain",
        "",
        "- Image: grayscale 600×600 → 384×384 bilinear; `_R` cases flip horizontally; R285 input is `/255`, then per-image mean/std normalization.",
        "- Proposal: sigmoid R285 logits, threshold 0.5, largest component per member channel with area ≥16; value-weighted centroid and `sqrt(area/pi)` scale; nearest 3 pairs with normalized distance ≤2.5.",
        "- Pair input: zero-padded crop; R316C high detail 256×256 and 128→256 context, both `/255`; two Gaussian center maps with σ=clip(0.15×mean scale, 3, 12); features are zeros(2).",
        "- Output: `sigmoid(relation) × sigmoid(heat)`, bilinear resized into the 384 crop and max-pasted. `.npy` remains the float32 score; `.png` is `clip(score×255)` for viewing.",
        "- Dependencies: R323 map builder → R259 crop helper/R316C inputs, R322 proposal geometry, R258B `RelationPriorNet`, R284 nnU-Net baseline, and R256 target geometry.",
        "",
        "## Fixed train-case audit",
        "",
        "The table reports actual saved maps against the declared GT gap proxy. `gap AP` is computed on background pixels only. `gap P/R@0.5` uses the binary proxy support (`target > 1e-4`); ordinary-FG and overlap rates are response diagnostics, not segmentation scores.",
        "",
        "| case | eligible GT pairs | gap px | map mean | gap AP | soft gap Dice | gap P/R@0.5 | non-gap BG FP@0.5 | ordinary FG@0.5 | overlap@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["cases"]:
        metric = row["map_metrics"]
        td = metric["threshold_diagnostics"]["0.5"]
        lines.append(
            f"| `{row['case']}` | {row['pair_target']['eligible_pairs']} | {metric['gap_pixels']} | {metric['mean']:.4f} | {metric['background_average_precision'] if metric['background_average_precision'] is not None else 'NA'} | {metric['soft_gap_dice']:.4f} | {td['gap_precision']:.3f}/{td['gap_recall']:.3f} | {td['non_gap_background_fp_rate']:.3f} | {td['ordinary_foreground_response_rate']:.3f} | {td['overlap_response_rate']:.3f} |"
        )
    aggregate = summary["selected_aggregate"]
    lines.append(
        f"| **macro mean ({aggregate['cases']})** | — | {aggregate['total_gt_gap_pixels']} | — | {aggregate['background_average_precision']:.3f} | {aggregate['soft_gap_dice']:.3f} | {aggregate['gap_precision_at_0p5']:.3f}/{aggregate['gap_recall_at_0p5']:.3f} | {aggregate['non_gap_background_fp_at_0p5']:.3f} | {aggregate['ordinary_foreground_response_at_0p5']:.3f} | {aggregate['overlap_response_at_0p5']:.3f} |"
    )
    lines.extend([
        "",
        "Interpretation: this audit can establish source recovery, coordinate/score semantics, and whether saved S responses concentrate in a declared local-gap proxy. It cannot establish a C0→CS segmentation delta. The correct status remains `CONDITIONAL_VALUE_PENDING` until a same-C0 segmentation comparison is run under the parent run lock.",
        "",
        "## Limitations",
        "",
    ])
    lines.extend(f"- {item}" for item in summary["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--map-root", type=Path, default=DEFAULT_MAP_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()
    if not 8 <= args.cases <= 12:
        raise ValueError("--cases must be between 8 and 12")
    summary = run_audit(args.dataset_root, args.map_root, args.output_root, args.cases)
    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "S_SOURCE_AUDIT.json"
    md_path = args.output_root / "S_SOURCE_AUDIT.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "cases": summary["scope"]["selected_cases"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
