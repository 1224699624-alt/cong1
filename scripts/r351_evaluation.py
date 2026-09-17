#!/usr/bin/env python3
"""Reusable R351 evaluation guards for TSRS and RAM.

The R350 training script uses teacher predictions when constructing its state
masks.  That is appropriate for a loss, but it makes its ``seam_fp`` value a
model-dependent diagnostic.  R351 keeps the evaluation ROI fixed: a
per-image, min-max-normalised seam prior is thresholded and intersected with
the background of the ground-truth union.  Predictions are used only to count
false positives inside that already-fixed ROI.

This module deliberately has no torch/scipy/project-model imports.  It accepts
NumPy arrays, Python sequences, and detached CPU/GPU torch tensors (through a
small duck-typed conversion helper), so it can be integrated into existing
RAM/TSRS collectors without importing a training stack in tests.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SEAM_THRESHOLD = 0.25
DEFAULT_STRICT_OVERLAP_TOLERANCE = 0.0
DEFAULT_NEAR_TIE_TOLERANCE = 0.0005
RESULT_SCHEMA = "R351_RESULT_V1"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _to_numpy(value: Any) -> np.ndarray:
    """Convert an array-like value, including a torch tensor, to NumPy.

    The conversion intentionally uses only duck typing.  Importing torch here
    would make the evaluator unnecessarily expensive and would make CPU unit
    tests depend on the project's training environment.
    """

    converted = value
    detach = getattr(converted, "detach", None)
    if callable(detach):
        converted = detach()
    cpu = getattr(converted, "cpu", None)
    if callable(cpu):
        converted = cpu()
    as_numpy = getattr(converted, "numpy", None)
    if callable(as_numpy):
        converted = as_numpy()
    return np.asarray(converted)


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"seam threshold must be finite and in [0, 1], got {threshold!r}")
    return threshold


def _collapse_spatial(
    value: Any,
    *,
    expected_shape: tuple[int, int] | None,
    name: str,
    reducer: str,
) -> np.ndarray:
    """Collapse a 2-D map or a channel-first/channel-last 3-D map to HxW."""

    array = _to_numpy(value)
    # A single leading batch dimension is common when a collector passes one
    # DataLoader item directly.  More than one image is intentionally rejected:
    # the public API is per-image and silently mixing cases is unsafe.
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        if expected_shape is not None and tuple(array.shape) != expected_shape:
            raise ValueError(f"{name} shape {tuple(array.shape)} != expected {expected_shape}")
        return array
    if array.ndim != 3:
        raise ValueError(
            f"{name} must be HxW, CxHxW, or HxWxC (one optional batch), got {array.shape}"
        )

    if expected_shape is not None and tuple(array.shape[1:]) == expected_shape:
        axis = 0  # C x H x W
    elif expected_shape is not None and tuple(array.shape[:2]) == expected_shape:
        axis = 2  # H x W x C
    elif expected_shape is None:
        # With no reference shape, prefer the conventional channel-first form
        # but recognise singleton/channel-small trailing dimensions too.  The
        # singleton checks also make tiny HxWx1 unit-test maps unambiguous.
        if array.shape[0] == 1:
            axis = 0
        elif array.shape[-1] == 1:
            axis = 2
        elif array.shape[0] <= min(array.shape[1:]):
            axis = 0
        elif array.shape[-1] <= min(array.shape[:2]):
            axis = 2
        else:
            axis = 0  # documented default for channel-first data
    else:
        raise ValueError(f"{name} shape {tuple(array.shape)} has no spatial match to {expected_shape}")

    if reducer == "max":
        return np.max(array, axis=axis)
    if reducer == "any":
        return np.any(array, axis=axis)
    raise ValueError(f"unsupported reducer {reducer!r}")


def _spatial_prior(seam_prior: Any) -> np.ndarray:
    """Return a floating point HxW seam prior, unioning optional channels."""

    prior = _collapse_spatial(
        seam_prior,
        expected_shape=None,
        name="seam_prior",
        reducer="max",
    ).astype(np.float64, copy=False)
    if prior.ndim != 2:
        raise ValueError(f"seam_prior did not collapse to HxW: {prior.shape}")
    return prior


def _foreground_union(
    value: Any,
    *,
    expected_shape: tuple[int, int],
    name: str,
    threshold: float,
) -> np.ndarray:
    """Convert binary/probability instance maps to one HxW foreground union."""

    raw = _collapse_spatial(
        value,
        expected_shape=expected_shape,
        name=name,
        reducer="any" if _to_numpy(value).dtype == np.bool_ else "max",
    )
    if raw.dtype == np.bool_:
        return raw
    return np.asarray(raw >= float(threshold), dtype=bool)


def _normalise_prior(prior: np.ndarray) -> np.ndarray:
    """Apply the per-image min-max normalisation used by R350 state masks.

    A constant or all-non-finite prior has no usable seam contrast, so its
    normalised map is all zero and produces an empty corridor at the default
    positive threshold.  Non-finite values are treated as the finite minimum.
    """

    finite = np.isfinite(prior)
    if not finite.any():
        return np.zeros_like(prior, dtype=np.float64)
    finite_values = prior[finite]
    lo = float(np.min(finite_values))
    hi = float(np.max(finite_values))
    safe = np.where(finite, prior, lo)
    if hi <= lo:
        return np.zeros_like(prior, dtype=np.float64)
    return np.clip((safe - lo) / (hi - lo), 0.0, 1.0)


def fixed_seam_background_roi(
    seam_prior: Any,
    gt_union: Any,
    *,
    seam_threshold: float = DEFAULT_SEAM_THRESHOLD,
    target_threshold: float = 0.5,
) -> np.ndarray:
    """Build the fixed RAM seam/background evaluation ROI.

    The ROI is exactly ``minmax(seam_prior) >= seam_threshold`` intersected
    with ``NOT(gt_union)``.  ``gt_union`` is the logical union over all GT
    instance channels.  No prediction, logits, teacher output, or refiner
    state is inspected by this function.

    The function is evaluation-only because it uses the GT union.  It returns
    a boolean HxW array and leaves all input arrays untouched.
    """

    threshold = _validate_threshold(seam_threshold)
    prior = _spatial_prior(seam_prior)
    shape = tuple(int(x) for x in prior.shape)
    union = _foreground_union(
        gt_union,
        expected_shape=shape,
        name="gt_union",
        threshold=target_threshold,
    )
    corridor = _normalise_prior(prior) >= threshold
    return np.logical_and(corridor, np.logical_not(union))


# A descriptive alias for callers that prefer a RAM-specific name.
build_ram_seam_background_roi = fixed_seam_background_roi


def _roi_checksum(roi: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(roi, dtype=np.uint8).tobytes()).hexdigest()


def _empty_policy() -> dict[str, str]:
    return {
        "per_image_fp_pixels": "0 when roi_area is zero",
        "per_image_fp_rate": "null when roi_area is zero",
        "aggregate_rate": "pooled FP pixels / pooled non-empty ROI area; empty images excluded",
        "comparison": "an empty ROI has no rate and is never treated as an improvement",
    }


def _row_for_prediction(
    prediction: Any,
    roi: np.ndarray,
    *,
    image: str,
    prediction_threshold: float,
    roi_checksum: str,
) -> dict[str, Any]:
    pred_union = _foreground_union(
        prediction,
        expected_shape=tuple(int(x) for x in roi.shape),
        name=f"prediction[{image}]",
        threshold=prediction_threshold,
    )
    roi_area = int(roi.sum())
    fp_pixels = int(np.logical_and(pred_union, roi).sum())
    fp_rate = None if roi_area == 0 else float(fp_pixels / roi_area)
    return {
        "image": image,
        "fp_pixels": fp_pixels,
        "roi_area": roi_area,
        "fp_rate": fp_rate,
        "fp_over_area": f"{fp_pixels}/{roi_area}",
        "prediction_union_area": int(pred_union.sum()),
        "empty_roi": roi_area == 0,
        "roi_checksum": roi_checksum,
    }


def _sorted_case_keys(*maps: Mapping[Any, Any]) -> list[Any]:
    if not maps:
        return []
    first = set(maps[0].keys())
    for index, mapping in enumerate(maps[1:], start=1):
        keys = set(mapping.keys())
        if keys != first:
            missing = sorted((str(key) for key in first - keys))
            extra = sorted((str(key) for key in keys - first))
            raise ValueError(f"case keys differ for map {index}: missing={missing}, extra={extra}")
    return sorted(first, key=str)


def aggregate_fixed_seam_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image fixed-ROI rows without diluting empty ROIs."""

    rows = list(rows)
    nonempty = [row for row in rows if int(row["roi_area"]) > 0]
    fp_pixels = int(sum(int(row["fp_pixels"]) for row in rows))
    roi_area = int(sum(int(row["roi_area"]) for row in rows))
    rates = [float(row["fp_rate"]) for row in nonempty if row.get("fp_rate") is not None]
    return {
        "num_images": len(rows),
        "nonempty_roi_images": len(nonempty),
        "empty_roi_images": len(rows) - len(nonempty),
        "fp_pixels": fp_pixels,
        "roi_area": roi_area,
        "fp_rate": None if roi_area == 0 else float(fp_pixels / roi_area),
        "mean_image_fp_rate": None if not rates else float(np.mean(rates)),
        "empty_roi_policy": _empty_policy(),
    }


def evaluate_fixed_seam_roi(
    predictions: Mapping[Any, Any],
    seam_priors: Mapping[Any, Any],
    targets: Mapping[Any, Any],
    *,
    seam_threshold: float = DEFAULT_SEAM_THRESHOLD,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate one prediction collection against fixed per-image RAM ROIs.

    ``predictions``, ``seam_priors``, and ``targets`` must contain exactly the
    same case IDs.  Every per-image result reports FP pixels and ROI area; the
    aggregate reports both pooled and mean-image FP rates.
    """

    case_keys = _sorted_case_keys(predictions, seam_priors, targets)
    rows: list[dict[str, Any]] = []
    for key in case_keys:
        image = str(key)
        roi = fixed_seam_background_roi(
            seam_priors[key],
            targets[key],
            seam_threshold=seam_threshold,
            target_threshold=target_threshold,
        )
        rows.append(
            _row_for_prediction(
                predictions[key],
                roi,
                image=image,
                prediction_threshold=prediction_threshold,
                roi_checksum=_roi_checksum(roi),
            )
        )
    return {
        "protocol": {
            "name": "R351_fixed_ram_seam_background_roi",
            "roi_definition": "minmax(seam_prior) >= seam_threshold AND NOT(GT union)",
            "seam_prior_is_fixed": True,
            "gt_union_background_is_fixed": True,
            "model_prediction_used_for_roi": False,
            "seam_threshold": float(seam_threshold),
            "prediction_threshold": float(prediction_threshold),
            "target_threshold": float(target_threshold),
            "empty_roi_policy": _empty_policy(),
        },
        "aggregate": aggregate_fixed_seam_rows(rows),
        "per_image": rows,
    }


def _delta_status(delta: float | None) -> str:
    if delta is None:
        return "undefined_empty_roi"
    if delta > 0:
        return "increased"
    if delta < 0:
        return "decreased"
    return "unchanged"


def _lower_is_better_change(baseline: float | None, candidate: float | None) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "delta_candidate_minus_baseline": None,
            "relative_improvement": None,
            "status": "undefined_empty_roi",
            "favorable": None,
        }
    delta = float(candidate - baseline)
    relative = None if baseline == 0 else float((baseline - candidate) / abs(baseline))
    return {
        "baseline": float(baseline),
        "candidate": float(candidate),
        "delta_candidate_minus_baseline": delta,
        "relative_improvement": relative,
        "status": "improved" if delta < 0 else ("degraded" if delta > 0 else "unchanged"),
        "favorable": delta <= 0,
    }


def compare_fixed_seam_roi(
    baseline_predictions: Mapping[Any, Any],
    adapted_predictions: Mapping[Any, Any],
    seam_priors: Mapping[Any, Any],
    targets: Mapping[Any, Any],
    *,
    seam_threshold: float = DEFAULT_SEAM_THRESHOLD,
    prediction_threshold: float = 0.5,
    target_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compare two models using exactly the same fixed seam/background ROIs."""

    _sorted_case_keys(baseline_predictions, adapted_predictions, seam_priors, targets)
    baseline = evaluate_fixed_seam_roi(
        baseline_predictions,
        seam_priors,
        targets,
        seam_threshold=seam_threshold,
        prediction_threshold=prediction_threshold,
        target_threshold=target_threshold,
    )
    adapted = evaluate_fixed_seam_roi(
        adapted_predictions,
        seam_priors,
        targets,
        seam_threshold=seam_threshold,
        prediction_threshold=prediction_threshold,
        target_threshold=target_threshold,
    )
    baseline_rows = {row["image"]: row for row in baseline["per_image"]}
    adapted_rows = {row["image"]: row for row in adapted["per_image"]}
    roi_invariant = all(
        baseline_rows[image]["roi_checksum"] == adapted_rows[image]["roi_checksum"]
        and baseline_rows[image]["roi_area"] == adapted_rows[image]["roi_area"]
        for image in baseline_rows
    )
    if not roi_invariant:
        raise RuntimeError("fixed ROI changed between baseline and adapted evaluation")

    baseline_aggregate = baseline["aggregate"]
    adapted_aggregate = adapted["aggregate"]
    fp_comparison = _lower_is_better_change(
        baseline_aggregate["fp_rate"], adapted_aggregate["fp_rate"]
    )
    per_image_comparison: list[dict[str, Any]] = []
    for image in sorted(baseline_rows):
        left = baseline_rows[image]
        right = adapted_rows[image]
        baseline_rate = left["fp_rate"]
        adapted_rate = right["fp_rate"]
        if baseline_rate is None or adapted_rate is None:
            change = {
                "baseline": baseline_rate,
                "adapted": adapted_rate,
                "delta_candidate_minus_baseline": None,
                "relative_improvement": None,
                "status": "undefined_empty_roi",
                "favorable": None,
            }
        else:
            change = _lower_is_better_change(float(baseline_rate), float(adapted_rate))
        per_image_comparison.append(
            {
                "image": image,
                "roi_area": left["roi_area"],
                "baseline_fp_pixels": left["fp_pixels"],
                "adapted_fp_pixels": right["fp_pixels"],
                "fp_rate": change,
            }
        )
    return {
        "protocol": baseline["protocol"],
        "roi_invariant": roi_invariant,
        "baseline": baseline,
        "adapted": adapted,
        "comparison": {
            "fp_rate": fp_comparison,
            "per_image": per_image_comparison,
            "interpretation": "positive relative_improvement means lower adapted FP rate",
        },
    }


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _normalise_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, float | None]:
    """Flatten nested result dictionaries while preserving null metric values."""

    output: dict[str, float | None] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, Mapping):
                output.update(_flatten_scalars(child, path))
            else:
                number = _finite_number(child)
                if number is not None or child is None:
                    output[path] = number
        return output
    number = _finite_number(value)
    if number is not None or value is None:
        output[prefix or "value"] = number
    return output


def _lookup_metric(flat: Mapping[str, float | None], aliases: Sequence[str]) -> tuple[float | None, str | None]:
    normalised = {_normalise_key(key): key for key in flat}
    for alias in aliases:
        alias_key = _normalise_key(alias)
        if alias_key in normalised:
            source = normalised[alias_key]
            return flat[source], source
    for alias in aliases:
        alias_key = _normalise_key(alias)
        suffixes = [f"_{alias_key}", f".{alias_key}"]
        for key, original in normalised.items():
            if any(key.endswith(suffix) for suffix in suffixes):
                source = original
                return flat[source], source
    return None, None


def _metric_change(
    baseline: float | None,
    candidate: float | None,
    *,
    direction: str | None,
    baseline_source: str | None = None,
    candidate_source: str | None = None,
) -> dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "delta_candidate_minus_baseline": None,
            "direction": direction or "unknown",
            "status": "missing",
            "favorable": None,
            "baseline_source": baseline_source,
            "candidate_source": candidate_source,
        }
    delta = float(candidate - baseline)
    if direction == "higher":
        favorable = delta >= 0
    elif direction == "lower":
        favorable = delta <= 0
    else:
        favorable = None
    if favorable is None:
        status = "increased" if delta > 0 else ("decreased" if delta < 0 else "unchanged")
    elif delta == 0:
        status = "unchanged"
    elif favorable:
        status = "improved"
    else:
        status = "degraded"
    return {
        "baseline": float(baseline),
        "candidate": float(candidate),
        "delta_candidate_minus_baseline": delta,
        "direction": direction or "unknown",
        "status": status,
        "favorable": favorable,
        "baseline_source": baseline_source,
        "candidate_source": candidate_source,
    }


def check_dice_iou_non_degradation(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    formal_tolerance: float = DEFAULT_STRICT_OVERLAP_TOLERANCE,
    near_tie_tolerance: float = DEFAULT_NEAR_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Apply the strict Dice/IoU gate and separately report near ties.

    ``formal_tolerance`` defaults to zero.  A negative delta therefore fails
    the formal gate even when its magnitude is <= 0.0005.  The latter is only
    reported as ``near_tie`` and is never relabelled as an improvement.
    """

    formal_tolerance = float(formal_tolerance)
    near_tie_tolerance = float(near_tie_tolerance)
    if not math.isfinite(formal_tolerance) or formal_tolerance < 0:
        raise ValueError("formal_tolerance must be finite and non-negative")
    if not math.isfinite(near_tie_tolerance) or near_tie_tolerance < formal_tolerance:
        raise ValueError("near_tie_tolerance must be finite and >= formal_tolerance")

    rows: dict[str, dict[str, Any]] = {}
    passed = True
    for metric in ("dice", "iou"):
        baseline = _finite_number(baseline_metrics.get(metric))
        candidate = _finite_number(candidate_metrics.get(metric))
        delta = None if baseline is None or candidate is None else float(candidate - baseline)
        strict_passed = delta is not None and delta >= -formal_tolerance
        approximate_tie = delta is not None and delta < -formal_tolerance and delta >= -near_tie_tolerance
        if delta is None:
            status = "missing"
        elif delta > 0:
            status = "improved"
        elif delta == 0:
            status = "unchanged"
        elif approximate_tie:
            status = "near_tie"
        else:
            status = "degraded"
        rows[metric] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta_candidate_minus_baseline": delta,
            "formal_tolerance": formal_tolerance,
            "near_tie_tolerance": near_tie_tolerance,
            "strict_passed": strict_passed,
            "approximate_tie": approximate_tie,
            "status": status,
        }
        passed = bool(passed and strict_passed)
    return {
        "passed": passed,
        "formal_tolerance": formal_tolerance,
        "near_tie_tolerance": near_tie_tolerance,
        "metrics": rows,
        "interpretation": "negative deltas are degradation or near_tie; never an improvement",
    }


_TSRS_PRIMARY_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("gap_fp", ("gap_region_fp_rate", "gap_fp", "gap_fp_rate"), "lower"),
    ("merge", ("component_merge_rate", "merge_rate", "merge"), "lower"),
)

_RAM_PRIMARY_SPECS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("overlap_dsc", ("overlap_dsc", "overlap_region_metrics.dsc", "overlap_region_dsc", "overlap_metrics.dsc"), "higher"),
    ("overlap_iou", ("overlap_iou", "overlap_region_metrics.iou", "overlap_region_iou"), "higher"),
    ("overlap_nsd_2px", ("overlap_nsd_2px", "overlap_region_metrics.nsd_2px"), "higher"),
    ("overlap_msd_px", ("overlap_msd_px", "overlap_region_metrics.msd_px"), "lower"),
    ("pair_msd_px", ("pair_msd_px", "pair_msd", "overlap_pair_msd_px", "overlap_pair_intersection_metrics.msd_px"), "lower"),
)


def _resolve_overlap_metrics(flat: Mapping[str, float | None]) -> tuple[dict[str, float | None], dict[str, str | None]]:
    # Prefer RAM overlap-region values over overall-instance values.  A plain
    # suffix lookup would otherwise select ``overall_instance_metrics.dsc``
    # merely because it appears earlier in an official result dictionary.
    dice, dice_source = _lookup_metric(
        flat,
        ("overlap_dsc", "overlap_region_metrics.dsc", "overlap_region_dsc",
         "overlap_metrics.dsc", "dice", "dsc", "overall_dsc"),
    )
    iou, iou_source = _lookup_metric(
        flat,
        ("overlap_iou", "overlap_region_metrics.iou", "overlap_region_iou",
         "iou", "overall_iou"),
    )
    if iou is None:
        voe, voe_source = _lookup_metric(
            flat,
            ("overlap_voe", "overlap_region_metrics.voe", "overlap_region_voe",
             "voe", "overall_voe"),
        )
        if voe is not None:
            iou = float(1.0 - voe)
            iou_source = f"derived_from_{voe_source}:1-voe"
    return {"dice": dice, "iou": iou}, {"dice": dice_source, "iou": iou_source}


def _direction_for_metric(key: str) -> str | None:
    normal = _normalise_key(key)
    if any(token in normal for token in ("gap_fp", "fp_rate", "merge_rate", "count_mae", "msd", "assd", "hd95", "ravd", "voe")):
        return "lower"
    if any(token in normal for token in ("dice", "dsc", "iou", "nsd", "precision", "recall", "specificity", "boundary", "surface")):
        return "higher"
    return None


def assess_comparison(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    *,
    domain: str,
    formal_tolerance: float = DEFAULT_STRICT_OVERLAP_TOLERANCE,
    near_tie_tolerance: float = DEFAULT_NEAR_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Assess all reported metrics plus domain-specific primary metrics.

    TSRS primary metrics are gap FP and component merge rate.  RAM primary
    metrics are overlap quality/surface metrics and pair MSD.  Dice/IoU safety
    is always strict by default.  Auxiliary metric changes are reported in
    full, but no universal non-degradation guarantee is claimed for them.
    """

    domain = str(domain).lower()
    if domain not in {"tsrs", "ram"}:
        raise ValueError("domain must be 'tsrs' or 'ram'")
    baseline_flat = _flatten_scalars(baseline_metrics)
    candidate_flat = _flatten_scalars(candidate_metrics)
    overlap_baseline, overlap_baseline_sources = _resolve_overlap_metrics(baseline_flat)
    overlap_candidate, overlap_candidate_sources = _resolve_overlap_metrics(candidate_flat)
    overlap_gate = check_dice_iou_non_degradation(
        overlap_baseline,
        overlap_candidate,
        formal_tolerance=formal_tolerance,
        near_tie_tolerance=near_tie_tolerance,
    )
    all_keys = sorted(set(baseline_flat) | set(candidate_flat), key=_normalise_key)
    metric_changes: dict[str, Any] = {}
    for key in all_keys:
        direction = _direction_for_metric(key)
        metric_changes[key] = _metric_change(
            baseline_flat.get(key),
            candidate_flat.get(key),
            direction=direction,
        )

    specs = _TSRS_PRIMARY_SPECS if domain == "tsrs" else _RAM_PRIMARY_SPECS
    primary: dict[str, Any] = {}
    required_missing: list[str] = []
    for name, aliases, direction in specs:
        baseline_value, baseline_source = _lookup_metric(baseline_flat, aliases)
        candidate_value, candidate_source = _lookup_metric(candidate_flat, aliases)
        row = _metric_change(
            baseline_value,
            candidate_value,
            direction=direction,
            baseline_source=baseline_source,
            candidate_source=candidate_source,
        )
        primary[name] = row
        # overlap_iou and overlap_nsd are useful when available but RAM can be
        # evaluated from official DSC/MSD/pair-MSD outputs that omit IoU/NSD.
        optional = domain == "ram" and name in {"overlap_iou", "overlap_nsd_2px"}
        if not optional and row["status"] == "missing":
            required_missing.append(name)

    available_primary = [row for row in primary.values() if row["status"] != "missing"]
    primary_all_non_degraded = bool(
        not required_missing
        and available_primary
        and all(row["favorable"] is True for row in available_primary)
    )
    primary_any_improved = bool(any(row["status"] == "improved" for row in available_primary))
    primary_claim_passed = bool(primary_all_non_degraded and primary_any_improved)
    primary_names: set[str] = set()
    primary_sources: set[str] = set()
    for name, aliases, _direction in specs:
        primary_names.add(_normalise_key(name))
        primary_names.update(_normalise_key(alias) for alias in aliases)
        row = primary[name]
        for source in (row.get("baseline_source"), row.get("candidate_source")):
            if source is not None:
                primary_sources.add(_normalise_key(source))
    overlap_sources = {
        _normalise_key(source)
        for source in (*overlap_baseline_sources.values(), *overlap_candidate_sources.values())
        if source is not None
    }
    auxiliary = {
        key: change
        for key, change in metric_changes.items()
        if _normalise_key(key) not in primary_names
        and _normalise_key(key) not in primary_sources
        and _normalise_key(key) not in overlap_sources
    }
    auxiliary_degraded = sorted(
        [key for key, change in auxiliary.items() if change["status"] == "degraded"],
        key=_normalise_key,
    )
    return {
        "domain": domain,
        "overlap_safety": {
            **overlap_gate,
            "baseline_sources": overlap_baseline_sources,
            "candidate_sources": overlap_candidate_sources,
        },
        "primary": {
            "metrics": primary,
            "all_non_degraded": primary_all_non_degraded,
            "any_improved": primary_any_improved,
            "claim_passed": primary_claim_passed,
            "required_missing": required_missing,
            "interpretation": (
                "primary claim requires every required primary metric to be favorable or unchanged "
                "and at least one to improve"
            ),
        },
        "all_metrics": metric_changes,
        "auxiliary": auxiliary,
        "auxiliary_degraded": auxiliary_degraded,
        "passed": bool(overlap_gate["passed"] and primary_claim_passed),
        "guarantees": {
            "dice_iou_formal_non_degradation": bool(overlap_gate["passed"]),
            "primary_metric_non_degradation": primary_all_non_degraded,
            "universal_metric_non_degradation": False,
            "statement": "This result does not claim a universal no-degradation guarantee for auxiliary metrics.",
        },
    }


def _json_ready(value: Any) -> Any:
    """Convert NumPy scalars/containers to strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("result contains a non-finite float")
    return value


def _collect_epoch_values(value: Any) -> list[int]:
    found: list[int] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = _normalise_key(str(key))
            if key_name in {"epoch", "checkpoint_epoch", "selected_epoch", "best_epoch"}:
                if isinstance(child, (int, np.integer)) and not isinstance(child, bool):
                    found.append(int(child))
                elif isinstance(child, float) and child.is_integer():
                    found.append(int(child))
            elif key_name in {"epochs", "checkpoint_epochs"} and isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                for item in child:
                    if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
                        found.append(int(item))
            found.extend(_collect_epoch_values(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.extend(_collect_epoch_values(child))
    return found


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a checkpoint SHA-256 for result provenance binding."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_bound_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the single-epoch/single-checkpoint R351 result envelope."""

    errors: list[str] = []
    if payload.get("schema") != RESULT_SCHEMA:
        errors.append(f"schema must be {RESULT_SCHEMA}")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        errors.append("selection must be an object")
        selection = {}
    epoch = selection.get("epoch")
    if not isinstance(epoch, (int, np.integer)) or isinstance(epoch, bool) or int(epoch) < 0:
        errors.append("selection.epoch must be a non-negative integer")
        selected_epoch = None
    else:
        selected_epoch = int(epoch)
    checkpoint_sha = selection.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or _SHA256_RE.fullmatch(checkpoint_sha) is None:
        errors.append("selection.checkpoint_sha256 must be a 64-character hexadecimal SHA-256")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics must be an object")
        metrics = {}
    epochs = _collect_epoch_values(metrics)
    if selected_epoch is not None and any(value != selected_epoch for value in epochs):
        errors.append(f"metrics contain epoch values {sorted(set(epochs))}, selected epoch is {selected_epoch}")
    if len(set(epochs)) > 1:
        errors.append(f"metrics mix multiple epochs: {sorted(set(epochs))}")
    return {
        "passed": not errors,
        "errors": errors,
        "selected_epoch": selected_epoch,
        "metrics_epochs": sorted(set(epochs)),
        "single_checkpoint": bool(
            isinstance(checkpoint_sha, str) and _SHA256_RE.fullmatch(checkpoint_sha) is not None
        ),
    }


def bind_single_checkpoint_result(
    *,
    experiment: str,
    split: str,
    epoch: int,
    checkpoint_sha256: str,
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a result envelope that cannot silently mix epochs.

    ``checkpoint_sha256`` is required even when ``checkpoint_path`` is also
    supplied.  A caller can use :func:`sha256_file` to calculate it.  Any
    epoch values embedded in ``metrics`` must equal the selected epoch.
    """

    if not isinstance(epoch, (int, np.integer)) or isinstance(epoch, bool) or int(epoch) < 0:
        raise ValueError("epoch must be a non-negative integer")
    if not isinstance(checkpoint_sha256, str) or _SHA256_RE.fullmatch(checkpoint_sha256) is None:
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal SHA-256")
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a mapping")
    if checkpoint_path is not None:
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha.lower() != checkpoint_sha256.lower():
            raise ValueError(
                f"checkpoint_sha256 does not match checkpoint_path: {actual_sha} != {checkpoint_sha256}"
            )
    copied_metrics = _json_ready(copy.deepcopy(dict(metrics)))
    epochs = _collect_epoch_values(copied_metrics)
    selected_epoch = int(epoch)
    if any(value != selected_epoch for value in epochs) or len(set(epochs)) > 1:
        raise ValueError(
            f"metrics contain epoch values {sorted(set(epochs))}; expected only selected epoch {selected_epoch}"
        )
    payload: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "experiment": str(experiment),
        "split": str(split),
        "selection": {
            "epoch": selected_epoch,
            "checkpoint_sha256": checkpoint_sha256.lower(),
        },
        "metrics": copied_metrics,
        "provenance": {
            "single_epoch": True,
            "single_checkpoint": True,
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        },
    }
    if comparison is not None:
        payload["comparison"] = _json_ready(copy.deepcopy(dict(comparison)))
    if config is not None:
        payload["config"] = _json_ready(copy.deepcopy(dict(config)))
    validation = validate_bound_result(payload)
    if not validation["passed"]:
        raise ValueError(f"invalid R351 result envelope: {validation['errors']}")
    return payload


def write_bound_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Validate and write one bound R351 result JSON file."""

    validation = validate_bound_result(payload)
    if not validation["passed"]:
        raise ValueError(f"invalid R351 result envelope: {validation['errors']}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return output


def _self_test() -> dict[str, Any]:
    """Small dependency-light smoke used by the command-line verifier."""

    seam = np.array([[0.0, 0.5, 1.0], [0.0, 0.25, 0.75]], dtype=np.float32)
    target = np.zeros((1, 2, 3), dtype=np.uint8)
    target[0, 0, 2] = 1
    baseline_pred = np.zeros_like(target)
    adapted_pred = baseline_pred.copy()
    adapted_pred[0, 1, 1] = 1
    baseline = evaluate_fixed_seam_roi({"a": baseline_pred}, {"a": seam}, {"a": target})
    adapted = evaluate_fixed_seam_roi({"a": adapted_pred}, {"a": seam}, {"a": target})
    comparison = compare_fixed_seam_roi(
        {"a": baseline_pred}, {"a": adapted_pred}, {"a": seam}, {"a": target}
    )
    assert comparison["roi_invariant"]
    assert comparison["comparison"]["fp_rate"]["status"] == "degraded"
    strict = check_dice_iou_non_degradation({"dice": 0.9, "iou": 0.8}, {"dice": 0.9 - 1e-6, "iou": 0.8})
    assert not strict["passed"] and strict["metrics"]["dice"]["approximate_tie"]
    payload = bind_single_checkpoint_result(
        experiment="R351_SMOKE",
        split="val",
        epoch=1,
        checkpoint_sha256="0" * 64,
        metrics={"dice": 0.9, "iou": 0.8},
    )
    assert validate_bound_result(payload)["passed"]
    return {
        "passed": True,
        "roi_area": baseline["aggregate"]["roi_area"],
        "adapted_fp_rate": adapted["aggregate"]["fp_rate"],
        "strict_overlap_gate": strict["passed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="R351 dependency-light evaluator smoke.")
    parser.add_argument("--self-test", action="store_true", help="run the built-in CPU smoke")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("only --self-test is provided; import the functions for project evaluation")
    print(json.dumps(_self_test(), indent=2))


__all__ = [
    "DEFAULT_NEAR_TIE_TOLERANCE",
    "DEFAULT_SEAM_THRESHOLD",
    "DEFAULT_STRICT_OVERLAP_TOLERANCE",
    "RESULT_SCHEMA",
    "aggregate_fixed_seam_rows",
    "assess_comparison",
    "bind_single_checkpoint_result",
    "build_ram_seam_background_roi",
    "check_dice_iou_non_degradation",
    "compare_fixed_seam_roi",
    "evaluate_fixed_seam_roi",
    "fixed_seam_background_roi",
    "sha256_file",
    "validate_bound_result",
    "write_bound_result",
]


if __name__ == "__main__":
    main()
