#!/usr/bin/env python3
"""R351 RAM dual-prior adapter/refiner pilot.

The mature R332 nnU-Net, its six-channel completion refiner, and the learned
image/seam overlap prior are frozen.  Only the zero-initialized input adapter
and the seven-channel output completion prior are updated.  The input adapter
is deliberately evaluated outside ``torch.no_grad`` before the frozen
nnU-Net: its output must retain a gradient path through the frozen network.

This script only opens RAM ``train`` and ``val``.  Official metrics retain the
historical padded arrays for compatibility, while the separately reported
seam ROI always includes the fixed prior/ground-truth/valid mask and therefore
never counts right/bottom padding.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients
from r351_data import AlignedRamDataset
from r351_dual_prior import (
    ImageOverlapPrior,
    OutputCompletionPrior,
    SeamInputAdapter,
    local_terms,
    normalize_prior,
)
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r327_ram_native_instance_completion_prior import (
    InstanceCompletionRefiner,
    seed_all,
)
from train_r332_ram_joint_iterative_refinement import iterative_refine


EXPERIMENT = "R351_RAM_DUAL_PRIOR"
DEFAULT_AUXILIARY_WEIGHTS = {
    "separation": 0.035,
    "overlap": 0.035,
    "preserve": 0.10,
}
STRUCTURE_MAX_KEYS = (
    "overlap_nsd_2px",
    "overlap_dsc",
    "overlap_iou",
    "pair_nsd_2px",
    "pair_dsc",
    "overall_nsd_2px",
)
STRUCTURE_MIN_KEYS = (
    "overlap_msd_px",
    "pair_msd_px",
)


def sha256(path: Path) -> str:
    """Return the content hash used to tie a run to frozen source weights."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_record(path: Path) -> dict[str, str | None]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256(resolved) if resolved.exists() else None,
    }


def _assert_fresh_output(path: Path) -> None:
    """R351 runs are immutable: a non-empty output directory is a mistake."""
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty R351 output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return payload


def _load_state(payload: dict[str, Any], key: str, path: Path) -> dict[str, torch.Tensor]:
    state = payload.get(key)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {path} has no state mapping under {key!r}")
    return state


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _case_names(value: Any, batch_size: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if batch_size == 1:
        return [str(value)]
    raise RuntimeError(f"Cannot unpack case field for batch size {batch_size}: {type(value)!r}")


def _batch_tensors(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    image = batch["image"].to(device, non_blocking=True)
    target = batch["mask"].to(device, non_blocking=True).float()
    seam = batch["seam"].to(device, non_blocking=True).float()
    valid = batch["valid"].to(device, non_blocking=True).float()
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if image.ndim != 4 or target.ndim != 4 or seam.ndim != 4 or valid.ndim != 4:
        raise RuntimeError({
            "image": tuple(image.shape), "mask": tuple(target.shape),
            "seam": tuple(seam.shape), "valid": tuple(valid.shape),
        })
    if target.shape[0] != image.shape[0] or seam.shape != valid.shape:
        raise RuntimeError({
            "image": tuple(image.shape), "mask": tuple(target.shape),
            "seam": tuple(seam.shape), "valid": tuple(valid.shape),
        })
    return image, target, seam, valid


def _overlap_probability(
    overlap_prior: ImageOverlapPrior,
    image: torch.Tensor,
    seam: torch.Tensor,
) -> torch.Tensor:
    # The prior is frozen and its prediction is an inference feature.  It is
    # intentionally detached before entering the trainable adapter/refiner.
    with torch.no_grad():
        return torch.sigmoid(overlap_prior(image, seam)).detach()


def student_forward(
    network: NnUNetMultiLabelPrior,
    overlap_prior: ImageOverlapPrior,
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    image: torch.Tensor,
    seam: torch.Tensor,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the trainable R351 path and preserve adapter-to-backbone gradients."""
    overlap_probability = _overlap_probability(overlap_prior, image, seam)
    adapted_image = adapter(image, seam, overlap_probability)
    # Do not put this call under no_grad.  ``network`` is frozen, but its
    # Jacobian with respect to ``adapted_image`` is required by the adapter.
    logits, _ = network(adapted_image, False)
    refined = refiner.refine(image, logits, seam, overlap_probability, steps=steps)
    return refined, overlap_probability, adapted_image


@torch.inference_mode()
def teacher_forward(
    network: NnUNetMultiLabelPrior,
    teacher_refiner: InstanceCompletionRefiner,
    image: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Use the original R332 image and original six-channel refiner path."""
    base_logits, _ = network(image, False)
    return iterative_refine(image, base_logits, teacher_refiner, steps)[0][-1]


def flatten_official(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Flatten the historical official metric structure for checkpoint gates."""
    overall = metrics["overall_instance_metrics"]
    overlap = metrics["overlap_region_metrics"]
    pair = metrics["overlap_pair_intersection_metrics"]

    def iou(section: dict[str, Any]) -> float | None:
        value = section.get("voe")
        return None if value is None else 1.0 - float(value)

    return {
        "overall_dsc": overall.get("dsc"),
        "overall_iou": iou(overall),
        "overall_nsd_2px": overall.get("nsd_2px"),
        "overall_msd_px": overall.get("msd_px"),
        "overall_msd_fail_rate": overall.get("msd_fail_rate"),
        "overall_ravd": overall.get("ravd"),
        "overlap_dsc": overlap.get("dsc"),
        "overlap_iou": iou(overlap),
        "overlap_nsd_2px": overlap.get("nsd_2px"),
        "overlap_msd_px": overlap.get("msd_px"),
        "overlap_msd_fail_rate": overlap.get("msd_fail_rate"),
        "overlap_ravd": overlap.get("ravd"),
        "pair_dsc": pair.get("dsc"),
        "pair_iou": iou(pair),
        "pair_nsd_2px": pair.get("nsd_2px"),
        "pair_msd_px": pair.get("msd_px"),
        "pair_ravd": pair.get("ravd"),
        "pair_msd_fail_rate": pair.get("msd_fail_rate"),
    }


def _max_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0e30
    return number if math.isfinite(number) else -1.0e30


def _min_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1.0e30
    return -number if math.isfinite(number) else -1.0e30


def structure_score(flat: dict[str, Any]) -> tuple[float, ...]:
    """Prefer overlap surface/shape quality after the strict native gate."""
    return tuple(
        [_max_score(flat.get(key)) for key in STRUCTURE_MAX_KEYS]
        + [_min_score(flat.get(key)) for key in STRUCTURE_MIN_KEYS]
    )


def structure_score_json(flat: dict[str, Any]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for key in STRUCTURE_MAX_KEYS + STRUCTURE_MIN_KEYS:
        value = flat.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            values[key] = None
            continue
        values[key] = number if math.isfinite(number) else None
    return values


def native_gate(
    candidate: dict[str, Any],
    native: dict[str, Any],
) -> tuple[bool, dict[str, bool], bool]:
    """Apply the overall gate and the required 5% RAM structure improvement."""
    checks: dict[str, bool] = {}
    for name in ("overall_dsc", "overall_iou"):
        try:
            left, right = float(candidate.get(name)), float(native.get(name))
            checks[f"{name}_ge_native"] = bool(
                math.isfinite(left) and math.isfinite(right) and left >= right
            )
        except (TypeError, ValueError):
            checks[f"{name}_ge_native"] = False
    region_gate = all(checks.values())
    for name in ("overlap_msd_px", "pair_msd_px"):
        try:
            left, right = float(candidate.get(name)), float(native.get(name))
            if not math.isfinite(left) or not math.isfinite(right) or right < 0:
                passed = False
            elif right == 0:
                passed = left <= 0
            else:
                passed = left <= right * 0.95
        except (TypeError, ValueError):
            passed = False
        checks[f"{name}_relative_reduction_5pct"] = bool(passed)
    structure_gate = all(
        checks[name] for name in (
            "overlap_msd_px_relative_reduction_5pct",
            "pair_msd_px_relative_reduction_5pct",
        )
    )
    for name in ("overall_msd_fail_rate", "overlap_msd_fail_rate", "pair_msd_fail_rate"):
        left, right = candidate.get(name), native.get(name)
        checks[f"{name}_not_increased"] = bool(left is not None and right is not None
            and math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) <= float(right))
    for name in ("overlap_dsc", "overlap_nsd_2px", "pair_nsd_2px"):
        left, right = candidate.get(name), native.get(name)
        checks[f"{name}_not_decreased"] = bool(left is not None and right is not None
            and math.isfinite(float(left)) and math.isfinite(float(right)) and float(left) >= float(right))
    checks["structure_gate"] = structure_gate
    return bool(all(checks.values())), checks, bool(region_gate)


def _selection_key(accepted: bool, flat: dict[str, Any]) -> tuple[float, ...]:
    # Gate first, then structural score, then the two gate metrics only as
    # deterministic tie-breakers.  This keeps selection from using the final
    # epoch or silently mixing metrics from different checkpoints.
    return (
        float(int(accepted)),
        *structure_score(flat),
        _max_score(flat.get("overall_dsc")),
        _max_score(flat.get("overall_iou")),
    )


def _seam_roi(
    prediction: torch.Tensor,
    target: torch.Tensor,
    seam: torch.Tensor,
    valid: torch.Tensor,
    cases: list[str],
    threshold: float,
) -> dict[str, Any]:
    """Compute fixed per-case false-positive pixels/ROI area without padding."""
    prior = normalize_prior(seam.float())
    union_background = target.sum(1, keepdim=True) < 0.5
    roi = (prior >= threshold) & union_background & (valid > 0.5)
    false_positive = prediction.any(1, keepdim=True) & roi
    fp_pixels = false_positive.sum((1, 2, 3)).detach().cpu().tolist()
    roi_area = roi.sum((1, 2, 3)).detach().cpu().tolist()
    rows = []
    for case, fp, area in zip(cases, fp_pixels, roi_area):
        fp_value, area_value = int(fp), int(area)
        rows.append({
            "case": case,
            "fp_pixels": fp_value,
            "roi_area": area_value,
            "fp_rate": (float(fp_value) / area_value) if area_value else None,
            "roi_empty": area_value == 0,
        })
    total_fp = int(sum(fp_pixels))
    total_area = int(sum(roi_area))
    rates = [row["fp_rate"] for row in rows if row["fp_rate"] is not None]
    return {
        "threshold": threshold,
        "definition": "normalize(seam) >= threshold AND GT union background AND valid",
        "per_case": rows,
        "total_fp_pixels": total_fp,
        "total_roi_area": total_area,
        "fp_per_area": (float(total_fp) / total_area) if total_area else None,
        "mean_case_fp_rate": float(np.mean(rates)) if rates else None,
        "nonempty_cases": len(rates),
        "n_cases": len(rows),
    }


def _add_cases(
    destination: dict[str, dict[str, np.ndarray]],
    batch: dict[str, Any],
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> list[str]:
    batch_size = int(prediction.shape[0])
    cases = _case_names(batch["case"], batch_size)
    if len(cases) != batch_size:
        raise RuntimeError({"cases": cases, "batch_size": batch_size})
    pred_np = prediction.detach().cpu().numpy().astype(bool)
    target_np = (target > 0.5).detach().cpu().numpy().astype(bool)
    for index, case in enumerate(cases):
        if case in destination:
            raise RuntimeError(f"Duplicate case in validation collection: {case}")
        destination[case] = {"pred": pred_np[index], "target": target_np[index]}
    return cases


@torch.inference_mode()
def collect_native(
    model: NnUNetMultiLabelPrior,
    loader: DataLoader,
    device: torch.device,
    pairs: list[tuple[int, int]],
    seam_threshold: float,
) -> dict[str, Any]:
    model.eval()
    cases: dict[str, dict[str, np.ndarray]] = {}
    roi_rows: list[dict[str, Any]] = []
    for batch in loader:
        image, target, seam, valid = _batch_tensors(batch, device)
        logits, _ = model(image, False)
        prediction = torch.sigmoid(logits) >= 0.5
        names = _add_cases(cases, batch, prediction, target)
        roi_rows.extend(_seam_roi(prediction, target, seam, valid, names, seam_threshold)["per_case"])
    official = official_metrics(cases, pairs)
    flat = flatten_official(official)
    roi = _aggregate_roi_rows(roi_rows, seam_threshold)
    return {"official": official, "flat": flat, "seam_roi": roi, "cases": len(cases)}


@torch.inference_mode()
def collect_teacher(
    network: NnUNetMultiLabelPrior,
    teacher_refiner: InstanceCompletionRefiner,
    loader: DataLoader,
    device: torch.device,
    pairs: list[tuple[int, int]],
    steps: int,
    seam_threshold: float,
) -> dict[str, Any]:
    network.eval()
    teacher_refiner.eval()
    cases: dict[str, dict[str, np.ndarray]] = {}
    roi_rows: list[dict[str, Any]] = []
    for batch in loader:
        image, target, seam, valid = _batch_tensors(batch, device)
        logits = teacher_forward(network, teacher_refiner, image, steps)
        prediction = torch.sigmoid(logits) >= 0.5
        names = _add_cases(cases, batch, prediction, target)
        roi_rows.extend(_seam_roi(prediction, target, seam, valid, names, seam_threshold)["per_case"])
    official = official_metrics(cases, pairs)
    flat = flatten_official(official)
    roi = _aggregate_roi_rows(roi_rows, seam_threshold)
    return {"official": official, "flat": flat, "seam_roi": roi, "cases": len(cases)}


@torch.inference_mode()
def collect_adapted(
    network: NnUNetMultiLabelPrior,
    overlap_prior: ImageOverlapPrior,
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    loader: DataLoader,
    device: torch.device,
    pairs: list[tuple[int, int]],
    steps: int,
    seam_threshold: float,
) -> dict[str, Any]:
    network.eval()
    overlap_prior.eval()
    adapter.eval()
    refiner.eval()
    cases: dict[str, dict[str, np.ndarray]] = {}
    roi_rows: list[dict[str, Any]] = []
    for batch in loader:
        image, target, seam, valid = _batch_tensors(batch, device)
        logits, _, _ = student_forward(
            network, overlap_prior, adapter, refiner, image, seam, steps
        )
        prediction = torch.sigmoid(logits) >= 0.5
        names = _add_cases(cases, batch, prediction, target)
        roi_rows.extend(_seam_roi(prediction, target, seam, valid, names, seam_threshold)["per_case"])
    official = official_metrics(cases, pairs)
    flat = flatten_official(official)
    roi = _aggregate_roi_rows(roi_rows, seam_threshold)
    return {"official": official, "flat": flat, "seam_roi": roi, "cases": len(cases)}


def _aggregate_roi_rows(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    total_fp = int(sum(int(row["fp_pixels"]) for row in rows))
    total_area = int(sum(int(row["roi_area"]) for row in rows))
    rates = [float(row["fp_rate"]) for row in rows if row["fp_rate"] is not None]
    return {
        "threshold": threshold,
        "definition": "normalize(seam) >= threshold AND GT union background AND valid",
        "per_case": rows,
        "total_fp_pixels": total_fp,
        "total_roi_area": total_area,
        "fp_per_area": (float(total_fp) / total_area) if total_area else None,
        "mean_case_fp_rate": float(np.mean(rates)) if rates else None,
        "nonempty_cases": len(rates),
        "n_cases": len(rows),
    }


def _source_metadata(
    r332_checkpoint: Path,
    overlap_checkpoint: Path,
    native_checkpoint: Path,
) -> dict[str, dict[str, str | None]]:
    # R332's model and six-channel refiner are both sourced from one payload;
    # retaining separate names makes the frozen dependency explicit.
    r332 = _source_record(r332_checkpoint)
    overlap = _source_record(overlap_checkpoint)
    native = _source_record(native_checkpoint)
    return {
        "r332_model": dict(r332),
        "r332_refiner": dict(r332),
        "overlap_prior": dict(overlap),
        "native_baseline": dict(native),
    }


def _checkpoint_payload(
    epoch: int,
    row: dict[str, Any],
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    frozen_weights: dict[str, dict[str, str | None]],
    accepted: bool,
    selection_key: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "epoch": int(epoch),
        "evaluation_epoch": int(row["evaluation_epoch"]),
        "input": _cpu_state(adapter),
        "refiner": _cpu_state(refiner),
        "accepted": bool(accepted),
        "selection_score": list(selection_key),
        "evaluation": {
            "epoch": int(row["evaluation_epoch"]),
            "metrics_sha256": row["evaluation_metrics_sha256"],
        },
        "frozen_weights": frozen_weights,
        "row": row,
    }


def _save_checkpoint(
    path: Path,
    epoch: int,
    row: dict[str, Any],
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    frozen_weights: dict[str, dict[str, str | None]],
    accepted: bool,
    selection_key: tuple[float, ...],
) -> str:
    payload = _checkpoint_payload(
        epoch, row, adapter, refiner, frozen_weights, accepted, selection_key
    )
    torch.save(payload, path)
    return sha256(path)


@torch.inference_mode()
def identity_audit(
    network: NnUNetMultiLabelPrior,
    old_refiner: InstanceCompletionRefiner,
    new_refiner: OutputCompletionPrior,
    overlap_prior: ImageOverlapPrior,
    adapter: SeamInputAdapter,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    max_cases: int = 2,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Check new 7-channel initialization against the old 6-channel path."""
    network.eval()
    old_refiner.eval()
    new_refiner.eval()
    overlap_prior.eval()
    adapter.eval()
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_cases:
            break
        image, target, seam, valid = _batch_tensors(batch, device)
        base_logits, _ = network(image, False)
        old_logits = iterative_refine(image, base_logits, old_refiner, steps)[0][-1]
        overlap_probability = _overlap_probability(overlap_prior, image, seam)
        new_logits = new_refiner.refine(
            image, base_logits, seam, overlap_probability, steps=steps
        )
        adapted_image = adapter(image, seam, overlap_probability)
        names = _case_names(batch["case"], int(image.shape[0]))
        rows.extend({
            "case": case,
            "max_abs_logit_difference": float(
                (new_logits[index:index + 1] - old_logits[index:index + 1]).abs().max()
            ),
            "max_abs_input_adapter_difference": float(
                (adapted_image[index:index + 1] - image[index:index + 1]).abs().max()
            ),
            "valid_pixels": int(valid[index:index + 1].sum()),
            "target_pixels": int(target[index:index + 1].sum()),
        } for index, case in enumerate(names))
    max_difference = max(
        (row["max_abs_logit_difference"] for row in rows), default=float("inf")
    )
    max_adapter_difference = max(
        (row["max_abs_input_adapter_difference"] for row in rows), default=float("inf")
    )
    first_weight = new_refiner.enc1.net[0].weight
    seam_channel_nonzero = int(torch.count_nonzero(first_weight[:, 6]).item())
    return {
        "cases": len(rows),
        "per_case": rows,
        "max_abs_logit_difference": max_difference,
        "max_abs_input_adapter_difference": max_adapter_difference,
        "seventh_input_channel_nonzero": seam_channel_nonzero,
        "seventh_input_channel_abs_sum": float(first_weight[:, 6].abs().sum()),
        "tolerance": tolerance,
        "passed": bool(
            len(rows) == max_cases
            and seam_channel_nonzero == 0
            and max_difference <= tolerance
            and max_adapter_difference <= tolerance
        ),
    }


def _make_student_terms(
    network: NnUNetMultiLabelPrior,
    overlap_prior: ImageOverlapPrior,
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    teacher_refiner: InstanceCompletionRefiner,
    batch: dict[str, Any],
    device: torch.device,
    steps: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor, torch.Tensor]:
    image, target, seam, valid = _batch_tensors(batch, device)
    with torch.no_grad():
        teacher_logits = teacher_forward(network, teacher_refiner, image, steps)
        teacher_probability = torch.sigmoid(teacher_logits)
    student_logits, overlap_probability, adapted_image = student_forward(
        network, overlap_prior, adapter, refiner, image, seam, steps
    )
    terms, states = local_terms(
        student_logits,
        target,
        teacher_probability,
        seam,
        overlap_probability,
        valid,
        membership_observed=True,
    )
    return terms, states, adapted_image, student_logits


def _assert_finite_terms(terms: dict[str, torch.Tensor], epoch: int) -> None:
    bad = [name for name, value in terms.items() if not bool(torch.isfinite(value).all())]
    if bad:
        raise RuntimeError({"epoch": epoch, "nonfinite_terms": bad})


def _routing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("separation", "overlap", "preserve"):
        selected = [row for row in rows if row["name"] == name]
        if not selected:
            output[name] = {"count": 0}
            continue
        output[name] = {
            "count": len(selected),
            "conflict_rate": float(np.mean([row["conflicted"] for row in selected])),
            "mean_scale": float(np.mean([row["scale"] for row in selected])),
            "mean_base_norm": float(np.mean([row["base_norm"] for row in selected])),
            "mean_raw_norm": float(np.mean([row["raw_norm"] for row in selected])),
            "mean_projected_norm": float(np.mean([row["projected_norm"] for row in selected])),
            "mean_applied_norm": float(np.mean([row["applied_norm"] for row in selected])),
            "mean_cosine_before": float(np.mean([row["cosine_before"] for row in selected])),
            "mean_cosine_after": float(np.mean([row["cosine_after"] for row in selected])),
        }
    return output


def _run_audit_backward(
    network: NnUNetMultiLabelPrior,
    overlap_prior: ImageOverlapPrior,
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    teacher_refiner: InstanceCompletionRefiner,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    auxiliary_weights: dict[str, float],
    max_aux_ratio: float,
    max_cases: int = 2,
) -> dict[str, Any]:
    """Run real forward/autograd routing on two cases without updating weights."""
    adapter.train()
    refiner.train()
    network.eval()
    overlap_prior.eval()
    trainable = list(adapter.parameters()) + list(refiner.parameters())
    per_case = []
    all_routing: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_cases:
            break
        terms, states, adapted_image, student_logits = _make_student_terms(
            network, overlap_prior, adapter, refiner, teacher_refiner, batch, device, steps
        )
        _assert_finite_terms(terms, 0)
        gradients, routed = route_parameter_gradients(
            trainable,
            terms["base"],
            [(name, terms[name], auxiliary_weights[name]) for name in
             ("separation", "overlap", "preserve")],
            max_aux_ratio=max_aux_ratio,
        )
        assign_gradients(trainable, gradients)
        adapter_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in adapter.parameters() if parameter.grad is not None
        )
        refiner_gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in refiner.parameters() if parameter.grad is not None
        )
        names = _case_names(batch["case"], int(student_logits.shape[0]))
        per_case.append({
            "cases": names,
            "finite": True,
            "channels": int(student_logits.shape[1]),
            "adapted_image_max_abs_delta": float((adapted_image - batch["image"].to(device)).abs().max()),
            "losses": {name: float(value.detach()) for name, value in terms.items()},
            "states": states,
            "adapter_gradient_sum": adapter_gradient,
            "refiner_gradient_sum": refiner_gradient,
            "gradient_routing": [vars(row) for row in routed],
        })
        all_routing.extend(vars(row) for row in routed)
        for parameter in trainable:
            parameter.grad = None
    passed = bool(
        len(per_case) == max_cases
        and all(row["channels"] == 14 for row in per_case)
        and all(row["adapter_gradient_sum"] > 0 and row["refiner_gradient_sum"] > 0
                for row in per_case)
        and all(
            row["cosine_after"] >= -1e-6
            and row["applied_norm"] <= max_aux_ratio * row["base_norm"] + 1e-6
            for row in all_routing
        )
    )
    return {
        "cases": len(per_case),
        "per_case": per_case,
        "gradient_routing": all_routing,
        "passed": passed,
        "max_aux_ratio": max_aux_ratio,
    }


def _train_epoch(
    epoch: int,
    network: NnUNetMultiLabelPrior,
    overlap_prior: ImageOverlapPrior,
    adapter: SeamInputAdapter,
    refiner: OutputCompletionPrior,
    teacher_refiner: InstanceCompletionRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps: int,
    auxiliary_weights: dict[str, float],
    aux_ramp: float,
    max_aux_ratio: float,
    max_grad_norm: float,
) -> dict[str, Any]:
    network.eval()
    overlap_prior.eval()
    adapter.train()
    refiner.train()
    totals = {name: 0.0 for name in ("base", "separation", "overlap", "preserve")}
    state_totals = {name: 0.0 for name in (
        "separation_mass", "overlap_observed_mass", "overlap_prior_mean")}
    routing_rows: list[dict[str, Any]] = []
    batches = 0
    started = time.time()
    trainable = list(adapter.parameters()) + list(refiner.parameters())
    for batch in loader:
        terms, states, _, _ = _make_student_terms(
            network, overlap_prior, adapter, refiner, teacher_refiner, batch, device, steps
        )
        _assert_finite_terms(terms, epoch)
        optimizer.zero_grad(set_to_none=True)
        gradients, routed = route_parameter_gradients(
            trainable,
            terms["base"],
            [(name, terms[name], auxiliary_weights[name] * aux_ramp) for name in
             ("separation", "overlap", "preserve")],
            max_aux_ratio=max_aux_ratio,
        )
        assign_gradients(trainable, gradients)
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        batches += 1
        for name, value in terms.items():
            totals[name] += float(value.detach())
        for name in state_totals:
            state_totals[name] += float(states[name])
        routing_rows.extend(vars(row) for row in routed)
    if batches == 0:
        raise RuntimeError("R351 training loader is empty")
    return {
        "batches": batches,
        "seconds": time.time() - started,
        "aux_ramp": aux_ramp,
        "losses": {name: value / batches for name, value in totals.items()},
        "state_mass": {name: value / batches for name, value in state_totals.items()},
        "gradient_routing": _routing_summary(routing_rows),
    }


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--r332-checkpoint", type=Path, required=True)
    parser.add_argument("--overlap-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--native-checkpoint", "--r325-checkpoint", "--baseline-checkpoint",
        dest="native_checkpoint",
        type=Path, required=True,
        help="Native R325 checkpoint used once for the validation gate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--aux-ramp-epochs", type=int, default=5)
    parser.add_argument("--max-aux-ratio", type=float, default=0.15)
    parser.add_argument("--max-grad-norm", type=float, default=8.0)
    parser.add_argument("--seam-threshold", type=float, default=0.25)
    parser.add_argument("--max-pairs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3511)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.aux_ramp_epochs < 1:
        raise ValueError("--aux-ramp-epochs must be positive")
    if args.max_aux_ratio < 0:
        raise ValueError("--max-aux-ratio must be non-negative")
    _assert_fresh_output(args.output)
    seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("R351 requires CUDA for native-resolution RAM forward/backward")
    device = torch.device("cuda")

    # The aligned dataset intentionally disables geometric augmentation: the
    # seam map and image must remain in the same canonical native coordinates.
    train_set = AlignedRamDataset(args.dataset_root, args.prior_root, "train", augment=False)
    val_set = AlignedRamDataset(args.dataset_root, args.prior_root, "val", augment=False)
    if args.audit:
        train_set.mask_files = train_set.mask_files[:2]
        val_set.mask_files = val_set.mask_files[:2]
    if len(train_set) < 2 or len(val_set) < 2:
        raise RuntimeError("R351 audit/training requires at least two train and val cases")
    if not args.audit and (len(train_set), len(val_set)) != (425, 69):
        raise RuntimeError({"expected_train_val": [425, 69], "actual": [len(train_set), len(val_set)]})
    worker_count = 0 if args.audit else max(0, args.num_workers)
    loader_options = {
        "batch_size": 1,
        "pin_memory": True,
        "num_workers": worker_count,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    frozen_weights = _source_metadata(
        args.r332_checkpoint, args.overlap_checkpoint, args.native_checkpoint
    )
    r332_payload = _load_payload(args.r332_checkpoint, device)
    overlap_payload = _load_payload(args.overlap_checkpoint, device)
    native_payload = _load_payload(args.native_checkpoint, device)

    network = NnUNetMultiLabelPrior().to(device)
    network.load_state_dict(_load_state(r332_payload, "model", args.r332_checkpoint), strict=True)
    network.requires_grad_(False)
    network.eval()

    old_refiner = InstanceCompletionRefiner().to(device)
    old_refiner.load_state_dict(
        _load_state(r332_payload, "refiner", args.r332_checkpoint), strict=True
    )
    old_refiner.requires_grad_(False)
    old_refiner.eval()

    refiner = OutputCompletionPrior().to(device)
    mature_initialization = refiner.load_mature(
        _load_state(r332_payload, "refiner", args.r332_checkpoint), identity=False
    )
    refiner.train()

    overlap_prior = ImageOverlapPrior().to(device)
    overlap_prior.load_state_dict(
        _load_state(overlap_payload, "model", args.overlap_checkpoint), strict=True
    )
    overlap_prior.requires_grad_(False)
    overlap_prior.eval()
    # The payload mappings are no longer needed after strict loading.  Release
    # their GPU copies before the native baseline and native-resolution audit.
    del r332_payload, overlap_payload
    torch.cuda.empty_cache()

    adapter = SeamInputAdapter().to(device)
    trainable = list(adapter.parameters()) + list(refiner.parameters())
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    # Add the output prior parameters as a second group after construction so
    # the CLI's single learning rate controls both trainable modules.
    optimizer.add_param_group({"params": refiner.parameters(), "lr": args.lr})
    pairs = top_training_pairs(args.dataset_root, args.max_pairs)

    protocol = {
        "experiment": EXPERIMENT,
        "split": "RAM train/validation only",
        "test_used": False,
        "expected_case_counts": {"train": 425, "val": 69},
        "actual_case_counts": {"train": len(train_set), "val": len(val_set)},
        "audit_mode": args.audit,
        "spatial_preprocessing": "native pixels; right/bottom padding only; no resize",
        "alignment": "R323 canonical seam maps; geometric augmentation disabled",
        "official_metric_padding_compatibility": "full padded prediction/target arrays",
        "seam_roi": "normalize(seam) >= .25 AND GT union background AND valid",
        "threshold": 0.5,
        "steps": args.steps,
        "trainable_modules": ["SeamInputAdapter", "OutputCompletionPrior"],
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "frozen_modules": ["NnUNetMultiLabelPrior", "ImageOverlapPrior", "InstanceCompletionRefiner teacher"],
        "frozen_parameter_count": sum(parameter.numel() for parameter in network.parameters())
        + sum(parameter.numel() for parameter in overlap_prior.parameters())
        + sum(parameter.numel() for parameter in old_refiner.parameters()),
        "mature_initialization": mature_initialization,
        "frozen_weights": frozen_weights,
        "config": _config(args),
    }
    _write_json(args.output / "protocol.json", protocol)

    identity = identity_audit(
        network, old_refiner, refiner, overlap_prior, adapter, val_loader, device,
        args.steps, max_cases=2,
    )
    _write_json(args.output / "identity_audit.json", identity)

    auxiliary_weights = dict(DEFAULT_AUXILIARY_WEIGHTS)
    if args.audit:
        audit = _run_audit_backward(
            network, overlap_prior, adapter, refiner, old_refiner, train_loader, device,
            args.steps, auxiliary_weights, args.max_aux_ratio, max_cases=2,
        )
        smoke = {"experiment": EXPERIMENT, "identity": identity, "backward": audit,
                 "passed": bool(identity["passed"] and audit["passed"]),
                 "config": _config(args)}
        _write_json(args.output / "smoke_audit.json", smoke)
        print(json.dumps(smoke, indent=2, default=_json_default), flush=True)
        if not smoke["passed"]:
            raise RuntimeError("R351 audit failed")
        return

    # Capture the immutable native R325 reference exactly once before any
    # student updates.  The aligned validation loader has the same image,
    # target, and padding as NativeWristDataset plus the seam field needed for
    # the paired ROI statistic.
    native_model = NnUNetMultiLabelPrior().to(device)
    native_model.load_state_dict(_load_state(native_payload, "model", args.native_checkpoint), strict=True)
    native_model.requires_grad_(False)
    native_model.eval()
    native_baseline = collect_native(
        native_model, val_loader, device, pairs, args.seam_threshold
    )
    del native_model, native_payload
    torch.cuda.empty_cache()

    # The R332 teacher is also immutable and is useful as the exact starting
    # reference for the dual-prior path.  It is evaluated once for provenance;
    # epoch rows always contain their own adapted official/flat values.
    teacher_baseline = collect_teacher(
        network, old_refiner, val_loader, device, pairs, args.steps, args.seam_threshold
    )
    native_flat = native_baseline["flat"]
    history_path = args.output / "history.jsonl"
    manifest_path = args.output / "checkpoint_manifest.jsonl"
    best_key: tuple[float, ...] | None = None
    best_row: dict[str, Any] | None = None
    best_hash: str | None = None

    for epoch in range(1, args.epochs + 1):
        aux_ramp = min(1.0, float(epoch) / float(args.aux_ramp_epochs))
        train_report = _train_epoch(
            epoch, network, overlap_prior, adapter, refiner, old_refiner,
            train_loader, optimizer, device, args.steps, auxiliary_weights,
            aux_ramp, args.max_aux_ratio, args.max_grad_norm,
        )
        adapted = collect_adapted(
            network, overlap_prior, adapter, refiner, val_loader, device, pairs,
            args.steps, args.seam_threshold,
        )
        accepted, checks, region_gate = native_gate(adapted["flat"], native_flat)
        structure_gate = bool(checks["structure_gate"])
        selection = _selection_key(accepted, adapted["flat"])
        row: dict[str, Any] = {
            "experiment": EXPERIMENT,
            "epoch": epoch,
            "evaluation_epoch": epoch,
            "accepted": bool(accepted),
            "gate": bool(accepted),
            "region_gate": bool(region_gate),
            "structure_gate": structure_gate,
            "gate_checks": checks,
            "selection_score": list(selection),
            "structure_score": structure_score_json(adapted["flat"]),
            "train": train_report,
            "official": {
                "native": native_baseline["official"],
                "r332_teacher": teacher_baseline["official"],
                "adapted": adapted["official"],
            },
            "flat": {
                "native": native_flat,
                "r332_teacher": teacher_baseline["flat"],
                "adapted": adapted["flat"],
            },
            "seam_roi": {
                "native": native_baseline["seam_roi"],
                "r332_teacher": teacher_baseline["seam_roi"],
                "adapted": adapted["seam_roi"],
            },
            "evaluation_metrics_sha256": _hash_json({
                "official": adapted["official"],
                "flat": adapted["flat"],
                "seam_roi": adapted["seam_roi"],
            }),
            "frozen_weights": frozen_weights,
        }

        candidate_path: Path | None = None
        candidate_hash: str | None = None
        if not accepted:
            candidate_path = args.output / f"candidate_epoch_{epoch:03d}.pth"
            candidate_hash = _save_checkpoint(
                candidate_path, epoch, row, adapter, refiner, frozen_weights,
                accepted=False, selection_key=selection,
            )

        last_path = args.output / "last.pth"
        last_hash = _save_checkpoint(
            last_path, epoch, row, adapter, refiner, frozen_weights,
            accepted=accepted, selection_key=selection,
        )

        best_updated = best_key is None or selection > best_key
        current_best_hash: str | None = None
        if best_updated:
            best_path = args.output / "best.pth"
            current_best_hash = _save_checkpoint(
                best_path, epoch, row, adapter, refiner, frozen_weights,
                accepted=accepted, selection_key=selection,
            )
            best_key = selection
            best_hash = current_best_hash

        row["checkpoint"] = {
            "evaluation_epoch": epoch,
            "evaluation_metrics_sha256": row["evaluation_metrics_sha256"],
            "last": {"path": str(last_path), "epoch": epoch, "sha256": last_hash},
            "best_updated": bool(best_updated),
            "best": ({"path": str(args.output / "best.pth"), "epoch": epoch,
                      "sha256": current_best_hash} if best_updated else None),
            "candidate": ({"path": str(candidate_path), "epoch": epoch,
                           "sha256": candidate_hash} if candidate_path is not None else None),
        }
        manifest = {
            "epoch": epoch,
            "evaluation_epoch": epoch,
            "evaluation_metrics_sha256": row["evaluation_metrics_sha256"],
            "accepted": bool(accepted),
            "files": row["checkpoint"],
        }
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, default=_json_default) + "\n")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")
        print(json.dumps(row, default=_json_default), flush=True)

        if best_updated:
            best_row = copy.deepcopy(row)

    if best_row is None or best_key is None:
        raise RuntimeError("R351 did not produce a selected checkpoint")
    result = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "split": "validation",
        "test_used": False,
        "accepted": bool(best_row["accepted"]),
        "best_epoch": int(best_row["epoch"]),
        "best_checkpoint_sha256": best_hash,
        "selection_score": list(best_key),
        "selection_rule": "strict native overall DSC/IoU plus 5% overlap/pair MSD reduction gate, then structural score",
        "native_baseline": native_baseline,
        "r332_teacher_baseline": teacher_baseline,
        "best": best_row,
        "checkpoints": {
            "best": str(args.output / "best.pth"),
            "last": str(args.output / "last.pth"),
            "manifest": str(manifest_path),
        },
        "frozen_weights": frozen_weights,
        "config": _config(args),
    }
    _write_json(args.output / "result.json", result)
    print(json.dumps({
        "status": result["status"],
        "accepted": result["accepted"],
        "best_epoch": result["best_epoch"],
        "best_checkpoint_sha256": result["best_checkpoint_sha256"],
    }, indent=2, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
