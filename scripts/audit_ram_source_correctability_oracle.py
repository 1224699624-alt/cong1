#!/usr/bin/env python3
"""Read-only RAM source-correctability oracle.

This script is deliberately an audit, rather than a training or model
selection entry point.  It runs a locked C0 checkpoint on RAM-W600 train and
validation independently, records the source/step1/step2 binary states, and
constructs the fixed hard-FN oracle

    H0 = valid & GT & GT-membership>=2 & C0-final-FN
         & source-logit < -5 - 1e-5.

The oracle result is only ``P_C0 | H0``.  It is labelled
``ORACLE_DIAGNOSTIC_NOT_MODEL_RESULT`` everywhere it is reported.

The normal invocation is GPU inference followed by CPU-only metric
evaluation.  Prediction caches contain packed binary masks and compact
diagnostic summaries; source logits are intentionally not retained.  A
second invocation with ``--eval-only --predictions-dir`` can therefore finish
or repeat the CPU evaluation without loading a model or using a GPU.

The implementation reuses the repository's NativeWristDataset, the R329
official evaluator, and the R332 flattening convention.  No optimizer is
created and no checkpoint or input is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Keep pure diagnostic helpers importable in a light-weight test env.
    import torch
except Exception:  # pragma: no cover - exercised only without PyTorch.
    torch = None  # type: ignore[assignment]

if torch is None:  # pragma: no cover - only for pure helper imports.
    def _inference_mode():
        def decorator(function: Any) -> Any:
            return function

        return decorator
else:
    _inference_mode = torch.inference_mode


ORACLE_LABEL = "ORACLE_DIAGNOSTIC_NOT_MODEL_RESULT"
PROBABILITY_THRESHOLD = 0.5
SOURCE_LOGIT_CUTOFF = -5.0 - 1e-5
EXPECTED_SPLIT_COUNTS = {"train": 425, "val": 69}
PACK_BITORDER = "little"
PREDICTION_STATES = ("source", "step1", "step2", "oracle")
MODEL_STATES = ("source", "step1", "step2")


def sha256(path: Path) -> str:
    """Return the SHA256 of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    """Hash a JSON value deterministically for cache provenance."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _as_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def resolve_path(value: Any, *, lock_path: Path, cwd: Path | None = None) -> Path:
    """Resolve a lock path while preserving absolute paths verbatim.

    Locks produced on the server often contain absolute paths.  Local smoke
    locks tend to contain paths relative to the lock or repository root, so we
    try those locations in a deterministic order.
    """

    raw = _as_path(value)
    if raw is None:
        raise ValueError(f"Expected a path, got {value!r}")
    if raw.is_absolute() and raw.exists():
        return raw
    candidates = [
        lock_path.parent / raw,
        (cwd or Path.cwd()) / raw,
        lock_path.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the lock-relative path for a useful later error message.
    return candidates[0].resolve()


def _extract_hash_entries(lock: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Extract common lock hash layouts into ``(path, sha256)`` pairs."""

    entries: list[tuple[str, str]] = []
    for container_name in (
        "file_hashes",
        "hashes",
        "input_hashes",
        "asset_hashes",
        "component_hashes",
    ):
        container = lock.get(container_name)
        if isinstance(container, Mapping):
            for raw_path, raw_hash in container.items():
                expected: Any = raw_hash
                if isinstance(raw_hash, Mapping):
                    expected = raw_hash.get("sha256", raw_hash.get("hash"))
                if isinstance(raw_path, str) and isinstance(expected, str):
                    entries.append((raw_path, expected))
        elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                raw_path = item.get("path", item.get("file"))
                expected = item.get("sha256", item.get("hash"))
                if isinstance(raw_path, str) and isinstance(expected, str):
                    entries.append((raw_path, expected))

    # Also accept the compact {checkpoint: ..., checkpoint_sha256: ...} form.
    direct_pairs = (
        ("checkpoint", "checkpoint_sha256"),
        ("c0_checkpoint", "c0_checkpoint_sha256"),
        ("reference_checkpoint", "reference_checkpoint_sha256"),
        ("dataset_manifest", "dataset_manifest_sha256"),
        ("code", "code_sha256"),
        ("script", "script_sha256"),
    )
    for path_key, hash_key in direct_pairs:
        raw_path, expected = lock.get(path_key), lock.get(hash_key)
        if isinstance(raw_path, str) and isinstance(expected, str):
            entries.append((raw_path, expected))

    # A lock may keep C0 under a nested object.
    c0 = lock.get("c0")
    if isinstance(c0, Mapping):
        raw_path = c0.get("checkpoint", c0.get("path"))
        expected = c0.get("sha256", c0.get("checkpoint_sha256"))
        if isinstance(raw_path, str) and isinstance(expected, str):
            entries.append((raw_path, expected))
    return entries


def verify_lock(lock_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Verify every file hash declared by a run lock.

    A diagnostic run without an input hash is not reproducible, so the lock is
    rejected when it contains no verifiable file hash.  Repeated paths are
    accepted only when their expected hashes agree.
    """

    if lock.get("test_used") is True:
        raise ValueError("A test-used lock cannot authorize the oracle")
    status = lock.get("status")
    status_text = str(status).upper() if status is not None else None
    if status_text is not None and not (
        status_text.startswith("LOCKED") or status_text in {"PASS", "READY"}
    ):
        raise ValueError(f"Lock status is not executable: {status!r}")

    entries = _extract_hash_entries(lock)
    if not entries:
        raise ValueError(
            "lock must declare file_hashes/hashes or a checkpoint SHA256"
        )
    checked: dict[str, str] = {}
    for raw_path, expected in entries:
        expected = expected.lower()
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid SHA256 in lock for {raw_path!r}")
        path = resolve_path(raw_path, lock_path=lock_path)
        if not path.is_file():
            raise FileNotFoundError(f"Locked file is missing: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
        key = str(path)
        old = checked.get(key)
        if old is not None and old != actual:
            raise ValueError(f"Conflicting hash entries for {path}")
        checked[key] = actual

    return {
        "status": "PASS",
        "lock_sha256": sha256(lock_path),
        "checked_files": checked,
        "checked_file_count": len(checked),
    }


def _nested_get(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def resolve_run_inputs(
    lock_path: Path,
    lock: Mapping[str, Any],
    *,
    dataset_override: Path | None = None,
    checkpoint_override: Path | None = None,
    model_checkpoint_override: Path | None = None,
    refiner_checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    """Resolve the dataset/checkpoint paths and architecture from a lock."""

    data_value = dataset_override or _nested_get(
        lock,
        "dataset_root",
        "dataset",
        "data_root",
        "ram_dataset",
    )
    checkpoint_value = checkpoint_override or _nested_get(
        lock,
        "checkpoint",
        "c0_checkpoint",
        "fresh100_checkpoint",
        "reference_checkpoint",
    )
    c0 = lock.get("c0")
    if isinstance(c0, Mapping):
        if checkpoint_value is None:
            checkpoint_value = _nested_get(c0, "checkpoint", "path")
    checkpoints = lock.get("checkpoints")
    if checkpoint_value is None and isinstance(checkpoints, Mapping):
        checkpoint_value = _nested_get(
            checkpoints, "c0", "fresh100", "reference", "model"
        )

    if data_value is None or checkpoint_value is None:
        raise ValueError(
            "lock must specify dataset_root and C0/fresh100 checkpoint paths"
        )
    dataset_root = resolve_path(data_value, lock_path=lock_path)
    checkpoint = resolve_path(checkpoint_value, lock_path=lock_path)
    model_checkpoint = model_checkpoint_override
    refiner_checkpoint = refiner_checkpoint_override
    if isinstance(checkpoints, Mapping):
        model_checkpoint = model_checkpoint or _as_path(
            _nested_get(checkpoints, "model", "model_checkpoint")
        )
        refiner_checkpoint = refiner_checkpoint or _as_path(
            _nested_get(checkpoints, "refiner", "refiner_checkpoint")
        )
    if model_checkpoint is not None:
        model_checkpoint = resolve_path(model_checkpoint, lock_path=lock_path)
    if refiner_checkpoint is not None:
        refiner_checkpoint = resolve_path(refiner_checkpoint, lock_path=lock_path)

    architecture = str(
        _nested_get(lock, "architecture", "model_kind", "checkpoint_kind") or "auto"
    ).lower()
    if architecture in {"v31", "ram_v31", "ram332_v31", "two_channel"}:
        architecture = "v31"
    elif architecture in {"r332", "r332_nnunet", "one_channel", "nnunet"}:
        architecture = "r332"
    elif architecture != "auto":
        raise ValueError(f"Unsupported RAM architecture in lock: {architecture}")

    counts = _nested_get(lock, "split_counts", "dataset_counts", "counts")
    split_counts = dict(EXPECTED_SPLIT_COUNTS)
    if isinstance(counts, Mapping):
        for split in split_counts:
            value = counts.get(split)
            if value is not None:
                split_counts[split] = int(value)

    if "cpu_threads" not in lock:
        raise ValueError("lock must declare cpu_threads; this oracle requires the locked thread count")
    cpu_threads = int(lock["cpu_threads"])
    if cpu_threads != 64:
        raise ValueError(f"RAM oracle requires exact locked cpu_threads=64, got {cpu_threads}")

    return {
        "dataset_root": dataset_root,
        "checkpoint": checkpoint,
        "model_checkpoint": model_checkpoint,
        "refiner_checkpoint": refiner_checkpoint,
        "architecture": architecture,
        "split_counts": split_counts,
        "max_seconds": float(lock.get("max_seconds", lock.get("runtime_budget_seconds", 7200))),
        "cpu_eval_max_seconds": float(lock.get("cpu_eval_max_seconds", lock.get("max_seconds", 7200))),
        "divisor": int(lock.get("padding_divisor", 32)),
        "seed": int(lock.get("seed", 3321)),
        "cpu_threads": cpu_threads,
    }


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for model inference")
    return torch


def pack_bool(mask: np.ndarray) -> np.ndarray:
    """Pack a boolean mask for compact, lossless cache storage."""

    array = np.asarray(mask, dtype=bool)
    return np.packbits(array.reshape(-1), bitorder=PACK_BITORDER)


def unpack_bool(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    shape_tuple = tuple(int(x) for x in shape)
    count = int(np.prod(shape_tuple, dtype=np.int64))
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8), bitorder=PACK_BITORDER, count=count
    ).reshape(shape_tuple).astype(bool, copy=False)


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if float(denominator) <= 0:
        return None
    return float(numerator) / float(denominator)


def _array_stats(value: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite source/refinement value")
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _membership_counts(mask: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    membership = np.asarray(mask, dtype=bool).sum(axis=0)
    region = np.asarray(valid, dtype=bool)
    return {
        "membership_0": int(((membership == 0) & region).sum()),
        "membership_1": int(((membership == 1) & region).sum()),
        "membership_2_plus": int(((membership >= 2) & region).sum()),
    }


def _state_counts(mask: np.ndarray, valid: np.ndarray) -> dict[str, int]:
    region = np.asarray(valid, dtype=bool)
    return {
        **_membership_counts(mask, region),
        "predicted_member_pixels": int(np.asarray(mask, dtype=bool)[:, region].sum()),
        "predicted_unique_pixels": int(
            (np.asarray(mask, dtype=bool).any(axis=0) & region).sum()
        ),
    }


def build_h0_oracle(
    target: np.ndarray,
    valid: np.ndarray,
    final_prediction: np.ndarray,
    source_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the fixed H0 set and the binary ``P | H0`` oracle.

    All arrays are expected in full native padded space.  ``valid`` may be
    either ``[H,W]`` or ``[1,H,W]``; target/prediction/source are
    ``[14,H,W]``.  GT is consumed only after the model forward has completed.
    """

    target_b = np.asarray(target, dtype=bool)
    final_b = np.asarray(final_prediction, dtype=bool)
    source = np.asarray(source_logits, dtype=np.float32)
    valid_b = np.asarray(valid, dtype=bool)
    if valid_b.ndim == 3 and valid_b.shape[0] == 1:
        valid_b = valid_b[0]
    if target_b.ndim != 3 or final_b.shape != target_b.shape or source.shape != target_b.shape:
        raise ValueError(
            f"Expected target/final/source [14,H,W], got "
            f"{target_b.shape}, {final_b.shape}, {source.shape}"
        )
    if valid_b.shape != target_b.shape[-2:]:
        raise ValueError(f"valid shape {valid_b.shape} != {target_b.shape[-2:]}")
    if not np.isfinite(source).all():
        raise FloatingPointError("source logits contain non-finite values")
    gt_membership = target_b.sum(axis=0)
    source_below = source < SOURCE_LOGIT_CUTOFF
    h0 = (
        valid_b[None]
        & target_b
        & (gt_membership[None] >= 2)
        & ~final_b
        & source_below
    )
    oracle = final_b | h0
    return {
        "h0": h0,
        "oracle": oracle,
        "gt_membership": gt_membership,
        "source_below_threshold": source_below,
    }


def _conditional_counts(
    target: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    region: np.ndarray,
) -> dict[str, int | float | None]:
    member_target = np.asarray(target, dtype=bool) & np.asarray(valid, dtype=bool)[None] & region
    fn = member_target & ~np.asarray(prediction, dtype=bool)
    return {
        "fn": int(fn.sum()),
        "gt_member_pixels": int(member_target.sum()),
        "fn_rate": _safe_rate(int(fn.sum()), int(member_target.sum())),
    }


def make_case_diagnostics(
    target: np.ndarray,
    valid: np.ndarray,
    source_logits: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    h0: np.ndarray,
) -> dict[str, Any]:
    """Return compact per-case source, membership, and conditional-FN stats."""

    target_b = np.asarray(target, dtype=bool)
    valid_b = np.asarray(valid, dtype=bool)
    if valid_b.ndim == 3 and valid_b.shape[0] == 1:
        valid_b = valid_b[0]
    h0_b = np.asarray(h0, dtype=bool)
    gt_membership = target_b.sum(axis=0)
    gt_unique = target_b.any(axis=0)
    h0_unique = h0_b.any(axis=0)
    source = np.asarray(source_logits, dtype=np.float32)
    source_values = source[:, valid_b]

    gt_categories = _membership_counts(target_b, valid_b)
    gt_info = {
        **gt_categories,
        "member_pixels": int(target_b[:, valid_b].sum()),
        "unique_pixels": int((gt_unique & valid_b).sum()),
        "overlap_unique_pixels": int(((gt_membership >= 2) & valid_b).sum()),
    }
    h0_info = {
        "member_pixels": int(h0_b.sum()),
        "unique_pixels": int((h0_unique & valid_b).sum()),
        "source_below_threshold_member_pixels": int(h0_b.sum()),
        "gt_overlap_member_pixels": int(
            (target_b & (gt_membership[None] >= 2) & valid_b[None]).sum()
        ),
    }
    outside_member = valid_b[None] & target_b & ~h0_b
    h0_member = h0_b
    outside_unique = valid_b & gt_unique & ~h0_unique
    h0_unique_region = valid_b & h0_unique

    state_info: dict[str, Any] = {}
    for state in PREDICTION_STATES:
        if state not in predictions:
            raise KeyError(f"missing prediction state {state}")
        pred = np.asarray(predictions[state], dtype=bool)
        if pred.shape != target_b.shape:
            raise ValueError(f"{state} prediction shape {pred.shape} != {target_b.shape}")
        fn = valid_b[None] & target_b & ~pred
        one_member = valid_b & (gt_membership == 1)
        overlap_member = valid_b & (gt_membership >= 2)
        member_regions = {
            "h0": _conditional_counts(target_b, pred, valid_b, h0_member),
            "outside_h0": _conditional_counts(target_b, pred, valid_b, outside_member),
            "all": _conditional_counts(target_b, pred, valid_b, valid_b[None]),
        }
        # Unique-pixel conditional errors are kept alongside member FN rates;
        # the former measures overlap-region failure as a spatial event.
        pred_unique = pred.any(axis=0)
        unique_fn = valid_b & gt_unique & ~pred_unique
        unique_regions = {
            "h0": {
                "fn": int((unique_fn & h0_unique_region).sum()),
                "gt_unique_pixels": int(h0_unique_region.sum()),
                "fn_rate": _safe_rate(int((unique_fn & h0_unique_region).sum()), int(h0_unique_region.sum())),
            },
            "outside_h0": {
                "fn": int((unique_fn & outside_unique).sum()),
                "gt_unique_pixels": int(outside_unique.sum()),
                "fn_rate": _safe_rate(int((unique_fn & outside_unique).sum()), int(outside_unique.sum())),
            },
            "all": {
                "fn": int(unique_fn.sum()),
                "gt_unique_pixels": int((valid_b & gt_unique).sum()),
                "fn_rate": _safe_rate(int(unique_fn.sum()), int((valid_b & gt_unique).sum())),
            },
        }
        state_info[state] = {
            "predicted_membership": _state_counts(pred, valid_b),
            "member_fn": member_regions,
            "unique_fn": unique_regions,
            "conditional_fn_by_gt_membership": {
                "membership_1": _conditional_counts(target_b, pred, valid_b, one_member[None]),
                "membership_2_plus": _conditional_counts(target_b, pred, valid_b, overlap_member[None]),
            },
            "fn_member_pixels": int(fn.sum()),
            "fn_unique_pixels": int(unique_fn.sum()),
        }

    return {
        "gt": gt_info,
        "h0": h0_info,
        "source": {
            "logit": _array_stats(source_values),
            "below_threshold_member_pixels": int(
                (source < SOURCE_LOGIT_CUTOFF)[:, valid_b].sum()
            ),
            "threshold": SOURCE_LOGIT_CUTOFF,
        },
        "states": state_info,
    }


def aggregate_case_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate count fields, then derive rates from aggregate denominators."""

    result: dict[str, Any] = {
        "cases": len(rows),
        "gt": {
            "member_pixels": 0,
            "unique_pixels": 0,
            "overlap_unique_pixels": 0,
            "membership_0": 0,
            "membership_1": 0,
            "membership_2_plus": 0,
        },
        "h0": {
            "member_pixels": 0,
            "unique_pixels": 0,
            "source_below_threshold_member_pixels": 0,
            "gt_overlap_member_pixels": 0,
        },
        "states": {},
    }
    for state in PREDICTION_STATES:
        result["states"][state] = {
            "predicted_membership": {
                "membership_0": 0,
                "membership_1": 0,
                "membership_2_plus": 0,
                "predicted_member_pixels": 0,
                "predicted_unique_pixels": 0,
            },
            "member_fn": {
                region: {"fn": 0, "gt_member_pixels": 0}
                for region in ("h0", "outside_h0", "all")
            },
            "unique_fn": {
                region: {"fn": 0, "gt_unique_pixels": 0}
                for region in ("h0", "outside_h0", "all")
            },
            "conditional_fn_by_gt_membership": {
                region: {"fn": 0, "gt_member_pixels": 0}
                for region in ("membership_1", "membership_2_plus")
            },
            "fn_member_pixels": 0,
            "fn_unique_pixels": 0,
        }

    def add_counts(dst: dict[str, int], src: Mapping[str, Any], keys: Iterable[str]) -> None:
        for key in keys:
            dst[key] += int(src.get(key, 0))

    for row in rows:
        add_counts(
            result["gt"],
            row["gt"],
            ("member_pixels", "unique_pixels", "overlap_unique_pixels", "membership_0", "membership_1", "membership_2_plus"),
        )
        add_counts(
            result["h0"],
            row["h0"],
            ("member_pixels", "unique_pixels", "source_below_threshold_member_pixels", "gt_overlap_member_pixels"),
        )
        for state in PREDICTION_STATES:
            out = result["states"][state]
            src = row["states"][state]
            add_counts(out["predicted_membership"], src["predicted_membership"], out["predicted_membership"].keys())
            for region in out["member_fn"]:
                add_counts(out["member_fn"][region], src["member_fn"][region], ("fn", "gt_member_pixels"))
            for region in out["unique_fn"]:
                add_counts(out["unique_fn"][region], src["unique_fn"][region], ("fn", "gt_unique_pixels"))
            for region in out["conditional_fn_by_gt_membership"]:
                add_counts(
                    out["conditional_fn_by_gt_membership"][region],
                    src["conditional_fn_by_gt_membership"][region],
                    ("fn", "gt_member_pixels"),
                )
            out["fn_member_pixels"] += int(src["fn_member_pixels"])
            out["fn_unique_pixels"] += int(src["fn_unique_pixels"])

    for state in PREDICTION_STATES:
        out = result["states"][state]
        for region in out["member_fn"].values():
            region["fn_rate"] = _safe_rate(region["fn"], region["gt_member_pixels"])
        for region in out["unique_fn"].values():
            region["fn_rate"] = _safe_rate(region["fn"], region["gt_unique_pixels"])
        for region in out["conditional_fn_by_gt_membership"].values():
            region["fn_rate"] = _safe_rate(region["fn"], region["gt_member_pixels"])
    return result


def _official_metrics_impl(data: Mapping[str, Mapping[str, np.ndarray]], pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    # Delayed import keeps pure mask/oracle tests independent of the heavy
    # nnU-Net dependency tree while still using the repository evaluator.
    from evaluate_r329_ram_official_metrics_val import official_metrics as evaluator

    return evaluator(data, list(pairs))


def official_metrics(data: Mapping[str, Mapping[str, np.ndarray]], pairs: Sequence[tuple[int, int]]) -> dict[str, Any]:
    """Call the unchanged R329 official evaluator."""

    return _official_metrics_impl(data, pairs)


def flatten(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Use the repository's R332 flattening convention for R329 metrics."""

    try:
        from train_r332_ram_joint_iterative_refinement import flatten as evaluator_flatten

        return evaluator_flatten(dict(metrics))
    except Exception:
        overall = metrics["overall_instance_metrics"]
        overlap = metrics["overlap_region_metrics"]
        pair = metrics["overlap_pair_intersection_metrics"]
        return {
            "overall_dsc": overall["dsc"],
            "overall_iou": 1.0 - overall["voe"],
            "overall_nsd_2px": overall["nsd_2px"],
            "overall_msd_px": overall["msd_px"],
            "overall_ravd": overall["ravd"],
            "overlap_dsc": overlap["dsc"],
            "overlap_iou": 1.0 - overlap["voe"],
            "overlap_nsd_2px": overlap["nsd_2px"],
            "overlap_msd_px": overlap["msd_px"],
            "overlap_ravd": overlap["ravd"],
            "pair_dsc": pair["dsc"],
            "pair_iou": 1.0 - pair["voe"],
            "pair_nsd_2px": pair["nsd_2px"],
            "pair_msd_px": pair["msd_px"],
            "pair_ravd": pair["ravd"],
            "pair_msd_fail_rate": pair["msd_fail_rate"],
        }


class OfficialAccumulator:
    """Streaming exact equivalent of R329 ``_aggregate`` output.

    A full RAM split can occupy several GB when all four binary states are
    materialized at once.  We invoke ``official_metrics`` for one case at a
    time, then merge the evaluator's per-section means using its row counts.
    MSD has a nullable value, so its finite count is recovered from the
    evaluator's MSD failure rate.
    """

    _SECTIONS = (
        "overall_instance_metrics",
        "overlap_region_metrics",
        "overlap_pair_intersection_metrics",
    )
    _MEAN_KEYS = ("dsc", "nsd_2px", "voe", "ravd")

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, float]] = {
            section: {
                "n": 0.0,
                **{f"sum_{key}": 0.0 for key in self._MEAN_KEYS},
                "msd_sum": 0.0,
                "msd_n": 0.0,
                "msd_fail_n": 0.0,
            }
            for section in self._SECTIONS
        }
        self.pair_count = 0

    def add(self, metrics: Mapping[str, Any]) -> None:
        for section in self._SECTIONS:
            row = metrics[section]
            n = int(row["n"])
            acc = self._rows[section]
            acc["n"] += n
            for key in self._MEAN_KEYS:
                value = row.get(key)
                if value is not None and math.isfinite(float(value)):
                    acc[f"sum_{key}"] += float(value) * n
            fail_rate = row.get("msd_fail_rate")
            fail_n = (float(fail_rate) * n) if fail_rate is not None and n > 0 else 0.0
            acc["msd_fail_n"] += fail_n
            if row.get("msd_px") is not None and math.isfinite(float(row["msd_px"])):
                acc["msd_sum"] += float(row["msd_px"]) * max(n - fail_n, 0.0)
                acc["msd_n"] += max(n - fail_n, 0.0)
        self.pair_count += int(metrics.get("overlap_pair_count", 0))

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for section, acc in self._rows.items():
            n = int(round(acc["n"]))
            row: dict[str, Any] = {"n": n}
            for key in self._MEAN_KEYS:
                row[key] = _safe_rate(acc[f"sum_{key}"], acc["n"])
            row["msd_px"] = _safe_rate(acc["msd_sum"], acc["msd_n"])
            row["msd_fail_rate"] = _safe_rate(acc["msd_fail_n"], acc["n"])
            out[section] = row
        out["overlap_pair_count"] = self.pair_count
        return out


def _import_native_dataset() -> Any:
    from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset

    return NativeWristDataset


def verify_native_protocol(
    dataset_root: Path,
    split: str,
    *,
    expected_count: int | None = None,
    divisor: int = 32,
) -> dict[str, Any]:
    """Verify canonical ``_R`` flip and native right/bottom padding exactly."""

    from PIL import Image

    NativeWristDataset = _import_native_dataset()
    dataset = NativeWristDataset(Path(dataset_root), split, augment=False, divisor=divisor)
    if expected_count is not None and len(dataset) != int(expected_count):
        raise ValueError(f"{split} count {len(dataset)} != locked {expected_count}")
    original_hw: dict[str, int] = {}
    padded_hw: dict[str, int] = {}
    right_flipped = 0
    for index, mask_path in enumerate(dataset.mask_files):
        image_path = Path(dataset_root) / "BoneSegmentation" / "images" / f"{mask_path.stem}.bmp"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image for {split}/{mask_path.stem}: {image_path}")
        raw_mask = (np.load(mask_path, mmap_mode="r") > 0).astype(bool)
        with Image.open(image_path) as image:
            width, height = image.size
            image_array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        if raw_mask.shape != (14, height, width):
            raise ValueError({"case": mask_path.stem, "mask": raw_mask.shape, "image": (height, width)})
        sample = dataset[index]
        sample_mask = np.asarray(sample["mask"].numpy(), dtype=bool)
        sample_image = np.asarray(sample["image"].numpy()[0], dtype=np.float32)
        sample_valid = np.asarray(sample["valid"].numpy()[0], dtype=bool)
        padded_h = ((height + divisor - 1) // divisor) * divisor
        padded_w = ((width + divisor - 1) // divisor) * divisor
        if sample_mask.shape != (14, padded_h, padded_w) or sample_image.shape != (padded_h, padded_w):
            raise ValueError(f"Unexpected padded shape for {mask_path.stem}")
        expected_valid = np.zeros((padded_h, padded_w), dtype=bool)
        expected_valid[:height, :width] = True
        if not np.array_equal(sample_valid, expected_valid):
            raise ValueError(f"valid mask is not top-left native area for {mask_path.stem}")
        expected_mask = raw_mask
        expected_image = image_array
        if mask_path.stem.endswith("_R"):
            expected_mask = np.ascontiguousarray(expected_mask[:, :, ::-1])
            expected_image = np.ascontiguousarray(expected_image[:, ::-1])
            right_flipped += 1
        if not np.array_equal(sample_mask[:, :height, :width], expected_mask):
            raise ValueError(f"target spatial flip mismatch for {mask_path.stem}")
        if np.any(sample_mask[:, height:, :]) or np.any(sample_mask[:, :, width:]):
            raise ValueError(f"target padding is not zero for {mask_path.stem}")
        mean, std = float(expected_image.mean()), float(expected_image.std())
        expected_image = (expected_image - mean) / (std + 1e-6)
        if not np.allclose(sample_image[:height, :width], expected_image, atol=2e-6, rtol=2e-5):
            raise ValueError(f"image preprocessing mismatch for {mask_path.stem}")
        if not np.allclose(sample_image[height:, :], 0.0) or not np.allclose(sample_image[:, width:], 0.0):
            raise ValueError(f"image padding is not zero for {mask_path.stem}")
        original_hw[f"{height}x{width}"] = original_hw.get(f"{height}x{width}", 0) + 1
        padded_hw[f"{padded_h}x{padded_w}"] = padded_hw.get(f"{padded_h}x{padded_w}", 0) + 1
        original_hw_value = sample["original_hw"].numpy().tolist()
        if [int(x) for x in original_hw_value] != [height, width]:
            raise ValueError(f"original_hw mismatch for {mask_path.stem}")
    return {
        "split": split,
        "num_images": len(dataset),
        "expected_num_images": expected_count,
        "original_hw_counts": original_hw,
        "padded_hw_counts": padded_hw,
        "right_flip_cases": right_flipped,
        "resize_used": False,
        "interpolation_used_for_spatial_size": False,
        "padding": "right/bottom zero padding to a multiple of the locked divisor",
        "augmentation": False,
    }


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    t = _require_torch()
    payload = t.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise ValueError(f"C0 checkpoint has no model state: {path}")
    return payload


class C0Runner:
    """FP32 inference wrapper for the two repository RAM checkpoint families."""

    def __init__(self, model: Any, refiner: Any, architecture: str, device: Any) -> None:
        self.model = model.float().to(device).eval()
        self.refiner = refiner.float().to(device).eval()
        self.architecture = architecture
        self.device = device
        for module in (self.model, self.refiner):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @_inference_mode()
    def predict(self, image: Any) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        t = _require_torch()
        image = image.to(self.device, dtype=t.float32)
        # The target is intentionally not an argument to this function.  The
        # second RAM channel is the protocol's explicit zero channel.
        if self.architecture == "v31":
            from r332_v31_port import iterative_refine_o

            zero_aux = t.zeros_like(image)
            source = self.model(t.cat((image, zero_aux), dim=1))
            if isinstance(source, tuple):
                source = source[0]
            states, deltas = iterative_refine_o(
                image, source, self.refiner, 2, zero_aux
            )
        else:
            from train_r332_ram_joint_iterative_refinement import iterative_refine

            source = self.model(image, False)
            if isinstance(source, tuple):
                source = source[0]
            states, deltas = iterative_refine(image, source, self.refiner, 2)
        source_cpu = source.detach().float().cpu().numpy()[0]
        if source_cpu.shape[0] != 14:
            raise ValueError(f"C0 source has {source_cpu.shape[0]} channels, expected 14")
        predictions = [
            (t.sigmoid(state.float()) >= PROBABILITY_THRESHOLD)
            .detach()
            .cpu()
            .numpy()[0]
            .astype(bool, copy=False)
            for state in states
        ]
        if len(predictions) != 3:
            raise ValueError(f"Expected source/step1/step2, got {len(predictions)} states")
        delta_arrays = [
            delta.detach().float().cpu().numpy() for delta in deltas
        ]
        return source_cpu, predictions, delta_arrays


def load_c0_runner(
    checkpoint: Path,
    *,
    architecture: str,
    device: Any,
    model_checkpoint: Path | None = None,
    refiner_checkpoint: Path | None = None,
) -> tuple[C0Runner, Mapping[str, Any], str]:
    """Instantiate and strictly load C0, without creating an optimizer."""

    t = _require_torch()
    payload = _load_checkpoint_payload(checkpoint)
    model_state = payload["model"]
    refiner_state: Mapping[str, Any]
    if model_checkpoint is not None:
        model_state = _load_checkpoint_payload(model_checkpoint)["model"]
    if refiner_checkpoint is not None:
        refiner_payload = _load_checkpoint_payload(refiner_checkpoint)
        refiner_state = refiner_payload.get("refiner", refiner_payload.get("model"))
        if not isinstance(refiner_state, Mapping):
            raise ValueError("refiner checkpoint has no refiner/model state")
    else:
        refiner_state = payload.get("refiner")  # type: ignore[assignment]
        if not isinstance(refiner_state, Mapping):
            raise ValueError("C0 checkpoint has no refiner state")

    if architecture == "auto":
        architecture = "r332" if "prior_residual.weight" in model_state else "v31"
    if architecture == "v31":
        from r332_v31_port import R332RefinerWithO, load_mature_with_zero_o
        from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel

        model = RamNnUNetTwoChannel()
        refiner = R332RefinerWithO()
        model.load_state_dict(model_state, strict=True)
        # Mature C0 checkpoints have a six-channel refiner input; a checkpoint
        # already containing the explicit zero O channel is also accepted.
        first = refiner_state.get("enc1.net.0.weight")
        if getattr(first, "shape", None) == (12, 6, 3, 3):
            load_mature_with_zero_o(refiner, refiner_state)
        else:
            refiner.load_state_dict(refiner_state, strict=True)
    elif architecture == "r332":
        from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior
        from train_r327_ram_native_instance_completion_prior import InstanceCompletionRefiner

        model = NnUNetMultiLabelPrior()
        refiner = InstanceCompletionRefiner()
        model.load_state_dict(model_state, strict=True)
        refiner.load_state_dict(refiner_state, strict=True)
    else:
        raise ValueError(f"Unknown architecture {architecture}")
    return C0Runner(model, refiner, architecture, device), payload, architecture


class RuntimeBudget:
    def __init__(self, start: float, maximum_seconds: float) -> None:
        self.start = start
        self.maximum_seconds = float(maximum_seconds)

    def check(self) -> None:
        elapsed = time.monotonic() - self.start
        if elapsed > self.maximum_seconds:
            raise TimeoutError(
                f"locked oracle runtime budget exceeded: {elapsed:.1f}s > {self.maximum_seconds:.1f}s"
            )


def _write_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _save_case_cache(
    path: Path,
    *,
    target: np.ndarray,
    valid: np.ndarray,
    source_prediction: np.ndarray,
    step1: np.ndarray,
    step2: np.ndarray,
    oracle: np.ndarray,
    h0: np.ndarray,
    source_below_threshold: np.ndarray,
    original_hw: Sequence[int],
) -> None:
    shape = np.asarray(target.shape, dtype=np.int64)
    _write_npz(
        path,
        shape=shape,
        original_hw=np.asarray([int(original_hw[0]), int(original_hw[1])], dtype=np.int64),
        target=pack_bool(target),
        valid=pack_bool(valid),
        source=pack_bool(source_prediction),
        step1=pack_bool(step1),
        step2=pack_bool(step2),
        oracle=pack_bool(oracle),
        h0=pack_bool(h0),
        source_below=pack_bool(source_below_threshold),
    )


def _save_h0_cache(
    path: Path,
    h0: np.ndarray,
    source_below_threshold: np.ndarray,
    original_hw: Sequence[int],
) -> None:
    _write_npz(
        path,
        shape=np.asarray(h0.shape, dtype=np.int64),
        original_hw=np.asarray([int(original_hw[0]), int(original_hw[1])], dtype=np.int64),
        h0=pack_bool(h0),
        source_below=pack_bool(source_below_threshold),
        label=np.asarray([ORACLE_LABEL]),
    )


def _delta_snapshots(deltas: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    result = []
    for index, delta in enumerate(deltas, 1):
        value = np.asarray(delta, dtype=np.float32)
        result.append({"step": index, "absolute": _array_stats(np.abs(value)), "signed": _array_stats(value)})
    return result


def infer_split(
    dataset: Any,
    split: str,
    runner: C0Runner,
    *,
    predictions_root: Path,
    h0_root: Path,
    expected_count: int,
    budget: RuntimeBudget,
    progress_every: int = 50,
) -> dict[str, Any]:
    """Run one split and write compact per-case caches/snapshot JSONL."""

    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    split_prediction_root = predictions_root / split
    split_h0_root = h0_root / split
    split_prediction_root.mkdir(parents=True, exist_ok=True)
    split_h0_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = predictions_root / f"{split}_snapshots.jsonl"
    snapshot_path.unlink(missing_ok=True)
    started = time.monotonic()
    forward_seconds = 0.0
    count = 0
    cases: list[str] = []
    with snapshot_path.open("w", encoding="utf-8") as snapshot_file:
        for batch in loader:
            budget.check()
            case = str(batch["case"][0])
            if case in cases:
                raise ValueError(f"duplicate case in {split}: {case}")
            image = batch["image"].float()
            tick = time.monotonic()
            source_logits, predictions, deltas = runner.predict(image)
            if torch is not None and runner.device.type == "cuda":
                torch.cuda.synchronize(runner.device)
            forward_seconds += time.monotonic() - tick
            # GT and valid are read only after the model forward, making the
            # no-GT-in-forward property explicit and auditable.
            target = np.asarray((batch["mask"].numpy() > 0.5)[0], dtype=bool)
            valid = np.asarray(batch["valid"].numpy()[0, 0], dtype=bool)
            original_hw = [int(x) for x in batch["original_hw"].numpy()[0].tolist()]
            built = build_h0_oracle(target, valid, predictions[2], source_logits)
            state_predictions = {
                "source": predictions[0],
                "step1": predictions[1],
                "step2": predictions[2],
                "oracle": built["oracle"],
            }
            diagnostics = make_case_diagnostics(
                target, valid, source_logits, state_predictions, built["h0"]
            )
            _save_case_cache(
                split_prediction_root / f"{case}.npz",
                target=target,
                valid=valid,
                source_prediction=predictions[0],
                step1=predictions[1],
                step2=predictions[2],
                oracle=built["oracle"],
                h0=built["h0"],
                source_below_threshold=built["source_below_threshold"],
                original_hw=original_hw,
            )
            _save_h0_cache(
                split_h0_root / f"{case}.npz",
                built["h0"],
                built["source_below_threshold"],
                original_hw,
            )
            snapshot = {
                "case": case,
                "split": split,
                "original_hw": original_hw,
                "padded_hw": [int(x) for x in target.shape[-2:]],
                "source_step": "source",
                "step1_step": "step1",
                "step2_step": "step2",
                "c0_final_state": "step2",
                "diagnostic_label": ORACLE_LABEL,
                "forward_dtype": "float32",
                "gt_used_in_forward": False,
                "source": diagnostics["source"],
                "step1": diagnostics["states"]["step1"],
                "step2": diagnostics["states"]["step2"],
                "source_state": diagnostics["states"]["source"],
                "oracle": diagnostics["states"]["oracle"],
                "h0": diagnostics["h0"],
                "gt": diagnostics["gt"],
                "refinement": _delta_snapshots(deltas),
            }
            snapshot_file.write(json.dumps(snapshot, allow_nan=False) + "\n")
            snapshot_file.flush()
            cases.append(case)
            count += 1
            if progress_every > 0 and (count % progress_every == 0 or count == expected_count):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "cases": count,
                            "expected": expected_count,
                            "forward_seconds": forward_seconds,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    if count != expected_count:
        raise ValueError(f"{split} inferred {count} cases, expected {expected_count}")
    return {
        "split": split,
        "cases": count,
        "case_ids": cases,
        "forward_seconds": forward_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "snapshot_file": str(snapshot_path),
        "prediction_dir": str(split_prediction_root),
        "h0_dir": str(split_h0_root),
    }


def _read_snapshots(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            case = str(row["case"])
            if case in rows:
                raise ValueError(f"duplicate snapshot case: {case}")
            rows[case] = row
    return rows


def _read_case_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        shape = tuple(int(x) for x in payload["shape"].tolist())
        expected_mask_shape = shape
        valid_shape = shape[-2:]
        arrays = {
            "target": unpack_bool(payload["target"], expected_mask_shape),
            "valid": unpack_bool(payload["valid"], valid_shape),
            "source": unpack_bool(payload["source"], expected_mask_shape),
            "step1": unpack_bool(payload["step1"], expected_mask_shape),
            "step2": unpack_bool(payload["step2"], expected_mask_shape),
            "oracle": unpack_bool(payload["oracle"], expected_mask_shape),
            "h0": unpack_bool(payload["h0"], expected_mask_shape),
            "source_below": unpack_bool(payload["source_below"], expected_mask_shape),
        }
    return arrays


def _read_h0_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        shape = tuple(int(x) for x in payload["shape"].tolist())
        original_hw = np.asarray(payload["original_hw"], dtype=np.int64).tolist()
        if payload.get("label", np.asarray([ORACLE_LABEL]))[0] != ORACLE_LABEL:
            raise ValueError(f"unexpected H0 cache label: {path}")
        return {
            "shape": np.asarray(shape, dtype=np.int64),
            "original_hw": np.asarray(original_hw, dtype=np.int64),
            "h0": unpack_bool(payload["h0"], shape),
            "source_below": unpack_bool(payload["source_below"], shape),
        }


def _merge_case_metrics(
    accumulators: Mapping[str, OfficialAccumulator],
    case: str,
    arrays: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    per_case: dict[str, Any] = {"case": case, "diagnostic_label": ORACLE_LABEL}
    target = arrays["target"]
    for state in PREDICTION_STATES:
        metrics = official_metrics(
            {case: {"pred": arrays[state], "target": target}}, pairs
        )
        accumulators[state].add(metrics)
        per_case[state] = {"official": metrics, "flatten": flatten(metrics)}
    return per_case


def evaluate_split(
    split: str,
    *,
    predictions_root: Path,
    pairs: Sequence[tuple[int, int]],
    expected_count: int,
    budget: RuntimeBudget,
    output_root: Path,
) -> dict[str, Any]:
    """Evaluate packed caches with the unchanged R329 evaluator, streaming."""

    split_prediction_root = predictions_root / split
    split_h0_root = predictions_root / "h0" / split
    snapshot_path = predictions_root / f"{split}_snapshots.jsonl"
    snapshots = _read_snapshots(snapshot_path)
    paths = sorted(split_prediction_root.glob("*.npz"))
    if len(paths) != expected_count:
        raise ValueError(f"{split} cache count {len(paths)} != {expected_count}")
    accumulators = {state: OfficialAccumulator() for state in PREDICTION_STATES}
    diag_rows: list[Mapping[str, Any]] = []
    per_case_path = output_root / f"{split}_per_case.jsonl"
    per_case_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with per_case_path.open("w", encoding="utf-8") as out:
        for index, path in enumerate(paths, 1):
            budget.check()
            case = path.stem
            arrays = _read_case_cache(path)
            h0_path = split_h0_root / path.name
            if not h0_path.is_file():
                raise FileNotFoundError(f"missing persisted train/val H0 cache: {h0_path}")
            h0_cache = _read_h0_cache(h0_path)
            if not np.array_equal(h0_cache["h0"], arrays["h0"]):
                raise ValueError(f"persisted H0 differs from prediction cache for {split}/{case}")
            if not np.array_equal(h0_cache["source_below"], arrays["source_below"]):
                raise ValueError(f"persisted source mask differs for {split}/{case}")
            if case not in snapshots:
                raise ValueError(f"missing snapshot for {split}/{case}")
            # The oracle is copied from the C0 final binary state exactly at
            # evaluation time; no GT/model recomputation can alter it.
            expected_oracle = arrays["step2"] | arrays["h0"]
            if not np.array_equal(expected_oracle, arrays["oracle"]):
                raise ValueError(f"oracle cache is not step2 | H0 for {split}/{case}")
            if np.any(arrays["h0"] & ~arrays["target"]):
                raise ValueError(f"H0 contains GT-negative pixels for {split}/{case}")
            if np.any(arrays["h0"] & ~arrays["valid"][None]):
                raise ValueError(f"H0 contains padded pixels for {split}/{case}")
            row = _merge_case_metrics(accumulators, case, arrays, pairs)
            row["snapshot"] = snapshots[case]
            out.write(json.dumps(row, allow_nan=False) + "\n")
            diag_rows.append(snapshots[case]["diagnostic_payload"] if "diagnostic_payload" in snapshots[case] else {
                "gt": snapshots[case]["gt"],
                "h0": snapshots[case]["h0"],
                "source": snapshots[case]["source"],
                "states": {
                    state: snapshots[case]["source_state" if state == "source" else state]
                    for state in PREDICTION_STATES
                },
            })
            if index % 50 == 0 or index == expected_count:
                print(json.dumps({"cpu_eval_split": split, "cases": index, "expected": expected_count}), flush=True)
    official = {state: accumulators[state].result() for state in PREDICTION_STATES}
    flat = {state: flatten(official[state]) for state in PREDICTION_STATES}
    return {
        "split": split,
        "cases": expected_count,
        "official_metrics": official,
        "flatten": flat,
        "diagnostics": aggregate_case_diagnostics(diag_rows),
        "per_case_file": str(per_case_path),
        "cpu_eval_seconds": time.monotonic() - started,
        "diagnostic_label": ORACLE_LABEL,
        "state_annotations": {
            "source": "C0_SOURCE_MODEL_PREDICTION",
            "step1": "C0_ONE_STEP_MODEL_PREDICTION",
            "step2": "C0_FINAL_MODEL_PREDICTION",
            "oracle": ORACLE_LABEL,
        },
    }


def _load_pairs(dataset_root: Path, predictions_root: Path, lock_pairs: Any = None) -> list[tuple[int, int]]:
    if lock_pairs is not None:
        pairs = [tuple(int(x) for x in pair) for pair in lock_pairs]
        if len(pairs) != 15:
            raise ValueError(f"locked pair list has {len(pairs)} entries, expected top 15")
        return pairs
    from train_r284_ramw600_overlap_prior import top_training_pairs

    pairs = top_training_pairs(dataset_root, 15)
    if len(pairs) != 15:
        raise ValueError(f"train top15 pair list has {len(pairs)} entries")
    return [(int(i), int(j)) for i, j in pairs]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _read_prediction_manifest(predictions_root: Path, lock_sha: str) -> dict[str, Any]:
    path = predictions_root / "prediction_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"prediction cache manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("lock_sha256") != lock_sha:
        raise ValueError("prediction cache was produced under a different lock")
    if manifest.get("test_used") is not False:
        raise ValueError("prediction cache is not train/val-only")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="locked JSON input manifest")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True, help="new output directory")
    parser.add_argument("--dataset-root", type=Path, help="optional lock-consistent dataset override")
    parser.add_argument("--checkpoint", type=Path, help="optional lock-consistent C0 checkpoint override")
    parser.add_argument("--model-checkpoint", type=Path, help="optional separate model checkpoint")
    parser.add_argument("--refiner-checkpoint", type=Path, help="optional separate refiner checkpoint")
    parser.add_argument("--architecture", choices=("auto", "v31", "r332"), default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--eval-only", action="store_true", help="evaluate an existing packed prediction cache")
    parser.add_argument("--predictions-dir", type=Path, help="cache root for --eval-only or an explicit cache root")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    lock_path = args.lock.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing oracle output: {out}")
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError("lock JSON must be an object")
    lock_report = verify_lock(lock_path, lock)
    inputs = resolve_run_inputs(
        lock_path,
        lock,
        dataset_override=args.dataset_root,
        checkpoint_override=args.checkpoint,
        model_checkpoint_override=args.model_checkpoint,
        refiner_checkpoint_override=args.refiner_checkpoint,
    )
    if args.architecture is not None:
        inputs["architecture"] = args.architecture
    if inputs["max_seconds"] <= 0 or inputs["cpu_eval_max_seconds"] <= 0:
        raise ValueError("runtime budgets must be positive")
    if torch is None:
        raise RuntimeError("PyTorch is required by the locked RAM oracle")
    torch.set_num_threads(inputs["cpu_threads"])
    if args.predictions_dir is not None:
        predictions_root = args.predictions_dir.resolve()
    else:
        predictions_root = out / "predictions"

    start = time.monotonic()
    protocol_reports: dict[str, Any] = {}
    if args.eval_only:
        manifest = _read_prediction_manifest(predictions_root, lock_report["lock_sha256"])
        pairs = [tuple(int(x) for x in pair) for pair in manifest["top_training_pairs"]]
        split_counts = manifest["split_counts"]
    else:
        dataset_root = inputs["dataset_root"]
        divisor = inputs["divisor"]
        split_counts = inputs["split_counts"]
        # The explicit count defaults enforce the requested RAM train425/val69
        # scope, while a lock may deliberately provide smoke counts.
        protocol_reports["train"] = verify_native_protocol(
            dataset_root, "train", expected_count=split_counts["train"], divisor=divisor
        )
        protocol_reports["val"] = verify_native_protocol(
            dataset_root, "val", expected_count=split_counts["val"], divisor=divisor
        )
        train_cases = set(
            Path(dataset_root / "BoneSegmentation" / "masks" / "train").glob("*.npy")
        )
        val_case_ids = {
            path.stem
            for path in Path(dataset_root / "BoneSegmentation" / "masks" / "val").glob("*.npy")
        }
        train_case_ids = {path.stem for path in train_cases}
        if train_case_ids & val_case_ids:
            raise ValueError("train/val case overlap detected")
        pairs = _load_pairs(dataset_root, predictions_root, lock.get("top_training_pairs"))
        out.mkdir(parents=True, exist_ok=False)
        predictions_root.mkdir(parents=True, exist_ok=False)
        _write_json(
            predictions_root / "prediction_manifest.json",
            {
                "schema": "RAM_SOURCE_CORRECTABILITY_ORACLE_CACHE_V1",
                "lock_sha256": lock_report["lock_sha256"],
                "dataset_root": str(dataset_root),
                "checkpoint": str(inputs["checkpoint"]),
                "checkpoint_sha256": sha256(inputs["checkpoint"]),
                "architecture": inputs["architecture"],
                "dtype": "float32",
                "threshold": PROBABILITY_THRESHOLD,
                "source_logit_cutoff": SOURCE_LOGIT_CUTOFF,
                "states": list(PREDICTION_STATES),
                "split_counts": split_counts,
                "top_training_pairs": [list(pair) for pair in pairs],
                "test_used": False,
                "optimizer_updates": 0,
                "cache_masks_packed": True,
                "source_logits_stored": False,
                "h0_persistence": "predictions/h0/<split>/<case>.npz",
            },
        )
        predictions_root.joinpath("h0").mkdir(parents=True, exist_ok=True)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device(args.device)
        torch.set_grad_enabled(False)
        runner, payload, architecture = load_c0_runner(
            inputs["checkpoint"],
            architecture=inputs["architecture"],
            device=device,
            model_checkpoint=inputs["model_checkpoint"],
            refiner_checkpoint=inputs["refiner_checkpoint"],
        )
        inputs["architecture"] = architecture
        budget = RuntimeBudget(start, inputs["max_seconds"])
        NativeWristDataset = _import_native_dataset()
        train_dataset = NativeWristDataset(dataset_root, "train", augment=False, divisor=inputs["divisor"])
        val_dataset = NativeWristDataset(dataset_root, "val", augment=False, divisor=inputs["divisor"])
        inference_reports = {}
        inference_reports["train"] = infer_split(
            train_dataset,
            "train",
            runner,
            predictions_root=predictions_root,
            h0_root=predictions_root / "h0",
            expected_count=split_counts["train"],
            budget=budget,
            progress_every=args.progress_every,
        )
        inference_reports["val"] = infer_split(
            val_dataset,
            "val",
            runner,
            predictions_root=predictions_root,
            h0_root=predictions_root / "h0",
            expected_count=split_counts["val"],
            budget=budget,
            progress_every=args.progress_every,
        )
        # Release all model/GPU state before the CPU evaluator starts.
        del train_dataset, val_dataset, runner, payload
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        protocol_reports["inference"] = inference_reports
    if args.eval_only:
        out.mkdir(parents=True, exist_ok=False)

    eval_budget = RuntimeBudget(time.monotonic(), inputs["cpu_eval_max_seconds"])
    split_results: dict[str, Any] = {}
    # Evaluate val first so the requested validation diagnostic is available
    # before the longer train425 CPU pass; both splits remain independent.
    for split in ("val", "train"):
        split_results[split] = evaluate_split(
            split,
            predictions_root=predictions_root,
            pairs=pairs,
            expected_count=int(split_counts[split]),
            budget=eval_budget,
            output_root=out,
        )

    result = {
        "status": "COMPLETE",
        "experiment": "RAM_SOURCE_CORRECTABILITY_ORACLE",
        "diagnostic_label": ORACLE_LABEL,
        "validation_performance_claim": False,
        "model_result": False,
        "test_used": False,
        "optimizer_updates": 0,
        "lock": lock_report,
        "lock_path": str(lock_path),
        "dataset_root": str(inputs["dataset_root"]),
        "checkpoint": {
            "path": str(inputs["checkpoint"]),
            "sha256": sha256(inputs["checkpoint"]),
            "label": "C0_fresh100",
            "architecture": inputs["architecture"],
        },
        "scope": {
            "splits": ["train", "val"],
            "split_counts": split_counts,
            "members": 14,
            "c0_final_state": "step2",
            "inference": "full14 FP32; native pixels; right/bottom padding only; canonical _R horizontal flip",
            "threshold": "sigmoid >= 0.5",
            "source_logit_cutoff": SOURCE_LOGIT_CUTOFF,
            "h0": "valid & GT positive & GT membership>=2 & C0 final FN & source<-5-1e-5",
            "oracle": "P_C0_final | H0; binary copy only",
            "gt_used_in_forward": False,
            "gt_read_after_forward": True,
        },
        "top_training_pairs": [list(pair) for pair in pairs],
        "protocol_reports": protocol_reports,
        "predictions_dir": str(predictions_root),
        "cpu_evaluation": True,
        "cpu_threads": inputs["cpu_threads"],
        "splits": split_results,
        "runtime_seconds": time.monotonic() - start,
    }
    _write_json(out / "result.json", result)
    print(json.dumps({"status": result["status"], "output": str(out), "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
