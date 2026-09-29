#!/usr/bin/env python3
"""Read-only fixed-H0 endpoint audit for the RAM final-logit target arm.

The training run is already complete when this script is invoked.  The script
loads the locked C0 checkpoint and the four locked arm checkpoints, runs the
repository's v3.1 full-14-member two-step refiner with the explicit O channel
set to zero, and records diagnostic responses on a fixed set of cases.

This is a diagnostic audit, not an official evaluator.  In particular, the
GT is read only after a model forward has finished, H0 is copied from the
packed oracle cache after the C0 endpoint has been checked, and no optimizer
or gradient operation is created.  The output is deliberately case-level so
that a positive H0 response can be separated from new errors elsewhere.

Typical invocation::

    python scripts/audit_ram_source_correctability_response.py \
        --lock /path/to/RESPONSE_LOCK.json \
        --out /path/to/new-response-audit

The lock is the authority for paths, hashes, split sizes, and the fixed
update/held-out case selection.  Existing output directories are never
overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Keep pure diagnostics importable in a light-weight test environment.
    import torch
except Exception:  # pragma: no cover - exercised only without PyTorch.
    torch = None  # type: ignore[assignment]


ARMS = ("FIN_F0S1",)
STATES = ("source", "step1", "step2")
EXPECTED_SPLIT_COUNTS = {"train": 425, "val": 69}
PACK_BITORDER = "little"
PROBABILITY_THRESHOLD = 0.5
SOURCE_CUTOFF = -5.0 - 1e-5
SOURCE_CROSSING = -5.0
QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
CONTROL_OTHER_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)
CONTROL_OWN_EDGES = np.asarray([0.0, 0.0001, 0.001, 0.003, 0.0067,
                                0.02, 0.05, 0.1, 0.25, 0.5, 1.0], dtype=np.float64)
ORACLE_LABEL = "ORACLE_DIAGNOSTIC_NOT_MODEL_RESULT"


def json_default(value: Any) -> Any:
    """JSON conversion for values used in trace and result records."""

    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_default),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_bool(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    return np.packbits(array.reshape(-1), bitorder=PACK_BITORDER)


def unpack_bool(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    shape_tuple = tuple(int(x) for x in shape)
    count = int(np.prod(shape_tuple, dtype=np.int64))
    return np.unpackbits(
        np.asarray(packed, dtype=np.uint8).reshape(-1),
        bitorder=PACK_BITORDER,
        count=count,
    ).reshape(shape_tuple).astype(bool, copy=False)


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _path_candidates(raw: Any, base: Path, repository_root: Path | None = None) -> list[Path]:
    path = Path(str(raw))
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend((base / path, Path.cwd() / path))
        if repository_root is not None:
            candidates.append(repository_root / path)
    # Server locks sometimes carry a path with a different drive/root.  The
    # basename fallback is intentionally not used: silently selecting a same-
    # named artifact would defeat input-lock validation.
    return candidates


def resolve_path(raw: Any, *, base: Path, repository_root: Path | None = None) -> Path:
    if raw is None:
        raise ValueError("missing path value")
    candidates = _path_candidates(raw, base, repository_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if not candidates:
        raise ValueError(f"invalid path value: {raw!r}")
    return candidates[0].resolve()


def _hash_value(value: Any) -> str | None:
    if isinstance(value, str):
        candidate = value.lower()
        return candidate if len(candidate) == 64 and all(c in '0123456789abcdef' for c in candidate) else None
    if isinstance(value, Mapping):
        for key in ("sha256", "sha", "hash"):
            candidate = _hash_value(value.get(key))
            if candidate is not None:
                return candidate
    return None


def _declared_hashes(lock: Mapping[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for container_key in ("file_hashes", "input_hashes", "hashes", "asset_hashes"):
        container = lock.get(container_key)
        if isinstance(container, Mapping):
            for raw_path, value in container.items():
                expected = _hash_value(value)
                if isinstance(raw_path, str) and expected:
                    entries.append((raw_path, expected))
        elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                raw_path = _first(item, "path", "file")
                expected = _hash_value(item)
                if isinstance(raw_path, str) and expected:
                    entries.append((raw_path, expected))
    # Compact locks often keep important assets at the top level.
    direct = (
        ("checkpoint", "checkpoint_sha256"),
        ("c0_checkpoint", "c0_checkpoint_sha256"),
        ("fresh100_checkpoint", "fresh100_checkpoint_sha256"),
        ("reference_checkpoint", "reference_checkpoint_sha256"),
        ("r332", "r332_sha256"),
        ("overlap", "overlap_sha256"),
        ("selection_and_schedule", "selection_and_schedule_sha256"),
        ("schedule_manifest", "schedule_sha256"),
        ("code_path", "code_sha256"),
    )
    for path_key, hash_key in direct:
        raw_path, expected = lock.get(path_key), lock.get(hash_key)
        if isinstance(raw_path, str) and isinstance(expected, str):
            entries.append((raw_path, expected.lower()))
    return entries


def _expected_for_path(
    lock: Mapping[str, Any],
    path: Path,
    labels: Sequence[str] = (),
) -> str | None:
    """Find an expected hash by exact path, lock label, or basename."""

    for label in labels:
        for key in (f"{label}_sha256", f"{label}_sha", f"{label}_hash"):
            value = _hash_value(lock.get(key))
            if value:
                return value
        nested = lock.get(label)
        value = _hash_value(nested)
        if value:
            return value
    text = str(path)
    raw_entries = _declared_hashes(lock)
    for raw, expected in raw_entries:
        if raw == text:
            return expected
    basename_matches = [expected for raw, expected in raw_entries if Path(raw).name == path.name]
    if len(set(basename_matches)) == 1 and basename_matches:
        return basename_matches[0]
    return None


def verify_input_hashes(
    lock_path: Path,
    lock: Mapping[str, Any],
    required: Mapping[str, Path],
) -> dict[str, Any]:
    """Verify declared hashes and require hashes for execution-critical files.

    ``required`` contains resolved files whose contents determine the audit.
    A missing hash for any such file is an error.  Additional entries in a
    lock are also checked, which prevents a stale code/checkpoint artifact
    from being silently ignored.
    """

    status = str(lock.get("status", "")).upper()
    if status and not (status.startswith("LOCKED") or status in {"READY", "PASS"}):
        raise ValueError(f"lock status is not executable: {lock.get('status')!r}")
    if lock.get("test_used") is True:
        raise ValueError("test-used lock cannot authorize a response audit")
    if "cpu_threads" not in lock:
        raise ValueError("lock must declare cpu_threads=64")
    if int(lock.get("cpu_threads", 64)) != 64:
        raise ValueError("response audit requires locked cpu_threads=64")
    if int(lock.get("split_counts", {}).get("train", 425)) != 425:
        raise ValueError("response audit requires RAM train425")
    if int(lock.get("split_counts", {}).get("val", 69)) != 69:
        raise ValueError("response audit requires RAM val69")
    architecture = str(lock.get("architecture", lock.get("model_kind", "v31"))).lower()
    if architecture not in {"v31", "ram_v31", "ram332_v31", "two_channel"}:
        raise ValueError(f"response audit requires v3.1 architecture, got {architecture!r}")
    if float(lock.get("threshold", PROBABILITY_THRESHOLD)) != PROBABILITY_THRESHOLD:
        raise ValueError("response audit requires locked sigmoid threshold 0.5")
    if int(lock.get("padding_divisor", lock.get("divisor", 32))) != 32:
        raise ValueError("response audit requires native RAM padding divisor 32")

    entries = _declared_hashes(lock)
    if not entries:
        raise ValueError("lock must declare file_hashes/input_hashes or direct SHA256 fields")
    checked: dict[str, str] = {}
    for raw_path, expected in entries:
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"invalid SHA256 for {raw_path!r}")
        path = resolve_path(raw_path, base=lock_path.parent, repository_root=Path(__file__).resolve().parents[1])
        if not path.is_file():
            raise FileNotFoundError(f"hashed input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
        checked[str(path)] = actual

    missing: list[str] = []
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"required input is missing: {label}: {path}")
        expected = _expected_for_path(lock, path, (label,))
        if expected is None:
            missing.append(label)
            continue
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for required {label}: {path}")
        checked[str(path)] = actual
    if missing:
        raise ValueError(f"lock has no declared SHA256 for required inputs: {missing}")
    return {
        "status": "PASS",
        "lock_sha256": sha256_file(lock_path),
        "checked_files": checked,
        "checked_file_count": len(checked),
    }


def _resolve_from_lock(
    lock: Mapping[str, Any], lock_path: Path, *keys: str
) -> Path | None:
    paths = lock.get("paths")
    values: list[Any] = []
    for key in keys:
        if key in lock:
            values.append(lock[key])
        if isinstance(paths, Mapping) and key in paths:
            values.append(paths[key])
    for value in values:
        if isinstance(value, Mapping):
            value = _first(value, "path", "checkpoint", "file")
        if value is None:
            continue
        return resolve_path(
            value,
            base=lock_path.parent,
            repository_root=Path(__file__).resolve().parents[1],
        )
    for container_key in ("checkpoints", "artifacts", "inputs"):
        container = lock.get(container_key)
        if not isinstance(container, Mapping):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, Mapping):
                value = _first(value, "path", "checkpoint", "file")
            if value is not None:
                return resolve_path(
                    value,
                    base=lock_path.parent,
                    repository_root=Path(__file__).resolve().parents[1],
                )
    return None


def _resolve_arm_paths(lock: Mapping[str, Any], lock_path: Path) -> dict[str, Path]:
    containers: list[Mapping[str, Any]] = []
    for key in ("arm_checkpoints", "diagnostic_final_paths", "arm_paths", "arms", "checkpoints"):
        value = lock.get(key)
        if isinstance(value, Mapping):
            nested = value.get("arms")
            containers.append(nested if isinstance(nested, Mapping) else value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            listed: dict[str, Any] = {}
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                name = _first(item, "arm", "name", "id")
                path = _first(item, "checkpoint", "path", "file")
                if name is not None and path is not None:
                    listed[str(name)] = path
            if listed:
                containers.append(listed)
    result: dict[str, Path] = {}
    for arm in ARMS:
        value: Any = None
        for container in containers:
            if arm in container:
                value = container[arm]
                break
            for alias in (arm.lower(), arm.replace("SRC_", ""), arm.replace("_", "-") ):
                if alias in container:
                    value = container[alias]
                    break
            if value is not None:
                break
        if isinstance(value, Mapping):
            value = _first(value, "checkpoint", "path", "file")
        if value is None:
            raise KeyError(f"lock has no checkpoint path for arm {arm}")
        result[arm] = resolve_path(
            value,
            base=lock_path.parent,
            repository_root=Path(__file__).resolve().parents[1],
        )
    return result


def resolve_inputs(lock_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _resolve_from_lock(lock, lock_path, "dataset_root", "dataset", "ram_dataset")
    c0 = _resolve_from_lock(
        lock,
        lock_path,
        "c0_checkpoint",
        "c0",
        "checkpoint",
        "fresh100_checkpoint",
        "reference_checkpoint",
    )
    r332 = _resolve_from_lock(lock, lock_path, "r332", "r332_checkpoint", "r332_source")
    overlap = _resolve_from_lock(
        lock, lock_path, "overlap", "overlap_checkpoint", "image_only_overlap"
    )
    oracle = _resolve_from_lock(
        lock,
        lock_path,
        "oracle_h0_root",
        "h0_predictions_root",
        "oracle_predictions_h0",
        "oracle_root",
        "h0_root",
        "oracle_v3",
    )
    selection = _resolve_from_lock(
        lock,
        lock_path,
        "selection_and_schedule",
        "selection_and_schedule_path",
        "selection_and_schedule_json",
        "selection_schedule",
        "selection_manifest",
        "schedule_selection",
        "schedule_selection_manifest",
        "selection",
    )
    if dataset is None or c0 is None or r332 is None or overlap is None:
        raise ValueError("lock must specify dataset_root, C0, r332, and overlap paths")
    if oracle is None:
        raise ValueError("lock must specify the packed H0 oracle root")
    if selection is None:
        raise ValueError("lock must specify SELECTION_AND_SCHEDULE.json")
    # Permit a lock to point at the oracle_v3 bundle or its predictions root,
    # while keeping the actual contract unambiguous.
    if oracle.name == "oracle_v3" or (oracle / "predictions" / "h0").is_dir():
        oracle = oracle / "predictions" / "h0"
    elif oracle.name == "predictions" or (oracle / "h0").is_dir():
        oracle = oracle / "h0"
    if not oracle.is_dir():
        raise FileNotFoundError(f"packed H0 oracle root is missing: {oracle}")
    arms = _resolve_arm_paths(lock, lock_path)
    split_counts = lock.get("split_counts", {})
    if not isinstance(split_counts, Mapping):
        split_counts = {}
    return {
        "dataset": dataset,
        "c0": c0,
        "r332": r332,
        "overlap": overlap,
        "oracle": oracle,
        "selection": selection,
        "arms": arms,
        "split_counts": {
            "train": int(split_counts.get("train", 425)),
            "val": int(split_counts.get("val", 69)),
        },
        "cpu_threads": int(lock.get("cpu_threads", 64)),
        "divisor": int(lock.get("padding_divisor", lock.get("divisor", 32))),
        "seed": int(lock.get("seed", 4421)),
        "max_seconds": float(lock.get("max_seconds", lock.get("runtime_budget_seconds", 7200))),
    }


def _selection_lists(selection_path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("SELECTION_AND_SCHEDULE.json must contain an object")
    source: Mapping[str, Any] = payload
    for key in ("selection", "case_selection", "train_selection"):
        if isinstance(source.get(key), Mapping):
            source = source[key]
            break
    update = source.get("update_cases")
    heldout = source.get("heldout_cases", source.get("holdout_cases"))
    if not isinstance(update, Sequence) or isinstance(update, (str, bytes)):
        raise ValueError("selection manifest must contain update_cases")
    if not isinstance(heldout, Sequence) or isinstance(heldout, (str, bytes)):
        raise ValueError("selection manifest must contain heldout_cases")
    update_cases = [str(case) for case in update]
    heldout_cases = [str(case) for case in heldout]
    if not update_cases or not heldout_cases:
        raise ValueError("update_cases and heldout_cases must both be non-empty")
    if len(set(update_cases)) != len(update_cases) or len(set(heldout_cases)) != len(heldout_cases):
        raise ValueError("selection manifest contains duplicate cases")
    if set(update_cases) & set(heldout_cases):
        raise ValueError("update_cases and heldout_cases overlap")
    return update_cases, heldout_cases


def _validate_selection_provenance(selection_path: Path, c0_checkpoint: Path) -> dict[str, Any]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("SELECTION_AND_SCHEDULE.json must contain an object")
    if payload.get("test_used") is True:
        raise ValueError("test-used selection manifest cannot authorize this audit")
    expected = _first(
        payload,
        "C0_sha256",
        "c0_sha256",
        "checkpoint_sha256",
        "fresh100_sha256",
    )
    actual = sha256_file(c0_checkpoint)
    if expected is not None and str(expected).lower() != actual:
        raise ValueError(
            f"selection manifest C0 hash mismatch: expected {expected}, got {actual}"
        )
    return {
        "path": str(selection_path),
        "sha256": sha256_file(selection_path),
        "c0_hash_declared": expected is not None,
        "c0_sha256": actual,
        "test_used": False,
    }


def load_h0_cache(path: Path, expected_shape: Sequence[int]) -> dict[str, Any]:
    """Read a packed H0 cache and normalize it to the native padded shape."""

    expected = tuple(int(x) for x in expected_shape)
    with np.load(path, allow_pickle=False) as archive:
        h0_key = "h0" if "h0" in archive else ("H0" if "H0" in archive else None)
        if "shape" not in archive or h0_key is None:
            raise ValueError(f"packed H0 cache requires shape and h0: {path}")
        shape = tuple(int(x) for x in np.asarray(archive["shape"]).reshape(-1).tolist())
        if len(shape) != 3 or shape[0] != 14:
            raise ValueError(f"invalid H0 shape {shape}: {path}")
        h0 = unpack_bool(archive[h0_key], shape)
        original_hw = None
        if "original_hw" in archive:
            original_hw = tuple(int(x) for x in np.asarray(archive["original_hw"]).reshape(-1).tolist())
            if len(original_hw) != 2:
                raise ValueError(f"invalid original_hw in {path}")
        source_below = None
        if "source_below" in archive:
            source_below = unpack_bool(archive["source_below"], shape)
        label = None
        if "label" in archive:
            value = np.asarray(archive["label"]).reshape(-1)
            label = str(value[0]) if value.size else None
        optional = {}
        for key in ("oracle", "source", "step1", "step2", "c0_final", "final"):
            if key in archive:
                optional[key] = unpack_bool(archive[key], shape)

    def normalize(array: np.ndarray | None) -> np.ndarray | None:
        if array is None:
            return None
        if array.shape == expected:
            return array
        if array.shape[0] != expected[0] or original_hw is None:
            raise ValueError(f"cache shape {array.shape} != expected {expected}: {path}")
        if array.shape[1:] != original_hw or any(array.shape[d] > expected[d] for d in (1, 2)):
            raise ValueError(f"cache shape/original_hw mismatch: {path}")
        padded = np.zeros(expected, dtype=bool)
        padded[:, : array.shape[1], : array.shape[2]] = array
        return padded

    h0 = normalize(h0)
    source_below = normalize(source_below)
    optional = {key: normalize(value) for key, value in optional.items()}
    if h0 is None:
        raise AssertionError("unreachable")
    return {
        "h0": h0,
        "source_below": source_below,
        "shape": expected,
        "original_hw": original_hw,
        "label": label,
        "optional": optional,
        "sha256": sha256_file(path),
    }


def _quantile_summary(values: np.ndarray | Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "quantiles": {}}
    if not np.isfinite(array).all():
        raise FloatingPointError("diagnostic values contain non-finite numbers")
    return {
        "n": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": {f"q{q:g}": float(np.quantile(array, q)) for q in QUANTILES},
    }


def _safe_rate(num: int | float, den: int | float) -> float | None:
    return None if float(den) <= 0 else float(num) / float(den)


def _mask_stats(
    values: np.ndarray, mask: np.ndarray | None = None, *, max_samples: int | None = None
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    if mask is not None:
        selection = np.asarray(mask, dtype=bool)
        if selection.shape != array.shape:
            try:
                selection = np.broadcast_to(selection, array.shape)
            except ValueError as exc:
                raise ValueError(f"mask shape {selection.shape} cannot broadcast to {array.shape}") from exc
        array = array[selection]
    population_n = int(array.size)
    stride = 1
    if max_samples is not None and population_n > max_samples:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        stride = (population_n + max_samples - 1) // max_samples
        array = array[::stride]
    result = _quantile_summary(array)
    if stride > 1:
        result["sampling"] = "deterministic_flat_stride_for_background_distribution_only"
        result["population_n"] = population_n
        result["sampling_stride"] = stride
    return result


def _logit_margin_stats(logits: np.ndarray, h0: np.ndarray) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float32)[np.asarray(h0, dtype=bool)]
    margin = values.astype(np.float64) - SOURCE_CROSSING
    return {
        "logit": _quantile_summary(values),
        "margin_to_minus5": _quantile_summary(margin),
        "deficit_below_minus5": _quantile_summary(-margin),
        "below_locked_cutoff": int((values < SOURCE_CUTOFF).sum()),
        "at_or_above_minus5": int((values >= SOURCE_CROSSING).sum()),
        "crossing_fraction": _safe_rate(int((values >= SOURCE_CROSSING).sum()), int(values.size)),
    }


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    result = np.searchsorted(edges, values, side="right") - 1
    return np.clip(result, 0, len(edges) - 2).astype(np.int16)


def make_matched_controls(
    target: np.ndarray,
    valid: np.ndarray,
    h0: np.ndarray,
    c0_source_logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match H0 member pixels to fixed GT-negative member controls.

    A control has the same member index and the same two ten-bin C0 *source*
    features: the member's own probability and the maximum probability among
    the other 13 members.  Matching is deterministic, per case, and does not
    inspect any arm checkpoint.

    Returned columns are ``member, h0_y, h0_x, control_y, control_x,
    exact_bin, reused``.  An unmatched row is represented by ``-1`` control
    coordinates and is retained for transparent counts.
    """

    target = np.asarray(target, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    h0 = np.asarray(h0, dtype=bool)
    logits = np.asarray(c0_source_logits, dtype=np.float32)
    if target.shape != h0.shape or logits.shape != target.shape:
        raise ValueError("target, H0, and C0 final logits must have the same [14,H,W] shape")
    if valid.shape != target.shape[-2:]:
        raise ValueError("valid shape does not match target")
    if torch is not None:
        probability = torch.sigmoid(
            torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
        ).numpy().astype(np.float32, copy=False)
    else:  # pragma: no cover - endpoint inference requires torch.
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -80.0, 80.0)))
    other = np.empty_like(probability)
    for member in range(target.shape[0]):
        values = probability.copy()
        values[member] = -np.inf
        other[member] = values.max(axis=0)
    own_bins = _bin_index(probability, CONTROL_OWN_EDGES)
    other_bins = _bin_index(other, CONTROL_OTHER_EDGES)

    h0_member, h0_y, h0_x = np.nonzero(h0)
    # Only bins observed at H0 positions can be requested.  Restricting the
    # candidate pool this way avoids sorting/uniquing every negative pixel in
    # all 14 members for every case, which matters on the 69-case val pass.
    needed_bins: dict[int, set[int]] = {}
    for member, y, x in zip(h0_member, h0_y, h0_x):
        code = int(own_bins[member, y, x]) * 10 + int(other_bins[member, y, x])
        needed_bins.setdefault(int(member), set()).add(code)
    groups: dict[tuple[int, int, int], np.ndarray] = {}
    for member, codes_needed in needed_bins.items():
        candidates = valid & ~target[member]
        candidate_flat = np.flatnonzero(candidates.reshape(-1))
        if candidate_flat.size == 0:
            continue
        code_grid = own_bins[member].astype(np.int16) * 10 + other_bins[member].astype(np.int16)
        candidate_codes = code_grid.reshape(-1)[candidate_flat]
        for code in sorted(codes_needed):
            selected = candidate_codes == code
            if not np.any(selected):
                continue
            flat = candidate_flat[selected]
            ys, xs = np.unravel_index(flat, valid.shape)
            groups[(member, code // 10, code % 10)] = np.stack(
                (ys, xs), axis=1
            ).astype(np.int32, copy=False)

    rows = np.full((h0_member.size, 7), -1, dtype=np.int32)
    used: dict[tuple[int, int, int], set[int]] = {}
    cursors: dict[tuple[int, int, int], int] = {}
    exact_count = fallback_count = reused_count = unmatched_count = 0

    def take(key: tuple[int, int, int], allow_reuse: bool = False) -> tuple[int, int] | None:
        values = groups.get(key)
        if values is None or values.size == 0:
            return None
        occupied = used.setdefault(key, set())
        cursor = cursors.get(key, 0)
        while cursor < len(values) and cursor in occupied:
            cursor += 1
        cursors[key] = cursor
        if cursor < len(values):
            occupied.add(cursor)
            cursors[key] = cursor + 1
            return int(values[cursor, 0]), int(values[cursor, 1])
        if allow_reuse:
            return int(values[0, 0]), int(values[0, 1])
        return None

    group_keys_by_member: dict[int, list[tuple[int, int, int]]] = {}
    for key in groups:
        group_keys_by_member.setdefault(key[0], []).append(key)
    for member in group_keys_by_member:
        group_keys_by_member[member].sort(key=lambda key: (key[1], key[2]))

    for row_index, (member, y, x) in enumerate(zip(h0_member, h0_y, h0_x)):
        own_bin = int(own_bins[member, y, x])
        other_bin = int(other_bins[member, y, x])
        key = (int(member), own_bin, other_bin)
        match = take(key)
        exact = 1
        reused = 0
        if match is None:
            exact = 0
            candidates = group_keys_by_member.get(int(member), [])
            candidates = sorted(
                candidates,
                key=lambda candidate: (
                    abs(candidate[1] - own_bin) + abs(candidate[2] - other_bin),
                    candidate[1],
                    candidate[2],
                ),
            )
            for candidate in candidates:
                match = take(candidate)
                if match is not None:
                    break
            if match is None:
                # A fixed matched control is preferable to silently dropping
                # an H0 member when the finite negative pool is exhausted.
                for candidate in candidates:
                    match = take(candidate, allow_reuse=True)
                    if match is not None:
                        reused = 1
                        break
            if match is not None:
                fallback_count += 1
        else:
            exact_count += 1
        if match is None:
            unmatched_count += 1
            continue
        if reused:
            reused_count += 1
        rows[row_index] = [int(member), int(y), int(x), int(match[0]), int(match[1]), exact, reused]

    metadata = {
        "h0_members": int(h0_member.size),
        "matched": int((rows[:, 3] >= 0).sum()),
        "exact_bin_matches": exact_count,
        "fallback_bin_matches": fallback_count,
        "reused_controls": reused_count,
        "unmatched": unmatched_count,
        "bin_edges": {"own": CONTROL_OWN_EDGES.tolist(), "other_max": CONTROL_OTHER_EDGES.tolist()},
        "feature_definition": "C0 source sigmoid own probability and max sigmoid probability of the other 13 members",
    }
    metadata["pair_sha256"] = hashlib.sha256(rows.tobytes()).hexdigest()
    return rows, metadata


def _capture_rng() -> tuple[Any, Any, Any, list[Any]]:
    if torch is None:
        return random.getstate(), np.random.get_state(), None, []
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
        [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )


def _restore_rng(state: tuple[Any, Any, Any, list[Any]]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    if torch is not None and state[2] is not None:
        torch.set_rng_state(state[2])
        if torch.cuda.is_available() and state[3]:
            torch.cuda.set_rng_state_all(state[3])


def _rng_digest(state: tuple[Any, Any, Any, list[Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state[0]).encode("utf-8"))
    numpy_state = state[1]
    digest.update(repr(numpy_state[0]).encode("utf-8"))
    digest.update(np.asarray(numpy_state[1]).tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
    if state[2] is not None:
        digest.update(state[2].numpy().tobytes())
    for value in state[3]:
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _module_state_digest(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SaturationProbe:
    """Low-cost forward hooks for the refiner gate and tanh residual."""

    def __init__(self, refiner: Any) -> None:
        self._handles: list[Any] = []
        self.total = {"gate": 0, "tanh": 0}
        self.saturated = {"gate_low": 0, "gate_high": 0, "tanh_abs_gt_0_99": 0}
        self.minimum = {"gate": math.inf, "tanh": math.inf}
        self.maximum = {"gate": -math.inf, "tanh": -math.inf}
        self.sum = {"gate": 0.0, "tanh": 0.0}
        self.available = False
        gate = getattr(refiner, "gate", None)
        residual = getattr(refiner, "residual", None)
        if gate is not None and hasattr(gate, "register_forward_hook"):
            self._handles.append(gate.register_forward_hook(self._gate_hook))
            self.available = True
        if residual is not None and hasattr(residual, "register_forward_hook"):
            self._handles.append(residual.register_forward_hook(self._residual_hook))
            self.available = True

    def _update(self, key: str, value: Any) -> None:
        array = value.detach().float()
        if key == "gate":
            array = torch.sigmoid(array)
        else:
            array = torch.tanh(array)
        flat = array.reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return
        self.total[key] += int(finite.numel())
        self.minimum[key] = min(self.minimum[key], float(finite.min()))
        self.maximum[key] = max(self.maximum[key], float(finite.max()))
        self.sum[key] += float(finite.double().sum())
        if key == "gate":
            self.saturated["gate_low"] += int((finite < 1e-3).sum())
            self.saturated["gate_high"] += int((finite > 1.0 - 1e-3).sum())
        else:
            self.saturated["tanh_abs_gt_0_99"] += int((finite.abs() > 0.99).sum())

    def _gate_hook(self, module: Any, inputs: Any, output: Any) -> None:
        self._update("gate", output)

    def _residual_hook(self, module: Any, inputs: Any, output: Any) -> None:
        self._update("tanh", output)

    def snapshot(self, reset: bool = True) -> dict[str, Any]:
        result = {
            "available": self.available,
            "gate": {
                "n": self.total["gate"],
                "min": None if self.total["gate"] == 0 else self.minimum["gate"],
                "max": None if self.total["gate"] == 0 else self.maximum["gate"],
                "mean": None if self.total["gate"] == 0 else self.sum["gate"] / self.total["gate"],
                "saturated_low": self.saturated["gate_low"],
                "saturated_high": self.saturated["gate_high"],
                "saturated_low_fraction": _safe_rate(self.saturated["gate_low"], self.total["gate"]),
                "saturated_high_fraction": _safe_rate(self.saturated["gate_high"], self.total["gate"]),
            },
            "tanh": {
                "n": self.total["tanh"],
                "min": None if self.total["tanh"] == 0 else self.minimum["tanh"],
                "max": None if self.total["tanh"] == 0 else self.maximum["tanh"],
                "mean": None if self.total["tanh"] == 0 else self.sum["tanh"] / self.total["tanh"],
                "saturated_abs_gt_0_99": self.saturated["tanh_abs_gt_0_99"],
                "saturated_abs_gt_0_99_fraction": _safe_rate(
                    self.saturated["tanh_abs_gt_0_99"], self.total["tanh"]
                ),
            },
        }
        if reset:
            self.total = {"gate": 0, "tanh": 0}
            self.saturated = {"gate_low": 0, "gate_high": 0, "tanh_abs_gt_0_99": 0}
            self.minimum = {"gate": math.inf, "tanh": math.inf}
            self.maximum = {"gate": -math.inf, "tanh": -math.inf}
            self.sum = {"gate": 0.0, "tanh": 0.0}
        return result

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@dataclass
class CaseContext:
    split: str
    subset: str
    case: str
    index: int
    shape: tuple[int, int, int]
    original_hw: tuple[int, int]
    h0_packed: np.ndarray
    c0_source_packed: np.ndarray
    c0_step1_packed: np.ndarray
    c0_step2_packed: np.ndarray
    c0_global_fn_packed: np.ndarray
    controls: np.ndarray
    control_metadata: dict[str, Any]
    h0_cache_sha256: str
    h0_source_margin: dict[str, Any]
    c0_source_below_cached: bool | None
    c0_verification: dict[str, Any]

    def mask(self, name: str) -> np.ndarray:
        values = {
            "h0": self.h0_packed,
            "c0_source": self.c0_source_packed,
            "c0_step1": self.c0_step1_packed,
            "c0_step2": self.c0_step2_packed,
            "c0_global_fn": self.c0_global_fn_packed,
        }
        if name not in values:
            raise KeyError(name)
        shape = self.shape if name != "c0_global_fn" else self.shape[1:]
        return unpack_bool(values[name], shape)


def _state_array(logits: Any) -> np.ndarray:
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("model/refiner output is not a tensor")
    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]
    if logits.ndim != 3 or logits.shape[0] != 14:
        raise ValueError(f"expected [14,H,W] logits, got {tuple(logits.shape)}")
    return logits.detach().float().cpu().numpy()


def _delta_array(delta: Any, shape: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(delta, torch.Tensor):
        raise TypeError("refiner delta is not a tensor")
    value = delta.detach().float()
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 3 or tuple(value.shape) != shape:
        raise ValueError(f"expected delta shape {shape}, got {tuple(value.shape)}")
    return value.cpu().numpy()


def forward_full14(model: Any, refiner: Any, image: Any, legacy: Any) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Forward an image only; GT is intentionally absent from this API."""

    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    image = image.to(next(model.parameters()).device, dtype=torch.float32)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    zero_o = torch.zeros_like(image)
    with torch.inference_mode():
        source = model(torch.cat((image, zero_o), dim=1))
        if isinstance(source, (tuple, list)):
            source = source[0]
        states, deltas = legacy.iterative_refine_o(
            image, source, refiner, 2, zero_o
        )
    source_array = _state_array(source)
    shape = tuple(int(x) for x in source_array.shape)
    state_arrays = [_state_array(value) for value in states]
    if len(state_arrays) != 3 or any(tuple(x.shape) != shape for x in state_arrays):
        raise ValueError("v3.1 iterative_refine_o did not return source/step1/step2 full14 states")
    delta_arrays = [_delta_array(value, shape) for value in deltas]
    if len(delta_arrays) != 2:
        raise ValueError("v3.1 iterative_refine_o did not return two deltas")
    return source_array, state_arrays, delta_arrays


def _prediction(logits: np.ndarray) -> np.ndarray:
    # Keep the repository evaluator's exact FP32 threshold path.  A direct
    # ``logits >= 0`` comparison is mathematically equivalent in real
    # arithmetic but can disagree for tiny negative logits after FP32 sigmoid
    # rounding (sigmoid(-1e-8) may round to exactly 0.5).
    if torch is None:  # pragma: no cover - endpoint inference requires torch.
        values = np.asarray(logits, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-values)) >= 0.5).astype(bool, copy=False)
    values = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
    return (torch.sigmoid(values) >= 0.5).numpy().astype(bool, copy=False)


def _array_digest(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def _state_delta_summary(
    delta: np.ndarray,
    valid: np.ndarray,
    h0: np.ndarray,
) -> dict[str, Any]:
    valid_members = np.asarray(valid, dtype=bool)[None]
    return {
        "all_valid": _mask_stats(delta, valid_members, max_samples=8192),
        "h0": _mask_stats(delta, h0),
        "h0_abs": _mask_stats(np.abs(delta), h0),
        "all_valid_abs": _mask_stats(np.abs(delta), valid_members, max_samples=8192),
        "signed_sum_h0": float(np.asarray(delta, dtype=np.float64)[h0].sum()),
        "positive_h0": int((np.asarray(delta)[h0] > 0).sum()),
        "negative_h0": int((np.asarray(delta)[h0] < 0).sum()),
    }


def _control_summary(controls: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict[str, Any]:
    valid_rows = controls[:, 3] >= 0 if controls.size else np.zeros(0, dtype=bool)
    matched = controls[valid_rows] if controls.size else np.empty((0, 7), dtype=np.int32)
    result: dict[str, Any] = {
        "h0_members": int(len(controls)),
        "matched": int(len(matched)),
        "exact_bin_matches": int((matched[:, 5] == 1).sum()) if len(matched) else 0,
        "reused_controls": int((matched[:, 6] == 1).sum()) if len(matched) else 0,
        "unmatched": int(len(controls) - len(matched)),
        "pair_sha256": hashlib.sha256(controls.tobytes()).hexdigest(),
        "states": {},
    }
    if not len(matched):
        return result
    members, h0_y, h0_x, control_y, control_x = [matched[:, i] for i in range(5)]
    for state, prediction in predictions.items():
        h0_positive = prediction[members, h0_y, h0_x]
        control_positive = prediction[members, control_y, control_x]
        result["states"][state] = {
            "h0_member_positive": int(h0_positive.sum()),
            "control_member_false_positive": int(control_positive.sum()),
            "h0_member_positive_rate": _safe_rate(int(h0_positive.sum()), len(matched)),
            "control_false_positive_rate": _safe_rate(int(control_positive.sum()), len(matched)),
            "paired_recovery_minus_control": int(h0_positive.sum() - control_positive.sum()),
        }
    return result


def _state_summary(
    state: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    h0: np.ndarray,
    c0_final: np.ndarray,
    c0_global_fn: np.ndarray,
) -> dict[str, Any]:
    valid_member = np.asarray(valid, dtype=bool)[None]
    target_member = np.asarray(target, dtype=bool)
    pred = np.asarray(prediction, dtype=bool)
    h0 = np.asarray(h0, dtype=bool)
    outside_h0 = valid_member & target_member & ~h0
    h0_gt = h0.sum()
    all_gt = (valid_member & target_member).sum()
    outside_gt = outside_h0.sum()
    member_fn = valid_member & target_member & ~pred
    fp = valid_member & ~target_member & pred
    unique_gt = valid & target_member.any(axis=0)
    unique_pred = pred.any(axis=0)
    unique_h0 = h0.any(axis=0)
    unique_fn = valid & unique_gt & ~unique_pred
    # A global overlap pixel is recovered when at least two members are
    # predicted, matching the locked overlap-FN definition used by the H0
    # grouping report.  ``unique_pred`` would count a one-member rescue.
    global_recovered = c0_global_fn & (pred.sum(axis=0) >= 2)
    # Error changes outside H0 are useful for spotting a compensating-looking
    # response that paid for recovery with a new error elsewhere.
    c0_member = np.asarray(c0_final, dtype=bool)
    return {
        "member_gt_pixels": int(all_gt),
        "member_gt_pixels_outside_h0": int(outside_gt),
        "member_fn_all": int(member_fn.sum()),
        "member_fn_h0": int((member_fn & h0).sum()),
        "member_fn_outside_h0": int((member_fn & outside_h0).sum()),
        "member_fn_rate_outside_h0": _safe_rate(int((member_fn & outside_h0).sum()), int(outside_gt)),
        "h0_member_pixels": int(h0_gt),
        "h0_member_recovered": int((h0 & pred).sum()),
        "h0_member_recovery_rate": _safe_rate(int((h0 & pred).sum()), int(h0_gt)),
        "fp_all_valid": int(fp.sum()),
        "fp_outside_h0": int(fp.sum()),
        "unique_gt_pixels": int(unique_gt.sum()),
        "unique_fn_all": int(unique_fn.sum()),
        "unique_fn_h0": int((unique_fn & unique_h0).sum()),
        "unique_global_overlap_fn_total": int(c0_global_fn.sum()),
        "unique_global_overlap_fn_recovered": int(global_recovered.sum()),
        "unique_global_overlap_fn_recovery_rate": _safe_rate(int(global_recovered.sum()), int(c0_global_fn.sum())),
        "new_member_fn_outside_h0_vs_c0": int((valid_member & target_member & c0_member & ~pred & ~h0).sum()),
        "member_fn_removed_outside_h0_vs_c0": int((valid_member & target_member & ~c0_member & pred & ~h0).sum()),
        "new_fp_vs_c0": int((valid_member & ~target_member & ~c0_member & pred).sum()),
        "fp_removed_vs_c0": int((valid_member & ~target_member & c0_member & ~pred).sum()),
        "logit_valid": _mask_stats(state, valid_member, max_samples=8192),
        "logit_h0": _mask_stats(state, h0),
    }


def _make_c0_context(
    *,
    split: str,
    subset: str,
    index: int,
    case: str,
    image: Any,
    sample: Mapping[str, Any],
    c0_source: np.ndarray,
    c0_states: list[np.ndarray],
    cache: Mapping[str, Any],
) -> CaseContext:
    # This function is called only after the image-only C0 forward has
    # completed.  GT enters here for validation and accounting.
    target = np.asarray(sample["mask"].numpy() > 0.5, dtype=bool)
    valid = np.asarray(sample["valid"].numpy()[0], dtype=bool)
    shape = tuple(int(x) for x in target.shape)
    if shape != tuple(int(x) for x in cache["h0"].shape):
        raise ValueError(f"H0 shape {cache['h0'].shape} does not match C0 target {shape} for {case}")
    h0 = np.asarray(cache["h0"], dtype=bool)
    c0_source_below = c0_source < SOURCE_CUTOFF
    c0_final = _prediction(c0_states[-1])
    gt_membership = target.sum(axis=0)
    computed_h0 = valid[None] & target & (gt_membership[None] >= 2) & ~c0_final & c0_source_below
    if not np.array_equal(computed_h0, h0):
        differing = int(np.count_nonzero(computed_h0 != h0))
        raise ValueError(f"C0/H0 oracle mismatch for {split}/{case}: {differing} member pixels")
    if cache.get("source_below") is not None and not np.array_equal(cache["source_below"], c0_source_below):
        raise ValueError(f"C0 source threshold mask mismatch for {split}/{case}")
    if cache.get("label") not in (None, ORACLE_LABEL):
        raise ValueError(f"unexpected oracle label for {split}/{case}: {cache['label']!r}")
    optional = cache.get("optional", {})
    for key in ("c0_final", "final", "step2"):
        if key in optional and not np.array_equal(optional[key], c0_final):
            raise ValueError(f"oracle cache {key} differs from C0 final mask for {split}/{case}")
    if "oracle" in optional and not np.array_equal(optional["oracle"], c0_final | h0):
        raise ValueError(f"oracle cache oracle mask differs from C0 final|H0 for {split}/{case}")
    if np.any(h0 & ~(target & valid[None])):
        raise ValueError(f"H0 is outside GT/valid pixels for {split}/{case}")
    controls, control_metadata = make_matched_controls(target, valid, h0, c0_source)
    c0_global_fn = valid & (gt_membership >= 2) & (c0_final.sum(axis=0) < 2)
    original_hw_value = np.asarray(sample["original_hw"].numpy()).reshape(-1).tolist()
    original_hw = tuple(int(x) for x in original_hw_value)
    c0_margin = _logit_margin_stats(c0_source, h0)
    cached_source_below = cache.get("source_below")
    return CaseContext(
        split=split,
        subset=subset,
        case=case,
        index=index,
        shape=shape,
        original_hw=original_hw,
        h0_packed=pack_bool(h0),
        c0_source_packed=pack_bool(_prediction(c0_source)),
        c0_step1_packed=pack_bool(_prediction(c0_states[1])),
        c0_step2_packed=pack_bool(c0_final),
        c0_global_fn_packed=pack_bool(c0_global_fn),
        controls=controls,
        control_metadata=control_metadata,
        h0_cache_sha256=str(cache["sha256"]),
        h0_source_margin=c0_margin,
        c0_source_below_cached=None if cached_source_below is None else bool(np.array_equal(cached_source_below, c0_source_below)),
        c0_verification={
            "h0_exact": True,
            "source_below_exact": cached_source_below is not None and bool(np.array_equal(cached_source_below, c0_source_below)),
            "optional_final_mask_checked": any(key in optional for key in ("c0_final", "final", "step2")),
            "optional_oracle_mask_checked": "oracle" in optional,
            "computed_h0_member_pixels": int(h0.sum()),
            "computed_h0_unique_pixels": int(h0.any(axis=0).sum()),
            "c0_global_overlap_fn_pixels": int(c0_global_fn.sum()),
            "c0_mask_digest": _array_digest([_prediction(c0_source), _prediction(c0_states[1]), c0_final]),
        },
    )


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or "model" not in payload or "refiner" not in payload:
        raise ValueError(f"checkpoint must contain model and refiner states: {path}")
    return payload


def _load_refiner_state(refiner: Any, state: Mapping[str, Any], legacy: Any) -> None:
    try:
        refiner.load_state_dict(state, strict=True)
    except RuntimeError:
        # A pre-explicit-O checkpoint is accepted only through the repository's
        # exact zero-O port.  It remains an Ozero response audit.
        from r332_v31_port import load_mature_with_zero_o

        load_mature_with_zero_o(refiner, state)


def load_pair_into(model: Any, refiner: Any, checkpoint: Path, legacy: Any) -> Mapping[str, Any]:
    payload = _load_checkpoint(checkpoint)
    model.load_state_dict(payload["model"], strict=True)
    _load_refiner_state(refiner, payload["refiner"], legacy)
    model.eval().requires_grad_(False)
    refiner.eval().requires_grad_(False)
    return payload


def _write_trace(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, allow_nan=False, default=json_default) + "\n")
    handle.flush()


def _new_aggregate() -> dict[str, Any]:
    return {
        "cases": 0,
        "h0_member_pixels": 0,
        "h0_unique_pixels": 0,
        "c0_global_overlap_fn_pixels": 0,
        "h0_source_crossing_c0": 0,
        "control_h0_members": 0,
        "control_matched": 0,
        "control_unmatched": 0,
        "control_reused": 0,
        "states": {state: {} for state in STATES},
        "source_margin_values": [],
        "gate_tanh": {"available_cases": 0, "gate_values": 0, "tanh_values": 0, "gate_low": 0, "gate_high": 0, "tanh_saturated": 0},
    }


def add_arm_row(aggregate: dict[str, Any], row: Mapping[str, Any]) -> None:
    aggregate["cases"] += 1
    h0 = row["h0"]
    aggregate["h0_member_pixels"] += int(h0["member_pixels"])
    aggregate["h0_unique_pixels"] += int(h0["unique_pixels"])
    aggregate["c0_global_overlap_fn_pixels"] += int(h0["c0_global_overlap_fn_pixels"])
    aggregate["h0_source_crossing_c0"] += int(h0["c0_source_at_or_above_minus5"])
    controls = row["matched_gt_negative_control"]
    for source, destination in (("h0_members", "control_h0_members"), ("matched", "control_matched"), ("unmatched", "control_unmatched"), ("reused_controls", "control_reused")):
        aggregate[destination] += int(controls[source])
    # The per-case logit summary is retained in the trace.  Pool the raw
    # margins only for the compact by-split summary.
    aggregate["source_margin_values"].extend(row["source"].pop("_raw_margin_values", []))
    state_count_keys = (
        "member_gt_pixels",
        "member_gt_pixels_outside_h0",
        "member_fn_all",
        "member_fn_h0",
        "member_fn_outside_h0",
        "h0_member_pixels",
        "h0_member_recovered",
        "fp_all_valid",
        "fp_outside_h0",
        "unique_gt_pixels",
        "unique_fn_all",
        "unique_fn_h0",
        "unique_global_overlap_fn_total",
        "unique_global_overlap_fn_recovered",
        "new_member_fn_outside_h0_vs_c0",
        "member_fn_removed_outside_h0_vs_c0",
        "new_fp_vs_c0",
        "fp_removed_vs_c0",
    )
    for state in STATES:
        for key in state_count_keys:
            value = row["states"][state].get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                aggregate["states"][state][key] = aggregate["states"][state].get(key, 0) + value
    saturation = row.get("gate_tanh", {})
    if saturation.get("available"):
        aggregate["gate_tanh"]["available_cases"] += 1
    gate = saturation.get("gate", {})
    tanh = saturation.get("tanh", {})
    aggregate["gate_tanh"]["gate_values"] += int(gate.get("n", 0) or 0)
    aggregate["gate_tanh"]["tanh_values"] += int(tanh.get("n", 0) or 0)
    aggregate["gate_tanh"]["gate_low"] += int(gate.get("saturated_low", 0) or 0)
    aggregate["gate_tanh"]["gate_high"] += int(gate.get("saturated_high", 0) or 0)
    aggregate["gate_tanh"]["tanh_saturated"] += int(tanh.get("saturated_abs_gt_0_99", 0) or 0)


def finalize_aggregate(value: dict[str, Any]) -> dict[str, Any]:
    output = dict(value)
    margins = np.asarray(output.pop("source_margin_values"), dtype=np.float64)
    output["source_margin_to_minus5_pooled"] = _quantile_summary(margins)
    for state in STATES:
        summary = output["states"][state]
        summary["h0_member_recovery_rate"] = _safe_rate(
            summary.get("h0_member_recovered", 0), summary.get("h0_member_pixels", 0)
        )
        summary["member_fn_rate_outside_h0"] = _safe_rate(
            summary.get("member_fn_outside_h0", 0), summary.get("member_gt_pixels_outside_h0", 0)
        )
        summary["unique_global_overlap_fn_recovery_rate"] = _safe_rate(
            summary.get("unique_global_overlap_fn_recovered", 0),
            summary.get("unique_global_overlap_fn_total", 0),
        )
    saturation = output["gate_tanh"]
    saturation["gate_low_fraction"] = _safe_rate(saturation["gate_low"], saturation["gate_values"])
    saturation["gate_high_fraction"] = _safe_rate(saturation["gate_high"], saturation["gate_values"])
    saturation["tanh_saturated_fraction"] = _safe_rate(saturation["tanh_saturated"], saturation["tanh_values"])
    return output


def _case_row(
    ctx: CaseContext,
    arm: str,
    target: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
    states: list[np.ndarray],
    deltas: list[np.ndarray],
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    h0 = ctx.mask("h0")
    c0_final = ctx.mask("c0_step2")
    c0_global_fn = ctx.mask("c0_global_fn")
    predictions = {state: _prediction(logit) for state, logit in zip(STATES, states)}
    state_rows = {
        state: _state_summary(logit, predictions[state], target, valid, h0, c0_final, c0_global_fn)
        for state, logit in zip(STATES, states)
    }
    margin_values = (source[h0] - SOURCE_CROSSING).astype(np.float64)
    source_report = _logit_margin_stats(source, h0)
    source_report["_raw_margin_values"] = margin_values.tolist()
    c0_source = ctx.mask("c0_source")
    controls = _control_summary(ctx.controls, predictions)
    return {
        "record_type": "arm_case_response",
        "diagnostic_label": ORACLE_LABEL,
        "arm": arm,
        "split": ctx.split,
        "subset": ctx.subset,
        "case": ctx.case,
        "shape": list(ctx.shape),
        "original_hw": list(ctx.original_hw),
        "no_gt_in_forward": True,
        "dtype": "float32",
        "O_channel": "zeros_like(image)",
        "h0": {
            "member_pixels": int(h0.sum()),
            "unique_pixels": int(h0.any(axis=0).sum()),
            "cache_sha256": ctx.h0_cache_sha256,
            "c0_source_below_locked_cutoff": int(ctx.h0_source_margin["below_locked_cutoff"]),
            "c0_source_at_or_above_minus5": int((ctx.h0_source_margin["at_or_above_minus5"] or 0)),
            "c0_global_overlap_fn_pixels": int(c0_global_fn.sum()),
            "c0_mask_digest": ctx.c0_verification["c0_mask_digest"],
        },
        "source": source_report,
        "states": state_rows,
        "step_deltas": {
            "step1": _state_delta_summary(deltas[0], valid, h0),
            "step2": _state_delta_summary(deltas[1], valid, h0),
            "returned_delta_vs_state_difference_max_abs": {
                "step1": float(np.max(np.abs(deltas[0] - (states[1] - states[0])))),
                "step2": float(np.max(np.abs(deltas[1] - (states[2] - states[1])))),
            },
        },
        "source_crossing_minus5": {
            state: {
                "at_or_above_minus5": int((logit[h0] >= SOURCE_CROSSING).sum()),
                "below_locked_cutoff": int((logit[h0] < SOURCE_CUTOFF).sum()),
            }
            for state, logit in zip(STATES, states)
        },
        "matched_gt_negative_control": controls,
        "gate_tanh": dict(telemetry),
        "c0_reference": {
            "source_prediction_member_pixels": int(c0_source.sum()),
            "final_prediction_member_pixels": int(c0_final.sum()),
            "final_h0_recovered": int((h0 & c0_final).sum()),
        },
        "model_state_observation": "read-only inference; no optimizer or parameter update",
    }


def _c0_trace_row(ctx: CaseContext) -> dict[str, Any]:
    h0 = ctx.mask("h0")
    c0_source = ctx.mask("c0_source")
    c0_step1 = ctx.mask("c0_step1")
    c0_step2 = ctx.mask("c0_step2")
    c0_global = ctx.mask("c0_global_fn")
    return {
        "record_type": "c0_oracle_verification",
        "diagnostic_label": ORACLE_LABEL,
        "arm": "C0_REFERENCE",
        "split": ctx.split,
        "subset": ctx.subset,
        "case": ctx.case,
        "shape": list(ctx.shape),
        "original_hw": list(ctx.original_hw),
        "no_gt_in_forward": True,
        "dtype": "float32",
        "O_channel": "zeros_like(image)",
        "h0": {
            "member_pixels": int(h0.sum()),
            "unique_pixels": int(h0.any(axis=0).sum()),
            "cache_sha256": ctx.h0_cache_sha256,
            "source_margin": ctx.h0_source_margin,
            "c0_global_overlap_fn_pixels": int(c0_global.sum()),
        },
        "c0_states": {
            "source_member_positive": int(c0_source.sum()),
            "step1_member_positive": int(c0_step1.sum()),
            "step2_member_positive": int(c0_step2.sum()),
        },
        "oracle_verification": ctx.c0_verification,
        "matched_gt_negative_control": ctx.control_metadata,
        "model_result": False,
        "optimizer_updates": 0,
    }


def _load_dataset(legacy: Any, root: Path, split: str, divisor: int, expected: int) -> Any:
    dataset = legacy.NativeWristDataset(root, split, augment=False, divisor=divisor)
    if len(dataset) != expected:
        raise ValueError(f"{split} dataset count {len(dataset)} != locked {expected}")
    return dataset


def _case_indices(dataset: Any) -> dict[str, int]:
    return {path.stem: index for index, path in enumerate(dataset.mask_files)}


def _h0_path(oracle_root: Path, split: str, case: str) -> Path:
    return oracle_root / split / f"{case}.npz"


def _validate_case_sets(
    train_dataset: Any,
    val_dataset: Any,
    update_cases: Sequence[str],
    heldout_cases: Sequence[str],
) -> dict[str, list[str]]:
    train_ids = {path.stem for path in train_dataset.mask_files}
    val_ids = {path.stem for path in val_dataset.mask_files}
    if train_ids & val_ids:
        raise ValueError("train and val case IDs overlap")
    for label, cases in (("update_cases", update_cases), ("heldout_cases", heldout_cases)):
        missing = sorted(set(cases) - train_ids)
        if missing:
            raise ValueError(f"{label} contain cases missing from train: {missing[:5]}")
    return {
        "train-update": list(update_cases),
        "train-heldout": list(heldout_cases),
        "val": sorted(val_ids),
    }


def _digest_h0_manifest(oracle_root: Path, split: str, cases: Sequence[str]) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    missing: list[str] = []
    for case in sorted(cases):
        path = _h0_path(oracle_root, split, case)
        if not path.is_file():
            missing.append(case)
            continue
        digest.update(case.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), missing


def _declared_manifest_digest(lock: Mapping[str, Any], split: str) -> str | None:
    for key in (
        f"h0_{split}_manifest_sha256",
        f"oracle_{split}_manifest_sha256",
        f"{split}_h0_manifest_sha256",
    ):
        value = lock.get(key)
        if isinstance(value, str):
            return value.lower()
    manifests = lock.get("oracle_manifest_digests")
    if isinstance(manifests, Mapping) and isinstance(manifests.get(split), str):
        return manifests[split].lower()
    return None


def _resolve_device(value: str) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="locked response-audit input JSON")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True, help="new output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out = args.out.resolve()
    # This check deliberately happens before lock/model reads: an audit can
    # never replace an earlier response artifact.
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    lock_path = args.lock.resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError("response lock must be a JSON object")
    inputs = resolve_inputs(lock_path, lock)
    arm_paths = inputs["arms"]
    required = {
        "c0": inputs["c0"],
        "r332": inputs["r332"],
        "overlap": inputs["overlap"],
        "selection": inputs["selection"],
        **{f"arm_{arm}": path for arm, path in arm_paths.items()},
    }
    lock_report = verify_input_hashes(lock_path, lock, required)
    if "test" in str(lock.get("dataset_split", "")).lower() or lock.get("test_used"):
        raise ValueError("test split/use is forbidden in the response audit")
    if any("TSRS" in str(path).upper() for path in required.values()):
        raise ValueError("TSRS inputs are forbidden in the response audit")

    update_cases, heldout_cases = _selection_lists(inputs["selection"])
    selection_provenance = _validate_selection_provenance(inputs["selection"], inputs["c0"])
    device = _resolve_device(args.device)
    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    previous_threads = torch.get_num_threads()
    rng_original = _capture_rng()
    torch.set_num_threads(64)
    seed = int(inputs["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng_before = _capture_rng()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    trace_path = out / "TRAINING_TRACE_OR_EQUIV.jsonl"
    limitations: list[str] = [
        "diagnostic counts only; official Overall/Overlap/Pair metrics are not computed here",
        "test split is not loaded or evaluated",
        "H0 and oracle masks are fixed C0 diagnostics, never model results",
        "GT is read after each image-only forward; no GT tensor is passed to model/refiner",
        "the final-target arm is evaluated with explicit O channel equal to zeros_like(image)",
    ]
    protocol: dict[str, Any] = {}
    references: dict[str, dict[str, Any]] = {}
    arm_counts: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    arm_state_records: dict[str, dict[str, Any]] = {}
    datasets: dict[str, Any] = {}
    contexts: dict[str, list[CaseContext]] = {}
    state_observations: dict[str, Any] = {}
    legacy = None
    model = refiner = generator = None
    try:
        # Importing this module supplies the exact repository model, loader,
        # and iterative_refine_o implementation used by the v3.1 run.
        import run_v31_ram_mature as legacy_module

        legacy = legacy_module
        datasets["train"] = _load_dataset(legacy, inputs["dataset"], "train", inputs["divisor"], 425)
        datasets["val"] = _load_dataset(legacy, inputs["dataset"], "val", inputs["divisor"], 69)
        selected = _validate_case_sets(
            datasets["train"], datasets["val"], update_cases, heldout_cases
        )
        protocol["dataset"] = {
            "root": str(inputs["dataset"]),
            "split_counts": {"train": len(datasets["train"]), "val": len(datasets["val"])},
            "selected_counts": {key: len(value) for key, value in selected.items()},
            "padding_divisor": inputs["divisor"],
            "augmentation": False,
            "dtype": "float32",
        }
        protocol["selection_and_schedule"] = selection_provenance
        index_maps = {split: _case_indices(dataset) for split, dataset in datasets.items()}
        for split, cases in (("train", update_cases + heldout_cases), ("val", selected["val"])):
            digest, missing = _digest_h0_manifest(inputs["oracle"], split, cases)
            if missing:
                raise FileNotFoundError(f"missing packed H0 caches for {split}: {missing[:5]}")
            expected_digest = _declared_manifest_digest(lock, split)
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(f"H0 manifest digest mismatch for {split}: {digest} != {expected_digest}")
            protocol.setdefault("oracle", {})[split] = {
                "root": str(inputs["oracle"]),
                "selected_cache_count": len(cases),
                "selected_cache_digest": digest,
                "declared_manifest_digest_checked": expected_digest is not None,
            }
        if _declared_manifest_digest(lock, "train") is None or _declared_manifest_digest(lock, "val") is None:
            limitations.append("per-cache H0 files were existence-checked and hashed; no lock-wide H0 manifest digest was declared for at least one split")

        # load_components is intentionally used even though every checkpoint
        # below carries model/refiner states: it locks the v3.1 architecture,
        # R332 mapping, and explicit-O refiner construction to the repository.
        overlap_sha = _expected_for_path(lock, inputs["overlap"], ("overlap",)) or sha256_file(inputs["overlap"])
        _, model, refiner, generator, backbone_mapping, refiner_mapping = legacy.load_components(
            SimpleNamespace(r332=inputs["r332"], overlap=inputs["overlap"], overlap_sha=overlap_sha),
            device,
        )
        del generator
        generator = None
        state_observations["architecture"] = "v31"
        state_observations["backbone_mapping"] = backbone_mapping
        state_observations["refiner_mapping"] = refiner_mapping
        c0_payload = load_pair_into(model, refiner, inputs["c0"], legacy)
        c0_model_digest = _module_state_digest(model)
        c0_refiner_digest = _module_state_digest(refiner)
        protocol["c0"] = {
            "path": str(inputs["c0"]),
            "sha256": sha256_file(inputs["c0"]),
            "updates_in_payload": c0_payload.get("updates"),
            "model_state_digest": c0_model_digest,
            "refiner_state_digest": c0_refiner_digest,
            "optimizer_created": False,
            "gt_used_in_forward": False,
        }

        train_index = index_maps["train"]
        val_index = index_maps["val"]
        context_plan = (("train", "train-update", selected["train-update"], train_index),
                        ("train", "train-heldout", selected["train-heldout"], train_index),
                        ("val", "val", selected["val"], val_index))
        with trace_path.open("w", encoding="utf-8") as trace:
            # C0 pass: each forward sees only image.  Target and oracle are
            # loaded/checked after the forward returns.
            for split, subset, cases, index_map in context_plan:
                rows: list[CaseContext] = []
                for case in cases:
                    if time.monotonic() - started > inputs["max_seconds"]:
                        raise TimeoutError("response audit runtime budget exceeded during C0 verification")
                    index = index_map[case]
                    sample = datasets[split][index]
                    image = sample["image"]
                    c0_source, c0_states, _ = forward_full14(model, refiner, image, legacy)
                    target_shape = tuple(int(x) for x in np.asarray(sample["mask"].shape))
                    cache_path = _h0_path(inputs["oracle"], split, case)
                    cache = load_h0_cache(cache_path, target_shape)
                    context = _make_c0_context(
                        split=split, subset=subset, index=index, case=case, image=image,
                        sample=sample, c0_source=c0_source, c0_states=c0_states, cache=cache,
                    )
                    rows.append(context)
                    _write_trace(trace, _c0_trace_row(context))
                contexts[subset] = rows
                references[subset] = {
                    "cases": len(rows),
                    "h0_member_pixels": int(sum(int(ctx.mask("h0").sum()) for ctx in rows)),
                    "h0_unique_pixels": int(sum(int(ctx.mask("h0").any(axis=0).sum()) for ctx in rows)),
                    "c0_global_overlap_fn_pixels": int(sum(int(ctx.mask("c0_global_fn").sum()) for ctx in rows)),
                    "oracle_verification_passed": True,
                }

            c0_digest_after = (_module_state_digest(model), _module_state_digest(refiner))
            if c0_digest_after != (c0_model_digest, c0_refiner_digest):
                raise RuntimeError("C0 model/refiner state changed during read-only verification")

            # Endpoint responses are independent: every arm starts from its
            # own immutable diagnostic_final.pth and no arm receives another
            # arm's state or optimizer/scaler.
            for arm in ARMS:
                arm_payload = load_pair_into(model, refiner, arm_paths[arm], legacy)
                model_digest = _module_state_digest(model)
                refiner_digest = _module_state_digest(refiner)
                arm_state_records[arm] = {
                    "path": str(arm_paths[arm]),
                    "sha256": sha256_file(arm_paths[arm]),
                    "updates_in_payload": arm_payload.get("updates"),
                    "model_state_digest_after_load": model_digest,
                    "refiner_state_digest_after_load": refiner_digest,
                    "optimizer_created": False,
                }
                arm_counts[arm] = {}
                for subset in ("train-update", "train-heldout", "val"):
                    aggregate = _new_aggregate()
                    split = "val" if subset == "val" else "train"
                    protocol.setdefault("gate_tanh_hook_cases", {})[subset] = contexts[subset][0].case
                    for case_number, context in enumerate(contexts[subset]):
                        # Hook every selected subset's first fixed case for
                        # scale/saturation.  The full14 logits, both residuals,
                        # H0 response, and negative-control counts remain
                        # exhaustive over all 97 cases for every arm.
                        probe = SaturationProbe(refiner) if case_number == 0 else None
                        try:
                            sample = datasets[split][context.index]
                            source, states, deltas = forward_full14(model, refiner, sample["image"], legacy)
                            target = np.asarray(sample["mask"].numpy() > 0.5, dtype=bool)
                            valid = np.asarray(sample["valid"].numpy()[0], dtype=bool)
                            if context.case != str(sample["case"]):
                                raise ValueError(f"dataset case order changed for {split}/{context.case}")
                            telemetry = (probe.snapshot(reset=True) if probe is not None else
                                         {"available": False, "reason": "hook sampled on first fixed case per subset"})
                            row = _case_row(context, arm, target, valid, source, states, deltas, telemetry)
                            add_arm_row(aggregate, row)
                            _write_trace(trace, row)
                        finally:
                            if probe is not None:
                                probe.close()
                    arm_counts[arm][subset] = finalize_aggregate(aggregate)
                if _module_state_digest(model) != model_digest or _module_state_digest(refiner) != refiner_digest:
                    raise RuntimeError(f"arm state changed during read-only inference: {arm}")

        state_observations["arm_checkpoints"] = arm_state_records
    finally:
        # Restore the caller's RNG state, including the state before the
        # deterministic audit seed was installed.
        _restore_rng(rng_original)
        torch.set_num_threads(previous_threads)
        if generator is not None:
            del generator
        if torch is not None and device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "status": "COMPLETE",
        "experiment": "RAM_FINAL_H0_TARGET_RESPONSE_AUDIT",
        "diagnostic_label": ORACLE_LABEL,
        "model_result": False,
        "official_metrics_computed": False,
        "test_used": False,
        "optimizer_updates": 0,
        "read_only": True,
        "lock": lock_report,
        "lock_path": str(lock_path),
        "inputs": {
            "dataset_root": str(inputs["dataset"]),
            "oracle_h0_root": str(inputs["oracle"]),
            "selection_and_schedule": str(inputs["selection"]),
            "c0_checkpoint": {"path": str(inputs["c0"]), "sha256": sha256_file(inputs["c0"])},
            "arms": {arm: {"path": str(path), "sha256": sha256_file(path)} for arm, path in arm_paths.items()},
            "r332": {"path": str(inputs["r332"]), "sha256": sha256_file(inputs["r332"])},
            "overlap": {"path": str(inputs["overlap"]), "sha256": sha256_file(inputs["overlap"])},
        },
        "scope": {
            "splits": ["train-update", "train-heldout", "val"],
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "selected_case_counts": {subset: len(value) for subset, value in selected.items()},
            "members": 14,
            "states": list(STATES),
            "steps": 2,
            "dtype": "float32",
            "cpu_threads": 64,
            "threshold": PROBABILITY_THRESHOLD,
            "source_locked_cutoff": SOURCE_CUTOFF,
            "forward": "run_v31_ram_mature architecture; iterative_refine_o full14; overlap/O input zeros_like(image)",
            "gt_used_in_forward": False,
            "h0": "packed oracle cache copied only after C0 final mask/source threshold exact verification",
            "control": "same member, GT-negative, binned C0 source own probability and other-member max; fixed per case independent of arm",
        },
        "protocol": protocol,
        "state_observations": state_observations,
        "c0_reference_by_split": references,
        "by_split_counts": {
            "c0_reference": references,
            "arms": {arm: {subset: value for subset, value in subsets.items()} for arm, subsets in arm_counts.items()},
        },
        "limitations": limitations,
        "trace": {
            "path": str(trace_path),
            "schema": "TRAINING_TRACE_OR_EQUIV_V1",
            "records": sum(len(rows) for rows in contexts.values()) * (1 + len(ARMS)),
            "includes_c0_oracle_verification": True,
            "includes_gate_tanh_hooks": True,
        },
        "rng": {
            "restored_after_audit": True,
            "original_digest": _rng_digest(rng_original),
            "audit_start_digest": _rng_digest(rng_before),
            "after_digest": _rng_digest(_capture_rng()),
            "preserved": _rng_digest(rng_original) == _rng_digest(_capture_rng()),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    write_json(out / "RESULT.json", result)
    write_json(out / "by_split_counts.json", result["by_split_counts"])
    print(json.dumps({"status": result["status"], "output": str(out), "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
