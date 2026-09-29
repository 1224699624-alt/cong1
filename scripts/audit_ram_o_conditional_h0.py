#!/usr/bin/env python3
"""Read-only conditional information audit for the RAM image-only prior.

The audit asks whether the frozen image-only O generator carries information at
the fixed C0 hard-FN set (H0) after conditioning on simple C0 and image
features.  It deliberately does not train a segmenter, choose a fusion point,
or report a segmentation-performance claim.

The executable path is locked with ``--lock`` and ``--out``.  The lock names
the fresh100 C0 checkpoint, the R332/v3.1 construction checkpoint, the frozen
image-only O checkpoint, the RAM dataset, and the packed H0 cache root.  C0 and
O are forwarded from the image alone; ground truth is read only after those
forwards have completed.  H0 is then checked against the fresh C0 FP32 source
and zero-O two-step final prediction.

The pure matching/readout helpers are intentionally usable without importing
the repository's nnU-Net dependency tree.  This keeps the focused synthetic
tests quick and makes the matching contract reviewable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:  # Tests exercise the pure helpers even on a light-weight environment.
    import torch
except Exception:  # pragma: no cover - endpoint execution requires torch.
    torch = None  # type: ignore[assignment]


ORACLE_LABEL = "ORACLE_DIAGNOSTIC_NOT_MODEL_RESULT"
DIAGNOSTIC_LABEL = "CONDITIONAL_O_INFORMATION_DIAGNOSTIC_NOT_SEGMENTATION_RESULT"
EXPECTED_SPLIT_COUNTS = {"train": 425, "val": 69}
N_MEMBERS = 14
PACK_BITORDER = "little"
PROBABILITY_THRESHOLD = 0.5
SOURCE_LOGIT_CUTOFF = -5.0 - 1e-5
DEFAULT_SEED = 4421
DEFAULT_MAX_SECONDS = 3600.0
DEFAULT_MAX_H0_ROWS_PER_CASE = 512
DEFAULT_MAX_CANDIDATES_PER_GROUP = 512
DEFAULT_BOOTSTRAPS = 1000

# Matching uses fixed bins only to build small deterministic candidate groups;
# the actual acceptance decision uses the fixed continuous calipers below.
PROBABILITY_EDGES = np.asarray(
    [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.97, 1.0],
    dtype=np.float32,
)
INTENSITY_EDGES = np.asarray(
    [-np.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf],
    dtype=np.float32,
)
GRADIENT_EDGES = np.asarray(
    [-np.inf, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.01],
    dtype=np.float32,
)
MATCH_FEATURE_NAMES = (
    "source_top1",
    "source_top2",
    "final_top1",
    "final_top2",
    "image_intensity",
    "image_gradient",
)
# Fixed before seeing the audit outputs.  These are deliberately wide enough
# to leave a measurable strict singleton pool while still conditioning on the
# C0/image summaries; all deviations and fallback counts remain reported.
DEFAULT_CALIPERS = np.asarray([0.10, 0.10, 0.10, 0.10, 0.50, 0.20], dtype=np.float32)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
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


def resolve_path(raw: Any, *, lock_path: Path) -> Path:
    """Resolve an absolute or lock-relative path without basename guessing."""

    if raw is None:
        raise ValueError("missing path in lock")
    value = Path(str(raw))
    candidates = [value] if value.is_absolute() else [lock_path.parent / value, Path.cwd() / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _hash_entries(lock: Mapping[str, Any]) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for container_key in ("file_hashes", "input_hashes", "hashes", "asset_hashes"):
        container = lock.get(container_key)
        if isinstance(container, Mapping):
            for raw_path, raw_hash in container.items():
                if isinstance(raw_hash, Mapping):
                    raw_hash = raw_hash.get("sha256", raw_hash.get("hash"))
                if isinstance(raw_path, str) and isinstance(raw_hash, str):
                    entries.append((raw_path, raw_hash.lower()))
        elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
            for item in container:
                if not isinstance(item, Mapping):
                    continue
                raw_path = item.get("path", item.get("file"))
                raw_hash = item.get("sha256", item.get("hash"))
                if isinstance(raw_path, str) and isinstance(raw_hash, str):
                    entries.append((raw_path, raw_hash.lower()))
    direct = (
        ("fresh100_checkpoint", "fresh100_checkpoint_sha256"),
        ("c0_checkpoint", "c0_checkpoint_sha256"),
        ("r332", "r332_sha256"),
        ("overlap_checkpoint", "overlap_sha256"),
        ("overlap", "overlap_sha256"),
        ("oracle_manifest", "oracle_manifest_sha256"),
        ("code_path", "code_sha256"),
        ("script", "script_sha256"),
    )
    for path_key, hash_key in direct:
        raw_path, raw_hash = lock.get(path_key), lock.get(hash_key)
        if isinstance(raw_path, str) and isinstance(raw_hash, str):
            entries.append((raw_path, raw_hash.lower()))
    return entries


def _expected_hash(lock: Mapping[str, Any], path: Path, labels: Sequence[str]) -> str | None:
    for label in labels:
        for key in (f"{label}_sha256", f"{label}_sha", f"{label}_hash"):
            value = lock.get(key)
            if isinstance(value, str) and len(value) == 64:
                return value.lower()
    matches = [value for raw, value in _hash_entries(lock) if raw == str(path)]
    if not matches:
        matches = [value for raw, value in _hash_entries(lock) if Path(raw).name == path.name]
    return matches[0] if len(set(matches)) == 1 else None


def _path_from_lock(lock: Mapping[str, Any], lock_path: Path, *keys: str) -> Path | None:
    containers = [lock]
    for name in ("paths", "inputs", "artifacts", "checkpoints"):
        value = lock.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in keys:
            if key not in container:
                continue
            value = container[key]
            if isinstance(value, Mapping):
                value = value.get("path", value.get("file", value.get("checkpoint")))
            if value is not None:
                return resolve_path(value, lock_path=lock_path)
    return None


def verify_lock(lock_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable protocol and all declared file hashes."""

    status = str(lock.get("status", "")).upper()
    if status and not (status.startswith("LOCKED") or status in {"READY", "PASS"}):
        raise ValueError(f"lock status is not executable: {lock.get('status')!r}")
    if lock.get("test_used") is True:
        raise ValueError("test-used lock cannot authorize a diagnostic run")
    if int(lock.get("cpu_threads", -1)) != 64:
        raise ValueError("RAM O audit requires locked cpu_threads=64")
    split_counts = lock.get("split_counts", {})
    if not isinstance(split_counts, Mapping):
        raise ValueError("split_counts must be an object")
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if int(split_counts.get(split, -1)) != expected:
            raise ValueError(f"requires RAM {split}{expected}")
    threshold = float(lock.get("threshold", PROBABILITY_THRESHOLD))
    cutoff = float(lock.get("source_cutoff", SOURCE_LOGIT_CUTOFF))
    if threshold != PROBABILITY_THRESHOLD or cutoff != SOURCE_LOGIT_CUTOFF:
        raise ValueError("threshold/source cutoff differ from locked FP32 protocol")
    if int(lock.get("padding_divisor", lock.get("divisor", 32))) != 32:
        raise ValueError("requires native RAM padding divisor 32")
    entries = _hash_entries(lock)
    if not entries:
        raise ValueError("lock must declare file_hashes or direct SHA256 fields")
    checked: dict[str, str] = {}
    for raw_path, expected in entries:
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"invalid SHA256 for {raw_path!r}")
        path = resolve_path(raw_path, lock_path=lock_path)
        if not path.is_file():
            raise FileNotFoundError(f"hashed input is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
        checked[str(path)] = actual
    return {
        "status": "PASS",
        "lock_sha256": sha256_file(lock_path),
        "checked_files": checked,
        "checked_file_count": len(checked),
    }


def _cache_path(root: Path, split: str, case: str) -> Path:
    candidates = [root / split / f"{case}.npz", root / f"{split}_{case}.npz", root / f"{case}.npz"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no packed H0 cache for {split}/{case} under {root}")


def unpack_bool(packed: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    shape_tuple = tuple(int(x) for x in shape)
    count = int(np.prod(shape_tuple, dtype=np.int64))
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8).reshape(-1), bitorder=PACK_BITORDER, count=count)
    return bits.reshape(shape_tuple).astype(bool, copy=False)


def load_h0_cache(path: Path, expected_shape: Sequence[int]) -> dict[str, Any]:
    """Load a packed H0 cache and pad a native-shape payload if necessary."""

    expected = tuple(int(x) for x in expected_shape)
    with np.load(path, allow_pickle=False) as archive:
        if "shape" not in archive or not ("h0" in archive or "H0" in archive):
            raise ValueError(f"packed H0 cache requires shape and h0: {path}")
        shape = tuple(int(x) for x in np.asarray(archive["shape"]).reshape(-1).tolist())
        if len(shape) != 3 or shape[0] != N_MEMBERS:
            raise ValueError(f"invalid H0 shape {shape}: {path}")
        key = "h0" if "h0" in archive else "H0"
        h0 = unpack_bool(archive[key], shape)
        original_hw = None
        if "original_hw" in archive:
            values = tuple(int(x) for x in np.asarray(archive["original_hw"]).reshape(-1).tolist())
            if len(values) != 2:
                raise ValueError(f"invalid original_hw in {path}")
            original_hw = values
        label = None
        if "label" in archive:
            values = np.asarray(archive["label"]).reshape(-1)
            label = str(values[0]) if values.size else None
        cached_source_below = None
        if "source_below" in archive:
            cached_source_below = unpack_bool(archive["source_below"], shape)

    if h0.shape != expected:
        if original_hw is None or h0.shape[0] != expected[0] or h0.shape[1:] != original_hw:
            raise ValueError(f"cache shape {h0.shape} != expected {expected}: {path}")
        padded = np.zeros(expected, dtype=bool)
        padded[:, : h0.shape[1], : h0.shape[2]] = h0
        h0 = padded
        if cached_source_below is not None:
            padded = np.zeros(expected, dtype=bool)
            padded[:, : cached_source_below.shape[1], : cached_source_below.shape[2]] = cached_source_below
            cached_source_below = padded
    if h0.shape != expected:
        raise ValueError(f"normalized H0 shape {h0.shape} != expected {expected}: {path}")
    return {
        "h0": h0,
        "source_below": cached_source_below,
        "label": label,
        "shape": expected,
        "sha256": sha256_file(path),
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return None if float(denominator) <= 0 else float(numerator) / float(denominator)


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "q": {}}
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite diagnostic values")
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "q": {f"q{q:g}": float(np.quantile(array, q)) for q in (0.05, 0.5, 0.95)},
    }


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.sum() == 0 or labels.sum() == labels.size:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    pos = labels == 1
    neg = ~pos
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum()))


def _logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = np.clip(np.asarray(scores, dtype=np.float64).reshape(-1), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return float("nan")
    return float(-(y * np.log(p) + (1.0 - y) * np.log1p(-p)).mean())


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2).astype(np.int16)


def _feature_key(count_class: int, top1_id: int, vector: np.ndarray) -> tuple[int, ...]:
    bins = [
        int(_bin_index(np.asarray([vector[0]], dtype=np.float32), PROBABILITY_EDGES)[0]),
        int(_bin_index(np.asarray([vector[1]], dtype=np.float32), PROBABILITY_EDGES)[0]),
        int(_bin_index(np.asarray([vector[2]], dtype=np.float32), PROBABILITY_EDGES)[0]),
        int(_bin_index(np.asarray([vector[3]], dtype=np.float32), PROBABILITY_EDGES)[0]),
        int(_bin_index(np.asarray([vector[4]], dtype=np.float32), INTENSITY_EDGES)[0]),
        int(_bin_index(np.asarray([vector[5]], dtype=np.float32), GRADIENT_EDGES)[0]),
    ]
    return (int(count_class), int(top1_id), *bins)


def _feature_key_matrix(count_class: np.ndarray, top1_id: np.ndarray,
                        vectors: np.ndarray) -> np.ndarray:
    """Vectorized version of ``_feature_key`` for candidate-pool indexing."""

    return np.column_stack((
        np.asarray(count_class, dtype=np.int16),
        np.asarray(top1_id, dtype=np.int16),
        _bin_index(vectors[:, 0], PROBABILITY_EDGES),
        _bin_index(vectors[:, 1], PROBABILITY_EDGES),
        _bin_index(vectors[:, 2], PROBABILITY_EDGES),
        _bin_index(vectors[:, 3], PROBABILITY_EDGES),
        _bin_index(vectors[:, 4], INTENSITY_EDGES),
        _bin_index(vectors[:, 5], GRADIENT_EDGES),
    )).astype(np.int16, copy=False)


def _candidate_groups(flat: np.ndarray, count_class: np.ndarray, top1_id: np.ndarray,
                      vectors: np.ndarray) -> dict[tuple[int, ...], np.ndarray]:
    if flat.size == 0:
        return {}
    rows = vectors[flat]
    keys = _feature_key_matrix(count_class[flat], top1_id[flat], rows)
    # A structured key keeps grouping deterministic without a Python dict per pixel.
    dtype = [(f"f{k}", np.int16) for k in range(keys.shape[1])]
    structured = np.asarray([tuple(row) for row in keys], dtype=dtype)
    order = np.argsort(structured, kind="mergesort", order=[f"f{k}" for k in range(keys.shape[1])])
    sorted_structured = structured[order]
    unique, starts = np.unique(sorted_structured, return_index=True)
    groups: dict[tuple[int, ...], np.ndarray] = {}
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else len(order)
        groups[tuple(int(sorted_structured[start][name]) for name in structured.dtype.names)] = flat[order[start:stop]]
    return groups


def _base_groups(flat: np.ndarray, count_class: np.ndarray,
                 top1_id: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for value in np.asarray(flat, dtype=np.int64):
        groups[(int(count_class[value]), int(top1_id[value]))].append(int(value))
    return {key: np.asarray(values, dtype=np.int64) for key, values in groups.items()}


def _cap_flat_indices(values: np.ndarray, cap: int) -> np.ndarray:
    values = np.sort(np.asarray(values, dtype=np.int64), kind="mergesort")
    if values.size <= cap:
        return values
    positions = np.linspace(0, values.size - 1, int(cap), dtype=np.int64)
    return values[positions]


def _nearest_unused(candidates: np.ndarray, query: np.ndarray, vectors: np.ndarray,
                    used: set[int], calipers: np.ndarray, strict: bool,
                    max_scan: int) -> tuple[int | None, float | None]:
    if candidates.size == 0:
        return None, None
    available = np.asarray([int(value) for value in candidates if int(value) not in used], dtype=np.int64)
    if available.size == 0:
        return None, None
    if available.size > max_scan:
        available = _cap_flat_indices(available, max_scan)
    differences = np.abs(vectors[available] - query[None, :])
    normalized = differences / np.maximum(calipers[None, :], 1e-8)
    quality = normalized.max(axis=1)
    if strict:
        keep = quality <= 1.0 + 1e-12
        if not np.any(keep):
            return None, float(quality.min())
        quality = np.where(keep, quality, np.inf)
    best = int(np.lexsort((available, quality))[0])
    if not np.isfinite(quality[best]):
        return None, float(quality.min())
    return int(available[best]), float(quality[best])


def make_conditional_matches(
    target: np.ndarray,
    valid: np.ndarray,
    h0: np.ndarray,
    matching_features: Mapping[str, np.ndarray],
    *,
    calipers: Sequence[float] = DEFAULT_CALIPERS,
    max_rows: int = DEFAULT_MAX_H0_ROWS_PER_CASE,
    max_candidate_scan: int = DEFAULT_MAX_CANDIDATES_PER_GROUP,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match H0 member rows to same-case GT-single foreground controls.

    The first pass requires exact ``C0 final member-count class`` and
    ``C0 final top-1 member ID`` plus the fixed continuous calipers.  It uses
    only C0/image features.  A deterministic nearest candidate is used after
    a caliper failure; if no singleton foreground exists, a broad GT-negative
    pool and finally deterministic reuse are reported separately.

    Columns are ``member, h0_y, h0_x, control_y, control_x, status,
    singleton, normalized_distance``.  ``status=0`` is unmatched,
    ``1`` exact singleton/caliper, ``2`` singleton fallback, ``3`` broad
    non-singleton fallback, and ``4`` reused control.
    """

    target = np.asarray(target, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    h0 = np.asarray(h0, dtype=bool)
    if target.shape != h0.shape or target.ndim != 3 or target.shape[0] != N_MEMBERS:
        raise ValueError("target and h0 must have shape [14,H,W]")
    if valid.shape != target.shape[1:]:
        raise ValueError("valid shape mismatch")
    required = ("count_class", "top1_id", *MATCH_FEATURE_NAMES)
    for name in required:
        if name not in matching_features:
            raise ValueError(f"missing matching feature {name}")
    count_class = np.asarray(matching_features["count_class"], dtype=np.int8).reshape(valid.shape)
    top1_id = np.asarray(matching_features["top1_id"], dtype=np.int8).reshape(valid.shape)
    vectors = np.stack([np.asarray(matching_features[name], dtype=np.float32).reshape(-1)
                        for name in MATCH_FEATURE_NAMES], axis=1)
    expected_n = int(np.prod(valid.shape))
    if vectors.shape[0] != expected_n or not np.isfinite(vectors).all():
        raise ValueError("matching feature arrays must be finite and match valid shape")
    calipers_array = np.asarray(calipers, dtype=np.float32).reshape(-1)
    if calipers_array.shape != (len(MATCH_FEATURE_NAMES),) or np.any(calipers_array <= 0):
        raise ValueError("calipers must be six positive values")
    membership = target.sum(axis=0)
    result_rows: list[list[float]] = []
    counters = Counter()
    # Cap the total member-row exposure per case, rather than multiplying the
    # cap by 14 members.  Flat ordering is fixed and therefore reproducible.
    h0_member_flat = np.flatnonzero(h0 & valid[None])
    if h0_member_flat.size > max_rows:
        h0_member_flat = _cap_flat_indices(h0_member_flat, max_rows)
    h0_member_indices = np.unravel_index(h0_member_flat, h0.shape)
    per_member_positive: dict[int, np.ndarray] = {
        member: np.asarray(h0_member_indices[1][h0_member_indices[0] == member] * valid.shape[1]
                           + h0_member_indices[2][h0_member_indices[0] == member], dtype=np.int64)
        for member in range(N_MEMBERS)
    }
    for member in range(N_MEMBERS):
        positive_flat = per_member_positive[member]
        if positive_flat.size == 0:
            continue
        singleton_mask = valid & (membership == 1) & ~target[member]
        broad_mask = valid & ~target[member]
        singleton_flat = np.flatnonzero(singleton_mask.reshape(-1))
        broad_flat = np.flatnonzero(broad_mask.reshape(-1))
        # A deterministic bounded candidate pool keeps the audit practical on
        # native 600px RAM images while preserving the full H0 row set cap.
        pool_cap = max(int(max_candidate_scan) * 2, int(max_rows) * 2)
        singleton_flat = _cap_flat_indices(singleton_flat, pool_cap)
        broad_flat = _cap_flat_indices(broad_flat, pool_cap)
        singleton_groups = _candidate_groups(singleton_flat, count_class.reshape(-1), top1_id.reshape(-1), vectors)
        broad_groups = _candidate_groups(broad_flat, count_class.reshape(-1), top1_id.reshape(-1), vectors)
        count_flat = count_class.reshape(-1)
        top1_flat = top1_id.reshape(-1)
        singleton_by_base = _base_groups(singleton_flat, count_flat, top1_flat)
        broad_by_base = _base_groups(broad_flat, count_flat, top1_flat)
        # One per-member occupancy set prevents a broad fallback from silently
        # reusing a coordinate already consumed by an exact singleton match.
        used_all: set[int] = set()
        for flat in positive_flat:
            y, x = np.unravel_index(int(flat), valid.shape)
            query = vectors[int(flat)]
            query_key = _feature_key(int(count_class[y, x]), int(top1_id[y, x]), query)
            control = None
            distance = None
            status = 0
            singleton = 0
            group = singleton_groups.get(query_key, np.empty(0, dtype=np.int64))
            control, distance = _nearest_unused(group, query, vectors, used_all, calipers_array, True, max_candidate_scan)
            if control is not None:
                used_all.add(control)
                status, singleton = 1, 1
                counters["exact_singleton_caliper"] += 1
            else:
                # Same exact count/top1 stratum, nearest continuous feature.
                base_key = (int(count_class[y, x]), int(top1_id[y, x]))
                base_singletons = singleton_by_base.get(base_key, np.empty(0, dtype=np.int64))
                control, distance = _nearest_unused(base_singletons, query, vectors, used_all, calipers_array, False, max_candidate_scan)
                if control is not None:
                    used_all.add(control)
                    status, singleton = 2, 1
                    counters["singleton_nearest_fallback"] += 1
            if control is None:
                group = broad_groups.get(query_key, np.empty(0, dtype=np.int64))
                control, distance = _nearest_unused(group, query, vectors, used_all, calipers_array, True, max_candidate_scan)
                if control is not None:
                    used_all.add(control)
                    status, singleton = 3, 0
                    counters["broad_non_singleton_caliper"] += 1
            if control is None:
                base_key = (int(count_class[y, x]), int(top1_id[y, x]))
                base_broad = broad_by_base.get(base_key, np.empty(0, dtype=np.int64))
                control, distance = _nearest_unused(base_broad, query, vectors, used_all, calipers_array, False, max_candidate_scan)
                if control is not None:
                    used_all.add(control)
                    status, singleton = 3, int(membership[np.unravel_index(control, valid.shape)] == 1)
                    counters["broad_nearest_fallback"] += 1
            if control is None:
                # Pool exhaustion is explicit.  Reuse is allowed only after
                # every non-reused candidate path has been exhausted.
                for pool_name, pool, pool_singleton in (
                    ("singleton", singleton_flat, 1),
                    ("broad", broad_flat, 0),
                ):
                    if pool.size == 0:
                        continue
                    # Every non-reused candidate was exhausted.  Passing a
                    # fresh occupancy set makes the reuse explicit and keeps
                    # it separate from the strict/fallback counts.
                    control, distance = _nearest_unused(pool, query, vectors, set(), calipers_array, False, max_candidate_scan)
                    if control is not None:
                        used_all.add(control)
                        status, singleton = 4, pool_singleton
                        counters[f"reused_{pool_name}"] += 1
                        break
            if control is None:
                counters["unmatched"] += 1
                result_rows.append([int(member), int(y), int(x), -1, -1, 0, 0, float("nan")])
                continue
            cy, cx = np.unravel_index(control, valid.shape)
            counters["matched"] += 1
            counters["singleton_matched"] += singleton
            result_rows.append([int(member), int(y), int(x), int(cy), int(cx), status, singleton, float(distance if distance is not None else math.nan)])
    rows = np.asarray(result_rows, dtype=np.float64).reshape(-1, 8)
    metadata = {
        "h0_member_rows_available": int(np.count_nonzero(h0 & valid[None])),
        "h0_member_rows_capped": int(len(h0_member_flat)),
        "rows": int(len(rows)),
        "matched": int(counters["matched"]),
        "singleton_matched": int(counters["singleton_matched"]),
        "unmatched": int(counters["unmatched"]),
        "status_counts": {str(key): int(value) for key, value in sorted(counters.items())},
        "calipers": calipers_array.tolist(),
        "pair_sha256": hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest(),
    }
    if len(rows):
        finite_distances = rows[np.isfinite(rows[:, 7]), 7]
        metadata["normalized_distance"] = _summary(finite_distances)
        metadata["strict_fraction_of_matched"] = _safe_rate(
            int(np.count_nonzero((rows[:, 5] == 1) & (rows[:, 6] == 1))),
            int(np.count_nonzero(rows[:, 3] >= 0)),
        )
    return rows, metadata


def feature_bundle(
    source_probability: np.ndarray,
    final_probability: np.ndarray,
    image: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the fixed C0/image features used by matching and readout."""

    source = np.asarray(source_probability, dtype=np.float32)
    final = np.asarray(final_probability, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    if source.shape != final.shape or source.ndim != 3 or source.shape[0] != N_MEMBERS:
        raise ValueError("source/final probabilities must be [14,H,W]")
    if image.shape != source.shape[1:]:
        raise ValueError("image shape mismatch")
    source_top = np.sort(source, axis=0)[-2:]
    final_top = np.sort(final, axis=0)[-2:]
    final_count = np.minimum((final >= PROBABILITY_THRESHOLD).sum(axis=0), 2).astype(np.int8)
    final_top1_id = final.argmax(axis=0).astype(np.int8)
    gx = np.zeros_like(image)
    gy = np.zeros_like(image)
    gx[:, :-1] = np.abs(image[:, 1:] - image[:, :-1])
    gy[:-1, :] = np.abs(image[1:, :] - image[:-1, :])
    gradient = np.sqrt(gx * gx + gy * gy)
    gradient /= max(float(gradient.max()), 1e-6)
    return {
        "count_class": final_count,
        "top1_id": final_top1_id,
        "source_top1": source_top[1],
        "source_top2": source_top[0],
        "final_top1": final_top[1],
        "final_top2": final_top[0],
        "image_intensity": image,
        "image_gradient": gradient.astype(np.float32),
        "source_probability": source,
        "final_probability": final,
    }


def _base_readout_features(bundle: Mapping[str, np.ndarray], member: np.ndarray,
                           y: np.ndarray, x: np.ndarray) -> np.ndarray:
    member = np.asarray(member, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    n = len(member)
    selected_source = np.asarray(bundle["source_probability"])[member, y, x]
    selected_final = np.asarray(bundle["final_probability"])[member, y, x]
    scalar_names = MATCH_FEATURE_NAMES
    scalars = np.column_stack([np.asarray(bundle[name])[y, x] for name in scalar_names])
    member_onehot = np.zeros((n, N_MEMBERS), dtype=np.float32)
    member_onehot[np.arange(n), member] = 1.0
    count = np.asarray(bundle["count_class"])[y, x].astype(np.float32)
    count_onehot = np.zeros((n, 3), dtype=np.float32)
    count_onehot[np.arange(n), np.clip(count.astype(np.int64), 0, 2)] = 1.0
    top1_onehot = np.zeros((n, N_MEMBERS), dtype=np.float32)
    top1 = np.asarray(bundle["top1_id"])[y, x].astype(np.int64)
    top1_onehot[np.arange(n), np.clip(top1, 0, N_MEMBERS - 1)] = 1.0
    return np.column_stack((selected_source, selected_final, scalars, count_onehot,
                            top1_onehot, member_onehot)).astype(np.float32)


def build_readout_rows(
    bundle: Mapping[str, np.ndarray], o_probability: np.ndarray, matches: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand valid pairs into positive/negative logistic rows."""

    matches = np.asarray(matches, dtype=np.float64)
    valid_rows = matches[:, 3] >= 0 if matches.size else np.zeros(0, dtype=bool)
    matched = matches[valid_rows].astype(np.int64)
    if matched.size == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float32)
    members = matched[:, 0]
    py, px, cy, cx = matched[:, 1], matched[:, 2], matched[:, 3], matched[:, 4]
    base_positive = _base_readout_features(bundle, members, py, px)
    base_negative = _base_readout_features(bundle, members, cy, cx)
    o = np.asarray(o_probability, dtype=np.float32)
    o_positive = o[py, px]
    o_negative = o[cy, cx]
    base = np.vstack((base_positive, base_negative))
    o_column = np.concatenate((o_positive, o_negative))[:, None]
    o_threshold = (o_column >= PROBABILITY_THRESHOLD).astype(np.float32)
    labels = np.concatenate((np.ones(len(matched), dtype=np.int8), np.zeros(len(matched), dtype=np.int8)))
    return base, labels, np.column_stack((o_column, o_threshold)).astype(np.float32)


class FixedLogisticReadout:
    """Small deterministic L2 logistic regression with train-only scaling."""

    def __init__(self, l2: float = 1.0, steps: int = 80) -> None:
        self.l2 = float(l2)
        self.steps = int(steps)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.iterations_: int = 0
        self.converged_: bool = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FixedLogisticReadout":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or len(x) != len(y) or len(x) == 0:
            raise ValueError("logistic fit requires a non-empty [N,D] matrix")
        if np.unique(y).size < 2:
            raise ValueError("logistic fit requires both labels")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        z = (x - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(z)), z))
        beta = np.zeros(design.shape[1], dtype=np.float64)
        for iteration in range(1, self.steps + 1):
            logits = np.clip(design @ beta, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-logits))
            weight = np.maximum(p * (1.0 - p), 1e-5)
            # Use a diagonal Newton step instead of ``np.linalg.solve``.
            # The audit is often run beside PyTorch's MKL/OpenMP runtime;
            # avoiding a second dense LAPACK call prevents platform-specific
            # duplicate-runtime aborts while retaining a deterministic convex
            # L2 readout.
            residual = p - y
            gradient = np.sum(design * residual[:, None], axis=0)
            gradient[1:] += self.l2 * beta[1:]
            hessian_diagonal = np.sum(design * design * weight[:, None], axis=0)
            hessian_diagonal[1:] += self.l2
            step = gradient / np.maximum(hessian_diagonal, 1e-8)
            step = np.clip(step, -5.0, 5.0)
            beta -= step
            if float(np.max(np.abs(step))) < 1e-7:
                self.iterations_ = iteration
                self.converged_ = True
                break
        if self.iterations_ == 0:
            self.iterations_ = self.steps
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("readout is not fitted")
        z = (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_
        logits = np.clip(self.intercept_ + z @ self.coef_, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    def summary(self) -> dict[str, Any]:
        if self.coef_ is None:
            return {"fitted": False}
        return {"fitted": True, "l2": self.l2, "steps": self.steps,
                "iterations": self.iterations_, "converged": self.converged_,
                "intercept": self.intercept_, "coef": self.coef_.tolist(),
                "mean": self.mean_.tolist(), "scale": self.scale_.tolist()}


def case_bootstrap(labels: np.ndarray, scores: np.ndarray, cases: np.ndarray,
                   *, seed: int = DEFAULT_SEED, repetitions: int = DEFAULT_BOOTSTRAPS) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    cases = np.asarray(cases).reshape(-1)
    unique = np.asarray(sorted(set(str(value) for value in cases)), dtype=object)
    if len(labels) != len(scores) or len(labels) != len(cases):
        raise ValueError("bootstrap arrays have different lengths")
    rng = np.random.default_rng(int(seed))
    auc_values: list[float] = []
    loss_values: list[float] = []
    by_case = {str(case): np.flatnonzero(cases.astype(str) == str(case)) for case in unique}
    for _ in range(int(repetitions)):
        draw = rng.integers(0, len(unique), size=len(unique))
        indexes = np.concatenate([by_case[str(unique[index])] for index in draw]) if len(draw) else np.empty(0, dtype=np.int64)
        if indexes.size == 0:
            continue
        value = _auc(labels[indexes], scores[indexes])
        if value is not None:
            auc_values.append(value)
        loss_values.append(_logloss(labels[indexes], scores[indexes]))
    def interval(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "q025": None, "q50": None, "q975": None}
        array = np.asarray(values, dtype=np.float64)
        return {"n": int(len(array)), "q025": float(np.quantile(array, 0.025)),
                "q50": float(np.quantile(array, 0.5)), "q975": float(np.quantile(array, 0.975))}
    return {"cases": int(len(unique)), "repetitions": int(repetitions), "seed": int(seed),
            "auc": interval(auc_values), "logloss": interval(loss_values)}


def paired_case_bootstrap(labels: np.ndarray, base_scores: np.ndarray,
                          o_scores: np.ndarray, cases: np.ndarray, *,
                          seed: int = DEFAULT_SEED,
                          repetitions: int = DEFAULT_BOOTSTRAPS) -> dict[str, Any]:
    """Bootstrap base and +O on identical case draws and report deltas."""

    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    base_scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    o_scores = np.asarray(o_scores, dtype=np.float64).reshape(-1)
    cases = np.asarray(cases).reshape(-1)
    if not (len(labels) == len(base_scores) == len(o_scores) == len(cases)):
        raise ValueError("paired bootstrap arrays have different lengths")
    unique = np.asarray(sorted(set(str(value) for value in cases)), dtype=object)
    by_case = {str(case): np.flatnonzero(cases.astype(str) == str(case)) for case in unique}
    rng = np.random.default_rng(int(seed))
    delta_auc: list[float] = []
    delta_logloss: list[float] = []
    for _ in range(int(repetitions)):
        draw = rng.integers(0, len(unique), size=len(unique))
        indexes = np.concatenate([by_case[str(unique[index])] for index in draw]) if len(draw) else np.empty(0, dtype=np.int64)
        if indexes.size == 0:
            continue
        base_auc = _auc(labels[indexes], base_scores[indexes])
        o_auc = _auc(labels[indexes], o_scores[indexes])
        if base_auc is not None and o_auc is not None:
            delta_auc.append(float(o_auc - base_auc))
        delta_logloss.append(float(_logloss(labels[indexes], o_scores[indexes]) - _logloss(labels[indexes], base_scores[indexes])))
    def interval(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            return {"n": 0, "q025": None, "q50": None, "q975": None}
        array = np.asarray(values, dtype=np.float64)
        return {"n": int(len(array)), "q025": float(np.quantile(array, 0.025)),
                "q50": float(np.quantile(array, 0.5)), "q975": float(np.quantile(array, 0.975))}
    return {"cases": int(len(unique)), "repetitions": int(repetitions), "seed": int(seed),
            "delta_auc_o_minus_base": interval(delta_auc),
            "delta_logloss_o_minus_base": interval(delta_logloss)}


def o_pair_stats(matches: np.ndarray, o_probability: np.ndarray) -> dict[str, Any]:
    matches = np.asarray(matches, dtype=np.float64)
    valid_rows = matches[:, 3] >= 0 if matches.size else np.zeros(0, dtype=bool)
    rows = matches[valid_rows].astype(np.int64)
    if rows.size == 0:
        return {"pairs": 0, "auc": None, "positive": _summary(np.empty(0)), "control": _summary(np.empty(0)),
                "positive_threshold_rate": None, "control_threshold_rate": None}
    o = np.asarray(o_probability, dtype=np.float32)
    positive = o[rows[:, 1], rows[:, 2]]
    control = o[rows[:, 3], rows[:, 4]]
    labels = np.concatenate((np.ones(len(positive), dtype=np.int8), np.zeros(len(control), dtype=np.int8)))
    scores = np.concatenate((positive, control))
    return {
        "pairs": int(len(rows)),
        "auc": _auc(labels, scores),
        "paired_mean_delta": float(np.mean(positive - control)),
        "positive": _summary(positive), "control": _summary(control),
        "positive_threshold_rate": float(np.mean(positive >= PROBABILITY_THRESHOLD)),
        "control_threshold_rate": float(np.mean(control >= PROBABILITY_THRESHOLD)),
    }


def _state_arrays_to_numpy(source: Any, states: Sequence[Any], o: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    def array(value: Any) -> np.ndarray:
        if isinstance(value, (tuple, list)):
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("model output must be a tensor")
        value = value.float().detach().cpu()
        if value.ndim == 4:
            value = value[0]
        if value.ndim != 3 or value.shape[0] != N_MEMBERS:
            raise ValueError(f"expected [14,H,W], got {tuple(value.shape)}")
        return value.numpy().astype(np.float32, copy=False)
    source_logits = array(source)
    final_logits = array(states[-1])
    o_logits = o.float().detach().cpu()
    if o_logits.ndim == 4:
        o_logits = o_logits[0]
    if o_logits.ndim == 3:
        o_logits = o_logits[0]
    if o_logits.ndim != 2:
        raise ValueError(f"expected O [H,W], got {tuple(o_logits.shape)}")
    return source_logits, final_logits, torch.sigmoid(o_logits).numpy().astype(np.float32, copy=False)


def _prediction(logits: np.ndarray) -> np.ndarray:
    if torch is None:
        values = np.asarray(logits, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-values)) >= PROBABILITY_THRESHOLD)
    values = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
    return (torch.sigmoid(values) >= PROBABILITY_THRESHOLD).numpy()


def _forward_case(model: Any, refiner: Any, generator: Any, image: Any, legacy: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward C0 and O from image only; this function has no GT argument."""

    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    image = image.float()
    zero_o = torch.zeros_like(image)
    # Models and inputs are explicitly FP32; no AMP/autocast is allowed in the
    # H0 verification path because the cache contract is FP32 sigmoid>=0.5.
    with torch.inference_mode():
        source = model(torch.cat((image, zero_o), dim=1))
        if isinstance(source, (tuple, list)):
            source = source[0]
        # C0 H0 caches were defined on the zero-O two-step final endpoint.
        states, _ = legacy.iterative_refine_o(image, source, refiner, 2, zero_o)
        o_logits = generator(image)
    return _state_arrays_to_numpy(source, states, o_logits)


def _load_components(lock: Mapping[str, Any], paths: Mapping[str, Path], device: Any) -> tuple[Any, Any, Any, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required for endpoint inference")
    import run_v31_ram_mature as legacy

    overlap_sha = sha256_file(paths["overlap"])
    args = SimpleNamespace(r332=paths["r332"], overlap=paths["overlap"], overlap_sha=overlap_sha)
    _, model, refiner, generator, _, _ = legacy.load_components(args, device)
    payload = torch.load(paths["fresh100"], map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "model" not in payload or "refiner" not in payload:
        raise ValueError("fresh100 checkpoint must contain model and refiner state")
    model.load_state_dict(payload["model"], strict=True)
    refiner.load_state_dict(payload["refiner"], strict=True)
    for module in (model, refiner, generator):
        module.float().to(device).eval().requires_grad_(False)
    return model, refiner, generator, legacy


def _case_o_stats(o: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    values = np.asarray(o, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    return {"soft": _summary(values),
            "threshold_count": int((values >= PROBABILITY_THRESHOLD).sum()),
            "threshold_rate": _safe_rate(int((values >= PROBABILITY_THRESHOLD).sum()), len(values))}


def _manifest_digest(root: Path, split: str, cases: Sequence[str]) -> tuple[str, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for case in cases:
        path = _cache_path(root, split, case)
        rows.append({"case": str(case), "path": str(path), "sha256": sha256_file(path)})
    digest_builder = hashlib.sha256()
    for row in sorted(rows, key=lambda value: value["case"]):
        digest_builder.update(row["case"].encode("utf-8"))
        digest_builder.update(b"\0")
        digest_builder.update(row["sha256"].encode("ascii"))
        digest_builder.update(b"\n")
    digest = digest_builder.hexdigest()
    return digest, rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(lock_path: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {out}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    hash_report = verify_lock(lock_path, lock)
    paths = {
        "dataset": _path_from_lock(lock, lock_path, "dataset_root", "dataset", "ram_dataset"),
        "fresh100": _path_from_lock(lock, lock_path, "fresh100_checkpoint", "c0_checkpoint", "checkpoint"),
        "r332": _path_from_lock(lock, lock_path, "r332", "r332_checkpoint"),
        "overlap": _path_from_lock(lock, lock_path, "overlap_checkpoint", "overlap", "image_only_overlap"),
        "h0": _path_from_lock(lock, lock_path, "oracle_h0_root", "h0_root", "oracle_predictions_h0", "oracle_root"),
    }
    if any(value is None for value in paths.values()):
        raise ValueError("lock must specify dataset, fresh100, r332, overlap, and packed H0 root")
    paths = {key: value for key, value in paths.items() if value is not None}
    if paths["h0"].name == "oracle_v3" or (paths["h0"] / "predictions" / "h0").is_dir():
        paths["h0"] = paths["h0"] / "predictions" / "h0"
    elif paths["h0"].name == "predictions" or (paths["h0"] / "h0").is_dir():
        paths["h0"] = paths["h0"] / "h0"
    if not paths["dataset"].is_dir() or not paths["h0"].is_dir():
        raise FileNotFoundError("dataset root or packed H0 root is missing")
    for label in ("fresh100", "r332", "overlap"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"missing {label}: {paths[label]}")
        expected = _expected_hash(lock, paths[label], (label, "c0" if label == "fresh100" else label))
        if expected is None:
            raise ValueError(f"lock has no SHA256 for required {label}: {paths[label]}")
        actual = sha256_file(paths[label])
        if actual != expected:
            raise ValueError(f"required {label} hash mismatch: {actual} != {expected}")

    import run_v31_ram_mature as legacy  # imported only after lock/hash gates
    torch.set_num_threads(64)
    seed = int(lock.get("seed", DEFAULT_SEED))
    legacy.seed_all(seed)
    start = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and lock.get("require_cuda", True):
        raise RuntimeError("locked audit requires CUDA; use synthetic tests for CPU")
    model, refiner, generator, legacy = _load_components(lock, paths, device)
    max_seconds = float(lock.get("max_seconds", DEFAULT_MAX_SECONDS))
    max_rows = int(lock.get("max_h0_rows_per_case", DEFAULT_MAX_H0_ROWS_PER_CASE))
    calipers = np.asarray(lock.get("matching_calipers", DEFAULT_CALIPERS.tolist()), dtype=np.float32)
    bootstrap_repetitions = int(lock.get("bootstrap_repetitions", DEFAULT_BOOTSTRAPS))
    disk_floor_bytes = int(lock.get("disk_floor_bytes", 4 * 1024 ** 3))
    if max_rows <= 0 or bootstrap_repetitions <= 0:
        raise ValueError("caps/bootstrap repetitions must be positive")
    out.mkdir(parents=True)

    def check_disk_floor() -> None:
        free = shutil.disk_usage(out).free
        if free < disk_floor_bytes:
            raise RuntimeError(f"disk floor reached: {free} < {disk_floor_bytes}")

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    readout_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        split: {"base": [], "o": [], "labels": [], "cases": []} for split in split_rows
    }
    case_csv_rows: list[dict[str, Any]] = []
    match_csv_rows: list[dict[str, Any]] = []
    cache_manifest: dict[str, Any] = {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        dataset = legacy.NativeWristDataset(paths["dataset"], split, augment=False, divisor=32)
        if len(dataset) != expected_count:
            raise ValueError(f"{split} dataset count {len(dataset)} != {expected_count}")
        cases = [path.stem for path in dataset.mask_files]
        digest, digest_rows = _manifest_digest(paths["h0"], split, cases)
        cache_manifest[split] = {"digest": digest, "files": digest_rows}
        expected_manifest = lock.get(f"{split}_h0_manifest_sha256", lock.get("h0_manifest_sha256"))
        if expected_manifest is None:
            manifest_map = lock.get("oracle_manifest_digests")
            if isinstance(manifest_map, Mapping):
                expected_manifest = manifest_map.get(split)
        if expected_manifest is not None and str(expected_manifest).lower() != digest:
            raise ValueError(f"{split} H0 manifest hash mismatch: {digest} != {expected_manifest}")
        for index in range(len(dataset)):
            if time.monotonic() - start > max_seconds:
                raise TimeoutError("locked diagnostic time budget exceeded")
            sample = dataset[index]
            case = str(sample["case"])
            image = sample["image"].unsqueeze(0).to(device, dtype=torch.float32)
            # This is the only forward boundary.  Target/valid are deliberately
            # obtained only after model and generator forwards have returned.
            source_logits, final_logits, o_probability = _forward_case(model, refiner, generator, image, legacy)
            source_probability = 1.0 / (1.0 + np.exp(-np.clip(source_logits.astype(np.float64), -80.0, 80.0))).astype(np.float32)
            final_probability = 1.0 / (1.0 + np.exp(-np.clip(final_logits.astype(np.float64), -80.0, 80.0))).astype(np.float32)
            # Ground truth begins here, strictly after all forwards.
            target = np.asarray(sample["mask"].numpy() > 0.5, dtype=bool)
            valid = np.asarray(sample["valid"].numpy()[0], dtype=bool)
            image_array = np.asarray(sample["image"].numpy()[0], dtype=np.float32)
            cache_path = _cache_path(paths["h0"], split, case)
            cache = load_h0_cache(cache_path, target.shape)
            if cache["label"] is not None and cache["label"] != ORACLE_LABEL:
                raise ValueError(f"unexpected H0 cache label for {split}/{case}: {cache['label']!r}")
            h0 = cache["h0"]
            source_below = source_logits < SOURCE_LOGIT_CUTOFF
            source_pred = _prediction(source_logits)
            final_pred = _prediction(final_logits)
            expected_h0 = valid[None] & target & (target.sum(axis=0, keepdims=True) >= 2) & ~final_pred & source_below
            h0_equal = bool(np.array_equal(h0, expected_h0))
            if not h0_equal:
                raise ValueError(f"H0 cache mismatch for {split}/{case}")
            if cache["source_below"] is not None and not np.array_equal(cache["source_below"], source_below):
                raise ValueError(f"cached source-below mask mismatch for {split}/{case}")
            bundle = feature_bundle(source_probability, final_probability, image_array)
            h0_unique = h0.any(axis=0) & valid
            matching_features = {name: bundle[name] for name in ("count_class", "top1_id", *MATCH_FEATURE_NAMES)}
            matches, match_meta = make_conditional_matches(target, valid, h0, matching_features,
                                                            calipers=calipers, max_rows=max_rows)
            pair_stats = o_pair_stats(matches, o_probability)
            # The incremental readout is fit on the strict, GT-single,
            # within-caliper subset only.  Fallback/reused controls remain in
            # the trace and pair audit but are never silently promoted to
            # equivalent matched evidence.
            strict_matches = matches[(matches[:, 5] == 1) & (matches[:, 6] == 1)] if len(matches) else np.empty((0, 8))
            strict_pair_stats = o_pair_stats(strict_matches, o_probability)
            base, labels, o_features = build_readout_rows(bundle, o_probability, strict_matches)
            if len(labels):
                readout_arrays[split]["base"].append(base)
                readout_arrays[split]["o"].append(o_features)
                readout_arrays[split]["labels"].append(labels)
                readout_arrays[split]["cases"].append(np.asarray([case] * len(labels), dtype="U64"))
            counts = {
                "case": case,
                "split": split,
                "h0_member_pixels": int(h0.sum()),
                "h0_unique_pixels": int(h0_unique.sum()),
                "gt_overlap_unique_pixels": int(((target.sum(axis=0) >= 2) & valid).sum()),
                "c0_source_member_fn": int((valid[None] & target & ~source_pred).sum()),
                "c0_final_member_fn": int((valid[None] & target & ~final_pred).sum()),
                "c0_final_member_fp": int((valid[None] & ~target & final_pred).sum()),
                "c0_final_overlap_fn_unique": int((((target.sum(axis=0) >= 2) & valid) & ~(final_pred.sum(axis=0) >= 2)).sum()),
                "o_h0_unique_n": int(h0_unique.sum()),
                "o_control_n": int(pair_stats["pairs"]),
                "o_h0_unique": _case_o_stats(o_probability, h0_unique),
                "o_gt_single_foreground": _case_o_stats(o_probability, valid & (target.sum(axis=0) == 1)),
                "match": match_meta,
                "paired_o": pair_stats,
                "paired_o_strict_singleton": strict_pair_stats,
                "readout_pairs_strict_singleton": int(strict_pair_stats["pairs"]),
                "h0_cache_sha256": cache["sha256"],
                "h0_cache_label": cache["label"],
                "h0_verified": h0_equal,
                "source_cutoff": SOURCE_LOGIT_CUTOFF,
                "fp32_threshold": PROBABILITY_THRESHOLD,
            }
            split_rows[split].append(counts)
            case_csv_rows.append({"split": split, "case": case, "h0_member_pixels": counts["h0_member_pixels"],
                                  "h0_unique_pixels": counts["h0_unique_pixels"], "matched": match_meta["matched"],
                                  "singleton_matched": match_meta["singleton_matched"], "unmatched": match_meta["unmatched"],
                                  "paired_o_auc": pair_stats["auc"], "paired_o_delta": pair_stats.get("paired_mean_delta"),
                                  "h0_o_mean": pair_stats["positive"].get("mean"), "control_o_mean": pair_stats["control"].get("mean")})
            valid_match = matches[matches[:, 3] >= 0] if len(matches) else np.empty((0, 8))
            for row in valid_match:
                match_csv_rows.append({"split": split, "case": case, "member": int(row[0]), "h0_y": int(row[1]),
                                       "h0_x": int(row[2]), "control_y": int(row[3]), "control_x": int(row[4]),
                                       "status": int(row[5]), "singleton": int(row[6]), "normalized_distance": float(row[7]),
                                       "o_h0": float(o_probability[int(row[1]), int(row[2])]),
                                       "o_control": float(o_probability[int(row[3]), int(row[4])])})
            if index % 25 == 0:
                print(json.dumps({"split": split, "case_index": index + 1, "cases": len(dataset),
                                  "elapsed": time.monotonic() - start}), flush=True)

    outputs: dict[str, Any] = {}
    for split in split_rows:
        for key in readout_arrays[split]:
            if readout_arrays[split][key]:
                outputs[f"{split}_{key}"] = np.concatenate(readout_arrays[split][key])
            else:
                outputs[f"{split}_{key}"] = np.empty(0, dtype=np.float32 if key != "cases" else "U1")
    for split in split_rows:
        check_disk_floor()
        np.savez_compressed(out / f"TRACES_{split}.npz", **{
            "base": outputs[f"{split}_base"], "o": outputs[f"{split}_o"],
            "labels": outputs[f"{split}_labels"], "cases": outputs[f"{split}_cases"],
        })
        check_disk_floor()
        with (out / f"INFERENCE_TRACE_{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows[split]:
                handle.write(json.dumps(row, allow_nan=False, default=json_default) + "\n")

    train_base, train_o, train_y, train_cases = outputs["train_base"], outputs["train_o"], outputs["train_labels"], outputs["train_cases"]
    val_base, val_o, val_y, val_cases = outputs["val_base"], outputs["val_o"], outputs["val_labels"], outputs["val_cases"]
    strict_train_cases = {str(row["case"]) for row in split_rows["train"] if row["readout_pairs_strict_singleton"] > 0}
    strict_val_cases = {str(row["case"]) for row in split_rows["val"] if row["readout_pairs_strict_singleton"] > 0}
    strict_train_pair_count = int(sum(row["readout_pairs_strict_singleton"] for row in split_rows["train"]))
    strict_val_pair_count = int(sum(row["readout_pairs_strict_singleton"] for row in split_rows["val"]))
    can_fit = (len(train_y) > 0 and len(val_y) > 0 and np.unique(train_y).size == 2
               and np.unique(val_y).size == 2)
    logistic = {
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "fit_split": "train425 only",
        "evaluation_split": "held-out val69 only",
        "base_features": "C0 source/final selected-member probabilities, source/final top1/top2, final count class, final top1 ID, member ID, normalized image intensity/gradient",
        "incremental_features": "base plus image-only O soft probability and O>=0.5 indicator",
        "train_rows": int(len(train_y)), "val_rows": int(len(val_y)),
        "claim_policy": "diagnostic association only; no segmentation performance or fusion-point claim",
    }
    if can_fit:
        base_model = FixedLogisticReadout(l2=float(lock.get("logistic_l2", 1.0))).fit(train_base, train_y)
        o_model = FixedLogisticReadout(l2=float(lock.get("logistic_l2", 1.0))).fit(np.column_stack((train_base, train_o)), train_y)
        base_scores = base_model.predict_proba(val_base)
        o_scores = o_model.predict_proba(np.column_stack((val_base, val_o)))
        logistic.update({
            "base": {"auc": _auc(val_y, base_scores), "logloss": _logloss(val_y, base_scores), "fit": base_model.summary()},
            "base_plus_o": {"auc": _auc(val_y, o_scores), "logloss": _logloss(val_y, o_scores), "fit": o_model.summary()},
            "val_case_bootstrap_base": case_bootstrap(val_y, base_scores, val_cases, seed=seed, repetitions=bootstrap_repetitions),
            "val_case_bootstrap_base_plus_o": case_bootstrap(val_y, o_scores, val_cases, seed=seed, repetitions=bootstrap_repetitions),
            "val_case_bootstrap_paired_delta": paired_case_bootstrap(
                val_y, base_scores, o_scores, val_cases, seed=seed, repetitions=bootstrap_repetitions
            ),
        })
    else:
        logistic.update({
            "base": None, "base_plus_o": None,
            "val_case_bootstrap_base": None, "val_case_bootstrap_base_plus_o": None,
            "val_case_bootstrap_paired_delta": None,
        })
    logistic["strict_match_coverage"] = {
        "train_cases_with_strict_pairs": len(strict_train_cases),
        "val_cases_with_strict_pairs": len(strict_val_cases),
        "train_strict_pairs": strict_train_pair_count,
        "val_strict_pairs": strict_val_pair_count,
        "minimum_cases_for_interpretation": 10,
    }
    logistic["interpretation_status"] = (
        "INCONCLUSIVE_STRICT_MATCH_COVERAGE"
        if (not can_fit or len(strict_train_cases) < 10 or len(strict_val_cases) < 10
                or logistic["strict_match_coverage"]["val_strict_pairs"] < 100)
        else "DIAGNOSTIC_READOUT_AVAILABLE"
    )
    aggregate_pairs = {}
    for split in split_rows:
        pair_rows = [row for row in match_csv_rows if row["split"] == split]
        labels_all = np.asarray([1] * len(pair_rows) + [0] * len(pair_rows), dtype=np.int8)
        scores_all = np.asarray([row["o_h0"] for row in pair_rows] + [row["o_control"] for row in pair_rows], dtype=np.float64)
        strict_rows = [row for row in pair_rows if int(row["status"]) == 1 and int(row["singleton"]) == 1]
        labels_strict = np.asarray([1] * len(strict_rows) + [0] * len(strict_rows), dtype=np.int8)
        scores_strict = np.asarray([row["o_h0"] for row in strict_rows] + [row["o_control"] for row in strict_rows], dtype=np.float64)
        aggregate_pairs[split] = {"cases": len(split_rows[split]),
                                  "matched_pairs": int(len(pair_rows)),
                                  "strict_singleton_pairs": int(len(strict_rows)),
                                  "pooled_auc": _auc(labels_all, scores_all),
                                  "strict_singleton_pooled_auc": _auc(labels_strict, scores_strict),
                                  "pooled_h0_o": _summary(np.asarray([row["o_h0"] for row in pair_rows], dtype=np.float32)),
                                  "pooled_control_o": _summary(np.asarray([row["o_control"] for row in pair_rows], dtype=np.float32)),
                                  "strict_singleton_h0_o": _summary(np.asarray([row["o_h0"] for row in strict_rows], dtype=np.float32)),
                                  "strict_singleton_control_o": _summary(np.asarray([row["o_control"] for row in strict_rows], dtype=np.float32)),
                                  "case_auc_mean": float(np.mean([row["paired_o"]["auc"] for row in split_rows[split] if row["paired_o"]["auc"] is not None])) if any(row["paired_o"]["auc"] is not None for row in split_rows[split]) else None}
    result = {
        "status": "COMPLETE", "diagnostic_label": DIAGNOSTIC_LABEL,
        "oracle_label": ORACLE_LABEL, "optimizer_updates": 0, "test_used": False,
        "segmentation_performance_claim": False, "fusion_point_selected": False,
        "device": str(device), "cpu_threads": torch.get_num_threads(),
        "paths": {key: str(value) for key, value in paths.items()},
        "hash_audit": hash_report, "h0_cache_manifests": cache_manifest,
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "per_split_pair_summary": aggregate_pairs,
        "logistic_readout": logistic,
        "matching": {"calipers": calipers.tolist(), "max_rows_per_case": max_rows,
                      "feature_names": list(MATCH_FEATURE_NAMES),
                      "discrete_keys": ["c0_final_predicted_member_count_class", "c0_final_top1_member_id"],
                      "o_excluded_from_matching": True},
        "seconds": time.monotonic() - start,
        "outputs": {"per_case_json": "PER_CASE.json", "per_case_csv": "PER_CASE.csv", "matches_csv": "MATCHES.csv",
                    "trace_npz": ["TRACES_train.npz", "TRACES_val.npz"], "trace_npz_per_case": False},
    }
    check_disk_floor()
    write_json(out / "HASH_AUDIT.json", hash_report)
    check_disk_floor()
    write_json(out / "PER_CASE.json", {"train": split_rows["train"], "val": split_rows["val"]})
    check_disk_floor()
    _write_csv(out / "PER_CASE.csv", case_csv_rows,
               ("split", "case", "h0_member_pixels", "h0_unique_pixels", "matched", "singleton_matched", "unmatched",
                "paired_o_auc", "paired_o_delta", "h0_o_mean", "control_o_mean"))
    check_disk_floor()
    _write_csv(out / "MATCHES.csv", match_csv_rows,
               ("split", "case", "member", "h0_y", "h0_x", "control_y", "control_x", "status", "singleton",
                "normalized_distance", "o_h0", "o_control"))
    check_disk_floor()
    write_json(out / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.lock.resolve(), args.out.resolve())
    print(json.dumps({"status": result["status"], "out": str(args.out), "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
