#!/usr/bin/env python3
"""Read-only endpoint audit for the locked RAM C00/CS fit256 heads.

The audit replays the already trained C00 and CS heads on the fixed, patient
grouped ``probe_fit`` manifest.  It loads the exact C0 checkpoint, captures
only the native/padded source -> step1 -> step2 endpoints, and applies the
saved head checkpoints without constructing an optimizer or taking a step.
The fixed H0 cache is used only for reporting; it is never passed to the
model.  No intermediate training trajectory is inferred when only endpoint
checkpoints exist.

Typical remote invocation (the lock supplies the remote paths)::

    python scripts/audit_ram_targeted_fit_h0.py \
        --run-lock /root/ram_so_c00_cs_20260929/RUN_LOCK.json \
        --out /root/ram_targeted_fit_h0_20260929

All outputs are written below ``--out``.  The original C0 checkpoint, head
checkpoints, dataset, manifests, and H0 cache are opened read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_r329_ram_official_metrics_val import _official_one  # noqa: E402
from ram_highres_capture import C0_SHA256, load_frozen_c0, sha256_file  # noqa: E402
from ram_so_conditional_corrector import (  # noqa: E402
    PriorMaps,
    RAMSOConditionalCorrector,
)
from r332_v31_port import iterative_refine_o  # noqa: E402
from train_r284_ramw600_overlap_prior import top_training_pairs  # noqa: E402
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset  # noqa: E402


N_MEMBERS = 14
FIT_GROUP = "probe_fit"
FIT_CASES = 256
TRAIN_CASES = 425
LOGIT_THRESHOLD = 0.0
H0_SOURCE_THRESHOLD = -5.00001
BOUND_DEFAULT = 8.0
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
ARMS = ("C00", "CS")
STATE_NAMES = ("source", "step1", "step2", "C0_full14_final", "C00_full14_final", "CS_full14_final")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.ascontiguousarray(value.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _runtime_snapshot(modules: Sequence[torch.nn.Module]) -> dict[str, Any]:
    modes: list[bool] = []
    buffers: list[tuple[str, bytes]] = []
    for owner_index, owner in enumerate(modules):
        modes.extend(bool(module.training) for module in owner.modules())
        for name, value in owner.named_buffers():
            buffers.append((f"{owner_index}:{name}", np.ascontiguousarray(value.detach().cpu().numpy()).tobytes()))
    return {
        "modes": tuple(modes),
        "buffers": tuple(buffers),
        "state_digests": tuple(_state_digest(module) for module in modules),
    }


def _runtime_unchanged(before: Mapping[str, Any], modules: Sequence[torch.nn.Module]) -> bool:
    return before == _runtime_snapshot(modules)


@contextmanager
def _preserve_runtime(modules: Sequence[torch.nn.Module]):
    """Keep mode, buffers, and RNG unchanged across the read-only audit."""

    mode = {module: module.training for owner in modules for module in owner.modules()}
    buffers = {buffer: buffer.detach().clone() for owner in modules for buffer in owner.buffers()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for owner in modules:
            owner.eval()
        yield
    finally:
        for module, value in mode.items():
            module.train(value)
        for buffer, value in buffers.items():
            buffer.copy_(value)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _stats(values: np.ndarray | Iterable[float] | None) -> dict[str, Any]:
    if values is None:
        values_array = np.empty(0, dtype=np.float64)
    else:
        values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    if values_array.size == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "quantiles": {f"q{int(q * 100):02d}": None for q in QUANTILES},
        }
    if not np.isfinite(values_array).all():
        raise FloatingPointError("non-finite audit values")
    return {
        "n": int(values_array.size),
        "min": float(values_array.min()),
        "max": float(values_array.max()),
        "mean": float(values_array.mean()),
        "std": float(values_array.std()),
        "quantiles": {
            f"q{int(q * 100):02d}": float(np.quantile(values_array, q)) for q in QUANTILES
        },
    }


class _RunningStats:
    """Bounded-memory aggregate with exact moments and deterministic samples."""

    def __init__(self, sample_limit: int = 250_000, seed: int = 1729) -> None:
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.sample_limit = int(sample_limit)
        self.sample: list[float] = []
        self.rng = np.random.default_rng(seed)

    def add(self, values: np.ndarray | Iterable[float]) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0:
            return
        if not np.isfinite(array).all():
            raise FloatingPointError("non-finite aggregate values")
        self.n += int(array.size)
        self.sum += float(array.sum(dtype=np.float64))
        self.sumsq += float(np.square(array, dtype=np.float64).sum(dtype=np.float64))
        self.minimum = min(self.minimum, float(array.min()))
        self.maximum = max(self.maximum, float(array.max()))
        if self.sample_limit <= 0:
            return
        if len(self.sample) < self.sample_limit:
            take = min(self.sample_limit - len(self.sample), int(array.size))
            self.sample.extend(float(value) for value in array[:take])
            array = array[take:]
        if array.size == 0:
            return
        # Reservoir sampling keeps the aggregate quantiles representative
        # without retaining all 14*H*W category pixels for 256 cases.
        start = self.n - int(array.size)
        for offset, value in enumerate(array):
            seen = start + offset
            slot = int(self.rng.integers(0, seen + 1))
            if slot < self.sample_limit:
                self.sample[slot] = float(value)

    def summary(self) -> dict[str, Any]:
        if self.n == 0:
            return _stats(None)
        mean = self.sum / self.n
        variance = max(self.sumsq / self.n - mean * mean, 0.0)
        sample = np.asarray(self.sample, dtype=np.float64)
        return {
            "n": int(self.n),
            "min": float(self.minimum),
            "max": float(self.maximum),
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "quantiles": {
                f"q{int(q * 100):02d}": (float(np.quantile(sample, q)) if sample.size else None)
                for q in QUANTILES
            },
            "quantiles_sample_n": int(sample.size),
        }


def _residual_stats(values: np.ndarray, bound: float) -> dict[str, Any]:
    result = _stats(values)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        result.update({"positive_fraction": None, "negative_fraction": None, "zero_fraction": None,
                       "saturated_count": 0, "saturated_fraction": None, "bound": float(bound)})
        return result
    saturated = np.abs(array) >= float(bound) * (1.0 - 1e-4)
    result.update({
        "positive_fraction": float(np.mean(array > 0.0)),
        "negative_fraction": float(np.mean(array < 0.0)),
        "zero_fraction": float(np.mean(array == 0.0)),
        "saturated_count": int(saturated.sum()),
        "saturated_fraction": float(saturated.mean()),
        "bound": float(bound),
    })
    return result


def _residual_accumulator(bound: float, sample_limit: int = 250_000) -> dict[str, Any]:
    return {
        "values": _RunningStats(sample_limit),
        "positive": 0,
        "negative": 0,
        "zero": 0,
        "saturated": 0,
        "bound": float(bound),
    }


def _add_residual(accumulator: dict[str, Any], values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    accumulator["values"].add(array)
    if array.size:
        accumulator["positive"] += int((array > 0).sum())
        accumulator["negative"] += int((array < 0).sum())
        accumulator["zero"] += int((array == 0).sum())
        accumulator["saturated"] += int((np.abs(array) >= accumulator["bound"] * (1.0 - 1e-4)).sum())


def _finish_residual(accumulator: dict[str, Any]) -> dict[str, Any]:
    result = accumulator["values"].summary()
    n = int(result["n"])
    result.update({
        "positive_fraction": None if n == 0 else float(accumulator["positive"] / n),
        "negative_fraction": None if n == 0 else float(accumulator["negative"] / n),
        "zero_fraction": None if n == 0 else float(accumulator["zero"] / n),
        "saturated_count": int(accumulator["saturated"]),
        "saturated_fraction": None if n == 0 else float(accumulator["saturated"] / n),
        "bound": float(accumulator["bound"]),
    })
    return result


def _path_from(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value).expanduser()


def _first_existing(paths: Iterable[Path | None]) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def _resolve_h0_root(root: Path) -> Path:
    root = root.expanduser()
    if (root / "train").is_dir():
        return root
    if (root / "predictions" / "h0" / "train").is_dir():
        return root / "predictions" / "h0"
    if (root / "h0" / "train").is_dir():
        return root / "h0"
    raise FileNotFoundError(f"H0 cache root has no train/ directory: {root}")


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = SCRIPT_DIR.parent
    local_run_root = repo_root / "outputs" / "ram_so_c00_cs_20260929"
    local_lock = local_run_root / "RUN_LOCK.json"
    lock_path = _path_from(args.run_lock) or local_lock
    lock: dict[str, Any] = {}
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not isinstance(lock, dict):
            raise ValueError("RUN_LOCK must be a JSON object")

    def choose(cli: str | None, keys: Sequence[str], fallbacks: Sequence[Path]) -> Path:
        if cli:
            return Path(cli).expanduser()
        candidates: list[Path | None] = [path for path in fallbacks if path is not None]
        candidates.extend(_path_from(lock.get(key)) for key in keys)
        result = _first_existing(candidates)
        if result is None:
            values = ", ".join(str(path) for path in candidates if path is not None)
            raise FileNotFoundError(f"could not resolve required path ({keys}): {values}")
        return result

    dataset_fallback = repo_root.parent / "RAM-W600"
    c0_fallback = repo_root / "outputs" / "whitebox_adam_20260922" / "fresh_adam" / "diagnostic_final.pth"
    split_fallback = repo_root / "outputs" / "ram_highres_readout_20260928" / "PROBE_SPLIT_MANIFEST_v2_PATIENT.json"
    h0_fallback = repo_root / "outputs" / "ram_source_correctability_20260924" / "oracle_v3" / "predictions" / "h0"
    prior_fallback = repo_root / "outputs" / "priors" / "r323_ram_native_r317_single_seed"
    run_fallback = local_run_root / "results"

    dataset = choose(args.dataset_root, ("dataset_actual_local", "dataset"), (dataset_fallback,))
    c0 = choose(args.c0_checkpoint, ("c0",), (c0_fallback,))
    split_manifest = choose(args.split_manifest, ("split_manifest",), (split_fallback,))
    h0 = choose(args.h0_root, ("h0_root", "oracle_h0_root"), (h0_fallback,))
    prior = choose(args.s_prior_root, ("s_prior_root",), (prior_fallback,))
    run_root = choose(args.run_root, ("run_root",), (run_fallback,))
    out = Path(args.out).expanduser().resolve()
    return {
        "lock_path": lock_path,
        "lock": lock,
        "dataset": dataset,
        "c0": c0,
        "split_manifest": split_manifest,
        "h0_root": _resolve_h0_root(h0),
        "s_prior_root": prior,
        "run_root": run_root,
        "out": out,
    }


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must be a JSON object")
    if manifest.get("test_files_accessed") is True:
        raise ValueError("split manifest reports test access")
    groups = manifest.get("groups")
    if not isinstance(groups, dict) or FIT_GROUP not in groups:
        raise ValueError(f"split manifest lacks {FIT_GROUP}")
    cases = [str(case) for case in groups[FIT_GROUP]]
    if len(cases) != FIT_CASES or len(set(cases)) != FIT_CASES:
        raise ValueError(f"{FIT_GROUP} must contain exactly {FIT_CASES} unique cases")
    counts = manifest.get("counts", {})
    if counts.get(FIT_GROUP, FIT_CASES) != FIT_CASES:
        raise ValueError("manifest count disagrees with probe_fit")
    case_patient = manifest.get("case_patient_token")
    if not isinstance(case_patient, dict) or any(case not in case_patient for case in cases):
        raise ValueError("patient-grouped manifest lacks a patient token for every fit case")
    fit_patients = {str(case_patient[case]) for case in cases}
    if not fit_patients:
        raise ValueError("empty fit patient group")
    return manifest, cases


def _load_h0_cache(path: Path, expected_shape: Sequence[int], original_hw: Sequence[int], target: np.ndarray,
                   valid: np.ndarray) -> dict[str, Any]:
    expected = tuple(int(value) for value in expected_shape)
    original = tuple(int(value) for value in original_hw)
    with np.load(path, allow_pickle=False) as archive:
        if "h0" in archive:
            h0_key = "h0"
        elif "H0" in archive:
            h0_key = "H0"
        else:
            raise ValueError(f"H0 cache has no h0/H0 payload: {path}")
        if "shape" in archive:
            shape = tuple(int(value) for value in np.asarray(archive["shape"]).reshape(-1).tolist())
        else:
            shape = tuple(int(value) for value in np.asarray(archive[h0_key]).shape)
        if len(shape) != 3 or shape[0] != N_MEMBERS:
            raise ValueError(f"H0 cache shape must be [14,H,W], got {shape}: {path}")
        payload = np.asarray(archive[h0_key])
        if payload.dtype == np.bool_ and payload.size == int(np.prod(shape)):
            h0 = payload.reshape(shape).copy()
        else:
            bits = np.unpackbits(payload.astype(np.uint8).reshape(-1), bitorder="little")
            count = int(np.prod(shape))
            if bits.size < count:
                raise ValueError(f"packed H0 payload too short: {path}")
            h0 = bits[:count].reshape(shape).astype(bool, copy=False)
        cached_hw = tuple(int(value) for value in np.asarray(archive["original_hw"]).reshape(-1).tolist()) if "original_hw" in archive else None
        label = None
        if "label" in archive:
            labels = np.asarray(archive["label"]).reshape(-1)
            label = str(labels[0]) if labels.size else None
        optional: dict[str, np.ndarray] = {}
        for key in ("source", "step1", "step2", "c0_final", "final", "source_below"):
            if key not in archive:
                continue
            item = np.asarray(archive[key])
            if item.dtype == np.bool_ and item.size == int(np.prod(shape)):
                optional[key] = item.reshape(shape).copy()
            else:
                bits = np.unpackbits(item.astype(np.uint8).reshape(-1), bitorder="little")
                optional[key] = bits[: int(np.prod(shape))].reshape(shape).astype(bool, copy=False)
    if cached_hw is not None and cached_hw != original:
        raise ValueError(f"H0 original_hw mismatch for {path}: {cached_hw} != {original}")
    if h0.shape == expected:
        normalized = h0
    elif h0.shape[0] == expected[0] and h0.shape[1:] == original:
        normalized = np.zeros(expected, dtype=bool)
        normalized[:, : original[0], : original[1]] = h0
    elif h0.shape[0] == expected[0] and h0.shape[1] <= expected[1] and h0.shape[2] <= expected[2] and cached_hw == original:
        normalized = np.zeros(expected, dtype=bool)
        normalized[:, : h0.shape[1], : h0.shape[2]] = h0
    else:
        raise ValueError(f"H0 shape mismatch for {path}: {h0.shape} != {expected}")
    if bool((normalized & ~(target & valid[None])).any()):
        raise ValueError(f"H0 is not a valid fixed GT member subset: {path}")
    if bool(normalized[:, original[0] :, :].any()) or bool(normalized[:, :, original[1] :].any()):
        raise ValueError(f"H0 has true bits in padded rows/columns: {path}")
    normalized_optional: dict[str, np.ndarray] = {}
    for key, item in optional.items():
        if item.shape == expected:
            normalized_optional[key] = item
        elif item.shape[0] == expected[0] and item.shape[1:] == original:
            padded = np.zeros(expected, dtype=bool)
            padded[:, : original[0], : original[1]] = item
            normalized_optional[key] = padded
    return {"h0": normalized, "label": label, "optional": normalized_optional, "sha256": _sha256(path), "path": str(path)}


def _load_s_map(root: Path, case: str, native_hw: tuple[int, int], padded_hw: tuple[int, int]) -> torch.Tensor:
    npy = root / "train" / f"{case}.npy"
    png = root / "train" / f"{case}.png"
    path = npy if npy.is_file() else png
    if not path.is_file():
        raise FileNotFoundError(f"missing S map for {case}: {npy} or {png}")
    if path.suffix.lower() == ".npy":
        value = np.asarray(np.load(path), dtype=np.float32)
    else:
        from PIL import Image

        value = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    value = np.squeeze(value)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"invalid S map {path}: {value.shape}")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"S map outside [0,1]: {path}")
    tensor = torch.from_numpy(value).float()[None, None]
    tensor = F.interpolate(tensor, native_hw, mode="bilinear", align_corners=False)
    pad_h = padded_hw[0] - native_hw[0]
    pad_w = padded_hw[1] - native_hw[1]
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"native S map extent exceeds padded canvas for {case}")
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=0.0)


def _load_head(path: Path, arm: str, bound: float, device: torch.device) -> tuple[RAMSOConditionalCorrector, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("head"), Mapping):
        raise ValueError(f"head checkpoint has no head mapping: {path}")
    if str(payload.get("arm")) != arm:
        raise ValueError(f"head arm mismatch for {path}: {payload.get('arm')} != {arm}")
    if str(payload.get("C0_sha256", "")).lower() != C0_SHA256:
        raise ValueError(f"head checkpoint C0 SHA mismatch: {path}")
    steps = int(payload.get("steps", 0))
    if steps <= 0:
        raise ValueError(f"head checkpoint has no completed updates: {path}")
    head = RAMSOConditionalCorrector(bound=float(bound)).to(device).float()
    head.load_state_dict(payload["head"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, {
        "path": str(path),
        "sha256": _sha256(path),
        "arm": arm,
        "steps": steps,
        "C0_sha256": payload.get("C0_sha256"),
        "optimizer_loaded": False,
        "parameters": int(sum(parameter.numel() for parameter in head.parameters())),
    }


def _capture_endpoints(model: torch.nn.Module, refiner: torch.nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden: list[torch.Tensor] = []
    calls = 0

    def collect(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal calls
        if calls >= 5:
            hidden.append(output.detach())
        calls += 1

    handle = refiner.dec1.register_forward_hook(collect)
    try:
        with torch.inference_mode():
            zero_o = torch.zeros_like(image)
            source = model(torch.cat((image, zero_o), dim=1))
            if isinstance(source, (tuple, list)):
                source = source[0]
            states, _ = iterative_refine_o(image, source, refiner, 2, zero_o, chunk=3)
    finally:
        handle.remove()
    if not isinstance(source, torch.Tensor) or len(states) != 3 or calls != 10 or len(hidden) != 5:
        raise RuntimeError(f"unexpected C0 endpoint capture: calls={calls}, hidden={len(hidden)}, states={len(states)}")
    hidden_tensor = torch.cat(hidden, dim=0)
    expected = (1, N_MEMBERS, image.shape[-2], image.shape[-1])
    if tuple(source.shape) != expected or any(tuple(state.shape) != expected for state in states):
        raise RuntimeError(f"unexpected C0 endpoint shapes: source={tuple(source.shape)}, states={[tuple(s.shape) for s in states]}")
    if tuple(hidden_tensor.shape) != (N_MEMBERS, 12, image.shape[-2], image.shape[-1]):
        raise RuntimeError(f"unexpected step2 hidden shape: {tuple(hidden_tensor.shape)}")
    return source, states[1], states[2], hidden_tensor


def _crop(array: np.ndarray, native_hw: tuple[int, int]) -> np.ndarray:
    return array[..., : native_hw[0], : native_hw[1]]


def _endpoint_summary(logits: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    valid_members = np.broadcast_to(valid[None], logits.shape)
    invalid_members = ~valid_members
    return {
        "padded_canvas": _stats(logits),
        "native_valid": _stats(logits[valid_members]),
        "padded_invalid": _stats(logits[invalid_members]),
        "padded_shape": list(logits.shape),
        "native_shape": [int(logits.shape[0]), int(valid.shape[0]), int(valid.shape[1])],
    }


def _category_masks(target: np.ndarray, valid: np.ndarray, h0: np.ndarray) -> dict[str, np.ndarray]:
    h0_unique = h0.any(axis=0)
    valid_members = np.broadcast_to(valid[None], target.shape)
    return {
        "H0_member": h0,
        "matched_TN_control": valid_members & ~target & h0_unique[None],
        "correct_foreground": valid_members & target & ~h0,
        "ordinary_background": valid_members & ~target & ~h0_unique[None],
    }


def _state_counts(logits: np.ndarray, target: np.ndarray, valid: np.ndarray, h0: np.ndarray) -> dict[str, Any]:
    prediction = logits >= LOGIT_THRESHOLD
    valid_members = np.broadcast_to(valid[None], target.shape)
    h0_unique = h0.any(axis=0)
    gt_overlap = (target.sum(axis=0) >= 2) & valid
    predicted_overlap = (prediction.sum(axis=0) >= 2) & valid
    other_overlap = gt_overlap & ~h0_unique
    other_overlap_member = target & gt_overlap[None] & ~h0
    matched_tn = valid_members & ~target & h0_unique[None]
    return {
        "member_tp": int((prediction & target & valid_members).sum()),
        "member_fp": int((prediction & ~target & valid_members).sum()),
        "member_fn": int((~prediction & target & valid_members).sum()),
        "h0_member_total": int(h0.sum()),
        "h0_member_recovered": int((prediction & h0).sum()),
        "h0_unique_total": int(h0_unique.sum()),
        "h0_unique_recovered": int((predicted_overlap & h0_unique).sum()),
        "other_overlap_unique_total": int(other_overlap.sum()),
        "other_overlap_unique_fn": int((~predicted_overlap & other_overlap).sum()),
        "other_overlap_fn_member": int((~prediction & other_overlap_member & valid_members).sum()),
        "overlap_unique_total": int(gt_overlap.sum()),
        "overlap_unique_fn": int((~predicted_overlap & gt_overlap).sum()),
        "matched_tn_control_total": int(matched_tn.sum()),
        "matched_tn_fp": int((prediction & matched_tn).sum()),
        "existing_fp": int((prediction & ~target & valid_members).sum()),
        "correct_foreground_tp": int((prediction & target & ~h0 & valid_members).sum()),
        "ordinary_background_fp": int((prediction & ~target & ~h0_unique[None] & valid_members).sum()),
    }


def _correction_counts(base: np.ndarray, final: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    base_prediction = base >= LOGIT_THRESHOLD
    final_prediction = final >= LOGIT_THRESHOLD
    valid_members = np.broadcast_to(valid[None], target.shape)
    tp = target & valid_members
    fp = ~target & valid_members
    return {
        "new_tp": int((~base_prediction & final_prediction & tp).sum()),
        "new_fp": int((~base_prediction & final_prediction & fp).sum()),
        "lost_tp": int((base_prediction & ~final_prediction & tp).sum()),
        "removed_fp": int((base_prediction & ~final_prediction & fp).sum()),
        "new_tp_per_member": [int((~base_prediction[m] & final_prediction[m] & tp[m]).sum()) for m in range(N_MEMBERS)],
        "new_fp_per_member": [int((~base_prediction[m] & final_prediction[m] & fp[m]).sum()) for m in range(N_MEMBERS)],
        "lost_tp_per_member": [int((base_prediction[m] & ~final_prediction[m] & tp[m]).sum()) for m in range(N_MEMBERS)],
        "removed_fp_per_member": [int((base_prediction[m] & ~final_prediction[m] & fp[m]).sum()) for m in range(N_MEMBERS)],
    }


def _metric_rows(logits: np.ndarray, target: np.ndarray, native_hw: tuple[int, int], pairs: Sequence[tuple[int, int]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    pred = _crop(logits >= LOGIT_THRESHOLD, native_hw)
    truth = _crop(target, native_hw)
    members = [_official_one(pred[m], truth[m]) for m in range(N_MEMBERS)]
    overlap = _official_one(pred.sum(axis=0) >= 2, truth.sum(axis=0) >= 2)
    pair_rows = [_official_one(pred[i] & pred[j], truth[i] & truth[j]) for i, j in pairs if bool((truth[i] & truth[j]).any())]

    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        keys = ("dsc", "nsd_2px", "voe", "msd_px", "ravd")
        result = {"n": len(rows)}
        for key in keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
            result[key] = None if not values else float(np.mean(values))
        result["msd_fail_rate"] = None if not rows else float(np.mean([float(row["msd_failed"]) for row in rows]))
        return result

    return members, {"overall_instance_metrics": aggregate(members), "overlap_region_metrics": aggregate([overlap]),
                     "overlap_pair_intersection_metrics": aggregate(pair_rows), "overlap_pair_count": len(pair_rows)}, pair_rows


def _stats_row(prefix: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {f"{prefix}_n": stats.get("n")}
    for key in ("min", "max", "mean", "std"):
        row[f"{prefix}_{key}"] = stats.get(key)
    for q in QUANTILES:
        row[f"{prefix}_q{int(q * 100):02d}"] = stats.get("quantiles", {}).get(f"q{int(q * 100):02d}")
    return row


def _analyze_case(case: str, patient: str, arrays: Mapping[str, np.ndarray], target: np.ndarray, valid: np.ndarray,
                  h0: np.ndarray, native_hw: tuple[int, int], bound: float, pairs: Sequence[tuple[int, int]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    categories = _category_masks(target, valid, h0)
    state_records: dict[str, Any] = {}
    for state in STATE_NAMES:
        logits = arrays[state]
        state_records[state] = {
            "endpoint": _endpoint_summary(logits, valid),
            "counts": _state_counts(logits, target, valid, h0),
            "H0_margin": _stats(logits[h0]),
            "category_margin": {name: _stats(logits[mask]) for name, mask in categories.items()},
        }

    residuals: dict[str, Any] = {}
    corrections: dict[str, Any] = {}
    for arm in ARMS:
        delta = arrays[f"{arm}_delta"]
        residuals[arm] = {
            "H0": _residual_stats(delta[h0], bound),
            "matched_TN_control": _residual_stats(delta[categories["matched_TN_control"]], bound),
            "correct_foreground": _residual_stats(delta[categories["correct_foreground"]], bound),
            "ordinary_background": _residual_stats(delta[categories["ordinary_background"]], bound),
            "signed_correction": _stats(delta[np.broadcast_to(valid[None], delta.shape)]),
        }
        corrections[arm] = _correction_counts(arrays["step2"], arrays[f"{arm}_full14_final"], target, valid)

    metric_members: dict[str, list[dict[str, Any]]] = {}
    standard: dict[str, Any] = {}
    for state in ("C0_full14_final", "C00_full14_final", "CS_full14_final"):
        metric_members[state], standard[state], _ = _metric_rows(arrays[state], target, native_hw, pairs)

    member_rows: list[dict[str, Any]] = []
    for member in range(N_MEMBERS):
        row: dict[str, Any] = {"case": case, "patient": patient, "member": member,
                               "h0_member_pixels": int(h0[member].sum()),
                               "matched_tn_control_pixels": int(categories["matched_TN_control"][member].sum()),
                               "correct_foreground_pixels": int(categories["correct_foreground"][member].sum()),
                               "ordinary_background_pixels": int(categories["ordinary_background"][member].sum()),
                               "native_height": native_hw[0], "native_width": native_hw[1],
                               "padded_height": int(arrays["step2"].shape[1]), "padded_width": int(arrays["step2"].shape[2])}
        for state in STATE_NAMES:
            logits = arrays[state]
            hmask = h0[member]
            row[f"{state}_h0_recovered"] = int(((logits[member] >= LOGIT_THRESHOLD) & hmask).sum())
            row.update(_stats_row(f"{state}_h0_margin", _stats(logits[member][hmask])))
        for arm in ARMS:
            delta = arrays[f"{arm}_delta"][member]
            row.update(_stats_row(f"{arm}_h0_residual", _residual_stats(delta[h0[member]], bound)))
            row[f"{arm}_h0_saturated"] = int((np.abs(delta[h0[member]]) >= bound * (1.0 - 1e-4)).sum())
            for key in ("new_tp_per_member", "new_fp_per_member", "lost_tp_per_member", "removed_fp_per_member"):
                row[f"{arm}_{key.replace('_per_member', '')}"] = corrections[arm][key][member]
        for state in ("C0_full14_final", "C00_full14_final", "CS_full14_final"):
            metrics = metric_members[state][member]
            row.update({f"{state}_{key}": value for key, value in metrics.items()})
        member_rows.append(row)

    case_record = {
        "case": case,
        "patient": patient,
        "native_hw": list(native_hw),
        "padded_hw": [int(arrays["step2"].shape[1]), int(arrays["step2"].shape[2])],
        "fixed_h0_cache": {"member_pixels": int(h0.sum()), "unique_pixels": int(h0.any(axis=0).sum())},
        "states": state_records,
        "residuals": residuals,
        "corrections_vs_C0_step2": corrections,
        "standard_metrics": standard,
    }
    return case_record, member_rows, {
        "categories": categories,
        "metric_members": metric_members,
        "state_h0_values": {state: arrays[state][h0].copy() for state in STATE_NAMES},
        "residual_values": {
            arm: {
                "H0": arrays[f"{arm}_delta"][h0].copy(),
                "matched_TN_control": arrays[f"{arm}_delta"][categories["matched_TN_control"]].copy(),
                "correct_foreground": arrays[f"{arm}_delta"][categories["correct_foreground"]].copy(),
                "ordinary_background": arrays[f"{arm}_delta"][categories["ordinary_background"]].copy(),
                "signed_correction": arrays[f"{arm}_delta"][np.broadcast_to(valid[None], arrays[f"{arm}_delta"].shape)].copy(),
            }
            for arm in ARMS
        },
    }


def _make_accumulators(bound: float) -> dict[str, Any]:
    state_counts = {state: {} for state in STATE_NAMES}
    state_h0 = {state: _RunningStats(500_000, seed=1800 + index) for index, state in enumerate(STATE_NAMES)}
    category_values = {
        state: {category: _RunningStats(0) for category in ("H0_member", "matched_TN_control", "correct_foreground", "ordinary_background")}
        for state in STATE_NAMES
    }
    residuals = {arm: {category: _residual_accumulator(bound, 500_000 if category == "H0" else 0)
                       for category in ("H0", "matched_TN_control", "correct_foreground", "ordinary_background", "signed_correction")}
                  for arm in ARMS}
    corrections = {arm: {key: 0 for key in ("new_tp", "new_fp", "lost_tp", "removed_fp")} for arm in ARMS}
    standard = {state: {section: [] for section in ("overall_instance_metrics", "overlap_region_metrics", "overlap_pair_intersection_metrics")}
                for state in ("C0_full14_final", "C00_full14_final", "CS_full14_final")}
    return {"state_counts": state_counts, "state_h0": state_h0, "category_values": category_values,
            "residuals": residuals, "corrections": corrections, "standard": standard,
            "h0_members": 0, "h0_unique": 0, "cases": 0}


def _accumulate(acc: dict[str, Any], case_record: Mapping[str, Any], aux: Mapping[str, Any]) -> None:
    acc["cases"] += 1
    acc["h0_members"] += int(case_record["fixed_h0_cache"]["member_pixels"])
    acc["h0_unique"] += int(case_record["fixed_h0_cache"]["unique_pixels"])
    for state in STATE_NAMES:
        counts = case_record["states"][state]["counts"]
        for key, value in counts.items():
            acc["state_counts"][state][key] = acc["state_counts"][state].get(key, 0) + int(value)
        acc["state_h0"][state].add(aux["state_h0_values"][state])
        for category, stats in case_record["states"][state]["category_margin"].items():
            # Category accumulators are intentionally moments-only at this
            # stage; use each case's exact moments without retaining pixels.
            accumulator = acc["category_values"][state][category]
            n = int(stats["n"])
            if n:
                accumulator.n += n
                accumulator.sum += float(stats["mean"] * n)
                accumulator.sumsq += float((stats["std"] ** 2 + stats["mean"] ** 2) * n)
                accumulator.minimum = min(accumulator.minimum, float(stats["min"]))
                accumulator.maximum = max(accumulator.maximum, float(stats["max"]))
    for arm in ARMS:
        for category, values in aux["residual_values"][arm].items():
            _add_residual(acc["residuals"][arm][category], values)
        for key in acc["corrections"][arm]:
            acc["corrections"][arm][key] += int(case_record["corrections_vs_C0_step2"][arm][key])
    for state, result in case_record["standard_metrics"].items():
        for section in acc["standard"][state]:
            acc["standard"][state][section].append(result[section])


def _aggregate_standard(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "dsc": None, "nsd_2px": None, "voe": None, "msd_px": None, "msd_fail_rate": None, "ravd": None}
    result: dict[str, Any] = {"n": len(rows)}
    for key in ("dsc", "nsd_2px", "voe", "msd_px", "ravd"):
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        result[key] = None if not values else float(np.mean(values))
    result["msd_fail_rate"] = float(np.mean([float(row.get("msd_fail_rate", 0.0)) for row in rows]))
    return result


def _finish_aggregate(acc: Mapping[str, Any]) -> dict[str, Any]:
    states: dict[str, Any] = {}
    for state in STATE_NAMES:
        states[state] = {
            "counts": acc["state_counts"][state],
            "H0_margin": acc["state_h0"][state].summary(),
            "category_margin": {category: value.summary() for category, value in acc["category_values"][state].items()},
        }
    residuals = {arm: {category: _finish_residual(value) for category, value in categories.items()}
                 for arm, categories in acc["residuals"].items()}
    standard = {state: {section: _aggregate_standard(rows) for section, rows in sections.items()}
                for state, sections in acc["standard"].items()}
    return {"cases": int(acc["cases"]), "fixed_h0_member_pixels": int(acc["h0_members"]),
            "fixed_h0_unique_pixels": int(acc["h0_unique"]), "states": states,
            "residuals": residuals, "corrections_vs_C0_step2": acc["corrections"], "standard_metrics": standard}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_csv_rows(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state, value in aggregate["states"].items():
        for category, stats in {"H0_margin": value["H0_margin"], **value["category_margin"]}.items():
            row = {"kind": "margin", "state": state, "category": category}
            row.update(_stats_row("value", stats))
            rows.append(row)
        for key, number in value["counts"].items():
            rows.append({"kind": "count", "state": state, "category": "all", "metric": key, "value": number})
    for arm, categories in aggregate["residuals"].items():
        for category, stats in categories.items():
            row = {"kind": "residual", "arm": arm, "category": category}
            row.update(_stats_row("value", stats))
            row.update({"positive_fraction": stats.get("positive_fraction"), "negative_fraction": stats.get("negative_fraction"),
                        "saturated_count": stats.get("saturated_count"), "saturated_fraction": stats.get("saturated_fraction")})
            rows.append(row)
        for key, number in aggregate["corrections_vs_C0_step2"][arm].items():
            rows.append({"kind": "correction", "arm": arm, "category": "all", "metric": key, "value": number})
    return rows


def audit(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _resolve_inputs(args)
    out = inputs["out"]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    lock = inputs["lock"]
    manifest, fit_cases = _load_manifest(inputs["split_manifest"])
    if lock.get("c0_sha256") and str(lock["c0_sha256"]).lower() != C0_SHA256:
        raise ValueError("RUN_LOCK C0 SHA does not match the locked C0 constant")
    if lock.get("S_manifest_sha256") and (inputs["s_prior_root"] / "manifest.json").is_file():
        if _sha256(inputs["s_prior_root"] / "manifest.json") != str(lock["S_manifest_sha256"]).lower():
            raise ValueError("S manifest SHA mismatch")
    if not inputs["dataset"].is_dir():
        raise FileNotFoundError(inputs["dataset"])
    if not inputs["c0"].is_file() or _sha256(inputs["c0"]) != C0_SHA256:
        raise ValueError(f"C0 checkpoint missing or SHA mismatch: {inputs['c0']}")
    if len(list((inputs["dataset"] / "BoneSegmentation" / "masks" / "train").glob("*.npy"))) != TRAIN_CASES:
        raise ValueError("train dataset is not the locked RAM-W600 train425 split")
    h0_root = inputs["h0_root"]
    c0_head_path = inputs["run_root"] / "TRAIN_C00" / "final.pth"
    cs_head_path = inputs["run_root"] / "TRAIN_CS" / "final.pth"
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the endpoint audit")
    model, refiner, c0_payload = load_frozen_c0(inputs["c0"], device)
    bound = float(lock.get("bound", BOUND_DEFAULT))
    c00_head, c00_meta = _load_head(c0_head_path, "C00", bound, device)
    cs_head, cs_meta = _load_head(cs_head_path, "CS", bound, device)
    heads = {"C00": c00_head, "CS": cs_head}
    head_meta = {"C00": c00_meta, "CS": cs_meta}
    dataset = NativeWristDataset(inputs["dataset"], "train", augment=False)
    lookup = {path.stem: index for index, path in enumerate(dataset.mask_files)}
    if set(fit_cases) - set(lookup):
        raise ValueError(f"fit manifest cases missing from train dataset: {sorted(set(fit_cases) - set(lookup))[:5]}")
    pairs = top_training_pairs(inputs["dataset"], 15)
    loader = DataLoader(dataset, batch_size=1, sampler=[lookup[case] for case in fit_cases], num_workers=0)
    before_runtime = _runtime_snapshot((model, refiner))
    acc = _make_accumulators(bound)
    case_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    case_patient = manifest["case_patient_token"]
    with _preserve_runtime((model, refiner)), torch.inference_mode():
        for batch in loader:
            case = str(batch["case"][0])
            image = batch["image"].to(device, dtype=torch.float32)
            target = (batch["mask"].numpy()[0] > 0.5)
            valid = batch["valid"].numpy()[0, 0].astype(bool)
            native_hw = tuple(int(value) for value in batch["original_hw"][0].tolist())
            h0_path = h0_root / "train" / f"{case}.npz"
            cache = _load_h0_cache(h0_path, target.shape, native_hw, target, valid)
            h0 = cache["h0"]
            source_t, step1_t, step2_t, hidden_t = _capture_endpoints(model, refiner, image)
            source = source_t.detach().cpu().numpy()[0].copy()
            step1 = step1_t.detach().cpu().numpy()[0].copy()
            step2 = step2_t.detach().cpu().numpy()[0].copy()
            if bool((h0 & (source >= H0_SOURCE_THRESHOLD)).any()) or bool((h0 & (step2 >= LOGIT_THRESHOLD)).any()):
                raise ValueError(f"fixed H0 cache does not match C0 endpoint definition for {case}")
            padded_hw = (int(image.shape[-2]), int(image.shape[-1]))
            seam = _load_s_map(inputs["s_prior_root"], case, native_hw, padded_hw).to(device)
            prior_maps = PriorMaps(s=seam, o=None, image_extent=native_hw,
                                   metadata_hashes={"S_manifest": str(lock.get("S_manifest_sha256", "unknown"))})
            c00_t, c00_delta_t = heads["C00"](image, step2_t, hidden_t, prior_maps, arm="C00", chunk_size=3)
            cs_t, cs_delta_t = heads["CS"](image, step2_t, hidden_t, prior_maps, arm="CS", chunk_size=3)
            arrays = {
                "source": source,
                "step1": step1,
                "step2": step2,
                "C0_full14_final": step2,
                "C00_full14_final": c00_t.detach().cpu().numpy()[0].copy(),
                "CS_full14_final": cs_t.detach().cpu().numpy()[0].copy(),
                "C00_delta": c00_delta_t.detach().cpu().numpy()[0].copy(),
                "CS_delta": cs_delta_t.detach().cpu().numpy()[0].copy(),
            }
            record, rows, aux = _analyze_case(case, str(case_patient[case]), arrays, target, valid, h0, native_hw, bound, pairs)
            record["fixed_h0_cache"].update({"path": cache["path"], "sha256": cache["sha256"], "label": cache["label"]})
            case_rows.append(record)
            member_rows.extend(rows)
            _accumulate(acc, record, aux)
            del seam, prior_maps, source_t, step1_t, step2_t, hidden_t, c00_t, cs_t, c00_delta_t, cs_delta_t
    runtime_unchanged = _runtime_unchanged(before_runtime, (model, refiner))
    if not runtime_unchanged:
        raise RuntimeError("C0 model/refiner mode, buffers, or weights changed during read-only audit")
    aggregate = _finish_aggregate(acc)
    result = {
        "status": "COMPLETE",
        "experiment": "RAM_TARGETED_FIT_H0_ENDPOINT_AUDIT",
        "split": "train/probe_fit",
        "fit_cases": len(fit_cases),
        "fit_patients": len({str(case_patient[case]) for case in fit_cases}),
        "test_used": False,
        "val_used": False,
        "optimizer_updates": 0,
        "threshold": {"probability": 0.5, "logit": LOGIT_THRESHOLD},
        "fixed_h0_definition": "immutable C0 cache; valid & Y_k=1 & sum(Y)>=2 & C0_final_k<0 & C0_source_k<-5.00001",
        "endpoint_states": list(STATE_NAMES),
        "trajectory_policy": "endpoint-only; no intermediate trajectory fabricated or read",
        "inputs": {
            "run_lock": str(inputs["lock_path"]),
            "run_lock_sha256": _sha256(inputs["lock_path"]) if inputs["lock_path"].is_file() else None,
            "dataset": str(inputs["dataset"]),
            "split_manifest": str(inputs["split_manifest"]),
            "split_manifest_sha256": _sha256(inputs["split_manifest"]),
            "h0_root": str(h0_root),
            "s_prior_root": str(inputs["s_prior_root"]),
            "c0_checkpoint": str(inputs["c0"]),
            "c0_sha256": c0_payload["sha256"],
            "c0_load": c0_payload,
        },
        "head_checkpoints": head_meta,
        "bound": bound,
        "feature_contract": "C0 native padded source + R332 step1/step2 full14; frozen six-channel per-instance feature, own step2 logit, 12-channel step2 hidden, optional S only for CS",
        "runtime_preserved": runtime_unchanged,
        "aggregate": aggregate,
        "per_case": case_rows,
    }
    _write_json(out / "FIT_H0_AUDIT.json", result)
    _write_csv(out / "FIT_H0_PER_CASE_MEMBER.csv", member_rows)
    _write_csv(out / "FIT_H0_PER_CASE.csv", case_rows)
    _write_csv(out / "FIT_H0_AGGREGATE.csv", _aggregate_csv_rows(aggregate))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new isolated output directory")
    parser.add_argument("--run-lock", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--c0-checkpoint", "--c0", dest="c0_checkpoint", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--h0-root", type=Path, default=None)
    parser.add_argument("--s-prior-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda", help="torch device; CUDA is required by default")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = audit(args)
    print(json.dumps({"status": result["status"], "fit_cases": result["fit_cases"],
                      "fixed_h0_member_pixels": result["aggregate"]["fixed_h0_member_pixels"],
                      "fixed_h0_unique_pixels": result["aggregate"]["fixed_h0_unique_pixels"],
                      "runtime_preserved": result["runtime_preserved"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
