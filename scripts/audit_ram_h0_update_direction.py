#!/usr/bin/env python3
"""Read-only, one-step update-direction audit for the RAM C00 head.

The audit is deliberately counterfactual.  It loads the exact frozen C0
checkpoint and, for each of the eight cases locked by
``CORRECTOR_CHECK_LOCK.json``, creates a fresh C00 head and AdamW optimizer
from the saved step-512 state.  It computes the real full14 loss and its
component gradients, performs one isolated optimizer step, and evaluates the
fixed H0, co-located negative members, and C0-correct positive members before
and after that step.

The H0 cache is a diagnostic input only.  It is never passed to the model and
is loaded after the C0/head forward.  No checkpoint is written and no
historical head/optimizer state is mutated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ram_highres_capture import C0_SHA256, load_frozen_c0  # noqa: E402
from ram_so_conditional_corrector import (  # noqa: E402
    PriorMaps,
    RAMSOConditionalCorrector,
)
from r351_data import AlignedRamDataset  # noqa: E402
from run_ram_so_c00_cs_pair import (  # noqa: E402
    _capture_frozen,
    complete_loss as reference_complete_loss,
)
from train_r330_ram_pair_surface_volume_prior import (  # noqa: E402
    soft_surface_nsd_loss,
    volume_losses,
)


COMPONENT_WEIGHTS: dict[str, float] = {
    "bce": 0.55,
    "dice": 0.45,
    "surface": 0.03,
    "instance_volume": 0.005,
    "pair_volume": 0.01,
}
COMPONENT_ORDER = tuple(COMPONENT_WEIGHTS)
DEFAULT_CASES = (
    "0001_0001_L",
    "0006_0001_L",
    "0001_0001_R",
    "0006_0001_R",
    "EXT_0002_0001_L",
    "EXT_0003_0001_L",
    "EXT_0079_0001_R",
    "EXT_0081_0001_R",
)
DEFAULT_STEPS = 512
DEFAULT_CLIP_GRAD = 5.0
DEFAULT_HEAD_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
PACK_BITORDER = "little"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_candidates(raw: Any, lock_path: Path) -> list[Path]:
    if raw is None:
        return []
    value = Path(str(raw))
    candidates: list[Path] = []
    if value.is_absolute():
        candidates.append(value)
    else:
        candidates.extend(
            (
                lock_path.parent / value,
                Path.cwd() / value,
                SCRIPT_DIR.parent / value,
            )
        )
    # A lock written on the server can carry a POSIX absolute path.  Keep that
    # path in the error message, but never silently substitute a same-named
    # artifact from an unrelated directory.
    return [candidate.resolve() for candidate in candidates]


def resolve_existing(raw: Any, lock_path: Path, *, label: str) -> Path:
    candidates = _path_candidates(raw, lock_path)
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate
    shown = ", ".join(str(item) for item in candidates) or repr(raw)
    raise FileNotFoundError(f"{label} not found; checked {shown}")


def resolve_optional(raw: Any, lock_path: Path) -> Path | None:
    if raw is None:
        return None
    candidates = _path_candidates(raw, lock_path)
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _first(lock: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = lock.get(key)
        if value is not None:
            return value
    return None


def _resolve_from_values(values: Sequence[Any], lock_path: Path, *, label: str) -> Path:
    tried: list[Path] = []
    for raw in values:
        candidates = _path_candidates(raw, lock_path)
        tried.extend(candidates)
        for candidate in candidates:
            if candidate.is_file() or candidate.is_dir():
                return candidate
    shown = ", ".join(str(item) for item in tried) or "<none>"
    raise FileNotFoundError(f"{label} not found; checked {shown}")


def component_losses(
    final: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    original_hw: Sequence[int],
) -> dict[str, torch.Tensor]:
    """Return the differentiable five tensors used by the locked C00 loss.

    This is intentionally a direct reconstruction of
    ``run_ram_so_c00_cs_pair.complete_loss``.  The native crop is applied
    before every term, and the R330 geometry functions see [14,1,H,W].
    """

    if final.ndim != 4 or target.shape != final.shape:
        raise ValueError("final and target must both have shape [1,14,H,W]")
    if valid.ndim != 4 or valid.shape[0] != final.shape[0] or valid.shape[1] != 1:
        raise ValueError("valid must have shape [1,1,H,W]")
    h, w = (int(value) for value in original_hw)
    if not (0 < h <= final.shape[-2] and 0 < w <= final.shape[-1]):
        raise ValueError(f"invalid native extent {(h, w)} for {tuple(final.shape[-2:])}")
    logits = final[:, :, :h, :w].float()
    truth = target[:, :, :h, :w].float()
    native_valid = valid[:, :, :h, :w].float()
    if truth.shape[1] != 14 or not bool(native_valid.all()):
        raise ValueError("native valid/14 target contract failed")

    bce = (
        F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * native_valid
    ).sum() / native_valid.sum().clamp_min(1.0) / 14
    probability = logits.sigmoid()
    intersection = (probability * truth).sum((0, 2, 3))
    mass = probability.sum((0, 2, 3)) + truth.sum((0, 2, 3))
    dice = (1.0 - (2.0 * intersection + 1.0) / (mass + 1.0)).mean()
    members = probability[0, :, None]
    labels = truth[0, :, None]
    surface = soft_surface_nsd_loss(members, labels)
    instance_volume, pair_volume = volume_losses(members, labels)
    values = {
        "bce": bce,
        "dice": dice,
        "surface": surface,
        "instance_volume": instance_volume,
        "pair_volume": pair_volume,
    }
    if not all(bool(torch.isfinite(value)) for value in values.values()):
        raise FloatingPointError("nonfinite loss component")
    return values


def weighted_loss(components: Mapping[str, torch.Tensor]) -> torch.Tensor:
    missing = [name for name in COMPONENT_ORDER if name not in components]
    if missing:
        raise KeyError(f"missing loss components: {missing}")
    result = sum(
        (COMPONENT_WEIGHTS[name] * components[name] for name in COMPONENT_ORDER),
        torch.zeros_like(components["bce"]),
    )
    if not bool(torch.isfinite(result)):
        raise FloatingPointError("nonfinite weighted loss")
    return result


def assert_loss_parity(
    final: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    original_hw: Sequence[int],
    *,
    atol: float = 2e-7,
    rtol: float = 2e-6,
) -> dict[str, float]:
    """Compare the reconstruction with the real runner implementation."""

    rebuilt = component_losses(final, target, valid, original_hw)
    rebuilt_total = weighted_loss(rebuilt)
    reference_total, reference_parts = reference_complete_loss(final, target, valid, original_hw)
    differences = {
        "total": abs(float((rebuilt_total - reference_total).detach())),
    }
    for name in COMPONENT_ORDER:
        differences[name] = abs(float(rebuilt[name].detach()) - float(reference_parts[name]))
    if not torch.allclose(rebuilt_total.detach(), reference_total.detach(), atol=atol, rtol=rtol):
        raise AssertionError(f"full loss parity failed: {differences}")
    for name in COMPONENT_ORDER:
        if not math.isclose(
            float(rebuilt[name].detach()),
            float(reference_parts[name]),
            abs_tol=atol,
            rel_tol=rtol,
        ):
            raise AssertionError(f"{name} parity failed: {differences}")
    return differences


def _unpack_bool(packed: Any, shape: Sequence[int]) -> np.ndarray:
    shape_tuple = tuple(int(value) for value in shape)
    count = int(np.prod(shape_tuple, dtype=np.int64))
    values = np.asarray(packed)
    if values.dtype == np.bool_ and values.size == count:
        return values.reshape(shape_tuple).astype(bool, copy=False)
    values = values.astype(np.uint8, copy=False).reshape(-1)
    return np.unpackbits(values, bitorder=PACK_BITORDER, count=count).reshape(shape_tuple).astype(bool)


def _archive_value(archive: Any, *keys: str) -> Any:
    for key in keys:
        if isinstance(archive, Mapping) and key in archive:
            return archive[key]
        if hasattr(archive, "files") and key in archive.files:
            return archive[key]
    return None


def load_fixed_h0(path: Path, expected_shape: Sequence[int]) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load one packed fixed-H0 cache and normalize native caches to padding."""

    expected = tuple(int(value) for value in expected_shape)
    if len(expected) != 3 or expected[0] != 14:
        raise ValueError(f"expected H0 shape must be [14,H,W], got {expected}")
    payload: Any
    if path.suffix.lower() in {".pt", ".pth", ".pickle"}:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"H0 torch cache is not a mapping: {path}")
    else:
        payload = np.load(path, allow_pickle=False)
    try:
        packed = _archive_value(payload, "h0", "H0", "h0_packed", "h0_bits")
        if packed is None:
            raise ValueError(f"H0 cache requires h0/H0 packed data: {path}")
        shape_value = _archive_value(payload, "shape", "h0_shape")
        if shape_value is None:
            array = np.asarray(packed)
            if array.dtype != np.bool_ or array.ndim != 3:
                raise ValueError(f"packed H0 cache requires shape metadata: {path}")
            shape = tuple(int(value) for value in array.shape)
        else:
            shape = tuple(int(value) for value in np.asarray(shape_value).reshape(-1).tolist())
        h0 = _unpack_bool(packed, shape)
        original_hw_value = _archive_value(payload, "original_hw", "native_hw")
        original_hw = None
        if original_hw_value is not None:
            original_hw = tuple(int(value) for value in np.asarray(original_hw_value).reshape(-1).tolist())
            if len(original_hw) != 2:
                raise ValueError(f"invalid original_hw in H0 cache: {path}")
        if h0.shape != expected:
            if h0.shape[0] != expected[0] or original_hw is None or h0.shape[1:] != original_hw:
                raise ValueError(f"H0 shape {h0.shape} does not match expected {expected}: {path}")
            padded = np.zeros(expected, dtype=bool)
            padded[:, : h0.shape[1], : h0.shape[2]] = h0
            h0 = padded
        if h0.shape != expected:
            raise ValueError(f"normalized H0 shape {h0.shape} does not match {expected}: {path}")
        tensor = torch.from_numpy(h0.copy())
        metadata = {
            "path": str(path),
            "sha256": sha256_file(path),
            "stored_shape": list(shape),
            "normalized_shape": list(expected),
            "original_hw": list(original_hw) if original_hw is not None else None,
            "members": int(h0.sum()),
            "unique_pixels": int(h0.any(axis=0).sum()),
        }
        return tensor, metadata
    finally:
        if hasattr(payload, "close"):
            payload.close()


def _case_cache_path(root: Path, case: str) -> Path:
    candidates = (
        root / "train" / f"{case}.npz",
        root / "train" / f"{case}.pt",
        root / f"{case}.npz",
        root / f"{case}.pt",
        root / f"{case}.pth",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no fixed H0 cache for train/{case} under {root}")


def build_masks(
    target: torch.Tensor,
    valid: torch.Tensor,
    c0_final: torch.Tensor,
    h0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build fixed H0, co-located negative, and C0-correct positive masks.

    ``H`` is taken verbatim from the packed cache.  ``N`` is a true-negative
    member at an H0 union pixel whose original C0 final logit is negative.
    ``C`` is a true-positive member that C0 already predicts positive.  These
    definitions are fixed before the counterfactual optimizer step.
    """

    if not (target.ndim == valid.ndim == c0_final.ndim == h0.ndim == 4):
        raise ValueError("target, valid, c0_final, and h0 must be batched tensors")
    if target.shape != c0_final.shape or target.shape != h0.shape:
        raise ValueError("target, c0_final, and h0 shapes must match")
    if valid.shape != (target.shape[0], 1, target.shape[2], target.shape[3]):
        raise ValueError("valid must have shape [B,1,H,W]")
    if target.shape[0] != 1 or target.shape[1] != 14:
        raise ValueError("diagnostic masks require a single full14 batch")
    target_bool = target > 0.5
    valid_members = valid > 0.5
    h0_bool = h0.bool()
    if bool((h0_bool & ~(target_bool & valid_members)).any()):
        raise ValueError("fixed H0 is not a subset of true valid positive members")
    if bool((h0_bool & (c0_final >= 0)).any()):
        raise ValueError("fixed H0 contains a C0-final-positive member pixel")
    h0_union = h0_bool.any(dim=1, keepdim=True).expand_as(target_bool)
    original_negative = c0_final < 0
    masks = {
        "H": h0_bool,
        "N": valid_members & ~target_bool & h0_union & original_negative,
        "C": valid_members & target_bool & ~h0_bool & ~original_negative,
    }
    return masks


def _mask_values(logits: torch.Tensor, mask: torch.Tensor, *, positive: bool) -> torch.Tensor:
    selected = logits[mask]
    if selected.numel() == 0:
        return selected
    return F.softplus(-selected if positive else selected)


def q_values(logits: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for name, positive in (("H", True), ("N", False), ("C", True)):
        selected = _mask_values(logits, masks[name], positive=positive)
        values[name] = {
            "n": int(selected.numel()),
            "mean": None if selected.numel() == 0 else float(selected.mean().detach()),
            "sum": None if selected.numel() == 0 else float(selected.sum().detach()),
        }
    return values


def _quantiles(values: torch.Tensor) -> dict[str, Any]:
    if values.numel() == 0:
        return {"n": 0, "mean": None, "q05": None, "q50": None, "q95": None, "min": None, "max": None}
    values = values.detach().float().cpu()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "q05": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def margin_stats(logits: torch.Tensor, masks: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _quantiles(logits[mask]) for name, mask in masks.items()}


def _tensor_l2(value: torch.Tensor | None) -> float:
    if value is None:
        return 0.0
    return float(value.detach().float().square().sum().sqrt())


def _gradient_vector(
    value: torch.Tensor | None,
    parameters: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor]:
    if value is None or not value.requires_grad:
        return [torch.zeros_like(parameter) for parameter in parameters]
    gradients = torch.autograd.grad(
        value,
        tuple(parameters),
        retain_graph=True,
        allow_unused=True,
    )
    return [
        torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
        for parameter, gradient in zip(parameters, gradients)
    ]


def _per_parameter_norms(
    gradients: Sequence[torch.Tensor],
    names: Sequence[str],
) -> dict[str, float]:
    return {name: _tensor_l2(gradient) for name, gradient in zip(names, gradients)}


def _component_gradient_report(
    components: Mapping[str, torch.Tensor],
    parameters: Sequence[torch.nn.Parameter],
    names: Sequence[str],
    clip_scale: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in COMPONENT_ORDER:
        raw = _gradient_vector(components[component], parameters)
        weight = COMPONENT_WEIGHTS[component]
        weighted = [weight * gradient for gradient in raw]
        result[component] = {
            "weight": weight,
            "raw_l2": math.sqrt(sum(_tensor_l2(value) ** 2 for value in raw)),
            "weighted_l2": math.sqrt(sum(_tensor_l2(value) ** 2 for value in weighted)),
            "clipped_l2_global_scale": math.sqrt(
                sum(_tensor_l2(value) ** 2 for value in weighted)
            )
            * clip_scale,
            "per_parameter_raw_l2": _per_parameter_norms(raw, names),
            "per_parameter_weighted_l2": _per_parameter_norms(weighted, names),
            "per_parameter_clipped_l2_global_scale": _per_parameter_norms(
                [clip_scale * value for value in weighted], names
            ),
        }
    return result


def _optimizer_state_report(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    names: Sequence[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, parameter in zip(names, parameters):
        state = optimizer.state.get(parameter, {})
        step = state.get("step", 0)
        if isinstance(step, torch.Tensor):
            step = float(step.detach().cpu())
        report[name] = {
            "step": float(step),
            "exp_avg_l2": _tensor_l2(state.get("exp_avg")),
            "exp_avg_sq_l2": _tensor_l2(state.get("exp_avg_sq")),
            "exp_avg_sq_sum": None
            if state.get("exp_avg_sq") is None
            else float(state["exp_avg_sq"].detach().float().sum()),
        }
    return report


def _head_path(lock: Mapping[str, Any], lock_path: Path, override: Path | None) -> Path:
    values: list[Any] = []
    if override is not None:
        values.append(override)
    for key in ("c00_checkpoint", "head_checkpoint", "c00_head", "checkpoint"):
        if lock.get(key) is not None:
            values.append(lock[key])
    run_root = lock.get("run_root")
    if run_root is not None:
        values.append(Path(str(run_root)) / "TRAIN_C00" / "final.pth")
    # This fallback is only for the checked-in run layout; a server lock with
    # another run root still requires an explicit path or matching lock field.
    values.append(SCRIPT_DIR.parent / "outputs" / "ram_so_c00_cs_20260929" / "results" / "TRAIN_C00" / "final.pth")
    return _resolve_from_values(values, lock_path, label="C00 head checkpoint")


def _h0_root(lock: Mapping[str, Any], lock_path: Path, override: Path | None) -> Path:
    values: list[Any] = []
    if override is not None:
        values.append(override)
    for key in (
        "h0_root",
        "fixed_h0_root",
        "h0_cache_root",
        "fixed_H0_root",
        "h0_cache",
    ):
        if lock.get(key) is not None:
            values.append(lock[key])
    return _resolve_from_values(values, lock_path, label="fixed H0 cache root")


def _fresh_head_optimizer(
    checkpoint: Mapping[str, Any],
    lock: Mapping[str, Any],
    device: torch.device,
) -> tuple[RAMSOConditionalCorrector, torch.optim.Optimizer]:
    bound = float(lock.get("bound", 8.0))
    head = RAMSOConditionalCorrector(bound=bound).to(device).float()
    state = checkpoint.get("head")
    if not isinstance(state, Mapping):
        raise ValueError("C00 checkpoint has no head state mapping")
    head.load_state_dict(state, strict=True)
    head.eval()
    learning_rate = float(lock.get("head_lr", DEFAULT_HEAD_LR))
    weight_decay = float(lock.get("weight_decay", DEFAULT_WEIGHT_DECAY))
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("C00 checkpoint has no AdamW optimizer state")
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    return head, optimizer


def _case_counts(
    prediction: torch.Tensor,
    c0_prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    valid_members = valid.expand_as(target).bool()
    positive = target > 0.5
    negative = ~positive
    return {
        "H0_recovered": int((prediction & masks["H"]).sum()),
        "co_located_negative_FP": int((prediction & masks["N"]).sum()),
        "C0_correct_positive_TP": int((prediction & masks["C"]).sum()),
        "all_TP": int((prediction & positive & valid_members).sum()),
        "all_FP": int((prediction & negative & valid_members).sum()),
        "new_TP_vs_C0": int((prediction & ~c0_prediction & positive & valid_members).sum()),
        "new_FP_vs_C0": int((prediction & ~c0_prediction & negative & valid_members).sum()),
        "lost_TP_vs_C0": int((~prediction & c0_prediction & positive & valid_members).sum()),
        "removed_FP_vs_C0": int((~prediction & c0_prediction & negative & valid_members).sum()),
    }


def _mask_counts(masks: Mapping[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, mask in masks.items():
        result[name] = {
            "member_pixels": int(mask.sum()),
            "unique_pixels": int(mask.any(dim=1).sum()),
        }
    return result


def _make_batch(item: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    image = item["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
    target = item["mask"].unsqueeze(0).to(device=device, dtype=torch.float32)
    valid = item["valid"].unsqueeze(0).to(device=device, dtype=torch.float32)
    seam = item["seam"].unsqueeze(0).to(device=device, dtype=torch.float32)
    original_hw = tuple(int(value) for value in item["original_hw"].reshape(-1).tolist())
    if image.shape[1] != 1 or target.shape[1] != 14:
        raise ValueError(f"unexpected batch shapes: {tuple(image.shape)}, {tuple(target.shape)}")
    if valid.shape[1] != 1 or tuple(image.shape[-2:]) != tuple(valid.shape[-2:]):
        raise ValueError("image/valid shape mismatch")
    maps = PriorMaps(s=seam, o=None, image_extent=original_hw)
    return {
        "image": image,
        "target": target,
        "valid": valid,
        "seam": seam,
        "original_hw": original_hw,
        "maps": maps,
        "case": str(item["case"]),
    }


def analyze_case(
    *,
    model: torch.nn.Module,
    refiner: torch.nn.Module,
    checkpoint: Mapping[str, Any],
    batch: Mapping[str, Any],
    h0_path: Path,
    lock: Mapping[str, Any],
    device: torch.device,
    clip_grad: float,
) -> dict[str, Any]:
    image = batch["image"]
    target = batch["target"]
    valid = batch["valid"]
    original_hw = batch["original_hw"]

    # C0 is frozen and the head sees only detached C0 features and the
    # image-derived S map.  H0 is intentionally loaded after this forward.
    with torch.no_grad():
        _, _, c0_final, hidden = _capture_frozen(model, refiner, image)
    head, optimizer = _fresh_head_optimizer(checkpoint, lock, device)
    names_and_parameters = list(head.named_parameters())
    names = [name for name, _ in names_and_parameters]
    parameters = [parameter for _, parameter in names_and_parameters]

    with torch.enable_grad():
        final_before, delta_before = head(
            image,
            c0_final,
            hidden,
            batch["maps"],
            arm="C00",
            chunk_size=3,
        )
    h0_cpu, h0_metadata = load_fixed_h0(h0_path, target.shape[1:])
    h0 = h0_cpu.to(device=device).unsqueeze(0)
    masks = build_masks(target, valid, c0_final, h0)
    parity = assert_loss_parity(final_before, target, valid, original_hw)
    components = component_losses(final_before, target, valid, original_hw)
    total = weighted_loss(components)

    q_h_tensor = _mask_values(final_before, masks["H"], positive=True)
    q_h_scalar = q_h_tensor.mean() if q_h_tensor.numel() else total * 0.0
    q_h_grad = _gradient_vector(q_h_scalar, parameters)

    # Obtain the real pre-clip gradients while retaining the graph for the
    # component/Q diagnostics, then apply exactly the runner's global norm
    # clip before the isolated AdamW step.
    raw_total_grad = _gradient_vector(total, parameters)
    raw_total_l2 = math.sqrt(sum(_tensor_l2(value) ** 2 for value in raw_total_grad))
    if not math.isfinite(raw_total_l2):
        raise FloatingPointError(f"nonfinite total gradient for {batch['case']}")
    clip_scale = min(1.0, float(clip_grad) / max(raw_total_l2, 1e-12))
    component_report = _component_gradient_report(components, parameters, names, clip_scale)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
    clipped_total_grad = [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    ]
    clipped_total_l2 = math.sqrt(sum(_tensor_l2(value) ** 2 for value in clipped_total_grad))
    total_report = {
        "raw_l2": raw_total_l2,
        "weighted_l2": raw_total_l2,
        "clipped_l2": clipped_total_l2,
        "clip_scale": clip_scale,
        "per_parameter_raw_l2": _per_parameter_norms(raw_total_grad, names),
        "per_parameter_clipped_l2": _per_parameter_norms(clipped_total_grad, names),
    }

    before_parameters = {name: parameter.detach().clone() for name, parameter in names_and_parameters}
    optimizer_state_before = _optimizer_state_report(optimizer, parameters, names)
    optimizer.step()
    deltas = [parameter.detach() - before_parameters[name] for name, parameter in names_and_parameters]
    delta_l2 = math.sqrt(sum(_tensor_l2(value) ** 2 for value in deltas))
    g_h_delta = sum(
        float((gradient.detach().float() * delta.detach().float()).sum())
        for gradient, delta in zip(q_h_grad, deltas)
    )
    optimizer_state_after = _optimizer_state_report(optimizer, parameters, names)

    with torch.no_grad():
        final_after, delta_after = head(
            image,
            c0_final,
            hidden,
            batch["maps"],
            arm="C00",
            chunk_size=3,
        )
    c0_prediction = c0_final >= 0
    before_prediction = final_before >= 0
    after_prediction = final_after >= 0
    q_before = q_values(final_before, masks)
    q_after = q_values(final_after, masks)

    return {
        "case": batch["case"],
        "fixed_h0_cache": h0_metadata,
        "mask_counts": _mask_counts(masks),
        "c0_final_margin": margin_stats(c0_final, masks),
        "final_margin_before": margin_stats(final_before, masks),
        "final_margin_after": margin_stats(final_after, masks),
        "loss_parity_abs_diff": parity,
        "loss_before": {
            "total": float(total.detach()),
            "components": {name: float(components[name].detach()) for name in COMPONENT_ORDER},
        },
        "Q_before": q_before,
        "Q_after": q_after,
        "gradients": {
            "components": component_report,
            "total": total_report,
            "Q_H_raw": {
                "l2": math.sqrt(sum(_tensor_l2(value) ** 2 for value in q_h_grad)),
                "per_parameter_l2": _per_parameter_norms(q_h_grad, names),
            },
        },
        "optimizer": {
            "name": "AdamW",
            "lr": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "clip_grad": float(clip_grad),
            "AMP": False,
            "state_before": optimizer_state_before,
            "state_after": optimizer_state_after,
        },
        "delta_theta": {
            "total_l2": delta_l2,
            "per_parameter_l2": _per_parameter_norms(deltas, names),
            "dot_Q_H_gradient_delta_theta": g_h_delta,
        },
        "counts_before": _case_counts(before_prediction, c0_prediction, target > 0, valid, masks),
        "counts_after": _case_counts(after_prediction, c0_prediction, target > 0, valid, masks),
        "correction_before": {
            "mean": float(delta_before.detach().mean()),
            "abs_mean": float(delta_before.detach().abs().mean()),
            "max_abs": float(delta_before.detach().abs().max()),
        },
        "correction_after": {
            "mean": float(delta_after.detach().mean()),
            "abs_mean": float(delta_after.detach().abs().mean()),
            "max_abs": float(delta_after.detach().abs().max()),
        },
    }


def _load_cases(
    dataset_root: Path,
    prior_root: Path,
    cases: Sequence[str],
) -> list[dict[str, Any]]:
    dataset = AlignedRamDataset(dataset_root, prior_root, "train", False)
    index_by_case = {path.stem: index for index, path in enumerate(dataset.mask_files)}
    missing = [case for case in cases if case not in index_by_case]
    if missing:
        raise ValueError(f"locked diagnostic cases absent from train dataset: {missing}")
    return [dataset[index_by_case[case]] for case in cases]


def run(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("test_used") is True:
        raise ValueError("test-used lock cannot authorize this diagnostic")
    cases = tuple(str(value) for value in lock.get("cases", DEFAULT_CASES))
    if cases != DEFAULT_CASES:
        raise ValueError(f"expected the locked eight-case set {DEFAULT_CASES}, got {cases}")
    if args.out.exists():
        raise FileExistsError(args.out)

    c0_raw = args.c0 if args.c0 is not None else _first(lock, "c0", "c0_checkpoint", "checkpoint")
    c0_path = resolve_existing(c0_raw, lock_path, label="C0 checkpoint")
    expected_c0 = str(_first(lock, "c0_sha256", "C0_sha256") or C0_SHA256)
    if expected_c0 != C0_SHA256:
        raise ValueError(f"lock C0 SHA differs from required frozen C0: {expected_c0}")
    c0_actual = sha256_file(c0_path)
    if c0_actual != C0_SHA256:
        raise ValueError(f"C0 SHA256 mismatch: expected {C0_SHA256}, got {c0_actual}")

    head_path = _head_path(lock, lock_path, args.head)
    head_checkpoint_sha = sha256_file(head_path)
    expected_head_sha = lock.get("head_sha256")
    if expected_head_sha is not None and head_checkpoint_sha != expected_head_sha:
        raise ValueError(f"C00 head SHA256 mismatch: {head_checkpoint_sha}")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("C00 checkpoint must be a mapping")
    if checkpoint.get("arm") not in (None, "C00"):
        raise ValueError(f"expected C00 head checkpoint, got {checkpoint.get('arm')!r}")
    if int(checkpoint.get("steps", DEFAULT_STEPS)) != DEFAULT_STEPS:
        raise ValueError("audit requires the C00 step-512 checkpoint")

    dataset_raw = args.dataset if args.dataset is not None else _first(lock, "dataset", "dataset_root")
    prior_raw = args.s_prior_root if args.s_prior_root is not None else _first(lock, "s_prior_root", "prior_root")
    dataset_root = resolve_existing(dataset_raw, lock_path, label="RAM dataset root")
    prior_root = resolve_existing(prior_raw, lock_path, label="RAM S prior root")
    h0_root = _h0_root(lock, lock_path, args.h0_root)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.set_num_threads(max(1, int(args.cpu_threads)))

    model, refiner, c0_payload = load_frozen_c0(c0_path, device)
    model_digest_before = _state_digest(model)
    refiner_digest_before = _state_digest(refiner)
    batches = [_make_batch(item, device) for item in _load_cases(dataset_root, prior_root, cases)]
    rows: list[dict[str, Any]] = []
    for batch in batches:
        h0_path = _case_cache_path(h0_root, batch["case"])
        rows.append(
            analyze_case(
                model=model,
                refiner=refiner,
                checkpoint=checkpoint,
                batch=batch,
                h0_path=h0_path,
                lock=lock,
                device=device,
                clip_grad=float(args.clip_grad if args.clip_grad is not None else lock.get("clip_grad", DEFAULT_CLIP_GRAD)),
            )
        )
    if _state_digest(model) != model_digest_before or _state_digest(refiner) != refiner_digest_before:
        raise RuntimeError("frozen C0 model/refiner changed during audit")

    max_parity = max(
        (max(float(value) for value in row["loss_parity_abs_diff"].values()) for row in rows),
        default=0.0,
    )
    result = {
        "status": "COMPLETE",
        "experiment": "RAM_C00_H0_UPDATE_DIRECTION_AUDIT",
        "diagnostic_only": True,
        "optimizer_steps_per_case": 1,
        "cases": list(cases),
        "device": str(device),
        "test_used": False,
        "test_split_read": False,
        "C0_sha256": c0_actual,
        "C0_checkpoint": str(c0_path),
        "C0_payload": c0_payload,
        "C00_head_checkpoint": str(head_path),
        "C00_head_checkpoint_sha256": head_checkpoint_sha,
        "C00_checkpoint_step": int(checkpoint.get("steps", DEFAULT_STEPS)),
        "loss_contract": {
            "source": "scripts/run_ram_so_c00_cs_pair.py::complete_loss",
            "weights": COMPONENT_WEIGHTS,
            "scope": "full14; native original_hw crop; padded pixels excluded",
            "R330_surface_and_volume": True,
        },
        "loss_component_numeric_parity_max_abs": max_parity,
        "per_case": rows,
        "caveats": [
            "Q_H is fixed positive BCE on cached H0 member pixels.",
            "Q_N is negative BCE on true-negative members at the H0 union where C0 final is negative.",
            "Q_C is positive BCE on C0-correct true-positive member pixels.",
            "The one-step dot product is local and does not establish generalization.",
            "The H0 cache is a fixed diagnostic mask and is never an inference input.",
        ],
    }
    write_json(args.out, result)
    return result


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head", type=Path, help="C00 step-512 head+Adam checkpoint override")
    parser.add_argument("--c0", type=Path, help="frozen C0 checkpoint override")
    parser.add_argument("--dataset", type=Path, help="RAM-W600 dataset root override")
    parser.add_argument("--s-prior-root", type=Path, help="canonical S map root override")
    parser.add_argument("--h0-root", type=Path, help="fixed packed H0 cache root override")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-grad", type=float)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.clip_grad is not None and (not math.isfinite(args.clip_grad) or args.clip_grad <= 0):
        raise ValueError("--clip-grad must be finite and positive")
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(result["per_case"]),
                "loss_component_numeric_parity_max_abs": result["loss_component_numeric_parity_max_abs"],
                "out": str(args.out),
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
