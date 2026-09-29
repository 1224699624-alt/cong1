"""Run a matched zero-O vs real-O RAM source-head fusion pilot.

The runner deliberately has a small surface area.  It consumes a lock file and
a pre-materialised, CPU-only schedule; it never samples the dataset during the
update loop.  Each invocation owns one arm and one output directory, so the
four arms can be launched independently from the same C0 checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_v31_ram_mature as legacy
from ram_o_source_head_adapter import attach_o_source_head


ARMS = ("OHEAD_ZERO", "OHEAD_REAL")
VARIANT_ID = "O_SOURCE_HEAD_GATED_1X1"
REFINER_LR = 2.1912418526906397e-7
DECODER_LR = 7e-6
WEIGHT_DECAY = 1e-4
INITIAL_SCALE = 1024.0
OLD_OPTIMIZER_STEP = 100
EXTRA_LAMBDA = 2.2
REFINER_CLIP = 12.0
DECODER_CLIP = 5.0
EXPECTED_D2_PARAMS = 10
EXPECTED_D2_NUMEL = 574336


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default), encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_path(value: Any, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _as_locked_file(value: Any, base: Path) -> Path:
    """Resolve a lock path, allowing repository-relative code paths as a fallback."""
    path = _as_path(value, base)
    if path.exists():
        return path
    repository_relative = (Path(__file__).resolve().parents[1] / Path(str(value))).resolve()
    return repository_relative if repository_relative.exists() else path


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _artifact_path(lock: Mapping[str, Any], lock_path: Path, *keys: str) -> Path:
    paths = lock.get("paths")
    values: list[Any] = []
    for key in keys:
        if key in lock:
            values.append(lock[key])
        if isinstance(paths, Mapping) and key in paths:
            values.append(paths[key])
    if not values:
        raise KeyError(f"lock has no artifact path for {keys}")
    return _as_locked_file(values[0], lock_path.parent)


def _expected_hash(lock: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = lock.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for nested in ("sha256", "sha", "hash"):
                if isinstance(value.get(nested), str):
                    return value[nested]
    hashes = lock.get("hashes")
    if isinstance(hashes, Mapping):
        for key in keys:
            value = hashes.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, Mapping):
                for nested in ("sha256", "sha", "hash"):
                    if isinstance(value.get(nested), str):
                        return value[nested]
    return None


def _verify_hash(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {path} expected {expected}, got {actual}")
    return actual


def _verify_lock_hashes(lock: Mapping[str, Any], lock_path: Path) -> dict[str, str]:
    """Verify every explicitly listed file and the runner/checkpoint hashes."""
    verified: dict[str, str] = {}
    file_hashes = lock.get("file_hashes", {})
    if file_hashes is not None:
        if not isinstance(file_hashes, Mapping):
            raise TypeError("lock.file_hashes must be an object")
        for raw_path, expected in file_hashes.items():
            path = _as_locked_file(raw_path, lock_path.parent)
            if isinstance(expected, Mapping):
                expected = _first(expected, "sha256", "sha", "hash")
            verified[str(path)] = _verify_hash(path, str(expected), "locked file")

    runner_expected = _expected_hash(lock, "runner_sha256", "code_sha256", "source_code_sha256")
    runner_actual = _verify_hash(Path(__file__).resolve(), runner_expected, "runner")
    verified["runner"] = runner_actual

    for names, label in (
        (("r332", "r332_checkpoint", "model_source"), "R332 source checkpoint"),
        (("overlap", "overlap_checkpoint", "image_only_overlap"), "overlap checkpoint"),
        (("checkpoint", "fresh100", "fresh100_checkpoint", "c0_checkpoint"), "C0 checkpoint"),
    ):
        try:
            path = _artifact_path(lock, lock_path, *names)
        except KeyError:
            continue
        expected = _expected_hash(lock, *(f"{name}_sha256" for name in names))
        verified[label] = _verify_hash(path, expected, label)

    return verified


def _load_schedule(lock: Mapping[str, Any], lock_path: Path) -> tuple[list[dict[str, Any]], Path | None, str]:
    candidate = _first(lock, "schedule_manifest", "schedule_path", "manifest_path")
    if candidate is None and isinstance(lock.get("schedule"), (str, Path)):
        candidate = lock["schedule"]

    manifest_path: Path | None = None
    if candidate is not None:
        manifest_path = _as_path(candidate, lock_path.parent)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"schedule manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = _expected_hash(lock, "schedule_sha256", "manifest_sha256", "schedule_manifest_sha256")
        manifest_hash = _verify_hash(manifest_path, expected, "schedule manifest")
    elif isinstance(lock.get("schedule"), Mapping):
        manifest = lock["schedule"]
        manifest_hash = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")
        ).hexdigest()
    elif isinstance(lock.get("steps"), list):
        manifest = lock
        manifest_hash = hashlib.sha256(
            json.dumps(lock["steps"], sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")
        ).hexdigest()
    else:
        raise KeyError("lock must provide schedule_manifest/schedule_path or schedule.steps")

    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("steps"), list):
        raise ValueError("schedule manifest must contain a steps list")
    steps: list[dict[str, Any]] = []
    for number, item in enumerate(manifest["steps"], 1):
        if not isinstance(item, Mapping):
            raise TypeError(f"schedule step {number} must be an object")
        required = ("path", "case", "batch_sha256", "indices", "denominator_c0")
        missing = [key for key in required if key not in item]
        if missing:
            raise KeyError(f"schedule step {number} missing {missing}")
        row = dict(item)
        row["step"] = number
        row["path"] = _as_path(item["path"], manifest_path.parent if manifest_path else lock_path.parent)
        row["case"] = str(item["case"])
        row["indices"] = [int(index) for index in item["indices"]]
        if len(row["indices"]) != 4 or len(set(row["indices"])) != 4 or any(i < 0 or i >= 14 for i in row["indices"]):
            raise ValueError(f"schedule step {number} must have four unique member indices in [0,13]")
        row["denominator_c0"] = float(item["denominator_c0"])
        if not math.isfinite(row["denominator_c0"]) or row["denominator_c0"] <= 0:
            raise ValueError(f"schedule step {number} has invalid denominator_c0")
        steps.append(row)
    if not steps:
        raise ValueError("schedule is empty")
    expected_steps = _first(lock, "schedule_steps", "num_steps", "updates")
    if expected_steps is not None and int(expected_steps) != len(steps):
        raise ValueError(f"schedule length mismatch: expected {expected_steps}, got {len(steps)}")
    return steps, manifest_path, manifest_hash


def _load_step_payload(row: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = Path(row["path"])
    if not path.is_file():
        raise FileNotFoundError(f"schedule payload does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"schedule payload is not an object: {path}")
    required = ("image", "mask", "valid", "h0")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"schedule payload {path} missing {missing}")
    image, mask, valid, h0 = (payload[key] for key in required)
    if not all(isinstance(value, torch.Tensor) for value in (image, mask, valid, h0)):
        raise TypeError(f"schedule payload {path} values must be tensors")
    if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 1:
        raise ValueError(f"image must be float32 [1,H,W], got {tuple(image.shape)} {image.dtype}")
    if mask.dtype not in (torch.float32, torch.uint8) or mask.ndim != 3 or mask.shape[0] != 14:
        raise ValueError(f"mask must be float32/uint8 [14,H,W], got {tuple(mask.shape)} {mask.dtype}")
    if valid.dtype != torch.bool or valid.ndim != 3 or valid.shape[0] != 1:
        raise ValueError(f"valid must be bool [1,H,W], got {tuple(valid.shape)} {valid.dtype}")
    if h0.dtype != torch.bool or h0.ndim != 3 or h0.shape[0] != 14:
        raise ValueError(f"h0 must be bool [14,H,W], got {tuple(h0.shape)} {h0.dtype}")
    if image.shape[1:] != mask.shape[1:] or image.shape[1:] != valid.shape[1:] or image.shape[1:] != h0.shape[1:]:
        raise ValueError(f"schedule tensor spatial shapes disagree in {path}")
    if not bool(torch.isfinite(image).all()) or not bool(torch.isfinite(mask.float()).all()):
        raise ValueError(f"nonfinite schedule tensor in {path}")
    target = mask.bool()
    valid_members = valid.expand_as(mask)
    if bool((h0 & ~(target & valid_members)).any()):
        raise ValueError(f"H0 is not a subset of GT and valid pixels in {path}")

    hash_keys = row.get("hash_keys", ("image", "mask", "valid"))
    if tuple(hash_keys) != ("image", "mask", "valid"):
        raise ValueError("schedule hash_keys must be the locked image/mask/valid protocol")
    digest = hashlib.sha256(
        image.contiguous().numpy().tobytes()
        + mask.contiguous().numpy().tobytes()
        + valid.contiguous().numpy().tobytes()
    ).hexdigest()
    if digest != str(row["batch_sha256"]):
        raise ValueError(f"dataset batch SHA256 mismatch for {row['case']}: expected {row['batch_sha256']}, got {digest}")
    for key in ("image", "mask", "valid", "h0"):
        expected_tensor_sha = row.get(key + "_sha256")
        if expected_tensor_sha is not None:
            actual_tensor_sha = hashlib.sha256(payload[key].contiguous().numpy().tobytes()).hexdigest()
            if actual_tensor_sha != expected_tensor_sha:
                raise ValueError(f"locked {key} SHA256 mismatch for {row['case']}")
    expected_file_hash = row.get("path_sha256")
    if expected_file_hash is not None and sha256_file(path) != str(expected_file_hash):
        raise ValueError(f"schedule payload SHA256 mismatch: {path}")
    return {
        "image": image,
        "mask": mask,
        "valid": valid,
        "h0": h0,
        "case": str(row["case"]),
        "indices": list(row["indices"]),
        "denominator_c0": float(row["denominator_c0"]),
        "batch_sha256": digest,
        "path": str(path),
        "step": int(row["step"]),
    }, digest


def _state_step(state: Mapping[str, Any]) -> int:
    value = state.get("step", 0)
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _state_moments(state: Mapping[str, Any], parameter: torch.Tensor) -> dict[str, Any]:
    if not state:
        return {"present": False, "step": 0, "exp_avg_l2": 0.0, "exp_avg_sq_l2": 0.0, "exp_avg_sq_sum": 0.0}
    avg = state.get("exp_avg")
    avg_sq = state.get("exp_avg_sq")
    return {
        "present": True,
        "step": _state_step(state),
        "exp_avg_l2": float(avg.detach().float().norm()) if isinstance(avg, torch.Tensor) else 0.0,
        "exp_avg_sq_l2": float(avg_sq.detach().float().norm()) if isinstance(avg_sq, torch.Tensor) else 0.0,
        "exp_avg_sq_sum": float(avg_sq.detach().float().sum()) if isinstance(avg_sq, torch.Tensor) else 0.0,
    }


def _capture_rng() -> tuple[Any, Any, torch.Tensor, list[torch.Tensor]]:
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def _restore_rng(state: tuple[Any, Any, torch.Tensor, list[torch.Tensor]]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if torch.cuda.is_available() and state[3]:
        torch.cuda.set_rng_state_all(state[3])


@contextmanager
def preserve_runtime(*modules: nn.Module):
    """Keep diagnostic observers from changing modes, buffers, grads, or RNG."""
    rng = _capture_rng()
    mode = {module: module.training for owner in modules for module in owner.modules()}
    buffers = {parameter: parameter.detach().clone() for owner in modules for parameter in owner.buffers()}
    grads = {parameter: None if parameter.grad is None else parameter.grad.detach().clone()
             for owner in modules for parameter in owner.parameters()}
    try:
        yield
    finally:
        for module, training in mode.items():
            module.training = training
        for buffer, value in buffers.items():
            buffer.copy_(value)
        for parameter, grad in grads.items():
            parameter.grad = None if grad is None else grad
        _restore_rng(rng)


def _configure_modes(model: nn.Module, refiner: nn.Module) -> None:
    model.eval()
    for index in (2, 3, 4):
        model.backbone.decoder.stages[index].train()
        model.backbone.decoder.transpconvs[index].train()
    model.backbone.decoder.seg_layers[4].train()
    refiner.train()
    for module in model.modules():
        if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)) and module.track_running_stats:
            raise RuntimeError("InstanceNorm running statistics are forbidden by the lock")


def _source_logits(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    return model(torch.cat((image, torch.zeros_like(image)), dim=1))


def _c0_denominator(model: nn.Module, image: torch.Tensor, target: torch.Tensor, indices: Sequence[int]) -> float:
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=image.is_cuda):
            logits = _source_logits(model, image)
            selected_target = target[0, list(indices)].unsqueeze(1)
            selected_prob = logits.sigmoid()[0, list(indices)].unsqueeze(1)
            local = F.max_pool2d((torch.abs(selected_prob - selected_target) > 0.15).float(), 11, 1, 5)
            denominator = (1.0 + 3.0 * local).sum()
    return float(denominator.detach().cpu())


def _validate_schedule_against_c0(model: nn.Module, steps: Sequence[Mapping[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for row in steps:
        payload, _ = _load_step_payload(row)
        image = payload["image"].unsqueeze(0).to(device)
        target = payload["mask"].float().unsqueeze(0).to(device)
        actual = _c0_denominator(model, image, target, payload["indices"])
        expected = float(payload["denominator_c0"])
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError(f"denominator_c0 mismatch at {payload['case']}: expected {expected}, got {actual}")
        payload["denominator_c0_actual"] = actual
        loaded.append(payload)
    return loaded


def _group_summary(named_groups: Mapping[str, Sequence[tuple[str, torch.Tensor]]], gradients: Mapping[int, torch.Tensor | None]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group, named in named_groups.items():
        squared = 0.0
        elements = 0
        missing = 0
        nonfinite = 0
        for _, parameter in named:
            elements += parameter.numel()
            gradient = gradients.get(id(parameter))
            if gradient is None:
                missing += parameter.numel()
                continue
            value = gradient.detach().float()
            squared += float(value.square().sum())
            nonfinite += int((~torch.isfinite(value)).sum())
        output[group] = {
            "l2": squared ** 0.5,
            "elements": elements,
            "missing_elements": missing,
            "nonfinite_elements": nonfinite,
        }
    return output


def _autograd_group_report(named_groups: Mapping[str, Sequence[tuple[str, torch.Tensor]]], value: torch.Tensor,
                           trainable: Sequence[torch.Tensor], retain_graph: bool = True) -> tuple[dict[str, Any], dict[int, torch.Tensor | None]]:
    if not value.requires_grad:
        gradients = {id(parameter): None for parameter in trainable}
        chosen_scale = None
    else:
        raw = None
        chosen_scale = None
        for scale in (1024.0, 64.0, 1.0):
            candidate = torch.autograd.grad(value * scale, trainable, retain_graph=retain_graph, allow_unused=True)
            if all(grad is None or bool(torch.isfinite(grad).all()) for grad in candidate):
                raw = candidate
                chosen_scale = scale
                break
        if raw is None:
            raise FloatingPointError("all functional AMP gradient scales are nonfinite")
        gradients = {id(parameter): None if grad is None else grad.detach().float() / chosen_scale
                     for parameter, grad in zip(trainable, raw)}
    report = _group_summary(named_groups, gradients)
    report["functional_scale"] = chosen_scale
    return report, gradients


def _margin_stats(logits: torch.Tensor, mask: torch.Tensor, positive: bool) -> dict[str, Any] | None:
    values = logits[mask]
    if values.numel() == 0:
        return None
    values = values.detach().float()
    margin = values if positive else -values
    quantiles = torch.quantile(margin, torch.tensor([0.05, 0.50, 0.95], device=margin.device))
    return {
        "count": int(margin.numel()),
        "raw_logit_mean": float(values.mean()),
        "signed_margin_mean": float(margin.mean()),
        "signed_margin_min": float(margin.min()),
        "signed_margin_max": float(margin.max()),
        "signed_margin_q05_q50_q95": [float(item) for item in quantiles],
    }


def _observe_margins(model: nn.Module, refiner: nn.Module, image: torch.Tensor, target: torch.Tensor,
                     valid: torch.Tensor, h0: torch.Tensor, indices: torch.Tensor) -> dict[str, Any]:
    with preserve_runtime(model, refiner):
        with torch.inference_mode():
            model.eval();refiner.eval()
            base = _source_logits(model, image)
            overlap = torch.zeros_like(image)
            gate_outputs: list[torch.Tensor] = []
            residual_outputs: list[torch.Tensor] = []
            gh = refiner.gate.register_forward_hook(lambda _m, _i, out: gate_outputs.append(out.detach()))
            rh = refiner.residual.register_forward_hook(lambda _m, _i, out: residual_outputs.append(out.detach()))
            try:
                # Full 14-member inference, as used by the standard evaluator.
                states, _ = legacy.iterative_refine_o(image, base, refiner, 2, overlap)
            finally:
                gh.remove();rh.remove()
            final = states[-1]
            valid_members = valid.expand_as(target).bool()
            matched_negative = valid_members & ~target.bool() & h0.any(dim=1, keepdim=True)
            ordinary_positive = valid_members & target.bool() & ~h0
            predicted = final.sigmoid() >= 0.5
            residuals = [states[1] - states[0], states[2] - states[1]]
            def activation_summary(outputs: list[torch.Tensor], fn, is_gate: bool) -> dict[str, float]:
                if not outputs:return {'mean':0.,'near_saturation_fraction':0.}
                values = torch.cat([fn(x.float()).reshape(-1) for x in outputs])
                return {'mean':float(values.mean()),
                        'near_saturation_fraction':float(((values<.01)|(values>.99)).float().mean()) if is_gate
                                                   else float((values.abs()>.99).float().mean())}
            return {
                "protocol":"full14_two_step_FP32_fixed_training_case",
                "source": {"H0": _margin_stats(base, h0, True), "matched_negative": _margin_stats(base, matched_negative, False),
                           "other_true_positive": _margin_stats(base, ordinary_positive, True)},
                "step1": {"H0": _margin_stats(states[1], h0, True),
                           "matched_negative": _margin_stats(states[1], matched_negative, False)},
                "step2_final": {"H0": _margin_stats(final, h0, True), "matched_negative": _margin_stats(final, matched_negative, False),
                                "other_true_positive": _margin_stats(final, ordinary_positive, True)},
                "residuals": [{"H0": _margin_stats(delta, h0, True),
                               "absolute_max":float(delta.detach().float().abs().max()),
                               "near_limit_fraction":float((delta.detach().float().abs()>2.4).float().mean())}
                              for delta in residuals],
                "gate_sigmoid":activation_summary(gate_outputs, torch.sigmoid, True),
                "residual_tanh":activation_summary(residual_outputs, torch.tanh, False),
                "counts":{"H0_members":int(h0.sum()),"H0_source_above_minus5":int((h0 & (base.float()>=-5.)).sum()),
                          "H0_final_recovered":int((h0 & predicted).sum()),
                          "matched_negative_FP":int((matched_negative & predicted).sum()),
                          "other_true_positive_FN":int((ordinary_positive & ~predicted).sum())},
            }


def _tensor_snapshot(named_groups: Mapping[str, Sequence[tuple[str, torch.Tensor]]]) -> dict[str, list[torch.Tensor]]:
    return {group: [parameter.detach().clone() for _, parameter in named] for group, named in named_groups.items()}


def _update_report(named_groups: Mapping[str, Sequence[tuple[str, torch.Tensor]]], before: Mapping[str, Sequence[torch.Tensor]],
                   initial: Mapping[str, Sequence[torch.Tensor]], optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for group, named in named_groups.items():
        delta_sq = 0.0
        cumulative_sq = 0.0
        parameter_sq = 0.0
        changed = 0
        elements = 0
        moment_rows = []
        for position, (_, parameter) in enumerate(named):
            after = parameter.detach()
            previous = before[group][position]
            initial_value = initial[group][position]
            delta = after.float() - previous.float()
            cumulative = after.detach().cpu().float() - initial_value.float()
            delta_sq += float(delta.square().sum())
            cumulative_sq += float(cumulative.square().sum())
            parameter_sq += float(previous.detach().float().square().sum())
            changed += int(torch.count_nonzero(delta))
            elements += parameter.numel()
            moment_rows.append(_state_moments(optimizer.state.get(parameter, {}), parameter))
        if moment_rows:
            steps = [int(row["step"]) for row in moment_rows]
            avg_norm = sum(float(row["exp_avg_l2"]) ** 2 for row in moment_rows) ** 0.5
            avg_sq_norm = sum(float(row["exp_avg_sq_l2"]) ** 2 for row in moment_rows) ** 0.5
            avg_sq_sum = sum(float(row["exp_avg_sq_sum"]) for row in moment_rows)
            present = sum(bool(row["present"]) for row in moment_rows)
        else:
            steps, avg_norm, avg_sq_norm, avg_sq_sum, present = [], 0.0, 0.0, 0.0, 0
        report[group] = {
            "delta_theta_l2": delta_sq ** 0.5,
            "cumulative_offset_l2": cumulative_sq ** 0.5,
            "parameter_l2_before": parameter_sq ** 0.5,
            "changed_fraction": changed / elements if elements else 0.0,
            "elements": elements,
            "adam_state_present_parameters": present,
            "adam_step_min": min(steps) if steps else 0,
            "adam_step_max": max(steps) if steps else 0,
            "adam_exp_avg_l2": avg_norm,
            "adam_exp_avg_sq_l2": avg_sq_norm,
            "adam_exp_avg_sq_sum": avg_sq_sum,
        }
    return report


def _alignment_report(named_groups: Mapping[str, Sequence[tuple[str, torch.Tensor]]], q_gradients: Mapping[int, torch.Tensor | None],
                      before: Mapping[str, Sequence[torch.Tensor]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for group, named in named_groups.items():
        dot = 0.0
        for position, (_, parameter) in enumerate(named):
            gradient = q_gradients.get(id(parameter))
            if gradient is not None:
                delta = parameter.detach() - before[group][position]
                dot += float((gradient.detach().float() * delta.float()).sum())
        output[group] = dot
    return output


def _optimizer_groups(named_groups: Mapping[str, Sequence[tuple[str, torch.Tensor]]]) -> list[dict[str, Any]]:
    return [
        {"name": "refiner", "params": [parameter for _, parameter in named_groups["refiner"]], "lr": REFINER_LR, "weight_decay": WEIGHT_DECAY},
        {"name": "decoder_existing", "params": [parameter for _, parameter in named_groups["decoder_existing"]], "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY},
    ] + ([{"name": "decoder_D2", "params": [parameter for _, parameter in named_groups["decoder_D2"]], "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY}]
         if named_groups["decoder_D2"] and named_groups["decoder_D2"][0][1].requires_grad else [])


def _build_state(model: nn.Module, refiner: nn.Module, optimizer: torch.optim.Optimizer, scaler: Any,
                 arm: str, step: int, lock_hash: str, rng: Any) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "refiner": refiner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng,
        "arm": arm,
        "step": step,
        "lock_sha256": lock_hash,
    }


def _make_scaler(checkpoint: Mapping[str, Any]) -> Any:
    if "scaler" not in checkpoint or not isinstance(checkpoint["scaler"], Mapping):
        raise KeyError("C0 checkpoint must contain scaler state")
    scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=INITIAL_SCALE)
    state = copy.deepcopy(dict(checkpoint["scaler"]))
    state["scale"] = INITIAL_SCALE
    scaler.load_state_dict(state)
    if float(scaler.get_scale()) != INITIAL_SCALE:
        raise RuntimeError("shared AMP initial scale was not restored to 1024")
    return scaler


def _prepare_components(lock: Mapping[str, Any], lock_path: Path, arm: str, device: torch.device):
    r332 = _artifact_path(lock, lock_path, "r332", "r332_checkpoint", "model_source")
    overlap = _artifact_path(lock, lock_path, "overlap", "overlap_checkpoint", "image_only_overlap")
    checkpoint_path = _artifact_path(lock, lock_path, "checkpoint", "fresh100", "fresh100_checkpoint", "c0_checkpoint")
    overlap_sha = _expected_hash(lock, "overlap_sha256", "overlap_sha") or sha256_file(overlap)
    args = SimpleNamespace(r332=r332, overlap=overlap, overlap_sha=overlap_sha)
    payload, model, refiner, generator, backbone_mapping, refiner_mapping = legacy.load_components(args, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model" not in checkpoint or "refiner" not in checkpoint:
        raise ValueError("fresh100 checkpoint must contain model and refiner state")
    model.load_state_dict(checkpoint["model"], strict=True)
    refiner.load_state_dict(checkpoint["refiner"], strict=True)
    if int(checkpoint.get("updates", OLD_OPTIMIZER_STEP)) != OLD_OPTIMIZER_STEP:
        raise ValueError(f"C0 checkpoint must be fresh100, got updates={checkpoint.get('updates')}")

    # Freeze everything first, then recover exactly the existing 22 decoder parameters.
    legacy.set_joint_trainability(model, False)
    legacy.set_joint_trainability(model, True)
    existing_decoder = [(f"model.{name}", parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    d2_named = [(f"model.backbone.decoder.transpconvs.2.{name}", parameter)
                for name, parameter in model.backbone.decoder.transpconvs[2].named_parameters()]
    d2_named += [(f"model.backbone.decoder.stages.2.{name}", parameter)
                 for name, parameter in model.backbone.decoder.stages[2].named_parameters()]
    if len(existing_decoder) != 22:
        raise RuntimeError(f"expected 22 existing decoder parameters, got {len(existing_decoder)}")
    if len(d2_named) != EXPECTED_D2_PARAMS or sum(parameter.numel() for _, parameter in d2_named) != EXPECTED_D2_NUMEL:
        raise RuntimeError("logical D2 parameter count does not match the locked architecture")
    for _, parameter in d2_named:
        parameter.requires_grad_(arm in ("SRC_F1S0", "SRC_F1S1"))
    named_groups = {
        "refiner": [(f"refiner.{name}", parameter) for name, parameter in refiner.named_parameters()],
        "decoder_existing": existing_decoder,
        "decoder_D2": d2_named,
    }
    optimizer = torch.optim.AdamW(
        [{"params": [parameter for _, parameter in named_groups["refiner"]], "lr": REFINER_LR, "weight_decay": WEIGHT_DECAY},
         {"params": [parameter for _, parameter in named_groups["decoder_existing"]], "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY}],
        weight_decay=WEIGHT_DECAY,
    )
    old_optimizer = checkpoint.get("optimizer")
    if not isinstance(old_optimizer, Mapping):
        raise KeyError("fresh100 checkpoint must contain the restored AdamW state")
    old_groups = old_optimizer.get("param_groups", [])
    if len(old_groups) != 2 or [len(group.get("params", [])) for group in old_groups] != [40, 22]:
        raise ValueError("C0 AdamW state must contain 40 refiner and 22 existing-decoder parameters")
    optimizer.load_state_dict(copy.deepcopy(old_optimizer))
    if len(optimizer.param_groups) != 2:
        raise RuntimeError("restored AdamW group count changed")
    optimizer.param_groups[0]["lr"] = REFINER_LR
    optimizer.param_groups[1]["lr"] = DECODER_LR
    optimizer.param_groups[0]["name"] = "refiner"
    optimizer.param_groups[1]["name"] = "decoder_existing"
    for group_index,group_name in enumerate(("refiner", "decoder_existing")):
        names=named_groups[group_name];source_ids=old_groups[group_index]["params"]
        if len(names)!=len(source_ids):raise ValueError(f"optimizer group shape changed for {group_name}")
        for (parameter_name, parameter),source_id in zip(names,source_ids):
            restored=optimizer.state.get(parameter, {})
            original=old_optimizer["state"].get(source_id, {})
            if set(original)!={"step","exp_avg","exp_avg_sq"} or set(restored)!=set(original):
                raise ValueError(f"incomplete AdamW mapping for {parameter_name}")
            if _state_step(restored)!=OLD_OPTIMIZER_STEP or _state_step(original)!=OLD_OPTIMIZER_STEP:
                raise ValueError(f"restored AdamW state for {parameter_name} is not step 100")
            for key in ("exp_avg","exp_avg_sq"):
                if tuple(original[key].shape)!=tuple(parameter.shape) or not torch.equal(restored[key].detach().cpu(),original[key]):
                    raise ValueError(f"AdamW {key} name/order/shape mismatch for {parameter_name}")
    if arm in ("SRC_F1S0", "SRC_F1S1"):
        optimizer.add_param_group({"params": [parameter for _, parameter in d2_named], "lr": DECODER_LR,
                                   "weight_decay": WEIGHT_DECAY, "name": "decoder_D2"})
        for _, parameter in d2_named:
            optimizer.state[parameter] = {
                "step": torch.tensor(0.0, device=parameter.device),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
    scaler = _make_scaler(checkpoint)
    _configure_modes(model, refiner)
    return (payload, model, refiner, generator, optimizer, scaler, named_groups,
            backbone_mapping, refiner_mapping, checkpoint_path)


def _run_validation(lock: Mapping[str, Any], model: nn.Module, refiner: nn.Module, generator: nn.Module,
                    device: torch.device, lock_path: Path) -> dict[str, Any] | None:
    if lock.get("run_validation", True) is False:
        return {"status": "SKIPPED_BY_LOCK"}
    dataset_value = _first(lock, "validation_dataset", "dataset", default=None)
    if dataset_value is None:
        return {"status": "SKIPPED_NO_DATASET"}
    dataset = _as_path(dataset_value, lock_path.parent)
    if not dataset.is_dir():
        raise FileNotFoundError(f"validation dataset does not exist: {dataset}")
    with preserve_runtime(model, refiner):
        val = legacy.NativeWristDataset(dataset, "val", False)
        expected_cases = int(_first(lock, "validation_cases", "val_cases", default=69))
        if len(val) != expected_cases:
            raise ValueError(f"validation split size mismatch: expected val{expected_cases}, got {len(val)}")
        loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=0)
        pairs = legacy.top_training_pairs(dataset, 15)
        result = legacy.collect(model, refiner, generator, loader, device, False, pairs)
    return {"status": "COMPLETE", "protocol": "legacy.collect / R329 official", "result": result}


def _prepare_o_adapter(model: nn.Module, generator: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       named_groups: dict[str, list[tuple[str, torch.Tensor]]],
                       use_real_o: bool):
    """Attach the same new source-head capacity to both matched arms."""
    attachment = attach_o_source_head(model, generator, use_real_o=use_real_o)
    adapter_named = [(f"model.o_source_adapter.{name}", parameter)
                     for name, parameter in attachment.adapter.named_parameters()]
    if len(adapter_named) != 1 or sum(parameter.numel() for _, parameter in adapter_named) != 448:
        raise RuntimeError("source-head adapter must have exactly 448 zero-initialized weights")
    if any(bool(parameter.detach().count_nonzero()) for _, parameter in adapter_named):
        raise RuntimeError("source-head adapter must start at zero")
    named_groups["o_source_adapter"] = adapter_named
    optimizer.add_param_group({"params": [parameter for _, parameter in adapter_named],
                               "lr": DECODER_LR, "weight_decay": WEIGHT_DECAY,
                               "name": "o_source_adapter"})
    for _, parameter in adapter_named:
        optimizer.state[parameter] = {
            "step": torch.tensor(0.0, device=parameter.device),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    return attachment


def _main(args: argparse.Namespace) -> None:
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise TypeError("lock must be a JSON object")
    if args.arm not in ARMS:
        raise ValueError(args.arm)
    if lock.get("variant_id") != VARIANT_ID:
        raise ValueError("source-head runner requires the locked variant ID")
    if args.out.exists():
        raise FileExistsError(f"output must be isolated and absent: {args.out}")
    verified_hashes = _verify_lock_hashes(lock, lock_path)
    steps, manifest_path, manifest_hash = _load_schedule(lock, lock_path)
    if not torch.cuda.is_available():
        raise RuntimeError("SRC correctability runner requires CUDA")
    out = args.out.resolve()
    out.mkdir(parents=True)
    start = time.monotonic()
    max_seconds = float(_first(lock, "max_seconds", "GPU_seconds_hard_limit", "gpu_seconds_hard_limit",
                               "time_limit_seconds", default=7200.0))
    disk_floor = int(_first(lock, "disk_floor_bytes", default=4 * 1024**3))
    seed = int(_first(lock, "seed", default=4421))
    lock_hash = sha256_file(lock_path)
    device = torch.device("cuda")

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "arm": args.arm,
        "variant_id": VARIANT_ID,
        "lock": str(lock_path),
        "lock_sha256": lock_hash,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": verified_hashes["runner"],
        "schedule_manifest": str(manifest_path) if manifest_path else None,
        "schedule_sha256": manifest_hash,
        "schedule_steps": len(steps),
        "seed": seed,
        "test_used": False,
        "no_scheduler": True,
        "C0": {"updates": OLD_OPTIMIZER_STEP, "refiner_lr": REFINER_LR, "decoder_lr": DECODER_LR,
               "weight_decay": WEIGHT_DECAY, "amp_initial_scale": INITIAL_SCALE},
        "hashes_verified": verified_hashes,
    }
    write_json(out / "run_manifest.json", manifest)

    try:
        seed_all(seed)
        (payload, model, refiner, generator, optimizer, scaler, named_groups,
         backbone_mapping, refiner_mapping, checkpoint_path) = _prepare_components(lock, lock_path, args.arm, device)
        attachment = _prepare_o_adapter(model, generator, optimizer, named_groups,
                                        use_real_o=args.arm == "OHEAD_REAL")
        manifest.update({"backbone_mapping": backbone_mapping, "refiner_mapping": refiner_mapping,
                         "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                         "extra_block": "backbone.decoder.transpconvs.2 + backbone.decoder.stages.2",
                         "extra_block_params": EXPECTED_D2_PARAMS, "extra_block_numel": EXPECTED_D2_NUMEL,
                         "source_use_o": args.arm == "OHEAD_REAL", "refiner_use_o": False,
                         "source_head_adapter_params": 448,
                         "mode_protocol": "model.eval; decoder 2/3/4 + transpconvs 2/3/4 + seg_layers.4 train; refiner.train; o_source_adapter trainable"})
        _configure_modes(model, refiner)
        attachment.adapter.train()
        loaded_steps = _validate_schedule_against_c0(model, steps, device)
        initial = {group: [parameter.detach().cpu().clone() for _, parameter in named]
                   for group, named in named_groups.items()}
        detailed_steps = set(int(step) for step in _first(lock, "detailed_steps", default=[]))
        snapshot_steps = set(int(step) for step in _first(
            lock, "snapshot_steps", "checkpoint_steps", "expected_checkpoint_steps", default=sorted(detailed_steps)))
        checkpoint_dir = out / "checkpoints"
        if snapshot_steps:
            checkpoint_dir.mkdir()
        trace_path = out / "TRAINING_TRACE.jsonl"
        original_selector = legacy.choose_pair_instances
        successful_steps = 0
        cases: list[str] = []
        wall_rows: list[float] = []
        try:
            for row in loaded_steps:
                step = int(row["step"])
                if time.monotonic() - start > max_seconds:
                    raise TimeoutError("time budget exceeded before step")
                if shutil.disk_usage(out).free < disk_floor:
                    raise RuntimeError("disk floor reached before step")
                indices = torch.tensor(row["indices"], dtype=torch.long, device=device)
                image = row["image"].unsqueeze(0).to(device, non_blocking=True)
                target = row["mask"].float().unsqueeze(0).to(device, non_blocking=True)
                valid = row["valid"].bool().unsqueeze(0).to(device, non_blocking=True)
                h0 = row["h0"].bool().unsqueeze(0).to(device, non_blocking=True)
                # The selector is the only legacy hook: all four arms consume the same fixed indices.
                def fixed_selector(probability: torch.Tensor, target_value: torch.Tensor, k: int,
                                   fixed=indices) -> torch.Tensor:
                    if int(k) != 4:
                        raise ValueError(f"R332 loss requested {k} members, lock requires four")
                    return fixed.to(probability.device)

                legacy.choose_pair_instances = fixed_selector
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                with torch.cuda.amp.autocast(enabled=True):
                    captured_base: list[torch.Tensor] = []
                    capture_handle = model.register_forward_hook(lambda _module, _inputs, output: captured_base.append(output))
                    try:
                        native_loss, parts, details, _, selected, supervision = legacy.loss_for_batch(
                            model, refiner, generator, image, target, valid, False)
                    finally:
                        capture_handle.remove()
                    if len(captured_base) != 1 or not isinstance(captured_base[0], torch.Tensor):
                        raise RuntimeError("could not capture the native R332 source logits")
                    source_logits = captured_base[0]
                    if selected.tolist() != row["indices"]:
                        raise RuntimeError("fixed selector was not used by the native R332 loss")
                    source_softplus_sum = F.softplus(-source_logits.float())[h0].sum()
                    extra_loss = EXTRA_LAMBDA * source_softplus_sum / float(row["denominator_c0"])
                    total_loss = native_loss + extra_loss
                if not torch.isfinite(native_loss) or not torch.isfinite(extra_loss) or not torch.isfinite(total_loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}")
                flow_before = attachment.last_trace
                if flow_before is None or flow_before["use_real_o"] != (args.arm == "OHEAD_REAL"):
                    raise RuntimeError("source-head O flow trace missing or wrong arm")

                source_before = source_logits.detach().float().clone()
                q_gradients: dict[int, torch.Tensor | None] = {}
                per_loss_gradients: dict[str, Any] = {}
                if step in detailed_steps:
                    trainable = [parameter for named in named_groups.values() for _, parameter in named if parameter.requires_grad]
                    for name, value in (("native", native_loss), ("global", parts["global"]),
                                        ("monotonic", parts["monotonic"]), ("contraction", parts["contraction"]),
                                        ("source_aux", extra_loss)):
                        report, _ = _autograd_group_report(named_groups, value, trainable, retain_graph=True)
                        per_loss_gradients[name] = report
                    if bool(h0.any()):
                        q_h = F.softplus(-source_logits.float())[h0].mean()
                        _, q_gradients = _autograd_group_report(named_groups, q_h, trainable, retain_graph=True)
                    else:
                        q_h = None
                else:
                    q_h = None

                margins_before = None
                if step in detailed_steps:
                    margins_before = _observe_margins(model, refiner, image, target, valid, h0, indices)

                before = _tensor_snapshot(named_groups)
                grad_before: dict[str, Any] | None = None
                if step in detailed_steps:
                    grad_map = {id(parameter): parameter.grad for named in named_groups.values() for _, parameter in named}
                    grad_before = _group_summary(named_groups, grad_map)
                scale_before = float(scaler.get_scale())
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_map = {id(parameter): parameter.grad for named in named_groups.values() for _, parameter in named}
                grad_before_clip = _group_summary(named_groups, grad_map)
                if any(group["nonfinite_elements"] for group in grad_before_clip.values()):
                    write_json(out / "FAILED_NUMERIC.json", {"step":step,"case":row["case"],
                        "reason":"nonfinite_unscaled_gradient","nonfinite_elements_by_group":
                        {name:group["nonfinite_elements"] for name,group in grad_before_clip.items()},
                        "batch_sha256":row["batch_sha256"],"scale":float(scaler.get_scale())})
                    raise FloatingPointError("nonfinite unscaled gradient before optimizer step")
                refiner_clip_value = float(torch.nn.utils.clip_grad_norm_([parameter for _, parameter in named_groups["refiner"]], REFINER_CLIP, error_if_nonfinite=True))
                decoder_params = [parameter for _, parameter in named_groups["decoder_existing"]]
                decoder_params += [parameter for _, parameter in named_groups["o_source_adapter"]]
                if args.arm.startswith("SRC_F1"):
                    decoder_params += [parameter for _, parameter in named_groups["decoder_D2"]]
                decoder_clip_value = float(torch.nn.utils.clip_grad_norm_(decoder_params, DECODER_CLIP, error_if_nonfinite=True))
                grad_map_after = {id(parameter): parameter.grad for named in named_groups.values() for _, parameter in named}
                grad_after_clip = _group_summary(named_groups, grad_map_after)
                steps_before = {group: max((_state_step(optimizer.state.get(parameter, {})) for _, parameter in named), default=0)
                                for group, named in named_groups.items()}
                scaler.step(optimizer)
                scaler.update()
                steps_after = {group: max((_state_step(optimizer.state.get(parameter, {})) for _, parameter in named), default=0)
                               for group, named in named_groups.items()}
                amp_skipped = all(steps_after[group] == steps_before[group] for group in ("refiner", "decoder_existing"))
                successful_steps += int(not amp_skipped)
                if amp_skipped:
                    write_json(out / "FAILED_NUMERIC.json", {"step":step,"case":row["case"],
                        "reason":"AMP_skipped_optimizer_step","batch_sha256":row["batch_sha256"],
                        "scale_before":scale_before,"scale_after":float(scaler.get_scale())})
                    raise RuntimeError("AMP skipped a locked update; stop before unmatched continuation")
                alignment = _alignment_report(named_groups, q_gradients, before) if step in detailed_steps else None
                update = _update_report(named_groups, before, initial, optimizer)
                row_trace: dict[str, Any] = {
                    "step": step,
                    "case": row["case"],
                    "batch_sha256": row["batch_sha256"],
                    "fixed_indices": row["indices"],
                    "h0_exposure": {"total": int(h0.sum()), "per_member": [int(value) for value in h0[0].sum(dim=(-2, -1))],
                                     "valid_total": int((h0 & valid.expand_as(h0)).sum())},
                    "denominator_c0": float(row["denominator_c0"]),
                    "native_loss": float(native_loss.detach()),
                    "extra_source_loss": float(extra_loss.detach()),
                    "extra_source_raw_sum": float(source_softplus_sum.detach()),
                    "total_loss": float(total_loss.detach()),
                    "o_source_flow": {
                        "O_mean": flow_before["o"]["mean"],
                        "O_max": flow_before["o"]["max"],
                        "adapter_response_l2": flow_before["adapter_response"]["l2"],
                        "adapter_response_abs_mean": flow_before["adapter_response"]["abs_mean"],
                        "stage4_shape": flow_before["stage4_input"]["shape"],
                        "source_shape": flow_before["source_logits_after"]["shape"],
                    },
                    "native_parts": {key: float(value.detach()) for key, value in parts.items()},
                    "native_details": details,
                    "supervision": supervision,
                    "lr": {str(group.get("name", index)): float(group["lr"])
                           for index, group in enumerate(optimizer.param_groups)},
                    "clip": {"refiner_max_norm": REFINER_CLIP, "refiner_returned_norm": refiner_clip_value,
                             "decoder_max_norm": DECODER_CLIP, "decoder_returned_norm": decoder_clip_value,
                             "before": grad_before_clip, "after": grad_after_clip},
                    "AMP": {"scale_before": scale_before, "scale_after": float(scaler.get_scale()), "skipped": amp_skipped,
                            "finite": True},
                    "optimizer_steps": {"attempted": step, "successful": successful_steps, "before": steps_before, "after": steps_after},
                    "parameter_updates": update,
                    "seconds": time.monotonic() - tick,
                }
                if step in detailed_steps:
                    row_trace["per_loss_gradients"] = per_loss_gradients
                    row_trace["Q_H"] = {"mean_softplus_minus_source": None if q_h is None else float(q_h.detach()),
                                         "gradient_dot_actual_delta": alignment}
                    row_trace["margins_before"] = margins_before
                    row_trace["o_source_flow_before_step"] = flow_before
                # Margin after uses the post-step parameters; the observer restores modes/RNG/buffers.
                if step in detailed_steps:
                    row_trace["margins_after"] = _observe_margins(model, refiner, image, target, valid, h0, indices)
                    row_trace["o_source_flow_after_step"] = attachment.last_trace
                    row_trace["source_logits_before_stats"] = {
                        "H0_mean": float(source_before[h0].mean()) if bool(h0.any()) else None,
                        "negative_mean": float(source_before[(valid.expand_as(h0) & ~target.bool())].mean()) if bool((valid.expand_as(h0) & ~target.bool()).any()) else None,
                    }
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row_trace, allow_nan=False, default=json_default) + "\n")
                cases.append(str(row["case"]))
                if step in snapshot_steps:
                    torch.save(_build_state(model, refiner, optimizer, scaler, args.arm, step, lock_hash, _capture_rng()),
                               checkpoint_dir / f"step_{step:05d}.pth")
                torch.cuda.synchronize()
                wall_rows.append(time.monotonic() - tick)
                print(json.dumps({"arm": args.arm, "step": step, "case": row["case"], "native_loss": row_trace["native_loss"],
                                  "extra_source_loss": row_trace["extra_source_loss"], "successful_optimizer_steps": successful_steps}), flush=True)
        finally:
            legacy.choose_pair_instances = original_selector

        validation = _run_validation(lock, model, refiner, generator, device, lock_path)
        if validation is not None:
            write_json(out / "FINAL_VALIDATION.json", validation)
        final_path = out / "diagnostic_final.pth"
        torch.save(_build_state(model, refiner, optimizer, scaler, args.arm, len(loaded_steps), lock_hash, _capture_rng()), final_path)
        manifest.update({"status": "COMPLETE", "optimizer_steps": successful_steps, "attempted_steps": len(loaded_steps),
                         "cases": cases, "case_order_sha256": hashlib.sha256(json.dumps(cases).encode()).hexdigest(),
                         "seconds": time.monotonic() - start, "step_seconds": wall_rows,
                         "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "final_checkpoint": str(final_path),
                         "final_checkpoint_sha256": sha256_file(final_path), "test_used": False,
                         "validation_status": None if validation is None else validation.get("status")})
        write_json(out / "run_manifest.json", manifest)
    except Exception as error:
        manifest.update({"status": "FAILED", "error": f"{type(error).__name__}: {error}",
                         "seconds": time.monotonic() - start, "test_used": False})
        write_json(out / "run_manifest.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="locked JSON protocol")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new isolated output directory")
    _main(parser.parse_args())


if __name__ == "__main__":
    main()
