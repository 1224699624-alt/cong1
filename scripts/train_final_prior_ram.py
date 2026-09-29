#!/usr/bin/env python3
"""Native-start RAM final-prior runner.

This runner is intentionally self contained.  All four nnU-Net arms start
from the same R325 native checkpoint and use the same optimizer, schedule,
seed, train/validation split, and validation selection rule:

``A`` native continue, ``B`` seam/input prior, ``C`` overlap/output prior,
and ``D`` both priors.

Only RAM ``train`` and ``val`` are opened.  The official metric implementation
is the existing R329 evaluator.  The optional R332 checkpoint is used only to
initialize the output refiner with ``identity=True``; its segmentation model
is never used as the final-run backbone.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_r329_ram_official_metrics_val import official_metrics
from r349_parameter_gradient_gate import assign_gradients, route_parameter_gradients
from r351_data import AlignedRamDataset
from r351_dual_prior import (
    ImageOverlapPrior,
    OutputCompletionPrior,
    SeamInputAdapter,
    local_terms,
)
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior, top_training_pairs
from train_r327_ram_native_instance_completion_prior import seed_all
from train_r351_ram_dual_prior import collect_native as r351_collect_native

try:  # The sibling TSRS final runner supplies this shared thin wrapper.
    from final_prior_plugin import BackboneAdapter, StructuralPriorPlugin
except ImportError:  # pragma: no cover - compatibility for staged checkouts.
    BackboneAdapter = None  # type: ignore[assignment,misc]
    StructuralPriorPlugin = None  # type: ignore[assignment,misc]


EXPERIMENT = "FINAL_PRIOR_RAM_NATIVE_START"
EXPECTED_NATIVE_SHA256 = "88f8f9168d969ee21238aeb77bbb701218a5fef31285f1fc2ceb71e3aa1076c9"
EXPECTED_CASE_COUNTS = {"train": 425, "val": 69}
VARIANTS = ("A", "B", "C", "D")
DEFAULT_SEED = 3511
DEFAULT_EPOCHS = 2
DEFAULT_ITERS_PER_EPOCH = 100
DEFAULT_BACKBONE_LR = 2e-6
DEFAULT_ADAPTER_LR = 2e-5
DEFAULT_STEPS = 2
DEFAULT_MAX_AUX_RATIO = 0.15
DEFAULT_AUXILIARY_WEIGHTS = {
    "separation": 0.035,
    "overlap": 0.035,
    "preserve": 0.10,
}


class RatioPolyLRScheduler:
    """Matched polynomial schedule that preserves backbone/prior LR ratio."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_lrs: list[float], max_steps: int,
                 exponent: float = 0.9) -> None:
        if len(base_lrs) != len(optimizer.param_groups) or max_steps < 1:
            raise ValueError("invalid scheduler groups or max_steps")
        self.optimizer = optimizer
        self.base_lrs = [float(value) for value in base_lrs]
        self.max_steps = int(max_steps)
        self.exponent = float(exponent)
        self.last_step = 0
        self.step(0)

    def step(self, current_step: int | None = None) -> None:
        self.last_step = self.last_step + 1 if current_step is None else int(current_step)
        multiplier = max(0.0, 1.0 - self.last_step / float(self.max_steps)) ** self.exponent
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base * multiplier

    def state_dict(self) -> dict[str, Any]:
        return {"base_lrs": self.base_lrs, "max_steps": self.max_steps,
                "exponent": self.exponent, "last_step": self.last_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.max_steps = int(state["max_steps"])
        self.exponent = float(state["exponent"])
        self.step(int(state.get("last_step", 0)))


def sha256(path: Path) -> str:
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


def _state(payload: dict[str, Any], keys: Iterable[str], path: Path) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    raise RuntimeError(f"Checkpoint {path} has no state mapping under {tuple(keys)!r}")


def _load_payload(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    return payload


def _source_record(path: Path, *, expected: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    actual = sha256(resolved)
    record: dict[str, Any] = {"path": str(resolved), "sha256": actual}
    if expected is not None:
        record["expected_sha256"] = expected
        if actual.lower() != expected.lower():
            raise RuntimeError({"checkpoint_sha256_mismatch": record})
    return record


def _assert_fresh_output(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty final-prior output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _case_name(value: Any, batch_size: int) -> list[str]:
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value]
    elif isinstance(value, np.ndarray):
        names = [str(item) for item in value.tolist()]
    elif batch_size == 1:
        names = [str(value)]
    else:
        raise RuntimeError(f"Cannot unpack case field for batch size {batch_size}: {type(value)!r}")
    if len(names) != batch_size:
        raise RuntimeError({"case_names": names, "batch_size": batch_size})
    return names


def _prior_path(prior_root: Path, split: str, case: str) -> Path | None:
    npy = prior_root / split / f"{case}.npy"
    png = prior_root / split / f"{case}.png"
    if npy.exists():
        return npy
    if png.exists():
        return png
    return None


def audit_data_contract(dataset_root: Path, prior_root: Path) -> dict[str, Any]:
    """Validate canonical train/val sizes, disjointness, and all prior maps."""
    train = AlignedRamDataset(dataset_root, prior_root, "train", augment=False)
    val = AlignedRamDataset(dataset_root, prior_root, "val", augment=False)
    train_cases = [path.stem for path in train.mask_files]
    val_cases = [path.stem for path in val.mask_files]
    if len(train_cases) != len(set(train_cases)) or len(val_cases) != len(set(val_cases)):
        raise RuntimeError("Duplicate case ids in RAM train or val split")
    overlap = sorted(set(train_cases).intersection(val_cases))
    if overlap:
        raise RuntimeError({"train_val_identity_overlap": overlap[:10], "count": len(overlap)})
    counts = {"train": len(train_cases), "val": len(val_cases)}
    if counts != EXPECTED_CASE_COUNTS:
        raise RuntimeError({"expected_case_counts": EXPECTED_CASE_COUNTS, "actual_case_counts": counts})

    coverage: dict[str, dict[str, Any]] = {}
    for split, cases in (("train", train_cases), ("val", val_cases)):
        missing = [case for case in cases if _prior_path(prior_root, split, case) is None]
        if missing:
            raise RuntimeError({"missing_prior_maps": {"split": split, "count": len(missing), "cases": missing[:20]}})
        coverage[split] = {
            "dataset_cases": len(cases),
            "prior_cases": len(cases),
            "missing": [],
            "complete": True,
        }
    return {
        "dataset_root": str(dataset_root.resolve()),
        "prior_root": str(prior_root.resolve()),
        "case_counts": counts,
        "train_val_disjoint": True,
        "prior_coverage": coverage,
    }


def _batch_tensors(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    image = batch["image"].to(device, non_blocking=True)
    target = batch["mask"].to(device, non_blocking=True).float()
    seam = batch["seam"].to(device, non_blocking=True).float()
    valid = batch["valid"].to(device, non_blocking=True).float()
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    if image.ndim != 4 or target.ndim != 4 or seam.ndim != 4 or valid.ndim != 4:
        raise RuntimeError({"image": tuple(image.shape), "mask": tuple(target.shape),
                            "seam": tuple(seam.shape), "valid": tuple(valid.shape)})
    if target.shape[0] != image.shape[0] or seam.shape != valid.shape:
        raise RuntimeError({"image": tuple(image.shape), "mask": tuple(target.shape),
                            "seam": tuple(seam.shape), "valid": tuple(valid.shape)})
    return image, target, seam, valid


class _FallbackPlugin(torch.nn.Module):
    """Small compatibility wrapper used only before the shared plugin lands."""

    def __init__(self, overlap_prior: ImageOverlapPrior, input_adapter: SeamInputAdapter,
                 output_refiner: OutputCompletionPrior, enable_input: bool,
                 enable_output: bool, output_residual_scale: float = 1.0) -> None:
        super().__init__()
        self.overlap_prior = overlap_prior
        self.input_adapter = input_adapter
        self.output_refiner = output_refiner
        self.enable_input = bool(enable_input)
        self.enable_output = bool(enable_output)
        self.output_residual_scale = float(output_residual_scale)

    def overlap_probability(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.overlap_prior(image, seam)).detach()

    def adapt_input(self, image: torch.Tensor, seam: torch.Tensor,
                    overlap: torch.Tensor | None = None) -> torch.Tensor:
        if not self.enable_input:
            return image
        overlap = self.overlap_probability(image, seam) if overlap is None else overlap
        return self.input_adapter(image, seam, overlap)

    def refine_output(self, image: torch.Tensor, logits: torch.Tensor, seam: torch.Tensor,
                      overlap: torch.Tensor | None = None, steps: int = 1) -> torch.Tensor:
        if not self.enable_output:
            return logits
        overlap = self.overlap_probability(image, seam) if overlap is None else overlap
        refined = self.output_refiner.refine(image, logits, seam, overlap, steps=steps)
        return logits + self.output_residual_scale * (refined - logits)


class _RamBackboneAdapter(torch.nn.Module):
    """Expose RAM nnU-Net through the shared ``forward_logits`` contract."""

    def __init__(self, model: NnUNetMultiLabelPrior) -> None:
        super().__init__()
        self.model = model

    def forward_logits(self, image_or_input: torch.Tensor) -> torch.Tensor:
        return self.model(image_or_input, False)[0]


def _build_plugin(overlap_prior: ImageOverlapPrior, input_adapter: SeamInputAdapter,
                  output_refiner: OutputCompletionPrior, enable_input: bool,
                  enable_output: bool) -> torch.nn.Module:
    if StructuralPriorPlugin is None:
        return _FallbackPlugin(overlap_prior, input_adapter, output_refiner,
                               enable_input, enable_output)

    # The project contract is keyword based.  Two positional fallbacks keep
    # this runner usable during the short period in which sibling branches are
    # importing an older revision of the shared wrapper.
    candidates = (
        lambda: StructuralPriorPlugin(
            seam_prior=None,
            overlap_prior=overlap_prior,
            input_adapter=input_adapter,
            output_refiner=output_refiner,
            enable_input=enable_input,
            enable_output=enable_output,
            output_residual_scale=1.0,
        ),
        lambda: StructuralPriorPlugin(
            overlap_prior, input_adapter, output_refiner,
            enable_input=enable_input, enable_output=enable_output,
            output_residual_scale=1.0,
        ),
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            plugin = candidate()
            if not isinstance(plugin, torch.nn.Module):
                raise TypeError("StructuralPriorPlugin must be an nn.Module")
            return plugin
        except (TypeError, ValueError) as exc:
            errors.append(repr(exc))
    raise RuntimeError({"cannot_construct_shared_final_prior_plugin": errors})


def _call_plugin_method(method: Any, image: torch.Tensor, logits_or_seam: torch.Tensor,
                        seam_or_overlap: torch.Tensor | None = None,
                        overlap: torch.Tensor | None = None, *, steps: int | None = None) -> torch.Tensor:
    """Call both revisions of the thin shared plugin without passing targets."""
    try:
        signature = inspect.signature(method)
        parameters = list(signature.parameters.values())
    except (TypeError, ValueError):
        parameters = []
    names = {parameter.name for parameter in parameters}
    positional = [parameter for parameter in parameters
                  if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)]
    args: list[Any] = [image, logits_or_seam]
    if seam_or_overlap is not None and len(positional) >= 3:
        args.append(seam_or_overlap)
    kwargs: dict[str, Any] = {}
    if overlap is not None:
        if "overlap" in names:
            kwargs["overlap"] = overlap
        elif "overlap_probability" in names:
            kwargs["overlap_probability"] = overlap
    if steps is not None and "steps" in names:
        kwargs["steps"] = steps
    return method(*args, **kwargs)


def _plugin_overlap(plugin: torch.nn.Module, overlap_prior: ImageOverlapPrior,
                    image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    method = getattr(plugin, "overlap_probability", None)
    if method is not None:
        with torch.no_grad():
            return method(image, seam).detach()
    with torch.no_grad():
        return torch.sigmoid(overlap_prior(image, seam)).detach()


def forward_variant(model: NnUNetMultiLabelPrior, plugin: torch.nn.Module,
                    overlap_prior: ImageOverlapPrior, image: torch.Tensor,
                    seam: torch.Tensor, *, enable_input: bool,
                    enable_output: bool, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward path with no target/ground-truth argument by construction."""
    overlap = _plugin_overlap(plugin, overlap_prior, image, seam)
    adapted = image
    if enable_input:
        adapted = _call_plugin_method(plugin.adapt_input, image, seam, overlap=overlap)
    logits = model(adapted, False)[0]
    refined = logits
    if enable_output:
        refined = _call_plugin_method(
            plugin.refine_output, image, logits, seam, overlap=overlap, steps=steps
        )
    return refined, overlap, adapted


def _flatten_official(metrics: dict[str, Any]) -> dict[str, float | None]:
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
        "overall_ravd": overall.get("ravd"),
        "overlap_dsc": overlap.get("dsc"),
        "overlap_nsd_2px": overlap.get("nsd_2px"),
        "overlap_msd_px": overlap.get("msd_px"),
        "overlap_ravd": overlap.get("ravd"),
        "pair_dsc": pair.get("dsc"),
        "pair_nsd_2px": pair.get("nsd_2px"),
        "pair_msd_px": pair.get("msd_px"),
        "pair_ravd": pair.get("ravd"),
    }


@torch.inference_mode()
def collect_adapted(model: NnUNetMultiLabelPrior, plugin: torch.nn.Module,
                    overlap_prior: ImageOverlapPrior, loader: DataLoader,
                    device: torch.device, pairs: list[tuple[int, int]],
                    enable_input: bool, enable_output: bool, steps: int) -> dict[str, Any]:
    """Collect full validation predictions and score with the official evaluator."""
    model.eval(); plugin.eval(); overlap_prior.eval()
    cases: dict[str, dict[str, np.ndarray]] = {}
    for batch in loader:
        image, target, seam, _valid = _batch_tensors(batch, device)
        logits, _overlap, _adapted = forward_variant(
            model, plugin, overlap_prior, image, seam,
            enable_input=enable_input, enable_output=enable_output, steps=steps,
        )
        prediction = (torch.sigmoid(logits) >= 0.5).detach().cpu().numpy().astype(bool)
        target_np = (target > 0.5).detach().cpu().numpy().astype(bool)
        names = _case_name(batch["case"], int(prediction.shape[0]))
        for index, name in enumerate(names):
            if name in cases:
                raise RuntimeError(f"Duplicate validation case: {name}")
            cases[name] = {"pred": prediction[index], "target": target_np[index]}
    if len(cases) != EXPECTED_CASE_COUNTS["val"]:
        raise RuntimeError({"expected_val_cases": 69, "actual_val_cases": len(cases)})
    official = official_metrics(cases, pairs)
    return {"official": official, "flat": _flatten_official(official), "cases": len(cases)}


def _load_native_model(native_checkpoint: Path, device: torch.device) -> tuple[NnUNetMultiLabelPrior, dict[str, Any]]:
    source = _source_record(native_checkpoint, expected=EXPECTED_NATIVE_SHA256)
    payload = _load_payload(native_checkpoint, device)
    source["checkpoint_epoch"] = payload.get("row", {}).get("epoch", payload.get("epoch"))
    model = NnUNetMultiLabelPrior().to(device)
    # This is deliberately strict: a different architecture or source model
    # must fail before any output directory is populated.
    model.load_state_dict(_state(payload, ("model",), native_checkpoint), strict=True)
    return model, source


def _build_prior_state(args: argparse.Namespace, device: torch.device,
                       enable_input: bool, enable_output: bool) -> tuple[torch.nn.Module, ImageOverlapPrior,
                                                                         SeamInputAdapter, OutputCompletionPrior,
                                                                         dict[str, Any]]:
    overlap_payload = _load_payload(args.overlap_checkpoint, device)
    overlap_source = _source_record(args.overlap_checkpoint)
    overlap_prior = ImageOverlapPrior().to(device)
    overlap_prior.load_state_dict(
        _state(overlap_payload, ("model", "overlap_prior", "prior"), args.overlap_checkpoint),
        strict=True,
    )
    overlap_prior.requires_grad_(False)
    overlap_prior.eval()

    input_adapter = SeamInputAdapter().to(device)
    output_refiner = OutputCompletionPrior().to(device)
    r332_source: dict[str, Any] | None = None
    if enable_output:
        r332_payload = _load_payload(args.r332_checkpoint, device)
        r332_source = _source_record(args.r332_checkpoint)
        # This copies only the historical output-prior weights and zeros the
        # final residual, so the native-start forward remains an identity.
        output_refiner.load_mature(
            _state(r332_payload, ("refiner",), args.r332_checkpoint), identity=True
        )
    plugin = _build_plugin(overlap_prior, input_adapter, output_refiner,
                           enable_input, enable_output).to(device)
    plugin.eval()
    frozen = {
        "overlap_prior": overlap_source,
        "r332_refiner": r332_source,
        "native_backbone": "loaded separately from the required R325 native source",
    }
    return plugin, overlap_prior, input_adapter, output_refiner, frozen


def _set_trainability(model: torch.nn.Module, plugin: torch.nn.Module,
                      overlap_prior: torch.nn.Module, input_adapter: torch.nn.Module,
                      output_refiner: torch.nn.Module, *, mode: str,
                      enable_input: bool, enable_output: bool) -> list[torch.nn.Parameter]:
    if mode not in ("adapter_only", "joint_finetune"):
        raise ValueError(mode)
    model.requires_grad_(mode == "joint_finetune")
    overlap_prior.requires_grad_(False)
    input_adapter.requires_grad_(mode == "adapter_only" and enable_input or enable_input)
    output_refiner.requires_grad_(mode == "adapter_only" and enable_output or enable_output)
    if mode == "joint_finetune":
        # Joint runs update the native backbone and enabled prior branches.
        model.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameters.extend(parameter for parameter in input_adapter.parameters() if parameter.requires_grad)
    parameters.extend(parameter for parameter in output_refiner.parameters() if parameter.requires_grad)
    if not parameters:
        raise RuntimeError("No trainable parameters; adapter_only cannot represent native_continue A")
    return parameters


def _teacher_probability(teacher: NnUNetMultiLabelPrior, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = teacher(image, False)[0]
        return torch.sigmoid(logits).detach()


def _loss_and_state(model: NnUNetMultiLabelPrior, plugin: torch.nn.Module,
                    overlap_prior: ImageOverlapPrior, teacher: NnUNetMultiLabelPrior,
                    input_adapter: SeamInputAdapter, output_refiner: OutputCompletionPrior,
                    batch: dict[str, Any], device: torch.device, *, enable_input: bool,
                    enable_output: bool, steps: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor]:
    image, target, seam, valid = _batch_tensors(batch, device)
    logits, overlap, _adapted = forward_variant(
        model, plugin, overlap_prior, image, seam,
        enable_input=enable_input, enable_output=enable_output, steps=steps,
    )
    teacher_probability = _teacher_probability(teacher, image)
    terms, state = local_terms(
        logits, target, teacher_probability, seam, overlap, valid,
        membership_observed=True,
    )
    return terms, state, logits


def _train_epoch(model: NnUNetMultiLabelPrior, plugin: torch.nn.Module,
                 overlap_prior: ImageOverlapPrior, teacher: NnUNetMultiLabelPrior,
                 input_adapter: SeamInputAdapter, output_refiner: OutputCompletionPrior,
                 loader: DataLoader, optimizer: torch.optim.Optimizer,
                 scheduler: RatioPolyLRScheduler, device: torch.device,
                 *, enable_input: bool, enable_output: bool, steps: int,
                 iterations: int, max_aux_ratio: float) -> dict[str, Any]:
    model.train(); plugin.train(); overlap_prior.eval(); teacher.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(parameter for parameter in input_adapter.parameters() if parameter.requires_grad)
    trainable.extend(parameter for parameter in output_refiner.parameters() if parameter.requires_grad)
    totals = {name: 0.0 for name in ("base", "separation", "overlap", "preserve")}
    routing: list[dict[str, Any]] = []
    iterator = iter(loader)
    batches = 0
    started = time.time()
    for _ in range(iterations):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        terms, state, _logits = _loss_and_state(
            model, plugin, overlap_prior, teacher, input_adapter, output_refiner,
            batch, device, enable_input=enable_input, enable_output=enable_output, steps=steps,
        )
        if not all(torch.isfinite(value).item() for value in terms.values()):
            raise RuntimeError({"non_finite_loss": {name: float(value.detach()) for name, value in terms.items()}})
        optimizer.zero_grad(set_to_none=True)
        auxiliaries = [
            ("separation", terms["separation"], DEFAULT_AUXILIARY_WEIGHTS["separation"] if enable_input else 0.0),
            ("overlap", terms["overlap"], DEFAULT_AUXILIARY_WEIGHTS["overlap"] if enable_output else 0.0),
            ("preserve", terms["preserve"], DEFAULT_AUXILIARY_WEIGHTS["preserve"]),
        ]
        gradients, rows = route_parameter_gradients(
            trainable, terms["base"], auxiliaries, max_aux_ratio=max_aux_ratio,
        )
        assign_gradients(trainable, gradients)
        torch.nn.utils.clip_grad_norm_(trainable, 8.0)
        optimizer.step()
        scheduler.step()
        batches += 1
        routing.extend(vars(row) for row in rows)
        for name, value in terms.items():
            totals[name] += float(value.detach())
    if batches != iterations:
        raise RuntimeError({"expected_iterations": iterations, "actual": batches})
    return {
        "batches": batches,
        "seconds": time.time() - started,
        "losses": {name: value / batches for name, value in totals.items()},
        "last_state": state,
        "routing": routing,
    }


def _identity_audit(model: NnUNetMultiLabelPrior, plugin: torch.nn.Module,
                    overlap_prior: ImageOverlapPrior, loader: DataLoader,
                    device: torch.device, *, enable_input: bool, enable_output: bool,
                    steps: int, max_batches: int = 2) -> dict[str, Any]:
    model.eval(); plugin.eval(); overlap_prior.eval()
    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        image, _target, seam, _valid = _batch_tensors(batch, device)
        with torch.no_grad():
            native = model(image, False)[0]
            final, _overlap, adapted = forward_variant(
                model, plugin, overlap_prior, image, seam,
                enable_input=enable_input, enable_output=enable_output, steps=steps,
            )
        rows.append({
            "cases": _case_name(batch["case"], int(image.shape[0])),
            "max_logit_delta": float((final - native).abs().max()),
            "max_input_delta": float((adapted - image).abs().max()),
            "finite": bool(torch.isfinite(final).all().item()),
        })
    passed = bool(rows) and all(row["finite"] and row["max_logit_delta"] <= 1e-5
                                and row["max_input_delta"] <= 1e-6 for row in rows)
    return {"batches": len(rows), "rows": rows, "identity_tolerance": 1e-5, "passed": passed}


def _backward_audit(model: NnUNetMultiLabelPrior, plugin: torch.nn.Module,
                    overlap_prior: ImageOverlapPrior, teacher: NnUNetMultiLabelPrior,
                    input_adapter: SeamInputAdapter, output_refiner: OutputCompletionPrior,
                    loader: DataLoader, device: torch.device, *, enable_input: bool,
                    enable_output: bool, steps: int, max_aux_ratio: float,
                    max_batches: int = 2) -> dict[str, Any]:
    model.train(); plugin.train(); overlap_prior.eval(); teacher.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(parameter for parameter in input_adapter.parameters() if parameter.requires_grad)
    trainable.extend(parameter for parameter in output_refiner.parameters() if parameter.requires_grad)
    rows: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        terms, _state, _logits = _loss_and_state(
            model, plugin, overlap_prior, teacher, input_adapter, output_refiner,
            batch, device, enable_input=enable_input, enable_output=enable_output, steps=steps,
        )
        gradients, routed = route_parameter_gradients(
            trainable, terms["base"], [
                ("separation", terms["separation"], DEFAULT_AUXILIARY_WEIGHTS["separation"] if enable_input else 0.0),
                ("overlap", terms["overlap"], DEFAULT_AUXILIARY_WEIGHTS["overlap"] if enable_output else 0.0),
                ("preserve", terms["preserve"], DEFAULT_AUXILIARY_WEIGHTS["preserve"]),
            ], max_aux_ratio=max_aux_ratio,
        )
        assign_gradients(trainable, gradients)
        grad_sum = float(sum(parameter.grad.abs().sum() for parameter in trainable if parameter.grad is not None))
        rows.append({
            "cases": _case_name(batch["case"], int(_batch_tensors(batch, device)[0].shape[0])),
            "finite": all(torch.isfinite(value).item() for value in terms.values()),
            "gradient_sum": grad_sum,
            "routing": [vars(row) for row in routed],
        })
        for parameter in trainable:
            parameter.grad = None
    passed = bool(rows) and all(row["finite"] and row["gradient_sum"] > 0 for row in rows)
    return {"batches": len(rows), "rows": rows, "passed": passed}


def _cpu_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _save_best(path: Path, model: NnUNetMultiLabelPrior, input_adapter: SeamInputAdapter,
               output_refiner: OutputCompletionPrior, *, epoch: int, variant: str,
               training_mode: str, source: dict[str, Any], row: dict[str, Any]) -> str:
    payload = {
        "model": _cpu_state(model),
        "input_adapter": _cpu_state(input_adapter),
        "output_refiner": _cpu_state(output_refiner),
        "epoch": int(epoch),
        "variant": variant,
        "training_mode": training_mode,
        "source_checkpoint": source,
        "selection_metric": "val.overall_dsc",
        "row": row,
    }
    torch.save(payload, path)
    return sha256(path)


def _selection_value(report: dict[str, Any]) -> float:
    value = report["flat"].get("overall_dsc")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _make_loaders(dataset_root: Path, prior_root: Path, *, audit: bool, seed: int,
                  num_workers: int) -> tuple[AlignedRamDataset, AlignedRamDataset, DataLoader, DataLoader, dict[str, Any]]:
    contract = audit_data_contract(dataset_root, prior_root)
    train_set = AlignedRamDataset(dataset_root, prior_root, "train", augment=False)
    val_set = AlignedRamDataset(dataset_root, prior_root, "val", augment=False)
    if audit:
        train_set.mask_files = train_set.mask_files[:2]
        val_set.mask_files = val_set.mask_files[:2]
    generator = torch.Generator().manual_seed(seed)
    worker_count = 0 if audit else max(0, num_workers)
    options = {"batch_size": 1, "pin_memory": torch.cuda.is_available(), "num_workers": worker_count}
    if worker_count:
        options["persistent_workers"] = True
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **options)
    val_loader = DataLoader(val_set, shuffle=False, **options)
    return train_set, val_set, train_loader, val_loader, contract


def run_baseline(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    _assert_fresh_output(args.output)
    _train, val_set, _train_loader, val_loader, contract = _make_loaders(
        args.dataset_root, args.prior_root, audit=False, seed=args.seed, num_workers=args.num_workers,
    )
    model, source = _load_native_model(args.native_checkpoint, device)
    pairs = top_training_pairs(args.dataset_root, 15)
    # This is the exact R351 collector, over all 69 validation cases.
    report = r351_collect_native(model, val_loader, device, pairs, args.seam_threshold)
    result = {
        "experiment": EXPERIMENT,
        "status": "baseline_only",
        "variant": "A",
        "split": "RAM train/validation only",
        "test_used": False,
        "test_used_for_selection": False,
        "git_commit": "737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353",
        "validation_case_count": len(val_set),
        "source_checkpoint": source,
        "native_checkpoint_sha256": source["sha256"],
        "selection_metric": "val.overall_dsc",
        "data_contract": contract,
        "official": report["official"],
        "flat": report["flat"],
        "config": _config(args),
    }
    _write_json(args.output / "baseline_result.json", result)
    _write_json(args.output / "run_manifest.json", result)
    print(json.dumps({"status": result["status"], "overall_dsc": report["flat"]["overall_dsc"],
                      "sha256": source["sha256"]}, default=_json_default), flush=True)
    return result


def run_audit(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    _assert_fresh_output(args.output)
    _train, _val, train_loader, val_loader, contract = _make_loaders(
        args.dataset_root, args.prior_root, audit=True, seed=args.seed, num_workers=0,
    )
    model, source = _load_native_model(args.native_checkpoint, device)
    enable_input, enable_output = args.variant in ("B", "D"), args.variant in ("C", "D")
    plugin, overlap_prior, input_adapter, output_refiner, prior_sources = _build_prior_state(
        args, device, enable_input, enable_output,
    )
    teacher, _teacher_source = _load_native_model(args.native_checkpoint, device)
    _set_trainability(model, plugin, overlap_prior, input_adapter, output_refiner,
                      mode=args.training_mode, enable_input=enable_input, enable_output=enable_output)
    identity = _identity_audit(model, plugin, overlap_prior, val_loader, device,
                               enable_input=enable_input, enable_output=enable_output, steps=args.steps)
    backward = _backward_audit(model, plugin, overlap_prior, teacher, input_adapter, output_refiner,
                               train_loader, device, enable_input=enable_input,
                               enable_output=enable_output, steps=args.steps,
                               max_aux_ratio=args.max_aux_ratio)
    result = {
        "experiment": EXPERIMENT,
        "status": "audit",
        "variant": args.variant,
        "training_mode": args.training_mode,
        "split": "RAM train/validation only",
        "test_used": False,
        "test_used_for_selection": False,
        "git_commit": "737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353",
        "source_checkpoint": source,
        "prior_sources": prior_sources,
        "data_contract": contract,
        "identity": identity,
        "backward": backward,
        "passed": bool(identity["passed"] and backward["passed"]),
        "config": _config(args),
    }
    _write_json(args.output / "audit_result.json", result)
    _write_json(args.output / "run_manifest.json", result)
    print(json.dumps({"status": result["status"], "passed": result["passed"]}), flush=True)
    if not result["passed"]:
        raise RuntimeError("final RAM audit failed")
    return result


def run_training(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.training_mode == "adapter_only" and args.variant == "A":
        raise ValueError("variant A native_continue requires joint_finetune; adapter_only A is not a continue control")
    _assert_fresh_output(args.output)
    train_set, val_set, train_loader, val_loader, contract = _make_loaders(
        args.dataset_root, args.prior_root, audit=False, seed=args.seed, num_workers=args.num_workers,
    )
    model, source = _load_native_model(args.native_checkpoint, device)
    teacher, _teacher_source = _load_native_model(args.native_checkpoint, device)
    enable_input, enable_output = args.variant in ("B", "D"), args.variant in ("C", "D")
    plugin, overlap_prior, input_adapter, output_refiner, prior_sources = _build_prior_state(
        args, device, enable_input, enable_output,
    )
    trainable = _set_trainability(
        model, plugin, overlap_prior, input_adapter, output_refiner,
        mode=args.training_mode, enable_input=enable_input, enable_output=enable_output,
    )
    model.eval(); teacher.eval(); overlap_prior.eval()
    parameter_groups: list[dict[str, Any]] = []
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adapter_parameters = [parameter for parameter in input_adapter.parameters() if parameter.requires_grad]
    adapter_parameters.extend(parameter for parameter in output_refiner.parameters() if parameter.requires_grad)
    if model_parameters:
        parameter_groups.append({"params": model_parameters, "lr": args.backbone_lr})
    if adapter_parameters:
        parameter_groups.append({"params": adapter_parameters, "lr": args.adapter_lr})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    total_steps = args.epochs * args.iterations_per_epoch
    scheduler = RatioPolyLRScheduler(
        optimizer, [float(group["lr"]) for group in parameter_groups], total_steps,
    )
    pairs = top_training_pairs(args.dataset_root, 15)
    protocol = {
        "experiment": EXPERIMENT,
        "variant": args.variant,
        "backbone": "nnU-Net PlainConvUNet RAM-W600",
        "input_seam_prior": enable_input,
        "output_overlap_prior": enable_output,
        "training_mode": args.training_mode,
        "split": "RAM train/validation only",
        "test_used": False,
        "case_counts": {"train": len(train_set), "val": len(val_set)},
        "iterations": {"epochs": args.epochs, "iterations_per_epoch": args.iterations_per_epoch,
                        "total": total_steps},
        "optimizer": "AdamW",
        "scheduler": "ratio-preserving polynomial",
        "learning_rates": {"backbone": args.backbone_lr, "adapters": args.adapter_lr},
        "seed": args.seed,
        "selection_rule": "maximum validation overall_dsc; fixed official evaluator; no test reads",
        "source_checkpoint": source,
        "prior_sources": prior_sources,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "frozen_teacher": "same required native checkpoint, raw logits, no GT in forward",
        "data_contract": contract,
        "config": _config(args),
    }
    _write_json(args.output / "protocol.json", protocol)

    best_value = -math.inf
    best_epoch: int | None = None
    best_hash: str | None = None
    manifest_path = args.output / "checkpoint_manifest.jsonl"
    history_path = args.output / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_report = _train_epoch(
            model, plugin, overlap_prior, teacher, input_adapter, output_refiner,
            train_loader, optimizer, scheduler, device,
            enable_input=enable_input, enable_output=enable_output, steps=args.steps,
            iterations=args.iterations_per_epoch, max_aux_ratio=args.max_aux_ratio,
        )
        if args.variant == "A":
            # A is a real native continuation, evaluated with the same native
            # collector as the immutable baseline.
            val_report = r351_collect_native(model, val_loader, device, pairs, args.seam_threshold)
        else:
            val_report = collect_adapted(
                model, plugin, overlap_prior, val_loader, device, pairs,
                enable_input, enable_output, args.steps,
            )
        selection_value = _selection_value(val_report)
        metrics_hash = _hash_json({"official": val_report["official"], "flat": val_report["flat"]})
        row = {
            "epoch": epoch,
            "evaluation_epoch": epoch,
            "variant": args.variant,
            "train": train_report,
            "val": val_report,
            "selection_metric": "overall_dsc",
            "selection_value": selection_value,
            "evaluation_metrics_sha256": metrics_hash,
            "source_checkpoint_sha256": source["sha256"],
        }
        best_updated = best_epoch is None or selection_value > best_value
        if best_updated:
            best_value = selection_value
            best_epoch = epoch
            best_hash = _save_best(
                args.output / "best.pth", model, input_adapter, output_refiner,
                epoch=epoch, variant=args.variant, training_mode=args.training_mode,
                source=source, row=row,
            )
        manifest_row = {
            "epoch": epoch,
            "selection_metric": "val.overall_dsc",
            "selection_value": selection_value,
            "evaluation_metrics_sha256": metrics_hash,
            "best_updated": best_updated,
            "best_checkpoint": {"path": str(args.output / "best.pth"), "epoch": best_epoch,
                                "sha256": best_hash} if best_epoch is not None else None,
        }
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_row, default=_json_default) + "\n")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=_json_default) + "\n")
        print(json.dumps({"epoch": epoch, "overall_dsc": selection_value,
                          "best_updated": best_updated}, default=_json_default), flush=True)

    if best_epoch is None or best_hash is None:
        raise RuntimeError("no final-prior checkpoint selected")
    result = {
        "experiment": EXPERIMENT,
        "status": "complete",
        "variant": args.variant,
        "training_mode": args.training_mode,
        "split": "RAM train/validation only",
        "test_used": False,
        "test_used_for_selection": False,
        "git_commit": "737cc4f3a1c395f1ae6fe842c4a2e51d6cf98353",
        "source_checkpoint": source,
        "native_checkpoint_sha256": source["sha256"],
        "best_checkpoint": {"path": str(args.output / "best.pth"), "epoch": best_epoch,
                             "sha256": best_hash},
        "selection_rule": "maximum validation overall_dsc; fixed official evaluator",
        "history": str(history_path),
        "checkpoint_manifest": str(manifest_path),
        "protocol": str(args.output / "protocol.json"),
        "data_contract": contract,
        "prior_sources": prior_sources,
        "config": _config(args),
    }
    _write_json(args.output / "run_manifest.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/root/autodl-tmp/YOLO_SAM_generic_src/data/remote_variants/RAM-W600"))
    parser.add_argument("--prior-root", type=Path,
                        default=Path("/root/autodl-tmp/YOLO_SAM_generic_src/outputs/priors/r323_ram_native_r317_single_seed"))
    parser.add_argument("--native-checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/YOLO_SAM_generic_src/outputs/ram_w600/r325_nnunet_native_resolution_overlap_iem/plain_native_best.pth"))
    parser.add_argument("--r332-checkpoint", type=Path,
                        default=Path("/root/autodl-tmp/YOLO_SAM_generic_src/outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth"))
    parser.add_argument("--overlap-checkpoint", type=Path,
                        default=Path("/root/r351_workspace/overlap/full/best.pth"))
    parser.add_argument("--output", type=Path, default=Path("/root/final_prior_v1/ram"))
    parser.add_argument("--variant", choices=VARIANTS, default="D")
    parser.add_argument("--training-mode", choices=("adapter_only", "joint_finetune"),
                        default="joint_finetune")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--iterations-per-epoch", type=int, default=DEFAULT_ITERS_PER_EPOCH)
    parser.add_argument("--backbone-lr", type=float, default=DEFAULT_BACKBONE_LR)
    parser.add_argument("--adapter-lr", type=float, default=DEFAULT_ADAPTER_LR)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--max-aux-ratio", type=float, default=DEFAULT_MAX_AUX_RATIO)
    parser.add_argument("--seam-threshold", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--audit", action="store_true", help="Run true two-batch identity/backward audit")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Collect all 69 validation cases from the native checkpoint and stop")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.audit and args.baseline_only:
        raise ValueError("--audit and --baseline-only are mutually exclusive")
    if args.epochs < 1 or args.iterations_per_epoch < 1:
        raise ValueError("epochs and iterations-per-epoch must be positive")
    if args.backbone_lr <= 0 or args.adapter_lr <= 0:
        raise ValueError("learning rates must be positive")
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.audit:
        run_audit(args, device)
    elif args.baseline_only:
        run_baseline(args, device)
    else:
        if device.type != "cuda":
            raise RuntimeError("native-resolution RAM final training requires CUDA")
        run_training(args, device)


if __name__ == "__main__":
    main()
