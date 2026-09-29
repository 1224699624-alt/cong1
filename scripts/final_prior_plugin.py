"""Thin, backbone-independent TSRS structural-prior plugin.

The final TSRS runs keep the native nnU-Net input contract while routing the
seam map only through the plugin.  In particular, the second native backbone
channel is *always* zero.  This prevents the matched control from receiving a
hidden seam input and makes the A/B/C/D comparison a native-start comparison.

The implementation intentionally reuses the R351 modules.  No target or other
ground-truth tensor is accepted by an inference forward method.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from r351_dual_prior import (
    ImageOverlapPrior,
    OutputCompletionPrior,
    SeamInputAdapter,
    normalize_prior,
)


NATIVE_CHECKPOINT_SHA256 = (
    "55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b"
)

_VARIANT_ALIASES = {
    "T0": "A",
    "T1": "B",
    "R0": "A",
    "R1": "C",
    "A": "A",
    "NATIVE_CONTINUE": "A",
    "NATIVE-CONTINUE": "A",
    "B": "B",
    "SEAM_ONLY": "B",
    "SEAM-ONLY": "B",
    "C": "C",
    "OVERLAP_ONLY": "C",
    "OVERLAP-ONLY": "C",
    "D": "D",
    "DUAL": "D",
}


def normalize_variant(value: str) -> str:
    """Return the canonical A/B/C/D name for a final-run variant."""

    key = str(value).strip().upper()
    try:
        return _VARIANT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown final prior variant {value!r}; expected A/B/C/D or "
            "native_continue/seam_only/overlap_only/dual"
        ) from exc


def variant_config(variant: str) -> dict[str, Any]:
    """Return the immutable branch contract for a final variant."""

    canonical = normalize_variant(variant)
    return {
        "A": {
            "name": "native_continue",
            "enable_input_prior": False,
            "enable_output_prior": False,
        },
        "B": {
            "name": "seam_only",
            "enable_input_prior": True,
            "enable_output_prior": False,
        },
        "C": {
            "name": "overlap_only",
            "enable_input_prior": False,
            "enable_output_prior": True,
        },
        "D": {
            "name": "dual",
            "enable_input_prior": True,
            "enable_output_prior": True,
        },
    }[canonical] | {"variant": canonical}


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a checkpoint without loading its potentially large tensor payload."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_state(payload: Any, preferred: str | None = None) -> Mapping[str, torch.Tensor]:
    """Extract a state dictionary from the checkpoint formats used in R351."""

    if isinstance(payload, Mapping):
        if preferred is not None and isinstance(payload.get(preferred), Mapping):
            return payload[preferred]
        for key in ("network_weights", "state_dict", "model", "refiner"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return value
        if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise TypeError("checkpoint does not contain a tensor state dictionary")


def load_state_dict_from_checkpoint(
    path: str | os.PathLike[str],
    *,
    preferred: str | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a state dictionary while accepting native/RAM checkpoint wrappers."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    return dict(_payload_state(payload, preferred))


def assert_native_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str = NATIVE_CHECKPOINT_SHA256,
) -> dict[str, Any]:
    """Verify the immutable native source and its zero-prior input contract."""

    source = Path(path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"native checkpoint is missing or empty: {source}")
    digest = sha256_file(source)
    expected = str(expected_sha256).lower()
    if digest != expected:
        raise RuntimeError(f"native checkpoint hash mismatch: {digest} != {expected}")
    state = load_state_dict_from_checkpoint(source, preferred="network_weights")
    first_keys = (
        "encoder.stages.0.0.convs.0.conv.weight",
        "encoder.stages.0.0.convs.0.all_modules.0.weight",
        "decoder.encoder.stages.0.0.convs.0.conv.weight",
        "decoder.encoder.stages.0.0.all_modules.0.weight",
        "decoder.encoder.stages.0.0.convs.0.all_modules.0.weight",
    )
    checked: list[str] = []
    for key in first_keys:
        if key not in state:
            continue
        value = state[key]
        if value.ndim != 4 or value.shape[1] != 2:
            raise RuntimeError(f"native first convolution is not two-channel: {key} {tuple(value.shape)}")
        if int(torch.count_nonzero(value[:, 1])) != 0:
            raise RuntimeError(f"native prior channel is nonzero in {key}")
        checked.append(key)
    if not checked:
        raise RuntimeError("native checkpoint has no recognized first convolution")
    return {
        "path": str(source),
        "sha256": digest,
        "expected_sha256": expected,
        "checked_zero_prior_keys": checked,
        "prior_channel_nonzero": 0,
    }


def zero_native_prior_channel(image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """Build the only tensor that may be passed to the native backbone."""

    if image.ndim != 4 or seam.ndim != 4:
        raise ValueError(f"expected BCHW image/seam, got {image.shape} and {seam.shape}")
    if image.shape[0] != seam.shape[0] or image.shape[-2:] != seam.shape[-2:]:
        raise ValueError(f"image/seam shape mismatch: {image.shape} vs {seam.shape}")
    if image.shape[1] != 1 or seam.shape[1] != 1:
        raise ValueError("final TSRS contract expects one image and one seam channel")
    return torch.cat((image, torch.zeros_like(seam)), dim=1)


class StructuralPriorPlugin(nn.Module):
    """R351 input/output priors with explicit branch switches.

    ``adapt_input`` and ``refine_output`` are deliberately GT-free.  The
    overlap map is computed by the frozen image-conditioned prior.  The TSRS
    trainer supplies no membership target, so its strong overlap loss remains
    disabled in the trainer even when this context is used at inference.
    """

    def __init__(
        self,
        seam_prior: nn.Module | None = None,
        overlap_prior: nn.Module | None = None,
        input_adapter: SeamInputAdapter | None = None,
        output_refiner: OutputCompletionPrior | None = None,
        *,
        enable_input: bool = True,
        enable_output: bool = True,
        enable_input_prior: bool | None = None,
        enable_output_prior: bool | None = None,
        output_residual_scale: float = 1.0,
        overlap_context_scale: float = 1.0,
        refinement_steps: int = 1,
        refinement_chunk: int = 3,
    ) -> None:
        super().__init__()
        if enable_input_prior is not None:
            enable_input = bool(enable_input_prior)
        if enable_output_prior is not None:
            enable_output = bool(enable_output_prior)
        if output_residual_scale < 0:
            raise ValueError("output_residual_scale must be non-negative")
        if overlap_context_scale < 0:
            raise ValueError("overlap_context_scale must be non-negative")
        if refinement_steps < 1 or refinement_chunk < 1:
            raise ValueError("refinement_steps and refinement_chunk must be positive")

        # Keep the R351 module names and state-dict paths stable for transfer.
        self.input_adapter = input_adapter if input_adapter is not None else SeamInputAdapter()
        self.output_refiner = output_refiner if output_refiner is not None else OutputCompletionPrior()
        self.overlap_prior = overlap_prior if overlap_prior is not None else ImageOverlapPrior()
        self.enable_input_prior = bool(enable_input)
        self.enable_output_prior = bool(enable_output)
        self.output_residual_scale = float(output_residual_scale)
        self.overlap_context_scale = float(overlap_context_scale)
        self.refinement_steps = int(refinement_steps)
        self.refinement_chunk = int(refinement_chunk)
        self.overlap_prior.requires_grad_(False)
        self.overlap_prior.eval()

    # Short aliases are useful to callers that use the R351 names directly.
    @property
    def input_prior(self) -> SeamInputAdapter:
        return self.input_adapter

    @property
    def output_prior(self) -> OutputCompletionPrior:
        return self.output_refiner

    @property
    def enable_input(self) -> bool:
        return self.enable_input_prior

    @property
    def enable_output(self) -> bool:
        return self.enable_output_prior

    def set_branches(
        self,
        *,
        enable_input: bool | None = None,
        enable_output: bool | None = None,
        output_residual_scale: float | None = None,
        overlap_context_scale: float | None = None,
    ) -> None:
        """Update branch controls without replacing any learned module."""

        if enable_input is not None:
            self.enable_input_prior = bool(enable_input)
        if enable_output is not None:
            self.enable_output_prior = bool(enable_output)
        if output_residual_scale is not None:
            if output_residual_scale < 0:
                raise ValueError("output_residual_scale must be non-negative")
            self.output_residual_scale = float(output_residual_scale)
        if overlap_context_scale is not None:
            if overlap_context_scale < 0:
                raise ValueError("overlap_context_scale must be non-negative")
            self.overlap_context_scale = float(overlap_context_scale)

    def overlap_probability(self, image: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
        """Return frozen, GT-free overlap evidence in the input geometry."""

        if not (self.enable_input_prior or self.enable_output_prior):
            return torch.zeros_like(image)
        with torch.no_grad():
            probability = torch.sigmoid(self.overlap_prior(image.float(), seam.float()))
        return probability * self.overlap_context_scale

    def adapt_input(
        self,
        image: torch.Tensor,
        seam: torch.Tensor,
        overlap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the R351 seam adapter, or return the image unchanged."""

        if not self.enable_input_prior:
            return image
        if overlap is None:
            overlap = self.overlap_probability(image, seam)
        return self.input_adapter(image, seam, overlap)

    def refine_output(
        self,
        image: torch.Tensor,
        logits: torch.Tensor,
        seam: torch.Tensor,
        overlap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply an optional R351 output residual with an explicit scale."""

        if not self.enable_output_prior or self.output_residual_scale == 0:
            return logits
        if logits.ndim != 4 or logits.shape[1] < 1:
            raise ValueError(f"refine_output expects BCHW logits, got {logits.shape}")
        if overlap is None:
            overlap = self.overlap_probability(image, seam)
        refined = self.output_refiner.refine(
            image.float(), logits.float(), seam.float(), overlap.float(),
            steps=self.refinement_steps, chunk=self.refinement_chunk,
        )
        delta = refined - logits.float()
        # Do not branch on delta's current numerical value.  The residual head
        # is zero-initialized, but its parameter gradient is nonzero at the
        # identity point; an early return here would make the output prior
        # permanently untrainable.
        return logits + self.output_residual_scale * delta.to(logits.dtype)

    def active_parameter_names(self) -> list[str]:
        """List prior parameters that are allowed to receive gradients."""

        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def forward(
        self,
        image: torch.Tensor,
        logits: torch.Tensor,
        seam: torch.Tensor,
        overlap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GT-free output-side convenience call."""

        return self.refine_output(image, logits, seam, overlap)


class BackboneAdapter(nn.Module):
    """Minimal logits contract for backbone-independent callers."""

    def forward_logits(self, image_or_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def configure_plugin_trainability(
    plugin: StructuralPriorPlugin,
    *,
    training_mode: str,
) -> dict[str, Any]:
    """Set prior-module gradients for ``adapter_only`` or ``joint_finetune``."""

    mode = str(training_mode).strip().lower()
    if mode not in {"adapter_only", "joint_finetune"}:
        raise ValueError("training_mode must be adapter_only or joint_finetune")
    plugin.overlap_prior.requires_grad_(False)
    plugin.input_adapter.requires_grad_(mode == "adapter_only" or plugin.enable_input_prior)
    plugin.output_refiner.requires_grad_(mode == "adapter_only" or plugin.enable_output_prior)
    if not plugin.enable_input_prior:
        plugin.input_adapter.requires_grad_(False)
    if not plugin.enable_output_prior:
        plugin.output_refiner.requires_grad_(False)
    return {
        "training_mode": mode,
        "enable_input_prior": plugin.enable_input_prior,
        "enable_output_prior": plugin.enable_output_prior,
        "active_parameter_names": plugin.active_parameter_names(),
    }


def load_r351_output_refiner(
    refiner: OutputCompletionPrior,
    checkpoint: str | os.PathLike[str],
    *,
    identity: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load the R332 refiner into the R351 seven-channel wrapper.

    The transfer is restricted to the output prior.  The native backbone is
    never loaded from this checkpoint.
    """

    state = load_state_dict_from_checkpoint(checkpoint, preferred="refiner", map_location=map_location)
    report = refiner.load_mature(state, identity=identity)
    report.update({"checkpoint": str(Path(checkpoint)), "sha256": sha256_file(checkpoint), "identity": bool(identity)})
    return report


def make_run_manifest(
    *,
    output_folder: str | os.PathLike[str],
    dataset: str,
    split: str,
    variant: str,
    training_mode: str,
    seed: int,
    native_checkpoint: str | os.PathLike[str],
    native_source: Mapping[str, Any],
    overlap_checkpoint: str | os.PathLike[str] | None,
    output_refiner_checkpoint: str | os.PathLike[str] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the immutable provenance record for a final run."""

    canonical = normalize_variant(variant)
    branches = variant_config(canonical)
    overlap = Path(overlap_checkpoint) if overlap_checkpoint else None
    refiner = Path(output_refiner_checkpoint) if output_refiner_checkpoint else None
    git_commit = os.environ.get("FINAL_PRIOR_GIT_COMMIT", "unknown")
    evaluator_version = os.environ.get("FINAL_PRIOR_EVALUATOR_VERSION", "R201 standard evaluator")
    evaluator_path = os.environ.get("FINAL_PRIOR_EVALUATOR_SCRIPT")
    evaluator_sha = (
        sha256_file(evaluator_path)
        if evaluator_path and Path(evaluator_path).is_file()
        else os.environ.get("FINAL_PRIOR_EVALUATOR_SHA256")
    )
    return {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "backbone": "nnU-Net native architecture",
        "variant": canonical,
        "variant_name": branches["name"],
        "seed": int(seed),
        "git_commit": os.environ.get("FINAL_PRIOR_GIT_COMMIT"),
        "git_commit": git_commit,
        "training_mode": str(training_mode),
        "source_checkpoint": str(native_checkpoint),
        "source_checkpoint_sha256": native_source.get("sha256"),
        "source_checkpoint_expected_sha256": native_source.get("expected_sha256"),
        "source_checkpoint_zero_prior_channel": True,
        "source_overlap_checkpoint": str(overlap) if overlap else None,
        "source_overlap_checkpoint_sha256": sha256_file(overlap) if overlap and overlap.is_file() else None,
        "source_output_refiner_checkpoint": str(refiner) if refiner else None,
        "source_output_refiner_checkpoint_sha256": sha256_file(refiner) if refiner and refiner.is_file() else None,
        "backbone_initialization_policy": "native checkpoint only; R350/R332 are forbidden as backbone sources",
        "output_refiner_initialization_policy": "R332 transfer allowed for output refiner with identity residual initialization",
        "native_input_contract": {
            "image_channel": 0,
            "backbone_prior_channel": "constant zero",
            "seam_routed_to_plugin_only": True,
        },
        "branch_controls": branches,
        "config": dict(config),
        "evaluation_protocol": "TSRS standard R201 metrics on train/val; test is never used for selection",
        "evaluator_version": evaluator_version,
        "evaluator_sha256": evaluator_sha,
        "validation_selection": {
            "split": "val",
            "metric": "nnU-Net EMA foreground Dice",
            "direction": "max",
            "test_used_for_selection": False,
        },
        "selected_checkpoint": None,
        "selected_checkpoint_sha256": None,
        "selected_epoch": None,
        "status": "initialized",
    }


def write_manifest_exclusive(output_folder: str | os.PathLike[str], manifest: Mapping[str, Any]) -> Path:
    """Write a manifest once and reject accidental reruns/overwrites."""

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "run_manifest.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing final run: {path}")
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def update_manifest(path: str | os.PathLike[str], updates: Mapping[str, Any]) -> None:
    """Update the current run manifest in place after a validated event."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(dict(updates))
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


__all__ = [
    "NATIVE_CHECKPOINT_SHA256",
    "StructuralPriorPlugin",
    "assert_native_checkpoint",
    "configure_plugin_trainability",
    "load_r351_output_refiner",
    "load_state_dict_from_checkpoint",
    "make_run_manifest",
    "normalize_prior",
    "normalize_variant",
    "sha256_file",
    "update_manifest",
    "variant_config",
    "write_manifest_exclusive",
    "zero_native_prior_channel",
]
