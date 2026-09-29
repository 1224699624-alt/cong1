"""Read-only high-resolution feature capture for the locked RAM C0 model.

The module intentionally has no optimizer, overlap generator, labels, or
training path.  It loads the exact two-channel C0 backbone and seven-channel
R332 refiner, runs the native two-step full14 forward with an all-zero
explicit O channel, and returns one case of detached NumPy FP32 arrays.

The two captured representations are deliberately different:

* ``backbone_h`` is the output of ``decoder.stages[4]`` immediately before
  the 14-channel ``decoder.seg_layers[4]`` source head.
* ``refiner_h`` is the output of ``refiner.dec1`` during the second R332
  iteration, concatenated in the exact member order 0..13.

No GT or target tensor is accepted by the public capture API.
"""
from __future__ import annotations

import hashlib
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn


_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from r332_v31_port import R332RefinerWithO, iterative_refine_o  # noqa: E402
from train_r322_ram_nnunet_r317_seam_prior import RamNnUNetTwoChannel  # noqa: E402


C0_SHA256 = "ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384"
N_MEMBERS = 14
P0_WIDTH = 12
P1_WIDTH = 32
P0_CHUNKS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), (12, 13))


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of ``path`` without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device_of(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration as exc:
        raise ValueError("module has no parameters from which to infer a device") from exc


def load_frozen_c0(checkpoint: Path, device: torch.device | str) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Strictly load the locked C0 backbone and refiner onto ``device``.

    The returned ``payload`` is metadata only.  The checkpoint's optimizer and
    scaler states are intentionally not returned or instantiated, because this
    reader is inference-only.
    """

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_sha = sha256_file(checkpoint)
    if actual_sha != C0_SHA256:
        raise ValueError(f"C0 SHA256 mismatch: expected {C0_SHA256}, got {actual_sha}")

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
        raise ValueError("C0 checkpoint has no model state mapping")
    if not isinstance(raw.get("refiner"), Mapping):
        raise ValueError("C0 checkpoint has no refiner state mapping")

    model = RamNnUNetTwoChannel()
    refiner = R332RefinerWithO()
    model.load_state_dict(raw["model"], strict=True)
    refiner.load_state_dict(raw["refiner"], strict=True)

    target_device = torch.device(device)
    model.to(target_device).float().eval()
    refiner.to(target_device).float().eval()
    for module in (model, refiner):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    model_input_weight = model.backbone.encoder.stages[0][0].convs[0].conv.weight
    refiner_input_weight = refiner.enc1.net[0].weight
    if tuple(model_input_weight.shape) != (32, 2, 3, 3):
        raise ValueError(f"unexpected C0 model input weight shape: {tuple(model_input_weight.shape)}")
    if tuple(refiner_input_weight.shape) != (12, 7, 3, 3):
        raise ValueError(f"unexpected C0 refiner input weight shape: {tuple(refiner_input_weight.shape)}")

    payload = {
        "checkpoint": str(checkpoint),
        "sha256": actual_sha,
        "updates": raw.get("updates"),
        "epoch": raw.get("epoch"),
        "model_key_count": len(raw["model"]),
        "refiner_key_count": len(raw["refiner"]),
        "optimizer_state_loaded": False,
        "scaler_state_loaded": False,
        "device": str(target_device),
        "model_input_channels": 2,
        "refiner_input_channels": 7,
    }
    return model, refiner, payload


@contextmanager
def _preserve_runtime(*modules: nn.Module) -> Iterator[None]:
    """Preserve training flags, buffers, and torch RNG around a read-only pass."""

    modes = {module: module.training for owner in modules for module in owner.modules()}
    buffers = {
        buffer: buffer.detach().clone()
        for owner in modules
        for buffer in owner.buffers()
    }
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        for module in modules:
            module.eval()
        yield
    finally:
        for module, training in modes.items():
            module.train(training)
        for buffer, value in buffers.items():
            buffer.copy_(value)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _as_member_array(value: torch.Tensor, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1:
        raise ValueError(f"{name} must be [1,C,H,W], got {type(value)!r} {getattr(value, 'shape', None)}")
    return value.detach().float().cpu().numpy()[0].copy()


def _reference_logits(model: nn.Module, refiner: nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the exact C0 source + two-step full14 forward without feature hooks."""

    zero_o = torch.zeros_like(image)
    source = model(torch.cat((image, zero_o), dim=1))
    if isinstance(source, (tuple, list)):
        source = source[0]
    if not isinstance(source, torch.Tensor):
        raise TypeError(f"C0 model returned {type(source)!r}, expected logits tensor")
    states, _ = iterative_refine_o(image, source, refiner, 2, zero_o, indices=None, chunk=3)
    if len(states) != 3:
        raise RuntimeError(f"expected source/step1/step2 states, got {len(states)}")
    return source, states


def forward_c0(model: nn.Module, refiner: nn.Module, image: torch.Tensor) -> dict[str, np.ndarray]:
    """Return unhooked C0 source, step1, and step2 logits for parity checks."""

    image = _prepare_image(model, image)
    with _preserve_runtime(model, refiner), torch.inference_mode():
        source, states = _reference_logits(model, refiner, image)
        return {
            "source": _as_member_array(source, "source"),
            "step1": _as_member_array(states[1], "step1"),
            "step2": _as_member_array(states[2], "step2"),
        }


def _prepare_image(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image)!r}")
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or tuple(image.shape[:2]) != (1, 1):
        raise ValueError(f"image must have shape [1,1,H,W] or [1,H,W], got {tuple(image.shape)}")
    if image.shape[-2] <= 0 or image.shape[-1] <= 0:
        raise ValueError(f"image has invalid spatial shape: {tuple(image.shape)}")
    return image.to(device=_device_of(model), dtype=torch.float32, non_blocking=False)


def capture_case(model: nn.Module, refiner: nn.Module, image: torch.Tensor) -> dict[str, Any]:
    """Capture one case's C0 logits and P0/P1 features as NumPy FP32 arrays.

    ``image`` is the only case tensor accepted.  The function never accepts or
    reads GT.  The explicit O channel is constructed internally as zeros.  All
    returned arrays are detached CPU copies, so hooks and GPU tensors are
    released when this function returns.
    """

    image = _prepare_image(model, image)
    stage4_outputs: list[torch.Tensor] = []
    dec1_second_outputs: list[torch.Tensor] = []
    dec1_calls = 0

    def capture_stage4(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("decoder stage4 did not return a tensor")
        stage4_outputs.append(output.detach().float().cpu())

    def capture_dec1(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal dec1_calls
        if not isinstance(output, torch.Tensor):
            raise TypeError("refiner dec1 did not return a tensor")
        call_index = dec1_calls
        dec1_calls += 1
        # Full14, two-step chunk=3 produces five calls per step.  Keep only
        # step2 so a single case does not retain duplicate step1 features.
        if call_index >= len(P0_CHUNKS):
            dec1_second_outputs.append(output.detach().float().cpu())

    stage4_handle = model.backbone.decoder.stages[4].register_forward_hook(capture_stage4)
    dec1_handle = refiner.dec1.register_forward_hook(capture_dec1)
    try:
        with _preserve_runtime(model, refiner), torch.inference_mode():
            source, states = _reference_logits(model, refiner, image)
    finally:
        stage4_handle.remove()
        dec1_handle.remove()

    if len(stage4_outputs) != 1:
        raise RuntimeError(f"expected one decoder stage4 call, got {len(stage4_outputs)}")
    if dec1_calls != 2 * len(P0_CHUNKS):
        raise RuntimeError(f"expected {2 * len(P0_CHUNKS)} refiner dec1 calls, got {dec1_calls}")
    if len(dec1_second_outputs) != len(P0_CHUNKS):
        raise RuntimeError(f"expected {len(P0_CHUNKS)} second-step dec1 calls, got {len(dec1_second_outputs)}")

    p1 = stage4_outputs[0]
    if p1.ndim != 4 or tuple(p1.shape[:2]) != (1, P1_WIDTH):
        raise RuntimeError(f"unexpected P1 stage4 shape: {tuple(p1.shape)}")
    p0_shapes = [tuple(value.shape) for value in dec1_second_outputs]
    # dec1 outputs are [K,12,H,W], not batched.
    expected_shapes = [(len(chunk), P0_WIDTH, image.shape[-2], image.shape[-1]) for chunk in P0_CHUNKS]
    if p0_shapes != expected_shapes:
        raise RuntimeError(f"unexpected P0 second-step shapes: {p0_shapes}, expected {expected_shapes}")

    refiner_h = torch.cat(dec1_second_outputs, dim=0)
    if tuple(refiner_h.shape) != (N_MEMBERS, P0_WIDTH, image.shape[-2], image.shape[-1]):
        raise RuntimeError(f"unexpected concatenated P0 shape: {tuple(refiner_h.shape)}")

    image_hw = [int(image.shape[-2]), int(image.shape[-1])]
    return {
        "source": _as_member_array(source, "source"),
        "step1": _as_member_array(states[1], "step1"),
        "step2": _as_member_array(states[2], "step2"),
        "backbone_h": p1[0].numpy().copy(),
        "refiner_h": refiner_h.numpy().copy(),
        "backbone_h_shape": [P1_WIDTH, *image_hw],
        "refiner_h_shape": [N_MEMBERS, P0_WIDTH, *image_hw],
        "member_ids": list(range(N_MEMBERS)),
        "member_chunk_ids": [list(chunk) for chunk in P0_CHUNKS],
        "decoder_stage4_calls": len(stage4_outputs),
        "refiner_dec1_calls_total": dec1_calls,
        "refiner_dec1_calls_step1": len(P0_CHUNKS),
        "refiner_dec1_calls_step2": len(dec1_second_outputs),
        "explicit_o": "zeros_like(image), appended as refiner feature channel 6",
        "spatial_alignment": "identity_native_padded",
        "image_hw": image_hw,
    }


__all__ = [
    "C0_SHA256",
    "capture_case",
    "forward_c0",
    "load_frozen_c0",
    "sha256_file",
]
