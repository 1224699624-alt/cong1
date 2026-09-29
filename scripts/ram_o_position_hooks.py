"""Attach a frozen, caller-supplied ``O`` prior at two R332 positions.

The mature C0/R332 weights are left in place.  Each attachment owns two
separate, zero-initialized trainable adapters with the same capacity::

    Conv2d(1, 12, 3, padding=1, bias=False)

``INPUT_O`` adds the early adapter response after the first two-channel
backbone convolution.  A fixed 12-to-32 orthonormal projection maps that
response to the convolution output width.  ``REFINER_O`` adds the late
adapter response after ``refiner.enc1.net[0]``.  The latter is already a
12-channel response, so its projection is the identity.  ``ZERO`` keeps both
paths inactive while still running them through a zero gate; this gives the
inactive adapters registered, finite, all-zero gradients during a normal
backward pass.

The runner is responsible for constructing O.  ``set_o`` only stores a
detached copy.  No generator, data access, or random operation is performed
from a hook.
"""
from __future__ import annotations

import weakref
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ARM_ZERO = "ZERO"
ARM_INPUT_O = "INPUT_O"
ARM_REFINER_O = "REFINER_O"
SUPPORTED_ARMS = (ARM_ZERO, ARM_INPUT_O, ARM_REFINER_O)

ADAPTER_CHANNELS = 12
INPUT_FEATURE_CHANNELS = 32
INPUT_CHANNELS = 2
REFINER_INPUT_CHANNELS = 7
REFINER_FEATURE_CHANNELS = 12


def _lookup_path(module: nn.Module, path: tuple[str, ...]) -> nn.Module | None:
    """Resolve a mixed attribute/index path without swallowing bad modules."""

    value: Any = module
    try:
        for name in path:
            if name.isdigit():
                value = value[int(name)]
            else:
                value = getattr(value, name)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None
    return value if isinstance(value, nn.Module) else None


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Infer the device/dtype from the first parameter or buffer."""

    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


def _find_input_conv(model: nn.Module) -> nn.Conv2d:
    """Find the mature two-channel first backbone convolution."""

    exact = _lookup_path(
        model,
        ("backbone", "encoder", "stages", "0", "0", "convs", "0", "conv"),
    )
    candidates: list[nn.Module] = []
    if exact is not None:
        candidates.append(exact)
    backbone = getattr(model, "backbone", None)
    encoder = getattr(backbone, "encoder", None)
    if isinstance(encoder, nn.Module):
        candidates.extend(encoder.modules())
    for candidate in candidates:
        if isinstance(candidate, nn.Conv2d):
            if candidate.in_channels == INPUT_CHANNELS and candidate.out_channels == INPUT_FEATURE_CHANNELS:
                if candidate.kernel_size == (3, 3):
                    return candidate
    raise TypeError(
        "model must expose a 32-channel Conv2d(2, 32, 3, ...) under "
        "backbone.encoder.stages[0][0].convs[0].conv"
    )


def _find_refiner_conv(refiner: nn.Module) -> nn.Conv2d:
    """Find the mature R332 seven-channel first refiner convolution."""

    exact = _lookup_path(refiner, ("enc1", "net", "0"))
    candidates: list[nn.Module] = []
    if exact is not None:
        candidates.append(exact)
    enc1 = getattr(refiner, "enc1", None)
    if isinstance(enc1, nn.Module):
        candidates.extend(enc1.modules())
    for candidate in candidates:
        if isinstance(candidate, nn.Conv2d):
            if candidate.in_channels == REFINER_INPUT_CHANNELS and candidate.out_channels == REFINER_FEATURE_CHANNELS:
                if candidate.kernel_size == (3, 3):
                    return candidate
    raise TypeError(
        "refiner must expose Conv2d(7, 12, 3, ...) at enc1.net[0]"
    )


class _FixedOrthonormalProjection(nn.Module):
    """A deterministic, frozen 12-to-32 pointwise orthonormal projection.

    The padded identity basis has orthonormal columns (``Q.T @ Q == I``), is
    deterministic, and does not consume the caller's RNG state.  It also
    makes the capacity comparison explicit: only the preceding 1-to-12
    spatial adapter is trainable.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        weight = torch.zeros((INPUT_FEATURE_CHANNELS, ADAPTER_CHANNELS), dtype=dtype, device=device)
        weight[:ADAPTER_CHANNELS, :] = torch.eye(ADAPTER_CHANNELS, dtype=dtype, device=device)
        self.register_buffer("weight", weight[:, :, None, None], persistent=True)

    def forward(self, value: Tensor) -> Tensor:
        weight = self.weight
        if weight.device != value.device or weight.dtype != value.dtype:
            weight = weight.to(device=value.device, dtype=value.dtype)
        return F.conv2d(value, weight)


class PositionHookAttachment(nn.Module):
    """Live hooks and trainable O adapters for one position arm.

    The attachment itself is an ``nn.Module`` so ``attachment.parameters()``
    contains exactly the two trainable adapters.  The mature model/refiner
    are intentionally held only through weak references and are never
    registered as children of this module.
    """

    def __init__(
        self,
        model: nn.Module,
        refiner: nn.Module,
        arm: str,
        input_conv: nn.Conv2d,
        refiner_conv: nn.Conv2d,
    ) -> None:
        super().__init__()
        self.arm = _normalize_arm(arm)
        self._model_ref = weakref.ref(model)
        self._refiner_ref = weakref.ref(refiner)
        self._input_conv_ref = weakref.ref(input_conv)
        self._refiner_conv_ref = weakref.ref(refiner_conv)

        input_device, input_dtype = _module_device_dtype(input_conv)
        refiner_device, refiner_dtype = _module_device_dtype(refiner_conv)
        self.input_adapter = nn.Conv2d(
            1,
            ADAPTER_CHANNELS,
            kernel_size=3,
            padding=1,
            bias=False,
            device=input_device,
            dtype=input_dtype,
        )
        self.refiner_adapter = nn.Conv2d(
            1,
            ADAPTER_CHANNELS,
            kernel_size=3,
            padding=1,
            bias=False,
            device=refiner_device,
            dtype=refiner_dtype,
        )
        nn.init.zeros_(self.input_adapter.weight)
        nn.init.zeros_(self.refiner_adapter.weight)

        self.early_projection = _FixedOrthonormalProjection(input_device, input_dtype)
        self.late_projection = nn.Identity()

        self._o: Tensor | None = None
        self._closed = False
        self._input_hook_calls = 0
        self._refiner_hook_calls = 0
        self._input_applied_calls = 0
        self._refiner_applied_calls = 0

        # Both hooks stay installed for every arm.  An inactive arm is
        # multiplied by zero in its hook, which keeps its gradient explicitly
        # zero while preserving an optimizer-visible parameter tensor.
        self._input_handle = input_conv.register_forward_hook(self._on_input_output)
        self._refiner_handle = refiner_conv.register_forward_hook(self._on_refiner_output)

    @property
    def input_projection(self) -> nn.Module:
        """Alias for the fixed early 12-to-32 projection."""

        return self.early_projection

    @property
    def hook_calls(self) -> dict[str, int]:
        """Return hook invocation counts without exposing mutable internals."""

        return {
            "input": self._input_hook_calls,
            "refiner": self._refiner_hook_calls,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """Return compact attachment state and hook counters."""

        return {
            "arm": self.arm,
            "closed": self._closed,
            "o_set": self._o is not None,
            "input_hook_calls": self._input_hook_calls,
            "refiner_hook_calls": self._refiner_hook_calls,
            "input_applied_calls": self._input_applied_calls,
            "refiner_applied_calls": self._refiner_applied_calls,
        }

    @property
    def input_hook_calls(self) -> int:
        return self._input_hook_calls

    @property
    def refiner_hook_calls(self) -> int:
        return self._refiner_hook_calls

    def reset_stats(self) -> None:
        """Reset hook counters while leaving adapters and O unchanged."""

        self._input_hook_calls = 0
        self._refiner_hook_calls = 0
        self._input_applied_calls = 0
        self._refiner_applied_calls = 0

    def set_o(self, value: Tensor | None) -> Tensor | None:
        """Store a detached O tensor for subsequent model/refiner calls.

        ``value`` must be ``[B, 1, H, W]``.  ``None`` clears O and makes all
        hooks use zeros, which is the exact C0/R332 parity path at zero
        initialization.
        """

        if value is None:
            self._o = None
            return None
        if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"O must have shape [B,1,H,W], got {getattr(value, 'shape', None)}")
        if value.shape[0] <= 0 or value.shape[-2] <= 0 or value.shape[-1] <= 0:
            raise ValueError(f"O has invalid shape {tuple(value.shape)}")
        detached = value.detach().clone()
        if not bool(torch.isfinite(detached).all().item()):
            raise ValueError("O contains non-finite values")
        self._o = detached
        return detached

    def clear_o(self) -> None:
        """Clear O; hooks will use zeros until the next :meth:`set_o`."""

        self._o = None

    def _source_for(self, batch: int, spatial: tuple[int, int], target: Tensor) -> Tensor:
        """Broadcast stored O to a hook's batch/spatial shape."""

        if self._o is None:
            return torch.zeros((batch, 1, *spatial), device=target.device, dtype=target.dtype)
        value = self._o
        if tuple(value.shape[-2:]) != spatial:
            raise ValueError(
                f"O spatial shape {tuple(value.shape[-2:])} does not match hook output {spatial}"
            )
        if value.shape[0] == batch:
            source = value
        elif value.shape[0] == 1:
            source = value.repeat(batch, 1, 1, 1)
        elif batch % value.shape[0] == 0:
            source = value.repeat_interleave(batch // value.shape[0], dim=0)
        else:
            raise ValueError(f"cannot repeat O batch {value.shape[0]} to hook batch {batch}")
        return source.to(device=target.device, dtype=target.dtype)

    @staticmethod
    def _validate_output(output: Any, name: str) -> Tensor:
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise TypeError(f"{name} hook expected a [B,C,H,W] tensor, got {type(output)!r}")
        return output

    def _on_input_output(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Tensor:
        self._input_hook_calls += 1
        value = self._validate_output(output, "backbone input")
        source = self._source_for(value.shape[0], tuple(value.shape[-2:]), value)
        # The adapter is initialized to zero, and an inactive arm is gated by
        # zero.  Both cases preserve the mature conv output exactly.
        response = self.early_projection(self.input_adapter(source))
        if self.arm == ARM_INPUT_O:
            self._input_applied_calls += 1
        else:
            response = response * 0.0
        return value + response.to(device=value.device, dtype=value.dtype)

    def _on_refiner_output(self, _module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Tensor:
        self._refiner_hook_calls += 1
        value = self._validate_output(output, "refiner enc1")
        source = self._source_for(value.shape[0], tuple(value.shape[-2:]), value)
        response = self.late_projection(self.refiner_adapter(source))
        if self.arm == ARM_REFINER_O:
            self._refiner_applied_calls += 1
        else:
            response = response * 0.0
        return value + response.to(device=value.device, dtype=value.dtype)

    def close(self) -> None:
        """Remove both hooks; adapter modules remain available for saving."""

        if self._closed:
            return
        self._input_handle.remove()
        self._refiner_handle.remove()
        self._closed = True
        self._o = None

    def __enter__(self) -> "PositionHookAttachment":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def _normalize_arm(arm: str) -> str:
    if not isinstance(arm, str):
        raise TypeError(f"arm must be one of {SUPPORTED_ARMS}, got {type(arm)!r}")
    normalized = arm.strip().upper()
    if normalized not in SUPPORTED_ARMS:
        raise ValueError(f"arm must be one of {SUPPORTED_ARMS}, got {arm!r}")
    return normalized


def attach_position_hooks(model: nn.Module, refiner: nn.Module, arm: str) -> PositionHookAttachment:
    """Attach the selected O-position arm after mature component loading.

    ``model`` and ``refiner`` are untouched structurally: the returned
    attachment owns the new trainable tensors, while the two forward hooks
    only add their zero-initialized responses to existing conv outputs.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(refiner, nn.Module):
        raise TypeError("refiner must be a torch.nn.Module")
    normalized = _normalize_arm(arm)
    input_conv = _find_input_conv(model)
    refiner_conv = _find_refiner_conv(refiner)
    return PositionHookAttachment(model, refiner, normalized, input_conv, refiner_conv)


__all__ = [
    "ADAPTER_CHANNELS",
    "ARM_INPUT_O",
    "ARM_REFINER_O",
    "ARM_ZERO",
    "INPUT_FEATURE_CHANNELS",
    "PositionHookAttachment",
    "SUPPORTED_ARMS",
    "attach_position_hooks",
]
