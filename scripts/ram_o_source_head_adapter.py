"""Image-only ``O`` fusion at the native RAM source head.

The mature RAM model is loaded first and this module is attached afterwards.
It deliberately adds one new parameter tensor at the end of the model's
parameter list while leaving every existing backbone parameter untouched:

``source = seg_layers[4](stage4) + O * Conv2d_32_to_14(stage4)``.

``O`` is generated from the image only.  The generator is evaluated under
``no_grad`` and with its runtime state restored, so it cannot become part of
the training graph and cannot update running statistics or consume RNG state.
The zero arm uses an all-zero ``O`` of the same shape.  The source-head hook
is intentionally local to ``decoder.seg_layers[4]``; the refiner never sees
this tensor.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


ADAPTER_NAME = "o_source_adapter"
FEATURE_CHANNELS = 32
MEMBER_CHANNELS = 14


def _capture_rng() -> tuple[Any, Any, Tensor, list[Tensor]]:
    """Capture the process RNG streams used by PyTorch modules.

    The adapter itself does not sample.  Saving the streams around the
    frozen-generator call makes that contract explicit even if a replacement
    image-only generator contains a stochastic layer.
    """

    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    )


def _restore_rng(state: tuple[Any, Any, Tensor, list[Tensor]]) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if torch.cuda.is_available() and state[3]:
        torch.cuda.set_rng_state_all(state[3])


def _snapshot_buffers(module: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in module.named_buffers()}


def _restore_buffers(module: nn.Module, snapshot: Mapping[str, Tensor]) -> None:
    current = dict(module.named_buffers())
    if set(current) != set(snapshot):
        raise RuntimeError("frozen O generator changed its buffer structure during inference")
    for name, value in snapshot.items():
        current[name].copy_(value)


def _tensor_stats(value: Tensor) -> dict[str, Any]:
    """Return compact, detached statistics suitable for JSON traces."""

    detached = value.detach().float()
    stats: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "numel": int(detached.numel()),
    }
    if detached.numel() == 0:
        stats.update({"finite": True, "mean": None, "std": None, "min": None,
                      "max": None, "abs_mean": None, "l2": 0.0})
        return stats
    finite = bool(torch.isfinite(detached).all().item())
    stats.update({
        "finite": finite,
        "mean": float(detached.mean().cpu()),
        "std": float(detached.std(unbiased=False).cpu()),
        "min": float(detached.min().cpu()),
        "max": float(detached.max().cpu()),
        "abs_mean": float(detached.abs().mean().cpu()),
        "l2": float(detached.norm().cpu()),
    })
    return stats


def _decoder_stage4_head(model: nn.Module) -> nn.Module:
    try:
        head = model.backbone.decoder.seg_layers[4]
    except (AttributeError, IndexError, KeyError, TypeError) as error:
        raise TypeError("model must expose backbone.decoder.seg_layers[4]") from error
    if not isinstance(head, nn.Module):
        raise TypeError("backbone.decoder.seg_layers[4] must be a torch module")
    return head


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        return parameter.device, parameter.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return torch.device("cpu"), torch.float32


class OSourceHeadAttachment:
    """Handle returned by :func:`attach_o_source_head`.

    A normal runner can simply attach once after strict C0 loading.  The
    model forward pre-hook extracts its first input channel and computes O;
    callers that already materialise O may instead call :meth:`set_o` before
    the forward.  ``adapter`` is the registered trainable module and can be
    put in a separate optimizer group without touching the original 22
    decoder parameters.
    """

    def __init__(self, model: nn.Module, generator: nn.Module | None, use_real_o: bool,
                 head: nn.Module, trace_sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.model = model
        self.generator = generator
        self.use_real_o = bool(use_real_o)
        self.head = head
        self.trace_sink = trace_sink
        device, dtype = _module_device_dtype(head)
        self.adapter = nn.Conv2d(FEATURE_CHANNELS, MEMBER_CHANNELS, kernel_size=1,
                                  bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.adapter.weight)

        # Register after the strict C0 load.  The new key is appended after
        # the existing backbone keys and therefore cannot reorder old names.
        if hasattr(model, ADAPTER_NAME):
            raise ValueError(f"model already has {ADAPTER_NAME!r}")
        model.add_module(ADAPTER_NAME, self.adapter)

        self._manual_o: Tensor | None = None
        self._active_o: Tensor | None = None
        self._last_trace: dict[str, Any] | None = None
        self._closed = False
        self._pre_hook = model.register_forward_pre_hook(self._on_model_forward)
        self._head_hook = head.register_forward_hook(self._on_head_forward)

    @property
    def last_trace(self) -> dict[str, Any] | None:
        """Most recent O/input/adapter/source statistics, detached from autograd."""

        return None if self._last_trace is None else dict(self._last_trace)

    @property
    def trace(self) -> dict[str, Any] | None:
        """Alias retained for runners that call the record ``trace``."""

        return self.last_trace

    def set_o(self, value: Tensor | None) -> Tensor | None:
        """Use a caller-materialised O for subsequent forwards.

        ``value`` must be ``[B,1,H,W]``.  It is detached and cloned so a
        generator graph or a later in-place mutation cannot enter the source
        loss.  Passing ``None`` restores automatic generation/zero fallback.
        """

        if value is None:
            self._manual_o = None
            return None
        if not isinstance(value, Tensor) or value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(f"O must have shape [B,1,H,W], got {getattr(value, 'shape', None)}")
        if not bool(torch.isfinite(value.detach()).all().item()):
            raise ValueError("O contains non-finite values")
        self._manual_o = value.detach().clone()
        return self._manual_o

    def clear_o(self) -> None:
        self._manual_o = None
        self._active_o = None

    def compute_o(self, image: Tensor) -> Tensor:
        """Compute and cache O from a single-channel image.

        This method is useful when a runner wants to materialise O explicitly;
        ordinary model calls use the pre-hook and need no extra backbone pass.
        """

        self._validate_image(image)
        if not self.use_real_o:
            value = torch.zeros_like(image)
        else:
            value = self._generator_o(image)
        self.set_o(value)
        return self._manual_o  # type: ignore[return-value]

    def _generator_o(self, image: Tensor) -> Tensor:
        if self.generator is None:
            raise RuntimeError("use_real_o=True requires an image-only generator")
        generator = self.generator
        mode = {module: module.training for module in generator.modules()}
        buffers = _snapshot_buffers(generator)
        rng = _capture_rng()
        try:
            # eval prevents accidental BatchNorm/dropout updates.  Restore
            # every mode and buffer below so this helper has no runtime side
            # effects on a shared frozen generator.
            generator.eval()
            with torch.no_grad():
                raw = generator(image)
            if not isinstance(raw, Tensor):
                raise TypeError("image-only generator must return a tensor")
            if raw.shape != image.shape:
                raise ValueError(f"generator output must match image shape {tuple(image.shape)}, got {tuple(raw.shape)}")
            value = raw.sigmoid()
        finally:
            for module, training in mode.items():
                module.training = training
            _restore_buffers(generator, buffers)
            _restore_rng(rng)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("sigmoid(generator(image)) contains non-finite values")
        return value.detach()

    @staticmethod
    def _validate_image(image: Tensor) -> None:
        if not isinstance(image, Tensor) or image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"image must have shape [B,1,H,W], got {getattr(image, 'shape', None)}")

    def _on_model_forward(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            raise ValueError("source-head attachment requires the model input as its first positional tensor")
        model_input = inputs[0]
        if model_input.ndim != 4 or model_input.shape[1] < 1:
            raise ValueError(f"model input must be [B,C,H,W] with C>=1, got {tuple(model_input.shape)}")
        if self._manual_o is not None:
            self._active_o = self._manual_o
        elif self.use_real_o:
            self._active_o = self._generator_o(model_input[:, :1])
        else:
            # The head hook creates a zero tensor on the decoder output's
            # device/dtype, so this branch never moves/copies the input.
            self._active_o = None

    def _on_head_forward(self, _module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Tensor:
        active = self._active_o
        try:
            if not isinstance(output, Tensor):
                raise TypeError("seg_layers[4] must return a tensor")
            if not inputs or not isinstance(inputs[0], Tensor):
                raise TypeError("seg_layers[4] hook did not receive its stage-4 feature tensor")
            feature = inputs[0]
            if feature.ndim != 4 or feature.shape[1] != FEATURE_CHANNELS:
                raise ValueError(f"stage-4 feature must be [B,32,H,W], got {tuple(feature.shape)}")
            if output.ndim != 4 or output.shape[1] != MEMBER_CHANNELS:
                raise ValueError(f"source logits must be [B,14,H,W], got {tuple(output.shape)}")
            if feature.shape[0] != output.shape[0] or feature.shape[-2:] != output.shape[-2:]:
                raise ValueError("stage-4 feature and source logits coordinates differ")

            if active is None:
                prior = torch.zeros((output.shape[0], 1, *output.shape[-2:]),
                                    device=output.device, dtype=output.dtype)
            else:
                if active.ndim != 4 or active.shape[1] != 1:
                    raise ValueError(f"O must have shape [B,1,H,W], got {tuple(active.shape)}")
                if active.shape[0] != output.shape[0] or active.shape[-2:] != output.shape[-2:]:
                    raise ValueError("O and source-logit coordinates differ")
                prior = active.to(device=output.device, dtype=output.dtype)

            raw_adapter = self.adapter(feature)
            if raw_adapter.shape != output.shape:
                raise ValueError(f"adapter output must match source logits, got {tuple(raw_adapter.shape)} and {tuple(output.shape)}")
            response = prior * raw_adapter
            fused = output + response
            self._last_trace = {
                "use_real_o": self.use_real_o,
                "o": _tensor_stats(prior),
                "stage4_input": _tensor_stats(feature),
                "adapter_output": _tensor_stats(raw_adapter),
                "adapter_response": _tensor_stats(response),
                "source_logits_before": _tensor_stats(output),
                "source_logits_after": _tensor_stats(fused),
            }
            if self.trace_sink is not None:
                self.trace_sink(dict(self._last_trace))
            return fused
        finally:
            # Automatic O is per forward.  A manually set O remains available
            # until clear_o()/set_o(None), which supports paired diagnostics.
            self._active_o = None

    def close(self) -> None:
        """Remove hooks while leaving the registered adapter for checkpointing."""

        if self._closed:
            return
        self._head_hook.remove()
        self._pre_hook.remove()
        self._active_o = None
        self._closed = True

    def __enter__(self) -> "OSourceHeadAttachment":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def attach_o_source_head(model: nn.Module, generator: nn.Module | None = None,
                         use_real_o: bool = False,
                         *, trace_sink: Callable[[dict[str, Any]], None] | None = None) -> OSourceHeadAttachment:
    """Attach the zero-initialized native RAM O source-head adapter.

    The caller must invoke this *after* loading the strict C0 model state.
    ``model.o_source_adapter`` is registered after the existing backbone, so
    the original parameter names/order remain a prefix of ``named_parameters``.
    ``use_real_o`` selects ``sigmoid(generator(image))``; ``False`` is the
    matched zero-O control and does not call ``generator``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if use_real_o and generator is None:
        raise ValueError("use_real_o=True requires an image-only generator")
    if generator is not None and not isinstance(generator, nn.Module):
        raise TypeError("generator must be a torch.nn.Module or None")
    head = _decoder_stage4_head(model)
    return OSourceHeadAttachment(model, generator, use_real_o, head, trace_sink)


__all__ = [
    "ADAPTER_NAME",
    "FEATURE_CHANNELS",
    "MEMBER_CHANNELS",
    "OSourceHeadAttachment",
    "attach_o_source_head",
]
