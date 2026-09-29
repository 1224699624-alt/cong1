"""RAM S/O conditional correction after the mature R332 step-2 output.

This file contains only the frozen-feature interface and the new shared head.
It deliberately does not load a checkpoint, call a mature model, or perform an
optimizer step.  A caller runs the mature nnU-Net/R332 path first, detaches its
``z2`` and second-step ``dec1`` features, and then calls
``RAMSOConditionalCorrector``.

The input order is fixed and mirrors the legacy R332 function exactly::

    [per_instance_features(...), z2_k, dec1_step2_hidden_k, S, O]
    [             6 channels,       1,            12,          1, 1]

S and O are independent soft maps.  ``image_extent`` is metadata only; it is
never concatenated into the head input and is not treated as a label.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import sys
from pathlib import Path
from numbers import Integral
from typing import Any

import torch
from torch import Tensor, nn

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


LEGACY_FEATURE_CHANNELS = 6
Z2_CHANNELS = 1
R332_HIDDEN_CHANNELS = 12
PRIOR_CHANNELS = 2
HEAD_INPUT_CHANNELS = (
    LEGACY_FEATURE_CHANNELS
    + Z2_CHANNELS
    + R332_HIDDEN_CHANNELS
    + PRIOR_CHANNELS
)
DEFAULT_HEAD_WIDTH = 32
FEATURE_ORDER = (
    "image",
    "own_probability",
    "other_max",
    "other_sum_clamped_div2",
    "uncertainty",
    "image_gradient",
    "z2_logit",
    "r332_dec1_step2_hidden",
    "S",
    "O",
)
ARMS = ("C00", "CS", "CO", "CSO")
_REQUIRED_PRIORS = {
    "C00": (),
    "CS": ("s",),
    "CO": ("o",),
    "CSO": ("s", "o"),
}


def _canonical_prior_key(key: str) -> str:
    value = str(key).strip().lower()
    if value not in {"s", "o"}:
        raise ValueError(f"prior presence keys must be S/O, got {key!r}")
    return value


def _normalize_extent(extent: Any) -> tuple[int, int] | None:
    if extent is None:
        return None
    if isinstance(extent, torch.Size):
        values = tuple(extent)
    elif isinstance(extent, Sequence) and not isinstance(extent, (str, bytes)):
        values = tuple(extent)
    else:
        raise TypeError("image_extent must be a (height, width) sequence")
    if len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise ValueError("image_extent must contain two integer dimensions")
    result = (int(values[0]), int(values[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError(f"image_extent must be positive, got {result}")
    return result


@dataclass(frozen=True)
class PriorMaps:
    """Independent S/O maps and provenance metadata for one image batch.

    ``s`` and ``o`` may be ``None`` only when an arm does not require that
    prior.  This makes a missing required map fail loudly instead of silently
    replacing it with a zero map.  Presence and hashes are metadata; they do
    not alter the forward computation.
    """

    s: Tensor | None = None
    o: Tensor | None = None
    image_extent: tuple[int, int] | None = None
    presence: Mapping[str, bool] = field(default_factory=dict)
    metadata_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        extent = _normalize_extent(self.image_extent)
        object.__setattr__(self, "image_extent", extent)

        normalized_presence: dict[str, bool] = {}
        for key, value in dict(self.presence).items():
            canonical = _canonical_prior_key(key)
            if not isinstance(value, bool):
                raise TypeError(f"presence[{key!r}] must be bool")
            normalized_presence[canonical] = value
        for key, value in (("s", self.s), ("o", self.o)):
            inferred = value is not None
            if key in normalized_presence and normalized_presence[key] != inferred:
                raise ValueError(
                    f"presence[{key!r}]={normalized_presence[key]} disagrees with map availability={inferred}"
                )
            normalized_presence.setdefault(key, inferred)
        object.__setattr__(self, "presence", normalized_presence)

        hashes = dict(self.metadata_hashes)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in hashes.items()):
            raise TypeError("metadata_hashes must map string names to string hashes")
        object.__setattr__(self, "metadata_hashes", hashes)

    @property
    def S(self) -> Tensor | None:
        """Uppercase alias used by experiment records."""

        return self.s

    @property
    def O(self) -> Tensor | None:
        """Uppercase alias used by experiment records."""

        return self.o

    @property
    def s_map(self) -> Tensor | None:
        return self.s

    @property
    def o_map(self) -> Tensor | None:
        return self.o

    @classmethod
    def from_tensors(
        cls,
        s: Tensor | None,
        o: Tensor | None,
        *,
        image_extent: Sequence[int] | None = None,
        presence: Mapping[str, bool] | None = None,
        metadata_hashes: Mapping[str, str] | None = None,
    ) -> "PriorMaps":
        return cls(
            s=s,
            o=o,
            image_extent=None if image_extent is None else tuple(image_extent),
            presence={} if presence is None else presence,
            metadata_hashes={} if metadata_hashes is None else metadata_hashes,
        )


def _require_floating_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating dtype, got {value.dtype}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinity")


def _validate_soft_map(
    name: str,
    value: Tensor,
    expected_shape: tuple[int, int, int, int],
) -> None:
    _require_floating_finite(name, value)
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(value.shape)}")
    minimum = float(value.detach().amin())
    maximum = float(value.detach().amax())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{name} must be a soft map in [0, 1], got range [{minimum}, {maximum}]")


def _normalize_arm(arm: str) -> str:
    if not isinstance(arm, str):
        raise TypeError(f"arm must be one of {ARMS}, got {type(arm).__name__}")
    value = arm.upper()
    if value not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return value


def _normalize_indices(indices: Tensor | Sequence[int] | None, members: int, device: torch.device) -> Tensor:
    if indices is None:
        result = torch.arange(members, dtype=torch.long, device=device)
    elif isinstance(indices, Tensor):
        if indices.ndim != 1:
            raise ValueError(f"indices must be one-dimensional, got {tuple(indices.shape)}")
        if indices.dtype == torch.bool or indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError(f"indices must contain integers, got {indices.dtype}")
        result = indices.to(device=device, dtype=torch.long)
    else:
        if isinstance(indices, (str, bytes)):
            raise TypeError("indices must be an integer sequence")
        try:
            raw = list(indices)
        except TypeError as exc:
            raise TypeError("indices must be an integer sequence") from exc
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
            raise TypeError("indices must contain integers")
        result = torch.tensor(raw, dtype=torch.long, device=device)
    if result.numel() == 0:
        raise ValueError("indices cannot be empty")
    values = result.detach().cpu().tolist()
    if len(set(values)) != len(values):
        raise ValueError(f"indices must be unique, got {values}")
    if min(values) < 0 or max(values) >= members:
        raise ValueError(f"indices must lie in [0, {members}), got {values}")
    # Keep caller order exactly.  The legacy R332 feature function uses this
    # order to build the own/other-member rows; sorting here would break its
    # permutation contract.
    return result


def _normalize_hidden(hidden: Tensor, batch: int, members: int, spatial: tuple[int, int]) -> Tensor:
    _require_floating_finite("r332_dec1_step2_hidden", hidden)
    if hidden.ndim == 4:
        if batch != 1:
            raise ValueError(
                "a [K, 12, H, W] hidden tensor is only valid for batch size one; "
                "use [B, K, 12, H, W] for batched input"
            )
        hidden = hidden.unsqueeze(0)
    elif hidden.ndim != 5:
        raise ValueError(
            "r332_dec1_step2_hidden must have shape [K, 12, H, W] or [B, K, 12, H, W]"
        )
    expected = (batch, members, R332_HIDDEN_CHANNELS, *spatial)
    if tuple(hidden.shape) != expected:
        raise ValueError(f"r332_dec1_step2_hidden must have shape {expected}, got {tuple(hidden.shape)}")
    return hidden


def _validate_prior_maps(
    prior_maps: PriorMaps,
    arm: str,
    batch: int,
    spatial: tuple[int, int],
) -> tuple[Tensor | None, Tensor | None]:
    if not isinstance(prior_maps, PriorMaps):
        raise TypeError("prior_maps must be a PriorMaps instance")
    expected_shape = (batch, 1, *spatial)
    # Native image extent may be smaller than the padded tensor canvas.
    # It is image geometry metadata, not a label-known mask or head feature.
    if prior_maps.image_extent is not None and any(
        native > padded for native, padded in zip(prior_maps.image_extent, spatial)
    ):
        raise ValueError(
            f"prior image_extent {prior_maps.image_extent} exceeds padded input spatial extent {spatial}"
        )
    for key, value in (("s", prior_maps.s), ("o", prior_maps.o)):
        if value is not None:
            _validate_soft_map(key.upper(), value, expected_shape)
        elif prior_maps.presence.get(key, False):
            raise ValueError(f"PriorMaps marks {key.upper()} present but supplies no tensor")
    for key in _REQUIRED_PRIORS[arm]:
        value = getattr(prior_maps, key)
        if value is None:
            raise ValueError(f"arm {arm} requires PriorMaps.{key} but it is missing")
    return prior_maps.s, prior_maps.o


def _legacy_features(
    image: Tensor,
    probabilities: Tensor,
    indices: Tensor,
    own_probability: Tensor,
) -> Tensor:
    """Call the repository's exact six-channel R332 feature function."""

    try:
        from train_r327_ram_native_instance_completion_prior import per_instance_features
    except ImportError as exc:  # pragma: no cover - only an installation/path error
        try:
            from scripts.train_r327_ram_native_instance_completion_prior import per_instance_features
        except ImportError:
            raise ImportError(
                "the legacy R332 feature function could not be imported; run from scripts/ "
                "or add scripts/ to PYTHONPATH"
            ) from exc
    value = per_instance_features(image, probabilities, indices, own_probability)
    if value.ndim != 4 or value.shape[1] != LEGACY_FEATURE_CHANNELS:
        raise RuntimeError(
            "legacy per_instance_features changed its contract: "
            f"expected [K, 6, H, W], got {tuple(value.shape)}"
        )
    return value


class SharedConditionalCorrectionHead(nn.Module):
    """Shared post-R332-step2 ``21 -> 32 -> 1`` pointwise correction head."""

    input_channels = HEAD_INPUT_CHANNELS
    hidden_channels = DEFAULT_HEAD_WIDTH

    def __init__(self, input_channels: int = HEAD_INPUT_CHANNELS, width: int = DEFAULT_HEAD_WIDTH) -> None:
        super().__init__()
        if input_channels != HEAD_INPUT_CHANNELS:
            raise ValueError(f"input_channels is fixed at {HEAD_INPUT_CHANNELS}, got {input_channels}")
        if isinstance(width, bool) or not isinstance(width, Integral) or width <= 0:
            raise ValueError(f"width must be a positive integer, got {width!r}")
        self.input_projection = nn.Conv2d(input_channels, int(width), kernel_size=1)
        self.activation = nn.ReLU(inplace=False)
        self.final_layer = nn.Conv2d(int(width), 1, kernel_size=1)
        # The new path is an exact identity at construction.  Only the newly
        # added final layer is zeroed; input_projection keeps normal Conv2d
        # initialization so its first backward gradient is intentionally zero
        # because final_layer is zero, rather than because both layers were
        # silently reset.
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    @property
    def output_layer(self) -> nn.Conv2d:
        """Readable alias without registering the same module twice."""

        return self.final_layer

    @property
    def new_final(self) -> nn.Conv2d:
        """Alias used by white-box checks for the newly added final layer."""

        return self.final_layer

    @property
    def new_final_layer(self) -> nn.Conv2d:
        return self.final_layer

    @property
    def final(self) -> nn.Conv2d:
        """Short alias for callers that inspect the head layout."""

        return self.final_layer

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1] != self.input_channels:
            raise ValueError(
                f"head expects [N, {self.input_channels}, H, W], got {tuple(value.shape)}"
            )
        _require_floating_finite("conditional head input", value)
        return self.final_layer(self.activation(self.input_projection(value)))


class RAMSOConditionalCorrector(nn.Module):
    """Frozen-input S/O conditional correction around mature R332 ``z2``.

    The module owns only the new shared head.  Mature model/refiner objects are
    intentionally absent: callers provide detached logits and hidden features,
    so the new head cannot create a hidden ``step2 -> source`` loop or update a
    generator through S/O.  No target, H0, Q map, F_high branch, or identity
    table is accepted by this interface.
    """

    def __init__(self, bound: float, *, width: int = DEFAULT_HEAD_WIDTH) -> None:
        super().__init__()
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise TypeError("bound must be a finite positive scalar")
        if not math.isfinite(float(bound)) or float(bound) <= 0.0:
            raise ValueError(f"bound must be finite and positive, got {bound!r}")
        self.bound = float(bound)
        self.head = SharedConditionalCorrectionHead(width=width)

    @property
    def correction_bound(self) -> float:
        return self.bound

    def head_parameters(self):
        """Return only the parameters intended for the new head optimizer."""

        return self.head.parameters()

    def _validate_inputs(
        self,
        image: Tensor,
        z2: Tensor,
        r332_dec1_step2_hidden: Tensor,
        prior_maps: PriorMaps,
        arm: str,
        indices: Tensor | Sequence[int] | None,
        chunk_size: int | None,
    ) -> tuple[str, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        _require_floating_finite("image", image)
        _require_floating_finite("z2", z2)
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"image must have shape [B, 1, H, W], got {tuple(image.shape)}")
        if z2.ndim != 4:
            raise ValueError(f"z2 must have shape [B, K, H, W], got {tuple(z2.shape)}")
        if image.shape[0] != z2.shape[0] or tuple(image.shape[-2:]) != tuple(z2.shape[-2:]):
            raise ValueError("image and z2 must agree in batch and spatial dimensions")
        if image.device != z2.device:
            raise ValueError("image and z2 must be on the same device")
        batch, members, height, width = map(int, z2.shape)
        if members <= 0:
            raise ValueError("z2 must contain at least one member channel")
        spatial = (height, width)
        hidden = _normalize_hidden(r332_dec1_step2_hidden, batch, members, spatial)
        if hidden.device != z2.device:
            raise ValueError("r332_dec1_step2_hidden and z2 must be on the same device")
        arm_value = _normalize_arm(arm)
        s_map, o_map = _validate_prior_maps(prior_maps, arm_value, batch, spatial)
        selected = _normalize_indices(indices, members, z2.device)
        if chunk_size is None:
            chunk = int(selected.numel())
        elif isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        else:
            chunk = int(chunk_size)
        return arm_value, image, z2, hidden, s_map, o_map, chunk

    def forward(
        self,
        image: Tensor,
        z2: Tensor,
        r332_dec1_step2_hidden: Tensor,
        prior_maps: PriorMaps,
        arm: str = "C00",
        indices: Tensor | Sequence[int] | None = None,
        chunk_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(z2 + delta, delta)`` with both tensors shaped like ``z2``.

        ``arm`` gates only the two prior values:

        * C00: S=0, O=0
        * CS:  S=provided, O=0
        * CO:  S=0, O=provided
        * CSO: S=provided, O=provided

        The input tensors are detached before feature construction.  This keeps
        the mature model/refiner and frozen prior generators outside the new
        optimizer graph while preserving gradients through ``self.head``.
        """

        arm_value, image_in, z2_in, hidden_in, s_map, o_map, chunk = self._validate_inputs(
            image, z2, r332_dec1_step2_hidden, prior_maps, arm, indices, chunk_size
        )
        batch, members, height, width = map(int, z2_in.shape)
        selected = _normalize_indices(indices, members, z2_in.device)

        # These are explicitly frozen inputs.  ``detach`` also prevents an
        # accidental caller from making a step2->source feedback edge through
        # this post-R332 module.
        image_frozen = image_in.detach()
        z2_frozen = z2_in.detach()
        hidden_frozen = hidden_in.detach()
        probability = torch.sigmoid(z2_frozen)

        zero = z2_frozen.new_zeros((1, 1, height, width))
        s_active = s_map.detach() if arm_value in {"CS", "CSO"} else zero.expand(batch, -1, -1, -1)
        o_active = o_map.detach() if arm_value in {"CO", "CSO"} else zero.expand(batch, -1, -1, -1)
        if s_active is None or o_active is None:  # defensive; required maps were validated above
            raise RuntimeError(f"arm {arm_value} reached forward without required prior values")
        s_active = s_active.to(device=z2_in.device, dtype=z2_in.dtype)
        o_active = o_active.to(device=z2_in.device, dtype=z2_in.dtype)

        # Build rows as a Python list and stack at the end.  This avoids an
        # in-place scatter into a non-grad tensor, which could hide gradients
        # from the shared head during a first-backward audit.
        per_batch: list[Tensor] = []
        for batch_index in range(batch):
            image_one = image_frozen[batch_index : batch_index + 1]
            probability_one = probability[batch_index : batch_index + 1]
            member_delta: list[Tensor | None] = [None] * members
            for start in range(0, int(selected.numel()), chunk):
                selected_chunk = selected[start : start + chunk]
                own = probability_one[:, selected_chunk]
                legacy = _legacy_features(image_one, probability_one, selected_chunk, own)
                z_rows = z2_frozen[batch_index, selected_chunk].unsqueeze(1)
                hidden_rows = hidden_frozen[batch_index, selected_chunk]
                s_rows = s_active[batch_index : batch_index + 1].expand(selected_chunk.numel(), -1, -1, -1)
                o_rows = o_active[batch_index : batch_index + 1].expand(selected_chunk.numel(), -1, -1, -1)
                head_input = torch.cat((legacy, z_rows, hidden_rows, s_rows, o_rows), dim=1)
                if head_input.shape[1] != HEAD_INPUT_CHANNELS:
                    raise RuntimeError(
                        f"conditional head input must have {HEAD_INPUT_CHANNELS} channels, "
                        f"got {head_input.shape[1]}"
                    )
                raw_delta = self.head(head_input)
                delta_chunk = self.bound * torch.tanh(raw_delta[:, 0])
                if not bool(torch.isfinite(delta_chunk).all()):
                    raise FloatingPointError("conditional correction produced NaN or infinity")
                for row, member_index in enumerate(selected_chunk.detach().cpu().tolist()):
                    member_delta[member_index] = delta_chunk[row]
            zeros = z2_frozen.new_zeros((height, width))
            per_batch.append(torch.stack([value if value is not None else zeros for value in member_delta], dim=0))

        delta = torch.stack(per_batch, dim=0)
        final = z2_frozen + delta
        if tuple(final.shape) != tuple(z2_in.shape) or tuple(delta.shape) != tuple(z2_in.shape):
            raise RuntimeError("conditional corrector changed z2 shape")
        return final, delta

    # A named method is useful in experiment launchers that distinguish a
    # correction call from the module's generic nn.Module.__call__ interface.
    def correct(self, *args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        return self.forward(*args, **kwargs)


# Explicit aliases make the architecture name discoverable without creating
# separate implementations or parameter sets.
PostR332SharedCorrectionHead = SharedConditionalCorrectionHead
ConditionalCorrectionHead = SharedConditionalCorrectionHead
RAMSOConditionalHead = SharedConditionalCorrectionHead
UnifiedConditionalCorrector = RAMSOConditionalCorrector
ConditionalCorrector = RAMSOConditionalCorrector


def apply_conditional_correction(
    corrector: RAMSOConditionalCorrector,
    image: Tensor,
    z2: Tensor,
    r332_dec1_step2_hidden: Tensor,
    prior_maps: PriorMaps,
    *,
    arm: str = "C00",
    indices: Tensor | Sequence[int] | None = None,
    chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Functional wrapper returning the same ``(final_z2, delta)`` tuple."""

    if not isinstance(corrector, RAMSOConditionalCorrector):
        raise TypeError("corrector must be a RAMSOConditionalCorrector")
    return corrector(
        image,
        z2,
        r332_dec1_step2_hidden,
        prior_maps,
        arm=arm,
        indices=indices,
        chunk_size=chunk_size,
    )


__all__ = [
    "ARMS",
    "FEATURE_ORDER",
    "HEAD_INPUT_CHANNELS",
    "PriorMaps",
    "RAMSOConditionalCorrector",
    "SharedConditionalCorrectionHead",
    "PostR332SharedCorrectionHead",
    "ConditionalCorrectionHead",
    "RAMSOConditionalHead",
    "UnifiedConditionalCorrector",
    "ConditionalCorrector",
    "apply_conditional_correction",
]
