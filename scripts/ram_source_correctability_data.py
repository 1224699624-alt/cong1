"""Deterministic RAM source-correctability data preparation.

This module deliberately keeps the R325 ``NativeWristDataset`` preprocessing
contract local to this file.  The legacy dataset is not modified: an H0 mask
is loaded from a per-case oracle file, transformed with the same spatial
operations and random draws as the image/mask pair, and padded at the end.

The schedule builder runs the unchanged C0 ``choose_pair_instances`` function
once per augmented case, records the resulting four member IDs and the C0
local-loss denominator, and stores CPU tensors in replayable ``.pt`` files.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


ORACLE_KEYS = ("shape", "original_hw", "h0")
SCHEDULE_VERSION = "ram_source_correctability_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor | np.ndarray) -> str:
    """Hash tensor contents exactly as stored in the replay payload."""

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    return _sha256_bytes(array.tobytes())


def batch_sha256(image: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor) -> str:
    """Match the existing C0 trace hash: image, mask, then valid raw bytes."""

    digest = hashlib.sha256()
    for tensor in (image, mask, valid):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rng_snapshot() -> dict[str, Any]:
    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": cuda_states,
    }


def _restore_rng(snapshot: Mapping[str, Any]) -> None:
    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch"])
    if snapshot.get("cuda") is not None:
        torch.cuda.set_rng_state_all(snapshot["cuda"])


def _rng_digest(snapshot: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(snapshot["python"]).encode("utf-8"))
    numpy_state = snapshot["numpy"]
    digest.update(repr(numpy_state[0]).encode("utf-8"))
    digest.update(str(numpy_state[1].dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(numpy_state[1]).tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
    digest.update(snapshot["torch"].detach().cpu().contiguous().numpy().tobytes())
    for state in snapshot.get("cuda") or []:
        digest.update(state.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


class NativeWristH0Dataset(Dataset):
    """R325-compatible native RAM samples with a fixed transformed H0 mask.

    The oracle contract is ``h0_dir/<split>/<case>.npz`` with:

    * ``shape``: integer ``[14, padded_h, padded_w]`` shape of the mask;
    * ``h0``: ``numpy.packbits`` uint8 payload, little-endian bit order;
    * ``original_hw``: integer ``[height, width]`` before right/bottom padding.

    H0 was saved by the oracle after the canonical right-wrist flip. It is
    cropped to ``original_hw`` and receives only the same augmentation as the
    already-flipped legacy mask. It is padded alongside the legacy mask and
    is returned as ``sample["h0"]`` with dtype ``torch.bool``.
    """

    def __init__(self, root: Path, split: str, augment: bool,
                 h0_dir: Path, divisor: int = 32) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.augment = bool(augment)
        self.h0_dir = Path(h0_dir)
        self.divisor = int(divisor)
        if self.divisor <= 0:
            raise ValueError("divisor must be positive")
        self.mask_files = sorted(
            (self.root / "BoneSegmentation" / "masks" / self.split).glob("*.npy")
        )
        if not self.mask_files:
            raise RuntimeError(f"No masks for split {self.split}: {self.root}")

    def __len__(self) -> int:
        return len(self.mask_files)

    def _load_h0(self, case: str, height: int, width: int) -> torch.Tensor:
        path = self.h0_dir / self.split / f"{case}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing H0 oracle for {self.split}/{case}: {path}")
        with np.load(path, allow_pickle=False) as archive:
            missing = [key for key in ORACLE_KEYS if key not in archive]
            if missing:
                # Keep the initial provisional contract readable for old local
                # fixtures, while all newly generated oracle files use the
                # packed ``shape``/``h0`` contract above.
                if "H0" not in archive or "original_hw" not in archive:
                    raise ValueError(f"H0 oracle missing keys {missing}: {path}")
                h0_array = np.asarray(archive["H0"])
            else:
                shape = tuple(int(value) for value in np.asarray(archive["shape"]).reshape(-1).tolist())
                if len(shape) != 3:
                    raise ValueError(f"oracle shape must have 3 entries, got {shape}: {path}")
                packed = np.asarray(archive["h0"], dtype=np.uint8).reshape(-1)
                unpacked = np.unpackbits(packed, bitorder="little")
                count = int(np.prod(shape, dtype=np.int64))
                if unpacked.size < count:
                    raise ValueError({"path": str(path), "packed_bits": int(unpacked.size), "required": count})
                h0_array = unpacked[:count].reshape(shape).astype(np.bool_, copy=False)
            original_hw = np.asarray(archive["original_hw"])
        if h0_array.dtype != np.bool_:
            raise ValueError(f"H0 must be bool, got {h0_array.dtype}: {path}")
        if h0_array.ndim != 3 or h0_array.shape[0] != 14:
            raise ValueError(f"H0 must have shape [14,H,W], got {h0_array.shape}: {path}")
        if original_hw.shape != (2,):
            raise ValueError(f"original_hw must have shape [2], got {original_hw.shape}: {path}")
        oracle_h, oracle_w = (int(original_hw[0]), int(original_hw[1]))
        if (oracle_h, oracle_w) != (height, width):
            raise ValueError({
                "case": case,
                "image_mask_hw": [height, width],
                "oracle_original_hw": [oracle_h, oracle_w],
                "path": str(path),
            })
        if h0_array.shape[1] < height or h0_array.shape[2] < width:
            raise ValueError({
                "case": case,
                "h0_shape": list(h0_array.shape),
                "required_original_hw": [height, width],
                "path": str(path),
            })
        return torch.from_numpy(h0_array[:, :height, :width].copy())

    def __getitem__(self, index: int) -> dict[str, object]:
        mask_path = self.mask_files[index]
        case = mask_path.stem
        image_path = self.root / "BoneSegmentation" / "images" / f"{case}.bmp"
        image = Image.open(image_path).convert("L")
        mask_array = (np.load(mask_path) > 0).astype(np.float32)
        if mask_array.shape[-2:] != (image.height, image.width):
            raise RuntimeError({"case": case, "image": image.size, "mask": mask_array.shape})

        height, width = int(image.height), int(image.width)
        image_t = TF.pil_to_tensor(image).float() / 255.0
        mask = torch.from_numpy(mask_array)
        h0 = self._load_h0(case, height, width)

        # Keep this order and these random calls byte-for-byte aligned with
        # NativeWristDataset.__getitem__.
        if mask_path.stem.endswith("_R"):
            image_t = torch.flip(image_t, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        if self.augment:
            angle = random.uniform(-7.0, 7.0)
            translate = [random.randint(-10, 10), random.randint(-10, 10)]
            scale = random.uniform(0.94, 1.06)
            image_t = TF.affine(image_t, angle, translate, scale, 0.0,
                                interpolation=InterpolationMode.BILINEAR, fill=0.0)
            mask = TF.affine(mask, angle, translate, scale, 0.0,
                             interpolation=InterpolationMode.NEAREST, fill=0.0)
            h0 = TF.affine(h0, angle, translate, scale, 0.0,
                           interpolation=InterpolationMode.NEAREST, fill=0.0)
            image_t = TF.adjust_gamma(image_t.clamp(0, 1), random.uniform(0.85, 1.15))
            image_t = (image_t * random.uniform(0.90, 1.10)).clamp(0, 1)

        image_t = (image_t - image_t.mean()) / (image_t.std() + 1e-6)
        height, width = image_t.shape[-2:]
        padded_h = ((height + self.divisor - 1) // self.divisor) * self.divisor
        padded_w = ((width + self.divisor - 1) // self.divisor) * self.divisor
        padding = (0, padded_w - width, 0, padded_h - height)
        image_t = F.pad(image_t, padding, value=0.0)
        mask = F.pad(mask, padding, value=0.0)
        h0 = F.pad(h0, padding, value=False).bool()
        valid = torch.zeros((1, padded_h, padded_w), dtype=torch.bool)
        valid[:, :height, :width] = True
        return {
            "image": image_t,
            "mask": mask,
            "valid": valid,
            "h0": h0,
            "original_hw": torch.tensor([height, width], dtype=torch.int64),
            "case": case,
        }


def _choose_pair_instances():
    # Import lazily so dataset-only users do not instantiate the full R330
    # dependency graph. This is the unchanged C0 function, not a copy.
    from train_r330_ram_pair_surface_volume_prior import choose_pair_instances

    return choose_pair_instances


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping):
        if "logits" not in output:
            raise ValueError("C0 forward mapping has no 'logits' key")
        output = output["logits"]
    elif isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("C0 forward returned an empty sequence")
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[1] != 14:
        raise ValueError(f"C0 logits must be [B,14,H,W], got {type(output)} {getattr(output, 'shape', None)}")
    return output


def _c0_forward(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Call the existing C0 model once without changing its architecture."""

    first_conv = next((module for module in model.modules() if isinstance(module, nn.Conv2d)), None)
    if first_conv is not None and first_conv.in_channels == 2:
        output = model(torch.cat((image, torch.zeros_like(image)), dim=1))
    else:
        parameters = list(inspect.signature(model.forward).parameters.values())
        required = [p for p in parameters if p.default is inspect.Parameter.empty]
        if len(required) >= 2:
            output = model(image, False)
        else:
            output = model(image)
    return _extract_logits(output)


def _model_device(model: nn.Module) -> torch.device:
    for parameter in model.parameters():
        return parameter.device
    for buffer in model.buffers():
        return buffer.device
    return torch.device("cpu")


def _selected_denominator(probability: torch.Tensor, target: torch.Tensor,
                          indices: torch.Tensor) -> float:
    selected_target = target[0, indices].unsqueeze(1)
    initial_error = torch.abs(probability[0, indices].unsqueeze(1) - selected_target)
    local = F.max_pool2d((initial_error > 0.15).float(), 11, 1, 5)
    return float((1.0 + 3.0 * local).sum().detach().cpu())


def _order_from_seed(length: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(length, generator=generator).tolist()


def build_deterministic_schedule(
    dataset: NativeWristH0Dataset,
    model: nn.Module,
    output_dir: Path,
    seed: int,
    case_order: Sequence[int] | None = None,
    selected_count: int = 4,
) -> dict[str, Any]:
    """Build a fixed augmented C0 schedule and CPU replay payloads.

    The caller supplies the already-loaded C0 model.  The model is evaluated
    exactly once per schedule step; no optimizer or backward pass occurs.
    Python/NumPy/Torch RNG states are restored before return, so schedule
    construction cannot perturb a surrounding training run.
    """

    if selected_count != 4:
        raise ValueError("The source-correctability schedule requires selected_count=4")
    output_dir = Path(output_dir)
    payload_dir = output_dir / "steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_dir.mkdir(parents=True, exist_ok=True)
    if case_order is None:
        order = _order_from_seed(len(dataset), seed)
    else:
        # An explicit locked schedule may contain a small subset and repeated
        # exposures. This is required for the fixed H0/update-set protocol.
        order = [int(index) for index in case_order]
        if not order or any(index < 0 or index >= len(dataset) for index in order):
            raise ValueError("case_order must be a nonempty sequence of valid dataset indices")

    choose_pair_instances = _choose_pair_instances()
    before_rng = _rng_snapshot()
    original_training = bool(model.training)
    device = _model_device(model)
    records: list[dict[str, Any]] = []
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) & 0xFFFFFFFF)
        torch.manual_seed(int(seed))
        model.eval()
        with torch.inference_mode():
            for step, index in enumerate(order):
                before_aug = _rng_snapshot()
                sample = dataset[index]
                after_aug = _rng_snapshot()
                image = sample["image"]
                mask = sample["mask"]
                valid = sample["valid"]
                h0 = sample["h0"]
                if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 1:
                    raise ValueError("dataset image must be [1,H,W]")
                if not isinstance(mask, torch.Tensor) or mask.shape[:1] != (14,):
                    raise ValueError("dataset mask must be [14,H,W]")
                if not isinstance(valid, torch.Tensor) or valid.shape[0] != 1:
                    raise ValueError("dataset valid must be [1,H,W]")
                if not isinstance(h0, torch.Tensor) or h0.dtype != torch.bool or h0.shape != mask.shape:
                    raise ValueError("dataset h0 must be bool [14,H,W]")

                image_batch = image.unsqueeze(0).to(device)
                target_batch = mask.unsqueeze(0).to(device)
                # The mature training loss evaluates source/error/local in AMP.
                # Match that numerical path when freezing its selected members
                # and denominator, while the oracle H0 itself stays FP32.
                with torch.cuda.amp.autocast(enabled=image_batch.is_cuda):
                    logits = _c0_forward(model, image_batch)
                    probability = logits.sigmoid()
                    indices = choose_pair_instances(probability, target_batch, selected_count)
                    denominator = _selected_denominator(probability, target_batch, indices)
                after_forward = _rng_snapshot()

                payload = {
                    "image": image.detach().cpu().contiguous(),
                    "mask": mask.detach().cpu().contiguous(),
                    "valid": valid.detach().cpu().contiguous().bool(),
                    "h0": h0.detach().cpu().contiguous().bool(),
                }
                payload_path = payload_dir / f"step_{step:06d}.pt"
                torch.save(payload, payload_path)
                relative_path = payload_path.relative_to(output_dir).as_posix()
                records.append({
                    "step": step,
                    "dataset_index": int(index),
                    "path": relative_path,
                    "case": str(sample["case"]),
                    "batch_sha256": batch_sha256(payload["image"], payload["mask"], payload["valid"]),
                    "image_sha256": tensor_sha256(payload["image"]),
                    "mask_sha256": tensor_sha256(payload["mask"]),
                    "valid_sha256": tensor_sha256(payload["valid"]),
                    "h0_sha256": tensor_sha256(payload["h0"]),
                    "indices": [int(value) for value in indices.detach().cpu().tolist()],
                    "denominator_c0": denominator,
                    "rng_before_aug_sha256": _rng_digest(before_aug),
                    "rng_after_aug_sha256": _rng_digest(after_aug),
                    "rng_after_c0_forward_sha256": _rng_digest(after_forward),
                    "shapes": {
                        "image": list(payload["image"].shape),
                        "mask": list(payload["mask"].shape),
                        "valid": list(payload["valid"].shape),
                        "h0": list(payload["h0"].shape),
                    },
                })
    finally:
        model.train(original_training)
        _restore_rng(before_rng)

    manifest = {
        "schedule_version": SCHEDULE_VERSION,
        "split": dataset.split,
        "dataset_root": str(dataset.root),
        "h0_dir": str(dataset.h0_dir),
        "seed": int(seed),
        "selected_count": 4,
        "updates": len(records),
        "case_order": [record["case"] for record in records],
        "dataset_indices": [record["dataset_index"] for record in records],
        "unique_case_count": len({record["case"] for record in records}),
        "repeated_case_count": len(records) - len({record["case"] for record in records}),
        "c0_forward_calls": len(records),
        "choose_pair_instances": "train_r330_ram_pair_surface_volume_prior.choose_pair_instances",
        "h0_oracle_contract": {
            "relative_path": "<h0_dir>/<split>/<case>.npz",
            "keys": ["shape", "original_hw", "h0"],
            "shape": "[14,padded_h,padded_w] int",
            "h0": "numpy.packbits(..., bitorder='little') uint8",
        },
        "rng_before_sha256": _rng_digest(before_rng),
        "rng_after_restore_sha256": _rng_digest(_rng_snapshot()),
        "steps": records,
    }
    manifest_path = output_dir / "schedule.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class SourceCorrectabilityReplayDataset(Dataset):
    """Simple no-RNG replay view over a schedule manifest."""

    def __init__(self, manifest: Path | Mapping[str, Any], verify_hashes: bool = True) -> None:
        self.verify_hashes = bool(verify_hashes)
        if isinstance(manifest, (str, Path)):
            self.manifest_path = Path(manifest)
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest_path = None
            self.manifest = dict(manifest)
        self.steps = list(self.manifest.get("steps", []))
        if not self.steps:
            raise ValueError("schedule has no steps")
        self.root = self.manifest_path.parent if self.manifest_path is not None else Path.cwd()

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.steps[index]
        path = Path(record["path"])
        if not path.is_absolute():
            path = self.root / path
        payload = _load_torch(path)
        required = {"image", "mask", "valid", "h0"}
        if not required.issubset(payload):
            raise ValueError(f"replay payload missing {required - set(payload)}: {path}")
        image, mask, valid, h0 = (payload[key] for key in ("image", "mask", "valid", "h0"))
        if image.shape[0] != 1 or mask.shape[0] != 14 or valid.shape[0] != 1 or h0.shape != mask.shape:
            raise ValueError(f"invalid replay shapes in {path}")
        if valid.dtype != torch.bool or h0.dtype != torch.bool:
            raise ValueError(f"valid/h0 must be bool in {path}")
        if self.verify_hashes:
            checks = {
                "batch_sha256": batch_sha256(image, mask, valid),
                "image_sha256": tensor_sha256(image),
                "mask_sha256": tensor_sha256(mask),
                "valid_sha256": tensor_sha256(valid),
                "h0_sha256": tensor_sha256(h0),
            }
            for key, actual in checks.items():
                if actual != record.get(key):
                    raise ValueError({"step": index, "field": key, "expected": record.get(key), "actual": actual})
        return {
            "image": image,
            "mask": mask,
            "valid": valid,
            "h0": h0,
            "case": str(record["case"]),
            "indices": torch.tensor(record["indices"], dtype=torch.long),
            "denominator_c0": float(record["denominator_c0"]),
        }


# Short aliases for callers that prefer the protocol terminology.
ReplayDataset = SourceCorrectabilityReplayDataset
build_schedule = build_deterministic_schedule


__all__ = [
    "NativeWristH0Dataset",
    "SourceCorrectabilityReplayDataset",
    "ReplayDataset",
    "batch_sha256",
    "tensor_sha256",
    "build_deterministic_schedule",
    "build_schedule",
]
