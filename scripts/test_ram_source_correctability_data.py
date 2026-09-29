"""CPU tests for ``ram_source_correctability_data``.

The test uses the first two local RAM cases when available.  Since oracle files
are produced outside this repository, an isolated packed-H0 fixture is created
for those cases when no ``--h0-dir`` is supplied.  If the RAM data is absent,
the same checks run on two synthetic cases, including a right-wrist case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ram_source_correctability_data import (  # noqa: E402
    NativeWristH0Dataset,
    SourceCorrectabilityReplayDataset,
    batch_sha256,
    build_deterministic_schedule,
    tensor_sha256,
)
from train_r325_ram_nnunet_native_resolution_overlap_iem import (  # noqa: E402
    NativeWristDataset,
)


def _state_digest() -> str:
    numpy_state = np.random.get_state()
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))
    digest.update(repr(numpy_state[0]).encode("utf-8"))
    digest.update(str(numpy_state[1].dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(numpy_state[1]).tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
    digest.update(torch.get_rng_state().numpy().tobytes())
    return digest.hexdigest()


def _write_packed_oracle(h0_dir: Path, split: str, mask_path: Path) -> None:
    mask = np.load(mask_path) > 0
    height, width = (int(mask.shape[-2]), int(mask.shape[-1]))
    padded_h = ((height + 31) // 32) * 32
    padded_w = ((width + 31) // 32) * 32
    h0 = np.zeros((14, padded_h, padded_w), dtype=np.bool_)
    # A proper subset exercises the transformed-target relationship without
    # pretending that this fixture is a scientific oracle.
    h0[:, :height // 2, :width // 2] = mask[:, :height // 2, :width // 2]
    # The real oracle stores H0 after the dataset's canonical right flip.
    if mask_path.stem.endswith("_R"):
        h0[:, :height, :width] = h0[:, :height, :width][..., ::-1]
    path = h0_dir / split / f"{mask_path.stem}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        shape=np.asarray(h0.shape, dtype=np.int64),
        original_hw=np.asarray([height, width], dtype=np.int64),
        h0=np.packbits(h0.reshape(-1), bitorder="little"),
        source_below=np.packbits(np.zeros(h0.size, dtype=np.bool_), bitorder="little"),
    )


def _make_synthetic_root(root: Path) -> list[Path]:
    masks_dir = root / "BoneSegmentation" / "masks" / "train"
    images_dir = root / "BoneSegmentation" / "images"
    masks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    cases = [("synthetic_0001_L", 35, 37), ("synthetic_0002_R", 41, 29)]
    paths = []
    for case, height, width in cases:
        yy, xx = np.mgrid[:height, :width]
        image = ((xx * 7 + yy * 11) % 256).astype(np.uint8)
        mask = np.zeros((14, height, width), dtype=np.int8)
        for member in range(14):
            mask[member] = (((xx + member * 2) % 17 == 0) & ((yy + member) % 13 < 5)).astype(np.int8)
        Image.fromarray(image, mode="L").save(images_dir / f"{case}.bmp")
        mask_path = masks_dir / f"{case}.npy"
        np.save(mask_path, mask)
        paths.append(mask_path)
    return paths


class ZeroC0(nn.Module):
    """Deterministic 14-channel CPU C0 stand-in for schedule-unit testing."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 14, kernel_size=1, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.forward_calls = 0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return self.projection(image)


def _assert_sample_parity(legacy: dict[str, object], enhanced: dict[str, object]) -> None:
    for key in ("image", "mask", "valid"):
        assert tensor_sha256(legacy[key]) == tensor_sha256(enhanced[key]), key
        assert torch.equal(legacy[key], enhanced[key]), key
    h0 = enhanced["h0"]
    mask = enhanced["mask"]
    valid = enhanced["valid"].expand_as(h0)
    assert h0.dtype == torch.bool
    assert bool((h0 & ~valid).any()) is False
    assert bool((h0 & ~(mask > 0.5)).any()) is False
    assert torch.equal(legacy["original_hw"], enhanced["original_hw"])
    assert legacy["case"] == enhanced["case"]


def _compare_first_two(root: Path, h0_dir: Path, split: str = "train") -> dict[str, object]:
    legacy = NativeWristDataset(root, split, augment=True)
    enhanced = NativeWristH0Dataset(root, split, augment=True, h0_dir=h0_dir)
    assert len(legacy) == len(enhanced)
    rows = []
    for index in range(min(2, len(legacy))):
        seed = 9142 + index
        random.seed(seed)
        before_legacy = _state_digest()
        legacy_item = legacy[index]
        after_legacy = _state_digest()
        random.seed(seed)
        before_enhanced = _state_digest()
        enhanced_item = enhanced[index]
        after_enhanced = _state_digest()
        assert before_legacy == before_enhanced
        assert after_legacy == after_enhanced
        _assert_sample_parity(legacy_item, enhanced_item)
        rows.append({
            "index": index,
            "case": enhanced_item["case"],
            "batch_sha256": batch_sha256(
                enhanced_item["image"], enhanced_item["mask"], enhanced_item["valid"]
            ),
            "h0_sha256": tensor_sha256(enhanced_item["h0"]),
            "rng_before_after_equal": True,
        })
    return {"cases": rows, "count": len(rows)}


def _test_schedule(root: Path, h0_dir: Path) -> dict[str, object]:
    dataset = NativeWristH0Dataset(root, "train", augment=True, h0_dir=h0_dir)
    # Keep the schedule test small while retaining the real preprocessing path.
    dataset.mask_files = dataset.mask_files[:2]
    locked_order = [0, 1, 0, 1]
    c0_a = ZeroC0()
    c0_b = ZeroC0()
    with tempfile.TemporaryDirectory(prefix="ram_schedule_a_") as first_tmp, tempfile.TemporaryDirectory(prefix="ram_schedule_b_") as second_tmp:
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        before = _state_digest()
        first = build_deterministic_schedule(dataset, c0_a, Path(first_tmp), seed=4421, case_order=locked_order)
        after = _state_digest()
        assert before == after, "schedule builder changed caller RNG state"
        second = build_deterministic_schedule(dataset, c0_b, Path(second_tmp), seed=4421, case_order=locked_order)
        assert c0_a.forward_calls == 4 and c0_b.forward_calls == 4
        assert first["updates"] == 4 and first["unique_case_count"] == 2
        assert first["case_order"] == second["case_order"]
        assert first["dataset_indices"] == second["dataset_indices"]
        for left, right in zip(first["steps"], second["steps"]):
            for key in ("case", "batch_sha256", "image_sha256", "mask_sha256", "valid_sha256", "h0_sha256", "indices"):
                assert left[key] == right[key], key
            assert left["denominator_c0"] == right["denominator_c0"]
        replay = SourceCorrectabilityReplayDataset(Path(first_tmp) / "schedule.json")
        replay_rows = []
        for index in range(len(replay)):
            sample = replay[index]
            record = first["steps"][index]
            assert batch_sha256(sample["image"], sample["mask"], sample["valid"]) == record["batch_sha256"]
            assert sample["indices"].tolist() == record["indices"]
            assert sample["denominator_c0"] == record["denominator_c0"]
            replay_rows.append({"step": index, "case": sample["case"], "indices": sample["indices"].tolist()})
        return {
            "steps": len(replay),
            "forward_calls": c0_a.forward_calls,
            "rng_restored": True,
            "replay": replay_rows,
        }


def run_tests(dataset_root: Path, h0_dir: Path | None, threads: int = 64) -> dict[str, object]:
    torch.set_num_threads(int(threads))
    dataset_root = Path(dataset_root)
    real_data = (dataset_root / "BoneSegmentation" / "masks" / "train").is_dir()
    with tempfile.TemporaryDirectory(prefix="ram_h0_test_") as tmp:
        tmp_root = Path(tmp)
        if real_data:
            root = dataset_root
            if h0_dir is None:
                fixture_h0 = tmp_root / "oracle" / "predictions" / "h0"
                mask_files = sorted((root / "BoneSegmentation" / "masks" / "train").glob("*.npy"))[:2]
                for path in mask_files:
                    _write_packed_oracle(fixture_h0, "train", path)
                h0_root = fixture_h0
                oracle_source = "isolated_packed_fixture_for_local_ram"
            else:
                h0_root = Path(h0_dir)
                oracle_source = "caller_oracle"
        else:
            root = tmp_root / "synthetic"
            mask_files = _make_synthetic_root(root)
            h0_root = Path(h0_dir) if h0_dir is not None else tmp_root / "oracle" / "predictions" / "h0"
            if h0_dir is None:
                for path in mask_files:
                    _write_packed_oracle(h0_root, "train", path)
                oracle_source = "synthetic_packed_fixture"
            else:
                oracle_source = "caller_oracle"
        parity = _compare_first_two(root, h0_root)
        schedule = _test_schedule(root, h0_root)
        return {
            "status": "PASS",
            "threads": int(torch.get_num_threads()),
            "data_mode": "real_ram_first2" if real_data else "synthetic_fallback",
            "oracle_source": oracle_source,
            "parity": parity,
            "schedule": schedule,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "remote_variants" / "RAM-W600",
    )
    parser.add_argument("--h0-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=64)
    args = parser.parse_args()
    result = run_tests(args.dataset_root, args.h0_dir, args.threads)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
