"""Focused CPU tests for the read-only RAM high-resolution capture module."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ram_highres_capture import capture_case, forward_c0, load_frozen_c0  # noqa: E402
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset  # noqa: E402


CHECKPOINT = Path(os.environ.get(
    "RAM_C0_CHECKPOINT",
    str(REPO_ROOT / "outputs/whitebox_adam_20260922/fresh_adam/diagnostic_final.pth"),
))
DATASET_ROOT = Path(os.environ.get(
    "RAM_DATASET_ROOT",
    str(REPO_ROOT / "data/remote_variants/RAM-W600"),
))


@unittest.skipUnless(CHECKPOINT.is_file() and DATASET_ROOT.is_dir(), "local RAM C0/data assets are unavailable")
class RamHighresCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.refiner, cls.payload = load_frozen_c0(CHECKPOINT, torch.device("cpu"))
        cls.sample = NativeWristDataset(DATASET_ROOT, "val", False)[0]

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model
        del cls.refiner

    def test_capture_shapes_member_order_and_hook_counts(self) -> None:
        result = capture_case(self.model, self.refiner, self.sample["image"])
        self.assertEqual(result["source"].shape, (14, *self.sample["image"].shape[-2:]))
        self.assertEqual(result["step1"].shape, result["source"].shape)
        self.assertEqual(result["step2"].shape, result["source"].shape)
        self.assertEqual(result["backbone_h"].shape, (32, *self.sample["image"].shape[-2:]))
        self.assertEqual(result["refiner_h"].shape, (14, 12, *self.sample["image"].shape[-2:]))
        self.assertEqual(result["member_ids"], list(range(14)))
        self.assertEqual(result["member_chunk_ids"], [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13]])
        self.assertEqual(result["decoder_stage4_calls"], 1)
        self.assertEqual(result["refiner_dec1_calls_total"], 10)
        self.assertEqual(result["refiner_dec1_calls_step1"], 5)
        self.assertEqual(result["refiner_dec1_calls_step2"], 5)
        self.assertEqual(result["spatial_alignment"], "identity_native_padded")
        for key in ("source", "step1", "step2", "backbone_h", "refiner_h"):
            self.assertEqual(result[key].dtype, np.float32)

    def test_hooked_capture_matches_unhooked_forward_and_preserves_state(self) -> None:
        image = self.sample["image"]
        model_before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        refiner_before = {name: value.detach().clone() for name, value in self.refiner.state_dict().items()}
        reference = forward_c0(self.model, self.refiner, image)
        captured = capture_case(self.model, self.refiner, image)
        for key in ("source", "step1", "step2"):
            np.testing.assert_array_equal(captured[key], reference[key], err_msg=f"parity failed for {key}")
        for name, before in model_before.items():
            self.assertTrue(torch.equal(before, self.model.state_dict()[name]), name)
        for name, before in refiner_before.items():
            self.assertTrue(torch.equal(before, self.refiner.state_dict()[name]), name)

    def test_coordinate_identity_uses_native_padded_grid(self) -> None:
        image = self.sample["image"]
        result = capture_case(self.model, self.refiner, image)
        expected_hw = list(image.shape[-2:])
        self.assertEqual(result["image_hw"], expected_hw)
        self.assertEqual(result["backbone_h_shape"][-2:], expected_hw)
        self.assertEqual(result["refiner_h_shape"][-2:], expected_hw)


if __name__ == "__main__":
    unittest.main(verbosity=2)
