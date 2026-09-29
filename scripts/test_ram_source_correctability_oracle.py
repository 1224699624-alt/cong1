#!/usr/bin/env python3
"""Focused tests for the RAM source-correctability oracle helpers."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

import audit_ram_source_correctability_oracle as oracle


class OracleMaskTests(unittest.TestCase):
    def test_c0_runner_uses_fp32_image_and_all_14_channels(self) -> None:
        model = torch.nn.Conv2d(2, 14, 1)
        refiner = torch.nn.Identity()
        runner = oracle.C0Runner(model, refiner, "v31", torch.device("cpu"))

        def two_steps(image, source, refiner, steps, zero_aux):
            self.assertEqual(source.dtype, torch.float32)
            self.assertEqual(tuple(source.shape), (1, 14, 4, 4))
            self.assertEqual(int(torch.count_nonzero(zero_aux)), 0)
            return [source, source, source], [torch.zeros_like(source), torch.zeros_like(source)]

        with patch("r332_v31_port.iterative_refine_o", side_effect=two_steps):
            source, masks, _ = runner.predict(torch.ones((1, 1, 4, 4), dtype=torch.float64))
        self.assertEqual(source.dtype, np.float32)
        self.assertEqual(len(masks), 3)

    def test_h0_definition_and_oracle_are_exact(self) -> None:
        target = np.zeros((3, 3, 4), dtype=bool)
        # Pixel (0, 0) has two true members and is a hard final FN for both.
        target[0, 0, 0] = True
        target[1, 0, 0] = True
        # Pixel (0, 1) is unique and must never enter H0.
        target[0, 0, 1] = True
        # Pixel (1, 0) is padded/invalid and must never enter H0.
        target[2, 1, 0] = True
        valid = np.ones((3, 4), dtype=bool)
        valid[1, 0] = False
        final = np.zeros_like(target)
        final[0, 0, 1] = True  # not an FN
        source = np.zeros_like(target, dtype=np.float32)
        source[0, 0, 0] = -5.00002
        source[1, 0, 0] = -5.00002
        source[2, 1, 0] = -20.0

        built = oracle.build_h0_oracle(target, valid, final, source)
        h0 = built["h0"]
        self.assertEqual(int(h0.sum()), 2)
        self.assertTrue(h0[0, 0, 0] and h0[1, 0, 0])
        self.assertFalse(h0[0, 0, 1])
        self.assertFalse(h0[2, 1, 0])
        self.assertTrue(np.array_equal(built["oracle"], final | h0))
        self.assertEqual(int(built["source_below_threshold"].sum()), 3)

    def test_membership_and_conditional_rates_are_member_versus_unique(self) -> None:
        target = np.zeros((3, 2, 3), dtype=bool)
        target[0, 0, 0] = True
        target[1, 0, 0] = True  # overlap pixel: two member pixels, one unique pixel
        target[2, 0, 1] = True  # unique pixel
        valid = np.ones((2, 3), dtype=bool)
        source = np.zeros_like(target, dtype=np.float32)
        source[0, 0, 0] = -6.0
        source[1, 0, 0] = -6.0
        final = np.zeros_like(target)
        built = oracle.build_h0_oracle(target, valid, final, source)
        predictions = {
            "source": final.copy(),
            "step1": final.copy(),
            "step2": final.copy(),
            "oracle": built["oracle"],
        }
        diagnostics = oracle.make_case_diagnostics(
            target, valid, source, predictions, built["h0"]
        )
        self.assertEqual(diagnostics["gt"]["member_pixels"], 3)
        self.assertEqual(diagnostics["gt"]["unique_pixels"], 2)
        self.assertEqual(diagnostics["gt"]["overlap_unique_pixels"], 1)
        self.assertEqual(diagnostics["gt"]["membership_1"], 1)
        self.assertEqual(diagnostics["gt"]["membership_2_plus"], 1)
        self.assertEqual(diagnostics["states"]["step2"]["member_fn"]["h0"]["fn"], 2)
        self.assertEqual(diagnostics["states"]["step2"]["unique_fn"]["h0"]["fn"], 1)
        self.assertEqual(
            diagnostics["states"]["step2"]["conditional_fn_by_gt_membership"]["membership_2_plus"]["fn"],
            2,
        )
        self.assertEqual(diagnostics["states"]["oracle"]["member_fn"]["h0"]["fn"], 0)

    def test_pack_roundtrip_preserves_full_native_shape(self) -> None:
        value = np.zeros((3, 5, 7), dtype=bool)
        value[0, 0, 0] = True
        value[2, 4, 6] = True
        packed = oracle.pack_bool(value)
        restored = oracle.unpack_bool(packed, value.shape)
        self.assertEqual(packed.dtype, np.uint8)
        self.assertTrue(np.array_equal(restored, value))


class LockAndAccumulatorTests(unittest.TestCase):
    def test_lock_hash_verification_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset.bin"
            asset.write_bytes(b"locked")
            lock = root / "lock.json"
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            lock.write_text(
                json.dumps({"status": "LOCKED", "file_hashes": {"asset.bin": digest}}),
                encoding="utf-8",
            )
            report = oracle.verify_lock(lock, json.loads(lock.read_text(encoding="utf-8")))
            self.assertEqual(report["checked_file_count"], 1)
            asset.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                oracle.verify_lock(lock, json.loads(lock.read_text(encoding="utf-8")))

    def test_streaming_official_accumulator_matches_weighted_rows(self) -> None:
        accumulator = oracle.OfficialAccumulator()
        first = {
            "overall_instance_metrics": {"n": 2, "dsc": 0.5, "nsd_2px": 0.25, "voe": 0.5, "msd_px": 2.0, "msd_fail_rate": 0.0, "ravd": 0.1},
            "overlap_region_metrics": {"n": 1, "dsc": 0.2, "nsd_2px": 0.3, "voe": 0.8, "msd_px": None, "msd_fail_rate": 1.0, "ravd": 0.2},
            "overlap_pair_intersection_metrics": {"n": 0, "dsc": None, "nsd_2px": None, "voe": None, "msd_px": None, "msd_fail_rate": 0.0, "ravd": None},
            "overlap_pair_count": 0,
        }
        second = {
            "overall_instance_metrics": {"n": 1, "dsc": 1.0, "nsd_2px": 1.0, "voe": 0.0, "msd_px": None, "msd_fail_rate": 1.0, "ravd": 0.0},
            "overlap_region_metrics": {"n": 1, "dsc": 0.4, "nsd_2px": 0.5, "voe": 0.6, "msd_px": 4.0, "msd_fail_rate": 0.0, "ravd": 0.4},
            "overlap_pair_intersection_metrics": {"n": 1, "dsc": 0.7, "nsd_2px": 0.8, "voe": 0.3, "msd_px": 3.0, "msd_fail_rate": 0.0, "ravd": 0.5},
            "overlap_pair_count": 1,
        }
        accumulator.add(first)
        accumulator.add(second)
        result = accumulator.result()
        self.assertEqual(result["overall_instance_metrics"]["n"], 3)
        self.assertAlmostEqual(result["overall_instance_metrics"]["dsc"], 2.0 / 3.0)
        self.assertAlmostEqual(result["overall_instance_metrics"]["msd_px"], 2.0)
        self.assertAlmostEqual(result["overall_instance_metrics"]["msd_fail_rate"], 1.0 / 3.0)
        self.assertEqual(result["overlap_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
