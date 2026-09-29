"""Focused pure-function tests for the RAM response audit."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_ram_source_correctability_response as audit


class ResponseAuditHelperTests(unittest.TestCase):
    def test_packed_h0_cache_round_trip_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.npz"
            h0 = np.zeros((14, 4, 5), dtype=bool)
            h0[0, 1, 2] = True
            h0[13, 3, 4] = True
            with path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    shape=np.asarray(h0.shape, dtype=np.int64),
                    original_hw=np.asarray([4, 5], dtype=np.int64),
                    h0=audit.pack_bool(h0),
                    source_below=audit.pack_bool(h0),
                    label=np.asarray([audit.ORACLE_LABEL]),
                )
            loaded = audit.load_h0_cache(path, h0.shape)
            self.assertTrue(np.array_equal(loaded["h0"], h0))
            self.assertTrue(np.array_equal(loaded["source_below"], h0))
            self.assertEqual(loaded["label"], audit.ORACLE_LABEL)

    def test_h0_definition_uses_overlap_valid_final_fn_and_source_cutoff(self) -> None:
        target = np.zeros((3, 3, 4), dtype=bool)
        target[0, 0, 0] = True
        target[1, 0, 0] = True
        target[0, 0, 1] = True  # unique member: excluded
        valid = np.ones((3, 4), dtype=bool)
        valid[2, 2] = False
        final = np.zeros_like(target)
        final[0, 0, 1] = True  # already predicted: excluded
        source = np.zeros_like(target, dtype=np.float32)
        source[0, 0, 0] = audit.SOURCE_CUTOFF - 0.01
        source[1, 0, 0] = audit.SOURCE_CUTOFF - 0.01
        source[2, 2, 2] = audit.SOURCE_CUTOFF - 1.0  # padded: excluded
        expected = np.zeros_like(target)
        expected[0, 0, 0] = True
        expected[1, 0, 0] = True
        self.assertTrue(np.array_equal(
            audit.pack_bool(expected), audit.pack_bool(
                valid[None] & target & (target.sum(axis=0)[None] >= 2) & ~final & (source < audit.SOURCE_CUTOFF)
            )
        ))

    def test_prediction_uses_fp32_sigmoid_threshold_for_tiny_negative_logits(self) -> None:
        logits = np.asarray([-1e-8, -1e-6], dtype=np.float32)
        expected = (torch.sigmoid(torch.from_numpy(logits)) >= 0.5).numpy()
        self.assertTrue(np.array_equal(audit._prediction(logits), expected))
        self.assertTrue(bool(expected[0]))
        self.assertFalse(bool(expected[1]))

    def test_background_distribution_sampling_keeps_h0_exact(self) -> None:
        values = np.arange(20000, dtype=np.float32)
        valid = np.ones_like(values, dtype=bool)
        sampled = audit._mask_stats(values, valid, max_samples=8192)
        exact = audit._mask_stats(values, valid)
        self.assertEqual(sampled["population_n"], 20000)
        self.assertEqual(sampled["sampling_stride"], 3)
        self.assertEqual(sampled["n"], len(values[::3]))
        self.assertEqual(exact["n"], 20000)
        self.assertNotIn("sampling", exact)

    def test_control_matching_is_fixed_and_same_member(self) -> None:
        target = np.zeros((2, 3, 4), dtype=bool)
        target[0, 1, 1] = True
        target[1, 1, 1] = True
        valid = np.ones((3, 4), dtype=bool)
        h0 = np.zeros_like(target)
        h0[:, 1, 1] = True
        logits = np.zeros_like(target, dtype=np.float32)
        first, first_meta = audit.make_matched_controls(target, valid, h0, logits)
        second, second_meta = audit.make_matched_controls(target, valid, h0, logits)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_meta["pair_sha256"], second_meta["pair_sha256"])
        self.assertTrue(np.all(first[:, 0] == np.asarray([0, 1])))
        self.assertTrue(np.all(first[:, 3] >= 0))
        self.assertTrue(np.all(~target[first[:, 0], first[:, 3], first[:, 4]]))

    def test_global_overlap_fn_requires_two_predicted_members(self) -> None:
        target = np.zeros((2, 1, 1), dtype=bool)
        target[:, 0, 0] = True
        valid = np.ones((1, 1), dtype=bool)
        h0 = np.zeros_like(target)
        c0_final = np.zeros_like(target)
        c0_final[0, 0, 0] = True  # C0 member count is one: overlap FN.
        one_member = c0_final.copy()
        two_members = target.copy()
        one = audit._state_summary(
            np.zeros_like(target, dtype=np.float32), one_member, target, valid, h0, c0_final,
            np.ones((1, 1), dtype=bool),
        )
        two = audit._state_summary(
            np.zeros_like(target, dtype=np.float32), two_members, target, valid, h0, c0_final,
            np.ones((1, 1), dtype=bool),
        )
        self.assertEqual(one["unique_global_overlap_fn_recovered"], 0)
        self.assertEqual(two["unique_global_overlap_fn_recovered"], 1)

    def test_hash_validation_rejects_mutation_and_missing_required_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "asset.bin"
            asset.write_bytes(b"locked")
            lock_path = root / "lock.json"
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()
            lock = {
                "status": "LOCKED",
                "cpu_threads": 64,
                "split_counts": {"train": 425, "val": 69},
                "file_hashes": {"asset.bin": digest},
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            report = audit.verify_input_hashes(lock_path, lock, {"asset": asset})
            self.assertEqual(report["status"], "PASS")
            asset.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                audit.verify_input_hashes(lock_path, lock, {"asset": asset})

    def test_required_path_label_is_not_mistaken_for_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "r332.pth"
            asset.write_bytes(b"checkpoint")
            lock_path = root / "lock.json"
            lock = {
                "status": "LOCKED",
                "cpu_threads": 64,
                "split_counts": {"train": 425, "val": 69},
                "r332": str(asset),
                "file_hashes": {str(asset): hashlib.sha256(asset.read_bytes()).hexdigest()},
            }
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            self.assertEqual(
                audit.verify_input_hashes(lock_path, lock, {"r332": asset})["status"],
                "PASS",
            )


if __name__ == "__main__":
    unittest.main()
