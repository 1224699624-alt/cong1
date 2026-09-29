"""Focused pure-function tests for ``audit_ram_o_conditional_h0``."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_ram_o_conditional_h0 as audit


class RamOConditionalAuditTests(unittest.TestCase):
    def test_matching_is_deterministic_and_uses_singleton_controls(self) -> None:
        height, width = 5, 6
        target = np.zeros((14, height, width), dtype=bool)
        target[0, 0, 0] = True
        target[1, 0, 0] = True  # H0 is an overlap pixel for members 0 and 1.
        target[2, 1, 1] = True  # A GT-single foreground control for 0/1.
        valid = np.ones((height, width), dtype=bool)
        h0 = np.zeros_like(target)
        h0[0, 0, 0] = True
        h0[1, 0, 0] = True
        source = np.full((14, height, width), 0.2, dtype=np.float32)
        final = source.copy()
        image = np.zeros((height, width), dtype=np.float32)
        bundle = audit.feature_bundle(source, final, image)
        features = {name: bundle[name] for name in ("count_class", "top1_id", *audit.MATCH_FEATURE_NAMES)}

        first, first_meta = audit.make_conditional_matches(
            target, valid, h0, features, max_rows=32, max_candidate_scan=64
        )
        second, second_meta = audit.make_conditional_matches(
            target, valid, h0, features, max_rows=32, max_candidate_scan=64
        )
        self.assertTrue(np.array_equal(first, second, equal_nan=True))
        self.assertEqual(first_meta["pair_sha256"], second_meta["pair_sha256"])
        self.assertEqual(first_meta["matched"], 2)
        self.assertEqual(first_meta["singleton_matched"], 2)
        self.assertTrue(np.all(first[:, 5] == 1))
        for row in first.astype(int):
            member, _, _, cy, cx = row[:5]
            self.assertEqual(int(target[:, cy, cx].sum()), 1)
            self.assertFalse(bool(target[member, cy, cx]))

    def test_matching_reports_fallback_without_promoting_it_to_strict(self) -> None:
        height, width = 3, 4
        target = np.zeros((14, height, width), dtype=bool)
        target[0, 0, 0] = target[1, 0, 0] = True
        target[2, 1, 1] = True
        valid = np.ones((height, width), dtype=bool)
        h0 = np.zeros_like(target)
        h0[0, 0, 0] = h0[1, 0, 0] = True
        source = np.full((14, height, width), 0.2, dtype=np.float32)
        final = source.copy()
        image = np.zeros((height, width), dtype=np.float32)
        image[1, 1] = 10.0  # Force the singleton candidate outside the caliper.
        bundle = audit.feature_bundle(source, final, image)
        features = {name: bundle[name] for name in ("count_class", "top1_id", *audit.MATCH_FEATURE_NAMES)}
        rows, metadata = audit.make_conditional_matches(
            target, valid, h0, features, calipers=[1e-4] * 4 + [1e-4, 1e-4], max_rows=32, max_candidate_scan=64
        )
        self.assertEqual(metadata["matched"], 2)
        self.assertGreater(metadata["status_counts"].get("singleton_nearest_fallback", 0), 0)
        self.assertEqual(int(np.count_nonzero(rows[:, 5] == 1)), 0)

    def test_conditional_logistic_readout_and_paired_bootstrap(self) -> None:
        rng = np.random.default_rng(17)
        n = 120
        labels = np.asarray([0, 1] * (n // 2), dtype=np.int8)
        base = np.zeros((n, 3), dtype=np.float32)
        o = np.column_stack((labels + rng.normal(0.0, 0.05, n), labels.astype(np.float32))).astype(np.float32)
        cases = np.asarray([f"case_{i // 6:03d}" for i in range(n)])
        base_model = audit.FixedLogisticReadout(l2=1.0).fit(base, labels)
        o_model = audit.FixedLogisticReadout(l2=1.0).fit(np.column_stack((base, o)), labels)
        base_score = base_model.predict_proba(base)
        o_score = o_model.predict_proba(np.column_stack((base, o)))
        self.assertTrue(o_model.summary()["converged"])
        self.assertGreater(float(audit._auc(labels, o_score)), float(audit._auc(labels, base_score)))
        paired = audit.paired_case_bootstrap(labels, base_score, o_score, cases, seed=9, repetitions=100)
        self.assertEqual(paired["repetitions"], 100)
        self.assertGreater(paired["delta_auc_o_minus_base"]["q50"], 0.0)
        self.assertLess(paired["delta_logloss_o_minus_base"]["q50"], 0.0)

    def test_packed_h0_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "case.npz"
            h0 = np.zeros((14, 4, 5), dtype=bool)
            h0[0, 1, 2] = True
            with path.open("wb") as handle:
                np.savez_compressed(handle, shape=np.asarray(h0.shape), original_hw=np.asarray([4, 5]),
                                    h0=np.packbits(h0.reshape(-1), bitorder="little"),
                                    label=np.asarray([audit.ORACLE_LABEL]))
            loaded = audit.load_h0_cache(path, h0.shape)
            self.assertTrue(np.array_equal(loaded["h0"], h0))
            self.assertEqual(loaded["label"], audit.ORACLE_LABEL)


if __name__ == "__main__":
    unittest.main()
