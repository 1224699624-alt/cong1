"""Fast static/unit checks for the native-start RAM final-prior runner."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import torch

import train_final_prior_ram as runner


class FinalRamRunnerTests(unittest.TestCase):
    def test_native_sha_constant_and_variants(self) -> None:
        self.assertEqual(len(runner.EXPECTED_NATIVE_SHA256), 64)
        self.assertEqual(runner.VARIANTS, ("A", "B", "C", "D"))

    def test_forward_has_no_ground_truth_argument(self) -> None:
        names = tuple(inspect.signature(runner.forward_variant).parameters)
        self.assertNotIn("target", names)
        self.assertNotIn("mask", names)

    def test_branch_matrix(self) -> None:
        self.assertEqual((False, False), ("A" in ("B", "D"), "A" in ("C", "D")))
        self.assertEqual((True, False), ("B" in ("B", "D"), "B" in ("C", "D")))
        self.assertEqual((False, True), ("C" in ("B", "D"), "C" in ("C", "D")))
        self.assertEqual((True, True), ("D" in ("B", "D"), "D" in ("C", "D")))

    def test_selection_uses_overall_dsc(self) -> None:
        report = {"flat": {"overall_dsc": 0.8125}}
        self.assertEqual(runner._selection_value(report), 0.8125)

    def test_fresh_output_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new"
            runner._assert_fresh_output(path)
            self.assertTrue(path.is_dir())
            (path / "sentinel").write_text("x", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                runner._assert_fresh_output(path)

    def test_native_load_is_strict_in_source(self) -> None:
        source = inspect.getsource(runner._load_native_model)
        self.assertIn("strict=True", source)
        self.assertIn("EXPECTED_NATIVE_SHA256", inspect.getsource(runner._load_native_model))


if __name__ == "__main__":
    unittest.main()
