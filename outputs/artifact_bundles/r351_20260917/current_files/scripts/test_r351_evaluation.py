#!/usr/bin/env python3
"""Dependency-light tests for the reusable R351 evaluation guards."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from r351_evaluation import (
        assess_comparison,
        bind_single_checkpoint_result,
        check_dice_iou_non_degradation,
        compare_fixed_seam_roi,
        evaluate_fixed_seam_roi,
        fixed_seam_background_roi,
        sha256_file,
        validate_bound_result,
        write_bound_result,
    )
except ModuleNotFoundError:  # Supports ``python -m unittest scripts/...``.
    from scripts.r351_evaluation import (
        assess_comparison,
        bind_single_checkpoint_result,
        check_dice_iou_non_degradation,
        compare_fixed_seam_roi,
        evaluate_fixed_seam_roi,
        fixed_seam_background_roi,
        sha256_file,
        validate_bound_result,
        write_bound_result,
    )


class R351EvaluationTests(unittest.TestCase):
    def test_roi_uses_only_seam_prior_and_gt_union(self) -> None:
        seam = np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)
        target = np.zeros((2, 2, 2), dtype=np.uint8)
        target[0, 0, 1] = 1
        expected = np.array([[False, False], [True, False]])
        np.testing.assert_array_equal(fixed_seam_background_roi(seam, target), expected)

        # Changing predictions cannot change the fixed evaluation ROI.
        baseline = np.zeros((2, 2, 2), dtype=np.uint8)
        adapted = np.ones_like(baseline)
        base_eval = evaluate_fixed_seam_roi({"case": baseline}, {"case": seam}, {"case": target})
        adapted_eval = evaluate_fixed_seam_roi({"case": adapted}, {"case": seam}, {"case": target})
        self.assertEqual(base_eval["per_image"][0]["roi_checksum"], adapted_eval["per_image"][0]["roi_checksum"])
        self.assertFalse(base_eval["protocol"]["model_prediction_used_for_roi"])

    def test_per_image_fp_area_and_pooled_rate(self) -> None:
        seam = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        target = np.zeros((1, 2, 2), dtype=np.uint8)
        predictions = {
            "a": np.array([[False, True], [False, False]]),
            "b": np.zeros((2, 2), dtype=bool),
        }
        priors = {"a": seam, "b": np.zeros((2, 2), dtype=np.float32)}
        targets = {"a": target, "b": target}
        result = evaluate_fixed_seam_roi(predictions, priors, targets)
        rows = {row["image"]: row for row in result["per_image"]}
        self.assertEqual(rows["a"]["fp_over_area"], "1/2")
        self.assertEqual(rows["a"]["fp_pixels"], 1)
        self.assertEqual(rows["a"]["roi_area"], 2)
        self.assertEqual(rows["b"]["roi_area"], 0)
        self.assertIsNone(rows["b"]["fp_rate"])
        self.assertEqual(result["aggregate"]["fp_pixels"], 1)
        self.assertEqual(result["aggregate"]["roi_area"], 2)
        self.assertAlmostEqual(result["aggregate"]["fp_rate"], 0.5)
        self.assertEqual(result["aggregate"]["empty_roi_images"], 1)

    def test_compare_reports_degradation_without_calling_it_improvement(self) -> None:
        seam = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        target = np.zeros((2, 2), dtype=np.uint8)
        baseline = {"case": np.zeros((2, 2), dtype=bool)}
        adapted = {"case": np.array([[False, True], [False, False]])}
        result = compare_fixed_seam_roi(baseline, adapted, {"case": seam}, {"case": target})
        change = result["comparison"]["fp_rate"]
        self.assertEqual(change["status"], "degraded")
        self.assertFalse(change["favorable"])
        self.assertAlmostEqual(change["delta_candidate_minus_baseline"], 0.5)
        # A zero baseline has no meaningful relative denominator, but the
        # directional status still identifies the positive FP-rate change.
        self.assertIsNone(change["relative_improvement"])

    def test_empty_roi_comparison_is_undefined(self) -> None:
        seam = np.zeros((2, 2), dtype=np.float32)
        target = np.zeros((2, 2), dtype=np.uint8)
        result = compare_fixed_seam_roi(
            {"case": np.zeros((2, 2), dtype=bool)},
            {"case": np.ones((2, 2), dtype=bool)},
            {"case": seam},
            {"case": target},
        )
        self.assertEqual(result["comparison"]["fp_rate"]["status"], "undefined_empty_roi")
        self.assertIsNone(result["comparison"]["fp_rate"]["relative_improvement"])

    def test_strict_overlap_gate_and_near_tie_are_separate(self) -> None:
        result = check_dice_iou_non_degradation(
            {"dice": 0.9, "iou": 0.8},
            {"dice": 0.8999, "iou": 0.8},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["dice"]["status"], "near_tie")
        self.assertTrue(result["metrics"]["dice"]["approximate_tie"])
        self.assertEqual(result["metrics"]["iou"]["status"], "unchanged")

        improved = check_dice_iou_non_degradation(
            {"dice": 0.9, "iou": 0.8},
            {"dice": 0.9001, "iou": 0.8001},
        )
        self.assertTrue(improved["passed"])
        self.assertEqual(improved["metrics"]["dice"]["status"], "improved")

    def test_assess_tsrs_primary_and_auxiliary_statuses(self) -> None:
        baseline = {"mean": {"dice": 0.90, "iou": 0.82, "gap_region_fp_rate": 0.20,
                             "component_merge_rate": 0.50, "boundary_iou": 0.30}}
        candidate = {"mean": {"dice": 0.90, "iou": 0.82, "gap_region_fp_rate": 0.15,
                              "component_merge_rate": 0.40, "boundary_iou": 0.29}}
        result = assess_comparison(baseline, candidate, domain="tsrs")
        self.assertTrue(result["passed"])
        self.assertTrue(result["primary"]["claim_passed"])
        self.assertEqual(result["primary"]["metrics"]["gap_fp"]["status"], "improved")
        self.assertEqual(result["primary"]["metrics"]["merge"]["status"], "improved")
        self.assertEqual(result["auxiliary"]["mean.boundary_iou"]["status"], "degraded")
        self.assertIn("mean.boundary_iou", result["auxiliary_degraded"])
        self.assertFalse(result["guarantees"]["universal_metric_non_degradation"])

    def test_assess_ram_resolves_nested_official_metrics(self) -> None:
        baseline = {
            "overall_instance_metrics": {"dsc": 0.97},
            "overlap_region_metrics": {"dsc": 0.80, "voe": 0.40, "nsd_2px": 0.70, "msd_px": 2.0},
            "overlap_pair_intersection_metrics": {"msd_px": 1.5},
        }
        candidate = {
            "overall_instance_metrics": {"dsc": 0.97},
            "overlap_region_metrics": {"dsc": 0.81, "voe": 0.38, "nsd_2px": 0.72, "msd_px": 1.8},
            "overlap_pair_intersection_metrics": {"msd_px": 1.2},
        }
        result = assess_comparison(baseline, candidate, domain="ram")
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["overlap_safety"]["metrics"]["iou"]["candidate"], 0.62)
        self.assertEqual(result["overlap_safety"]["metrics"]["dice"]["baseline"], 0.80)
        self.assertEqual(result["primary"]["metrics"]["pair_msd_px"]["status"], "improved")
        self.assertEqual(result["primary"]["metrics"]["overlap_msd_px"]["status"], "improved")
        self.assertFalse(result["guarantees"]["universal_metric_non_degradation"])

    def test_result_binding_uses_one_epoch_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"r351 checkpoint")
            digest = sha256_file(checkpoint)
            payload = bind_single_checkpoint_result(
                experiment="R351",
                split="validation",
                epoch=7,
                checkpoint_sha256=digest,
                metrics={"epoch": 7, "dice": 0.9, "iou": 0.8},
            )
            self.assertTrue(validate_bound_result(payload)["passed"])
            output = write_bound_result(Path(directory) / "result.json", payload)
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(loaded["selection"]["epoch"], 7)
            self.assertEqual(loaded["selection"]["checkpoint_sha256"], digest)

    def test_result_binding_rejects_mixed_epoch(self) -> None:
        with self.assertRaises(ValueError):
            bind_single_checkpoint_result(
                experiment="R351",
                split="validation",
                epoch=7,
                checkpoint_sha256="a" * 64,
                metrics={"history": [{"epoch": 6}, {"epoch": 7}]},
            )

    def test_result_binding_rejects_checkpoint_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaises(ValueError):
                bind_single_checkpoint_result(
                    experiment="R351",
                    split="validation",
                    epoch=7,
                    checkpoint_sha256="a" * 64,
                    checkpoint_path=checkpoint,
                    metrics={"dice": 0.9, "iou": 0.8},
                )

    def test_case_key_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_fixed_seam_roi(
                {"a": np.zeros((2, 2))},
                {"b": np.zeros((2, 2))},
                {"a": np.zeros((2, 2))},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
