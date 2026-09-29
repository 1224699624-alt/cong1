#!/usr/bin/env python3
"""Synthetic regression tests for the TSRS anonymous-label decoder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tsrs_instance_labels import (
        audit_dataset,
        audit_instance_label,
        decode_anonymous_instances,
        write_audit_outputs,
    )
except ImportError:  # Allows both ``python scripts/test_...`` and ``-m`` use.
    from scripts.tsrs_instance_labels import (
        audit_dataset,
        audit_instance_label,
        decode_anonymous_instances,
        write_audit_outputs,
    )


class TsrsInstanceLabelTests(unittest.TestCase):
    def test_tsrs_instance_decode_palette(self) -> None:
        values = np.asarray(
            [
                [0, 1, 1, 0, 2],
                [0, 1, 1, 0, 2],
                [0, 0, 0, 0, 2],
            ],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "palette.png"
            image = Image.fromarray(values, mode="P")
            palette = [0, 0, 0, 255, 0, 0, 0, 255, 0]
            image.putpalette(palette + [0] * (768 - len(palette)))
            image.save(path)

            decoded = decode_anonymous_instances(path)
            report = audit_instance_label(path)

        self.assertEqual(decoded.dtype, np.int32)
        self.assertEqual(report["mode"], "P")
        self.assertEqual(report["anonymous_instance_count"], 2)
        self.assertEqual(report["raw_unique_value_count"], 3)
        self.assertTrue(report["binary_union_preserved"])
        np.testing.assert_array_equal(decoded > 0, values > 0)

    def test_tsrs_instance_decode_rgb_full_tuple(self) -> None:
        values = np.zeros((4, 8, 3), dtype=np.uint8)
        values[1:3, 1:3] = (17, 20, 30)
        values[1:3, 5:7] = (17, 40, 50)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rgb.png"
            Image.fromarray(values, mode="RGB").save(path)
            decoded = decode_anonymous_instances(path)
            report = audit_instance_label(path)

        # Both foreground colours have the same R channel.  A first-channel
        # reader would collapse them; a full-tuple decoder keeps two instances.
        self.assertEqual(report["anonymous_instance_count"], 2)
        self.assertEqual(report["first_channel_collision_count"], 1)
        self.assertEqual(report["first_channel_collision_color_count"], 2)
        self.assertTrue(report["first_channel_collision"])
        self.assertTrue(report["binary_union_preserved"])
        self.assertEqual(np.unique(decoded[decoded > 0]).size, 2)
        np.testing.assert_array_equal(decoded > 0, np.any(values != 0, axis=-1))

    def test_tsrs_same_color_disconnected_components_split(self) -> None:
        values = np.zeros((8, 10, 3), dtype=np.uint8)
        values[1:3, 1:3] = (9, 8, 7)
        values[5:7, 7:9] = (9, 8, 7)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "repeated_rgb.png"
            Image.fromarray(values, mode="RGB").save(path)
            decoded = decode_anonymous_instances(path)
            report = audit_instance_label(path)

        self.assertEqual(report["anonymous_instance_count"], 2)
        self.assertEqual(report["multi_component_value_count"], 1)
        self.assertEqual(report["recoverable_duplicate_color_instance_count"], 1)
        self.assertTrue(report["multi_component_image"])
        np.testing.assert_array_equal(decoded > 0, np.any(values != 0, axis=-1))

    def test_binary_union_unchanged_after_instance_remap(self) -> None:
        values = np.zeros((7, 7), dtype=np.uint8)
        values[1, 1] = 3
        values[1, 2] = 3
        values[5, 5] = 4
        values[5, 4] = 4
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "indexed.png"
            Image.fromarray(values, mode="L").save(path)
            decoded = decode_anonymous_instances(path)
            report = audit_instance_label(path)

        self.assertEqual(decoded.dtype, np.int32)
        self.assertTrue(np.array_equal(decoded > 0, values > 0))
        self.assertEqual(report["binary_union_preserved"], True)
        self.assertEqual(report["foreground_pixel_count"], int(np.count_nonzero(values)))

    def test_rgba_alpha_ambiguity_is_qc_only(self) -> None:
        values = np.zeros((2, 4, 4), dtype=np.uint8)
        values[0, 0] = (3, 4, 5, 0)  # transparent but non-background RGB
        values[0, 1] = (3, 4, 5, 255)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "alpha.png"
            Image.fromarray(values, mode="RGBA").save(path)
            decoded = decode_anonymous_instances(path)
            report = audit_instance_label(path)

        self.assertTrue(report["alpha_present"])
        self.assertTrue(report["alpha_ambiguity"])
        self.assertEqual(report["alpha_tuple_conflict_count"], 1)
        self.assertEqual(report["anonymous_instance_count"], 1)
        self.assertEqual(int(np.count_nonzero(decoded > 0)), 2)

    def test_dataset_audit_rejects_test_and_writes_json_csv(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            (root / "train_labels").mkdir(parents=True)
            (root / "val_labels").mkdir(parents=True)
            values = np.zeros((3, 3), dtype=np.uint8)
            values[1, 1] = 1
            Image.fromarray(values, mode="L").save(root / "train_labels" / "a.png")
            Image.fromarray(values, mode="L").save(root / "val_labels" / "b.png")

            with self.assertRaises(ValueError):
                audit_dataset(root, splits=("train", "test"))

            summary = audit_dataset(root, splits=("train", "val"))
            json_path, csv_path = write_audit_outputs(summary, Path(folder) / "out")
            self.assertEqual(summary["image_count"], 2)
            self.assertFalse(summary["test_read"])
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())


if __name__ == "__main__":
    unittest.main()

