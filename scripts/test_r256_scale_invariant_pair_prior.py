#!/usr/bin/env python3
"""Fast deterministic checks for the R256 proposal and augmentation geometry."""

from __future__ import annotations

import csv
import argparse
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from train_r256_scale_invariant_pair_prior import (
    PairDataset,
    build_samples,
    gaussian,
    make_heat_target,
    noisy_proposal,
    point_line_distance,
    read_metadata,
    transform_direction,
)


def peak_direction(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ay, ax = np.unravel_index(np.argmax(a), a.shape)
    by, bx = np.unravel_index(np.argmax(b), b.shape)
    delta = np.asarray([bx - ax, by - ay], dtype=float)
    delta /= np.linalg.norm(delta)
    return float(delta[0]), float(delta[1])


def test_rotations() -> None:
    a, b = gaussian(64, 18, 25, 2), gaussian(64, 44, 37, 2)
    dx, dy = peak_direction(a, b)
    for turns in range(4):
        observed = peak_direction(np.rot90(a, turns), np.rot90(b, turns))
        expected = transform_direction(dx, dy, turns=turns)
        assert np.allclose(observed, expected, atol=0.04), (turns, observed, expected)


def test_proposal_reproducibility_and_crop_coverage() -> None:
    ca, cb = np.asarray([30.0, 50.0]), np.asarray([95.0, 63.0])
    p1 = noisy_proposal(ca, cb, 400.0, 625.0, 256, 0.08, 0.12)
    p2 = noisy_proposal(ca, cb, 400.0, 625.0, 256, 0.08, 0.12)
    assert p1 == p2
    pa, pb = np.asarray(p1["pa"]), np.asarray(p1["pb"])
    side = math.ceil(max(32.0, np.linalg.norm(pb-pa) + 1.35 * (p1["psa"] + p1["psb"])))
    center = 0.5 * (pa + pb)
    assert np.all(np.abs(pa-center) < side/2) and np.all(np.abs(pb-center) < side/2)


def test_legal_mask_and_corridor() -> None:
    a = np.zeros((64, 64), bool); b = np.zeros((64, 64), bool); third = np.zeros((64, 64), bool)
    a[20:36, 10:28] = True; b[20:36, 32:50] = True; third[26:30, 29:31] = True
    target, valid = make_heat_target(a, b, a | b | third, 64, np.asarray([19., 28.]), np.asarray([41., 28.]), 18.0)
    assert valid
    assert float(target[(a | b | third)].max(initial=0.0)) == 0.0
    assert target[23, 30] > target[10, 30]


def test_metadata_parser() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "metadata.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "boneage", "male"])
            writer.writeheader(); writer.writerow({"id": "001", "boneage": "120", "male": "True"})
        parsed = read_metadata(path)
        assert parsed == {"001": (120.0, 1.0)}


def test_real_data(root: Path, train_metadata_path: Path, val_metadata_path: Path) -> None:
    train_metadata, val_metadata = read_metadata(train_metadata_path), read_metadata(val_metadata_path)
    assert len(train_metadata) == 875 and len(val_metadata) == 96
    assert not (set(train_metadata) & set(val_metadata))
    args = SimpleNamespace(limit_train=32, limit_val=16, max_positive_per_case=32,
                           max_negative_per_case=32, relative_pair_limit=0.8,
                           relative_close_threshold=0.2, center_noise_rel=0.08,
                           scale_noise_log_std=0.12, seed=256, crop_margin_scales=1.35,
                           crop_size=128)
    train_samples, train_audit = build_samples(root, "train", train_metadata, args)
    val_samples, val_audit = build_samples(root, "val", val_metadata, args)
    assert train_samples and val_samples
    dataset = PairDataset(val_samples, args, augment=False, use_development=True)
    labels, close, contact, valid = [], [], [], []
    for item in dataset:
        assert item["image"].shape == (3, 128, 128)
        assert item["heatmap"].shape == (1, 128, 128)
        assert np.isfinite(item["image"].numpy()).all() and np.isfinite(item["heatmap"].numpy()).all()
        labels.append(item["label"].item()); close.append(item["close"].item())
        contact.append(item["contact"].item()); valid.append(item["heat_valid"].item())
    labels, valid = np.asarray(labels) > 0.5, np.asarray(valid) > 0.5
    assert labels.any() and (~labels).any()
    assert abs(float(labels.mean()) - 0.5) < 1e-9
    assert np.asarray(contact).sum() > 0, "touching/near-touching pairs must not be silently excluded"
    print({"train_pairs": len(train_samples), "val_pairs": len(val_samples),
           "val_close": int(np.asarray(close).sum()), "val_contact": int(np.asarray(contact).sum()),
           "val_heat_coverage": float(valid[labels].mean()), "train_audit": train_audit, "val_audit": val_audit})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration-root", type=Path)
    parser.add_argument("--train-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_train.csv"))
    parser.add_argument("--val-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_val.csv"))
    cli = parser.parse_args()
    test_rotations()
    test_proposal_reproducibility_and_crop_coverage()
    test_legal_mask_and_corridor()
    test_metadata_parser()
    if cli.integration_root:
        test_real_data(cli.integration_root, cli.train_metadata, cli.val_metadata)
    print("R256 synthetic checks: PASS")
