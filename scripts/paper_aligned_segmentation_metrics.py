#!/usr/bin/env python3
"""RAM-W600 paper-aligned binary segmentation metrics using MONAI 1.4.0.

Reference implementation: YSongxiao/RAM-W600 evaluations/metrics.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from monai.metrics import DiceMetric, MeanIoU, SurfaceDiceMetric, SurfaceDistanceMetric


def compute_paper_metrics(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    nsd_tolerance_px: float = 2.0,
) -> dict[str, float | None]:
    """Compute DSC, NSD, VOE, symmetric MSD and RAVD for one binary mask pair."""
    pred = torch.as_tensor(prediction, dtype=torch.float32).squeeze()
    gt = torch.as_tensor(target, dtype=torch.float32).squeeze()
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError({"prediction": tuple(pred.shape), "target": tuple(gt.shape)})
    pred = (pred > 0.5)[None, None]
    gt = (gt > 0.5)[None, None]
    dsc = DiceMetric(include_background=True, reduction="none")(pred, gt).squeeze()
    nsd = SurfaceDiceMetric(
        class_thresholds=[nsd_tolerance_px], include_background=True, reduction="none"
    )(pred, gt).squeeze()
    iou = MeanIoU(include_background=True, reduction="none")(pred, gt).squeeze()
    msd = SurfaceDistanceMetric(
        include_background=True, symmetric=True, reduction="none"
    )(pred, gt).squeeze()
    gt_area = float(gt.sum())
    pred_area = float(pred.sum())
    ravd = 0.0 if gt_area == 0 else abs(pred_area - gt_area) / (gt_area + 1e-8)

    def finite_or_none(value: torch.Tensor) -> float | None:
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None

    iou_value = finite_or_none(iou)
    return {
        "dsc": finite_or_none(dsc),
        "nsd_2px": finite_or_none(nsd),
        "voe": None if iou_value is None else 1.0 - iou_value,
        "msd_px": finite_or_none(msd),
        "ravd": float(ravd),
        "msd_failed": float(not math.isfinite(float(msd))),
    }


def synthetic_self_test() -> None:
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = 1
    perfect = compute_paper_metrics(target, target)
    assert perfect["dsc"] == 1.0
    assert perfect["nsd_2px"] == 1.0
    assert perfect["voe"] == 0.0
    assert perfect["msd_px"] == 0.0
    assert perfect["ravd"] == 0.0


if __name__ == "__main__":
    synthetic_self_test()
    print("paper-aligned metric self-test passed")
