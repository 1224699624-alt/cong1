#!/usr/bin/env python3
"""Diagnostic grid evaluation for a trained boundary-band refiner checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from evaluate_masks import compute_metrics
from train_boundary_band_refiner import (
    BandDataset,
    BandUNet,
    add_structure_metrics,
    apply_band,
    mean_metrics,
    parse_floats,
    parse_ints,
    predict_prob,
    read_mask,
    resize_bool,
    write_mask,
    find_anchor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a boundary-band refiner checkpoint over fixed grids.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-root", default="data/raw_variants")
    parser.add_argument("--source-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--edit-radii", default="1,2,3,4,6,8,10,12")
    parser.add_argument("--max-edit-fracs", default="0,0.0025,0.005,0.01,0.02", help="0 means unlimited edits.")
    parser.add_argument("--top-full-metrics", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pred-root", default="")
    parser.add_argument("--output-exp", default="r082_boundary_band_readout")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def apply_limited_band(anchor: np.ndarray, prob: np.ndarray, threshold: float, radius: int, max_edit_frac: float) -> np.ndarray:
    pred = apply_band(anchor, prob, threshold, radius)
    changed = pred != anchor
    if max_edit_frac <= 0 or not changed.any():
        return pred
    max_edits = int(round(max_edit_frac * float(anchor.size)))
    if max_edits <= 0 or int(changed.sum()) <= max_edits:
        return pred
    confidence = np.abs(prob - threshold)
    idx = np.flatnonzero(changed.ravel())
    keep = idx[np.argsort(confidence.ravel()[idx])[-max_edits:]]
    limited = anchor.copy().ravel()
    limited[keep] = pred.ravel()[keep]
    return limited.reshape(anchor.shape)


@torch.no_grad()
def cache_items(model: torch.nn.Module, ds: BandDataset, device: str) -> list[dict[str, object]]:
    cached: list[dict[str, object]] = []
    roots = ds.roots
    for item in tqdm(ds, desc=f"cache/{ds.dataset}/{ds.split}"):
        prob = predict_prob(model, item["x"].unsqueeze(0), device).numpy()[0, 0]
        shape = item["shape"]
        prob_full = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        name = str(item["name"])
        gt = read_mask(ds.raw_root / ds.dataset / f"{ds.split}_labels" / name)
        anchor = resize_bool(read_mask(find_anchor(roots, ds.anchor_exp, ds.dataset, ds.split, name, ds.source_dataset)), gt.shape)
        cached.append({"name": name, "prob": prob_full, "anchor": anchor, "gt": gt})
    return cached


def evaluate_cached(
    cached: list[dict[str, object]],
    threshold: float,
    radius: int,
    max_edit_frac: float,
    boundary_kernel: int,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    for item in cached:
        name = str(item["name"])
        anchor = item["anchor"]  # type: ignore[assignment]
        prob = item["prob"]  # type: ignore[assignment]
        gt = item["gt"]  # type: ignore[assignment]
        pred = apply_limited_band(anchor, prob, threshold, radius, max_edit_frac)
        if write_dir is not None:
            write_mask(write_dir / name, pred)
        rec = compute_metrics(pred, gt, boundary_kernel)
        add_structure_metrics(rec, pred, gt)
        rec["image"] = name
        records.append(rec)
    return mean_metrics(records), records


def fast_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float(np.logical_and(pred, gt).sum())
    den = float(pred.sum() + gt.sum())
    return (2.0 * inter + 1e-6) / (den + 1e-6)


def evaluate_cached_fast(cached: list[dict[str, object]], threshold: float, radius: int, max_edit_frac: float) -> dict[str, float]:
    dice_values = []
    precision_values = []
    recall_values = []
    changed_values = []
    for item in cached:
        anchor = item["anchor"]  # type: ignore[assignment]
        prob = item["prob"]  # type: ignore[assignment]
        gt = item["gt"]  # type: ignore[assignment]
        pred = apply_limited_band(anchor, prob, threshold, radius, max_edit_frac)
        tp = float(np.logical_and(pred, gt).sum())
        fp = float(np.logical_and(pred, ~gt).sum())
        fn = float(np.logical_and(~pred, gt).sum())
        dice_values.append((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6))
        precision_values.append((tp + 1e-6) / (tp + fp + 1e-6))
        recall_values.append((tp + 1e-6) / (tp + fn + 1e-6))
        changed_values.append(float(np.mean(pred != anchor)))
    return {
        "dice": float(np.mean(dice_values)),
        "precision": float(np.mean(precision_values)),
        "recall": float(np.mean(recall_values)),
        "changed_frac": float(np.mean(changed_values)),
    }


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location=args.device)
    ckpt_args = state.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", args.base_channels))
    img_size = int(ckpt_args.get("img_size", args.img_size))
    model = BandUNet(base_channels).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()

    ds = BandDataset(
        Path(args.raw_root),
        args.dataset,
        args.split,
        [Path(p) for p in args.ablations_roots],
        args.anchor_exp,
        img_size,
        0,
        False,
        args.source_dataset,
    )
    cached = cache_items(model, ds, args.device)

    thresholds = parse_floats(args.thresholds)
    radii = parse_ints(args.edit_radii)
    max_edit_fracs = parse_floats(args.max_edit_fracs)
    fast_grid = []
    for radius in radii:
        for threshold in thresholds:
            for max_edit_frac in max_edit_fracs:
                mean = evaluate_cached_fast(cached, threshold, radius, max_edit_frac)
                row = {
                    "threshold": float(threshold),
                    "edit_radius": int(radius),
                    "max_edit_frac": float(max_edit_frac),
                    "fast_mean": mean,
                }
                fast_grid.append(row)
    fast_grid = sorted(fast_grid, key=lambda item: item["fast_mean"]["dice"], reverse=True)

    full_grid = []
    best = None
    for row in fast_grid[: max(1, args.top_full_metrics)]:
        mean, _ = evaluate_cached(
            cached,
            float(row["threshold"]),
            int(row["edit_radius"]),
            float(row["max_edit_frac"]),
            args.boundary_kernel,
        )
        full_row = {**row, "mean": mean}
        full_grid.append(full_row)
        if best is None or mean["dice"] > best["mean"]["dice"]:
            best = full_row
    assert best is not None

    per_image = []
    if args.pred_root:
        write_dir = Path(args.pred_root) / args.output_exp / args.dataset / args.split / "masks"
        _, per_image = evaluate_cached(
            cached,
            float(best["threshold"]),
            int(best["edit_radius"]),
            float(best["max_edit_frac"]),
            args.boundary_kernel,
            write_dir,
        )

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "checkpoint": args.checkpoint,
        "num_evaluated": len(cached),
        "best": best,
        "fast_grid": fast_grid,
        "full_grid": full_grid,
        "per_image": per_image,
        "note": "Diagnostic only if thresholds/radii/max_edit_frac are selected on this evaluated split.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
