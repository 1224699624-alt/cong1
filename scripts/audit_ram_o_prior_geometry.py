#!/usr/bin/env python3
"""Read-only RAM image-only O audit against actual multi-hot NPY overlap labels.

This diagnoses the prior probability map; it does not train, modify GT, tune a
segmenter, evaluate test, or turn a GT-derived H0 mask into an inference input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter
from torch.utils.data import DataLoader

from structural_prior_v3 import ImageOnlyOverlapPrior
from train_r325_ram_nnunet_native_resolution_overlap_iem import NativeWristDataset


EXPECTED_O_SHA = "e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5"
RADII = (0, 3, 7, 15)
SHIFTS = range(-4, 5)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_h0(path: Path, shape, original_hw) -> np.ndarray:
    with np.load(path, allow_pickle=False) as x:
        cached_shape = tuple(int(v) for v in x["shape"])
        if cached_shape != shape:
            raise ValueError(f"H0 shape mismatch {path}: {cached_shape} != {shape}")
        if tuple(int(v) for v in x["original_hw"]) != original_hw:
            raise ValueError(f"H0 original geometry mismatch {path}")
        bits = np.unpackbits(x["h0"], bitorder="little")[:np.prod(shape)]
    return bits.reshape(shape).astype(bool)


def shifted(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = image.shape
    out = np.zeros_like(image)
    ys = slice(max(-dy, 0), min(h - dy, h))
    xs = slice(max(-dx, 0), min(w - dx, w))
    yd = slice(max(dy, 0), min(h + dy, h))
    xd = slice(max(dx, 0), min(w + dx, w))
    out[yd, xd] = image[ys, xs]
    return out


def summary_empty():
    return {"n": 0, "sum": 0.0, "high": 0, "hist": np.zeros(10, dtype=np.int64)}


def add(group, values):
    values = np.asarray(values, dtype=np.float32)
    if not values.size:
        return
    group["n"] += int(values.size)
    group["sum"] += float(values.sum(dtype=np.float64))
    group["high"] += int((values >= .5).sum())
    group["hist"] += np.histogram(values, bins=np.linspace(0, 1, 11))[0]


def present(group):
    n = group["n"]
    return {"n": n, "mean_O": group["sum"] / n if n else None,
            "fraction_O_ge_0_5": group["high"] / n if n else None,
            "O_ge_0_5": group["high"], "histogram_0_0p1_to_0p9_1": group["hist"].tolist()}


def audit_split(root, split, generator, h0_root, out, device):
    dataset = NativeWristDataset(root, split, augment=False)
    expected = {"train": 425, "val": 69}[split]
    if len(dataset) != expected:
        raise ValueError(f"unexpected {split} cases {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    groups = {name: summary_empty() for name in (
        "GT_overlap_unique", "GT_single_unique", "GT_background_unique",
        "H0_member", "H0_unique", "GT_overlap_without_H0_unique")}
    side = {s: {name: summary_empty() for name in groups} for s in ("L", "R")}
    radius = {r: {"H0_unique_ge_0_5": 0, "H0_unique_n": 0,
                  "H0_member_ge_0_5": 0, "H0_member_n": 0} for r in RADII}
    h0_dist = []
    gt_depth_h0 = []
    gt_depth_easy = []
    shift_counts = {(dy, dx): [0, 0, 0] for dy in SHIFTS for dx in SHIFTS}
    per_case = []
    h0_total = 0
    for batch in loader:
        case = str(batch["case"][0])
        image = batch["image"].to(device)
        with torch.inference_mode():
            o = generator(image).sigmoid().float().cpu().numpy()[0, 0]
        target = batch["mask"].numpy()[0] > .5
        valid = batch["valid"].numpy()[0, 0].astype(bool)
        hw = tuple(int(x) for x in batch["original_hw"].numpy()[0])
        h0_path = h0_root / split / f"{case}.npz"
        if not h0_path.is_file(): raise FileNotFoundError(h0_path)
        h0 = load_h0(h0_path, target.shape, hw)
        membership = target.sum(0)
        gt = (membership >= 2) & valid
        single = (membership == 1) & valid
        background = (membership == 0) & valid
        unique_h0 = h0.any(0) & valid
        if np.any(h0 & ~(target & gt[None] & valid[None])):
            raise ValueError(f"H0 cache contains non-GT-overlap member: {case}")
        h0_total += int(h0.sum())
        o_valid = np.where(valid, o, 0.0)
        if not np.isfinite(o_valid).all() or o_valid.min() < 0 or o_valid.max() > 1:
            raise ValueError(f"invalid O map: {case}")
        case_groups = {
            "GT_overlap_unique": o[gt],
            "GT_single_unique": o[single],
            "GT_background_unique": o[background],
            "H0_member": np.broadcast_to(o, target.shape)[h0],
            "H0_unique": o[unique_h0],
            "GT_overlap_without_H0_unique": o[gt & ~unique_h0],
        }
        direction = case.rsplit("_", 1)[-1]
        if direction not in side: raise ValueError(case)
        for name, values in case_groups.items():
            add(groups[name], values)
            add(side[direction][name], values)
        for r in RADII:
            maximum = o if r == 0 else maximum_filter(o, size=2*r+1, mode="constant", cval=0)
            high = maximum >= .5
            radius[r]["H0_unique_ge_0_5"] += int((high & unique_h0).sum())
            radius[r]["H0_unique_n"] += int(unique_h0.sum())
            radius[r]["H0_member_ge_0_5"] += int(np.broadcast_to(high, target.shape)[h0].sum())
            radius[r]["H0_member_n"] += int(h0.sum())
        if unique_h0.any():
            # Zero at O>=.5; values quantify whether the high-O region lies
            # nearby rather than exactly on the known hard-FN coordinate.
            dist = distance_transform_edt(~(o >= .5))
            h0_dist.extend(dist[unique_h0].astype(np.float32).tolist())
        depth = distance_transform_edt(gt)
        gt_depth_h0.extend(depth[unique_h0].astype(np.float32).tolist())
        easy_depth = depth[gt & ~unique_h0].astype(np.float32)
        if easy_depth.size > 4096:
            easy_depth = easy_depth[np.linspace(0, easy_depth.size-1, 4096, dtype=np.int64)]
        gt_depth_easy.extend(easy_depth.tolist())
        if split == "val":
            pred = (o >= .5) & valid
            for (dy, dx), counts in shift_counts.items():
                moved = shifted(pred, dy, dx) & valid
                counts[0] += int((moved & gt).sum())
                counts[1] += int((moved & ~gt & valid).sum())
                counts[2] += int((~moved & gt).sum())
        per_case.append({"case": case, "side": direction, "H0_member_n": int(h0.sum()),
                         "H0_unique_n": int(unique_h0.sum()),
                         "H0_member_mean_O": float(case_groups["H0_member"].mean()) if h0.any() else None,
                         "H0_member_fraction_ge_0_5": float((case_groups["H0_member"] >= .5).mean()) if h0.any() else None,
                         "GT_overlap_n": int(gt.sum()),
                         "GT_overlap_mean_O": float(o[gt].mean()) if gt.any() else None,
                         "GT_overlap_recall_at_0_5": float(((o >= .5) & gt).sum()/gt.sum()) if gt.any() else None,
                         "GT_single_high_O_n": int(((o >= .5) & single).sum()),
                         "GT_background_high_O_n": int(((o >= .5) & background).sum())})
    expected_h0 = {"train": 228695, "val": 71335}[split]
    if h0_total != expected_h0:
        raise ValueError(f"H0 member count differs: {h0_total} vs {expected_h0}")
    def quantiles(values):
        return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, .9, .95)} if values else None
    g = {name: present(value) for name, value in groups.items()}
    overlap_high = g["GT_overlap_unique"]["O_ge_0_5"]
    total_high = sum(g[name]["O_ge_0_5"] for name in ("GT_overlap_unique", "GT_single_unique", "GT_background_unique"))
    result = {"split": split, "cases": len(per_case), "test_used": False,
              "H0_member_total": h0_total, "groups": g,
              "sides": {s: {name: present(value) for name, value in rows.items()} for s, rows in side.items()},
              "GT_overlap_precision_at_0_5": overlap_high / total_high,
              "GT_overlap_recall_at_0_5": overlap_high / g["GT_overlap_unique"]["n"],
              "nearby_high_O": {str(r): {**row,
                  "H0_unique_fraction": row["H0_unique_ge_0_5"] / row["H0_unique_n"],
                  "H0_member_fraction": row["H0_member_ge_0_5"] / row["H0_member_n"]}
                  for r, row in radius.items()},
              "distance_H0_to_O_ge_0_5_px_quantiles": quantiles(h0_dist),
              "distance_inside_GT_overlap_H0_px_quantiles": quantiles(gt_depth_h0),
              "distance_inside_GT_overlap_other_px_quantiles": quantiles(gt_depth_easy),
              "distance_inside_GT_overlap_other_sampling": "deterministic at most 4096 pixels per case",
              "per_case": per_case}
    if split == "val":
        shifts = []
        for (dy, dx), (tp, fp, fn) in shift_counts.items():
            shifts.append({"dy": dy, "dx": dx, "tp": tp, "fp": fp, "fn": fn,
                           "dice": 2*tp/(2*tp+fp+fn), "iou": tp/(tp+fp+fn)})
        result["shift_diagnostic"] = {
            "note": "GT-referenced read-only diagnostic; no shifted O used in training or evaluation",
            "unshifted": next(x for x in shifts if x["dy"] == 0 and x["dx"] == 0),
            "best_dice_shift": max(shifts, key=lambda x: x["dice"]),
            "all_shifts": shifts,
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split.upper()}_O_GEOMETRY.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": split, "cases": len(per_case), "H0": h0_total,
                      "global_precision": result["GT_overlap_precision_at_0_5"],
                      "global_recall": result["GT_overlap_recall_at_0_5"],
                      "H0_mean_O": g["H0_member"]["mean_O"],
                      "H0_ge0_5": g["H0_member"]["fraction_O_ge_0_5"]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--overlap", type=Path, required=True)
    ap.add_argument("--h0-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "val", "both"), default="both")
    a = ap.parse_args()
    if sha(a.overlap) != EXPECTED_O_SHA:
        raise ValueError("wrong image-only O checkpoint")
    payload = torch.load(a.overlap, map_location="cpu", weights_only=False)
    if payload["input_dependencies"] != ["image"]:
        raise ValueError("generator is not image-only")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageOnlyOverlapPrior().to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    torch.set_num_threads(16)
    for split in (("train", "val") if a.split == "both" else (a.split,)):
        audit_split(a.dataset, split, model, a.h0_root, a.output, device)


if __name__ == "__main__": main()
