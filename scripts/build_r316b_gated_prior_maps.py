#!/usr/bin/env python3
"""Build GT-free full-image R316B gated seam priors for full train/original-val.

The map keeps only pair predictions above a frozen relation threshold and
attenuates seam responses inside the predicted two-sided bone support. Ground
truth is never read during map generation.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from build_r259_frozen_prior_maps import crop_input, load_checkpoint_portable
from train_r256_scale_invariant_pair_prior import find_image, read_metadata, set_seed
from train_r258b_prediction_relation_prior import load_proposals, pair_geometry, sha256
from train_r316_safe_relation_selector import SafeRelationNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-root", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--train-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_train.csv"))
    p.add_argument("--val-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_val.csv"))
    p.add_argument("--train-proposals", type=Path, default=Path("outputs/analysis/r258_oof_center_basin_full_proposals.csv"))
    p.add_argument("--val-proposals", type=Path, default=Path("outputs/analysis/r257b_scale_repair_fullval_proposals.csv"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("outputs/priors/r319_r316b_gated_relation"))
    p.add_argument("--threshold", type=float, default=0.65)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--nearest-neighbors", type=int, default=4)
    p.add_argument("--proposal-distance-limit", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--r316c", action="store_true")
    p.add_argument("--global-size", type=int, default=128)
    p.add_argument("--center-sigma-rel", type=float, default=.15)
    p.add_argument("--center-sigma-min", type=float, default=3.)
    p.add_argument("--center-sigma-max", type=float, default=12.)
    p.add_argument("--png-only", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args(); set_seed(3161)
    joined = " ".join(map(str, vars(a).values())).lower()
    if a.variant_root.name != "TSRS_RSNA-Epiphysis_contrast_v1" or "clean-test" in joined or "articular" in joined:
        raise RuntimeError("R316B map generation is restricted to full Epiphysis train/original-val")
    expected_crop = 256 if a.r316c else 128
    if abs(a.threshold - 0.65) > 1e-12 or a.crop_size != expected_crop:
        raise RuntimeError(f"R316B/R316C freezes threshold=0.65 and crop_size={expected_crop}")

    payload = load_checkpoint_portable(a.checkpoint)
    config = payload.get("config", {})
    if int(config.get("seed", -1)) != 3161 or float(config.get("hard_negative_quantile", -1)) != 0.20:
        raise RuntimeError("Unexpected R316B checkpoint protocol")
    if not bool(config.get("require_full_data", False)) or len(payload.get("history", [])) != 8:
        raise RuntimeError("Checkpoint is not the completed full-data R316B run")
    if bool(config.get("r316c", False)) != a.r316c:
        raise RuntimeError("R316B/R316C checkpoint representation mismatch")
    if a.r316c:
        expected = {"crop_size": 256, "global_size": 128, "center_sigma_rel": .15,
                    "center_sigma_min": 3., "center_sigma_max": 12.}
        if any(config.get(k) != v for k, v in expected.items()):
            raise RuntimeError("Unexpected R316C gated-prior protocol")
    model = SafeRelationNet(4 if a.r316c else 3); model.load_state_dict(payload["model"], strict=True); model.to(a.device).eval()

    audit = {
        "experiment": "R316C_R316B_GATED_PRIOR_MAPS" if a.r316c else "R319_R316B_GATED_PRIOR_MAPS",
        "checkpoint": str(a.checkpoint), "checkpoint_sha256": sha256(a.checkpoint),
        "threshold": a.threshold, "support_suppression": "1-max(sigmoid(support_a),sigmoid(support_b))",
        "clean_test_used": False, "splits": {},
    }
    for split, meta_path, proposal_path, expected in (
        ("train", a.train_metadata, a.train_proposals, 875),
        ("val", a.val_metadata, a.val_proposals, 96),
    ):
        metadata = read_metadata(meta_path); stems = sorted(metadata)
        if len(stems) != expected:
            raise RuntimeError(f"Expected full {split} count {expected}, got {len(stems)}")
        proposals = load_proposals(proposal_path, a.variant_root, split)
        if set(proposals) != set(stems):
            raise RuntimeError(f"Proposal coverage mismatch for {split}")
        out_dir = a.output_root / split; out_dir.mkdir(parents=True, exist_ok=True)
        total_pairs = selected_pairs = nonzero_maps = 0
        score_sum = support_suppression_sum = 0.0
        for n, stem in enumerate(stems, 1):
            image = np.asarray(Image.open(find_image(a.variant_root / split, stem)).convert("L"))
            ps = proposals[stem]; pairs: set[tuple[int, int]] = set()
            for i, p in enumerate(ps):
                ranked = sorted((math.hypot(p["x"] - q["x"], p["y"] - q["y"]), j) for j, q in enumerate(ps) if j != i)
                for _, j in ranked[: a.nearest_neighbors]:
                    pair = tuple(sorted((i, j)))
                    if pair_geometry(ps[pair[0]], ps[pair[1]])[0] <= a.proposal_distance_limit:
                        pairs.add(pair)
            pairs = sorted(pairs); total_pairs += len(pairs); full = np.zeros(image.shape, np.float32)
            for start in range(0, len(pairs), a.batch_size):
                chunk = pairs[start : start + a.batch_size]; inputs, placements = [], []
                for i, j in chunk:
                    p, q = ps[i], ps[j]
                    ch, place = crop_input(image, np.array([p["x"], p["y"]]), np.array([q["x"], q["y"]]), p["scale"], q["scale"], a.crop_size,
                                           a.r316c, a.global_size, a.center_sigma_rel, a.center_sigma_min, a.center_sigma_max)
                    inputs.append(ch); placements.append(place)
                if not inputs:
                    continue
                features = torch.tensor([[metadata[stem][0] / 240.0, metadata[stem][1]]] * len(inputs), dtype=torch.float32, device=a.device)
                with torch.no_grad():
                    logit, seam, support = model(torch.from_numpy(np.stack(inputs)).to(a.device), features)
                    score = torch.sigmoid(logit)
                    keep = score >= a.threshold
                    suppression = 1.0 - torch.sigmoid(support).amax(1)
                    values = (score[:, None, None] * torch.sigmoid(seam[:, 0]) * suppression * keep[:, None, None]).cpu().numpy()
                    selected_pairs += int(keep.sum().item()); score_sum += float(score[keep].sum().item())
                    if keep.any():
                        support_suppression_sum += float((1.0 - suppression[keep]).mean((1, 2)).sum().item())
                for value, (x0, y0, side) in zip(values, placements):
                    if float(value.max()) <= 0:
                        continue
                    patch = cv2.resize(value, (side, side), interpolation=cv2.INTER_LINEAR)
                    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(image.shape[1], x0 + side), min(image.shape[0], y0 + side)
                    if sx1 > sx0 and sy1 > sy0:
                        local = patch[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
                        full[sy0:sy1, sx0:sx1] = np.maximum(full[sy0:sy1, sx0:sx1], local)
            if not a.png_only:
                np.save(out_dir / f"{stem}.npy", full)
            Image.fromarray(np.clip(full * 255, 0, 255).astype(np.uint8)).save(out_dir / f"{stem}.png")
            nonzero_maps += int(float(full.max()) > 0)
            if n % 50 == 0 or n == len(stems):
                print(json.dumps({"split": split, "done": n, "total": len(stems), "selected_pairs": selected_pairs}), flush=True)
        audit["splits"][split] = {
            "images": expected, "candidate_pairs": total_pairs, "selected_pairs": selected_pairs,
            "selection_rate": selected_pairs / max(total_pairs, 1), "nonzero_maps": nonzero_maps,
            "mean_selected_score": score_sum / max(selected_pairs, 1),
            "mean_support_suppression": support_suppression_sum / max(selected_pairs, 1),
        }
    a.output_root.mkdir(parents=True, exist_ok=True)
    (a.output_root / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
