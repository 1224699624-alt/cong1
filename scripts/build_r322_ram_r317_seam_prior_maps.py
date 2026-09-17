#!/usr/bin/env python3
"""Build GT-free RAM-W600 seam priors with frozen R317 relation models.

Instance centers and pair scales come only from a frozen RAM nnU-Net prediction.
No train/validation target is read by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label

from build_r259_frozen_prior_maps import crop_input, load_checkpoint_portable
from train_r258b_prediction_relation_prior import RelationPriorNet
from train_r284_ramw600_overlap_prior import NnUNetMultiLabelPrior


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_image(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize((size, size), Image.Resampling.BILINEAR)
    value = np.asarray(image)
    if path.stem.endswith("_R"):
        value = np.fliplr(value)
    return np.ascontiguousarray(value)


def prediction_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.from_numpy(image.astype(np.float32) / 255.0)[None, None]
    value = (value - value.mean((2, 3), keepdim=True)) / (value.std((2, 3), keepdim=True) + 1e-6)
    return value.to(device)


def predicted_instances(probability: np.ndarray, threshold: float) -> list[dict]:
    result: list[dict] = []
    for channel, values in enumerate(probability):
        components, count = label(values >= threshold)
        if count == 0:
            continue
        ids, sizes = np.unique(components[components > 0], return_counts=True)
        component = int(ids[int(np.argmax(sizes))])
        support = components == component
        if int(support.sum()) < 16:
            continue
        weights = values * support
        yy, xx = np.indices(values.shape)
        mass = float(weights.sum()) + 1e-6
        x = float((weights * xx).sum() / mass)
        y = float((weights * yy).sum() / mass)
        scale = float(max(3.0, np.sqrt(float(support.sum()) / np.pi)))
        result.append({"channel": channel, "point": np.array([x, y]), "scale": scale})
    return result


def nearest_pairs(instances: list[dict], neighbors: int, relative_limit: float) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i, item in enumerate(instances):
        ranked = sorted((float(np.linalg.norm(item["point"] - other["point"])), j)
                        for j, other in enumerate(instances) if j != i)
        for distance, j in ranked[:neighbors]:
            other = instances[j]
            # Scale-normalized restriction avoids unrelated long-range bone pairs.
            if distance / max(item["scale"] + other["scale"], 1e-6) <= relative_limit:
                pairs.add(tuple(sorted((i, j))))
    return sorted(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path(r"G:\gutou\RAM-W600"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--r317-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/priors/r322_ram_r317_seam"))
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--nearest-neighbors", type=int, default=3)
    parser.add_argument("--relative-pair-limit", type=float, default=2.5)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if "Articular-Surface" in str(args.dataset_root):
        raise RuntimeError("R322 is RAM-W600 only")
    device = torch.device(args.device)

    baseline = NnUNetMultiLabelPrior().to(device)
    payload = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline.load_state_dict(payload["model"], strict=True)
    baseline.eval()

    paths = sorted(args.r317_checkpoint_dir.glob("image_centers_seed*_selected.pt"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected exact R317 three-seed ensemble, got {paths}")
    priors = []
    checkpoint_rows = []
    for path in paths:
        p = load_checkpoint_portable(path)
        if p.get("use_development") is not False or int(p.get("selected_epoch", -1)) < 1:
            raise RuntimeError(f"Not an image-centers-only selected R317 checkpoint: {path}")
        model = RelationPriorNet(4).to(device)
        model.load_state_dict(p["model"], strict=True)
        model.eval()
        priors.append(model)
        checkpoint_rows.append({"path": str(path), "seed": int(p["seed"]),
                                "selected_epoch": int(p["selected_epoch"]), "sha256": sha256(path)})

    splits = [("train", args.limit_train), ("val", args.limit_val)]
    audit = {"experiment": "R322_RAM_R317_SEAM_PRIOR", "dataset": str(args.dataset_root),
             "source": "prediction-derived centers; GT-free map generation",
             "baseline_checkpoint": str(args.baseline_checkpoint),
             "baseline_sha256": sha256(args.baseline_checkpoint), "r317_checkpoints": checkpoint_rows,
             "size": args.size, "threshold": args.threshold,
             "nearest_neighbors": args.nearest_neighbors,
             "relative_pair_limit": args.relative_pair_limit, "splits": {}}
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split, limit in splits:
        image_paths = sorted((args.dataset_root / "BoneSegmentation" / "images").glob("*.bmp"))
        allowed = {p.stem for p in (args.dataset_root / "BoneSegmentation" / "masks" / split).glob("*.npy")}
        image_paths = [p for p in image_paths if p.stem in allowed][:limit or None]
        out = args.output_root / split
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, path in enumerate(image_paths, 1):
            image = canonical_image(path, args.size)
            with torch.no_grad():
                logits, _ = baseline(prediction_tensor(image, device), False)
                probability = torch.sigmoid(logits)[0].cpu().numpy()
            instances = predicted_instances(probability, args.threshold)
            pairs = nearest_pairs(instances, args.nearest_neighbors, args.relative_pair_limit)
            ensemble_maps = []
            for model in priors:
                full = np.zeros((args.size, args.size), np.float32)
                for i, j in pairs:
                    a, b = instances[i], instances[j]
                    channels, (x0, y0, side) = crop_input(
                        image, a["point"], b["point"], a["scale"], b["scale"], 256,
                        True, 128, .15, 3., 12.)
                    feature = torch.zeros((1, 2), dtype=torch.float32, device=device)
                    with torch.no_grad():
                        relation_logit, heat_logit = model(
                            torch.from_numpy(channels[None]).to(device), feature)
                        local = (torch.sigmoid(relation_logit)[:, None, None, None]
                                 * torch.sigmoid(heat_logit))[0, 0].cpu().numpy()
                    patch = cv2.resize(local, (side, side), interpolation=cv2.INTER_LINEAR)
                    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(args.size, x0 + side), min(args.size, y0 + side)
                    if sx1 > sx0 and sy1 > sy0:
                        crop = patch[sy0-y0:sy1-y0, sx0-x0:sx1-x0]
                        full[sy0:sy1, sx0:sx1] = np.maximum(full[sy0:sy1, sx0:sx1], crop)
                ensemble_maps.append(full)
            prior = np.mean(ensemble_maps, axis=0).astype(np.float32)
            np.save(out / f"{path.stem}.npy", prior)
            Image.fromarray(np.clip(prior * 255, 0, 255).astype(np.uint8)).save(out / f"{path.stem}.png")
            rows.append({"case": path.stem, "instances": len(instances), "pairs": len(pairs),
                         "mean": float(prior.mean()), "max": float(prior.max()),
                         "nonzero_fraction": float((prior > 0).mean())})
            if index % 25 == 0 or index == len(image_paths):
                print(json.dumps({"split": split, "done": index, "total": len(image_paths)}), flush=True)
        audit["splits"][split] = {"count": len(rows), "mean_pairs": float(np.mean([r["pairs"] for r in rows])),
                                  "mean_prior": float(np.mean([r["mean"] for r in rows])), "rows": rows}
    (args.output_root / "manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(args.output_root / 'manifest.json'), "splits": {
        k: v["count"] for k, v in audit["splits"].items()}}, indent=2))


if __name__ == "__main__":
    main()
