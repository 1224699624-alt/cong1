#!/usr/bin/env python3
"""R256 noisy-proposal, development-conditioned local bone-pair prior pilot.

This is an oracle-proposal mechanism study, not a deployable segmenter. Ground
truth is used to construct proposal labels and heat targets, but every model
input/crop/geometry feature is derived from a reproducibly perturbed proposal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset


EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-root", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_contrast_v1"))
    p.add_argument("--train-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_train.csv"))
    p.add_argument("--val-metadata", type=Path, default=Path("outputs/metadata/r256/filtered_val.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/pair_prior/r256_scale_invariant_pair_prior"))
    p.add_argument("--result-json", type=Path, default=Path("outputs/analysis/r256_scale_invariant_pair_prior_result.json"))
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-positive-per-case", type=int, default=32)
    p.add_argument("--max-negative-per-case", type=int, default=32)
    p.add_argument("--relative-pair-limit", type=float, default=0.8)
    p.add_argument("--relative-close-threshold", type=float, default=0.20)
    p.add_argument("--center-noise-rel", type=float, default=0.08)
    p.add_argument("--scale-noise-log-std", type=float, default=0.12)
    p.add_argument("--crop-margin-scales", type=float, default=1.35)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--seed", type=int, default=256)
    p.add_argument("--train-seeds", default="256,257,258")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def read_metadata(path: Path) -> dict[str, tuple[float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            key = str(row["id"]).strip()
            rows[key] = (float(row["boneage"]), float(str(row["male"]).strip().lower() in {"true", "1", "yes"}))
        return rows


def find_image(folder: Path, stem: str) -> Path:
    for extension in EXTENSIONS:
        path = folder / f"{stem}{extension}"
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(stem)


def read_instance(path: Path) -> np.ndarray:
    value = np.asarray(Image.open(path))
    return (value[..., 0] if value.ndim == 3 else value).astype(np.int32)


@lru_cache(maxsize=32)
def read_instance_cached(path: str) -> np.ndarray:
    return read_instance(Path(path))


@lru_cache(maxsize=64)
def read_gray_cached(path: str) -> np.ndarray:
    value = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise ValueError(f"Cannot read {path}")
    return value


def instance_records(instance: np.ndarray, min_area: int = 24) -> list[dict[str, Any]]:
    result = []
    for value in [int(v) for v in np.unique(instance) if int(v) > 0]:
        mask = instance == value
        yy, xx = np.nonzero(mask)
        if len(xx) < min_area:
            continue
        points = np.stack([yy, xx], axis=1).astype(np.float64)
        centered = points - points.mean(0, keepdims=True)
        covariance = centered.T @ centered / max(1, len(points) - 1)
        _, eigvecs = np.linalg.eigh(covariance)
        major = eigvecs[:, -1]
        boundary = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        by, bx = np.nonzero(boundary)
        if len(bx) > 384:
            take = np.linspace(0, len(bx) - 1, 384).astype(int)
            by, bx = by[take], bx[take]
        result.append({"id": value, "area": float(len(xx)), "cy": float(yy.mean()), "cx": float(xx.mean()),
                       "major_y": float(major[0]), "major_x": float(major[1]),
                       "boundary": np.stack([by, bx], axis=1), "pixels_y": yy, "pixels_x": xx})
    return result


def boundary_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    distance, _ = cKDTree(b["boundary"]).query(a["boundary"], k=1)
    return max(0.0, float(distance.min()) - 1.0)


def split_same_instance(mask: np.ndarray, major_y: float, major_x: float, quantile: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(mask)
    cy, cx = yy.mean(), xx.mean()
    projection = (yy - cy) * major_y + (xx - cx) * major_x
    first, second = np.zeros_like(mask), np.zeros_like(mask)
    threshold = float(np.quantile(projection, quantile))
    first[yy[projection <= threshold], xx[projection <= threshold]] = True
    second[yy[projection > threshold], xx[projection > threshold]] = True
    return first, second


def mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, float]:
    yy, xx = np.nonzero(mask)
    return np.asarray([float(xx.mean()), float(yy.mean())]), float(len(xx))


def split_record_geometry(record: dict[str, Any], quantile: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    yy, xx = record["pixels_y"], record["pixels_x"]
    projection = (yy - record["cy"]) * record["major_y"] + (xx - record["cx"]) * record["major_x"]
    first = projection <= float(np.quantile(projection, quantile))
    second = ~first
    ca = np.asarray([float(xx[first].mean()), float(yy[first].mean())])
    cb = np.asarray([float(xx[second].mean()), float(yy[second].mean())])
    return ca, cb, float(first.sum()), float(second.sum())


def noisy_proposal(ca: np.ndarray, cb: np.ndarray, area_a: float, area_b: float,
                   seed: int, center_noise_rel: float, scale_noise_log_std: float) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scale_a, scale_b = math.sqrt(max(area_a, 1.0)), math.sqrt(max(area_b, 1.0))
    pa = ca + rng.normal(0.0, center_noise_rel * scale_a, 2)
    pb = cb + rng.normal(0.0, center_noise_rel * scale_b, 2)
    proposal_scale_a = scale_a * math.exp(float(rng.normal(0.0, scale_noise_log_std)))
    proposal_scale_b = scale_b * math.exp(float(rng.normal(0.0, scale_noise_log_std)))
    relative_distance = float(np.linalg.norm(pb - pa) / max(math.sqrt(0.5 * (proposal_scale_a**2 + proposal_scale_b**2)), 1.0))
    return {"pa": pa.tolist(), "pb": pb.tolist(), "psa": proposal_scale_a, "psb": proposal_scale_b,
            "proposal_relative_distance": relative_distance,
            "proposal_abs_log_scale_ratio": abs(math.log(max(proposal_scale_a, 1e-6) / max(proposal_scale_b, 1e-6)))}


def build_samples(root: Path, split: str, metadata: dict[str, tuple[float, float]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    label_paths = sorted((root / f"{split}_labels").glob("*.png"))
    limit = args.limit_train if split == "train" else args.limit_val
    if limit > 0:
        label_paths = label_paths[:limit]
    samples: list[dict[str, Any]] = []
    audit = {"eligible_pairs": 0, "eligible_close_pairs": 0, "eligible_contact_pairs": 0,
             "selected_pairs": 0, "selected_close_pairs": 0, "selected_contact_pairs": 0,
             "included_cases": 0}
    for label_path in label_paths:
        stem = label_path.stem
        if stem not in metadata:
            raise KeyError(f"Missing metadata: {stem}")
        instance = read_instance(label_path)
        recs = instance_records(instance)
        if len(recs) < 2:
            continue
        positives = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                gap = boundary_gap(a, b)
                local_scale = math.sqrt(0.5 * (a["area"] + b["area"]))
                relative_gap = gap / max(local_scale, 1.0)
                if relative_gap <= args.relative_pair_limit:
                    center_distance = math.hypot(a["cy"] - b["cy"], a["cx"] - b["cx"])
                    positives.append((relative_gap, center_distance, a, b, gap))
        positives.sort(key=lambda item: (item[0], item[1]))
        audit["eligible_pairs"] += len(positives)
        audit["eligible_close_pairs"] += sum(x[0] <= args.relative_close_threshold for x in positives)
        audit["eligible_contact_pairs"] += sum(x[4] < 1.0 for x in positives)
        cap_pos = len(positives) if args.max_positive_per_case <= 0 else args.max_positive_per_case
        cap_neg = len(recs) if args.max_negative_per_case <= 0 else args.max_negative_per_case
        pair_count = min(len(positives), cap_pos, cap_neg)
        if pair_count == 0:
            continue
        selected_positives = positives[:pair_count]
        audit["included_cases"] += 1
        audit["selected_pairs"] += pair_count
        audit["selected_close_pairs"] += sum(x[0] <= args.relative_close_threshold for x in selected_positives)
        audit["selected_contact_pairs"] += sum(x[4] < 1.0 for x in selected_positives)
        image_path = find_image(root / split, stem)
        age, male = metadata[stem]
        for rank, (relative_gap, _, a, b, gap) in enumerate(selected_positives):
            proposal = noisy_proposal(np.asarray([a["cx"], a["cy"]]), np.asarray([b["cx"], b["cy"]]),
                                      a["area"], b["area"], stable_seed(args.seed, split, stem, "pos", rank),
                                      args.center_noise_rel, args.scale_noise_log_std)
            samples.append({"split": split, "stem": stem, "image": str(image_path), "label": str(label_path),
                            "a": a["id"], "b": b["id"], "same": False, "gap": gap,
                            "relative_gap": relative_gap, "age": age, "male": male, **proposal})
        # Match pseudo-negative geometry to each positive before applying the same
        # proposal noise, so age cannot exploit an obvious distance/scale shortcut.
        candidate_negatives = []
        for record in sorted(recs, key=lambda r: -r["area"]):
            for split_quantile in (0.20, 0.35, 0.50, 0.65, 0.80):
                ca, cb, area_a, area_b = split_record_geometry(record, split_quantile)
                norm_dist = float(np.linalg.norm(cb - ca) / max(math.sqrt(0.5 * (area_a + area_b)), 1.0))
                abs_log_ratio = abs(0.5 * math.log(max(area_a, 1.0) / max(area_b, 1.0)))
                candidate_negatives.append((record, split_quantile, ca, cb, area_a, area_b, norm_dist, abs_log_ratio))
        available = list(range(len(candidate_negatives)))
        for rank, (_, _, positive_a, positive_b, _) in enumerate(selected_positives):
            positive_norm_dist = math.hypot(positive_a["cx"] - positive_b["cx"], positive_a["cy"] - positive_b["cy"]) / max(math.sqrt(0.5 * (positive_a["area"] + positive_b["area"])), 1.0)
            positive_abs_log_ratio = abs(0.5 * math.log(max(positive_a["area"], 1.0) / max(positive_b["area"], 1.0)))
            pool = available if available else list(range(len(candidate_negatives)))
            chosen = min(pool, key=lambda idx: (candidate_negatives[idx][6] - positive_norm_dist) ** 2 + (candidate_negatives[idx][7] - positive_abs_log_ratio) ** 2)
            if chosen in available:
                available.remove(chosen)
            record, split_quantile, ca, cb, area_a, area_b, _, _ = candidate_negatives[chosen]
            proposal = noisy_proposal(ca, cb, area_a, area_b, stable_seed(args.seed, split, stem, "neg", rank),
                                      args.center_noise_rel, args.scale_noise_log_std)
            samples.append({"split": split, "stem": stem, "image": str(image_path), "label": str(label_path),
                            "a": record["id"], "b": record["id"], "same": True, "gap": -1.0,
                            "relative_gap": -1.0, "age": age, "male": male,
                            "major_y": record["major_y"], "major_x": record["major_x"],
                            "split_quantile": split_quantile, **proposal})
    for key in ("pairs", "close_pairs", "contact_pairs"):
        selected, eligible = audit[f"selected_{key}"], audit[f"eligible_{key}"]
        audit[f"proposal_{key}_coverage"] = float(selected / eligible) if eligible else 1.0
    positive_geometry = np.asarray([[s["proposal_relative_distance"], s["proposal_abs_log_scale_ratio"]] for s in samples if not s["same"]])
    negative_geometry = np.asarray([[s["proposal_relative_distance"], s["proposal_abs_log_scale_ratio"]] for s in samples if s["same"]])
    if len(positive_geometry) and len(negative_geometry):
        pooled_std = np.std(np.concatenate([positive_geometry, negative_geometry]), axis=0) + 1e-6
        standardized_difference = np.abs(positive_geometry.mean(0) - negative_geometry.mean(0)) / pooled_std
        audit["proposal_geometry_max_standardized_mean_difference"] = float(standardized_difference.max())
    return samples, {k: float(v) for k, v in audit.items()}


def gaussian(size: int, x: float, y: float, sigma: float = 5.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)).astype(np.float32)


def point_line_distance(size: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    ab = b - a
    denom = max(float(ab @ ab), 1e-6)
    t = np.clip(((xx - a[0]) * ab[0] + (yy - a[1]) * ab[1]) / denom, 0.0, 1.0)
    return np.sqrt((xx - (a[0] + t * ab[0])) ** 2 + (yy - (a[1] + t * ab[1])) ** 2)


def transform_direction(dx: float, dy: float, horizontal_flip: bool = False,
                        vertical_flip: bool = False, turns: int = 0) -> tuple[float, float]:
    """Transform an image-coordinate direction under flips and np.rot90 CCW turns."""
    if horizontal_flip:
        dx = -dx
    if vertical_flip:
        dy = -dy
    for _ in range(turns % 4):
        dx, dy = dy, -dx
    return dx, dy


def make_heat_target(mask_a_crop: np.ndarray, mask_b_crop: np.ndarray, full_foreground_crop: np.ndarray,
                     output_size: int, center_a: np.ndarray, center_b: np.ndarray,
                     proposal_scale_pixels: float) -> tuple[np.ndarray, bool]:
    """Create a separation target and strictly clear every GT foreground pixel after resize."""
    distance_a = cv2.distanceTransform((~mask_a_crop).astype(np.uint8), cv2.DIST_L2, 5)
    distance_b = cv2.distanceTransform((~mask_b_crop).astype(np.uint8), cv2.DIST_L2, 5)
    # Broad pair-scale target: the eligible-pair rule allows gaps up to 0.8 local
    # scale, so each side must reach beyond half that distance to overlap.
    radius = max(3.0, 0.55 * math.sqrt(0.5 * (float(mask_a_crop.sum()) + float(mask_b_crop.sum()))))
    raw = np.exp(-np.abs(distance_a-distance_b)/(0.35*radius+1e-6) - (distance_a+distance_b)/(2*radius+1e-6)).astype(np.float32)
    raw *= (~(mask_a_crop | mask_b_crop)) & (distance_a <= radius) & (distance_b <= radius)
    target = cv2.resize(raw, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    resized_foreground = cv2.resize(full_foreground_crop.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_NEAREST).astype(bool)
    corridor = point_line_distance(output_size, center_a, center_b) <= max(3.0, 0.45 * proposal_scale_pixels)
    target *= (~resized_foreground) & corridor
    target[target < 1e-4] = 0.0
    assert not np.any(target[resized_foreground] != 0), "heat target leaked into GT foreground"
    return target, bool(target.sum() > 1e-3)


class PairDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], args: argparse.Namespace, augment: bool, use_development: bool):
        self.samples, self.args, self.augment, self.use_development = samples, args, augment, use_development
        cases = sorted({str(sample["stem"]) for sample in samples})
        shift = max(1, len(cases) // 2)
        case_metadata = {str(sample["stem"]): (float(sample["age"]), float(sample["male"])) for sample in samples}
        self.shuffled_metadata = {case: case_metadata[cases[(i + shift) % len(cases)]] for i, case in enumerate(cases)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image = read_gray_cached(sample["image"])
        instance = read_instance_cached(sample["label"])
        mask_a = instance == int(sample["a"])
        if sample["same"]:
            mask_a, mask_b = split_same_instance(mask_a, float(sample["major_y"]), float(sample["major_x"]), float(sample["split_quantile"]))
        else:
            mask_b = instance == int(sample["b"])

        pa, pb = np.asarray(sample["pa"], dtype=np.float64), np.asarray(sample["pb"], dtype=np.float64)
        psa, psb = float(sample["psa"]), float(sample["psb"])
        center = 0.5 * (pa + pb)
        side = int(math.ceil(max(32.0, np.linalg.norm(pb - pa) + self.args.crop_margin_scales * (psa + psb))))
        x0, y0 = int(round(center[0] - side / 2)), int(round(center[1] - side / 2))
        x1, y1 = x0 + side, y0 + side

        def crop_pad(array: np.ndarray) -> np.ndarray:
            out = np.zeros((side, side), dtype=array.dtype)
            sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(array.shape[1], x1), min(array.shape[0], y1)
            if sx1 > sx0 and sy1 > sy0:
                out[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = array[sy0:sy1, sx0:sx1]
            return out

        size = self.args.crop_size
        image_crop = cv2.resize(crop_pad(image), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        a_crop, b_crop = crop_pad(mask_a), crop_pad(mask_b)
        scale_xy = size / side
        center_a = (pa - [x0, y0]) * scale_xy
        center_b = (pb - [x0, y0]) * scale_xy
        center_a_map = gaussian(size, *center_a)
        center_b_map = gaussian(size, *center_b)

        target = np.zeros((size, size), dtype=np.float32)
        heat_valid = False
        if not sample["same"]:
            all_foreground_crop = crop_pad(instance > 0)
            target, heat_valid = make_heat_target(a_crop, b_crop, all_foreground_crop, size, center_a, center_b,
                                                  0.5 * (psa + psb) * scale_xy)

        delta = pb - pa
        dist = float(np.linalg.norm(delta))
        proposal_scale = math.sqrt(0.5 * (psa**2 + psb**2))
        features = np.asarray([float(sample["age"]) / 240.0, float(sample["male"]),
                               dist / max(proposal_scale, 1.0), math.log(max(psa, 1e-6) / max(psb, 1e-6)),
                               float(delta[0] / max(dist, 1e-6)), float(delta[1] / max(dist, 1e-6))], dtype=np.float32)
        if not self.use_development:
            features[:2] = 0
        shuffled_age, shuffled_male = self.shuffled_metadata[str(sample["stem"])]
        shuffled_development = np.asarray([shuffled_age / 240.0, shuffled_male], dtype=np.float32)
        channels = np.stack([image_crop, center_a_map, center_b_map]).astype(np.float32)
        if self.augment:
            horizontal_flip, vertical_flip = random.random() < 0.5, random.random() < 0.5
            if horizontal_flip:
                channels, target = channels[:, :, ::-1].copy(), target[:, ::-1].copy()
            if vertical_flip:
                channels, target = channels[:, ::-1, :].copy(), target[::-1, :].copy()
            turns = random.randrange(4)
            if turns:
                channels, target = np.rot90(channels, turns, axes=(1, 2)).copy(), np.rot90(target, turns).copy()
            features[4], features[5] = transform_direction(float(features[4]), float(features[5]), horizontal_flip, vertical_flip, turns)
            if random.random() < 0.25:
                channels[0] = np.clip(channels[0] ** random.uniform(0.9, 1.1), 0, 1)
        close = (not sample["same"]) and float(sample["relative_gap"]) <= self.args.relative_close_threshold
        contact = (not sample["same"]) and float(sample["gap"]) < 1.0
        return {"image": torch.from_numpy(channels), "features": torch.from_numpy(features),
                "label": torch.tensor(float(not sample["same"])),
                "heatmap": torch.from_numpy(target[None]), "heat_valid": torch.tensor(float(heat_valid)),
                "close": torch.tensor(float(close)), "contact": torch.tensor(float(contact)),
                "case_id": sample["stem"], "shuffled_development": torch.from_numpy(shuffled_development)}


class Block(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.SiLU(),
                                 nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.SiLU())
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class PairPriorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e1, self.e2, self.e3, self.e4 = Block(3, 16), Block(16, 32), Block(32, 64), Block(64, 128)
        self.meta = nn.Sequential(nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 128))
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.SiLU(), nn.Linear(64, 1))
        self.u3, self.d3 = nn.ConvTranspose2d(128, 64, 2, 2), Block(128, 64)
        self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), Block(64, 32)
        self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), Block(32, 16)
        self.heat = nn.Conv2d(16, 1, 1)

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.e1(image); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        meta = self.meta(features); conditioned = e4 + meta[:, :, None, None]
        pooled = F.adaptive_avg_pool2d(e4, 1).flatten(1)
        logit = self.classifier(torch.cat([pooled, meta], 1)).squeeze(1)
        d3 = self.d3(torch.cat([self.u3(conditioned), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return logit, self.heat(d1)


def heat_loss(logits: torch.Tensor, target: torch.Tensor, label: torch.Tensor, heat_valid: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((1, 2, 3))
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((1, 2, 3))
    denominator = prob.sum((1, 2, 3)) + target.sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (denominator + 1)
    supervised = (label < 0.5) | (heat_valid > 0.5)
    per_sample = bce + torch.where((label > 0.5) & (heat_valid > 0.5), dice, torch.zeros_like(dice))
    return per_sample[supervised].mean() if supervised.any() else logits.sum() * 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, ablation: str = "none") -> dict[str, float]:
    model.eval()
    labels, scores, close, contact, heat_dice, heat_valid_all, cases = [], [], [], [], [], [], []
    for batch in loader:
        image = batch["image"].to(device)
        features = batch["features"].to(device)
        if ablation == "zero_xray": image[:, 0] = 0
        elif ablation == "zero_centers": image[:, 1:] = 0
        elif ablation == "zero_geometry": features[:, 2:] = 0
        elif ablation == "metadata_only": image[:] = 0; features[:, 2:] = 0
        elif ablation == "shuffled_metadata": features[:, :2] = batch["shuffled_development"].to(device)
        target = batch["heatmap"].to(device)
        logit, heat = model(image, features)
        labels.extend(batch["label"].numpy().tolist())
        scores.extend(torch.sigmoid(logit).cpu().numpy().tolist())
        close.extend(batch["close"].numpy().tolist()); contact.extend(batch["contact"].numpy().tolist())
        cases.extend(list(batch["case_id"]))
        valid = batch["heat_valid"].numpy() > 0.5; heat_valid_all.extend(valid.tolist())
        hp = torch.sigmoid(heat); inter = (hp * target).sum((1, 2, 3)); den = hp.sum((1, 2, 3)) + target.sum((1, 2, 3))
        dice = ((2 * inter + 1) / (den + 1)).cpu().numpy(); heat_dice.extend(dice[valid].tolist())
    y, p = np.asarray(labels), np.asarray(scores); prediction = p >= 0.5
    close_mask, contact_mask, negative = np.asarray(close) > 0.5, np.asarray(contact) > 0.5, y < 0.5
    positive = y > 0.5; valid = np.asarray(heat_valid_all)
    case_accuracy, case_auroc, case_close_recall, case_specificity = {}, {}, {}, {}
    for case in sorted(set(cases)):
        mask = np.asarray([x == case for x in cases])
        case_accuracy[case] = float((prediction[mask] == y[mask]).mean())
        case_auroc[case] = float(roc_auc_score(y[mask], p[mask])) if len(np.unique(y[mask])) == 2 else None
        case_close = mask & close_mask; case_negative = mask & negative
        case_close_recall[case] = float(prediction[case_close].mean()) if case_close.any() else None
        case_specificity[case] = float((~prediction[case_negative]).mean()) if case_negative.any() else None
    contact_positive = positive & contact_mask
    return {"num_pairs": float(len(y)), "positive_fraction": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "accuracy": float((prediction == y).mean()),
            "close_pair_count": float(close_mask.sum()),
            "close_pair_recall": float(prediction[close_mask].mean()) if close_mask.any() else 0.0,
            "contact_pair_count": float(contact_mask.sum()),
            "contact_pair_recall": float(prediction[contact_mask].mean()) if contact_mask.any() else 0.0,
            "same_instance_specificity": float((~prediction[negative]).mean()) if negative.any() else 0.0,
            "positive_heatmap_soft_dice": float(np.mean(heat_dice)) if heat_dice else 0.0,
            "heatmap_valid_count": float((positive & valid).sum()),
            "heatmap_target_coverage": float(valid[positive].mean()) if positive.any() else 0.0,
            "heatmap_zero_target_rate": float((~valid[positive]).mean()) if positive.any() else 1.0,
            "contact_heatmap_valid_count": float((contact_positive & valid).sum()),
            "contact_heatmap_coverage": float(valid[contact_positive].mean()) if contact_positive.any() else 0.0,
            "contact_heatmap_missed_count": float((contact_positive & ~valid).sum()),
            "mean_case_accuracy": float(np.mean(list(case_accuracy.values()))),
            "case_accuracy_by_id": case_accuracy,
            "case_auroc_by_id": case_auroc,
            "case_close_recall_by_id": case_close_recall,
            "case_specificity_by_id": case_specificity}


def train_variant(name: str, use_development: bool, train_samples: list[dict[str, Any]], val_samples: list[dict[str, Any]], stress_val_samples: list[dict[str, Any]], args: argparse.Namespace, run_seed: int) -> dict[str, Any]:
    set_seed(run_seed)
    train_ds, val_ds = PairDataset(train_samples, args, True, use_development), PairDataset(val_samples, args, False, use_development)
    stress_ds = PairDataset(stress_val_samples, args, False, use_development)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    stress_loader = DataLoader(stress_ds, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    model = PairPriorNet().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = args.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    positives = sum(not x["same"] for x in train_samples); negatives = len(train_samples) - positives
    pos_weight = torch.tensor(negatives / max(positives, 1), device=args.device)
    history = []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for batch in train_loader:
            image = batch["image"].to(args.device, non_blocking=True); features = batch["features"].to(args.device, non_blocking=True)
            label = batch["label"].to(args.device); target = batch["heatmap"].to(args.device); valid = batch["heat_valid"].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logit, heat = model(image, features)
                loss = F.binary_cross_entropy_with_logits(logit, label, pos_weight=pos_weight) + 0.35 * heat_loss(heat, target, label, valid)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, val_loader, args.device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **metrics})
        print(name, history[-1], flush=True)
    final = history[-1]
    ablations = {mode: evaluate(model, val_loader, args.device, mode)
                 for mode in ("zero_xray", "zero_centers", "zero_geometry", "metadata_only", "shuffled_metadata")}
    stress_metrics = evaluate(model, stress_loader, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{name}_seed{run_seed}_final.pt"
    torch.save({"model": model.state_dict(), "config": vars(args), "use_development": use_development, "run_seed": run_seed}, checkpoint)
    return {"name": name, "run_seed": run_seed, "checkpoint": str(checkpoint), "history": history, "final": final,
            "shortcut_ablations": ablations, "double_noise_stress": stress_metrics}


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [k for k, v in runs[0]["final"].items() if isinstance(v, (int, float))]
    final = {k: float(np.mean([run["final"][k] for run in runs])) for k in numeric_keys}
    for map_key in ("case_accuracy_by_id", "case_auroc_by_id", "case_close_recall_by_id", "case_specificity_by_id"):
        case_ids = sorted(set.intersection(*[set(run["final"][map_key]) for run in runs]))
        final[map_key] = {}
        for case in case_ids:
            values = [run["final"][map_key][case] for run in runs if run["final"][map_key][case] is not None]
            final[map_key][case] = float(np.mean(values)) if values else None
    modes = runs[0]["shortcut_ablations"]
    shortcuts = {}
    for mode in modes:
        keys = [k for k, v in runs[0]["shortcut_ablations"][mode].items() if isinstance(v, (int, float))]
        shortcuts[mode] = {k: float(np.mean([run["shortcut_ablations"][mode][k] for run in runs])) for k in keys}
    stress_keys = [k for k, v in runs[0]["double_noise_stress"].items() if isinstance(v, (int, float))]
    stress = {k: float(np.mean([run["double_noise_stress"][k] for run in runs])) for k in stress_keys}
    return {"runs": runs, "final": final, "shortcut_ablations": shortcuts, "double_noise_stress": stress}


def paired_case_bootstrap(geometry: dict[str, float | None], development: dict[str, float | None], seed: int, draws: int = 2000) -> dict[str, float]:
    cases = sorted(c for c in set(geometry) & set(development) if geometry[c] is not None and development[c] is not None)
    delta = np.asarray([development[c] - geometry[c] for c in cases], dtype=np.float64)
    if not len(delta):
        return {"num_cases": 0.0, "mean_delta_accuracy": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    sampled = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return {"num_cases": float(len(delta)), "mean_delta_accuracy": float(delta.mean()),
            "ci95_low": float(np.quantile(sampled, 0.025)), "ci95_high": float(np.quantile(sampled, 0.975))}


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    joined = str(args.variant_root).lower()
    if "clean-test" in joined or "articular" in joined or args.variant_root.name != "TSRS_RSNA-Epiphysis_contrast_v1":
        raise RuntimeError("R256 permits contrast-v1 Epiphysis train/val only")
    train_metadata, val_metadata = read_metadata(args.train_metadata), read_metadata(args.val_metadata)
    if set(train_metadata) & set(val_metadata):
        raise RuntimeError("Metadata train/val IDs overlap")
    train_samples, train_proposal_audit = build_samples(args.variant_root, "train", train_metadata, args)
    val_samples, val_proposal_audit = build_samples(args.variant_root, "val", val_metadata, args)
    stress_args = argparse.Namespace(**vars(args)); stress_args.center_noise_rel *= 2.0; stress_args.scale_noise_log_std *= 2.0
    stress_val_samples, stress_val_proposal_audit = build_samples(args.variant_root, "val", val_metadata, stress_args)
    if not train_samples or not val_samples:
        raise RuntimeError("No pair samples")
    train_seeds = [int(x.strip()) for x in args.train_seeds.split(",") if x.strip()]
    if not train_seeds:
        raise ValueError("--train-seeds must contain at least one integer")
    geometry = aggregate_runs([train_variant("geometry_only", False, train_samples, val_samples, stress_val_samples, args, seed) for seed in train_seeds])
    development = aggregate_runs([train_variant("development_conditioned", True, train_samples, val_samples, stress_val_samples, args, seed) for seed in train_seeds])
    g, d = geometry["final"], development["final"]
    bootstrap = {
        "accuracy": paired_case_bootstrap(g["case_accuracy_by_id"], d["case_accuracy_by_id"], args.seed),
        "auroc": paired_case_bootstrap(g["case_auroc_by_id"], d["case_auroc_by_id"], args.seed + 1),
        "close_recall": paired_case_bootstrap(g["case_close_recall_by_id"], d["case_close_recall_by_id"], args.seed + 2),
        "specificity": paired_case_bootstrap(g["case_specificity_by_id"], d["case_specificity_by_id"], args.seed + 3),
    }
    d_abl = development["shortcut_ablations"]
    checks = {
        "development_metrics_finite": all(np.isfinite(d[k]) for k in ("auroc", "average_precision", "close_pair_recall", "contact_pair_recall", "same_instance_specificity", "positive_heatmap_soft_dice")),
        "close_recall_ge_075": d["close_pair_recall"] >= 0.75,
        "specificity_ge_075": d["same_instance_specificity"] >= 0.75,
        "heatmap_dice_ge_020": d["positive_heatmap_soft_dice"] >= 0.20,
        "heatmap_coverage_ge_085": d["heatmap_target_coverage"] >= 0.85,
        "contact_heatmap_coverage_ge_085": d["contact_pair_count"] > 0 and d["contact_heatmap_coverage"] >= 0.85,
        "development_adds_minimum_value": (d["auroc"] - g["auroc"] >= 0.005 or d["close_pair_recall"] - g["close_pair_recall"] >= 0.02),
        "specificity_noninferior": d["same_instance_specificity"] >= g["same_instance_specificity"] - 0.02,
        "three_training_seeds": len(train_seeds) >= 3,
        "contact_proposal_coverage_eq_1": val_proposal_audit["proposal_contact_pairs_coverage"] >= 1.0,
        "close_proposal_coverage_ge_095": val_proposal_audit["proposal_close_pairs_coverage"] >= 0.95,
        "proposal_geometry_mean_matched": val_proposal_audit.get("proposal_geometry_max_standardized_mean_difference", float("inf")) <= 0.50,
        "metadata_only_not_predictive": d_abl["metadata_only"]["auroc"] <= 0.60,
        "real_metadata_beats_shuffled": d["auroc"] >= d_abl["shuffled_metadata"]["auroc"] + 0.003,
        "no_single_input_shortcut": d["auroc"] >= max(d_abl["zero_xray"]["auroc"], d_abl["zero_centers"]["auroc"], d_abl["zero_geometry"]["auroc"]) + 0.005,
        "case_bootstrap_noninferior": bootstrap["auroc"]["num_cases"] > 0 and bootstrap["close_recall"]["num_cases"] > 0 and bootstrap["specificity"]["num_cases"] > 0 and bootstrap["auroc"]["ci95_low"] >= -0.02 and bootstrap["close_recall"]["ci95_low"] >= -0.02 and bootstrap["specificity"]["ci95_low"] >= -0.02,
        "double_noise_stress_usable": development["double_noise_stress"]["close_pair_recall"] >= 0.65 and development["double_noise_stress"]["same_instance_specificity"] >= 0.65,
    }
    payload = {"run_id": "R256_SCALE_INVARIANT_PAIR_PRIOR", "study_type": "oracle_metadata_noisy_proposal_mechanism_pilot",
               "scope": "TSRS_RSNA-Epiphysis_contrast_v1 original train/val only", "clean_test_used": False,
               "num_train_pairs": len(train_samples), "num_val_pairs": len(val_samples), "config": vars(args),
               "train_seeds": train_seeds, "train_proposal_audit": train_proposal_audit, "val_proposal_audit": val_proposal_audit,
               "stress_val_proposal_audit": stress_val_proposal_audit,
               "geometry_only": geometry, "development_conditioned": development,
               "delta_development_vs_geometry": {k: d[k] - g[k] for k in d if isinstance(d[k], (int, float)) and k in g},
               "paired_case_bootstrap_development_vs_geometry": bootstrap,
               "gate_checks": checks, "gate_pass": all(checks.values()),
               "decision": "allow_center_detector_next" if all(checks.values()) else "no_go_prior_not_validated"}
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"decision": payload["decision"], "gate_checks": checks, "geometry": g, "development": d}, indent=2), flush=True)


if __name__ == "__main__":
    main()
