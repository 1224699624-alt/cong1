#!/usr/bin/env python3
"""R242 local crop scorer for reversed-topology seam/background cuts.

R239/R240 showed that background-channel cuts can improve seam/gap diagnostics,
but scalar gates cannot reliably distinguish true gap cuts from shallow bone
foreground cuts. R242 trains a small local crop scorer over the generated cut
itself. This is an original-val candidate-level diagnostic: it writes JSON/CSV
only, writes no masks, and must not be used on clean-test-v2 for selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audit_r228_oracle_pixel_seam_localizer import (
    feature_stack,
    fit_model,
    grouped_r224_rows,
    predict_prob,
    r224_positive_mask,
)
from run_r239_gtfree_bg_channel_seam_prob_editor import (
    application_names,
    as_float,
    candidate_rows_for_case,
    load_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate R242 local crop scorer with image-grouped CV.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--source-mode", choices=["source-csv", "all-anchors"], default="all-anchors")
    parser.add_argument("--source-candidate-csv", type=Path, default=Path("outputs/analysis/r239_gtfree_bg_channel_light_fullval_candidates.csv"))
    parser.add_argument("--r224-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--limit-source", type=int, default=0)
    parser.add_argument("--limit-train-rows", type=int, default=300)
    parser.add_argument("--samples-per-image", type=int, default=1024)
    parser.add_argument("--positive-oversample", type=int, default=512)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--max-components-per-image", type=int, default=3)
    parser.add_argument("--component-pad", type=int, default=18)
    parser.add_argument("--max-bg-components", type=int, default=6)
    parser.add_argument("--max-pairs-per-component", type=int, default=4)
    parser.add_argument("--corridor-radii", default="1,2,3")
    parser.add_argument("--action-fracs", default="0.15,0.25,0.35")
    parser.add_argument("--min-cut-area", type=int, default=3)
    parser.add_argument("--max-cut-frac", type=float, default=0.012)
    parser.add_argument("--max-candidates-per-image", type=int, default=120)
    parser.add_argument("--seam-prob-threshold", type=float, default=0.35)
    parser.add_argument("--seam-prob-p90-threshold", type=float, default=0.70)
    parser.add_argument("--min-bg-channel-frac", type=float, default=0.10)
    parser.add_argument("--require-bg-channel", action="store_true")
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--boundary-tol", type=float, default=0.0)
    parser.add_argument("--allow-component-mae-worsen", action="store_true")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--crop-size", type=int, default=56)
    parser.add_argument("--crop-pad", type=int, default=20)
    parser.add_argument("--grouped-cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-hidden", type=int, default=64)
    parser.add_argument("--useful-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--risk-thresholds", default="0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    parser.add_argument("--max-actions-per-image", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r242_candidate_crop_scorer_groupedcv.json"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r242_candidate_crop_scorer_threshold_grid.csv"))
    parser.add_argument("--probed-csv", type=Path, default=Path("outputs/analysis/r242_candidate_crop_scorer_probed_rows.csv"))
    parser.add_argument("--candidate-snapshot-csv", type=Path, default=Path(""))
    parser.add_argument("--progress-json", type=Path, default=Path(""))
    parser.add_argument("--flush-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607242)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row if key != "_crop"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)),
    )


def resize_crop(arr: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None]
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy().astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    out = image - float(image.mean())
    std = float(out.std())
    return out / std if std > 1e-6 else out


def normalize_distance(dist: np.ndarray) -> np.ndarray:
    dist = dist.astype(np.float32)
    denom = float(np.percentile(dist, 95)) if dist.size else 1.0
    if denom > 0:
        dist = dist / denom
    return np.clip(dist, 0.0, 2.0)


def crop_for_candidate(
    image: np.ndarray,
    anchor: np.ndarray,
    cut: np.ndarray,
    prob: np.ndarray,
    crop_size: int,
    crop_pad: int,
) -> np.ndarray:
    local = bbox_slice(cut, crop_pad)
    if local is None:
        return np.zeros((6, crop_size, crop_size), dtype=np.float32)
    anchor_bool = anchor.astype(bool)
    cut_bool = cut.astype(bool)
    dist_in = ndimage.distance_transform_edt(anchor_bool)
    dist_out = ndimage.distance_transform_edt(~anchor_bool)
    after = anchor_bool & ~cut_bool
    cut_channel_bg = ndimage.binary_dilation(cut_bool, structure=np.ones((5, 5), dtype=bool)) & ~after
    channels = [
        normalize_image(image)[local],
        anchor_bool[local].astype(np.float32),
        cut_bool[local].astype(np.float32),
        normalize_distance(dist_in[local]),
        normalize_distance(dist_out[local]),
        np.clip(prob[local].astype(np.float32), 0.0, 1.0),
    ]
    # Mark the opened background channel by slightly boosting the cut channel
    # around nearby background. This stays inference-time only.
    channels[2] = np.maximum(channels[2], 0.35 * cut_channel_bg[local].astype(np.float32))
    return np.stack([resize_crop(ch, crop_size) for ch in channels], axis=0).astype(np.float32)


def is_safe_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= 0.0
        and as_float(row, "delta_boundary_iou") >= 0.0
        and as_float(row, "delta_boundary_f1") >= 0.0
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
        and as_float(row, "cut_gt_fg_frac") <= 0.25
        and as_float(row, "cut_gt_gap_frac") >= 0.75
    )


def is_risk(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "cut_gt_fg_frac") > 0.5
        or as_float(row, "delta_recall") < 0.0
        or as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_boundary_f1") < 0.0
    )


def row_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("image")),
            str(row.get("candidate_family")),
            f"{as_float(row, 'pred_component_id'):.0f}",
            f"{as_float(row, 'corridor_radius'):.0f}",
            f"{as_float(row, 'action_frac'):.4f}",
            f"{as_float(row, 'cut_area'):.0f}",
            f"{as_float(row, 'bg_pair_distance'):.3f}",
            f"{as_float(row, 'bridge_score_p90'):.3f}",
        ]
    )


def sample_training_safe(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    grouped = grouped_r224_rows(args)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stats: dict[str, Any] = {"images": 0, "positive_pixels": 0, "sampled_positive": 0, "sampled_negative": 0, "skipped_training_images": []}
    for name, rows in tqdm(grouped.items(), desc="r242/sample/r224"):
        try:
            image, instance, _gt, anchor = load_case(args, name)
            pos_mask = r224_positive_mask(args, image, instance, anchor, rows)
        except Exception as exc:  # noqa: BLE001 - corrupt local images should not kill diagnostics.
            stats["skipped_training_images"].append({"image": name, "reason": type(exc).__name__, "message": str(exc)})
            continue
        if not pos_mask.any():
            continue
        feat = feature_stack(image, anchor)
        dist_in = ndimage.distance_transform_edt(anchor)
        eligible = anchor & (dist_in <= max(1.0, float(np.percentile(dist_in[anchor], 35))))
        pos = np.flatnonzero(pos_mask.ravel())
        neg = np.flatnonzero((eligible & ~pos_mask).ravel())
        if neg.size == 0:
            continue
        n_pos = min(args.positive_oversample, max(args.min_cut_area, pos.size * 4))
        n_neg = max(args.samples_per_image - n_pos, n_pos)
        pick_pos = rng.choice(pos, size=n_pos, replace=pos.size < n_pos)
        pick_neg = rng.choice(neg, size=n_neg, replace=neg.size < n_neg)
        idx = np.concatenate([pick_pos, pick_neg])
        xs.append(feat.reshape(-1, feat.shape[-1])[idx])
        target = np.zeros(idx.shape[0], dtype=np.float32)
        target[: len(pick_pos)] = 1.0
        ys.append(target)
        stats["images"] += 1
        stats["positive_pixels"] += int(pos.size)
        stats["sampled_positive"] += int(n_pos)
        stats["sampled_negative"] += int(n_neg)
    if not xs:
        raise RuntimeError("No R242/R228 pixel samples collected after skipping unreadable images.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), stats


def build_candidate_crops(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    x, y, train_info = sample_training_safe(args)
    seam_model = fit_model(args, x, y)
    rows: list[dict[str, Any]] = []
    crops: dict[str, np.ndarray] = {}
    skipped: list[dict[str, str]] = []
    app_names = application_names(args)
    for index, name in enumerate(tqdm(app_names, desc=f"r242/candidates/{args.split}"), start=1):
        try:
            image, instance, gt, anchor = load_case(args, name)
            prob = predict_prob(seam_model, image, anchor)
            case_rows = candidate_rows_for_case(args, image, instance, gt, anchor, prob, name)
        except Exception as exc:  # noqa: BLE001 - keep long remote sweeps alive and report skips.
            skipped.append({"image": name, "reason": type(exc).__name__, "message": str(exc)})
            continue
        for row in case_rows:
            cut = row.pop("_cut", None)
            if cut is None:
                continue
            key = row_key(row)
            if key in crops:
                continue
            crops[key] = crop_for_candidate(image, anchor, cut, prob, args.crop_size, args.crop_pad)
            row["r242_key"] = key
            row["r242_safe_useful_label"] = float(is_safe_useful(row))
            row["r242_risk_label"] = float(is_risk(row))
            rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            if str(args.candidate_snapshot_csv):
                write_csv(args.candidate_snapshot_csv, rows)
            if str(args.progress_json):
                args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                progress = {
                    "run_id": "R242-candidate-crop-scorer-progress",
                    "processed_images": index,
                    "total_images": len(app_names),
                    "num_candidate_rows": len(rows),
                    "num_candidate_images": len({str(row["image"]) for row in rows}),
                    "num_crops": len(crops),
                    "num_safe_useful": int(sum(is_safe_useful(row) for row in rows)),
                    "num_risk": int(sum(is_risk(row) for row in rows)),
                    "skipped_images": skipped,
                    "note": "Progress snapshot only; not final grouped-CV evidence.",
                }
                args.progress_json.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    meta = {
        "seam_train_info": {**train_info, "num_samples": int(len(y)), "positive_sample_rate": float(np.mean(y))},
        "skipped_images": skipped,
    }
    return rows, crops, meta


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], crops: dict[str, np.ndarray]):
        self.rows = [row for row in rows if str(row.get("r242_key")) in crops]
        self.crops = crops

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        x = torch.from_numpy(self.crops[str(row["r242_key"])])
        y = torch.tensor([float(is_safe_useful(row)), float(is_risk(row))], dtype=torch.float32)
        return x, y


class TinyCropScorer(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, hidden), nn.ReLU(inplace=True), nn.Dropout(0.15), nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def train_model(train_rows: list[dict[str, Any]], crops: dict[str, np.ndarray], args: argparse.Namespace, seed: int) -> TinyCropScorer:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ds = CropDataset(train_rows, crops)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyCropScorer(args.classifier_hidden).to(args.device)
    y = np.asarray([[float(is_safe_useful(row)), float(is_risk(row))] for row in ds.rows], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, 10.0), dtype=torch.float32, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _epoch in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def add_probs(rows: list[dict[str, Any]], crops: dict[str, np.ndarray], model: TinyCropScorer, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.eval()
    available = [row for row in rows if str(row.get("r242_key")) in crops]
    out: list[dict[str, Any]] = []
    for start in range(0, len(available), args.batch_size):
        chunk = available[start : start + args.batch_size]
        xb = torch.stack([torch.from_numpy(crops[str(row["r242_key"])]) for row in chunk]).to(args.device)
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        for row, prob in zip(chunk, probs):
            out.append({**row, "r242_useful_prob": float(prob[0]), "r242_risk_prob": float(prob[1])})
    return out


def grouped_cv(rows: list[dict[str, Any]], crops: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = [row for row in rows if str(row.get("r242_key")) in crops]
    images = np.asarray(sorted({str(row["image"]) for row in available}))
    if len(images) < 2:
        return [], {"skipped": True, "reason": "not_enough_images"}
    folds = min(args.grouped_cv_folds, len(images))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=args.seed)
    probed: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(images), start=1):
        train_images = set(images[train_idx].tolist())
        val_images = set(images[val_idx].tolist())
        train_rows = [row for row in available if row["image"] in train_images]
        val_rows = [row for row in available if row["image"] in val_images]
        model = train_model(train_rows, crops, args, args.seed + fold_idx)
        probed.extend(add_probs(val_rows, crops, model, args))
        fold_summaries.append({"fold": fold_idx, "train_images": len(train_images), "val_images": len(val_images), "train_rows": len(train_rows), "val_rows": len(val_rows)})
    return probed, {"skipped": False, "folds": folds, "fold_summaries": fold_summaries}


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    return grouped


def selected_rows(rows: list[dict[str, Any]], useful_thr: float, risk_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for image_rows in group_rows(rows).values():
        candidates = [
            row
            for row in image_rows
            if as_float(row, "r242_useful_prob") >= useful_thr and as_float(row, "r242_risk_prob") <= risk_thr
        ]
        candidates.sort(key=lambda row: (-as_float(row, "r242_useful_prob"), as_float(row, "r242_risk_prob"), -as_float(row, "delta_boundary_iou")))
        selected.extend(candidates[:max_per_image])
    return selected


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [value for value in vals if np.isfinite(value)]
    return float(np.mean(vals)) if vals else None


def summarize_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected": len(rows),
        "selected_images": len({str(row["image"]) for row in rows}),
        "safe_useful": int(sum(is_safe_useful(row) for row in rows)),
        "risk": int(sum(is_risk(row) for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
    }


def evaluate_grid(probed: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for useful_thr in parse_float_list(args.useful_thresholds):
        for risk_thr in parse_float_list(args.risk_thresholds):
            selected = selected_rows(probed, useful_thr, risk_thr, args.max_actions_per_image)
            grid.append({"useful_threshold": useful_thr, "risk_threshold": risk_thr, **summarize_selection(selected)})
    return grid


def choose_best(grid: list[dict[str, Any]]) -> dict[str, Any] | None:
    nonempty = [row for row in grid if int(row["selected"]) > 0]
    if not nonempty:
        return None
    viable = [row for row in nonempty if int(row["risk"]) <= 2 and (row["mean_delta_recall"] is None or float(row["mean_delta_recall"]) >= -1e-8)]
    pool = viable if viable else nonempty
    return sorted(
        pool,
        key=lambda row: (
            -int(row["risk"]),
            int(row["safe_useful"]),
            float(row["mean_delta_boundary_iou"] or -999.0),
            -float(row["mean_delta_gap_region_fp_rate"] or 999.0),
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    rows, crops, meta = build_candidate_crops(args)
    probed, cv_meta = grouped_cv(rows, crops, args)
    grid = evaluate_grid(probed, args) if probed else []
    best = choose_best(grid)
    report = {
        "run_id": "R242-candidate-crop-scorer",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "original_val_candidate_crop_grouped_cv_diagnostic",
        "idea": "score local background-channel cuts from image/cut crops to separate true bone gaps from foreground erosion",
        "num_candidate_rows": len(rows),
        "num_candidate_images": len({str(row["image"]) for row in rows}),
        "num_crops": len(crops),
        "num_safe_useful": int(sum(is_safe_useful(row) for row in rows)),
        "num_risk": int(sum(is_risk(row) for row in rows)),
        "device": args.device,
        "crop_size": args.crop_size,
        "epochs": args.epochs,
        "candidate_generation": "R239 GT-free bg components/seeds + seam probability map; GT only labels/audits candidate effects",
        "labels": {
            "safe_useful": "non-worse Dice/IoU/Recall/Boundary, gap-FP gain, component-count non-worse, cut_gt_fg_frac<=0.25, cut_gt_gap_frac>=0.75",
            "risk": "foreground-heavy cut, recall loss, overlap drop, or boundary drop",
        },
        "seam_probability_model": meta["seam_train_info"],
        "skipped_images": meta["skipped_images"],
        "grouped_cv": {**cv_meta, "best": best},
        "promotion_gate": {
            "candidate_level": "selected risk far below selected safe, Recall delta zero, positive Boundary IoU/F1, negative gap FP",
            "next_step": "only if candidate-level gate passes, write an isolated original-val mask editor; clean-test-v2 remains locked",
        },
        "warning": "Original-val development only; not clean-test-v2 or R201 final evidence.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.grid_csv, grid)
    write_csv(args.probed_csv, probed)
    print(json.dumps({"output_json": str(args.output_json), "grid_csv": str(args.grid_csv), "probed_csv": str(args.probed_csv), "best": best}, indent=2))


if __name__ == "__main__":
    main()
