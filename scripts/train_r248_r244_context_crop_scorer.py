#!/usr/bin/env python3
"""R248-F0 context crop scorer for R244 merge-targeted candidates.

R245/R247 showed that scalar GT-free filters over R244 candidates are not stable.
R248 keeps the R244 candidate generator, but trains a local crop scorer so the
selector can inspect image/anchor/component/cut context. This is original-val
candidate-level evidence only: it writes JSON/CSV, writes no masks, and must not
touch clean-test-v2 for model selection.
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

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, quick_metrics
from audit_r237_component_merge_targeted_candidates import local_bg_connectivity
from audit_r244_gtfree_merge_targeted_candidates import (
    anchor_path,
    bbox_stats,
    component_labels,
    component_match_metrics,
    cut_features,
    is_risk,
    is_safe_useful,
    metric_delta,
    parse_float_list,
    safe_divide,
)
from run_r209_component_preserving_feasibility_audit import read_instance
from run_r210_f1_neck_candidate_gate import candidate_components
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R248 local crop scorer over R244 candidates.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dist-percentiles", default="6,8,10,12")
    parser.add_argument("--min-cut-area", type=int, default=3)
    parser.add_argument("--max-cut-frac", type=float, default=0.008)
    parser.add_argument("--max-components-per-image", type=int, default=5)
    parser.add_argument("--max-candidates-generated", type=int, default=64)
    parser.add_argument("--crop-pad", type=int, default=24)
    parser.add_argument("--crop-size", type=int, default=56)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--dice-tol", type=float, default=5e-4)
    parser.add_argument("--iou-tol", type=float, default=8e-4)
    parser.add_argument("--recall-tol", type=float, default=0.0)
    parser.add_argument("--grouped-cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--useful-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--risk-thresholds", default="0.70,0.75,0.80,0.85,0.90,0.95,0.99")
    parser.add_argument("--max-actions-per-image", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r248_r244_context_crop_scorer.json"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r248_r244_context_crop_scorer_threshold_grid.csv"))
    parser.add_argument("--probed-csv", type=Path, default=Path("outputs/analysis/r248_r244_context_crop_scorer_probed_rows.csv"))
    parser.add_argument("--candidate-snapshot-csv", type=Path, default=Path(""))
    parser.add_argument("--progress-json", type=Path, default=Path(""))
    parser.add_argument("--flush-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=202607248)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row if not key.startswith("_")})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


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


def local_gradient(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    vmax = float(grad.max())
    return grad / vmax if vmax > 0 else grad


def crop_for_candidate(
    image: np.ndarray,
    anchor: np.ndarray,
    component: np.ndarray,
    cut: np.ndarray,
    crop_size: int,
    crop_pad: int,
) -> np.ndarray:
    local = bbox_slice(cut, crop_pad)
    if local is None:
        return np.zeros((6, crop_size, crop_size), dtype=np.float32)
    anchor_bool = anchor.astype(bool)
    cut_bool = cut.astype(bool)
    component_bool = component.astype(bool)
    dist_in = ndimage.distance_transform_edt(anchor_bool)
    grad = local_gradient(image)
    ring = ndimage.binary_dilation(cut_bool, structure=np.ones((5, 5), dtype=bool)) & ~cut_bool
    channels = [
        normalize_image(image)[local],
        anchor_bool[local].astype(np.float32),
        component_bool[local].astype(np.float32),
        cut_bool[local].astype(np.float32),
        normalize_distance(dist_in[local]),
        np.maximum(ring[local].astype(np.float32) * 0.5, grad[local].astype(np.float32)),
    ]
    return np.stack([resize_crop(ch, crop_size) for ch in channels], axis=0).astype(np.float32)


def build_rows_and_crops(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    crops: dict[str, np.ndarray] = {}
    skipped: list[dict[str, str]] = []
    case_names = names(args.raw_root, args.dataset, args.split)[: args.limit or None]
    for index, name in enumerate(tqdm(case_names, desc=f"r248/build-r244-crops/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            skipped.append({"image": name, "reason": "missing_anchor"})
            continue
        try:
            instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
            gt = instance > 0
            image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
            anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"image": name, "reason": type(exc).__name__, "message": str(exc)})
            continue

        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        anchor_match = component_match_metrics(anchor, instance, args.min_overlap_frac)
        comp_labels, _n_components = component_labels(anchor)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        seen: set[bytes] = set()
        for dist_percentile in parse_float_list(args.dist_percentiles):
            cuts = candidate_components(anchor, image, dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
            for cut_rank, raw_cut in enumerate(cuts, start=1):
                cut = raw_cut & anchor
                if int(cut.sum()) < args.min_cut_area or int(cut.sum()) > max_pixels:
                    continue
                key_bytes = np.packbits(cut.astype(bool).ravel()).tobytes()
                if key_bytes in seen:
                    continue
                seen.add(key_bytes)
                comp_ids = np.unique(comp_labels[cut])
                comp_ids = comp_ids[comp_ids > 0]
                if comp_ids.size == 0:
                    continue
                comp_id = int(comp_ids[np.argmax([(comp_labels[cut] == cid).sum() for cid in comp_ids])])
                component = comp_labels == comp_id
                trial = anchor & ~cut
                trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                trial_match = component_match_metrics(trial, instance, args.min_overlap_frac)
                row: dict[str, Any] = {
                    "run_id": "R248-F0-r244-context-crop-scorer",
                    "split": args.split,
                    "image": name,
                    "dist_percentile": float(dist_percentile),
                    "candidate_rank": float(cut_rank),
                    "pred_component_id": float(comp_id),
                    "cut_gt_fg_frac": safe_divide(float((cut & gt).sum()), float(cut.sum())),
                    "cut_gt_gap_frac": safe_divide(float((cut & gt_cache.gap_region).sum()), float(cut.sum())),
                }
                row.update(cut_features(image, anchor, component, cut))
                row.update(local_bg_connectivity(anchor, cut, args.crop_pad))
                for metric, value in trial_metrics.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                for metric, value in trial_match.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = float(value) - float(anchor_match[metric])
                row["safe_useful_label"] = float(is_safe_useful(row, args))
                row["risk_label"] = float(is_risk(row, args))
                key = f"{name}|{len(rows)}"
                row["r248_key"] = key
                crops[key] = crop_for_candidate(image, anchor, component, cut, args.crop_size, args.crop_pad)
                rows.append(row)

        if args.flush_every > 0 and index % args.flush_every == 0:
            if str(args.candidate_snapshot_csv):
                write_csv(args.candidate_snapshot_csv, rows)
            if str(args.progress_json):
                args.progress_json.parent.mkdir(parents=True, exist_ok=True)
                args.progress_json.write_text(
                    json.dumps(
                        {
                            "run_id": "R248-F0-progress",
                            "processed_images": index,
                            "total_images": len(case_names),
                            "num_candidate_rows": len(rows),
                            "num_candidate_images": len({str(row["image"]) for row in rows}),
                            "num_safe_useful": int(sum(as_float(row, "safe_useful_label") > 0.5 for row in rows)),
                            "num_risk": int(sum(as_float(row, "risk_label") > 0.5 for row in rows)),
                            "skipped": skipped,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    return rows, crops, skipped


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], crops: dict[str, np.ndarray]):
        self.rows = [row for row in rows if str(row.get("r248_key")) in crops]
        self.crops = crops

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        x = torch.from_numpy(self.crops[str(row["r248_key"])])
        y = torch.tensor([as_float(row, "safe_useful_label"), as_float(row, "risk_label")], dtype=torch.float32)
        return x, y


class TinyContextScorer(nn.Module):
    def __init__(self, hidden: int):
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


def train_model(train_rows: list[dict[str, Any]], crops: dict[str, np.ndarray], args: argparse.Namespace, seed: int) -> TinyContextScorer:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ds = CropDataset(train_rows, crops)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyContextScorer(args.hidden).to(args.device)
    y = np.asarray([[as_float(row, "safe_useful_label"), as_float(row, "risk_label")] for row in ds.rows], dtype=np.float32)
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
def add_probs(rows: list[dict[str, Any]], crops: dict[str, np.ndarray], model: TinyContextScorer, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.eval()
    available = [row for row in rows if str(row.get("r248_key")) in crops]
    out: list[dict[str, Any]] = []
    for start in range(0, len(available), args.batch_size):
        chunk = available[start : start + args.batch_size]
        xb = torch.stack([torch.from_numpy(crops[str(row["r248_key"])]) for row in chunk]).to(args.device)
        probs = torch.sigmoid(model(xb)).cpu().numpy()
        for row, prob in zip(chunk, probs):
            out.append({**row, "r248_useful_prob": float(prob[0]), "r248_risk_prob": float(prob[1])})
    return out


def grouped_cv(rows: list[dict[str, Any]], crops: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available = [row for row in rows if str(row.get("r248_key")) in crops]
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
            if as_float(row, "r248_useful_prob") >= useful_thr and as_float(row, "r248_risk_prob") <= risk_thr
        ]
        candidates.sort(key=lambda row: (-as_float(row, "r248_useful_prob"), as_float(row, "r248_risk_prob"), -as_float(row, "delta_boundary_iou")))
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
        "safe_useful": int(sum(as_float(row, "safe_useful_label") > 0.5 for row in rows)),
        "risk": int(sum(as_float(row, "risk_label") > 0.5 for row in rows)),
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
            row = {"useful_threshold": useful_thr, "risk_threshold": risk_thr, **summarize_selection(selected)}
            row["passes_gate"] = bool(
                row["selected"] >= 5
                and row["selected_images"] >= 5
                and row["safe_useful"] > row["risk"]
                and (row["mean_delta_recall"] is None or float(row["mean_delta_recall"]) >= 0.0)
                and (row["mean_delta_boundary_iou"] is not None and float(row["mean_delta_boundary_iou"]) > 0.0)
                and (row["mean_delta_gap_region_fp_rate"] is not None and float(row["mean_delta_gap_region_fp_rate"]) < 0.0)
                and (row["mean_cut_gt_fg_frac"] is not None and float(row["mean_cut_gt_fg_frac"]) <= 0.25)
            )
            grid.append(row)
    return grid


def choose_best(grid: list[dict[str, Any]]) -> dict[str, Any] | None:
    nonempty = [row for row in grid if int(row["selected"]) > 0]
    if not nonempty:
        return None
    passing = [row for row in nonempty if row.get("passes_gate")]
    pool = passing if passing else nonempty
    return sorted(
        pool,
        key=lambda row: (
            bool(row.get("passes_gate")),
            int(row["safe_useful"]) - int(row["risk"]),
            int(row["safe_useful"]),
            float(row["mean_delta_boundary_iou"] or -999.0),
            -float(row["mean_delta_gap_region_fp_rate"] or 999.0),
            -float(row["mean_cut_gt_fg_frac"] or 999.0),
        ),
        reverse=True,
    )[0]


def main() -> None:
    args = parse_args()
    rows, crops, skipped = build_rows_and_crops(args)
    probed, cv_meta = grouped_cv(rows, crops, args)
    grid = evaluate_grid(probed, args) if probed else []
    best = choose_best(grid)
    report = {
        "run_id": "R248-F0-r244-context-crop-scorer",
        "dataset": args.dataset,
        "split": args.split,
        "anchor_exp": args.anchor_exp,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "original_val_r244_candidate_context_crop_grouped_cv",
        "num_candidate_rows": len(rows),
        "num_candidate_images": len({str(row["image"]) for row in rows}),
        "num_crops": len(crops),
        "num_safe_useful": int(sum(as_float(row, "safe_useful_label") > 0.5 for row in rows)),
        "num_risk": int(sum(as_float(row, "risk_label") > 0.5 for row in rows)),
        "device": args.device,
        "crop_size": args.crop_size,
        "epochs": args.epochs,
        "grouped_cv": {**cv_meta, "best": best, "num_passing_configs": int(sum(bool(row.get("passes_gate")) for row in grid))},
        "skipped": skipped,
        "promotion_gate": "Only if grouped-CV best passes on full original-val, build isolated original-val mask editor; no clean-test-v2 tuning.",
        "warning": "Candidate generation is GT-free; GT labels/metrics are original-val audit only.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.grid_csv, grid)
    write_csv(args.probed_csv, probed)
    print(json.dumps({"output_json": str(args.output_json), "grid_csv": str(args.grid_csv), "probed_csv": str(args.probed_csv), "best": best}, indent=2))


if __name__ == "__main__":
    main()
