#!/usr/bin/env python3
"""R218 local patch action scorer for soft seam actions.

R216 showed partial seam actions are safer than full deletion, while R217
hand-crafted patch statistics did not improve grouped-CV gating. R218 trains a
small CNN directly on local action crops to classify useful and risk actions.

This script consumes R216/R217 candidate-action CSV rows, regenerates action
crops from the R110 anchor, runs image-grouped CV, and writes JSON/CSV only. It
does not write masks and must not use clean-test-v2 for model selection.
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

from audit_r216_soft_seam_action_candidates import soft_action_cut
from run_r210_f1_neck_candidate_gate import candidate_components, read_instance
from train_anchor_pixel_residual import image_path, read_gray, read_mask, resize_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate R218 patch action scorer with image-grouped CV.")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--dist-percentile", type=float, default=10.0)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-candidates-generated", type=int, default=8)
    parser.add_argument("--max-components-per-image", type=int, default=3)
    parser.add_argument("--action-fracs", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--crop-pad", type=int, default=18)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--grouped-cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classifier-hidden", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--useful-target", choices=["quick", "safe_channel"], default="quick")
    parser.add_argument("--risk-target", choices=["hard", "overerosion_or_hard"], default="hard")
    parser.add_argument("--useful-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--risk-thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--max-actions-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=202607218)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def is_quick_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= -8e-4
        and as_float(row, "delta_boundary_iou") > 0.0
        and as_float(row, "delta_boundary_f1") > 0.0
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
    )


def is_safe_channel_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "r216_cut_gt_fg_frac") <= 0.5
        and as_float(row, "r216_cut_gt_gap_frac") >= 0.5
        and is_quick_useful(row)
    )


def is_useful(row: dict[str, Any], target: str) -> bool:
    if target == "quick":
        return is_quick_useful(row)
    if target == "safe_channel":
        return is_safe_channel_useful(row)
    raise ValueError(target)


def is_hard_risk(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
        or as_float(row, "delta_recall") < -8e-4
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_boundary_f1") < 0.0
    )


def is_risk(row: dict[str, Any], target: str) -> bool:
    if target == "hard":
        return is_hard_risk(row)
    if target == "overerosion_or_hard":
        return bool(is_hard_risk(row) or as_float(row, "r216_cut_gt_fg_frac") > 0.5)
    raise ValueError(target)


def row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["image"]), f"{as_float(row, 'candidate_rank'):.6f}", f"{as_float(row, 'action_frac'):.6f}"


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), instance.shape)
    return image.astype(np.float32), anchor.astype(bool)


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def resize_crop(arr: np.ndarray, size: int) -> np.ndarray:
    tensor = torch.from_numpy(arr.astype(np.float32))[None, None]
    out = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
    return out[0, 0].numpy().astype(np.float32)


def crop_for_action(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray, crop_size: int, crop_pad: int) -> np.ndarray:
    local_slice = bbox_slice(cut, crop_pad)
    if local_slice is None:
        return np.zeros((5, crop_size, crop_size), dtype=np.float32)
    image_l = image[local_slice].astype(np.float32)
    anchor_l = anchor[local_slice].astype(np.float32)
    cut_l = cut[local_slice].astype(np.float32)
    dist_in = ndimage.distance_transform_edt(anchor)[local_slice].astype(np.float32)
    dist_out = ndimage.distance_transform_edt(~anchor)[local_slice].astype(np.float32)
    for arr in (dist_in, dist_out):
        denom = float(np.percentile(arr, 95)) if arr.size else 1.0
        if denom > 0:
            arr /= denom
    image_l = image_l - float(image_l.mean())
    std = float(image_l.std())
    if std > 1e-6:
        image_l /= std
    channels = [image_l, anchor_l, cut_l, np.clip(dist_in, 0.0, 2.0), np.clip(dist_out, 0.0, 2.0)]
    return np.stack([resize_crop(ch, crop_size) for ch in channels], axis=0).astype(np.float32)


def build_crops(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], np.ndarray]:
    images = sorted({str(row["image"]) for row in rows})
    if args.limit_images > 0:
        images = images[: args.limit_images]
    action_fracs = parse_float_list(args.action_fracs)
    crops: dict[tuple[str, str, str], np.ndarray] = {}
    for name in tqdm(images, desc=f"r218/crops/{args.split}"):
        if not anchor_path(args, name).exists():
            continue
        image, anchor = load_case(args, name)
        cuts = candidate_components(anchor, image, args.dist_percentile, args.min_cut_area)[: args.max_candidates_generated]
        for rank, raw_cut in enumerate(cuts[: args.max_components_per_image], start=1):
            for action_frac in action_fracs:
                cut = soft_action_cut(image, anchor, raw_cut, action_frac, args.min_cut_area)
                cut = np.logical_and(cut, anchor)
                if int(cut.sum()) < args.min_cut_area:
                    continue
                key = (name, f"{float(rank):.6f}", f"{float(action_frac):.6f}")
                crops[key] = crop_for_action(image, anchor, cut, args.crop_size, args.crop_pad)
    return crops


class ActionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], crops: dict[tuple[str, str, str], np.ndarray], useful_target: str, risk_target: str):
        self.rows = [row for row in rows if row_key(row) in crops]
        self.crops = crops
        self.useful_target = useful_target
        self.risk_target = risk_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        x = torch.from_numpy(self.crops[row_key(row)])
        y = torch.tensor([float(is_useful(row, self.useful_target)), float(is_risk(row, self.risk_target))], dtype=torch.float32)
        return x, y


class TinyPatchScorer(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
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


def train_model(train_rows: list[dict[str, Any]], crops: dict[tuple[str, str, str], np.ndarray], args: argparse.Namespace, seed: int) -> TinyPatchScorer:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    ds = ActionDataset(train_rows, crops, args.useful_target, args.risk_target)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = TinyPatchScorer(args.classifier_hidden).to(args.device)
    y = np.asarray([[float(is_useful(row, args.useful_target)), float(is_risk(row, args.risk_target))] for row in ds.rows], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    pos_weight = torch.tensor(np.clip(neg / np.maximum(pos, 1.0), 1.0, 8.0), dtype=torch.float32, device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()
    for _epoch in range(args.epochs):
        for x, yb in loader:
            x = x.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def add_probs(rows: list[dict[str, Any]], crops: dict[tuple[str, str, str], np.ndarray], model: TinyPatchScorer, args: argparse.Namespace) -> list[dict[str, Any]]:
    model.eval()
    out = []
    batch_rows = [row for row in rows if row_key(row) in crops]
    for start in range(0, len(batch_rows), args.batch_size):
        chunk = batch_rows[start : start + args.batch_size]
        x = torch.stack([torch.from_numpy(crops[row_key(row)]) for row in chunk]).to(args.device)
        probs = torch.sigmoid(model(x)).cpu().numpy()
        for row, prob in zip(chunk, probs):
            out.append(
                {
                    **row,
                    "r218_useful_label": float(is_useful(row, args.useful_target)),
                    "r218_risk_label": float(is_risk(row, args.risk_target)),
                    "r218_useful_prob": float(prob[0]),
                    "r218_risk_prob": float(prob[1]),
                }
            )
    return out


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    return grouped


def selected_rows(rows: list[dict[str, Any]], useful_thr: float, risk_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    selected = []
    for image_rows in group_rows(rows).values():
        candidates = [
            row
            for row in image_rows
            if as_float(row, "r218_useful_prob") >= useful_thr and as_float(row, "r218_risk_prob") <= risk_thr
        ]
        candidates.sort(key=lambda row: (-as_float(row, "r218_useful_prob"), as_float(row, "r218_risk_prob"), as_float(row, "candidate_rank"), as_float(row, "action_frac")))
        selected.extend(candidates[:max_per_image])
    return selected


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def evaluate_grid(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out = []
    for useful_thr in parse_float_list(args.useful_thresholds):
        for risk_thr in parse_float_list(args.risk_thresholds):
            selected = selected_rows(rows, useful_thr, risk_thr, args.max_actions_per_image)
            useful = int(sum(as_float(row, "r218_useful_label") > 0.5 for row in selected))
            risk = int(sum(as_float(row, "r218_risk_label") > 0.5 for row in selected))
            out.append(
                {
                    "useful_threshold": useful_thr,
                    "risk_threshold": risk_thr,
                    "accepted": len(selected),
                    "accepted_images": len({row["image"] for row in selected}),
                    "useful": useful,
                    "risk": risk,
                    "precision_useful": float(useful / max(1, len(selected))),
                    "risk_rate": float(risk / max(1, len(selected))),
                    "mean_action_frac": mean(selected, "action_frac"),
                    "mean_delta_dice": mean(selected, "delta_dice"),
                    "mean_delta_iou": mean(selected, "delta_iou"),
                    "mean_delta_recall": mean(selected, "delta_recall"),
                    "mean_delta_boundary_iou": mean(selected, "delta_boundary_iou"),
                    "mean_delta_boundary_f1": mean(selected, "delta_boundary_f1"),
                    "mean_delta_gap_region_fp_rate": mean(selected, "delta_gap_region_fp_rate"),
                }
            )
    return out


def frontier(grid: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for max_risk in [0, 1, 2, 5, 10, 20]:
        subset = [row for row in grid if int(row["risk"]) <= max_risk]
        out[f"risk_le_{max_risk}"] = sorted(
            subset,
            key=lambda row: (row["useful"], row["precision_useful"], row["accepted"], row["mean_delta_boundary_iou"] or -999.0),
            reverse=True,
        )[:20]
    return out


def grouped_cv(rows: list[dict[str, Any]], crops: dict[tuple[str, str, str], np.ndarray], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    available_rows = [row for row in rows if row_key(row) in crops]
    images = np.asarray(sorted({row["image"] for row in available_rows}))
    folds = min(args.grouped_cv_folds, len(images))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=args.seed)
    probed = []
    fold_summaries = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(images), start=1):
        train_images = set(images[train_idx].tolist())
        val_images = set(images[val_idx].tolist())
        train_rows = [row for row in available_rows if row["image"] in train_images]
        val_rows = [row for row in available_rows if row["image"] in val_images]
        model = train_model(train_rows, crops, args, args.seed + fold_idx)
        probed.extend(add_probs(val_rows, crops, model, args))
        fold_summaries.append({"fold": fold_idx, "train_images": len(train_images), "val_images": len(val_images), "train_rows": len(train_rows), "val_rows": len(val_rows)})
    grid = evaluate_grid(probed, args)
    return grid, {"folds": folds, "fold_summaries": fold_summaries, "frontier": frontier(grid)}


def main() -> None:
    args = parse_args()
    rows = read_csv(args.candidate_csv)
    if args.limit_images > 0:
        allowed = []
        seen = set()
        for row in rows:
            if row["image"] not in seen:
                seen.add(row["image"])
                allowed.append(row["image"])
        allowed = set(allowed[: args.limit_images])
        rows = [row for row in rows if row["image"] in allowed]
    crops = build_crops(args, rows)
    grid, cv_summary = grouped_cv(rows, crops, args)
    ranked = sorted(grid, key=lambda row: (row["risk"] <= 2, row["useful"], -row["risk"], row["mean_delta_boundary_iou"] or -999), reverse=True)
    report = {
        "run_id": "R218-patch-action-scorer",
        "candidate_csv": str(args.candidate_csv),
        "evidence_level": "candidate_action_grouped_cv_diagnostic",
        "clean_test_v2_used": False,
        "writes_masks": False,
        "num_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_crops": len(crops),
        "device": args.device,
        "crop_size": args.crop_size,
        "epochs": args.epochs,
        "useful_target": args.useful_target,
        "risk_target": args.risk_target,
        "num_useful": int(sum(is_useful(row, args.useful_target) for row in rows)),
        "num_risk": int(sum(is_risk(row, args.risk_target) for row in rows)),
        "grouped_cv": {**cv_summary, "best": ranked[0] if ranked else None},
        "warning": "candidate-action grouped-CV diagnostic only; not mask-level R201 evidence",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, grid)
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv), "best": report["grouped_cv"]["best"]}, indent=2))


if __name__ == "__main__":
    main()
