#!/usr/bin/env python3
"""R228 oracle pixel seam-localizer diagnostic.

R228-F0 tests whether R224 oracle seam pixels can train a local pixel scorer
that ranks GT-free R226-style background-connectivity cuts. This is deliberately
candidate-level and original-val only: it writes no masks and does not touch
clean-test-v2.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy import ndimage
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError:
    HistGradientBoostingClassifier = None
    SimpleImputer = None
    Pipeline = Any
from tqdm import tqdm

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, is_hard_risk, is_quick_useful, quick_metrics
from audit_r224_pairwise_instance_seam_candidates import pairwise_seam_cut, trim_cut_by_score as trim_r224_cut
from audit_r226_bg_connectivity_seam_scorer import bg_contact_seeds, seed_pair_cut, trim_cut_by_score as trim_r226_cut
from run_r209_component_preserving_feasibility_audit import read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R228 oracle-pixel seam localizer diagnostic.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--r224-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--source-candidate-csv", type=Path, default=Path("outputs/analysis/r224_pairwise_instance_seam_fullval_candidates.csv"))
    parser.add_argument("--limit-source", type=int, default=32)
    parser.add_argument("--limit-train-rows", type=int, default=0)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--positive-oversample", type=int, default=2048)
    parser.add_argument("--max-components-per-image", type=int, default=1)
    parser.add_argument("--max-bg-components", type=int, default=3)
    parser.add_argument("--max-pairs-per-component", type=int, default=2)
    parser.add_argument("--corridor-radius", type=int, default=1)
    parser.add_argument("--action-frac", type=float, default=0.35)
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.006)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--score-thresholds", default="0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r228_oracle_pixel_seam_localizer_summary.json"))
    parser.add_argument("--candidate-csv", type=Path, default=Path("outputs/analysis/r228_oracle_pixel_seam_localizer_candidates.csv"))
    parser.add_argument("--grid-csv", type=Path, default=Path("outputs/analysis/r228_oracle_pixel_seam_localizer_grid.csv"))
    parser.add_argument("--seed", type=int, default=202607228)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    try:
        out = float(value)
    except ValueError:
        return default
    return out if np.isfinite(out) else default


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def feature_stack(image: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    anchor_f = anchor.astype(np.float32)
    gy, gx = np.gradient(image)
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    if float(grad.max()) > 0.0:
        grad /= float(grad.max())
    dist_in = ndimage.distance_transform_edt(anchor).astype(np.float32)
    dist_out = ndimage.distance_transform_edt(~anchor).astype(np.float32)
    dist_in_n = dist_in / max(1.0, float(np.percentile(dist_in[anchor], 95)) if anchor.any() else 1.0)
    dist_out_n = dist_out / max(1.0, float(np.percentile(dist_out[~anchor], 95)) if (~anchor).any() else 1.0)
    boundary = ndimage.binary_dilation(anchor, structure=np.ones((3, 3), dtype=bool)) ^ ndimage.binary_erosion(anchor, structure=np.ones((3, 3), dtype=bool))
    bg_near = ndimage.distance_transform_edt(anchor).astype(np.float32)
    local = []
    for size in (3, 7, 15):
        local.append(ndimage.uniform_filter(image, size=size, mode="nearest").astype(np.float32))
        local.append(ndimage.uniform_filter(anchor_f, size=size, mode="nearest").astype(np.float32))
    yy, xx = np.indices(anchor.shape, dtype=np.float32)
    yy /= max(1.0, float(anchor.shape[0] - 1))
    xx /= max(1.0, float(anchor.shape[1] - 1))
    return np.stack(
        [
            image,
            anchor_f,
            grad,
            np.clip(dist_in_n, 0.0, 1.0),
            np.clip(dist_out_n, 0.0, 1.0),
            boundary.astype(np.float32),
            np.clip(bg_near / max(1.0, float(np.percentile(bg_near[anchor], 95)) if anchor.any() else 1.0), 0.0, 1.0),
            yy,
            xx,
            *local,
        ],
        axis=-1,
    ).astype(np.float32)


def grouped_r224_rows(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    rows = read_rows(args.r224_csv)
    useful = [
        row
        for row in rows[: args.limit_train_rows or None]
        if as_float(row, "r224_quick_useful") > 0.5
        and as_float(row, "r224_hard_risk") <= 0.5
        and as_float(row, "cut_gt_gap_frac") >= 0.75
        and as_float(row, "cut_gt_fg_frac") <= 0.25
    ]
    out: dict[str, list[dict[str, Any]]] = {}
    for row in useful:
        out.setdefault(str(row["image"]), []).append(row)
    return out


def r224_positive_mask(args: argparse.Namespace, image: np.ndarray, instance: np.ndarray, anchor: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    labels, _ = component_labels(anchor)
    pos = np.zeros_like(anchor, dtype=bool)
    for row in rows:
        comp_id = int(float(row["pred_component_id"]))
        component = labels == comp_id
        raw = pairwise_seam_cut(component, instance, int(float(row["gt_a"])), int(float(row["gt_b"])), int(float(row["seam_radius"])))
        cut = trim_r224_cut(image, anchor, raw, float(row["action_frac"]), args.min_cut_area)
        pos |= cut & anchor
    return pos


def sample_training(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    grouped = grouped_r224_rows(args)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stats = {"images": 0, "positive_pixels": 0, "sampled_positive": 0, "sampled_negative": 0}
    for name, rows in tqdm(grouped.items(), desc="r228/sample/r224"):
        if not anchor_path(args, name).exists():
            continue
        image, instance, _gt, anchor = load_case(args, name)
        pos_mask = r224_positive_mask(args, image, instance, anchor, rows)
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
        y = np.zeros(idx.shape[0], dtype=np.float32)
        y[: len(pick_pos)] = 1.0
        ys.append(y)
        stats["images"] += 1
        stats["positive_pixels"] += int(pos.size)
        stats["sampled_positive"] += int(n_pos)
        stats["sampled_negative"] += int(n_neg)
    if not xs:
        raise RuntimeError("No R228 pixel samples collected.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), stats


class PrototypeSeamScorer:
    """Small numpy fallback for remote environments without scikit-learn."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PrototypeSeamScorer":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y) > 0.5
        self.center = np.nanmean(x, axis=0)
        x = np.where(np.isfinite(x), x, self.center)
        self.scale = np.nanstd(x, axis=0) + 1e-6
        z = (x - self.center) / self.scale
        pos = z[y]
        neg = z[~y]
        if pos.size == 0 or neg.size == 0:
            raise RuntimeError("PrototypeSeamScorer needs both positive and negative samples.")
        self.pos_mean = pos.mean(axis=0)
        self.neg_mean = neg.mean(axis=0)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.where(np.isfinite(x), x, self.center)
        z = (x - self.center) / self.scale
        d_pos = np.sum((z - self.pos_mean) ** 2, axis=1)
        d_neg = np.sum((z - self.neg_mean) ** 2, axis=1)
        logits = np.clip(0.5 * (d_neg - d_pos), -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        return np.stack([1.0 - prob, prob], axis=1)


def fit_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> Pipeline:
    if HistGradientBoostingClassifier is None or SimpleImputer is None:
        return PrototypeSeamScorer().fit(x, y)
    model = Pipeline(
        [
            ("impute", SimpleImputer()),
            ("clf", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, l2_regularization=0.03, random_state=args.seed)),
        ]
    )
    model.fit(x, y.astype(np.int32))
    return model


def predict_prob(model: Pipeline, image: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    feat = feature_stack(image, anchor)
    flat = feat.reshape(-1, feat.shape[-1])
    prob = model.predict_proba(flat)[:, 1]
    return prob.reshape(anchor.shape)


def source_names(args: argparse.Namespace) -> list[str]:
    if args.source_candidate_csv.exists():
        ordered: list[str] = []
        seen: set[str] = set()
        for row in read_rows(args.source_candidate_csv):
            name = str(row.get("image") or "")
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered[: args.limit_source or None]
    return names(args.raw_root, args.dataset, args.split)[: args.limit_source or None]


def generate_r226_candidates(args: argparse.Namespace, model: Pipeline) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in tqdm(source_names(args), desc="r228/generate-r226-style"):
        if not anchor_path(args, name).exists():
            continue
        image, instance, gt, anchor = load_case(args, name)
        prob = predict_prob(model, image, anchor)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        comp_labels, n_components = component_labels(anchor)
        component_order = [(int((comp_labels == comp_id).sum()), comp_id) for comp_id in range(1, n_components + 1)]
        component_order.sort(reverse=True)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        for _area, comp_id in component_order[: args.max_components_per_image]:
            component = comp_labels == comp_id
            comp_area = int(component.sum())
            if comp_area < args.min_component_area or comp_area > args.max_component_area:
                continue
            seeds = bg_contact_seeds(anchor, component, args.max_bg_components)
            pair_specs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for i, a in enumerate(seeds):
                for b in seeds[i + 1 :]:
                    dist = float(np.hypot(float(a["y"]) - float(b["y"]), float(a["x"]) - float(b["x"])))
                    if dist >= 12.0:
                        pair_specs.append((-abs(dist - 42.0), a, b))
            pair_specs.sort(key=lambda item: item[0], reverse=True)
            for _score, seed_a, seed_b in pair_specs[: args.max_pairs_per_component]:
                raw_cut, bridge = seed_pair_cut(component, seed_a, seed_b, image, args.corridor_radius, args.min_cut_area)
                raw_cut &= anchor
                if int(raw_cut.sum()) < args.min_cut_area:
                    continue
                cut = trim_r226_cut(image, anchor, raw_cut, bridge, args.action_frac, args.min_cut_area) & anchor
                if int(cut.sum()) < args.min_cut_area or int(cut.sum()) > max_pixels:
                    continue
                trial = anchor & ~cut
                trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                row: dict[str, Any] = {
                    "image": name,
                    "pred_component_id": float(comp_id),
                    "cut_area": float(cut.sum()),
                    "seam_prob_mean": float(np.mean(prob[cut])),
                    "seam_prob_p90": float(np.percentile(prob[cut], 90)),
                    "cut_gt_fg_frac": safe_divide(float(np.logical_and(cut, instance > 0).sum()), float(cut.sum())),
                    "cut_gt_gap_frac": safe_divide(float(np.logical_and(cut, instance == 0).sum()), float(cut.sum())),
                }
                for metric, value in trial_metrics.items():
                    row[f"candidate_{metric}"] = value
                    row[f"delta_{metric}"] = float(value) - float(anchor_metrics[metric])
                row["r228_quick_useful"] = float(is_quick_useful(row))
                row["r228_hard_risk"] = float(is_hard_risk(row))
                row["r228_safe_gap_positive"] = float(row["cut_gt_fg_frac"] <= 0.25 and row["cut_gt_gap_frac"] >= 0.75)
                row["r228_overerosion_proxy"] = float(row["cut_gt_fg_frac"] > 0.5)
                rows.append(row)
    return rows


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [as_float(row, key, default=np.nan) for row in rows]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({str(row.get("image")) for row in rows}),
        "num_quick_useful": int(sum(as_float(row, "r228_quick_useful") > 0.5 for row in rows)),
        "num_hard_risk": int(sum(as_float(row, "r228_hard_risk") > 0.5 for row in rows)),
        "num_safe_gap_positive": int(sum(as_float(row, "r228_safe_gap_positive") > 0.5 for row in rows)),
        "num_overerosion_proxy": int(sum(as_float(row, "r228_overerosion_proxy") > 0.5 for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
        "mean_seam_prob_mean": mean_value(rows, "seam_prob_mean"),
        "mean_seam_prob_p90": mean_value(rows, "seam_prob_p90"),
    }


def evaluate_grid(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for thr in parse_float_list(args.score_thresholds):
        selected = [row for row in rows if as_float(row, "seam_prob_mean") >= thr and as_float(row, "r228_overerosion_proxy") <= 0.5]
        summary = summarize_group(selected)
        grid.append({"seam_prob_threshold": thr, "accepted": summary["num_rows"], **summary})
    return grid


def choose_best(grid: list[dict[str, Any]]) -> dict[str, Any] | None:
    viable = [
        row
        for row in grid
        if int(row.get("accepted") or 0) > 0
        and int(row.get("num_hard_risk") or 0) == 0
        and int(row.get("num_overerosion_proxy") or 0) == 0
        and (row.get("mean_delta_gap_region_fp_rate") is not None and float(row["mean_delta_gap_region_fp_rate"]) < 0.0)
    ]
    pool = viable if viable else grid
    if not pool:
        return None
    return sorted(
        pool,
        key=lambda row: (
            int(row.get("accepted") or 0) > 0,
            -int(row.get("num_hard_risk") or 0),
            -int(row.get("num_overerosion_proxy") or 0),
            float(row.get("mean_delta_boundary_iou") or -999.0),
            -float(row.get("mean_delta_gap_region_fp_rate") or 999.0),
        ),
        reverse=True,
    )[0]


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


def main() -> None:
    args = parse_args()
    x, y, train_info = sample_training(args)
    model = fit_model(args, x, y)
    candidates = generate_r226_candidates(args, model)
    grid = evaluate_grid(candidates, args)
    best = choose_best(grid)
    write_csv(args.candidate_csv, candidates)
    write_csv(args.grid_csv, grid)
    safe = [row for row in candidates if as_float(row, "r228_safe_gap_positive") > 0.5]
    over = [row for row in candidates if as_float(row, "r228_overerosion_proxy") > 0.5]
    report = {
        "run_id": "R228-oracle-pixel-seam-localizer-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "original_val_pixel_scorer_candidate_diagnostic",
        "train_info": {
            **train_info,
            "num_samples": int(len(y)),
            "positive_sample_rate": float(np.mean(y)),
        },
        "num_candidates": len(candidates),
        "overall": summarize_group(candidates),
        "safe_gap_positive": summarize_group(safe),
        "overerosion_proxy": summarize_group(over),
        "grid": grid,
        "best_grid": best,
        "promotion_gate": {
            "passed": bool(best and int(best.get("accepted") or 0) > 0 and int(best.get("num_hard_risk") or 0) == 0 and int(best.get("num_overerosion_proxy") or 0) == 0),
            "note": "candidate-level only; mask writing requires broader coverage and original-val mask-level evaluation",
        },
        "warning": "R224 oracle pixels supervise the scorer; this is not clean-test-v2 evidence and not R201 final evidence.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "candidate_csv": str(args.candidate_csv), "grid_csv": str(args.grid_csv), "best_grid": best}, indent=2))


if __name__ == "__main__":
    main()
