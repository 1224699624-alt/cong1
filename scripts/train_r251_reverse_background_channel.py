#!/usr/bin/env python3
"""R251-S0 dense reverse-background channel sanity gate.

Train a dense scorer inside a non-leaking anchor mask. The positive target is
background derived from inverted bone GT, while foreground bone pixels are the
bone-preservation negative class. GT is used only to build train/eval targets;
the prediction function accepts image and anchor only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import platform
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from audit_r228_oracle_pixel_seam_localizer import feature_stack
from audit_r244_gtfree_merge_targeted_candidates import parse_float_list
from run_r209_component_preserving_feasibility_audit import read_instance
from run_r210_f1_neck_candidate_gate import candidate_components
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train R251 dense reverse-background channel scorer.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="train")
    parser.add_argument("--limit-train", type=int, default=8)
    parser.add_argument("--limit-eval", type=int, default=8)
    parser.add_argument("--max-scan-images", type=int, default=64)
    parser.add_argument("--dist-percentiles", default="6,8,10,12")
    parser.add_argument("--min-cut-area", type=int, default=3)
    parser.add_argument("--max-cut-frac", type=float, default=0.008)
    parser.add_argument("--max-candidates-generated", type=int, default=32)
    parser.add_argument("--hard-negative-ring", type=int, default=5)
    parser.add_argument("--min-unique-positive", type=int, default=8)
    parser.add_argument("--min-unique-corridor-negative", type=int, default=8)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--positive-samples-per-image", type=int, default=2048)
    parser.add_argument("--eval-samples-per-image", type=int, default=8192)
    parser.add_argument("--max-iter", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2-regularization", type=float, default=0.03)
    parser.add_argument("--min-overfit-auc", type=float, default=0.75)
    parser.add_argument("--min-overfit-ap", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=202607251)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/analysis/r251_reverse_background_channel_smoke.json"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/models/r251_reverse_background_channel_smoke.pkl"),
    )
    return parser.parse_args()


def validate_scope(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset != "TSRS_RSNA-Epiphysis":
        raise ValueError("R251-S0 is locked to TSRS_RSNA-Epiphysis.")
    if args.train_split != "train" or args.eval_split != "train":
        raise ValueError("R251-S0 is a train/train overfit gate; other splits are forbidden.")
    if args.limit_train <= 0 or args.limit_eval <= 0 or args.limit_train != args.limit_eval:
        raise ValueError("R251-S0 requires equal positive --limit-train/--limit-eval values.")
    if args.max_scan_images < args.limit_train:
        raise ValueError("--max-scan-images must cover --limit-train.")
    if args.samples_per_image <= 1 or args.eval_samples_per_image <= 1:
        raise ValueError("Sample counts must be greater than one.")
    if not 0 < args.positive_samples_per_image < args.samples_per_image:
        raise ValueError("--positive-samples-per-image must be between zero and --samples-per-image.")
    if not 0.0 < args.max_cut_frac <= 1.0:
        raise ValueError("--max-cut-frac must be in (0, 1].")
    raw_root = args.raw_root.resolve()
    dataset_root = (raw_root / args.dataset).resolve()
    forbidden = "clean_test" in str(dataset_root).lower() or "articular-surface" in str(dataset_root).lower()
    if raw_root.name != "raw" or forbidden:
        raise ValueError("R251-S0 requires the original data/raw Epiphysis root.")
    required = [dataset_root / "train", dataset_root / "train_labels"]
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError(f"Missing original train paths: {required}")
    return {
        "dataset_root": str(dataset_root),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "clean_test_v2_used": False,
        "validated_original_raw": True,
    }


def anchor_path(args: argparse.Namespace, split: str, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / split / "masks" / name


def load_case(args: argparse.Namespace, split: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, split, name))
    anchor = resize_like(read_mask(anchor_path(args, split, name)), gt.shape)
    return image, gt, anchor


def candidate_corridor(args: argparse.Namespace, image: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Build GT-free R244-style corridor support from image and anchor only."""
    corridor = np.zeros_like(anchor, dtype=bool)
    max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
    for dist_percentile in parse_float_list(args.dist_percentiles):
        cuts = candidate_components(anchor, image, dist_percentile, args.min_cut_area)
        for raw_cut in cuts[: args.max_candidates_generated]:
            cut = raw_cut.astype(bool) & anchor
            area = int(cut.sum())
            if args.min_cut_area <= area <= max_pixels:
                corridor |= cut
    return corridor


def supervision_regions(
    args: argparse.Namespace,
    image: np.ndarray,
    gt: np.ndarray,
    anchor: np.ndarray,
) -> dict[str, np.ndarray]:
    corridor = candidate_corridor(args, image, anchor)
    positive = corridor & ~gt
    corridor_negative = corridor & gt
    ring = ndimage.binary_dilation(corridor, iterations=args.hard_negative_ring) & anchor & ~corridor & gt
    dist_in = ndimage.distance_transform_edt(anchor)
    cutoff = float(np.percentile(dist_in[anchor], 60)) if anchor.any() else 0.0
    interior = anchor & gt & (dist_in >= cutoff)
    hard_negative = corridor_negative | ring | interior
    return {
        "corridor": corridor,
        "positive": positive,
        "corridor_negative": corridor_negative,
        "ring_negative": ring,
        "interior_negative": interior,
        "hard_negative": hard_negative,
    }


def select_case_names(
    args: argparse.Namespace,
) -> tuple[list[str], list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]]]:
    selected: list[str] = []
    audit: list[dict[str, Any]] = []
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]] = {}
    for name in names(args.raw_root, args.dataset, args.train_split)[: args.max_scan_images]:
        if not anchor_path(args, args.train_split, name).exists():
            audit.append({"image": name, "selected": False, "reason": "missing_anchor"})
            continue
        image, gt, anchor = load_case(args, args.train_split, name)
        regions = supervision_regions(args, image, gt, anchor)
        counts = {key: int(value.sum()) for key, value in regions.items()}
        usable = bool(
            counts["positive"] >= args.min_unique_positive
            and counts["corridor_negative"] >= args.min_unique_corridor_negative
        )
        audit.append({"image": name, "selected": usable, **counts})
        if usable:
            selected.append(name)
            cache[name] = (image, gt, anchor, regions)
        if len(selected) >= args.limit_train:
            break
    return selected, audit, cache


def balanced_indices(
    positive: np.ndarray,
    negative: np.ndarray,
    total: int,
    positive_cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.flatnonzero(positive.ravel())
    neg = np.flatnonzero(negative.ravel())
    if pos.size == 0 or neg.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    n_pos = min(positive_cap, max(1, total // 2))
    n_neg = max(1, total - n_pos)
    pick_pos = rng.choice(pos, size=n_pos, replace=pos.size < n_pos)
    pick_neg = rng.choice(neg, size=n_neg, replace=neg.size < n_neg)
    idx = np.concatenate([pick_pos, pick_neg])
    labels = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(n_neg, dtype=np.float32)])
    order = rng.permutation(len(idx))
    return idx[order], labels[order]


def collect_training(
    args: argparse.Namespace,
    case_names: list[str],
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    stats: dict[str, Any] = {
        "requested_images": args.limit_train,
        "used_images": 0,
        "skipped_missing_anchor": 0,
        "skipped_single_class": 0,
        "anchor_pixels": 0,
        "background_positive_pixels": 0,
        "bone_negative_pixels": 0,
        "case_names": [],
        "per_region_counts": [],
    }
    for name in case_names:
        if not anchor_path(args, args.train_split, name).exists():
            stats["skipped_missing_anchor"] += 1
            continue
        image, gt, anchor, regions = cache[name]
        positive = regions["positive"]
        negative = regions["hard_negative"]
        idx, labels = balanced_indices(
            positive,
            negative,
            args.samples_per_image,
            args.positive_samples_per_image,
            rng,
        )
        if idx.size == 0:
            stats["skipped_single_class"] += 1
            continue
        feat = feature_stack(image, anchor)
        xs.append(feat.reshape(-1, feat.shape[-1])[idx])
        ys.append(labels)
        stats["used_images"] += 1
        stats["case_names"].append(name)
        stats["anchor_pixels"] += int(anchor.sum())
        stats["background_positive_pixels"] += int(positive.sum())
        stats["bone_negative_pixels"] += int(negative.sum())
        stats["per_region_counts"].append(
            {"image": name, **{key: int(value.sum()) for key, value in regions.items()}}
        )
    if not xs:
        raise RuntimeError("No two-class R251 training samples were collected.")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    stats["num_samples"] = int(len(y))
    stats["sample_positive_rate"] = float(np.mean(y))
    stats["raw_positive_rate_inside_anchor"] = float(
        stats["background_positive_pixels"] / max(1, stats["anchor_pixels"])
    )
    return x, y, stats


def fit_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("impute", SimpleImputer()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=args.max_iter,
                    learning_rate=args.learning_rate,
                    l2_regularization=args.l2_regularization,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(x, y.astype(np.int32))
    return model


def predict_background_prob(model: Pipeline, image: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """GT-free inference: only image and anchor-derived features are accepted."""
    feat = feature_stack(image, anchor)
    prob = model.predict_proba(feat.reshape(-1, feat.shape[-1]))[:, 1]
    return prob.reshape(anchor.shape)


def collect_eval(
    args: argparse.Namespace,
    model: Pipeline,
    case_names: list[str],
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(args.seed + 1)
    labels_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []
    missing = 0
    eval_case_names: list[str] = []
    for name in case_names:
        if not anchor_path(args, args.eval_split, name).exists():
            missing += 1
            continue
        image, gt, anchor, regions = cache[name]
        positive = regions["positive"]
        negative = regions["corridor_negative"]
        idx, labels = balanced_indices(
            positive,
            negative,
            args.eval_samples_per_image,
            args.eval_samples_per_image // 2,
            rng,
        )
        if idx.size == 0:
            continue
        prob = predict_background_prob(model, image, anchor).ravel()[idx]
        labels_all.append(labels)
        probs_all.append(prob.astype(np.float32))
        per_image.append(
            {
                "image": name,
                "num_samples": int(len(labels)),
                "positive_rate": float(np.mean(labels)),
                "roc_auc": float(roc_auc_score(labels, prob)),
                "average_precision": float(average_precision_score(labels, prob)),
            }
        )
        eval_case_names.append(name)
    if not labels_all:
        raise RuntimeError("No two-class R251 evaluation samples were collected.")
    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    summary = {
        "used_images": len(per_image),
        "missing_anchors": missing,
        "num_samples": int(len(labels)),
        "positive_rate": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "average_precision": float(average_precision_score(labels, probs)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prob_mean": float(np.mean(probs)),
        "per_image": per_image,
        "case_names": eval_case_names,
    }
    return labels, probs, summary


def main() -> None:
    args = parse_args()
    scope = validate_scope(args)
    case_names, selection_audit, cache = select_case_names(args)
    if len(case_names) != args.limit_train:
        raise RuntimeError(
            f"R251-S0 needs {args.limit_train} usable corridor cases, found {len(case_names)} "
            f"within {args.max_scan_images} scanned images."
        )
    x, y, train_info = collect_training(args, case_names, cache)
    model = fit_model(args, x, y)
    _labels, _probs, eval_info = collect_eval(args, model, case_names, cache)
    same_split = args.train_split == args.eval_split
    exact_case_identity = train_info["case_names"] == eval_info["case_names"] == case_names
    complete_case_set = bool(
        train_info["used_images"] == args.limit_train
        and eval_info["used_images"] == args.limit_eval
        and train_info["skipped_missing_anchor"] == 0
        and train_info["skipped_single_class"] == 0
        and eval_info["missing_anchors"] == 0
    )
    gate_pass = bool(
        same_split
        and exact_case_identity
        and complete_case_set
        and eval_info["roc_auc"] >= args.min_overfit_auc
        and eval_info["average_precision"] >= args.min_overfit_ap
        and eval_info["prob_max"] - eval_info["prob_min"] >= 0.25
    )
    report = {
        "run_id": "R251-S0-reverse-background-channel-sanity",
        "dataset": args.dataset,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "anchor_exp": args.anchor_exp,
        "clean_test_v2_used": scope["clean_test_v2_used"],
        "writes_masks": False,
        "scope_validation": scope,
        "target_definition": "positive = GT-free candidate corridor AND NOT bone_GT; negatives = bone_GT in corridor/ring/interior",
        "inference_inputs": ["image", "anchor"],
        "gt_allowed_at_inference": False,
        "train_info": train_info,
        "eval_info": eval_info,
        "selection_audit": selection_audit,
        "reproducibility": {
            "seed": args.seed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "feature_stack_source": "audit_r228_oracle_pixel_seam_localizer.feature_stack",
            "dist_percentiles": args.dist_percentiles,
            "max_candidates_generated": args.max_candidates_generated,
        },
        "sanity_gate": {
            "same_split_overfit": same_split,
            "exact_case_identity": exact_case_identity,
            "complete_case_set": complete_case_set,
            "min_roc_auc": args.min_overfit_auc,
            "min_average_precision": args.min_overfit_ap,
            "min_probability_range": 0.25,
            "passed": gate_pass,
        },
        "next_step": "prepare image-grouped train-to-original-val R251 candidate safety audit" if gate_pass else "fix target/features before full R251",
        "warning": "Train-only overfit sanity; not original-val model evidence and not R201/clean-test-v2 evidence.",
    }
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_json = args.output_json.with_name(f"{args.output_json.stem}_{stamp}{args.output_json.suffix}")
    report["timestamped_output_json"] = str(timestamped_json)
    report["model_written"] = gate_pass
    payload = json.dumps(report, indent=2)
    timestamped_json.parent.mkdir(parents=True, exist_ok=True)
    timestamped_json.write_text(payload, encoding="utf-8")
    args.output_json.write_text(payload, encoding="utf-8")
    model_output: str | None = None
    if gate_pass:
        timestamped_model = args.model_path.with_name(f"{args.model_path.stem}_{stamp}{args.model_path.suffix}")
        timestamped_model.parent.mkdir(parents=True, exist_ok=True)
        with timestamped_model.open("wb") as handle:
            pickle.dump(model, handle)
        shutil.copyfile(timestamped_model, args.model_path)
        model_output = str(args.model_path)
    print(json.dumps({"output_json": str(args.output_json), "timestamped_output_json": str(timestamped_json), "model_path": model_output, "sanity_gate": report["sanity_gate"], "eval": {key: eval_info[key] for key in ["used_images", "num_samples", "roc_auc", "average_precision", "prob_min", "prob_max"]}}, indent=2))
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
