#!/usr/bin/env python3
"""Train a candidate-level two-stage R213 useful/risk gate.

This script is validation-candidate analysis only. It trains on train
candidate CSV rows and evaluates on validation candidate CSV rows. It does not
produce masks and is not clean-test-v2 evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_r211_f1_learned_acceptance_gate import FEATURE_KEYS


DEFAULT_USEFUL_THRESHOLDS = "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"
DEFAULT_RISK_THRESHOLDS = "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate R213 two-stage candidate gate.")
    parser.add_argument("--train-candidate-csv", type=Path, required=True)
    parser.add_argument("--val-candidate-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(""))
    parser.add_argument("--classifier", choices=["logreg", "hgb"], default="hgb")
    parser.add_argument("--useful-thresholds", default=DEFAULT_USEFUL_THRESHOLDS)
    parser.add_argument("--risk-thresholds", default=DEFAULT_RISK_THRESHOLDS)
    parser.add_argument("--max-candidates-per-image", type=int, default=1)
    parser.add_argument("--grouped-cv-folds", type=int, default=0)
    parser.add_argument(
        "--extra-feature-prefix",
        action="append",
        default=[],
        help="Append numeric CSV columns whose names start with this prefix. Default keeps the original R213 feature set.",
    )
    parser.add_argument("--seed", type=int, default=202607213)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def is_strict_useful(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") >= -5e-4
        and as_float(row, "delta_iou") >= -8e-4
        and as_float(row, "delta_recall") >= -8e-4
        and as_float(row, "delta_boundary_iou") >= 1e-4
        and as_float(row, "delta_boundary_f1") >= 1e-4
        and as_float(row, "delta_surface_dice_2px") >= 1e-4
        and as_float(row, "delta_gap_region_fp_rate") < 0.0
        and as_float(row, "delta_component_count_mae") <= 0.0
    )


def is_hard_risk(row: dict[str, Any]) -> bool:
    return bool(
        as_float(row, "delta_dice") < -5e-4
        or as_float(row, "delta_iou") < -8e-4
        or as_float(row, "delta_recall") < -8e-4
        or as_float(row, "delta_boundary_iou") < 0.0
        or as_float(row, "delta_boundary_f1") < 0.0
        or as_float(row, "delta_surface_dice_2px") < 0.0
    )


def feature_keys(rows: list[dict[str, Any]], prefixes: list[str] | None = None) -> list[str]:
    keys = list(FEATURE_KEYS)
    prefixes = prefixes or []
    for prefix in prefixes:
        extra = sorted({key for row in rows for key in row if key.startswith(prefix)})
        for key in extra:
            if key not in keys:
                keys.append(key)
    return keys


def matrix(rows: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    return np.asarray([[as_float(row, key) for key in keys] for row in rows], dtype=np.float32)


def labels(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    if target == "useful":
        return np.asarray([int(is_strict_useful(row)) for row in rows], dtype=np.int32)
    if target == "risk":
        return np.asarray([int(is_hard_risk(row)) for row in rows], dtype=np.int32)
    raise ValueError(target)


def make_model(kind: str, seed: int) -> Pipeline:
    if kind == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        )
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.04,
                    random_state=seed,
                    l2_regularization=0.02,
                ),
            ),
        ]
    )


def fit_model(rows: list[dict[str, Any]], target: str, kind: str, seed: int, keys: list[str]) -> Pipeline:
    y = labels(rows, target)
    if len(set(y.tolist())) < 2:
        raise RuntimeError(f"{target} labels contain only one class")
    model = make_model(kind, seed)
    model.fit(matrix(rows, keys), y)
    return model


def add_predictions(rows: list[dict[str, Any]], useful_model: Pipeline, risk_model: Pipeline, keys: list[str]) -> None:
    x = matrix(rows, keys)
    useful_probs = useful_model.predict_proba(x)[:, 1]
    risk_probs = risk_model.predict_proba(x)[:, 1]
    for row, useful_prob, risk_prob in zip(rows, useful_probs, risk_probs):
        row["r213_useful_prob"] = float(useful_prob)
        row["r213_risk_prob"] = float(risk_prob)
        row["strict_useful"] = float(is_strict_useful(row))
        row["hard_risk"] = float(is_hard_risk(row))


def add_grouped_cv_predictions(rows: list[dict[str, Any]], kind: str, folds: int, seed: int, keys: list[str]) -> dict[str, Any]:
    images = sorted({str(row["image"]) for row in rows})
    if folds < 2:
        raise ValueError("--grouped-cv-folds must be >=2")
    folds = min(folds, len(images))
    image_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_to_rows[str(row["image"])].append(row)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_summaries = []
    for fold_idx, (train_index, val_index) in enumerate(splitter.split(images), start=1):
        train_images = {images[i] for i in train_index}
        val_images = {images[i] for i in val_index}
        train_rows = [row for image in train_images for row in image_to_rows[image]]
        val_rows = [row for image in val_images for row in image_to_rows[image]]
        useful_model = fit_model(train_rows, "useful", kind, seed + fold_idx, keys)
        risk_model = fit_model(train_rows, "risk", kind, seed + 100 + fold_idx, keys)
        x = matrix(val_rows, keys)
        useful_probs = useful_model.predict_proba(x)[:, 1]
        risk_probs = risk_model.predict_proba(x)[:, 1]
        for row, useful_prob, risk_prob in zip(val_rows, useful_probs, risk_probs):
            row["r213_useful_prob"] = float(useful_prob)
            row["r213_risk_prob"] = float(risk_prob)
            row["strict_useful"] = float(is_strict_useful(row))
            row["hard_risk"] = float(is_hard_risk(row))
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_images": len(train_images),
                "val_images": len(val_images),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_summary": class_summary(train_rows),
                "val_summary": class_summary(val_rows),
            }
        )
    return {"folds": folds, "fold_summaries": fold_summaries}


def group_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["image"])].append(row)
    return grouped


def evaluate_policy(
    grouped: dict[str, list[dict[str, Any]]],
    useful_threshold: float,
    risk_threshold: float,
    max_candidates_per_image: int,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        ranked = sorted(
            image_rows,
            key=lambda row: (as_float(row, "r213_useful_prob") - as_float(row, "r213_risk_prob"), as_float(row, "r213_useful_prob")),
            reverse=True,
        )
        image_accepted = 0
        for row in ranked:
            if image_accepted >= max_candidates_per_image:
                break
            if as_float(row, "r213_useful_prob") < useful_threshold:
                continue
            if as_float(row, "r213_risk_prob") > risk_threshold:
                continue
            accepted.append(row)
            image_accepted += 1
    n = len(accepted)
    useful = [row for row in accepted if as_float(row, "strict_useful") > 0.5]
    risk = [row for row in accepted if as_float(row, "hard_risk") > 0.5]
    positive = [row for row in accepted if as_float(row, "label_positive") > 0.5]
    score = len(useful) - 2.0 * len(risk) + 0.2 * n
    return {
        "useful_threshold": useful_threshold,
        "risk_threshold": risk_threshold,
        "accepted": n,
        "positive": len(positive),
        "strict_useful": len(useful),
        "hard_risk": len(risk),
        "precision_useful": float(len(useful) / n) if n else 0.0,
        "risk_rate": float(len(risk) / n) if n else 0.0,
        "score": float(score),
        "accepted_images": sorted({str(row["image"]) for row in accepted}),
        "useful_images": sorted({str(row["image"]) for row in useful}),
        "risk_images": sorted({str(row["image"]) for row in risk}),
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "useful_threshold",
        "risk_threshold",
        "accepted",
        "positive",
        "strict_useful",
        "hard_risk",
        "precision_useful",
        "risk_rate",
        "score",
    ]
    return {key: row[key] for key in keys}


def frontiers(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for max_risk in [0, 1, 2, 5, 10, 20]:
        subset = [row for row in results if int(row["hard_risk"]) <= max_risk]
        subset = sorted(subset, key=lambda row: (row["strict_useful"], row["precision_useful"], row["accepted"]), reverse=True)
        out[f"risk_le_{max_risk}"] = [compact(row) for row in subset[:20]]
    for min_useful in [2, 3, 5, 10, 15]:
        subset = [row for row in results if int(row["strict_useful"]) >= min_useful]
        subset = sorted(subset, key=lambda row: (-row["hard_risk"], row["precision_useful"], row["strict_useful"]), reverse=True)
        out[f"useful_ge_{min_useful}"] = [compact(row) for row in subset[:20]]
    return out


def class_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    useful = [row for row in rows if is_strict_useful(row)]
    risk = [row for row in rows if is_hard_risk(row)]
    return {
        "num_rows": len(rows),
        "num_images": len({str(row["image"]) for row in rows}),
        "strict_useful": len(useful),
        "hard_risk": len(risk),
        "strict_useful_images": len({str(row["image"]) for row in useful}),
        "hard_risk_images": len({str(row["image"]) for row in risk}),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "useful_threshold",
        "risk_threshold",
        "accepted",
        "positive",
        "strict_useful",
        "hard_risk",
        "precision_useful",
        "risk_rate",
        "score",
        "accepted_images",
        "useful_images",
        "risk_images",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ["accepted_images", "useful_images", "risk_images"]:
                out[key] = "|".join(out.get(key) or [])
            writer.writerow(out)


def evidence_level(args: argparse.Namespace) -> str:
    if args.grouped_cv_folds > 0:
        return "candidate_grouped_cv_diagnostic"
    if args.train_candidate_csv == args.val_candidate_csv:
        return "candidate_selffit_probe"
    return "candidate_train_val_diagnostic"


def protocol_warning(args: argparse.Namespace) -> str:
    level = evidence_level(args)
    if level == "candidate_selffit_probe":
        return "self-fit candidate probe only; optimistic and not valid model evidence"
    if level == "candidate_grouped_cv_diagnostic":
        return "image-grouped candidate CV diagnostic only; no masks and no clean-test-v2 evidence"
    return "candidate train-to-val diagnostic only; requires mask-level R201 validation before promotion"


def main() -> None:
    args = parse_args()
    train_rows = read_csv(args.train_candidate_csv)
    val_rows = read_csv(args.val_candidate_csv)
    keys = feature_keys(train_rows + val_rows, args.extra_feature_prefix)
    cv_summary = None
    useful_model = None
    risk_model = None
    if args.grouped_cv_folds > 0:
        if args.train_candidate_csv != args.val_candidate_csv:
            raise ValueError("--grouped-cv-folds expects train and val candidate CSVs to be the same file")
        cv_summary = add_grouped_cv_predictions(val_rows, args.classifier, args.grouped_cv_folds, args.seed, keys)
    else:
        useful_model = fit_model(train_rows, "useful", args.classifier, args.seed, keys)
        risk_model = fit_model(train_rows, "risk", args.classifier, args.seed + 17, keys)
        add_predictions(val_rows, useful_model, risk_model, keys)
    grouped = group_by_image(val_rows)
    results = []
    for useful_threshold in parse_float_list(args.useful_thresholds):
        for risk_threshold in parse_float_list(args.risk_thresholds):
            results.append(evaluate_policy(grouped, useful_threshold, risk_threshold, args.max_candidates_per_image))
    ranked = sorted(results, key=lambda row: (row["score"], row["strict_useful"], -row["hard_risk"]), reverse=True)
    report = {
        "run_id": "R213-two-stage-candidate-gate",
        "evidence_level": evidence_level(args),
        "protocol_warning": protocol_warning(args),
        "clean_test_v2_used": False,
        "writes_masks": False,
        "train_candidate_csv": str(args.train_candidate_csv),
        "val_candidate_csv": str(args.val_candidate_csv),
        "classifier": args.classifier,
        "grouped_cv_folds": args.grouped_cv_folds,
        "feature_keys": keys,
        "extra_feature_prefix": args.extra_feature_prefix,
        "train_summary": class_summary(train_rows),
        "val_summary": class_summary(val_rows),
        "cv_summary": cv_summary,
        "num_policies": len(results),
        "best_by_score": [compact(row) for row in ranked[:40]],
        "frontiers": frontiers(results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(args.output_csv, ranked[:200])
    if args.checkpoint != Path(".") and useful_model is not None and risk_model is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "useful_model": useful_model,
                "risk_model": risk_model,
                "feature_keys": keys,
                "args": vars(args),
            },
            args.checkpoint,
        )
    print(json.dumps({"output_json": str(args.output_json), "output_csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
