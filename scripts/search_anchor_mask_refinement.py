#!/usr/bin/env python3
"""Search conservative anchor-mask refinements from validation labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-search anchored mask add/remove rules.")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--source-apply-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--train-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--train-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-roots", nargs="+", default=["outputs/ablations_variants", "outputs/ablations"])
    parser.add_argument("--anchor-exp", required=True)
    parser.add_argument("--support-exps", nargs="+", required=True)
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--rule-json", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--add-mins", default="2,3,4,5,6")
    parser.add_argument("--remove-maxs", default="-1,0,1")
    parser.add_argument("--near-radii", default="0,1,2,3,5")
    parser.add_argument("--boundary-radii", default="0,1,2,3,5")
    parser.add_argument("--close-kernels", default="1,3")
    parser.add_argument("--open-kernels", default="1")
    parser.add_argument("--min-add-components", default="0,5,10,25,50")
    parser.add_argument("--min-keep-components", default="0,25,50")
    parser.add_argument("--search-boundary-kernel", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def label_names(raw_root: Path, dataset: str, split: str) -> list[str]:
    return sorted(p.name for p in (raw_root / dataset / f"{split}_labels").glob("*.png"))


def find_mask(roots: list[Path], exp: str, dataset: str, split: str, name: str, source_dataset: str | None = None) -> Path:
    datasets = [dataset]
    if source_dataset and source_dataset not in datasets:
        datasets.append(source_dataset)
    for root in roots:
        for ds in datasets:
            path = root / exp / ds / split / "masks" / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Missing mask exp={exp} dataset={dataset} split={split} name={name}")


def resize_like(mask: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if mask.shape == gt.shape:
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0


def remove_small(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            out |= labels == idx
    return out


def band(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones_like(mask, dtype=bool)
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dil = cv2.dilate(mask.astype(np.uint8), k).astype(bool)
    ero = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    return dil ^ ero


def refine(anchor: np.ndarray, support_count: np.ndarray, rule: dict[str, int]) -> np.ndarray:
    out = anchor.astype(bool).copy()
    if rule["add_min"] >= 0:
        add = (support_count >= rule["add_min"]) & ~out
        if rule["near_radius"] > 0:
            k = np.ones((2 * rule["near_radius"] + 1, 2 * rule["near_radius"] + 1), np.uint8)
            add &= cv2.dilate(out.astype(np.uint8), k).astype(bool)
        add = remove_small(add, rule["min_add_component"])
        out |= add
    if rule["remove_max"] >= 0:
        rem = (support_count <= rule["remove_max"]) & out
        if rule["boundary_radius"] > 0:
            rem &= band(out, rule["boundary_radius"])
        out &= ~rem
    out = remove_small(out, rule["min_keep_component"])
    if rule["open_kernel"] > 1:
        k = np.ones((rule["open_kernel"], rule["open_kernel"]), np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, k).astype(bool)
    if rule["close_kernel"] > 1:
        k = np.ones((rule["close_kernel"], rule["close_kernel"]), np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return out


def load_split(
    roots: list[Path],
    raw_root: Path,
    dataset: str,
    split: str,
    anchor_exp: str,
    support_exps: list[str],
    source_dataset: str | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    rows = []
    for name in label_names(raw_root, dataset, split):
        gt = read_mask(raw_root / dataset / f"{split}_labels" / name)
        anchor = resize_like(read_mask(find_mask(roots, anchor_exp, dataset, split, name, source_dataset)), gt)
        support_count = np.zeros_like(gt, dtype=np.uint8)
        for exp in support_exps:
            support_count += resize_like(read_mask(find_mask(roots, exp, dataset, split, name, source_dataset)), gt).astype(np.uint8)
        rows.append((name, gt, anchor, support_count))
    return rows


def evaluate(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]], rule: dict[str, int], boundary_kernel: int) -> tuple[dict[str, float], list[dict]]:
    records = []
    for name, gt, anchor, support_count in rows:
        pred = refine(anchor, support_count, rule)
        if boundary_kernel <= 0:
            tp = float(np.logical_and(pred, gt).sum())
            fp = float(np.logical_and(pred, ~gt).sum())
            fn = float(np.logical_and(~pred, gt).sum())
            item = {
                "dice": float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0,
                "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0,
                "boundary_iou": 0.0,
            }
        else:
            item = compute_metrics(pred, gt, boundary_kernel)
        item["image"] = name
        records.append(item)
    keys = [k for k in records[0] if k != "image"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}, records


def main() -> None:
    args = parse_args()
    roots = [Path(p) for p in args.ablations_roots]
    train_rows = load_split(roots, Path(args.train_raw_root), args.train_dataset, args.train_split, args.anchor_exp, args.support_exps)
    rules = [
        {
            "add_min": add_min,
            "remove_max": remove_max,
            "near_radius": near_radius,
            "boundary_radius": boundary_radius,
            "close_kernel": close_kernel,
            "open_kernel": open_kernel,
            "min_add_component": min_add_component,
            "min_keep_component": min_keep_component,
        }
        for add_min in parse_ints(args.add_mins)
        for remove_max in parse_ints(args.remove_maxs)
        for near_radius in parse_ints(args.near_radii)
        for boundary_radius in parse_ints(args.boundary_radii)
        for close_kernel in parse_ints(args.close_kernels)
        for open_kernel in parse_ints(args.open_kernels)
        for min_add_component in parse_ints(args.min_add_components)
        for min_keep_component in parse_ints(args.min_keep_components)
    ]
    best = None
    for rule in tqdm(rules, desc="search anchor rules"):
        mean, _ = evaluate(train_rows, rule, args.search_boundary_kernel)
        score = mean["dice"]
        item = {"rule": rule, "mean": mean, "score": score}
        if best is None or item["score"] > best["score"]:
            best = item
    assert best is not None
    Path(args.rule_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rule_json).write_text(json.dumps({"best": best, "anchor_exp": args.anchor_exp, "support_exps": args.support_exps}, indent=2), encoding="utf-8")

    apply_rows = load_split(
        roots,
        Path(args.apply_raw_root),
        args.apply_dataset,
        args.apply_split,
        args.anchor_exp,
        args.support_exps,
        args.source_apply_dataset,
    )
    out_dir = Path(args.ablations_roots[0]) / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    for name, _, anchor, support_count in tqdm(apply_rows, desc="apply anchor rule"):
        write_mask(out_dir / name, refine(anchor, support_count, best["rule"]))
    mean, records = evaluate(apply_rows, best["rule"], args.boundary_kernel)
    metrics = {
        "dataset": args.apply_dataset,
        "split": args.apply_split,
        "num_evaluated": len(records),
        "mean": mean,
        "per_image": records,
        "best": best,
        "output_exp": args.output_exp,
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"mean": mean, "best": best}, indent=2))


if __name__ == "__main__":
    main()
