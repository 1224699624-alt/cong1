#!/usr/bin/env python3
"""Validation-selected binary-mask ensemble for epiphysis refiner candidates."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics


@dataclass(frozen=True)
class Rule:
    base_exp: str
    add_exp: str
    add_radius: int
    max_add_component: int
    support_min: int
    close_kernel: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and apply a validation-calibrated mask ensemble.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--output-exp", required=True)
    parser.add_argument("--rule-json", required=True)
    parser.add_argument("--mode", choices=["select", "apply", "select-apply"], default="select-apply")
    parser.add_argument("--base-exps", nargs="+", required=True)
    parser.add_argument("--add-exps", nargs="+", required=True)
    parser.add_argument("--support-exps", nargs="+", default=[])
    parser.add_argument("--add-radii", default="0,1,2,3,5")
    parser.add_argument("--max-add-components", default="0,10,25,50,100,200,500,1000")
    parser.add_argument("--support-mins", default="0,1,2")
    parser.add_argument("--close-kernels", default="1,3")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--selection-boundary-kernel", type=int, default=0)
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def mask_path(root: Path, exp: str, dataset: str, split: str, name: str) -> Path:
    return root / exp / dataset / split / "masks" / name


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("L"))
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def image_names(raw_root: Path, dataset: str, split: str) -> list[str]:
    label_dir = raw_root / dataset / f"{split}_labels"
    return sorted(p.name for p in label_dir.glob("*.png"))


def gt_path(raw_root: Path, dataset: str, split: str, name: str) -> Path:
    return raw_root / dataset / f"{split}_labels" / name


def component_filter(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return np.zeros_like(mask, dtype=bool)
    if max_area >= int(mask.sum()):
        return mask
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area <= max_area:
            out |= labels == idx
    return out


def close_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return mask
    k = np.ones((kernel, kernel), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)


def apply_rule(
    root: Path,
    dataset: str,
    split: str,
    name: str,
    rule: Rule,
    support_exps: list[str],
) -> np.ndarray:
    base = read_mask(mask_path(root, rule.base_exp, dataset, split, name))
    add_src = read_mask(mask_path(root, rule.add_exp, dataset, split, name))
    add = add_src & ~base
    if rule.add_radius > 0:
        k = np.ones((rule.add_radius * 2 + 1, rule.add_radius * 2 + 1), np.uint8)
        near_base = cv2.dilate(base.astype(np.uint8), k).astype(bool)
        add &= near_base
    if support_exps and rule.support_min > 0:
        support_count = np.zeros_like(base, dtype=np.uint8)
        for exp in support_exps:
            support_count += read_mask(mask_path(root, exp, dataset, split, name)).astype(np.uint8)
        add &= support_count >= rule.support_min
    add = component_filter(add, rule.max_add_component)
    return close_mask(base | add, rule.close_kernel)


def evaluate_rule(
    raw_root: Path,
    ablations_root: Path,
    dataset: str,
    split: str,
    names: list[str],
    rule: Rule,
    support_exps: list[str],
    boundary_kernel: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for name in names:
        pred = apply_rule(ablations_root, dataset, split, name, rule, support_exps)
        gt = read_mask(gt_path(raw_root, dataset, split, name))
        metrics = compute_metrics(pred, gt, boundary_kernel)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    return {key: value / max(1, count) for key, value in totals.items()}


def preload_split(
    raw_root: Path,
    ablations_root: Path,
    dataset: str,
    split: str,
    names: list[str],
    exps: list[str],
    support_exps: list[str],
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray], list[np.ndarray]]:
    masks = {
        exp: [read_mask(mask_path(ablations_root, exp, dataset, split, name)) for name in names]
        for exp in exps
    }
    gts = [read_mask(gt_path(raw_root, dataset, split, name)) for name in names]
    support_counts: list[np.ndarray] = []
    for idx, name in enumerate(names):
        if support_exps:
            count = np.zeros_like(gts[idx], dtype=np.uint8)
            for exp in support_exps:
                if exp not in masks:
                    masks[exp] = [read_mask(mask_path(ablations_root, exp, dataset, split, n)) for n in names]
                count += masks[exp][idx].astype(np.uint8)
            support_counts.append(count)
        else:
            support_counts.append(np.zeros_like(gts[idx], dtype=np.uint8))
    return masks, gts, support_counts


def apply_rule_cached(
    masks: dict[str, list[np.ndarray]],
    support_counts: list[np.ndarray],
    idx: int,
    rule: Rule,
) -> np.ndarray:
    base = masks[rule.base_exp][idx]
    add_src = masks[rule.add_exp][idx]
    add = add_src & ~base
    if rule.add_radius > 0:
        k = np.ones((rule.add_radius * 2 + 1, rule.add_radius * 2 + 1), np.uint8)
        near_base = cv2.dilate(base.astype(np.uint8), k).astype(bool)
        add &= near_base
    if rule.support_min > 0:
        add &= support_counts[idx] >= rule.support_min
    add = component_filter(add, rule.max_add_component)
    return close_mask(base | add, rule.close_kernel)


def evaluate_rule_cached(
    masks: dict[str, list[np.ndarray]],
    gts: list[np.ndarray],
    support_counts: list[np.ndarray],
    rule: Rule,
    boundary_kernel: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for idx, gt in enumerate(gts):
        pred = apply_rule_cached(masks, support_counts, idx, rule)
        if boundary_kernel > 0:
            metrics = compute_metrics(pred, gt, boundary_kernel)
        else:
            pred = pred.astype(bool)
            gt = gt.astype(bool)
            tp = float(np.logical_and(pred, gt).sum())
            fp = float(np.logical_and(pred, ~gt).sum())
            fn = float(np.logical_and(~pred, gt).sum())
            metrics = {
                "dice": float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0,
                "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0,
                "boundary_iou": 0.0,
            }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    count = len(gts)
    return {key: value / max(1, count) for key, value in totals.items()}


def score(metrics: dict[str, float]) -> float:
    return (
        0.42 * metrics.get("dice", 0.0)
        + 0.24 * metrics.get("iou", 0.0)
        + 0.14 * metrics.get("precision", 0.0)
        + 0.08 * metrics.get("recall", 0.0)
        + 0.12 * metrics.get("boundary_iou", 0.0)
    )


def select_rule(args: argparse.Namespace) -> dict:
    raw_root = Path(args.raw_root)
    ablations_root = Path(args.ablations_root)
    names = image_names(raw_root, args.dataset, args.val_split)
    preload_exps = sorted(set(args.base_exps + args.add_exps + args.support_exps))
    masks, gts, support_counts = preload_split(
        raw_root,
        ablations_root,
        args.dataset,
        args.val_split,
        names,
        preload_exps,
        args.support_exps,
    )
    rules = [
        Rule(base, add, radius, max_comp, support_min, close_kernel)
        for base in args.base_exps
        for add in args.add_exps
        for radius in parse_ints(args.add_radii)
        for max_comp in parse_ints(args.max_add_components)
        for support_min in parse_ints(args.support_mins)
        for close_kernel in parse_ints(args.close_kernels)
    ]
    best: dict | None = None
    for rule in tqdm(rules, desc="select ensemble rule"):
        metrics = evaluate_rule_cached(masks, gts, support_counts, rule, args.selection_boundary_kernel)
        item = {"rule": asdict(rule), "metrics": metrics, "score": score(metrics)}
        if best is None or item["score"] > best["score"]:
            best = item
    assert best is not None
    output = {"dataset": args.dataset, "val_split": args.val_split, "support_exps": args.support_exps, "best": best}
    Path(args.rule_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rule_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def load_rule(path: Path) -> tuple[Rule, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Rule(**data["best"]["rule"]), list(data.get("support_exps", []))


def apply_selected(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    ablations_root = Path(args.ablations_root)
    rule, support_exps = load_rule(Path(args.rule_json))
    names = image_names(raw_root, args.dataset, args.apply_split)
    out_dir = ablations_root / args.output_exp / args.dataset / args.apply_split / "masks"
    for name in tqdm(names, desc=f"apply ensemble/{args.apply_split}"):
        pred = apply_rule(ablations_root, args.dataset, args.apply_split, name, rule, support_exps)
        write_mask(out_dir / name, pred)


def main() -> None:
    args = parse_args()
    if args.mode in {"select", "select-apply"}:
        select_rule(args)
    if args.mode in {"apply", "select-apply"}:
        apply_selected(args)


if __name__ == "__main__":
    main()
