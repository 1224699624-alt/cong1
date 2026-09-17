#!/usr/bin/env python3
"""R186 fast neck-cut gate with non-zero validation edit requirement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from evaluate_masks import compute_metrics


TARGET = 0.9317660066557425
R110_CLEAN_DICE = 0.9177231563529792
R110_CLEAN_BOUNDARY_IOU = 0.25189601044085763


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R186 fast neck gate.")
    parser.add_argument("--tune-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--tune-raw-root", default="data/raw")
    parser.add_argument("--apply-raw-root", default="data/raw_variants")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--output-exp", default="r186_fast_neck_gate")
    parser.add_argument("--val-summary-json", default="outputs/analysis/r186_fast_neck_gate_val_summary.json")
    parser.add_argument("--metrics-json", default="outputs/analysis/r186_fast_neck_gate_clean_test_v2_metrics.json")
    parser.add_argument("--result-json", default="outputs/analysis/r186_fast_neck_gate_result_summary.json")
    parser.add_argument("--neck-widths", default="1,2,3")
    parser.add_argument("--band-radii", default="2,3,5")
    parser.add_argument("--min-areas", default="1,2,4,8")
    parser.add_argument("--max-remove-fracs", default="0.00025,0.0005,0.001,0.002")
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--min-val-edited-frac", type=float, default=1e-7)
    parser.add_argument("--val-dice-drop-tol", type=float, default=0.00025)
    parser.add_argument("--val-boundary-drop-tol", type=float, default=0.0)
    parser.add_argument("--val-component-tol", type=float, default=0.0)
    parser.add_argument("--allow-clean-test-on-gate-fail", action="store_true")
    return parser.parse_args()


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def resize_like(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def component_count(mask: np.ndarray) -> int:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(n)


def separation_band(gt: np.ndarray, radius: int = 7) -> np.ndarray:
    structure = np.ones((radius, radius), dtype=bool)
    return ndimage.binary_dilation(gt, structure=structure) & ~gt


def add_structure_metrics(rec: dict[str, float], pred: np.ndarray, gt: np.ndarray, edited: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    sep = separation_band(gt)
    rec["pred_component_count"] = float(pred_count)
    rec["gt_component_count"] = float(gt_count)
    rec["component_count_error"] = float(abs(pred_count - gt_count))
    rec["false_bridge_flag"] = float(pred_count < gt_count)
    rec["sep_fp_rate"] = float(np.logical_and(pred, sep).sum() / max(1, int(sep.sum())))
    rec["edited_frac"] = float(edited.sum() / max(1, edited.size))
    return rec


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {k: float(np.mean([float(r[k]) for r in records])) for k in keys}


def anchor_root(ablations_root: Path, exp: str, dataset: str, split: str) -> Path:
    root = ablations_root / exp / dataset / split / "masks"
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def load_items(raw_root: Path, dataset: str, split: str, mask_root: Path, limit: int) -> list[dict[str, object]]:
    label_paths = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
    if limit and limit > 0:
        label_paths = label_paths[:limit]
    items = []
    for label_path in label_paths:
        gt = read_mask(label_path)
        anchor = resize_like(read_mask(mask_root / label_path.name), gt.shape)
        items.append({"name": label_path.name, "gt": gt, "anchor": anchor})
    if not items:
        raise FileNotFoundError(raw_root / dataset / f"{split}_labels")
    return items


def fast_neck_cut(anchor: np.ndarray, neck_width: int, band_radius: int, min_area: int, max_remove_frac: float) -> np.ndarray:
    structure = np.ones((max(1, neck_width), max(1, neck_width)), dtype=bool)
    opened = ndimage.binary_opening(anchor, structure=structure)
    thin = anchor & ~opened
    band = anchor & ndimage.binary_dilation(~anchor, structure=np.ones((band_radius, band_radius), dtype=bool))
    candidates = thin & band
    labels, n = ndimage.label(candidates, structure=np.ones((3, 3), dtype=np.uint8))
    pieces = []
    for idx in range(1, n + 1):
        comp = labels == idx
        area = int(comp.sum())
        if area >= min_area:
            pieces.append((area, comp))
    if not pieces:
        return anchor.copy()
    max_remove = int(round(float(max_remove_frac) * max(1, int(anchor.sum()))))
    if max_remove <= 0:
        return anchor.copy()
    pieces.sort(key=lambda x: x[0])
    remove = np.zeros_like(anchor, dtype=bool)
    used = 0
    for area, comp in pieces:
        if used + area > max_remove:
            continue
        remove |= comp
        used += area
    out = anchor.copy()
    out[remove] = False
    return out


def accept_with_gt(anchor: np.ndarray, candidate: np.ndarray, gt: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if np.array_equal(anchor, candidate):
        return anchor
    am = compute_metrics(anchor, gt, args.boundary_kernel)
    cm = compute_metrics(candidate, gt, args.boundary_kernel)
    anchor_comp = abs(component_count(anchor) - component_count(gt))
    cand_comp = abs(component_count(candidate) - component_count(gt))
    if cand_comp > anchor_comp + args.val_component_tol:
        return anchor
    if cm["dice"] < am["dice"] - args.val_dice_drop_tol:
        return anchor
    if cm["boundary_iou"] < am["boundary_iou"] - args.val_boundary_drop_tol:
        return anchor
    return candidate


def apply_cfg(items: list[dict[str, object]], cfg: dict[str, float], args: argparse.Namespace, accept: bool, write_dir: Path | None = None):
    records = []
    for item in items:
        gt = item["gt"]  # type: ignore[assignment]
        anchor = item["anchor"]  # type: ignore[assignment]
        cand = fast_neck_cut(anchor, int(cfg["neck_width"]), int(cfg["band_radius"]), int(cfg["min_area"]), float(cfg["max_remove_frac"]))
        pred = accept_with_gt(anchor, cand, gt, args) if accept else cand
        edited = pred != anchor
        if write_dir is not None:
            write_mask(write_dir / str(item["name"]), pred)
        rec = compute_metrics(pred, gt, args.boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt, edited)
        rec["image"] = str(item["name"])
        records.append(rec)
    return mean_records(records), records


def configs(args: argparse.Namespace) -> list[dict[str, float]]:
    out = []
    for neck in parse_ints(args.neck_widths):
        for band in parse_ints(args.band_radii):
            for min_area in parse_ints(args.min_areas):
                for max_remove in parse_floats(args.max_remove_fracs):
                    out.append(
                        {
                            "neck_width": float(neck),
                            "band_radius": float(band),
                            "min_area": float(min_area),
                            "max_remove_frac": float(max_remove),
                        }
                    )
    return out


def gate(mean: dict[str, float], anchor: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        mean["edited_frac"] >= args.min_val_edited_frac
        and mean["dice"] >= anchor["dice"] - args.val_dice_drop_tol
        and mean["boundary_iou"] >= anchor["boundary_iou"] - args.val_boundary_drop_tol
        and mean["component_count_error"] <= anchor["component_count_error"] + args.val_component_tol
        and mean["false_bridge_flag"] <= anchor["false_bridge_flag"] + 1e-9
    )


def main() -> None:
    args = parse_args()
    ab_root = Path(args.ablations_root)
    tune_items = load_items(Path(args.tune_raw_root), args.tune_dataset, args.tune_split, anchor_root(ab_root, args.anchor_exp, args.tune_dataset, args.tune_split), args.max_val_images)
    anchor_cfg = {"neck_width": 1.0, "band_radius": 1.0, "min_area": 999999.0, "max_remove_frac": 0.0}
    anchor_mean, anchor_records = apply_cfg(tune_items, anchor_cfg, args, accept=False)
    grid = []
    best = None
    cfg_list = configs(args)
    for idx, cfg in enumerate(cfg_list, start=1):
        print(json.dumps({"stage": "val_grid", "idx": idx, "total": len(cfg_list), "cfg": cfg}), flush=True)
        mean, _ = apply_cfg(tune_items, cfg, args, accept=True)
        passed = gate(mean, anchor_mean, args)
        item = {"cfg": cfg, "mean": mean, "passes_gate": passed}
        grid.append(item)
        if not passed:
            continue
        score = (
            mean["dice"] - anchor_mean["dice"],
            mean["boundary_iou"] - anchor_mean["boundary_iou"],
            -(mean["component_count_error"] - anchor_mean["component_count_error"]),
            -(mean["false_bridge_flag"] - anchor_mean["false_bridge_flag"]),
            mean["edited_frac"],
        )
        if best is None or score > best["score"]:
            best = item | {"score": score}
            print(json.dumps({"stage": "new_best", "idx": idx, "mean": mean, "score": score}), flush=True)
    val_summary = {
        "run_id": "R186",
        "anchor_exp": args.anchor_exp,
        "apply_anchor_exp": args.apply_anchor_exp,
        "anchor_mean": anchor_mean,
        "anchor_per_image": anchor_records,
        "gate_pass": best is not None,
        "best": best,
        "grid": grid,
    }
    Path(args.val_summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_summary_json).write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
    result = {
        "run_id": "R186",
        "status": "val_gate_failed",
        "val_summary_path": args.val_summary_json,
        "metrics_path": args.metrics_json,
        "target_dice": TARGET,
        "r110_clean_dice": R110_CLEAN_DICE,
        "r110_clean_boundary_iou": R110_CLEAN_BOUNDARY_IOU,
    }
    if best is None and not args.allow_clean_test_on_gate_fail:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return
    if best is None:
        raise RuntimeError("gate failed but clean-test application was forced")
    apply_items = load_items(Path(args.apply_raw_root), args.apply_dataset, args.apply_split, anchor_root(ab_root, args.apply_anchor_exp, args.apply_dataset, args.apply_split), args.max_apply_images)
    out_dir = ab_root / args.output_exp / args.apply_dataset / args.apply_split / "masks"
    mean, records = apply_cfg(apply_items, best["cfg"], args, accept=False, write_dir=out_dir)
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
    dice = float(mean["dice"])
    result.update(
        {
            "status": "target_met" if dice >= TARGET else ("new_best_below_target" if dice > R110_CLEAN_DICE else "below_best"),
            "mean": mean,
            "target_margin": dice - TARGET,
            "dice_delta_vs_r110": dice - R110_CLEAN_DICE,
            "boundary_iou_delta_vs_r110": float(mean["boundary_iou"]) - R110_CLEAN_BOUNDARY_IOU,
            "val_best": best,
        }
    )
    Path(args.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
