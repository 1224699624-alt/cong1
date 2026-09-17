#!/usr/bin/env python3
"""R204 conservative bridge/gap edit gate.

This run is intentionally validation-first. It searches conservative edits on
train/val anchors, rejects edits that hurt recall or component-count error, and
only writes clean-test-v2 masks when the validation gate passes. The clean-test
application uses locked validation parameters; it does not tune on clean-test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R204 conservative bridge/gap validation gate.")
    parser.add_argument("--tune-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--tune-raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--apply-raw-root", type=Path, default=Path("data/raw_variants"))
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--output-exp", default="r204_conservative_bridge_gate")
    parser.add_argument("--val-summary-json", type=Path, default=Path("outputs/analysis/r204_conservative_bridge_gate_val_summary.json"))
    parser.add_argument("--result-json", type=Path, default=Path("outputs/analysis/r204_conservative_bridge_gate_result_summary.json"))
    parser.add_argument("--metrics-json", type=Path, default=Path("outputs/analysis/r204_conservative_bridge_gate_clean_test_v2_metrics.json"))
    parser.add_argument("--progress-json", type=Path, default=Path("outputs/analysis/r204_conservative_bridge_gate_progress.json"))
    parser.add_argument("--gap-radii", default="3,5,7")
    parser.add_argument("--thin-radii", default="1,2")
    parser.add_argument("--distance-fracs", default="0.10,0.14,0.18,0.22")
    parser.add_argument("--max-remove-fracs", default="0.00025,0.0005,0.001,0.0015")
    parser.add_argument("--min-cut-areas", default="1,2,4")
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--min-edited-frac", type=float, default=1e-7)
    parser.add_argument("--dice-drop-tol", type=float, default=0.0002)
    parser.add_argument("--recall-drop-tol", type=float, default=0.0002)
    parser.add_argument("--boundary-drop-tol", type=float, default=0.0)
    parser.add_argument("--component-mae-tol", type=float, default=0.0)
    parser.add_argument("--gap-fp-tol", type=float, default=0.0)
    parser.add_argument("--allow-clean-test-on-gate-fail", action="store_true")
    return parser.parse_args()


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


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


def mask_root(ablations_root: Path, exp: str, dataset: str, split: str) -> Path:
    root = ablations_root / exp / dataset / split / "masks"
    if not root.exists():
        raise FileNotFoundError(root)
    return root


def load_items(raw_root: Path, dataset: str, split: str, pred_root: Path, limit: int) -> list[dict[str, Any]]:
    label_paths = sorted((raw_root / dataset / f"{split}_labels").glob("*.png"))
    if limit > 0:
        label_paths = label_paths[:limit]
    items: list[dict[str, Any]] = []
    for gt_path in label_paths:
        pred_path = pred_root / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        gt = read_mask(gt_path)
        pred = resize_like(read_mask(pred_path), gt.shape)
        items.append({"image": gt_path.name, "gt": gt, "anchor": pred})
    if not items:
        raise FileNotFoundError(raw_root / dataset / f"{split}_labels")
    return items


def component_count(mask: np.ndarray) -> int:
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return int(n)


def gap_region(gt: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((radius, radius), dtype=bool)
    return ndimage.binary_dilation(gt, structure=structure) & ~gt


def add_diag(rec: dict[str, float], pred: np.ndarray, gt: np.ndarray, edited: np.ndarray, gap_radius: int) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    gap = gap_region(gt, gap_radius)
    rec["component_count_mae_proxy"] = float(abs(pred_count - gt_count))
    rec["component_merge_proxy"] = float(pred_count < gt_count)
    rec["gap_fp_proxy"] = float(np.logical_and(pred, gap).sum() / max(1, int(gap.sum())))
    rec["edited_frac"] = float(edited.sum() / max(1, edited.size))
    rec["removed_frac_of_anchor"] = float(np.logical_and(~pred, edited).sum() / max(1, int(pred.sum() + np.logical_and(~pred, edited).sum())))
    return rec


def internal_bridge_candidates(anchor: np.ndarray, gap: np.ndarray, thin_radius: int, distance_frac: float, min_cut_area: int) -> list[np.ndarray]:
    dist = cv2.distanceTransform(anchor.astype(np.uint8), cv2.DIST_L2, 5)
    if float(dist.max()) <= 0:
        return []
    eroded = ndimage.binary_erosion(anchor, structure=np.ones((max(1, thin_radius), max(1, thin_radius)), dtype=bool))
    thin = anchor & ~eroded
    low_dist = anchor & (dist <= float(distance_frac) * float(dist.max()))
    cand = thin & low_dist & gap
    labels, n = ndimage.label(cand, structure=np.ones((3, 3), dtype=np.uint8))
    pieces = []
    for idx in range(1, n + 1):
        piece = labels == idx
        if int(piece.sum()) >= min_cut_area:
            pieces.append(piece)
    return sorted(pieces, key=lambda p: int(p.sum()))


def conservative_edit(anchor: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    gap = ndimage.binary_dilation(~anchor, structure=np.ones((int(cfg["gap_radius"]), int(cfg["gap_radius"])), dtype=bool))
    pieces = internal_bridge_candidates(
        anchor,
        gap,
        int(cfg["thin_radius"]),
        float(cfg["distance_frac"]),
        int(cfg["min_cut_area"]),
    )
    if not pieces:
        return anchor.copy()
    max_remove = int(round(float(cfg["max_remove_frac"]) * max(1, int(anchor.sum()))))
    if max_remove <= 0:
        return anchor.copy()
    remove = np.zeros_like(anchor, dtype=bool)
    used = 0
    for piece in pieces:
        area = int(piece.sum())
        if used + area > max_remove:
            continue
        remove |= piece
        used += area
    out = anchor.copy()
    out[remove] = False
    return out


def mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in records[0] if key != "image"] if records else []
    return {key: float(np.mean([float(row[key]) for row in records])) for key in keys}


def evaluate_items(
    items: list[dict[str, Any]],
    cfg: dict[str, float],
    boundary_kernel: int,
    write_dir: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    for item in items:
        gt = item["gt"]
        anchor = item["anchor"]
        pred = conservative_edit(anchor, cfg)
        edited = pred != anchor
        if write_dir is not None:
            write_mask(write_dir / str(item["image"]), pred)
        rec = compute_metrics(pred, gt, boundary_kernel)
        rec = add_diag(rec, pred, gt, edited, int(cfg["gap_radius"]))
        rec["image"] = str(item["image"])
        records.append(rec)
    return mean_records(records), records


def cfgs(args: argparse.Namespace) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for gap_radius in parse_ints(args.gap_radii):
        for thin_radius in parse_ints(args.thin_radii):
            for distance_frac in parse_floats(args.distance_fracs):
                for max_remove_frac in parse_floats(args.max_remove_fracs):
                    for min_cut_area in parse_ints(args.min_cut_areas):
                        out.append(
                            {
                                "gap_radius": float(gap_radius),
                                "thin_radius": float(thin_radius),
                                "distance_frac": float(distance_frac),
                                "max_remove_frac": float(max_remove_frac),
                                "min_cut_area": float(min_cut_area),
                            }
                        )
    return out


def gate_pass(mean: dict[str, float], anchor: dict[str, float], args: argparse.Namespace) -> bool:
    return bool(
        mean["edited_frac"] >= args.min_edited_frac
        and mean["dice"] >= anchor["dice"] - args.dice_drop_tol
        and mean["recall"] >= anchor["recall"] - args.recall_drop_tol
        and mean["boundary_iou"] >= anchor["boundary_iou"] - args.boundary_drop_tol
        and mean["component_count_mae_proxy"] <= anchor["component_count_mae_proxy"] + args.component_mae_tol
        and mean["gap_fp_proxy"] <= anchor["gap_fp_proxy"] + args.gap_fp_tol
    )


def score(mean: dict[str, float], anchor: dict[str, float]) -> tuple[float, ...]:
    return (
        anchor["component_merge_proxy"] - mean["component_merge_proxy"],
        anchor["component_count_mae_proxy"] - mean["component_count_mae_proxy"],
        anchor["gap_fp_proxy"] - mean["gap_fp_proxy"],
        mean["boundary_iou"] - anchor["boundary_iou"],
        mean["dice"] - anchor["dice"],
        mean["edited_frac"],
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    write_progress(args.progress_json, {"run_id": "R204", "stage": "loading_val_items"})
    tune_items = load_items(
        args.tune_raw_root,
        args.tune_dataset,
        args.tune_split,
        mask_root(args.ablations_root, args.anchor_exp, args.tune_dataset, args.tune_split),
        args.max_val_images,
    )
    write_progress(
        args.progress_json,
        {
            "run_id": "R204",
            "stage": "evaluating_anchor",
            "num_val_items": len(tune_items),
            "elapsed_sec": round(time.time() - started, 3),
        },
    )
    anchor_cfg = {"gap_radius": 3.0, "thin_radius": 1.0, "distance_frac": 0.0, "max_remove_frac": 0.0, "min_cut_area": 999999.0}
    anchor_mean, anchor_records = evaluate_items(tune_items, anchor_cfg, args.boundary_kernel)
    grid = []
    best = None
    configs = cfgs(args)
    write_progress(
        args.progress_json,
        {
            "run_id": "R204",
            "stage": "grid_search",
            "num_val_items": len(tune_items),
            "num_configs": len(configs),
            "anchor_mean": anchor_mean,
            "elapsed_sec": round(time.time() - started, 3),
        },
    )
    for idx, cfg in enumerate(configs, start=1):
        cfg_started = time.time()
        mean, _ = evaluate_items(tune_items, cfg, args.boundary_kernel)
        passed = gate_pass(mean, anchor_mean, args)
        item = {"idx": idx, "cfg": cfg, "mean": mean, "passes_gate": passed}
        grid.append(item)
        if passed:
            item_score = score(mean, anchor_mean)
            if best is None or item_score > best["score"]:
                best = item | {"score": item_score}
        write_progress(
            args.progress_json,
            {
                "run_id": "R204",
                "stage": "grid_search",
                "idx": idx,
                "num_configs": len(configs),
                "last_cfg_sec": round(time.time() - cfg_started, 3),
                "elapsed_sec": round(time.time() - started, 3),
                "last_passes_gate": passed,
                "best_idx": best["idx"] if best else None,
                "last_mean": {
                    key: mean.get(key)
                    for key in [
                        "dice",
                        "recall",
                        "boundary_iou",
                        "component_count_mae_proxy",
                        "component_merge_proxy",
                        "gap_fp_proxy",
                        "edited_frac",
                    ]
                },
            },
        )
    val_summary = {
        "run_id": "R204",
        "status": "val_gate_passed" if best else "val_gate_failed",
        "selection_rule": "train/val only; clean-test-v2 application is allowed only after gate pass",
        "anchor_mean": anchor_mean,
        "anchor_per_image": anchor_records,
        "best": best,
        "grid": grid,
        "gate": {
            "dice_drop_tol": args.dice_drop_tol,
            "recall_drop_tol": args.recall_drop_tol,
            "boundary_drop_tol": args.boundary_drop_tol,
            "component_mae_tol": args.component_mae_tol,
            "gap_fp_tol": args.gap_fp_tol,
        },
    }
    args.val_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.val_summary_json.write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
    write_progress(
        args.progress_json,
        {
            "run_id": "R204",
            "stage": "val_complete",
            "status": val_summary["status"],
            "best_idx": best["idx"] if best else None,
            "elapsed_sec": round(time.time() - started, 3),
        },
    )

    result: dict[str, Any] = {
        "run_id": "R204",
        "status": "val_gate_failed",
        "val_summary_json": str(args.val_summary_json),
        "clean_test_applied": False,
    }
    if best is not None or args.allow_clean_test_on_gate_fail:
        write_progress(
            args.progress_json,
            {
                "run_id": "R204",
                "stage": "applying_clean_test_v2",
                "status": val_summary["status"],
                "elapsed_sec": round(time.time() - started, 3),
            },
        )
        apply_items = load_items(
            args.apply_raw_root,
            args.apply_dataset,
            args.apply_split,
            mask_root(args.ablations_root, args.apply_anchor_exp, args.apply_dataset, args.apply_split),
            args.max_apply_images,
        )
        out_dir = args.ablations_root / args.output_exp / args.apply_dataset / args.apply_split / "masks"
        cfg = best["cfg"] if best is not None else grid[0]["cfg"]
        mean, records = evaluate_items(apply_items, cfg, args.boundary_kernel, out_dir)
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(
            json.dumps({"run_id": "R204", "cfg": cfg, "mean": mean, "per_image": records}, indent=2),
            encoding="utf-8",
        )
        result.update(
            {
                "status": "clean_test_applied_after_val_gate" if best is not None else "clean_test_forced_after_gate_fail",
                "clean_test_applied": True,
                "cfg": cfg,
                "metrics_json": str(args.metrics_json),
                "output_mask_dir": str(out_dir),
                "mean": mean,
            }
        )
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_progress(
        args.progress_json,
        {
            "run_id": "R204",
            "stage": "done",
            "status": result["status"],
            "clean_test_applied": result["clean_test_applied"],
            "elapsed_sec": round(time.time() - started, 3),
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
