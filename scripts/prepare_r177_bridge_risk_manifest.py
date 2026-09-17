#!/usr/bin/env python3
"""Prepare train/val bridge-risk manifest for R177."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from evaluate_masks import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare R177 bridge-risk manifest.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--anchor-exp", default="r100_like_r025b_r097_patch_basic_trainval")
    parser.add_argument("--ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--output-dir", default="outputs/analysis/r177_bridge_risk_manifest")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--gap-radius", type=int, default=9)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--min-risk-pixels", type=int, default=128)
    parser.add_argument("--min-train-risk-images", type=int, default=50)
    parser.add_argument("--min-val-risk-images", type=int, default=10)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def boundary(mask: np.ndarray, kernel: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(pil.filter(ImageFilter.MaxFilter(kernel))) > 0
    eroded = np.asarray(pil.filter(ImageFilter.MinFilter(kernel))) > 0
    return np.logical_xor(dilated, eroded)


def component_count(mask: np.ndarray) -> int:
    _, n = ndimage.label(mask)
    return int(n)


def separation_band(gt: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((radius, radius), dtype=bool)
    return np.logical_and(ndimage.binary_dilation(gt, structure=structure), ~gt)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def split_records(args: argparse.Namespace, split: str) -> list[dict[str, float | str | int]]:
    label_dir = Path(args.raw_root) / args.dataset / f"{split}_labels"
    pred_dir = Path(args.ablations_root) / args.anchor_exp / args.dataset / split / "masks"
    if not label_dir.exists():
        raise FileNotFoundError(f"Label dir not found: {label_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Anchor mask dir not found: {pred_dir}")

    records: list[dict[str, float | str | int]] = []
    label_paths = sorted(label_dir.glob("*.png"))
    if args.max_images_per_split and args.max_images_per_split > 0:
        label_paths = label_paths[: args.max_images_per_split]
    for label_path in label_paths:
        pred_path = pred_dir / label_path.name
        if not pred_path.exists():
            records.append({"split": split, "image": label_path.name, "missing_anchor": 1})
            continue
        gt = read_mask(label_path)
        anchor = read_mask(pred_path)
        if anchor.shape != gt.shape:
            anchor = np.asarray(Image.fromarray(anchor.astype(np.uint8)).resize(gt.shape[::-1], Image.NEAREST)) > 0

        metrics = compute_metrics(anchor, gt, args.boundary_kernel)
        gt_components = component_count(gt)
        anchor_components = component_count(anchor)
        sep = separation_band(gt, args.gap_radius)
        fp = np.logical_and(anchor, ~gt)
        fn = np.logical_and(~anchor, gt)
        b_gt = boundary(gt, args.boundary_kernel)
        b_anchor = boundary(anchor, args.boundary_kernel)
        b_tp = float(np.logical_and(b_gt, b_anchor).sum())
        b_fp = float(np.logical_and(~b_gt, b_anchor).sum())
        b_fn = float(np.logical_and(b_gt, ~b_anchor).sum())
        boundary_f1 = (2.0 * b_tp / (2.0 * b_tp + b_fp + b_fn)) if (2.0 * b_tp + b_fp + b_fn) > 0 else 0.0

        sep_pixels = int(sep.sum())
        sep_fp_pixels = int(np.logical_and(anchor, sep).sum())
        false_bridge = int(anchor_components < gt_components)
        risk = np.logical_or(np.logical_and(anchor, sep), np.logical_and(fn, ndimage.binary_dilation(b_gt, iterations=2)))
        risk_pixels = int(risk.sum())
        bridge_risk = int(false_bridge and risk_pixels >= args.min_risk_pixels)

        records.append({
            "split": split,
            "image": label_path.name,
            "missing_anchor": 0,
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "boundary_iou": float(metrics["boundary_iou"]),
            "boundary_f1": float(boundary_f1),
            "gt_components": gt_components,
            "anchor_components": anchor_components,
            "component_delta": int(anchor_components - gt_components),
            "component_error": int(abs(anchor_components - gt_components)),
            "false_bridge": false_bridge,
            "sep_pixels": sep_pixels,
            "sep_fp_pixels": sep_fp_pixels,
            "sep_fp_rate": float(sep_fp_pixels / sep_pixels) if sep_pixels > 0 else 0.0,
            "fp_pixels": int(fp.sum()),
            "fn_pixels": int(fn.sum()),
            "risk_pixels": risk_pixels,
            "bridge_risk": bridge_risk,
        })
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_records: list[dict[str, float | str | int]] = []
    for split in splits:
        all_records.extend(split_records(args, split))

    csv_path = output_dir / "r177_bridge_risk_manifest.csv"
    fieldnames = sorted({key for rec in all_records for key in rec})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    summary_by_split = {}
    for split in splits:
        rows = [r for r in all_records if r["split"] == split and not int(r.get("missing_anchor", 0))]
        risk_rows = [r for r in rows if int(r.get("bridge_risk", 0))]
        summary_by_split[split] = {
            "num_records": len(rows),
            "missing_anchor": sum(1 for r in all_records if r["split"] == split and int(r.get("missing_anchor", 0))),
            "risk_images": len(risk_rows),
            "false_bridge_images": sum(1 for r in rows if int(r.get("false_bridge", 0))),
            "dice": summarize([float(r["dice"]) for r in rows]),
            "boundary_iou": summarize([float(r["boundary_iou"]) for r in rows]),
            "boundary_f1": summarize([float(r["boundary_f1"]) for r in rows]),
            "sep_fp_rate": summarize([float(r["sep_fp_rate"]) for r in rows]),
            "component_error": summarize([float(r["component_error"]) for r in rows]),
            "risk_pixels": summarize([float(r["risk_pixels"]) for r in rows]),
            "top_risk": sorted(
                [
                    {
                        "image": str(r["image"]),
                        "risk_pixels": int(r["risk_pixels"]),
                        "false_bridge": int(r["false_bridge"]),
                        "component_delta": int(r["component_delta"]),
                        "dice": float(r["dice"]),
                        "boundary_iou": float(r["boundary_iou"]),
                    }
                    for r in rows
                ],
                key=lambda x: (x["false_bridge"], x["risk_pixels"], -x["dice"]),
                reverse=True,
            )[:30],
        }

    train_ok = summary_by_split.get("train", {}).get("risk_images", 0) >= args.min_train_risk_images
    val_ok = summary_by_split.get("val", {}).get("risk_images", 0) >= args.min_val_risk_images
    missing_ok = all(v.get("missing_anchor", 0) == 0 for v in summary_by_split.values())
    summary = {
        "run_id": "R177-prep",
        "dataset": args.dataset,
        "anchor_exp": args.anchor_exp,
        "gate": {
            "pass": bool(train_ok and val_ok and missing_ok),
            "train_ok": bool(train_ok),
            "val_ok": bool(val_ok),
            "missing_ok": bool(missing_ok),
            "min_train_risk_images": args.min_train_risk_images,
            "min_val_risk_images": args.min_val_risk_images,
            "min_risk_pixels": args.min_risk_pixels,
        },
        "summary_by_split": summary_by_split,
        "manifest_csv": str(csv_path),
    }
    summary_path = output_dir / "r177_bridge_risk_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": str(summary_path),
        "manifest": str(csv_path),
        "gate": summary["gate"],
        "risk_images": {k: v["risk_images"] for k, v in summary_by_split.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
