#!/usr/bin/env python3
"""Apply an instance-separation veto to an existing anchor mask.

This is a low-cost diagnostic after R068: use the learned separation/core heads
as an image-derived bridge veto on a strong YOLO+SAM/refiner mask, selecting
trim parameters on a validation anchor and applying the fixed rule to
clean-test-v2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluate_masks import compute_metrics


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply R068 separation veto to anchor masks.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--train-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--val-dataset", default=None)
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--val-anchor-mask-dir", required=True)
    parser.add_argument("--eval-anchor-mask-dir", required=True)
    parser.add_argument("--checkpoint", default="outputs/context_segmenter/r068_instance_separation_segmenter/best.pt")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--sep-thresholds", default="0.45,0.55,0.65,0.75,0.85")
    parser.add_argument("--core-thresholds", default="0.05,0.15,0.25,0.35")
    parser.add_argument("--dilate-radii", default="0,1")
    parser.add_argument("--max-remove-fracs", default="0.005,0.010,0.020,0.035")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--fast-grid-boundary-kernel", type=int, default=0)
    parser.add_argument("--output-exp", default="r069_r038_r068_separation_veto")
    parser.add_argument("--pred-root", default="outputs/ablations_variants")
    parser.add_argument("--metrics-json", default="outputs/analysis/r069_r038_r068_separation_veto_clean_test_v2_metrics.json")
    parser.add_argument("--control-metrics-json", default="outputs/analysis/r069_r038_r068_separation_veto_val_proxy_metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_r068_module():
    script = Path(__file__).with_name("train_instance_separation_segmenter.py")
    spec = importlib.util.spec_from_file_location("train_instance_separation_segmenter", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def find_image_path(root: Path, dataset: str, split: str, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        p = root / dataset / split / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"image not found for {stem}: {root}/{dataset}/{split}")


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image.astype(np.float32) / 255.0


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) > 0)


def add_structure_metrics(record: dict[str, float], pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_count = component_count(pred)
    gt_count = component_count(gt)
    record["pred_component_count"] = float(pred_count)
    record["gt_component_count"] = float(gt_count)
    record["component_count_error"] = float(abs(pred_count - gt_count))
    record["false_bridge_flag"] = float(pred_count < gt_count and pred.sum() >= gt.sum() * 0.90)
    return record


def fast_compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    tn = float(np.logical_and(~pred, ~gt).sum())
    return {
        "dice": float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 1.0,
        "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 1.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0,
        "boundary_iou": 0.0,
    }


def mean_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    keys = [k for k in records[0] if k != "image"] if records else []
    return {key: float(np.mean([r[key] for r in records])) for key in keys}


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    module = load_r068_module()
    state = torch.load(args.checkpoint, map_location=args.device)
    ckpt_args = state.get("args", {})
    base = int(ckpt_args.get("base_channels", args.base_channels))
    model = module.InstanceSeparationUNet(base).to(args.device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_sep_core(model: torch.nn.Module, image: np.ndarray, original_shape: tuple[int, int], img_size: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(image.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized[None, None].astype(np.float32)).to(device)
    out = model(tensor)
    sep = torch.sigmoid(out["sep"])[0, 0].detach().cpu().numpy()
    core = torch.sigmoid(out["core"])[0, 0].detach().cpu().numpy()
    sep = cv2.resize(sep, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    core = cv2.resize(core, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    return sep, core


def trim_anchor(anchor: np.ndarray, sep: np.ndarray, core: np.ndarray, rule: dict[str, float | int]) -> np.ndarray:
    risky = (sep >= float(rule["sep_threshold"])) & (core < float(rule["core_threshold"])) & anchor
    radius = int(rule["dilate_radius"])
    if radius > 0:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        risky = cv2.dilate(risky.astype(np.uint8), kernel, iterations=1).astype(bool) & anchor & (core < float(rule["core_threshold"]))
    max_remove = int(round(float(rule["max_remove_frac"]) * float(anchor.sum())))
    if max_remove <= 0 or int(risky.sum()) == 0:
        return anchor.copy()
    candidates = np.argwhere(risky)
    scores = sep[risky] - 0.50 * core[risky]
    order = np.argsort(-scores)
    keep_count = min(max_remove, len(order))
    out = anchor.copy()
    chosen = candidates[order[:keep_count]]
    out[chosen[:, 0], chosen[:, 1]] = False
    return out


def collect_items(
    model: torch.nn.Module,
    raw_root: Path,
    dataset: str,
    split: str,
    anchor_mask_dir: Path,
    img_size: int,
    device: str,
) -> list[dict[str, object]]:
    label_dir = raw_root / dataset / f"{split}_labels"
    items: list[dict[str, object]] = []
    for label_path in tqdm(sorted(label_dir.glob("*.png")), desc=f"collect/{dataset}/{split}"):
        anchor_path = anchor_mask_dir / label_path.name
        if not anchor_path.exists():
            continue
        gt = read_mask(label_path)
        anchor = read_mask(anchor_path)
        if anchor.shape != gt.shape:
            anchor = cv2.resize(anchor.astype(np.float32), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5
        image = read_gray(find_image_path(raw_root, dataset, split, label_path.stem))
        sep, core = predict_sep_core(model, image, gt.shape, img_size, device)
        items.append({"name": label_path.name, "gt": gt, "anchor": anchor, "sep": sep, "core": core})
    return items


def evaluate_items(items: list[dict[str, object]], rule: dict[str, float | int], boundary_kernel: int, out_dir: Path | None = None) -> tuple[dict[str, float], list[dict[str, float]]]:
    records: list[dict[str, float]] = []
    for item in items:
        pred = trim_anchor(item["anchor"], item["sep"], item["core"], rule)  # type: ignore[arg-type]
        gt = item["gt"]  # type: ignore[assignment]
        rec = fast_compute_metrics(pred, gt) if boundary_kernel <= 0 else compute_metrics(pred, gt, boundary_kernel=boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt)
        rec["image"] = str(item["name"])
        records.append(rec)
        if out_dir is not None:
            write_mask(out_dir / str(item["name"]), pred)
    return mean_metrics(records), records


def anchor_summary(items: list[dict[str, object]], boundary_kernel: int) -> dict[str, float]:
    records = []
    for item in items:
        pred = item["anchor"]  # type: ignore[assignment]
        gt = item["gt"]  # type: ignore[assignment]
        rec = compute_metrics(pred, gt, boundary_kernel=boundary_kernel)
        rec = add_structure_metrics(rec, pred, gt)
        records.append(rec)
    return mean_metrics(records)


def main() -> None:
    args = parse_args()
    model = load_model(args)
    val_dataset = args.val_dataset or args.train_dataset
    val_items = collect_items(model, Path(args.raw_root), val_dataset, args.val_split, Path(args.val_anchor_mask_dir), args.img_size, args.device)
    eval_items = collect_items(model, Path(args.eval_raw_root), args.eval_dataset, args.eval_split, Path(args.eval_anchor_mask_dir), args.img_size, args.device)
    if not val_items or not eval_items:
        raise RuntimeError(f"missing items: val={len(val_items)} eval={len(eval_items)}")

    rules = []
    for sep_thr in parse_floats(args.sep_thresholds):
        for core_thr in parse_floats(args.core_thresholds):
            for radius in parse_ints(args.dilate_radii):
                for max_frac in parse_floats(args.max_remove_fracs):
                    rules.append({
                        "sep_threshold": float(sep_thr),
                        "core_threshold": float(core_thr),
                        "dilate_radius": int(radius),
                        "max_remove_frac": float(max_frac),
                    })
    val_results = []
    for rule in tqdm(rules, desc="val/grid"):
        mean, _ = evaluate_items(val_items, rule, args.fast_grid_boundary_kernel)
        val_results.append({"rule": rule, "mean": mean})
    val_anchor = anchor_summary(val_items, args.boundary_kernel)
    eval_anchor = anchor_summary(eval_items, args.boundary_kernel)
    best = max(val_results, key=lambda x: x["mean"]["dice"])

    pred_dir = Path(args.pred_root) / args.output_exp / args.eval_dataset / args.eval_split / "masks"
    eval_mean, eval_records = evaluate_items(eval_items, best["rule"], args.boundary_kernel, pred_dir)
    val_mean, val_records = evaluate_items(val_items, best["rule"], args.boundary_kernel)

    val_summary = {
        "dataset": val_dataset,
        "split": args.val_split,
        "anchor_mask_dir": args.val_anchor_mask_dir,
        "anchor_mean": val_anchor,
        "selected_rule": best["rule"],
        "selected_val_mean": val_mean,
        "grid": val_results,
        "per_image": val_records,
    }
    eval_summary = {
        "dataset": args.eval_dataset,
        "split": args.eval_split,
        "anchor_mask_dir": args.eval_anchor_mask_dir,
        "anchor_mean": eval_anchor,
        "selected_rule": best["rule"],
        "mean": eval_mean,
        "per_image": eval_records,
    }
    Path(args.control_metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.control_metrics_json).write_text(json.dumps(val_summary, indent=2), encoding="utf-8")
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_rule": best["rule"],
        "val_anchor": val_anchor,
        "val_selected": val_mean,
        "eval_anchor": eval_anchor,
        "eval_selected": eval_mean,
        "metrics_json": args.metrics_json,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
