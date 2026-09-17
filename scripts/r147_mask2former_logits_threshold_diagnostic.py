#!/usr/bin/env python3
"""R147 diagnostic for Mask2Former mask-logit scale and threshold behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_masks import compute_metrics
from r144_mask2former_hf_smoke import add_hf_deps, load_split, mean_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose R146 Mask2Former mask logits.")
    parser.add_argument("--hf-deps", default=".tmp/r144_hf_deps")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--limit-train", type=int, default=16)
    parser.add_argument("--limit-val", type=int, default=4)
    parser.add_argument("--limit-eval", type=int, default=4)
    parser.add_argument("--num-queries", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--checkpoint", default="outputs/r146_mask2former_hf_maskonly_readout/model.pt")
    parser.add_argument(
        "--thresholds",
        default="0.001,0.003,0.005,0.01,0.02,0.05,0.1,0.2,0.35",
        help="Comma-separated mask probability thresholds.",
    )
    parser.add_argument("--output-json", default="outputs/analysis/r147_mask2former_logits_threshold_diagnostic.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_thresholds(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one threshold is required")
    return values


def quantiles(array: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 0.95, 0.99)) -> dict[str, float]:
    if array.size == 0:
        return {f"p{int(q * 100):02d}": 0.0 for q in qs}
    return {f"p{int(q * 100):02d}": float(np.quantile(array, q)) for q in qs}


def metric_record(pred: np.ndarray, gt: np.ndarray, image: str, selected_fraction: float) -> dict[str, float | str]:
    rec = compute_metrics(pred, gt, boundary_kernel=3)
    rec["image"] = image
    rec["selected_fraction"] = float(selected_fraction)
    return rec


@torch.no_grad()
def sample_diagnostic(model: torch.nn.Module, sample: dict[str, object], thresholds: list[float], device: str) -> dict[str, object]:
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    output = model(pixel_values=pixel_values)
    mask_logits = output.masks_queries_logits[0]
    class_logits = output.class_queries_logits[0]
    out_shape = sample["original_shape"]

    resized_logits = F.interpolate(
        mask_logits[:, None],
        size=out_shape,
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    probs = torch.sigmoid(resized_logits)
    class_probs = torch.softmax(class_logits, dim=-1)
    foreground_scores = class_probs[:, 0]
    keep = foreground_scores > 0.05

    union_all = probs.max(dim=0).values.detach().cpu().numpy()
    if keep.any():
        union_class_gated = probs[keep].max(dim=0).values.detach().cpu().numpy()
    else:
        union_class_gated = probs.max(dim=0).values.detach().cpu().numpy()
    gt = sample["gt_original"]
    image = str(sample["name"])

    threshold_metrics: dict[str, dict[str, object]] = {}
    for mode, union_prob in (("mask_only", union_all), ("class_gated_005", union_class_gated)):
        mode_records = []
        for threshold in thresholds:
            pred = union_prob >= threshold
            mode_records.append(metric_record(pred, gt, image, float(pred.mean())) | {"threshold": threshold})
        threshold_metrics[mode] = {"per_threshold": mode_records}

    foreground_np = foreground_scores.detach().cpu().numpy()
    mask_logit_np = resized_logits.detach().cpu().numpy()
    union_np = union_all
    return {
        "image": image,
        "foreground_pixels_gt": int(np.asarray(gt).sum()),
        "gt_fraction": float(np.asarray(gt).mean()),
        "num_queries": int(probs.shape[0]),
        "kept_queries_class_gated_005": int(keep.sum().detach().cpu()),
        "foreground_score": {
            "max": float(foreground_np.max()),
            "mean": float(foreground_np.mean()),
            **quantiles(foreground_np),
        },
        "mask_logit": {
            "max": float(mask_logit_np.max()),
            "mean": float(mask_logit_np.mean()),
            "min": float(mask_logit_np.min()),
            **quantiles(mask_logit_np.reshape(-1)),
        },
        "union_probability": {
            "max": float(union_np.max()),
            "mean": float(union_np.mean()),
            **quantiles(union_np.reshape(-1)),
        },
        "threshold_metrics": threshold_metrics,
    }


def summarize_thresholds(per_sample: list[dict[str, object]], thresholds: list[float], mode: str) -> list[dict[str, object]]:
    rows = []
    for threshold in thresholds:
        records = []
        for sample in per_sample:
            for rec in sample["threshold_metrics"][mode]["per_threshold"]:
                if abs(float(rec["threshold"]) - threshold) < 1e-12:
                    records.append(rec)
                    break
        mean = mean_metrics(records)
        rows.append(
            {
                "threshold": threshold,
                "mean": mean,
                "num_evaluated": len(records),
                "mean_selected_fraction": float(np.mean([float(r["selected_fraction"]) for r in records])) if records else 0.0,
            }
        )
    return rows


def summarize_split(name: str, samples: list[dict[str, object]], model: torch.nn.Module, thresholds: list[float], device: str) -> dict[str, object]:
    per_sample = [sample_diagnostic(model, sample, thresholds, device) for sample in samples]
    return {
        "split": name,
        "num_evaluated": len(per_sample),
        "probability_summary": {
            "mean_union_max": float(np.mean([s["union_probability"]["max"] for s in per_sample])) if per_sample else 0.0,
            "max_union_max": float(np.max([s["union_probability"]["max"] for s in per_sample])) if per_sample else 0.0,
            "mean_foreground_score_max": float(np.mean([s["foreground_score"]["max"] for s in per_sample])) if per_sample else 0.0,
            "mean_kept_queries_class_gated_005": float(np.mean([s["kept_queries_class_gated_005"] for s in per_sample])) if per_sample else 0.0,
        },
        "threshold_summary": {
            "mask_only": summarize_thresholds(per_sample, thresholds, "mask_only"),
            "class_gated_005": summarize_thresholds(per_sample, thresholds, "class_gated_005"),
        },
        "per_image": per_sample,
    }


def main() -> None:
    args = parse_args()
    add_hf_deps(args.hf_deps)
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    thresholds = parse_thresholds(args.thresholds)
    config = Mask2FormerConfig(
        num_labels=1,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feature_size=args.hidden_dim,
        mask_feature_size=args.hidden_dim,
        fpn_feature_size=args.hidden_dim,
        use_auxiliary_loss=True,
    )
    model = Mask2FormerForUniversalSegmentation(config).to(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    train_samples = load_split(Path(args.raw_root), args.dataset, "train", args.img_size, args.limit_train)
    val_samples = load_split(Path(args.raw_root), args.dataset, "val", args.img_size, args.limit_val)
    clean_samples = load_split(Path(args.eval_raw_root), args.eval_dataset, "test", args.img_size, args.limit_eval)

    report = {
        "status": "diagnostic_complete",
        "run_id": "R147",
        "checkpoint": str(checkpoint_path),
        "thresholds": thresholds,
        "architecture": "hf_mask2former",
        "config": {
            "num_queries": args.num_queries,
            "hidden_dim": args.hidden_dim,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
        },
        "clean_test_policy": "clean-test-v2 used only for diagnostic inference; no training or success threshold tuning",
        "train_trace_tail": checkpoint.get("train_trace", [])[-5:],
        "splits": {
            "train": summarize_split("train", train_samples, model, thresholds, args.device),
            "val": summarize_split("val", val_samples, model, thresholds, args.device),
            "clean_test_v2": summarize_split("clean_test_v2", clean_samples, model, thresholds, args.device),
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        split: {
            "probability_summary": data["probability_summary"],
            "best_mask_only_dice": max(row["mean"].get("dice", 0.0) for row in data["threshold_summary"]["mask_only"]),
        }
        for split, data in report["splits"].items()
    }
    print(json.dumps({"output_json": str(output_json), "summary": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
