#!/usr/bin/env python3
"""R148 rank diagnostic for extremely low Mask2Former mask probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_masks import compute_metrics
from r144_mask2former_hf_smoke import add_hf_deps, load_split, mean_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose whether tiny mask probabilities have useful spatial rank.")
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
    parser.add_argument("--thresholds", default="0.000001,0.000003,0.000005,0.0000075,0.00001,0.000015,0.00002,0.00005")
    parser.add_argument("--topk-factors", default="0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--output-json", default="outputs/analysis/r148_mask2former_rank_diagnostic.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


@torch.no_grad()
def union_probability(model: torch.nn.Module, sample: dict[str, object], device: str) -> np.ndarray:
    output = model(pixel_values=sample["pixel_values"].unsqueeze(0).to(device))
    logits = output.masks_queries_logits[0]
    resized = F.interpolate(logits[:, None], size=sample["original_shape"], mode="bilinear", align_corners=False)[:, 0]
    return torch.sigmoid(resized).max(dim=0).values.detach().cpu().numpy()


def topk_mask(prob: np.ndarray, k: int) -> np.ndarray:
    flat = prob.reshape(-1)
    k = max(0, min(int(k), flat.size))
    if k == 0:
        return np.zeros_like(prob, dtype=bool)
    index = np.argpartition(flat, flat.size - k)[flat.size - k :]
    mask = np.zeros(flat.size, dtype=bool)
    mask[index] = True
    return mask.reshape(prob.shape)


def analyze_sample(model: torch.nn.Module, sample: dict[str, object], thresholds: list[float], topk_factors: list[float], device: str) -> dict[str, object]:
    prob = union_probability(model, sample, device)
    gt = np.asarray(sample["gt_original"]).astype(bool)
    gt_pixels = int(gt.sum())
    threshold_records = []
    for threshold in thresholds:
        pred = prob >= threshold
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec.update({"threshold": threshold, "selected_fraction": float(pred.mean())})
        threshold_records.append(rec)
    topk_records = []
    for factor in topk_factors:
        pred = topk_mask(prob, int(round(gt_pixels * factor)))
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec.update({"factor": factor, "selected_fraction": float(pred.mean())})
        topk_records.append(rec)
    return {
        "image": str(sample["name"]),
        "gt_fraction": float(gt.mean()),
        "prob_max": float(prob.max()),
        "prob_mean": float(prob.mean()),
        "prob_p99": float(np.quantile(prob, 0.99)),
        "threshold_records": threshold_records,
        "topk_records": topk_records,
    }


def summarize(records: list[dict[str, object]], key: str, values: list[float]) -> list[dict[str, object]]:
    out = []
    records_key = "threshold_records" if key == "threshold" else "topk_records"
    for value in values:
        selected = []
        for record in records:
            for sub in record[records_key]:
                if abs(float(sub[key]) - value) < 1e-12:
                    row = dict(sub)
                    row["image"] = record["image"]
                    selected.append(row)
                    break
        out.append({"value": value, "mean": mean_metrics(selected), "num_evaluated": len(selected)})
    return out


def analyze_split(name: str, samples: list[dict[str, object]], model: torch.nn.Module, thresholds: list[float], topk_factors: list[float], device: str) -> dict[str, object]:
    records = [analyze_sample(model, sample, thresholds, topk_factors, device) for sample in samples]
    return {
        "split": name,
        "num_evaluated": len(records),
        "threshold_summary": summarize(records, "threshold", thresholds),
        "topk_gt_fraction_summary": summarize(records, "factor", topk_factors),
        "per_image": records,
    }


def main() -> None:
    args = parse_args()
    add_hf_deps(args.hf_deps)
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    thresholds = parse_float_list(args.thresholds)
    topk_factors = parse_float_list(args.topk_factors)
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
    checkpoint = torch.load(Path(args.checkpoint), map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    report = {
        "status": "diagnostic_complete",
        "run_id": "R148",
        "checkpoint": args.checkpoint,
        "clean_test_policy": "clean-test-v2 used only for diagnostic inference; top-k GT-fraction is an oracle diagnostic, not a deployable readout",
        "splits": {
            "train": analyze_split("train", load_split(Path(args.raw_root), args.dataset, "train", args.img_size, args.limit_train), model, thresholds, topk_factors, args.device),
            "val": analyze_split("val", load_split(Path(args.raw_root), args.dataset, "val", args.img_size, args.limit_val), model, thresholds, topk_factors, args.device),
            "clean_test_v2": analyze_split("clean_test_v2", load_split(Path(args.eval_raw_root), args.eval_dataset, "test", args.img_size, args.limit_eval), model, thresholds, topk_factors, args.device),
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {}
    for split, data in report["splits"].items():
        compact[split] = {
            "best_threshold_dice": max(row["mean"].get("dice", 0.0) for row in data["threshold_summary"]),
            "best_topk_oracle_dice": max(row["mean"].get("dice", 0.0) for row in data["topk_gt_fraction_summary"]),
        }
    print(json.dumps({"output_json": str(out), "summary": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
