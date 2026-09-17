#!/usr/bin/env python3
"""R149 tiny overfit rescue for the HuggingFace Mask2Former path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_masks import compute_metrics
from r144_mask2former_hf_smoke import add_hf_deps, load_split, mean_metrics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Force a tiny Mask2Former overfit before any full run.")
    parser.add_argument("--hf-deps", default=".tmp/r144_hf_deps")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--eval-raw-root", default="data/raw_variants")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--eval-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--limit-train", type=int, default=1)
    parser.add_argument("--limit-val", type=int, default=2)
    parser.add_argument("--limit-eval", type=int, default=2)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--num-queries", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-mode", choices=["instance", "union"], default="union")
    parser.add_argument("--class-weight", type=float, default=2.0)
    parser.add_argument("--mask-weight", type=float, default=5.0)
    parser.add_argument("--dice-weight", type=float, default=5.0)
    parser.add_argument("--no-object-weight", type=float, default=0.1)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--output-json", default="outputs/analysis/r149_mask2former_overfit_rescue_metrics.json")
    parser.add_argument("--checkpoint", default="outputs/r149_mask2former_overfit_rescue/model.pt")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("empty threshold list")
    return values


def target_tensors(sample: dict[str, object], target_mode: str, device: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if target_mode == "instance":
        return [sample["mask_labels"].to(device)], [sample["class_labels"].to(device)]
    gt_small = sample["mask_labels"].sum(dim=0).clamp(max=1.0).to(device)
    return [gt_small[None]], [torch.zeros(1, dtype=torch.long, device=device)]


@torch.no_grad()
def union_prob(model: torch.nn.Module, sample: dict[str, object], device: str) -> tuple[np.ndarray, dict[str, float]]:
    output = model(pixel_values=sample["pixel_values"].unsqueeze(0).to(device))
    logits = F.interpolate(
        output.masks_queries_logits[0][:, None],
        size=sample["original_shape"],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    probs = torch.sigmoid(logits)
    scores = torch.softmax(output.class_queries_logits[0], dim=-1)[:, 0]
    keep = scores > 0.05
    selected = probs[keep] if keep.any() else probs
    union = selected.max(dim=0).values.detach().cpu().numpy()
    return union, {
        "union_max": float(union.max()),
        "union_mean": float(union.mean()),
        "foreground_score_max": float(scores.max().detach().cpu()),
        "kept_queries_005": float(keep.sum().detach().cpu()),
    }


def best_threshold_metrics(prob: np.ndarray, gt: np.ndarray, thresholds: list[float]) -> dict[str, object]:
    records = []
    for threshold in thresholds:
        pred = prob >= threshold
        rec = compute_metrics(pred, gt, boundary_kernel=3)
        rec.update({"threshold": threshold, "selected_fraction": float(pred.mean())})
        records.append(rec)
    return max(records, key=lambda item: item.get("dice", 0.0)) | {"all_thresholds": records}


@torch.no_grad()
def evaluate_split(model: torch.nn.Module, samples: list[dict[str, object]], thresholds: list[float], device: str) -> dict[str, object]:
    per_image = []
    for sample in samples:
        prob, stats = union_prob(model, sample, device)
        rec = best_threshold_metrics(prob, sample["gt_original"], thresholds)
        rec["image"] = str(sample["name"])
        rec["probability"] = stats
        per_image.append(rec)
    compact_records = [{k: v for k, v in row.items() if k != "all_thresholds" and k != "probability"} for row in per_image]
    return {"num_evaluated": len(per_image), "mean": mean_metrics(compact_records), "per_image": per_image}


def main() -> None:
    args = parse_args()
    add_hf_deps(args.hf_deps)
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    set_seed(args.seed)
    thresholds = parse_float_list(args.thresholds)
    train_samples = load_split(Path(args.raw_root), args.dataset, "train", args.img_size, args.limit_train)
    val_samples = load_split(Path(args.raw_root), args.dataset, "val", args.img_size, args.limit_val)
    clean_samples = load_split(Path(args.eval_raw_root), args.eval_dataset, "test", args.img_size, args.limit_eval)

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
        class_weight=args.class_weight,
        mask_weight=args.mask_weight,
        dice_weight=args.dice_weight,
        no_object_weight=args.no_object_weight,
    )
    model = Mask2FormerForUniversalSegmentation(config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trace = []
    best_train_dice = -1.0
    best_state = None

    for step in range(1, args.steps + 1):
        model.train()
        sample = train_samples[(step - 1) % len(train_samples)]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(args.device)
        mask_labels, class_labels = target_tensors(sample, args.target_mode, args.device)
        opt.zero_grad(set_to_none=True)
        output = model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)
        loss = output.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            model.eval()
            train_eval = evaluate_split(model, train_samples, thresholds, args.device)
            train_dice = float(train_eval["mean"].get("dice", 0.0))
            if train_dice > best_train_dice:
                best_train_dice = train_dice
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "train_best_threshold_dice": train_dice,
                "train_union_max_mean": float(np.mean([r["probability"]["union_max"] for r in train_eval["per_image"]])),
                "train_eval": train_eval,
            }
            trace.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "train_eval"}), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    final_train = evaluate_split(model, train_samples, thresholds, args.device)
    final_val = evaluate_split(model, val_samples, thresholds, args.device)
    final_clean = evaluate_split(model, clean_samples, thresholds, args.device)
    status = "overfit_pass" if float(final_train["mean"].get("dice", 0.0)) >= 0.85 else "overfit_failed"
    report = {
        "status": status,
        "run_id": "R149",
        "architecture": "hf_mask2former",
        "target_mode": args.target_mode,
        "thresholds": thresholds,
        "config": {
            "num_queries": args.num_queries,
            "hidden_dim": args.hidden_dim,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "steps": args.steps,
            "lr": args.lr,
            "img_size": args.img_size,
            "class_weight": args.class_weight,
            "mask_weight": args.mask_weight,
            "dice_weight": args.dice_weight,
            "no_object_weight": args.no_object_weight,
        },
        "train_images": [str(s["name"]) for s in train_samples],
        "val_images": [str(s["name"]) for s in val_samples],
        "clean_test_v2_images": [str(s["name"]) for s in clean_samples],
        "clean_test_policy": "clean-test-v2 used only for post-overfit diagnostic inference, never for training or success tuning",
        "trace": trace,
        "train_eval": final_train,
        "val_eval": final_val,
        "clean_test_v2_eval": final_clean,
        "decision": "full run allowed only if status is overfit_pass",
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "trace": trace, "status": status}, checkpoint)
    print(json.dumps({"output_json": str(output_json), "status": status, "train": final_train["mean"], "val": final_val["mean"], "clean": final_clean["mean"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
