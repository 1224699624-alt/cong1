#!/usr/bin/env python3
"""Evaluate R289 relation-head gap masks with RAM-W600 paper-aligned metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_aligned_segmentation_metrics import compute_paper_metrics
from train_r289_tsrs_pair_gap_head import (
    FrozenNnUNetPairHead,
    PairCropDataset,
    relation_targets,
)


def finite_mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def classification_metrics(records: list[dict[str, object]]) -> dict[str, float | None]:
    tp = sum(bool(row["target_present"]) and bool(row["prediction_present"]) for row in records)
    tn = sum(not bool(row["target_present"]) and not bool(row["prediction_present"]) for row in records)
    fp = sum(not bool(row["target_present"]) and bool(row["prediction_present"]) for row in records)
    fn = sum(bool(row["target_present"]) and not bool(row["prediction_present"]) for row in records)

    def divide(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    sensitivity = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    precision = divide(tp, tp + fp)
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    accuracy = divide(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (None if sensitivity is None or specificity is None
                         else (sensitivity + specificity) / 2.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--gap-radius", type=int, default=8)
    parser.add_argument("--oracle-roi-clip", action="store_true")
    args = parser.parse_args()
    joined = " ".join(map(str, (args.dataset_root, args.output))).lower()
    if args.dataset_root.name != "TSRS_RSNA-Epiphysis" or "clean-test" in joined or "articular" in joined:
        raise RuntimeError("Paper-aligned R289 evaluation is original-val Epiphysis only")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = [tuple(map(int, pair)) for pair in manifest["pairs"]]
    dataset = PairCropDataset(
        args.dataset_root, "val", pairs, args.crop_size, max_pairs_per_val_case=0
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenNnUNetPairHead(args.backbone_checkpoint, len(pairs), device).to(device)
    relation = torch.load(args.relation_checkpoint, map_location=device, weights_only=False)
    if [tuple(pair) for pair in relation["pairs"]] != pairs:
        raise RuntimeError("Pair list differs between manifest and relation checkpoint")
    model.relation_head.load_state_dict(relation["relation_head"])
    model.half_projection.load_state_dict(relation["half_projection"])
    model.eval()

    records: list[dict[str, object]] = []
    by_pair: dict[int, list[dict[str, object]]] = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            instance = batch["instance"].to(device)
            focus = int(batch["focus_pair"].item())
            logits = model(image)
            _, roi, gap = relation_targets(instance, pairs, args.gap_radius)
            probability = torch.softmax(logits.float(), dim=2)[0, focus, 0]
            prediction = probability >= 0.5
            if args.oracle_roi_clip:
                prediction = prediction & roi[0, focus]
            target = gap[0, focus] & roi[0, focus]
            target_present = bool(target.any())
            prediction_present = bool(prediction.any())
            metrics = (compute_paper_metrics(prediction.cpu(), target.cpu(), nsd_tolerance_px=2.0)
                       if target_present else {
                           "dsc": None, "nsd_2px": None, "voe": None,
                           "msd_px": None, "ravd": None, "msd_failed": None,
                       })
            row: dict[str, object] = {
                "case": str(batch["case"][0]),
                "pair_index": focus,
                "label_i": pairs[focus][0],
                "label_j": pairs[focus][1],
                "target_present": target_present,
                "prediction_present": prediction_present,
                **metrics,
            }
            records.append(row)
            by_pair[focus].append(row)

    metric_names = ("dsc", "nsd_2px", "voe", "msd_px", "ravd", "msd_failed")
    valid_gap_records = [row for row in records if bool(row["target_present"])]
    overall = {name: finite_mean([row[name] for row in valid_gap_records]) for name in metric_names}
    presence = classification_metrics(records)
    per_pair = []
    for pair_index, pair in enumerate(pairs):
        rows = by_pair.get(pair_index, [])
        per_pair.append({
            "pair_index": pair_index,
            "label_i": pair[0],
            "label_j": pair[1],
            "num_crops": len(rows),
            "num_valid_gap_crops": sum(bool(row["target_present"]) for row in rows),
            **{name: finite_mean([row[name] for row in rows if bool(row["target_present"])])
               for name in metric_names},
            **{f"presence_{name}": value for name, value in classification_metrics(rows).items()
               if name != "confusion_matrix_tn_fp_fn_tp"},
        })
    payload = {
        "protocol": "RAM-W600 paper-aligned MONAI 1.4.0 metrics on TSRS gap ROI",
        "dataset": "TSRS_RSNA-Epiphysis",
        "split": "original-val",
        "num_cases": len({str(row["case"]) for row in records}),
        "num_pair_crops": len(records),
        "num_valid_gap_crops": len(valid_gap_records),
        "input_scale": f"native 512x512 crops; NSD tolerance={2}px",
        "threshold": 0.5,
        "threshold_search_used": False,
        "prediction_scope": ("GT local ROI (oracle diagnostic only)"
                             if args.oracle_roi_clip else "full 512x512 pair crop"),
        "gt_used_to_clip_prediction": args.oracle_roi_clip,
        "clean_test_v2_used": False,
        "gap_segmentation_metrics": overall,
        "gap_presence_classification": presence,
        "per_pair": per_pair,
        "legacy_precision_recall_are_diagnostic_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".per_pair.csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_pair[0]))
        writer.writeheader()
        writer.writerows(per_pair)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
