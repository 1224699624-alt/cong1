#!/usr/bin/env python3
"""Audit reannotated Epiphysis variant for safe train/val-only use."""

from __future__ import annotations

import json
from pathlib import Path


def stems(path: Path) -> set[str]:
    return {p.stem for p in path.glob("*") if p.is_file()}


def split_counts(root: Path) -> dict[str, dict[str, int]]:
    out = {}
    for split in ["train", "val", "test"]:
        out[split] = {
            "images": len(list((root / split).glob("*"))),
            "labels": len(list((root / f"{split}_labels").glob("*.png"))),
            "paired": len(stems(root / split) & stems(root / f"{split}_labels")),
            "image_without_label": len(stems(root / split) - stems(root / f"{split}_labels")),
            "label_without_image": len(stems(root / f"{split}_labels") - stems(root / split)),
        }
    return out


def main() -> None:
    original = Path("data/raw/TSRS_RSNA-Epiphysis")
    reann = Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1")
    reviewed = Path("data/raw_variants/TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1")
    original_sets = {split: stems(original / split) for split in ["train", "val", "test"]}
    re_sets = {split: stems(reann / split) for split in ["train", "val", "test"]}
    all_original = set().union(*original_sets.values())

    report = {
        "status": "complete",
        "original_dataset": str(original),
        "reannotated_dataset": str(reann),
        "reviewed_variant": str(reviewed),
        "split_counts": {
            "original": split_counts(original),
            "reannotated": split_counts(reann),
            "reviewed_variant": split_counts(reviewed) if reviewed.exists() else None,
        },
        "overlaps": {
            re_split: {
                orig_split: len(re_sets[re_split] & original_sets[orig_split])
                for orig_split in ["train", "val", "test"]
            }
            for re_split in ["train", "val", "test"]
        },
        "extras_vs_original": {
            split: len(re_sets[split] - all_original)
            for split in ["train", "val", "test"]
        },
        "unsafe_for_success_eval": {
            "reannotated_test_exists": (reann / "test").exists(),
            "reannotated_test_count": len(re_sets["test"]),
            "reason": "User policy forbids using new/reannotated test for success evaluation; success must remain clean-test-v2/test.",
        },
        "safe_training_candidate": {
            "can_use_reannotated_train": True,
            "can_use_reannotated_val": True,
            "must_exclude_reannotated_test_from_training_and_claims": True,
            "recommended_variant_name": "TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1",
            "recommended_policy": (
                "Build an isolated variant with reannotated train/val only, omit or copy original test only as control, "
                "and evaluate success exclusively on TSRS_RSNA-Epiphysis_clean_test_v2/test."
            ),
        },
        "gate_before_gpu": {
            "require_manifest": True,
            "require_no_reannotated_test_in_launcher": True,
            "require_clean_test_v2_eval_only_for_success": True,
            "require_train_val_pair_counts_match": True,
        },
    }
    out = Path("outputs/analysis/r161_reannotated_variant_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
