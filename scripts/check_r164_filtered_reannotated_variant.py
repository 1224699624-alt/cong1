#!/usr/bin/env python3
"""Check R164 filtered reannotated variant safety and pairing invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check R164 filtered variant.")
    parser.add_argument("--variant", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1"))
    parser.add_argument("--original", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--reannotated-source", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r164_filtered_reannotated_variant_check.json"))
    return parser.parse_args()


def stems(path: Path, pattern: str = "*") -> set[str]:
    return {p.stem for p in path.glob(pattern) if p.is_file()}


def split_report(root: Path, split: str) -> dict[str, object]:
    image_stems = stems(root / split)
    label_stems = stems(root / f"{split}_labels", "*.png")
    return {
        "images": len(image_stems),
        "labels": len(label_stems),
        "paired": len(image_stems & label_stems),
        "images_without_labels": sorted(image_stems - label_stems)[:20],
        "labels_without_images": sorted(label_stems - image_stems)[:20],
    }


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    metadata = args.variant / "r164_filtered_reannotated_metadata.json"
    if not metadata.exists():
        errors.append("missing_metadata")
        meta = {}
    else:
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        policy = str(meta.get("policy", ""))
        if "Reannotated test remains excluded" not in policy:
            errors.append("metadata_does_not_exclude_reannotated_test")
        if "clean_test_v2/test" not in policy and "clean-test-v2/test" not in policy:
            errors.append("metadata_does_not_preserve_clean_test_policy")

    splits = {split: split_report(args.variant, split) for split in ("train", "val", "test")}
    expected = {"train": 1159, "val": 95, "test": 97}
    for split, expected_count in expected.items():
        report = splits[split]
        if report["images"] != expected_count or report["labels"] != expected_count or report["paired"] != expected_count:
            errors.append(f"unexpected_{split}_counts")
        if report["images_without_labels"] or report["labels_without_images"]:
            errors.append(f"unpaired_{split}")

    variant_test = stems(args.variant / "test")
    original_test = stems(args.original / "test")
    reannotated_test = stems(args.reannotated_source / "test")
    if variant_test != original_test:
        errors.append("variant_test_not_equal_original_test")
    if variant_test == reannotated_test:
        errors.append("variant_test_equals_reannotated_test")

    report = {
        "ok": not errors,
        "errors": errors,
        "variant": str(args.variant),
        "metadata": str(metadata),
        "splits": splits,
        "test_equals_original_test": variant_test == original_test,
        "test_equals_reannotated_test": variant_test == reannotated_test,
        "reannotated_test_extra_vs_original_count": len(reannotated_test - original_test),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
