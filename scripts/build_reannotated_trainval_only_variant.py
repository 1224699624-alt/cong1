#!/usr/bin/env python3
"""Build an isolated Epiphysis variant using reannotated train/val only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build safe train/val-only reannotated Epiphysis variant.")
    parser.add_argument("--source-reannotated", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1"))
    parser.add_argument("--original", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--target-root", type=Path, default=Path("data/raw_variants"))
    parser.add_argument("--target-name", default="TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1")
    parser.add_argument("--audit-json", type=Path, default=Path("outputs/analysis/r161_reannotated_variant_audit.json"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def copy_split(src_root: Path, dst_root: Path, split: str, dry_run: bool) -> dict[str, int]:
    src_img = src_root / split
    src_lab = src_root / f"{split}_labels"
    dst_img = dst_root / split
    dst_lab = dst_root / f"{split}_labels"
    images = sorted(p for p in src_img.glob("*") if p.is_file())
    labels = sorted(p for p in src_lab.glob("*.png") if p.is_file())
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    if image_stems != label_stems:
        raise RuntimeError(f"Unpaired files in {split}: images_only={len(image_stems-label_stems)} labels_only={len(label_stems-image_stems)}")
    if not dry_run:
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        for path in images:
            shutil.copy2(path, dst_img / path.name)
        for path in labels:
            shutil.copy2(path, dst_lab / path.name)
    return {"images": len(images), "labels": len(labels), "paired": len(image_stems)}


def main() -> None:
    args = parse_args()
    target = args.target_root / args.target_name
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    if not audit.get("safe_training_candidate", {}).get("must_exclude_reannotated_test_from_training_and_claims"):
        raise RuntimeError("Audit does not explicitly require excluding reannotated test.")
    if target.exists() and not args.dry_run:
        raise RuntimeError(f"Target already exists: {target}")

    summary = {
        "status": "dry_run_ok" if args.dry_run else "created",
        "source_reannotated": str(args.source_reannotated),
        "original": str(args.original),
        "target": str(target),
        "policy": (
            "train and val are copied from the reannotated variant; test is copied only from the original dataset as a control. "
            "The reannotated test split is excluded from this variant and must not be used for success evaluation. "
            "Success evaluation remains TSRS_RSNA-Epiphysis_clean_test_v2/test."
        ),
        "splits": {},
    }
    split_sources = {
        "train": args.source_reannotated,
        "val": args.source_reannotated,
        "test": args.original,
    }
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=False)
    for split, source in split_sources.items():
        summary["splits"][split] = {
            "source": str(source),
            **copy_split(source, target, split, args.dry_run),
        }
    summary["excluded"] = {
        "reannotated_test": str(args.source_reannotated / "test"),
        "reannotated_test_count": audit["unsafe_for_success_eval"]["reannotated_test_count"],
    }
    out = Path("outputs/analysis/r161_reannotated_trainval_only_variant_build.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.dry_run:
        (target / "reannotated_trainval_only_metadata.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
