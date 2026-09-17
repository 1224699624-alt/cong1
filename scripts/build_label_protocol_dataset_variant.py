#!/usr/bin/env python3
"""Build an isolated train/val label-protocol dataset variant after review.

This script never edits the original dataset. It copies the source dataset into
`data/raw_variants/<target_dataset>` and optionally replaces reviewed train/val
labels from a correction directory. Clean-test-v2 rows in the R134 manifest are
diagnostic-only and are never copied into the variant.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
CONFIRMED_DECISIONS = {"label_ok_protocol_clear", "label_needs_correction"}
CORRECTION_DECISION = "label_needs_correction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an isolated label/protocol dataset variant from reviewed R134 manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json"))
    parser.add_argument("--source-raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--target-raw-root", type=Path, default=Path("data/raw_variants"))
    parser.add_argument("--source-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--target-dataset", default="TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1")
    parser.add_argument("--correction-dir", type=Path, default=None, help="Directory containing corrected label PNGs named like the original labels.")
    parser.add_argument("--min-reviewed-train", type=int, default=20)
    parser.add_argument("--min-confirmed-train", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metadata-json", type=Path, default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_safe_target(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved not in target_resolved.parents and root_resolved != target_resolved:
        raise ValueError(f"Refusing to touch target outside root: {target_resolved}")


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def reviewed_rows(manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = []
    for row in manifest.get(key, []):
        decision = str(row.get("human_decision", "")).strip()
        if decision:
            rows.append(row)
    return rows


def confirmed_rows(manifest: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [row for row in reviewed_rows(manifest, key) if row.get("human_decision") in CONFIRMED_DECISIONS]


def correction_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for key in ("train_review_queue", "val_audit_queue"):
        rows.extend(row for row in manifest.get(key, []) if row.get("human_decision") == CORRECTION_DECISION)
    return rows


def validate_gate(manifest: dict[str, Any], min_reviewed: int, min_confirmed: int) -> dict[str, Any]:
    clean_refs = manifest.get("clean_test_reference", [])
    train_reviewed = reviewed_rows(manifest, "train_review_queue")
    train_confirmed = confirmed_rows(manifest, "train_review_queue")
    errors = []
    for row in clean_refs:
        if str(row.get("human_decision", "")).strip():
            errors.append(f"clean-test-v2 row must not drive variant decisions: {row.get('image')}")
        if "diagnostic_reference_only" not in str(row.get("allowed_use", "")):
            errors.append(f"unsafe clean-test-v2 allowed_use: {row.get('image')}")
    if len(train_reviewed) < min_reviewed:
        errors.append(f"reviewed train rows {len(train_reviewed)} < required {min_reviewed}")
    if len(train_confirmed) < min_confirmed:
        errors.append(f"confirmed train rows {len(train_confirmed)} < required {min_confirmed}")
    return {
        "gate_pass": not errors,
        "errors": errors,
        "reviewed_train": len(train_reviewed),
        "confirmed_train": len(train_confirmed),
        "correction_rows": len(correction_rows(manifest)),
    }


def copy_split(source_dataset_dir: Path, target_dataset_dir: Path, split: str, replacements: dict[str, Path], dry_run: bool) -> dict[str, Any]:
    image_dir = source_dataset_dir / split
    label_dir = source_dataset_dir / f"{split}_labels"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing source split dirs: {image_dir} / {label_dir}")
    target_image_dir = target_dataset_dir / split
    target_label_dir = target_dataset_dir / f"{split}_labels"
    if not dry_run:
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    replaced = 0
    for label_path in sorted(label_dir.glob("*.png")):
        stem = label_path.stem
        image_path = find_image_for_stem(image_dir, stem)
        if image_path is None:
            raise FileNotFoundError(f"Image for stem '{stem}' not found in {image_dir}")
        replacement = replacements.get(label_path.name)
        if replacement is not None and not replacement.exists():
            raise FileNotFoundError(f"Correction label missing: {replacement}")
        if not dry_run:
            shutil.copy2(image_path, target_image_dir / image_path.name)
            shutil.copy2(replacement or label_path, target_label_dir / label_path.name)
        copied += 1
        if replacement is not None:
            replaced += 1
    return {"copied": copied, "label_replacements": replaced}


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    gate = validate_gate(manifest, args.min_reviewed_train, args.min_confirmed_train)
    source_dataset_dir = args.source_raw_root / args.source_dataset
    target_dataset_dir = args.target_raw_root / args.target_dataset
    ensure_safe_target(args.target_raw_root, target_dataset_dir)

    if not gate["gate_pass"]:
        summary = {
            "status": "gate_failed",
            "target_dataset": args.target_dataset,
            "dry_run": args.dry_run,
            "gate": gate,
            "notes": "Fill human_decision fields in the R134 manifest before building a variant.",
        }
        print(json.dumps(summary, indent=2))
        if args.metadata_json is not None:
            args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(3)

    if not source_dataset_dir.exists():
        raise FileNotFoundError(source_dataset_dir)
    if target_dataset_dir.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Target dataset exists: {target_dataset_dir}. Use --overwrite to replace it.")
    if target_dataset_dir.exists() and args.overwrite and not args.dry_run:
        shutil.rmtree(target_dataset_dir)

    replacements: dict[str, dict[str, Path]] = {split: {} for split in SPLITS}
    for row in correction_rows(manifest):
        split = str(row.get("split", ""))
        if split not in {"train", "val"}:
            raise ValueError(f"Corrections are allowed only for train/val rows, got {split}: {row.get('image')}")
        if args.correction_dir is None:
            raise ValueError("Rows marked label_needs_correction require --correction-dir.")
        name = str(row.get("image", ""))
        if not name:
            raise ValueError("Correction row missing image name.")
        replacements[split][name] = args.correction_dir / name

    split_summary = {}
    for split in SPLITS:
        split_summary[split] = copy_split(source_dataset_dir, target_dataset_dir, split, replacements[split], args.dry_run)

    metadata = {
        "status": "dry_run_ok" if args.dry_run else "created",
        "manifest": str(args.manifest),
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "source_raw_root": str(args.source_raw_root),
        "target_raw_root": str(args.target_raw_root),
        "correction_dir": str(args.correction_dir) if args.correction_dir else None,
        "gate": gate,
        "split_summary": split_summary,
        "clean_test_policy": "clean-test-v2 rows were diagnostic-only and were not used to build this variant",
    }
    if not args.dry_run:
        target_dataset_dir.mkdir(parents=True, exist_ok=True)
        (target_dataset_dir / "label_protocol_variant_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.metadata_json is not None:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
