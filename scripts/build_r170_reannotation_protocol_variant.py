#!/usr/bin/env python3
"""Build an isolated train/val variant from R168 review decisions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_WORKLIST = Path(
    "outputs/analysis/r168_reannotation_protocol_review_package/"
    "r168_reannotation_protocol_review_worklist.csv"
)
DEFAULT_REVIEWED = Path(
    "outputs/analysis/r168_reannotation_protocol_review_package/"
    "r168_reannotation_protocol_review_reviewed.csv"
)
DEFAULT_OUTPUT_JSON = Path("outputs/analysis/r170_reannotation_protocol_variant_build.json")
DEFAULT_TARGET = "TSRS_RSNA-Epiphysis_reannotation_protocol_reviewed_v1"
USABLE_DECISIONS = {
    "original_label_correct",
    "reannotated_label_correct",
    "both_wrong_needs_correction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-csv", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--original-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument(
        "--reannotated-root",
        type=Path,
        default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1"),
    )
    parser.add_argument("--target-raw-root", type=Path, default=Path("data/raw_variants"))
    parser.add_argument("--target-dataset", default=DEFAULT_TARGET)
    parser.add_argument("--correction-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--create", action="store_true", help="Actually create the variant; default is dry-run.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-reviewed-train", type=int, default=40)
    parser.add_argument("--min-usable-train", type=int, default=20)
    parser.add_argument("--min-reviewed-val", type=int, default=8)
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "image", "human_decision"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return rows


def key(row: dict[str, str]) -> tuple[str, str]:
    return str(row.get("split", "")).strip(), str(row.get("image", "")).strip()


def index_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in indexed:
            raise ValueError(f"duplicate row: {row_key}")
        indexed[row_key] = row
    return indexed


def preview_gate(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_json.with_name(args.output_json.stem + "_preview.json")
    command = [
        sys.executable,
        "scripts/preview_r168_reannotation_protocol_decisions.py",
        "--worklist-csv",
        str(args.worklist_csv),
        "--decisions-csv",
        str(args.decisions_csv),
        "--output-json",
        str(output),
        "--min-reviewed-train",
        str(args.min_reviewed_train),
        "--min-usable-train",
        str(args.min_usable_train),
        "--min-reviewed-val",
        str(args.min_reviewed_val),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if output.exists():
        summary = json.loads(output.read_text(encoding="utf-8"))
    else:
        summary = {
            "status": "preview_failed",
            "gate_pass_if_applied": False,
            "errors": [completed.stderr or completed.stdout],
        }
    summary["preview_command"] = command
    summary["preview_returncode"] = completed.returncode
    summary["preview_stdout"] = completed.stdout
    summary["preview_stderr"] = completed.stderr
    return summary


def merge_reviewed(worklist_path: Path, decisions_path: Path) -> list[dict[str, str]]:
    worklist = index_rows(load_csv(worklist_path))
    reviewed = load_csv(decisions_path)
    merged = {row_key: dict(row) for row_key, row in worklist.items()}
    for row in reviewed:
        row_key = key(row)
        if row_key not in merged:
            continue
        merged[row_key]["human_decision"] = str(row.get("human_decision", "")).strip()
        merged[row_key]["human_notes"] = str(row.get("human_notes", "")).strip()
    return list(merged.values())


def ensure_safe_target(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved != target_resolved and root_resolved not in target_resolved.parents:
        raise ValueError(f"Refusing target outside target root: {target_resolved}")


def find_image(split_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = split_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def split_label_names(root: Path, split: str) -> list[str]:
    label_dir = root / f"{split}_labels"
    if not label_dir.exists():
        raise FileNotFoundError(label_dir)
    return sorted(path.name for path in label_dir.glob("*.png"))


def copy_pair(
    *,
    source_root: Path,
    target_root: Path,
    split: str,
    label_name: str,
    label_source: Path,
    create: bool,
) -> None:
    stem = Path(label_name).stem
    image_path = find_image(source_root / split, stem)
    if image_path is None:
        raise FileNotFoundError(f"missing image for {split}/{label_name} under {source_root}")
    if not label_source.exists():
        raise FileNotFoundError(label_source)
    if not create:
        return
    (target_root / split).mkdir(parents=True, exist_ok=True)
    (target_root / f"{split}_labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, target_root / split / image_path.name)
    shutil.copy2(label_source, target_root / f"{split}_labels" / label_name)


def label_source_for_row(row: dict[str, str], args: argparse.Namespace) -> tuple[str, Path | None]:
    split = row["split"]
    name = row["image"]
    decision = row.get("human_decision", "")
    if decision == "original_label_correct":
        return "original", args.original_root / f"{split}_labels" / name
    if decision == "reannotated_label_correct":
        return "reannotated", args.reannotated_root / f"{split}_labels" / name
    if decision == "both_wrong_needs_correction":
        if args.correction_dir is None:
            return "missing_correction_dir", None
        return "corrected", args.correction_dir / name
    if decision == "exclude_from_training_variant":
        return "excluded", None
    if decision == "uncertain_second_review":
        return "uncertain", None
    return "default_original", args.original_root / f"{split}_labels" / name


def build_variant(args: argparse.Namespace, rows: list[dict[str, str]], preview: dict[str, Any]) -> dict[str, Any]:
    target = args.target_raw_root / args.target_dataset
    ensure_safe_target(args.target_raw_root, target)
    create = bool(args.create)
    errors: list[str] = []
    selected_counts = {
        "default_original": 0,
        "original": 0,
        "reannotated": 0,
        "corrected": 0,
        "excluded": 0,
        "uncertain": 0,
    }
    split_summary: dict[str, dict[str, Any]] = {}
    reviewed_by_key = {key(row): row for row in rows if row.get("split") in {"train", "val"}}

    if not preview.get("gate_pass_if_applied"):
        errors.append("R168 preview gate did not pass; refusing to build variant.")
    if target.exists() and create:
        if not args.overwrite:
            errors.append(f"target exists: {target}; use --overwrite to replace it")
        else:
            shutil.rmtree(target)
    if create and errors:
        raise RuntimeError("; ".join(errors))

    for split in ("train", "val"):
        copied = 0
        skipped = 0
        label_sources = Counter()
        missing: list[str] = []
        for name in split_label_names(args.original_root, split):
            row = reviewed_by_key.get((split, name), {"split": split, "image": name, "human_decision": ""})
            source_kind, label_path = label_source_for_row(row, args)
            selected_counts[source_kind] = selected_counts.get(source_kind, 0) + 1
            label_sources[source_kind] += 1
            if source_kind in {"excluded", "uncertain"}:
                skipped += 1
                continue
            if label_path is None:
                missing.append(name)
                continue
            copy_pair(
                source_root=args.original_root,
                target_root=target,
                split=split,
                label_name=name,
                label_source=label_path,
                create=create,
            )
            copied += 1
        if missing:
            errors.extend(f"missing corrected label for {split}/{name}" for name in missing)
        split_summary[split] = {
            "copied": copied,
            "skipped": skipped,
            "label_sources": dict(label_sources),
        }

    test_copied = 0
    for name in split_label_names(args.original_root, "test"):
        copy_pair(
            source_root=args.original_root,
            target_root=target,
            split="test",
            label_name=name,
            label_source=args.original_root / "test_labels" / name,
            create=create,
        )
        test_copied += 1
    split_summary["test"] = {
        "copied": test_copied,
        "skipped": 0,
        "label_sources": {"original_control_only": test_copied},
    }
    if errors and create:
        raise RuntimeError("; ".join(errors))
    metadata = {
        "status": "created" if create else ("dry_run_blocked" if errors else "dry_run_ok"),
        "dry_run": not create,
        "errors": errors,
        "target_dataset": args.target_dataset,
        "target_path": str(target),
        "worklist_csv": str(args.worklist_csv),
        "decisions_csv": str(args.decisions_csv),
        "original_root": str(args.original_root),
        "reannotated_root": str(args.reannotated_root),
        "correction_dir": str(args.correction_dir) if args.correction_dir else None,
        "preview": {
            "gate_pass_if_applied": preview.get("gate_pass_if_applied"),
            "counts": preview.get("counts", {}),
            "decision_counts": preview.get("decision_counts", {}),
            "errors": preview.get("errors", []),
            "warnings": preview.get("warnings", []),
        },
        "selected_counts": selected_counts,
        "split_summary": split_summary,
        "policy": {
            "source_test": "original TSRS_RSNA-Epiphysis/test only as control",
            "does_not_use_clean_test_v2_for_training": True,
            "does_not_use_reannotated_test": True,
            "success_eval_remains": "TSRS_RSNA-Epiphysis_clean_test_v2/test",
        },
    }
    if create:
        target.mkdir(parents=True, exist_ok=True)
        (target / "r170_reannotation_protocol_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return metadata


def main() -> None:
    args = parse_args()
    preview = preview_gate(args)
    rows = merge_reviewed(args.worklist_csv, args.decisions_csv)
    metadata = build_variant(args, rows, preview)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if metadata["errors"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
