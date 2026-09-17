#!/usr/bin/env python3
"""Preview R168 reannotation-protocol review decisions without mutating data."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_WORKLIST = Path(
    "outputs/analysis/r168_reannotation_protocol_review_package/"
    "r168_reannotation_protocol_review_worklist.csv"
)
DEFAULT_REVIEWED = Path(
    "outputs/analysis/r168_reannotation_protocol_review_package/"
    "r168_reannotation_protocol_review_reviewed.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/analysis/r168_reannotation_protocol_review_package/"
    "r168_review_preview_status.json"
)
ALLOWED_DECISIONS = {
    "original_label_correct",
    "reannotated_label_correct",
    "both_wrong_needs_correction",
    "exclude_from_training_variant",
    "uncertain_second_review",
}
TRAIN_USABLE_DECISIONS = {
    "original_label_correct",
    "reannotated_label_correct",
    "both_wrong_needs_correction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-csv", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
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


def index_worklist(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in indexed:
            raise ValueError(f"duplicate worklist row: {row_key}")
        indexed[row_key] = row
    return indexed


def merge_rows(
    worklist: dict[tuple[str, str], dict[str, str]],
    decisions: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    merged = {row_key: dict(row) for row_key, row in worklist.items()}
    seen_decision_keys: set[tuple[str, str]] = set()
    for csv_idx, row in enumerate(decisions, start=2):
        row_key = key(row)
        split, image = row_key
        decision = str(row.get("human_decision", "")).strip()
        notes = str(row.get("human_notes", "")).strip()
        if row_key in seen_decision_keys:
            errors.append(f"row {csv_idx}: duplicate decision row for {split}/{image}")
            continue
        seen_decision_keys.add(row_key)
        if split not in {"train", "val"}:
            errors.append(f"row {csv_idx}: forbidden split {split!r}; R168 accepts train/val only")
            continue
        if row_key not in merged:
            errors.append(f"row {csv_idx}: unknown R168 review row {split}/{image}")
            continue
        if decision and decision not in ALLOWED_DECISIONS:
            errors.append(f"row {csv_idx}: invalid decision for {split}/{image}: {decision}")
            continue
        if not decision and notes:
            warnings.append(f"row {csv_idx}: notes without decision for {split}/{image}")
        merged[row_key]["human_decision"] = decision
        merged[row_key]["human_notes"] = notes
    return list(merged.values()), errors, warnings


def summarize(rows: list[dict[str, str]], args: argparse.Namespace, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    reviewed_train = [row for row in train if row.get("human_decision")]
    reviewed_val = [row for row in val if row.get("human_decision")]
    usable_train = [row for row in train if row.get("human_decision") in TRAIN_USABLE_DECISIONS]
    corrections_needed = [row for row in rows if row.get("human_decision") == "both_wrong_needs_correction"]
    selected_reannotated = [row for row in rows if row.get("human_decision") == "reannotated_label_correct"]
    selected_original = [row for row in rows if row.get("human_decision") == "original_label_correct"]
    excluded = [row for row in rows if row.get("human_decision") == "exclude_from_training_variant"]
    uncertain = [row for row in rows if row.get("human_decision") == "uncertain_second_review"]
    decision_counts = Counter(row.get("human_decision") or "blank" for row in rows)
    train_decision_counts = Counter(row.get("human_decision") or "blank" for row in train)
    val_decision_counts = Counter(row.get("human_decision") or "blank" for row in val)
    gate_pass = (
        not errors
        and len(reviewed_train) >= args.min_reviewed_train
        and len(usable_train) >= args.min_usable_train
        and len(reviewed_val) >= args.min_reviewed_val
    )
    if gate_pass:
        if corrections_needed:
            next_action = "collect_corrected_pngs_then_build_isolated_variant"
        else:
            next_action = "build_isolated_variant_from_selected_original_or_reannotated_labels"
    else:
        next_action = "continue_human_review_or_fix_reviewed_csv"
    return {
        "status": "error" if errors else "ok",
        "worklist_csv": str(args.worklist_csv),
        "decisions_csv": str(args.decisions_csv),
        "gate_pass_if_applied": gate_pass,
        "errors": errors,
        "warnings": warnings,
        "thresholds": {
            "min_reviewed_train": args.min_reviewed_train,
            "min_usable_train": args.min_usable_train,
            "min_reviewed_val": args.min_reviewed_val,
        },
        "counts": {
            "rows": len(rows),
            "train_rows": len(train),
            "val_rows": len(val),
            "reviewed_train": len(reviewed_train),
            "usable_train": len(usable_train),
            "reviewed_val": len(reviewed_val),
            "selected_original": len(selected_original),
            "selected_reannotated": len(selected_reannotated),
            "corrections_needed": len(corrections_needed),
            "excluded": len(excluded),
            "uncertain": len(uncertain),
        },
        "decision_counts": dict(decision_counts),
        "train_decision_counts": dict(train_decision_counts),
        "val_decision_counts": dict(val_decision_counts),
        "correction_rows": [
            {"split": row["split"], "image": row["image"], "panel": row.get("panel", "")}
            for row in corrections_needed
        ],
        "policy": {
            "mutates_dataset": False,
            "uses_clean_test_v2": False,
            "uses_reannotated_test": False,
            "success_eval_remains": "TSRS_RSNA-Epiphysis_clean_test_v2/test",
        },
        "next_action": next_action,
    }


def main() -> None:
    args = parse_args()
    worklist = load_csv(args.worklist_csv)
    decisions = load_csv(args.decisions_csv)
    merged, errors, warnings = merge_rows(index_worklist(worklist), decisions)
    summary = summarize(merged, args, errors, warnings)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
