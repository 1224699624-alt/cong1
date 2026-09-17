#!/usr/bin/env python3
"""Preview R134 review CSV decisions before mutating the manifest.

This is a read-only companion to `apply_label_protocol_review_decisions.py`.
It answers: if this CSV were applied, would the review gate pass, and are
there any unsafe rows such as clean-test-v2 decisions or unknown images?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")
CONFIRMED_DECISIONS = {"label_ok_protocol_clear", "label_needs_correction"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview R134 human review CSV decisions without writing the manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--min-reviewed-train", type=int, default=20)
    parser.add_argument("--min-confirmed-train", type=int, default=8)
    parser.add_argument("--allow-clean-test-decisions", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_decisions(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", "split", "human_decision"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"decisions CSV missing columns: {sorted(missing)}")
    return rows


def manifest_index(payload: dict[str, Any]) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    out: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for section in ("clean_test_reference", "train_review_queue", "val_audit_queue"):
        for row in payload.get(section, []):
            key = (str(row.get("split", "")), str(row.get("image", "")))
            if key in out:
                raise ValueError(f"duplicate manifest key: {key}")
            out[key] = (section, row)
    return out


def main() -> None:
    args = parse_args()
    payload = load_json(args.manifest)
    rows = load_decisions(args.decisions_csv)
    allowed = set(payload.get("human_decision_schema", {}).get("human_decision_allowed", []))
    index = manifest_index(payload)
    decisions_by_key: dict[tuple[str, str], str] = {}
    notes_by_key: dict[tuple[str, str], str] = {}
    errors: list[str] = []
    warnings: list[str] = []
    skipped_blank = 0
    skipped_clean = 0

    for csv_idx, row in enumerate(rows, start=2):
        split = str(row.get("split", "")).strip()
        image = str(row.get("image", "")).strip()
        decision = str(row.get("human_decision", "")).strip()
        notes = str(row.get("human_notes", "")).strip()
        key = (split, image)
        if not decision and not notes:
            skipped_blank += 1
            continue
        if decision and decision not in allowed:
            errors.append(f"row {csv_idx}: invalid decision for {split}/{image}: {decision}")
            continue
        if key not in index:
            errors.append(f"row {csv_idx}: decision row not found in manifest: {split}/{image}")
            continue
        section, _ = index[key]
        if section == "clean_test_reference" and not args.allow_clean_test_decisions:
            if decision:
                skipped_clean += 1
                warnings.append(f"row {csv_idx}: clean-test-v2 decision ignored for {image}")
            continue
        if decision:
            decisions_by_key[key] = decision
        if notes:
            notes_by_key[key] = notes

    train_rows = payload.get("train_review_queue", [])
    val_rows = payload.get("val_audit_queue", [])
    reviewed_train = []
    confirmed_train = []
    reviewed_val = []
    for row in train_rows:
        key = (str(row.get("split", "")), str(row.get("image", "")))
        decision = decisions_by_key.get(key, str(row.get("human_decision", "")).strip())
        if decision:
            reviewed_train.append(row)
        if decision in CONFIRMED_DECISIONS:
            confirmed_train.append(row)
    for row in val_rows:
        key = (str(row.get("split", "")), str(row.get("image", "")))
        decision = decisions_by_key.get(key, str(row.get("human_decision", "")).strip())
        if decision:
            reviewed_val.append(row)

    gate_pass = (
        not errors
        and len(reviewed_train) >= args.min_reviewed_train
        and len(confirmed_train) >= args.min_confirmed_train
    )
    train_decision_counts = Counter()
    val_decision_counts = Counter()
    for row in train_rows:
        key = (str(row.get("split", "")), str(row.get("image", "")))
        train_decision_counts[decisions_by_key.get(key, str(row.get("human_decision", "")).strip()) or "blank"] += 1
    for row in val_rows:
        key = (str(row.get("split", "")), str(row.get("image", "")))
        val_decision_counts[decisions_by_key.get(key, str(row.get("human_decision", "")).strip()) or "blank"] += 1

    summary = {
        "status": "error" if errors else "ok",
        "manifest": str(args.manifest),
        "decisions_csv": str(args.decisions_csv),
        "gate_pass_if_applied": gate_pass,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "csv_rows": len(rows),
            "applied_decision_rows_preview": len(decisions_by_key),
            "notes_rows_preview": len(notes_by_key),
            "skipped_blank": skipped_blank,
            "skipped_clean_test_decisions": skipped_clean,
            "reviewed_train_if_applied": len(reviewed_train),
            "confirmed_train_if_applied": len(confirmed_train),
            "reviewed_val_if_applied": len(reviewed_val),
        },
        "train_decision_counts_if_applied": dict(train_decision_counts),
        "val_decision_counts_if_applied": dict(val_decision_counts),
        "next_action": (
            "apply_decisions_then_run_checker"
            if gate_pass
            else "continue_review_or_fix_csv_before_apply"
        ),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
