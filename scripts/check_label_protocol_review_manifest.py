#!/usr/bin/env python3
"""Validate an R134-style label/protocol review manifest.

The checker is intentionally read-only. It verifies that clean-test-v2 cases
remain diagnostic-only and reports whether enough train cases have been
reviewed to justify building an isolated corrected/verified data variant.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check R134 label/protocol review manifest gates.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV mirror to cross-check row count.")
    parser.add_argument("--min-reviewed-train", type=int, default=20)
    parser.add_argument("--min-confirmed-train", type=int, default=8)
    parser.add_argument("--require-val-reviewed", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def is_reviewed(row: dict[str, Any]) -> bool:
    return bool(str(row.get("human_decision", "")).strip())


def is_confirmed_for_variant(row: dict[str, Any]) -> bool:
    return row.get("human_decision") in {"label_ok_protocol_clear", "label_needs_correction"}


def rel_exists(path_text: str) -> bool:
    return not path_text or Path(path_text).exists()


def main() -> None:
    args = parse_args()
    payload = load_manifest(args.manifest)
    allowed = set(payload.get("human_decision_schema", {}).get("human_decision_allowed", []))
    errors: list[str] = []
    warnings: list[str] = []

    clean_refs = payload.get("clean_test_reference", [])
    train_rows = payload.get("train_review_queue", [])
    val_rows = payload.get("val_audit_queue", [])

    for row in clean_refs:
        if "diagnostic_reference_only" not in str(row.get("allowed_use", "")):
            errors.append(f"clean-test reference has unsafe allowed_use: {row.get('image')}")
        if row.get("split") != "clean-test-v2/test":
            errors.append(f"clean-test reference has unexpected split: {row.get('image')} {row.get('split')}")

    for section_name, rows in [
        ("clean_test_reference", clean_refs),
        ("train_review_queue", train_rows),
        ("val_audit_queue", val_rows),
    ]:
        for row in rows:
            decision = str(row.get("human_decision", "")).strip()
            if decision and decision not in allowed:
                errors.append(f"{section_name} invalid human_decision for {row.get('image')}: {decision}")
            panel = str(row.get("panel", "")).strip()
            if panel and not rel_exists(panel):
                warnings.append(f"{section_name} panel missing for {row.get('image')}: {panel}")

    if args.csv is not None:
        csv_rows = count_csv_rows(args.csv)
        expected = len(clean_refs) + len(train_rows) + len(val_rows)
        if csv_rows != expected:
            errors.append(f"CSV row count mismatch: {csv_rows} != {expected}")

    reviewed_train = [row for row in train_rows if is_reviewed(row)]
    confirmed_train = [row for row in train_rows if is_confirmed_for_variant(row)]
    reviewed_val = [row for row in val_rows if is_reviewed(row)]
    p0_confirmed = [row for row in confirmed_train if row.get("priority") == "P0"]
    gate_pass = (
        len(errors) == 0
        and len(reviewed_train) >= args.min_reviewed_train
        and len(confirmed_train) >= args.min_confirmed_train
        and (not args.require_val_reviewed or len(reviewed_val) == len(val_rows))
    )

    summary = {
        "manifest": str(args.manifest),
        "gate_pass": gate_pass,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "clean_test_reference": len(clean_refs),
            "train_review_queue": len(train_rows),
            "val_audit_queue": len(val_rows),
            "reviewed_train": len(reviewed_train),
            "confirmed_train_for_variant": len(confirmed_train),
            "confirmed_train_p0": len(p0_confirmed),
            "reviewed_val": len(reviewed_val),
        },
        "train_decision_counts": dict(Counter(str(row.get("human_decision", "")).strip() or "blank" for row in train_rows)),
        "val_decision_counts": dict(Counter(str(row.get("human_decision", "")).strip() or "blank" for row in val_rows)),
        "next_action": (
            "build_isolated_corrected_variant"
            if gate_pass
            else "continue_human_review_before_gpu_or_dataset_variant"
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
