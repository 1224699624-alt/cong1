#!/usr/bin/env python3
"""Apply human review decisions from CSV into an R134 manifest.

The script updates only `human_decision`, `human_notes`, and `review_status`.
By default it refuses decisions for clean-test-v2 diagnostic reference rows so
they cannot influence dataset-variant creation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply R134 human review decisions from CSV to JSON manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decisions-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Output manifest path. Defaults to in-place update.")
    parser.add_argument("--backup", action="store_true", help="When updating in place, write a timestamped backup first.")
    parser.add_argument("--allow-clean-test-decisions", action="store_true", help="Allow updating diagnostic clean-test-v2 rows. These still do not affect build gates.")
    parser.add_argument("--summary-json", type=Path, default=None)
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
    errors: list[str] = []
    warnings: list[str] = []
    applied = 0
    skipped_blank = 0
    skipped_clean = 0

    for row in rows:
        split = str(row.get("split", "")).strip()
        image = str(row.get("image", "")).strip()
        decision = str(row.get("human_decision", "")).strip()
        notes = str(row.get("human_notes", "")).strip()
        if not decision and not notes:
            skipped_blank += 1
            continue
        if decision and decision not in allowed:
            errors.append(f"invalid decision for {split}/{image}: {decision}")
            continue
        key = (split, image)
        if key not in index:
            errors.append(f"decision row not found in manifest: {split}/{image}")
            continue
        section, target = index[key]
        if section == "clean_test_reference" and not args.allow_clean_test_decisions:
            if decision:
                skipped_clean += 1
                warnings.append(f"ignored clean-test-v2 decision for {image}; use --allow-clean-test-decisions only for diagnostic notes")
            continue
        if decision:
            target["human_decision"] = decision
            target["review_status"] = "reviewed"
        if notes:
            target["human_notes"] = notes
        applied += 1

    if errors:
        summary = {
            "status": "error",
            "errors": errors,
            "warnings": warnings,
            "applied": applied,
            "skipped_blank": skipped_blank,
            "skipped_clean_test_decisions": skipped_clean,
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    output = args.output or args.manifest
    if output == args.manifest and args.backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = args.manifest.with_name(f"{args.manifest.stem}_{stamp}.bak{args.manifest.suffix}")
        shutil.copy2(args.manifest, backup)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "status": "ok",
        "manifest": str(args.manifest),
        "decisions_csv": str(args.decisions_csv),
        "output": str(output),
        "applied": applied,
        "skipped_blank": skipped_blank,
        "skipped_clean_test_decisions": skipped_clean,
        "warnings": warnings,
        "next_step": "run scripts/check_label_protocol_review_manifest.py on the updated manifest",
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
