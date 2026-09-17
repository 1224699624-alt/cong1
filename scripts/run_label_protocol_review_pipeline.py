#!/usr/bin/env python3
"""Run the safe R134 review-decision pipeline.

Default behavior is non-mutating:
1. preview reviewed CSV
2. apply decisions to a temporary manifest copy
3. run the gate checker on that temporary manifest
4. run the dataset-variant builder in dry-run mode

Use `--apply-in-place` only after the preview and temporary checker are clean.
Use `--create-variant` only after an in-place gate pass and correction labels,
if any, are ready.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")
DEFAULT_REVIEWED_CSV = Path("outputs/analysis/r134_label_protocol_review_manifest/review_package/r134_train_val_review_worklist_reviewed.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/r134_label_protocol_review_manifest/review_package")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely preview/apply/check/build R134 review decisions.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_REVIEWED_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply-in-place", action="store_true", help="Update the real manifest after preview passes.")
    parser.add_argument("--create-variant", action="store_true", help="Create the dataset variant instead of builder dry-run. Requires --apply-in-place.")
    parser.add_argument("--backup", action="store_true", help="When applying in place, create a timestamped manifest backup.")
    parser.add_argument("--correction-dir", type=Path, default=None)
    parser.add_argument("--min-reviewed-train", type=int, default=20)
    parser.add_argument("--min-confirmed-train", type=int, default=8)
    return parser.parse_args()


def run_step(command: list[str], *, allow_failure: bool = False) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True)
    step = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(json.dumps(step, indent=2, ensure_ascii=False))
    return step


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.create_variant and not args.apply_in_place:
        raise SystemExit("--create-variant requires --apply-in-place")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_manifest = args.output_dir / "r134_pipeline_preview_manifest.json"
    preview_json = args.output_dir / "r134_pipeline_preview_status.json"
    apply_json = args.output_dir / "r134_pipeline_apply_status.json"
    checker_json = args.output_dir / "r134_pipeline_checker_status.json"
    builder_json = args.output_dir / "r134_pipeline_builder_status.json"
    summary_json = args.output_dir / "r134_pipeline_summary.json"

    python = sys.executable
    preview_cmd = [
        python,
        "scripts/preview_label_protocol_review_decisions.py",
        "--manifest",
        str(args.manifest),
        "--decisions-csv",
        str(args.decisions_csv),
        "--min-reviewed-train",
        str(args.min_reviewed_train),
        "--min-confirmed-train",
        str(args.min_confirmed_train),
        "--output-json",
        str(preview_json),
    ]
    preview_step = run_step(preview_cmd)
    preview = load_json(preview_json)
    if not preview.get("gate_pass_if_applied"):
        summary = {
            "status": "preview_gate_failed",
            "mutated_manifest": False,
            "created_variant": False,
            "preview": preview,
            "steps": {"preview": preview_step},
            "next_action": "continue_review_or_fix_csv_before_apply",
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(3)

    apply_output = args.manifest if args.apply_in_place else temp_manifest
    apply_cmd = [
        python,
        "scripts/apply_label_protocol_review_decisions.py",
        "--manifest",
        str(args.manifest),
        "--decisions-csv",
        str(args.decisions_csv),
        "--output",
        str(apply_output),
        "--summary-json",
        str(apply_json),
    ]
    if args.apply_in_place and args.backup:
        apply_cmd.append("--backup")
    apply_step = run_step(apply_cmd)

    checker_cmd = [
        python,
        "scripts/check_label_protocol_review_manifest.py",
        "--manifest",
        str(apply_output),
        "--min-reviewed-train",
        str(args.min_reviewed_train),
        "--min-confirmed-train",
        str(args.min_confirmed_train),
        "--output-json",
        str(checker_json),
    ]
    checker_step = run_step(checker_cmd)
    checker = load_json(checker_json)
    if not checker.get("gate_pass"):
        summary = {
            "status": "checker_gate_failed",
            "mutated_manifest": bool(args.apply_in_place),
            "created_variant": False,
            "preview": preview,
            "checker": checker,
            "steps": {"preview": preview_step, "apply": apply_step, "checker": checker_step},
            "next_action": "inspect_checker_errors_before_dataset_variant",
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(4)

    builder_cmd = [
        python,
        "scripts/build_label_protocol_dataset_variant.py",
        "--manifest",
        str(apply_output),
        "--min-reviewed-train",
        str(args.min_reviewed_train),
        "--min-confirmed-train",
        str(args.min_confirmed_train),
        "--metadata-json",
        str(builder_json),
    ]
    if not args.create_variant:
        builder_cmd.append("--dry-run")
    if args.correction_dir is not None:
        builder_cmd.extend(["--correction-dir", str(args.correction_dir)])
    builder_step = run_step(builder_cmd)
    builder = load_json(builder_json)

    summary = {
        "status": "variant_created" if args.create_variant else "dry_run_ok",
        "mutated_manifest": bool(args.apply_in_place),
        "created_variant": bool(args.create_variant),
        "preview": preview,
        "checker": checker,
        "builder": builder,
        "steps": {"preview": preview_step, "apply": apply_step, "checker": checker_step, "builder": builder_step},
        "next_action": "launch_gpu_only_after_reviewed_variant_is_intentionally_created" if args.create_variant else "review_dry_run_then_apply_in_place_if_ready",
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
