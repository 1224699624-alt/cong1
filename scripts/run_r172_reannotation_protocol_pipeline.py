#!/usr/bin/env python3
"""Run the safe R168 -> R170 reannotation-protocol pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/r172_reannotation_protocol_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist-csv", type=Path, default=DEFAULT_WORKLIST)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_REVIEWED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--correction-dir", type=Path, default=None)
    parser.add_argument("--create-variant", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-reviewed-train", type=int, default=40)
    parser.add_argument("--min-usable-train", type=int, default=20)
    parser.add_argument("--min-reviewed-val", type=int, default=8)
    return parser.parse_args()


def run_step(command: list[str], *, allow_failure: bool = False) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True)
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0 and not allow_failure:
        raise RuntimeError(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    preview_json = args.output_dir / "r172_preview_status.json"
    build_json = Path("outputs/analysis/r170_reannotation_protocol_variant_build_status.json")
    summary_json = args.output_dir / "r172_pipeline_summary.json"

    preview_cmd = [
        python,
        "scripts/preview_r168_reannotation_protocol_decisions.py",
        "--worklist-csv",
        str(args.worklist_csv),
        "--decisions-csv",
        str(args.decisions_csv),
        "--output-json",
        str(preview_json),
        "--min-reviewed-train",
        str(args.min_reviewed_train),
        "--min-usable-train",
        str(args.min_usable_train),
        "--min-reviewed-val",
        str(args.min_reviewed_val),
    ]
    preview_step = run_step(preview_cmd, allow_failure=True)
    preview = load_json(preview_json) if preview_json.exists() else {"gate_pass_if_applied": False, "errors": ["preview did not write json"]}
    if not preview.get("gate_pass_if_applied"):
        summary = {
            "status": "preview_gate_failed",
            "mutated_dataset": False,
            "created_variant": False,
            "preview": preview,
            "steps": {"preview": preview_step},
            "next_action": "continue_human_review_or_fix_reviewed_csv",
        }
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit(3)

    build_cmd = [
        python,
        "scripts/build_r170_reannotation_protocol_variant.py",
        "--worklist-csv",
        str(args.worklist_csv),
        "--decisions-csv",
        str(args.decisions_csv),
        "--output-json",
        str(build_json),
        "--min-reviewed-train",
        str(args.min_reviewed_train),
        "--min-usable-train",
        str(args.min_usable_train),
        "--min-reviewed-val",
        str(args.min_reviewed_val),
    ]
    if args.correction_dir is not None:
        build_cmd.extend(["--correction-dir", str(args.correction_dir)])
    if args.create_variant:
        build_cmd.append("--create")
    if args.overwrite:
        build_cmd.append("--overwrite")
    build_step = run_step(build_cmd, allow_failure=True)
    build = load_json(build_json) if build_json.exists() else {"status": "missing_build_json", "errors": ["build did not write json"]}
    created = build.get("status") == "created"
    dry_ok = build.get("status") == "dry_run_ok"
    status = "variant_created" if created else ("dry_run_ok" if dry_ok else "build_blocked")
    if args.create_variant and not created:
        status = "create_failed"
    summary = {
        "status": status,
        "mutated_dataset": bool(created),
        "created_variant": bool(created),
        "preview": preview,
        "build": build,
        "steps": {
            "preview": preview_step,
            "build": build_step,
        },
        "next_action": (
            "launch_r171_guarded_training"
            if created
            else ("rerun_with_create_variant_if_dry_run_is_acceptable" if dry_ok else "fix_review_or_correction_inputs")
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if status in {"build_blocked", "create_failed"}:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
