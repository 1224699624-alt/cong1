#!/usr/bin/env python3
"""Static guard checks for the R140 reviewed-variant launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_LAUNCHER = Path("outputs/bridge_logs/run_r140_reviewed_variant_instance_sep.sh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check R140 reviewed-variant launcher invariants.")
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.launcher.exists():
        raise FileNotFoundError(args.launcher)
    text = args.launcher.read_text(encoding="utf-8")
    required = {
        "reviewed_variant_dataset": "TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1",
        "clean_test_eval_dataset": "--eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2",
        "clean_test_eval_root": "--eval-raw-root data/raw_variants",
        "original_control_dataset": "--control-dataset TSRS_RSNA-Epiphysis",
        "gate_checker": "scripts/check_label_protocol_review_manifest.py",
        "variant_metadata_guard": "label_protocol_variant_metadata.json",
        "clean_test_policy_guard": "clean-test-v2 rows were diagnostic-only and were not used to build this variant",
        "r140_output_exp": "r140_reviewed_variant_dinov3_instance_sep",
        "result_monitor": "scripts/monitor_r140_reviewed_variant_result.py",
    }
    forbidden = {
        "new_reannotated_test": "flat_output_suffix",
        "reannotated_dataset": "reannotated",
        "clean_test_train_dataset": "--dataset TSRS_RSNA-Epiphysis_clean_test_v2",
    }
    errors = [f"missing required token {name}: {token}" for name, token in required.items() if token not in text]
    errors.extend(f"forbidden token present {name}: {token}" for name, token in forbidden.items() if token in text)
    summary = {
        "launcher": str(args.launcher),
        "ok": not errors,
        "errors": errors,
        "required_checked": sorted(required),
        "forbidden_checked": sorted(forbidden),
    }
    output = json.dumps(summary, indent=2, ensure_ascii=False)
    print(output)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
