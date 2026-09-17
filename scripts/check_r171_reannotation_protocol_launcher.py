#!/usr/bin/env python3
"""Static guard checks for the R171 reannotation-protocol launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_LAUNCHER = Path("outputs/bridge_logs/run_r171_reannotation_protocol_variant_instance_sep.sh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, default=DEFAULT_LAUNCHER)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.launcher.exists():
        raise FileNotFoundError(args.launcher)
    text = args.launcher.read_text(encoding="utf-8")
    required = {
        "r171_dataset": "TSRS_RSNA-Epiphysis_reannotation_protocol_reviewed_v1",
        "r170_metadata_guard": "r170_reannotation_protocol_metadata.json",
        "r170_status_guard": "r170_reannotation_protocol_variant_build_status.json",
        "created_status_guard": "status.get(\"status\") != \"created\"",
        "clean_test_eval_dataset": "--eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2",
        "clean_test_eval_root": "--eval-raw-root data/raw_variants",
        "original_control_dataset": "--control-dataset TSRS_RSNA-Epiphysis",
        "original_control_root": "--control-raw-root data/raw",
        "r171_output_exp": "r171_reannotation_protocol_dinov3_instance_sep",
        "result_monitor": "scripts/monitor_r171_reannotation_protocol_result.py",
        "success_policy_guard": "TSRS_RSNA-Epiphysis_clean_test_v2/test",
        "no_clean_train_policy_guard": "does_not_use_clean_test_v2_for_training",
        "no_reannotated_test_policy_guard": "does_not_use_reannotated_test",
    }
    forbidden = {
        "flat_output_suffix": "flat_output_suffix",
        "raw_reannotated_trainval": "TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1",
        "filtered_reannotated_trainval": "TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1",
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
