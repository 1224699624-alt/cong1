#!/usr/bin/env python3
"""Static safety check for the R162 reannotated train/val-only launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check R162 launcher for dataset leakage guards.")
    parser.add_argument("--launcher", type=Path, default=Path("outputs/bridge_logs/run_r162_reannotated_trainval_instance_sep.sh"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r162_reannotated_trainval_launcher_check.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.launcher.read_text(encoding="utf-8")
    required = {
        "trainval_variant": "TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1",
        "clean_test_eval": "--eval-dataset TSRS_RSNA-Epiphysis_clean_test_v2",
        "eval_root_variants": "--eval-raw-root data/raw_variants",
        "original_control": "--control-dataset TSRS_RSNA-Epiphysis",
        "metadata_guard": "reannotated_trainval_only_metadata.json",
        "exclude_policy_guard": "reannotated test split is excluded",
        "original_test_equality_guard": "variant test is not exactly original test",
    }
    forbidden = {
        "reannotated_train_dataset": "--dataset TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1",
        "flat_output_suffix_train_direct": 'VARIANT="TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1"',
        "new_test_eval_dataset": "--eval-dataset TSRS_RSNA-Epiphysis_reannotated_flat_output_suffix_v1",
    }
    errors = []
    for name, token in required.items():
        if token not in text:
            errors.append(f"missing_required:{name}:{token}")
    for name, token in forbidden.items():
        if token in text:
            errors.append(f"forbidden_token:{name}:{token}")
    report = {
        "launcher": str(args.launcher),
        "ok": not errors,
        "errors": errors,
        "required_checked": sorted(required),
        "forbidden_checked": sorted(forbidden),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
