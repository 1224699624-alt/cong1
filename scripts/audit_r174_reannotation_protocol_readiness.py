#!/usr/bin/env python3
"""Audit R168-R173 readiness for the reannotation-protocol branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("outputs/analysis/r174_reannotation_protocol_readiness.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r168-worklist", type=Path, default=Path("outputs/analysis/r168_reannotation_protocol_review_package/r168_reannotation_protocol_review_worklist.csv"))
    parser.add_argument("--r168-reviewed", type=Path, default=Path("outputs/analysis/r168_reannotation_protocol_review_package/r168_reannotation_protocol_review_reviewed.csv"))
    parser.add_argument("--r172-summary", type=Path, default=Path("outputs/analysis/r172_reannotation_protocol_pipeline/r172_pipeline_summary.json"))
    parser.add_argument("--r170-status", type=Path, default=Path("outputs/analysis/r170_reannotation_protocol_variant_build_status.json"))
    parser.add_argument("--r171-launcher-check", type=Path, default=Path("outputs/analysis/r171_reannotation_protocol_launcher_check.json"))
    parser.add_argument("--r173-summary", type=Path, default=Path("outputs/analysis/r171_reannotation_protocol_dinov3_instance_sep_result_summary.json"))
    parser.add_argument("--target-variant", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotation_protocol_reviewed_v1"))
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def split_pair_counts(root: Path, split: str) -> dict[str, Any]:
    image_dir = root / split
    label_dir = root / f"{split}_labels"
    images = {p.stem for p in image_dir.glob("*") if p.is_file()} if image_dir.exists() else set()
    labels = {p.stem for p in label_dir.glob("*.png")} if label_dir.exists() else set()
    return {
        "images": len(images),
        "labels": len(labels),
        "paired": len(images & labels),
        "images_without_labels": sorted(images - labels)[:20],
        "labels_without_images": sorted(labels - images)[:20],
    }


def decide_next(
    *,
    reviewed_exists: bool,
    r172: dict[str, Any] | None,
    r170: dict[str, Any] | None,
    variant_exists: bool,
    launcher_ok: bool,
    r173: dict[str, Any] | None,
) -> str:
    if not reviewed_exists:
        return "human_review_r168_html_and_export_reviewed_csv"
    if not r172 or r172.get("status") == "preview_gate_failed":
        return "run_r172_pipeline_after_review_or_continue_review_until_gate_passes"
    if r172.get("status") == "dry_run_ok":
        return "rerun_r172_with_create_variant_if_dry_run_is_acceptable"
    if not r170 or r170.get("status") != "created" or not variant_exists:
        return "fix_or_create_r170_isolated_variant"
    if not launcher_ok:
        return "fix_r171_launcher_static_check_before_training"
    if not r173 or r173.get("status") == "waiting_for_metrics":
        return "launch_or_monitor_r171_guarded_training"
    if r173.get("status") == "target_met":
        return "run_result_to_claim_and_experiment_audit"
    if r173.get("status") == "new_best_below_target":
        return "analyze_r171_and_choose_next_branch"
    return "do_not_continue_r171_without_new_evidence"


def main() -> None:
    args = parse_args()
    r172 = load_json(args.r172_summary)
    r170 = load_json(args.r170_status)
    r171 = load_json(args.r171_launcher_check)
    r173 = load_json(args.r173_summary)
    variant_exists = args.target_variant.exists()
    launcher_ok = bool(r171 and r171.get("ok"))
    next_action = decide_next(
        reviewed_exists=args.r168_reviewed.exists(),
        r172=r172,
        r170=r170,
        variant_exists=variant_exists,
        launcher_ok=launcher_ok,
        r173=r173,
    )
    variant_counts = None
    if variant_exists:
        variant_counts = {split: split_pair_counts(args.target_variant, split) for split in ("train", "val", "test")}
    summary = {
        "status": "ready_for_training" if next_action == "launch_or_monitor_r171_guarded_training" else "not_ready_for_training",
        "next_action": next_action,
        "r168": {
            "worklist_exists": args.r168_worklist.exists(),
            "reviewed_csv_exists": args.r168_reviewed.exists(),
            "worklist": str(args.r168_worklist),
            "reviewed_csv": str(args.r168_reviewed),
        },
        "r172": {
            "summary_exists": r172 is not None,
            "status": r172.get("status") if r172 else None,
            "created_variant": r172.get("created_variant") if r172 else None,
            "preview_gate": (r172.get("preview", {}) or {}).get("gate_pass_if_applied") if r172 else None,
        },
        "r170": {
            "status_exists": r170 is not None,
            "status": r170.get("status") if r170 else None,
            "errors": r170.get("errors") if r170 else None,
            "target_path": r170.get("target_path") if r170 else str(args.target_variant),
            "target_exists": variant_exists,
            "variant_counts": variant_counts,
        },
        "r171": {
            "launcher_check_exists": r171 is not None,
            "launcher_ok": launcher_ok,
            "errors": r171.get("errors") if r171 else None,
        },
        "r173": {
            "summary_exists": r173 is not None,
            "status": r173.get("status") if r173 else None,
            "target_met": r173.get("target_met") if r173 else None,
            "new_best": r173.get("new_best") if r173 else None,
            "clean_test_v2": r173.get("clean_test_v2") if r173 else None,
        },
        "policy": {
            "success_eval_remains": "TSRS_RSNA-Epiphysis_clean_test_v2/test",
            "do_not_use_reannotated_test": True,
            "do_not_train_on_clean_test_v2": True,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
