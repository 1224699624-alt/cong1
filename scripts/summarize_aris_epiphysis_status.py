#!/usr/bin/env python3
"""Summarize current ARIS epiphysis status from authoritative local artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TARGET_DICE = 0.9317660066557425
BEST_VALID_DICE = 0.9177231563529792
DEFAULT_OUT_JSON = Path("outputs/analysis/aris_epiphysis_current_status.json")
DEFAULT_OUT_MD = Path("research-workflow/refine-logs/ARIS_EPIPHYSIS_CURRENT_STATUS.md")
MANIFEST = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")
MANIFEST_CSV = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.csv")
GATE_STATUS = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_review_gate_status.json")
REVIEWED_CSV = Path("outputs/analysis/r134_label_protocol_review_manifest/review_package/r134_train_val_review_worklist_reviewed.csv")
REVIEW_HTML = Path("outputs/analysis/r134_label_protocol_review_manifest/review_package/R134_HUMAN_REVIEW_WORK_PACKAGE.html")
VARIANT_DIR = Path("data/raw_variants/TSRS_RSNA-Epiphysis_label_protocol_reviewed_v1")
VARIANT_META = VARIANT_DIR / "label_protocol_variant_metadata.json"
R140_LAUNCHER = Path("outputs/bridge_logs/run_r140_reviewed_variant_instance_sep.sh")
R140_LAUNCHER_CHECK = Path("outputs/analysis/r134_label_protocol_review_manifest/r140_reviewed_variant_launcher_check.json")
R140_MONITOR = Path("outputs/analysis/r140_reviewed_variant_dinov3_instance_sep_result_summary.json")
R143_METRICS = Path("outputs/analysis/r143_highres_medical_recipe_unet_clean_test_v2_metrics.json")
R144_SMOKE_METRICS = Path("outputs/analysis/r144_mask2former_hf_smoke_clean_test_v2_metrics.json")
R145_METRICS = Path("outputs/analysis/r145_mask2former_hf_small_overfit_clean_test_v2_metrics.json")
R146_METRICS = Path("outputs/analysis/r146_mask2former_hf_maskonly_readout_clean_test_v2_metrics.json")
R147_DIAGNOSTIC = Path("outputs/analysis/r147_mask2former_logits_threshold_diagnostic.json")
R148_DIAGNOSTIC = Path("outputs/analysis/r148_mask2former_rank_diagnostic.json")
R149_METRICS = Path("outputs/analysis/r149_mask2former_union_overfit_rescue_metrics.json")
R149B_METRICS = Path("outputs/analysis/r149b_mask2former_instance_overfit_control_metrics.json")
R150_METRICS = Path("outputs/analysis/r150_mask2former_union_overfit_rescue_metrics.json")
R151_METRICS = Path("outputs/analysis/r151_mask2former_union_small_gate_metrics.json")
R152_DECISION = Path("research-workflow/refine-logs/R152_POST_HF_DIRECTION_DECISION.md")
R153_PROBE = Path("outputs/analysis/r153_faithful_external_feasibility_probe.json")
R153_PROBE_MD = Path("research-workflow/refine-logs/R153_FAITHFUL_EXTERNAL_FEASIBILITY_PROBE.md")
R154_DECISION = Path("research-workflow/refine-logs/R154_POST_DETECTRON2_BLOCK_DIRECTION.md")
R155_METRICS = Path("outputs/analysis/r155_aspp_context_smoke_clean_test_v2_metrics.json")
R155_HISTORY = Path("outputs/timm_aspp/r155_aspp_context_smoke/history.json")
R156_DECISION = Path("research-workflow/refine-logs/R156_POST_ASPP_COLLAPSE_DIRECTION_DECISION.md")
R157_METRICS = Path("outputs/analysis/r157_hf_segformer_smoke_clean_test_v2_metrics.json")
R157_HISTORY = Path("outputs/hf_segformer/r157_hf_segformer_smoke/history.json")
R158_HISTORY = Path("outputs/hf_segformer/r158_hf_segformer_one_image_overfit/history.json")
R158_SUMMARY = Path("research-workflow/refine-logs/R158_HF_SEGFORMER_GATE_SUMMARY.md")
R159_PROBE = Path("outputs/analysis/r159_cuda_toolkit_env_probe.json")
R159_SUMMARY = Path("research-workflow/refine-logs/R159_CUDA_TOOLKIT_ENV_PROBE.md")
R160_METRICS = Path("outputs/analysis/r160_hf_upernet_convnext_smoke_clean_test_v2_metrics.json")
R160_HISTORY = Path("outputs/hf_semantic/r160_hf_upernet_convnext_smoke/history.json")
R160_SUMMARY = Path("research-workflow/refine-logs/R160_HF_UPERNET_CONVNEXT_SMOKE_SUMMARY.md")
R161_AUDIT = Path("outputs/analysis/r161_reannotated_variant_audit.json")
R161_BUILD = Path("outputs/analysis/r161_reannotated_trainval_only_variant_build.json")
R161_VARIANT = Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1")
R161_SUMMARY = Path("research-workflow/refine-logs/R161_REANNOTATED_TRAINVAL_ONLY_VARIANT.md")
R162_METRICS = Path("outputs/analysis/r162_reannotated_trainval_dinov3_instance_sep_clean_test_v2_metrics.json")
R162_HISTORY = Path("outputs/timm_instance_sep/r162_reannotated_trainval_dinov3_instance_sep/history.json")
R162_LAUNCHER_CHECK = Path("outputs/analysis/r162_reannotated_trainval_launcher_check.json")
R163_PAIR_SHIFT = Path("outputs/analysis/r163_reannotated_pair_shift.json")
R163_DECISION = Path("research-workflow/refine-logs/R163_REANNOTATED_SHIFT_DECISION.md")
R164_BUILD = Path("outputs/analysis/r164_filtered_reannotated_variant_build.json")
R164_CHECK = Path("outputs/analysis/r164_filtered_reannotated_variant_check.json")
R164_VARIANT = Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1")
R164_SUMMARY = Path("research-workflow/refine-logs/R164_FILTERED_REANNOTATED_VARIANT.md")
R165_METRICS = Path("outputs/analysis/r165_filtered_reannotated_dinov3_instance_sep_clean_test_v2_metrics.json")
R165_CONTROL_METRICS = Path("outputs/analysis/r165_filtered_reannotated_dinov3_instance_sep_original_test_metrics.json")
R165_HISTORY = Path("outputs/timm_instance_sep/r165_filtered_reannotated_dinov3_instance_sep/history.json")
R165_LAUNCHER_CHECK = Path("outputs/analysis/r165_filtered_reannotated_launcher_check.json")
R165_SUMMARY = Path("research-workflow/refine-logs/R165_FILTERED_REANNOTATED_RESULT.md")
R166_COMPLEMENTARITY = Path("outputs/analysis/r166_r110_r165_filtered_reannotated_complementarity.json")
R167_DECISION = Path("research-workflow/refine-logs/R167_NEXT_MATERIAL_GATE.md")
R168_SUMMARY = Path("outputs/analysis/r168_reannotation_protocol_review_package/r168_reannotation_protocol_review_summary.json")
R168_REVIEW_HTML = Path("outputs/analysis/r168_reannotation_protocol_review_package/R168_REANNOTATION_PROTOCOL_REVIEW.html")
R168_WORKLIST = Path("outputs/analysis/r168_reannotation_protocol_review_package/r168_reannotation_protocol_review_worklist.csv")
R168_REVIEWED_CSV = Path("outputs/analysis/r168_reannotation_protocol_review_package/r168_reannotation_protocol_review_reviewed.csv")
R174_READINESS = Path("outputs/analysis/r174_reannotation_protocol_readiness.json")
R176_AUDIT = Path("outputs/analysis/r176_mask_synced_bridge_boundary_audit.json")
R177_RESULT = Path("outputs/analysis/r177_boundary_preserving_separation_arbitrator_result_summary.json")
R178_METRICS = Path("outputs/analysis/r178_delete_only_bridge_suppressor_clean_test_v2_metrics.json")
R179_METRICS = Path("outputs/analysis/r179_boundary_gap_constrained_dinov3_instance_sep_clean_test_v2_metrics.json")
R179_HISTORY = Path("outputs/timm_instance_sep/r179_boundary_gap_constrained_dinov3_instance_sep/history.json")
R180_AUDIT = Path("outputs/analysis/r180_r110_r179_complementarity_audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current ARIS epiphysis pipeline status.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--skip-refresh", action="store_true", help="Do not rerun checker/monitor helper scripts.")
    return parser.parse_args()


def run(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def reviewed_csv_counts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row.get("split") == "train"]
    val = [row for row in rows if row.get("split") == "val"]
    train_reviewed = [row for row in train if str(row.get("human_decision", "")).strip()]
    train_confirmed = [
        row
        for row in train
        if str(row.get("human_decision", "")).strip() in {"label_ok_protocol_clear", "label_needs_correction"}
    ]
    return {
        "exists": True,
        "rows": len(rows),
        "train_rows": len(train),
        "val_rows": len(val),
        "reviewed_train": len(train_reviewed),
        "confirmed_train": len(train_confirmed),
    }


def metric_dice(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    mean = payload.get("mean", {})
    return mean.get("dice")


def r162_status(payload: dict[str, Any] | None) -> str | None:
    dice = metric_dice(payload)
    if dice is None:
        return None
    if dice > TARGET_DICE:
        return f"target_met:dice={dice:.6f}"
    if dice > BEST_VALID_DICE:
        return f"new_best_below_target:dice={dice:.6f}"
    return f"below_best:dice={dice:.6f}"


def r163_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    train = payload.get("paired", {}).get("train", {}).get("summary", {})
    val = payload.get("paired", {}).get("val", {}).get("summary", {})
    train_dice = train.get("label_dice", {}).get("mean")
    val_dice = val.get("label_dice", {}).get("mean")
    extras = train.get("extras_in_reannotated")
    if train_dice is None or val_dice is None:
        return "complete"
    if train_dice < 0.90 and val_dice > 0.93:
        return f"train_protocol_shift:train_dice={train_dice:.6f},val_dice={val_dice:.6f},extras={extras}"
    return f"complete:train_dice={train_dice:.6f},val_dice={val_dice:.6f},extras={extras}"


def r147_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    if payload.get("status") != "diagnostic_complete":
        return str(payload.get("status"))
    train = payload.get("splits", {}).get("train", {})
    prob = train.get("probability_summary", {})
    max_union = prob.get("max_union_max")
    if max_union is not None and max_union < 1e-3:
        return "mask_logits_collapsed"
    return "diagnostic_complete"


def r148_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    if payload.get("status") != "diagnostic_complete":
        return str(payload.get("status"))
    train = payload.get("splits", {}).get("train", {})
    topk_rows = train.get("topk_gt_fraction_summary", [])
    best_topk = max((row.get("mean", {}).get("dice", 0.0) for row in topk_rows), default=0.0)
    if best_topk < 0.5:
        return "spatial_rank_poor"
    return "spatial_rank_usable"


def overfit_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    status = str(payload.get("status"))
    train_dice = payload.get("train_eval", {}).get("mean", {}).get("dice")
    if train_dice is None:
        return status
    return f"{status}:train_dice={train_dice:.6f}"


def probe_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    return str(payload.get("decision", {}).get("status"))


def r166_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    summary = payload.get("summary", payload)
    better = summary.get("r165_better_count")
    oracle = summary.get("per_image_oracle_dice") or summary.get("per_image_oracle_mean_dice")
    if better is None and oracle is None:
        return "complete"
    return f"r165_better={better}/81,oracle={oracle:.6f}" if oracle is not None else f"r165_better={better}/81"


def r168_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    rows = payload.get("num_rows") or payload.get("worklist_rows") or payload.get("total_rows")
    train = payload.get("train_rows") or payload.get("num_train")
    val = payload.get("val_rows") or payload.get("num_val")
    panels = payload.get("num_panels") or payload.get("panels")
    if rows or train or val:
        return f"ready:rows={rows},train={train},val={val},panels={panels}"
    return str(payload.get("status", "ready"))


def r174_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    return f"{payload.get('status')}:{payload.get('next_action')}"


def r176_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    systems = payload.get("systems", payload.get("results", {}))
    r110 = systems.get("r110_r100_r108_patch_basic") if isinstance(systems, dict) else None
    if not r110:
        return "complete"
    mean = r110.get("mean", r110)
    return (
        f"r110:dice={mean.get('dice'):.6f},boundary_iou={mean.get('boundary_iou'):.6f},"
        f"component_error={mean.get('component_count_error'):.6f},false_bridge={mean.get('false_bridge_flag'):.6f}"
    )


def r177_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    clean = payload.get("clean_test_v2") or payload.get("mean") or {}
    status = payload.get("status")
    dice = clean.get("dice")
    if dice is not None:
        return f"{status}:dice={dice:.6f}"
    return str(status)


def r178_status(payload: dict[str, Any] | None) -> str | None:
    dice = metric_dice(payload)
    if dice is None:
        return None
    return f"dice={dice:.6f},target_margin={dice - TARGET_DICE:.6f}"


def r179_status(payload: dict[str, Any] | None) -> str | None:
    dice = metric_dice(payload)
    if dice is None:
        return None
    mean = payload.get("mean", {})
    return (
        f"dice={dice:.6f},boundary_iou={mean.get('boundary_iou'):.6f},"
        f"component_error={mean.get('component_count_error'):.6f},false_bridge={mean.get('false_bridge_flag'):.6f}"
    )


def r180_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    summary = payload.get("summary", payload)
    better = summary.get("r179_better_count")
    oracle = summary.get("per_image_oracle_mean_dice")
    if oracle is not None:
        return f"r179_better={better}/81,oracle={oracle:.6f}"
    return "complete"


def r155_status(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    mean = payload.get("mean", {})
    dice = mean.get("dice")
    recall = mean.get("recall")
    specificity = mean.get("specificity")
    boundary_iou = mean.get("boundary_iou")
    if recall == 1.0 and specificity == 0.0:
        return f"failed_all_foreground:dice={dice:.6f}"
    if dice is not None and dice < 0.2 and boundary_iou == 0.0:
        return f"failed_collapse:dice={dice:.6f}"
    if dice is not None:
        return f"complete:dice={dice:.6f}"
    return "complete"


def history_best(history: list[dict[str, Any]] | None, key: str = "val_dice") -> float | None:
    if not history:
        return None
    values = [row.get(key) for row in history if row.get(key) is not None]
    return float(max(values)) if values else None


def determine_next_action(
    gate: dict[str, Any] | None,
    reviewed_csv: dict[str, Any],
    variant_exists: bool,
    r140: dict[str, Any] | None,
    r143: dict[str, Any] | None,
    r144_smoke: dict[str, Any] | None,
    r145: dict[str, Any] | None,
    r146: dict[str, Any] | None,
    r147: dict[str, Any] | None,
    r148: dict[str, Any] | None,
    r149: dict[str, Any] | None,
    r149b: dict[str, Any] | None,
    r150: dict[str, Any] | None,
    r151: dict[str, Any] | None,
    r152_exists: bool,
    r153: dict[str, Any] | None,
    r154_exists: bool,
    r155: dict[str, Any] | None,
    r157: dict[str, Any] | None,
    r158_history: list[dict[str, Any]] | None,
    r159_probe: dict[str, Any] | None,
    r160: dict[str, Any] | None,
    r161_build: dict[str, Any] | None,
    r162_metrics: dict[str, Any] | None,
    r162_history: list[dict[str, Any]] | None,
    r179_metrics: dict[str, Any] | None,
    r180_audit: dict[str, Any] | None,
    r168_reviewed_csv_exists: bool,
) -> str:
    if r180_audit:
        if r168_reviewed_csv_exists:
            return "run_r172_preview_and_r170_build_for_r168_reviewed_variant"
        return "manual_r168_review_csv_or_new_model_family_required_do_not_continue_dinov3_loss_sweeps"
    if r179_metrics:
        return "run_r180_r110_r179_complementarity_audit"
    if r162_metrics:
        dice = metric_dice(r162_metrics)
        if dice is not None and dice > TARGET_DICE:
            return "run_result_to_claim_and_experiment_audit_for_r162"
        if R165_METRICS.exists():
            return "close_reannotated_direct_training_branch_choose_next_material_direction"
        return "prepare_guarded_r165_training_on_filtered_reannotated_variant"
    if r162_history:
        return "monitor_r162_until_final_clean_test_v2_metrics"
    if r161_build and r161_build.get("status") == "created":
        return "prepare_guarded_r162_training_on_reannotated_trainval_only_variant"
    if r160:
        return "close_hf_semantic_branch_choose_env_data_or_new_mechanism"
    if r159_probe:
        return "need_cuda_toolkit_env_or_corrected_labels_or_new_no_custom_cuda_route"
    if r158_history:
        return "pause_gpu_and_choose_cuda_toolkit_env_or_new_materially_different_route"
    if r157:
        return "start_r158_segformer_tiny_gate_or_close_segformer"
    if r155:
        return "start_r156_post_aspp_collapse_direction_decision"
    if r154_exists:
        return "start_r155_aspp_context_decoder_smoke"
    if r153 and r153.get("decision"):
        return "start_r154_post_detectron2_block_direction_decision"
    if r152_exists:
        return "start_r153_faithful_external_implementation_feasibility_probe"
    if r151 and r151.get("status") in {"overfit_failed", "overfit_pass"}:
        return "start_r152_post_hf_direction_decision"
    if r150 and r150.get("status") in {"overfit_failed", "overfit_pass"}:
        return "start_r151_mask2former_small_multi_image_union_source_gate"
    if r149b and r149b.get("status") in {"overfit_failed", "overfit_pass"}:
        return "start_r150_mask2former_union_source_overfit_rescue"
    if r149 and r149.get("status") == "overfit_pass":
        return "run_r149b_instance_target_control_before_full_run"
    if r149 and r149.get("status") == "overfit_failed":
        return "run_r149b_instance_target_control_or_close_hf_mask2former"
    if r148 and r148.get("status") == "diagnostic_complete":
        if r148_status(r148) == "spatial_rank_poor":
            return "start_r149_mask2former_target_loss_wiring_rescue"
        return "design_mask2former_low_probability_readout_rescue"
    if r147 and r147.get("status") == "diagnostic_complete":
        return "start_r148_mask2former_spatial_rank_diagnostic"
    if r146 and r146.get("status") == "smoke_complete":
        return "start_r147_mask2former_logits_threshold_diagnostic"
    if r145 and r145.get("status") == "smoke_complete":
        return "run_r146_mask2former_maskonly_readout_diagnostic"
    if r144_smoke and r144_smoke.get("status") == "smoke_complete":
        return "start_r145_mask2former_small_overfit_gate"
    r143_dice = metric_dice(r143)
    if r143_dice is not None:
        if r143_dice > TARGET_DICE:
            return "run_result_to_claim_and_experiment_audit"
        return "start_r144_external_architecture_feasibility_smoke"
    if r140 and r140.get("target_met"):
        return "run_result_to_claim_and_experiment_audit"
    if r140 and r140.get("status") in {"below_best", "new_best_below_target"}:
        return "close_r140_branch_and_start_r143_direction_decision"
    if variant_exists:
        if r140 and r140.get("status") == "waiting_for_metrics":
            return "launch_or_monitor_r140"
        return "monitor_r140_or_analyze_completed_result"
    if gate and gate.get("gate_pass"):
        return "build_isolated_reviewed_variant_then_launch_r140"
    if reviewed_csv.get("exists"):
        return "run_review_pipeline_preview_then_apply_if_gate_passes"
    return "complete_human_review_in_html_and_export_reviewed_csv"


def write_markdown(path: Path, status: dict[str, Any]) -> None:
    gate = status["r134_gate"]
    reviewed = status["reviewed_csv"]
    r140 = status["r140_monitor"]
    lines = [
        "# ARIS Epiphysis Current Status",
        "",
        f"- Target clean-test-v2 Dice: `{status['target_dice']}`",
        f"- Current valid best R110 Dice: `{status['best_valid_dice']}`",
        f"- Reviewed CSV exists: `{reviewed.get('exists')}`",
        f"- R134 gate pass: `{gate.get('gate_pass') if gate else None}`",
        f"- Reviewed train: `{gate.get('counts', {}).get('reviewed_train') if gate else None}`",
        f"- Confirmed train: `{gate.get('counts', {}).get('confirmed_train_for_variant') if gate else None}`",
        f"- Reviewed variant exists: `{status['reviewed_variant']['exists']}`",
        f"- R140 launcher check: `{status['r140_launcher_check'].get('ok') if status['r140_launcher_check'] else None}`",
        f"- R140 monitor status: `{r140.get('status') if r140 else None}`",
        f"- R143 clean-test-v2 Dice: `{metric_dice(status.get('r143_metrics'))}`",
        f"- R144 Mask2Former smoke: `{status.get('r144_smoke_metrics', {}).get('status') if status.get('r144_smoke_metrics') else None}`",
        f"- R145 clean-test-v2 Dice: `{metric_dice(status.get('r145_metrics'))}`",
        f"- R146 clean-test-v2 Dice: `{metric_dice(status.get('r146_metrics'))}`",
        f"- R147 Mask2Former logits diagnostic: `{r147_status(status.get('r147_diagnostic'))}`",
        f"- R148 Mask2Former rank diagnostic: `{r148_status(status.get('r148_diagnostic'))}`",
        f"- R149 union overfit rescue: `{overfit_status(status.get('r149_metrics'))}`",
        f"- R149b instance overfit control: `{overfit_status(status.get('r149b_metrics'))}`",
        f"- R150 union-source overfit rescue: `{overfit_status(status.get('r150_metrics'))}`",
        f"- R151 union small gate: `{overfit_status(status.get('r151_metrics'))}`",
        f"- R152 post-HF decision exists: `{status.get('r152_decision', {}).get('exists')}`",
        f"- R153 faithful external feasibility: `{probe_status(status.get('r153_probe'))}`",
        f"- R154 post-Detectron2 decision exists: `{status.get('r154_decision', {}).get('exists')}`",
        f"- R155 ASPP/context smoke: `{r155_status(status.get('r155_metrics'))}`",
        f"- R156 post-ASPP decision exists: `{status.get('r156_decision', {}).get('exists')}`",
        f"- R157 SegFormer smoke Dice: `{metric_dice(status.get('r157_metrics'))}`",
        f"- R158 SegFormer tiny-gate best val Dice: `{history_best(status.get('r158_history'))}`",
        f"- R159 CUDA toolkit probe exists: `{status.get('r159_probe') is not None}`",
        f"- R160 UPerNet smoke Dice: `{metric_dice(status.get('r160_metrics'))}`",
        f"- R161 train/val-only reannotated variant exists: `{status.get('r161_variant', {}).get('exists')}`",
        f"- R162 clean-test-v2 Dice: `{metric_dice(status.get('r162_metrics'))}`",
        f"- R162 status: `{r162_status(status.get('r162_metrics'))}`",
        f"- R162 history epochs: `{len(status.get('r162_history') or [])}`",
        f"- R163 reannotated shift status: `{r163_status(status.get('r163_pair_shift'))}`",
        f"- R164 filtered variant exists: `{status.get('r164_variant', {}).get('exists')}`",
        f"- R164 filtered variant check: `{status.get('r164_check', {}).get('ok') if status.get('r164_check') else None}`",
        f"- R165 clean-test-v2 Dice: `{metric_dice(status.get('r165_metrics'))}`",
        f"- R165 original-test Dice: `{metric_dice(status.get('r165_control_metrics'))}`",
        f"- R166 R110/R165 complementarity: `{r166_status(status.get('r166_complementarity'))}`",
        f"- R167 next-material gate exists: `{status.get('r167_decision', {}).get('exists')}`",
        f"- R168 review package: `{r168_status(status.get('r168_summary'))}`",
        f"- R168 reviewed CSV exists: `{status.get('r168_reviewed_csv', {}).get('exists')}`",
        f"- R174 readiness: `{r174_status(status.get('r174_readiness'))}`",
        f"- R176 mask-synced constraint audit: `{r176_status(status.get('r176_audit'))}`",
        f"- R177 arbitrator result: `{r177_status(status.get('r177_result'))}`",
        f"- R178 delete-only result: `{r178_status(status.get('r178_metrics'))}`",
        f"- R179 boundary/gap constrained result: `{r179_status(status.get('r179_metrics'))}`",
        f"- R180 R110/R179 complementarity: `{r180_status(status.get('r180_audit'))}`",
        f"- Next action: `{status['next_action']}`",
        "",
        "## Evidence Files",
        "",
        f"- Review HTML: `{REVIEW_HTML}`",
        f"- Reviewed CSV: `{REVIEWED_CSV}`",
        f"- Gate status: `{GATE_STATUS}`",
        f"- R140 launcher: `{R140_LAUNCHER}`",
        f"- R140 monitor summary: `{R140_MONITOR}`",
        f"- R143 metrics: `{R143_METRICS}`",
        f"- R144 smoke metrics: `{R144_SMOKE_METRICS}`",
        f"- R145 metrics: `{R145_METRICS}`",
        f"- R146 metrics: `{R146_METRICS}`",
        f"- R147 diagnostic: `{R147_DIAGNOSTIC}`",
        f"- R148 diagnostic: `{R148_DIAGNOSTIC}`",
        f"- R149 metrics: `{R149_METRICS}`",
        f"- R149b metrics: `{R149B_METRICS}`",
        f"- R150 metrics: `{R150_METRICS}`",
        f"- R151 metrics: `{R151_METRICS}`",
        f"- R152 decision: `{R152_DECISION}`",
        f"- R153 probe: `{R153_PROBE_MD}`",
        f"- R154 decision: `{R154_DECISION}`",
        f"- R155 metrics: `{R155_METRICS}`",
        f"- R155 history: `{R155_HISTORY}`",
        f"- R156 decision: `{R156_DECISION}`",
        f"- R157 metrics: `{R157_METRICS}`",
        f"- R157 history: `{R157_HISTORY}`",
        f"- R158 history: `{R158_HISTORY}`",
        f"- R158 summary: `{R158_SUMMARY}`",
        f"- R159 probe: `{R159_PROBE}`",
        f"- R159 summary: `{R159_SUMMARY}`",
        f"- R160 metrics: `{R160_METRICS}`",
        f"- R160 history: `{R160_HISTORY}`",
        f"- R160 summary: `{R160_SUMMARY}`",
        f"- R161 audit: `{R161_AUDIT}`",
        f"- R161 build: `{R161_BUILD}`",
        f"- R161 variant: `{R161_VARIANT}`",
        f"- R161 summary: `{R161_SUMMARY}`",
        f"- R162 metrics: `{R162_METRICS}`",
        f"- R162 history: `{R162_HISTORY}`",
        f"- R162 launcher check: `{R162_LAUNCHER_CHECK}`",
        f"- R163 pair-shift audit: `{R163_PAIR_SHIFT}`",
        f"- R163 decision: `{R163_DECISION}`",
        f"- R164 build: `{R164_BUILD}`",
        f"- R164 check: `{R164_CHECK}`",
        f"- R164 variant: `{R164_VARIANT}`",
        f"- R164 summary: `{R164_SUMMARY}`",
        f"- R165 metrics: `{R165_METRICS}`",
        f"- R165 original-test control metrics: `{R165_CONTROL_METRICS}`",
        f"- R165 history: `{R165_HISTORY}`",
        f"- R165 launcher check: `{R165_LAUNCHER_CHECK}`",
        f"- R165 summary: `{R165_SUMMARY}`",
        f"- R166 complementarity audit: `{R166_COMPLEMENTARITY}`",
        f"- R167 decision: `{R167_DECISION}`",
        f"- R168 package summary: `{R168_SUMMARY}`",
        f"- R168 review HTML: `{R168_REVIEW_HTML}`",
        f"- R168 worklist CSV: `{R168_WORKLIST}`",
        f"- R168 reviewed CSV: `{R168_REVIEWED_CSV}`",
        f"- R174 readiness audit: `{R174_READINESS}`",
        f"- R176 full audit: `{R176_AUDIT}`",
        f"- R177 result summary: `{R177_RESULT}`",
        f"- R178 metrics: `{R178_METRICS}`",
        f"- R179 metrics: `{R179_METRICS}`",
        f"- R179 history: `{R179_HISTORY}`",
        f"- R180 audit: `{R180_AUDIT}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    refresh_steps = {}
    if not args.skip_refresh:
        py = sys.executable
        refresh_steps["gate"] = run([
            py,
            "scripts/check_label_protocol_review_manifest.py",
            "--manifest",
            str(MANIFEST),
            "--csv",
            str(MANIFEST_CSV),
            "--output-json",
            str(GATE_STATUS),
        ])
        refresh_steps["r140_launcher"] = run([
            py,
            "scripts/check_r140_reviewed_variant_launcher.py",
            "--output-json",
            str(R140_LAUNCHER_CHECK),
        ])
        refresh_steps["r140_monitor"] = run([
            py,
            "scripts/monitor_r140_reviewed_variant_result.py",
            "--output-json",
            str(R140_MONITOR),
        ])

    gate = load_json(GATE_STATUS)
    reviewed = reviewed_csv_counts(REVIEWED_CSV)
    variant = {
        "exists": VARIANT_DIR.exists(),
        "path": str(VARIANT_DIR),
        "metadata_exists": VARIANT_META.exists(),
        "metadata": load_json(VARIANT_META),
    }
    r140_monitor = load_json(R140_MONITOR)
    r143_metrics = load_json(R143_METRICS)
    r144_smoke_metrics = load_json(R144_SMOKE_METRICS)
    r145_metrics = load_json(R145_METRICS)
    r146_metrics = load_json(R146_METRICS)
    r147_diagnostic = load_json(R147_DIAGNOSTIC)
    r148_diagnostic = load_json(R148_DIAGNOSTIC)
    r149_metrics = load_json(R149_METRICS)
    r149b_metrics = load_json(R149B_METRICS)
    r150_metrics = load_json(R150_METRICS)
    r151_metrics = load_json(R151_METRICS)
    r153_probe = load_json(R153_PROBE)
    r155_metrics = load_json(R155_METRICS)
    r155_history = load_json(R155_HISTORY)
    r157_metrics = load_json(R157_METRICS)
    r157_history = load_json(R157_HISTORY)
    r158_history = load_json(R158_HISTORY)
    r159_probe = load_json(R159_PROBE)
    r160_metrics = load_json(R160_METRICS)
    r160_history = load_json(R160_HISTORY)
    r161_audit = load_json(R161_AUDIT)
    r161_build = load_json(R161_BUILD)
    r162_metrics = load_json(R162_METRICS)
    r162_history = load_json(R162_HISTORY)
    r162_launcher_check = load_json(R162_LAUNCHER_CHECK)
    r163_pair_shift = load_json(R163_PAIR_SHIFT)
    r164_build = load_json(R164_BUILD)
    r164_check = load_json(R164_CHECK)
    r165_metrics = load_json(R165_METRICS)
    r165_control_metrics = load_json(R165_CONTROL_METRICS)
    r165_history = load_json(R165_HISTORY)
    r165_launcher_check = load_json(R165_LAUNCHER_CHECK)
    r166_complementarity = load_json(R166_COMPLEMENTARITY)
    r168_summary = load_json(R168_SUMMARY)
    r174_readiness = load_json(R174_READINESS)
    r176_audit = load_json(R176_AUDIT)
    r177_result = load_json(R177_RESULT)
    r178_metrics = load_json(R178_METRICS)
    r179_metrics = load_json(R179_METRICS)
    r179_history = load_json(R179_HISTORY)
    r180_audit = load_json(R180_AUDIT)
    launcher_check = load_json(R140_LAUNCHER_CHECK)
    status = {
        "target_dice": TARGET_DICE,
        "best_valid_dice": BEST_VALID_DICE,
        "r134_gate": gate,
        "reviewed_csv": reviewed,
        "review_html": {"exists": REVIEW_HTML.exists(), "path": str(REVIEW_HTML)},
        "reviewed_variant": variant,
        "r140_launcher": {"exists": R140_LAUNCHER.exists(), "path": str(R140_LAUNCHER)},
        "r140_launcher_check": launcher_check,
        "r140_monitor": r140_monitor,
        "r143_metrics": r143_metrics,
        "r144_smoke_metrics": r144_smoke_metrics,
        "r145_metrics": r145_metrics,
        "r146_metrics": r146_metrics,
        "r147_diagnostic": r147_diagnostic,
        "r148_diagnostic": r148_diagnostic,
        "r149_metrics": r149_metrics,
        "r149b_metrics": r149b_metrics,
        "r150_metrics": r150_metrics,
        "r151_metrics": r151_metrics,
        "r152_decision": {"exists": R152_DECISION.exists(), "path": str(R152_DECISION)},
        "r153_probe": r153_probe,
        "r154_decision": {"exists": R154_DECISION.exists(), "path": str(R154_DECISION)},
        "r155_metrics": r155_metrics,
        "r155_history": r155_history,
        "r156_decision": {"exists": R156_DECISION.exists(), "path": str(R156_DECISION)},
        "r157_metrics": r157_metrics,
        "r157_history": r157_history,
        "r158_history": r158_history,
        "r158_summary": {"exists": R158_SUMMARY.exists(), "path": str(R158_SUMMARY)},
        "r159_probe": r159_probe,
        "r159_summary": {"exists": R159_SUMMARY.exists(), "path": str(R159_SUMMARY)},
        "r160_metrics": r160_metrics,
        "r160_history": r160_history,
        "r160_summary": {"exists": R160_SUMMARY.exists(), "path": str(R160_SUMMARY)},
        "r161_audit": r161_audit,
        "r161_build": r161_build,
        "r161_variant": {"exists": R161_VARIANT.exists(), "path": str(R161_VARIANT)},
        "r161_summary": {"exists": R161_SUMMARY.exists(), "path": str(R161_SUMMARY)},
        "r162_metrics": r162_metrics,
        "r162_history": r162_history,
        "r162_launcher_check": r162_launcher_check,
        "r163_pair_shift": r163_pair_shift,
        "r163_decision": {"exists": R163_DECISION.exists(), "path": str(R163_DECISION)},
        "r164_build": r164_build,
        "r164_check": r164_check,
        "r164_variant": {"exists": R164_VARIANT.exists(), "path": str(R164_VARIANT)},
        "r164_summary": {"exists": R164_SUMMARY.exists(), "path": str(R164_SUMMARY)},
        "r165_metrics": r165_metrics,
        "r165_control_metrics": r165_control_metrics,
        "r165_history": r165_history,
        "r165_launcher_check": r165_launcher_check,
        "r165_summary": {"exists": R165_SUMMARY.exists(), "path": str(R165_SUMMARY)},
        "r166_complementarity": r166_complementarity,
        "r167_decision": {"exists": R167_DECISION.exists(), "path": str(R167_DECISION)},
        "r168_summary": r168_summary,
        "r168_review_html": {"exists": R168_REVIEW_HTML.exists(), "path": str(R168_REVIEW_HTML)},
        "r168_worklist": {"exists": R168_WORKLIST.exists(), "path": str(R168_WORKLIST)},
        "r168_reviewed_csv": {"exists": R168_REVIEWED_CSV.exists(), "path": str(R168_REVIEWED_CSV)},
        "r174_readiness": r174_readiness,
        "r176_audit": r176_audit,
        "r177_result": r177_result,
        "r178_metrics": r178_metrics,
        "r179_metrics": r179_metrics,
        "r179_history": r179_history,
        "r180_audit": r180_audit,
        "refresh_steps": refresh_steps,
    }
    status["next_action"] = determine_next_action(
        gate,
        reviewed,
        bool(variant["exists"]),
        r140_monitor,
        r143_metrics,
        r144_smoke_metrics,
        r145_metrics,
        r146_metrics,
        r147_diagnostic,
        r148_diagnostic,
        r149_metrics,
        r149b_metrics,
        r150_metrics,
        r151_metrics,
        R152_DECISION.exists(),
        r153_probe,
        R154_DECISION.exists(),
        r155_metrics,
        r157_metrics,
        r158_history,
        r159_probe,
        r160_metrics,
        r161_build,
        r162_metrics,
        r162_history,
        r179_metrics,
        r180_audit,
        R168_REVIEWED_CSV.exists(),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, status)
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
