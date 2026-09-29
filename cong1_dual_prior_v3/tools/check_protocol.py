#!/usr/bin/env python3
"""Validate V3 handoff configuration without launching training.

Requires PyYAML. Paths in the configuration are relative to --repo (default:
current working directory), not the configuration directory. Evidence flags
are declarations, not independently verified scientific results. The checker
also verifies native file hashes when checking pilot/formal readiness.

Exit 0: static preflight is valid, OR the requested stage passes its configured
checks. Preflight may still list blockers. Exit 2: invalid or not ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

METHOD = "input_seam__nnunet__output_overlap__unified_coordination"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARM_FLAGS = {
    "A": (False, False, False), "B": (True, False, True),
    "C": (False, True, True), "D": (True, True, True),
    "D_no_routing": (True, True, False),
}
EVIDENCE = (
    "baseline_identity", "four_prior_semantics", "arm_information_isolation",
    "overlap_independent_channel", "local_observation_and_valid",
    "routing_none_regression", "routing_aggregate_cap", "tsrs_real_batch",
    "ram_real_batch", "evaluator_failure_cases",
)


def get(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def positive_number(value: Any, zero_allowed: bool = False) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value >= 0 if zero_allowed else value > 0))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(config: dict[str, Any], stage: str, repo: Path) -> dict[str, Any]:
    errors: list[str] = []
    blockers: list[str] = []

    def equal(path: str, expected: Any) -> None:
        value = get(config, path)
        if value != expected or (isinstance(expected, bool) and not isinstance(value, bool)):
            errors.append(f"{path}: expected {expected!r}, got {value!r}")

    def ready(path: str) -> None:
        if get(config, path) is not True:
            blockers.append(f"{path}: true required")

    def required(path: str) -> None:
        if not present(get(config, path)):
            blockers.append(f"{path}: unresolved")

    def number(path: str, zero_allowed: bool = False) -> None:
        if not positive_number(get(config, path), zero_allowed):
            blockers.append(f"{path}: finite {'nonnegative' if zero_allowed else 'positive'} number required")

    def resolve(path_key: str, directory: bool = False) -> Path | None:
        value = get(config, path_key)
        if not isinstance(value, str) or not value.strip():
            blockers.append(f"{path_key}: local path unresolved")
            return None
        path = Path(value).expanduser()
        path = path if path.is_absolute() else repo / path
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            blockers.append(f"{path_key}: {'directory' if directory else 'file'} missing: {path}")
            return None
        return path

    equal("schema_version", "3.0")
    equal("method_line", METHOD)
    equal("scope.datasets", ["TSRS", "RAM"])
    equal("scope.backbone", "nnunet")
    for path in (
        "scope.cross_backbone_enabled", "scope.test_enabled",
        "scope.reconstruct_native_from_scratch", "scope.mandatory_oof",
        "scope.replace_overlap_with_completion", "run.overwrite_existing",
        "run.auto_long_training", "contracts.input_adapter_reads_overlap",
        "contracts.allow_legacy_coupled_context_in_main_table",
        "contracts.unknown_overlap_is_negative", "contracts.prior_probabilities_forced_disjoint",
        "contracts.current_student_selects_weak_roi", "loss.weak_overlap_is_true_gt",
        "quality_gate.undefined_metric_is_pass",
    ):
        equal(path, False)
    for path in (
        "scope.same_image_can_have_both_structures", "contracts.clean_ablation",
        "contracts.output_overlap_has_independent_channel", "contracts.zero_init_input",
        "contracts.zero_init_output", "contracts.bounded_input_residual",
        "contracts.bounded_output_residual", "contracts.no_gt_in_inference",
        "contracts.native_output_semantics_preserved", "contracts.target_and_weight_separate",
        "contracts.valid_masks_required", "loss.base_loss_same_within_dataset",
        "loss.native_deep_supervision_preserved", "loss.routing.none_coordinates_as_zero",
        "quality_gate.distance_failure_rate_must_not_increase", "quality_gate.single_checkpoint_all_metrics",
        "quality_gate.same_case_cohort", "quality_gate.no_posthoc_evaluator_changes",
    ):
        equal(path, True)
    equal("contracts.overlap_generator_input", "image_only")
    equal("loss.teacher", "common_frozen_native")
    equal("quality_gate.reference_comparisons", ["N0", "A"])
    equal("quality_gate.causal_reference", "A")
    equal("prior_adaptation.no_test", True)
    equal("prior_adaptation.dataset", "RAM")
    equal("loss.observed_overlap", "known_membership_only")
    equal("loss.weak_overlap", "observed_union_under_frozen_overlap_evidence")
    for arm, flags in ARM_FLAGS.items():
        for key, expected in zip(("input_seam", "output_overlap", "routing"), flags):
            equal(f"arms.{arm}.{key}", expected)
    if get(config, "run.mode") not in ("joint_finetune", "adapter_only"):
        errors.append("run.mode must be joint_finetune or adapter_only")
    for name in ("TSRS", "RAM"):
        equal(f"datasets.{name}.overlap.requested_role", "bone_overlap_prior")
    equal("datasets.TSRS.primary_structure", "seam")
    equal("datasets.RAM.primary_structure", "overlap")

    # Runtime readiness is reported even at preflight, without accessing weights.
    for path in (
        "run.hardware_verified", "run.protocol_locked", "loss.base_loss_resolved",
        "loss.mass_parameters_locked", "quality_gate.selection_rule_locked",
    ):
        ready(path)
    for path in ("source_lock.actual_repo_commit", "source_lock.task_sha256"):
        required(path)
    for path in ("contracts.input_residual_limit", "contracts.output_residual_limit",
                 "contracts.numerical_identity_tolerance", "loss.lambda_seam", "loss.lambda_overlap", "loss.lambda_preserve",
                 "loss.routing.per_aux_cap", "loss.routing.aggregate_cap"):
        number(path)
    number("loss.beta_weak", zero_allowed=True)
    number("quality_gate.absolute_noninferiority_margin", zero_allowed=True)
    for name in ("TSRS", "RAM"):
        prefix = f"datasets.{name}"
        for field in ("root", "train_manifest", "val_manifest", "native.path", "native.observed_sha256",
                      "seam.path", "seam.normalization_rule", "seam.support_threshold", "overlap.path",
                      "matched_training.optimizer", "matched_training.learning_rate_schedule",
                      "matched_training.augmentation", "matched_training.sampling_manifest_sha256",
                      "matched_training.base_loss_id"):
            required(f"{prefix}.{field}")
        for field in ("native.hash_verified", "native.prior_free_verified", "native.input_contract_verified",
                      "native.baseline_reproduced", "seam.provenance_verified", "seam.coverage_verified",
                      "overlap.provenance_verified", "overlap.generator_image_only_verified",
                      "overlap.teacher_frozen", "overlap.evidence_level_locked", "evaluation.evaluator_verified",
                      "evaluation.manifest_and_coordinates_verified"):
            ready(f"{prefix}.{field}")
        for field in ("matched_training.extra_updates_per_arm", "matched_training.batch_size"):
            number(f"{prefix}.{field}")
    ready("datasets.TSRS.missing_1833_reconciled")
    if get(config, "datasets.TSRS.overlap.native_membership_annotation_audit") != "PASS":
        blockers.append("datasets.TSRS.overlap.native_membership_annotation_audit: PASS audit required (PASS does not mean labels exist)")
    ready("datasets.RAM.overlap.annotation_observation_mask_verified")
    ready("datasets.RAM.evaluation.right_wrist_and_valid_verified")
    required("datasets.RAM.evaluation.pair_list_sha256")
    weak = get(config, "datasets.TSRS.overlap.weak_local_term_enabled")
    if weak is True:
        ready("datasets.TSRS.overlap.weak_support_validated")
        required("datasets.TSRS.overlap.weak_support_rule")
        number("loss.beta_weak")
    elif weak is not False:
        errors.append("datasets.TSRS.overlap.weak_local_term_enabled must be boolean")
    if get(config, "datasets.TSRS.overlap.source_mode") == "transferred_weak_probe":
        if get(config, "datasets.TSRS.overlap.independently_tsrs_overlap_supervised") is not False:
            errors.append("A transferred weak probe cannot declare independent TSRS overlap supervision")
    for key in EVIDENCE:
        if get(config, f"required_evidence.{key}") != "PASS":
            blockers.append(f"required_evidence.{key}: PASS required")
    required("required_evidence.evidence_directory")

    if stage in ("pilot", "formal"):
        ready(f"run.{stage}_enabled")
        if get(config, "run.mode") != "joint_finetune":
            blockers.append("Clean matched A/B/C/D training check requires joint_finetune; adapter_only uses N0 reference, not Native-Continue")
        seeds = get(config, "run.formal_seeds" if stage == "formal" else "run.paired_seeds")
        min_seeds = 3 if stage == "formal" else 1
        if (not isinstance(seeds, list) or len(seeds) < min_seeds
                or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds)
                or len(seeds) != len(set(seeds))):
            blockers.append(f"run seeds: at least {min_seeds} distinct integer seeds required")
        for name in ("TSRS", "RAM"):
            prefix = f"datasets.{name}"
            resolve(f"{prefix}.root", directory=True)
            for field in ("train_manifest", "val_manifest", "seam.path", "overlap.path"):
                resolve(f"{prefix}.{field}")
            native = resolve(f"{prefix}.native.path")
            observed = get(config, f"{prefix}.native.observed_sha256")
            expected = get(config, f"{prefix}.native.expected_sha256")
            if not isinstance(observed, str) or not SHA256_RE.fullmatch(observed):
                blockers.append(f"{prefix}.native.observed_sha256: 64 lowercase hex characters required")
            if native is not None:
                actual = file_sha256(native)
                if actual != observed:
                    blockers.append(f"{prefix}.native: actual file SHA differs from observed_sha256")
                if actual != expected and not present(get(config, f"{prefix}.native.approved_replacement_record")):
                    blockers.append(f"{prefix}.native: differs from historical hash; approved_replacement_record required, not automatic rejection of an audited alternative")
            updates = get(config, f"{prefix}.matched_training.extra_updates_per_arm")
            if positive_number(updates) and not isinstance(updates, int):
                blockers.append(f"{prefix}.matched_training.extra_updates_per_arm: integer required")
        resolve("required_evidence.evidence_directory", directory=True)
    if stage == "formal":
        blockers.append("Formal training is outside this handoff's current authorization; freeze a separately authorized formal protocol first")
    if errors:
        status = "INVALID_PROTOCOL"
    elif stage == "preflight":
        status = "TEMPLATE_VALID_WITH_BLOCKERS" if blockers else "PREFLIGHT_VALID_REQUIRES_EVIDENCE_REVIEW"
    else:
        status = "BLOCKED_NOT_READY" if blockers else "CONFIG_READY_REQUIRES_EVIDENCE_REVIEW"
    return {
        "stage": stage, "status": status, "static_errors": errors, "blockers": blockers,
        "training_launched": False,
        "limitation": "This checks configuration declarations and selected file existence/hashes, not label truth, source-code correctness, hardware safety, or experimental success.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "pilot", "formal"), default="preflight")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; does not overwrite")
    args = parser.parse_args()
    try:
        import yaml
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Configuration root must be a mapping")
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output}")
        report = check(config, args.stage, args.repo.resolve())
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        print(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        return 2 if report["static_errors"] or (args.stage != "preflight" and report["blockers"]) else 0
    except Exception as exc:
        print(json.dumps({"status": "CHECK_ERROR", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
