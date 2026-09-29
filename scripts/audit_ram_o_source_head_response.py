#!/usr/bin/env python3
"""Read-only response audit for the matched RAM O source-head arms.

This module is a thin, explicit adaptation of
``audit_ram_source_correctability_response.py`` v4.  The fixed C0 model is
verified first with its original state.  Only after all 97 C0 contexts have
been checked is the zero-initialized source-head adapter attached.  The two
new arm checkpoints are then loaded strictly, including
``o_source_adapter.weight``:

``OHEAD_ZERO``
    source-head O is zero; the refiner O input is also zero.
``OHEAD_REAL``
    source-head O is the frozen image-only generator probability; the refiner
    O input remains zero.

No optimizer is constructed and no parameter is updated.  GT is read only
after each image-only forward, exactly as in the v4 audit.  The output keeps
the v4 full14 source/step1/step2 counts and matched negative controls while
adding a compact source-head response trace to each arm record.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

try:  # Keep helper tests importable in the same light-weight mode as v4.
    import torch
except Exception:  # pragma: no cover - endpoint execution requires PyTorch.
    torch = None  # type: ignore[assignment]

try:
    import audit_ram_source_correctability_response as base
except ModuleNotFoundError:  # Allow ``import scripts.audit_*`` from repo root.
    from scripts import audit_ram_source_correctability_response as base

try:
    from ram_o_source_head_adapter import attach_o_source_head
except ModuleNotFoundError:  # Allow package-style imports from repo root.
    try:
        from scripts.ram_o_source_head_adapter import attach_o_source_head
    except Exception:  # pragma: no cover - runtime reports a clearer error.
        attach_o_source_head = None  # type: ignore[assignment]
except Exception:  # pragma: no cover - runtime reports a clearer error.
    attach_o_source_head = None  # type: ignore[assignment]


ARMS = ("OHEAD_ZERO", "OHEAD_REAL")
STATES = base.STATES
EXPECTED_SPLIT_COUNTS = base.EXPECTED_SPLIT_COUNTS
ORACLE_LABEL = base.ORACLE_LABEL
PROBABILITY_THRESHOLD = base.PROBABILITY_THRESHOLD
SOURCE_CUTOFF = base.SOURCE_CUTOFF
SOURCE_CROSSING = base.SOURCE_CROSSING
CaseContext = base.CaseContext

# The v4 helpers resolve arm paths from this module-global value in their
# defining module.  Set it once so the existing lock/path/hash machinery uses
# the new pair without copying or editing the locked audit.
base.ARMS = ARMS


def _require_torch() -> Any:
    if torch is None or attach_o_source_head is None:
        raise RuntimeError("PyTorch and ram_o_source_head_adapter are required")
    return torch


def _load_pair_into_c0(model: Any, refiner: Any, checkpoint: Path, legacy: Any) -> Mapping[str, Any]:
    """Load the unmodified C0 pair before the adapter exists."""

    return base.load_pair_into(model, refiner, checkpoint, legacy)


def load_arm_pair_into(
    model: Any,
    refiner: Any,
    checkpoint: Path,
    legacy: Any,
) -> Mapping[str, Any]:
    """Strict-load one trained O source-head arm.

    The adapter must already be registered.  Requiring the new state key
    explicitly makes a stale pre-adapter checkpoint fail instead of silently
    producing a misleading response audit.
    """

    _require_torch()
    payload = base._load_checkpoint(checkpoint)
    model_state = payload["model"]
    if "o_source_adapter.weight" not in model_state:
        raise ValueError(
            f"O source-head arm checkpoint has no registered adapter weight: {checkpoint}"
        )
    model.load_state_dict(model_state, strict=True)
    # The matched OHEAD arm is a paired v3.1 checkpoint.  Do not apply the
    # v4 pre-explicit-O compatibility port here: accepting it would weaken
    # the arm's strict model/refiner lock and could hide a mismatched pair.
    refiner.load_state_dict(payload["refiner"], strict=True)
    model.eval().requires_grad_(False)
    refiner.eval().requires_grad_(False)
    return payload


def _freeze_generator(generator: Any) -> str:
    """Freeze/eval the image-only generator and return a state digest."""

    _require_torch()
    if not isinstance(generator, torch.nn.Module):
        raise TypeError("load_components must return an image-only generator module")
    generator.eval().requires_grad_(False)
    if any(parameter.requires_grad for parameter in generator.parameters()):
        raise RuntimeError("image-only generator was not frozen")
    return base._module_state_digest(generator)


def _state_snapshot(module: Any) -> dict[str, Any]:
    _require_torch()
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_original_state_preserved(module: Any, snapshot: Mapping[str, Any]) -> None:
    """Check that adapter registration did not alter any C0 state entry."""

    current = module.state_dict()
    for name, expected in snapshot.items():
        if name not in current or not torch.equal(current[name], expected):
            raise RuntimeError(f"C0 state changed while registering source-head adapter: {name}")


def attach_after_c0(
    model: Any,
    generator: Any,
    *,
    trace_sink: Any = None,
) -> Any:
    """Attach the adapter after C0 verification without advancing RNG."""

    _require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(generator, torch.nn.Module):
        raise TypeError("generator must be a torch module")
    # Conv2d construction samples an initial tensor before the adapter zeroes
    # it.  Restore the audit RNG stream so registration itself is read-only.
    rng = base._capture_rng()
    try:
        attachment = attach_o_source_head(
            model, generator, use_real_o=False, trace_sink=trace_sink
        )
    finally:
        base._restore_rng(rng)
    model.eval().requires_grad_(False)
    if not torch.equal(
        attachment.adapter.weight.detach(),
        torch.zeros_like(attachment.adapter.weight.detach()),
    ):
        raise RuntimeError("source-head adapter must start at exact zero")
    return attachment


def forward_full14(model: Any, refiner: Any, image: Any, legacy: Any):
    """Forward image-only input through source, step1 and step2.

    This delegates to v4, whose API has no target/GT argument and whose
    explicit refiner O input is ``zeros_like(image)``.  When an attachment is
    present, its model pre-hook computes source-head O from the image only.
    """

    return base.forward_full14(model, refiner, image, legacy)


def _c0_trace_row(context: CaseContext) -> dict[str, Any]:
    row = base._c0_trace_row(context)
    row["O_channel"] = "zeros_like(image) to refiner; no source-head adapter"
    row["refiner_O_channel"] = "zeros_like(image)"
    return row


def _arm_case_row(
    context: CaseContext,
    arm: str,
    target: np.ndarray,
    valid: np.ndarray,
    source: np.ndarray,
    states: list[np.ndarray],
    deltas: list[np.ndarray],
    telemetry: Mapping[str, Any],
    attachment: Any,
) -> dict[str, Any]:
    row = base._case_row(
        context, arm, target, valid, source, states, deltas, telemetry
    )
    real = arm == "OHEAD_REAL"
    row["O_channel"] = (
        "image_only_generator_probability at source head"
        if real else "zeros at source head"
    )
    row["refiner_O_channel"] = "zeros_like(image)"
    row["o_source_head"] = {
        "attached": True,
        "injection": "source_head_only",
        "use_real_o": real,
        "generator_frozen": True,
        "trace": None if attachment.last_trace is None else dict(attachment.last_trace),
    }
    row["no_gt_in_forward"] = True
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True, help="locked O source-head response-audit JSON")
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True, help="new isolated output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out}")
    lock_path = args.lock.resolve()
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(lock, Mapping):
        raise ValueError("response lock must be a JSON object")

    inputs = base.resolve_inputs(lock_path, lock)
    arm_paths = inputs["arms"]
    required = {
        "c0": inputs["c0"],
        "r332": inputs["r332"],
        "overlap": inputs["overlap"],
        "selection": inputs["selection"],
        **{f"arm_{arm}": path for arm, path in arm_paths.items()},
    }
    lock_report = base.verify_input_hashes(lock_path, lock, required)
    if "test" in str(lock.get("dataset_split", "")).lower() or lock.get("test_used"):
        raise ValueError("test split/use is forbidden in the response audit")
    if any("TSRS" in str(path).upper() for path in required.values()):
        raise ValueError("TSRS inputs are forbidden in the response audit")

    update_cases, heldout_cases = base._selection_lists(inputs["selection"])
    selection_provenance = base._validate_selection_provenance(
        inputs["selection"], inputs["c0"]
    )
    device = base._resolve_device(args.device)
    _require_torch()
    previous_threads = torch.get_num_threads()
    rng_original = base._capture_rng()
    torch.set_num_threads(64)
    seed = int(inputs["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng_before = base._capture_rng()
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    trace_path = out / "TRAINING_TRACE_OR_EQUIV.jsonl"

    limitations: list[str] = [
        "diagnostic counts only; official Overall/Overlap/Pair metrics are not computed here",
        "test split is not loaded or evaluated",
        "H0 and oracle masks are fixed C0 diagnostics, never model results",
        "GT is read after each image-only forward; no GT tensor is passed to model/refiner",
        "O enters the source head only: OHEAD_ZERO uses zero O and OHEAD_REAL uses the frozen image-only generator probability",
        "the refiner receives explicit zeros_like(image) O in both arms",
        "C0 is fully verified before adapter registration; each arm is strict-loaded with o_source_adapter.weight",
    ]
    protocol: dict[str, Any] = {}
    references: dict[str, dict[str, Any]] = {}
    arm_counts: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    arm_state_records: dict[str, dict[str, Any]] = {}
    datasets: dict[str, Any] = {}
    contexts: dict[str, list[CaseContext]] = {}
    state_observations: dict[str, Any] = {}
    legacy = None
    model = refiner = generator = attachment = None
    generator_digest: str | None = None

    try:
        import run_v31_ram_mature as legacy_module

        legacy = legacy_module
        datasets["train"] = base._load_dataset(
            legacy, inputs["dataset"], "train", inputs["divisor"], 425
        )
        datasets["val"] = base._load_dataset(
            legacy, inputs["dataset"], "val", inputs["divisor"], 69
        )
        selected = base._validate_case_sets(
            datasets["train"], datasets["val"], update_cases, heldout_cases
        )
        protocol["dataset"] = {
            "root": str(inputs["dataset"]),
            "split_counts": {"train": len(datasets["train"]), "val": len(datasets["val"])},
            "selected_counts": {key: len(value) for key, value in selected.items()},
            "padding_divisor": inputs["divisor"],
            "augmentation": False,
            "dtype": "float32",
        }
        protocol["selection_and_schedule"] = selection_provenance
        index_maps = {split: base._case_indices(dataset) for split, dataset in datasets.items()}
        for split, cases in (("train", update_cases + heldout_cases), ("val", selected["val"])):
            digest, missing = base._digest_h0_manifest(inputs["oracle"], split, cases)
            if missing:
                raise FileNotFoundError(f"missing packed H0 caches for {split}: {missing[:5]}")
            expected_digest = base._declared_manifest_digest(lock, split)
            if expected_digest is not None and digest != expected_digest:
                raise ValueError(f"H0 manifest digest mismatch for {split}: {digest} != {expected_digest}")
            protocol.setdefault("oracle", {})[split] = {
                "root": str(inputs["oracle"]),
                "selected_cache_count": len(cases),
                "selected_cache_digest": digest,
                "declared_manifest_digest_checked": expected_digest is not None,
            }
        if base._declared_manifest_digest(lock, "train") is None or base._declared_manifest_digest(lock, "val") is None:
            limitations.append(
                "per-cache H0 files were existence-checked and hashed; no lock-wide H0 manifest digest was declared for at least one split"
            )

        overlap_sha = base._expected_for_path(lock, inputs["overlap"], ("overlap",)) or base.sha256_file(inputs["overlap"])
        _, model, refiner, generator, backbone_mapping, refiner_mapping = legacy.load_components(
            SimpleNamespace(r332=inputs["r332"], overlap=inputs["overlap"], overlap_sha=overlap_sha),
            device,
        )
        generator_digest = _freeze_generator(generator)
        state_observations["architecture"] = "v31"
        state_observations["backbone_mapping"] = backbone_mapping
        state_observations["refiner_mapping"] = refiner_mapping
        state_observations["generator"] = {
            "frozen": True,
            "eval": not generator.training,
            "state_digest_before_audit": generator_digest,
        }

        # C0 is loaded and audited while the model still has its original
        # state_dict.  This ordering is a required part of the protocol.
        c0_payload = _load_pair_into_c0(model, refiner, inputs["c0"], legacy)
        c0_model_digest = base._module_state_digest(model)
        c0_refiner_digest = base._module_state_digest(refiner)
        c0_model_snapshot = _state_snapshot(model)
        protocol["c0"] = {
            "path": str(inputs["c0"]),
            "sha256": base.sha256_file(inputs["c0"]),
            "updates_in_payload": c0_payload.get("updates"),
            "model_state_digest": c0_model_digest,
            "refiner_state_digest": c0_refiner_digest,
            "adapter_attached": False,
            "optimizer_created": False,
            "gt_used_in_forward": False,
        }

        train_index = index_maps["train"]
        val_index = index_maps["val"]
        context_plan = (
            ("train", "train-update", selected["train-update"], train_index),
            ("train", "train-heldout", selected["train-heldout"], train_index),
            ("val", "val", selected["val"], val_index),
        )
        with trace_path.open("w", encoding="utf-8") as trace:
            # This pass deliberately has no attachment.  The sample target
            # and H0 cache are read only after each image-only C0 forward.
            for split, subset, cases, index_map in context_plan:
                rows: list[CaseContext] = []
                for case in cases:
                    if time.monotonic() - started > inputs["max_seconds"]:
                        raise TimeoutError("response audit runtime budget exceeded during C0 verification")
                    index = index_map[case]
                    sample = datasets[split][index]
                    image = sample["image"]
                    c0_source, c0_states, _ = forward_full14(model, refiner, image, legacy)
                    target_shape = tuple(int(x) for x in np.asarray(sample["mask"].shape))
                    cache = base.load_h0_cache(
                        base._h0_path(inputs["oracle"], split, case), target_shape
                    )
                    context = base._make_c0_context(
                        split=split, subset=subset, index=index, case=case, image=image,
                        sample=sample, c0_source=c0_source, c0_states=c0_states, cache=cache,
                    )
                    rows.append(context)
                    base._write_trace(trace, _c0_trace_row(context))
                contexts[subset] = rows
                references[subset] = {
                    "cases": len(rows),
                    "h0_member_pixels": int(sum(int(ctx.mask("h0").sum()) for ctx in rows)),
                    "h0_unique_pixels": int(sum(int(ctx.mask("h0").any(axis=0).sum()) for ctx in rows)),
                    "c0_global_overlap_fn_pixels": int(sum(int(ctx.mask("c0_global_fn").sum()) for ctx in rows)),
                    "oracle_verification_passed": True,
                }

            if (base._module_state_digest(model), base._module_state_digest(refiner)) != (
                c0_model_digest, c0_refiner_digest
            ):
                raise RuntimeError("C0 model/refiner state changed during original-model verification")

            # Only now is the source-head adapter registered.  Its generator
            # is frozen and the registration restores RNG exactly.
            _assert_original_state_preserved(model, c0_model_snapshot)
            attachment = attach_after_c0(model, generator)
            _assert_original_state_preserved(model, c0_model_snapshot)
            protocol["source_head_adapter"] = {
                "attached_after_c0_contexts": True,
                "module_name": "o_source_adapter",
                "weight_shape": list(attachment.adapter.weight.shape),
                "initial_weight_zero": True,
                "generator_frozen": True,
                "generator_state_digest": generator_digest,
                "refiner_o_input": "zeros_like(image)",
            }

            # Each arm starts from its own immutable strict checkpoint.  The
            # only runtime switch is attachment.use_real_o.
            for arm in ARMS:
                arm_payload = load_arm_pair_into(model, refiner, arm_paths[arm], legacy)
                attachment.use_real_o = arm == "OHEAD_REAL"
                attachment.clear_o()
                model_digest = base._module_state_digest(model)
                refiner_digest = base._module_state_digest(refiner)
                arm_state_records[arm] = {
                    "path": str(arm_paths[arm]),
                    "sha256": base.sha256_file(arm_paths[arm]),
                    "updates_in_payload": arm_payload.get("updates"),
                    "strict_model_load": True,
                    "registered_adapter_key": "o_source_adapter.weight",
                    "use_real_o": attachment.use_real_o,
                    "model_state_digest_after_load": model_digest,
                    "refiner_state_digest_after_load": refiner_digest,
                    "optimizer_created": False,
                }
                arm_counts[arm] = {}
                for subset in ("train-update", "train-heldout", "val"):
                    aggregate = base._new_aggregate()
                    split = "val" if subset == "val" else "train"
                    protocol.setdefault("gate_tanh_hook_cases", {})[subset] = contexts[subset][0].case
                    for case_number, context in enumerate(contexts[subset]):
                        probe = base.SaturationProbe(refiner) if case_number == 0 else None
                        try:
                            sample = datasets[split][context.index]
                            source, states, deltas = forward_full14(model, refiner, sample["image"], legacy)
                            target = np.asarray(sample["mask"].numpy() > 0.5, dtype=bool)
                            valid = np.asarray(sample["valid"].numpy()[0], dtype=bool)
                            if context.case != str(sample["case"]):
                                raise ValueError(f"dataset case order changed for {split}/{context.case}")
                            telemetry = (
                                probe.snapshot(reset=True)
                                if probe is not None
                                else {"available": False, "reason": "hook sampled on first fixed case per subset"}
                            )
                            row = _arm_case_row(
                                context, arm, target, valid, source, states, deltas,
                                telemetry, attachment,
                            )
                            base.add_arm_row(aggregate, row)
                            base._write_trace(trace, row)
                        finally:
                            if probe is not None:
                                probe.close()
                    arm_counts[arm][subset] = base.finalize_aggregate(aggregate)
                if (base._module_state_digest(model), base._module_state_digest(refiner)) != (
                    model_digest, refiner_digest
                ):
                    raise RuntimeError(f"arm state changed during read-only inference: {arm}")
                if base._module_state_digest(generator) != generator_digest:
                    raise RuntimeError(f"frozen image-only generator state changed during arm: {arm}")

        state_observations["arm_checkpoints"] = arm_state_records
        state_observations["generator"]["state_digest_after_audit"] = base._module_state_digest(generator)
    finally:
        base._restore_rng(rng_original)
        torch.set_num_threads(previous_threads)
        if attachment is not None:
            attachment.close()
        if generator is not None:
            del generator
        if torch is not None and device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "status": "COMPLETE",
        "experiment": "RAM_O_SOURCE_HEAD_RESPONSE_AUDIT",
        "diagnostic_label": ORACLE_LABEL,
        "model_result": False,
        "official_metrics_computed": False,
        "test_used": False,
        "optimizer_updates": 0,
        "read_only": True,
        "lock": lock_report,
        "lock_path": str(lock_path),
        "inputs": {
            "dataset_root": str(inputs["dataset"]),
            "oracle_h0_root": str(inputs["oracle"]),
            "selection_and_schedule": str(inputs["selection"]),
            "c0_checkpoint": {"path": str(inputs["c0"]), "sha256": base.sha256_file(inputs["c0"])},
            "arms": {arm: {"path": str(path), "sha256": base.sha256_file(path)} for arm, path in arm_paths.items()},
            "r332": {"path": str(inputs["r332"]), "sha256": base.sha256_file(inputs["r332"])},
            "overlap": {"path": str(inputs["overlap"]), "sha256": base.sha256_file(inputs["overlap"])},
        },
        "scope": {
            "splits": ["train-update", "train-heldout", "val"],
            "split_counts": EXPECTED_SPLIT_COUNTS,
            "selected_case_counts": {subset: len(value) for subset, value in selected.items()},
            "members": 14,
            "states": list(STATES),
            "steps": 2,
            "dtype": "float32",
            "cpu_threads": 64,
            "threshold": PROBABILITY_THRESHOLD,
            "source_locked_cutoff": SOURCE_CUTOFF,
            "forward": "run_v31_ram_mature; source input image+zero seam; source-head O only; iterative_refine_o full14 with refiner O zeros_like(image)",
            "gt_used_in_forward": False,
            "h0": "packed oracle cache copied only after C0 final mask/source threshold exact verification",
            "control": "same member, GT-negative, binned C0 source own probability and other-member max; fixed per case independent of arm",
            "arms": list(ARMS),
        },
        "protocol": protocol,
        "state_observations": state_observations,
        "c0_reference_by_split": references,
        "by_split_counts": {
            "c0_reference": references,
            "arms": {arm: dict(subsets) for arm, subsets in arm_counts.items()},
        },
        "limitations": limitations,
        "trace": {
            "path": str(trace_path),
            "schema": "TRAINING_TRACE_OR_EQUIV_V1_O_SOURCE_HEAD",
            "records": sum(len(rows) for rows in contexts.values()) * (1 + len(ARMS)),
            "includes_c0_oracle_verification": True,
            "includes_gate_tanh_hooks": True,
            "includes_source_head_response": True,
        },
        "rng": {
            "restored_after_audit": True,
            "original_digest": base._rng_digest(rng_original),
            "audit_start_digest": base._rng_digest(rng_before),
            "after_digest": base._rng_digest(base._capture_rng()),
            "preserved": base._rng_digest(rng_original) == base._rng_digest(base._capture_rng()),
        },
        "runtime_seconds": time.monotonic() - started,
    }
    base.write_json(out / "RESULT.json", result)
    base.write_json(out / "by_split_counts.json", result["by_split_counts"])
    print(json.dumps({"status": result["status"], "output": str(out), "runtime_seconds": result["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
