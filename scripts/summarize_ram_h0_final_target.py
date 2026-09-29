"""Verify the matched source-vs-final H0 run and summarize standard and H0 outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "overall_dsc", "overall_iou", "overall_nsd_2px", "overall_msd_px",
    "overlap_dsc", "overlap_iou", "overlap_nsd_2px", "overlap_msd_px",
    "pair_dsc", "pair_iou", "pair_nsd_2px", "pair_msd_px", "pair_msd_fail_rate",
)
ARM_REFERENCE = "SRC_F0S1"
ARM_VARIANT = "FIN_F0S1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def response_rows(path: Path, arm: str, subset: str):
    return [row for line in path.open(encoding="utf-8")
            if (row := json.loads(line)).get("record_type") == "arm_case_response"
            and row.get("arm") == arm and row.get("subset") == subset]


def response_aggregate(rows):
    n = sum(row["h0"]["member_pixels"] for row in rows)
    if not n:
        raise ValueError("empty fixed H0 response")
    weighted = lambda extract: sum(extract(row) * row["h0"]["member_pixels"] for row in rows) / n
    final_sum = lambda key: sum(row["states"]["step2"][key] for row in rows)
    return {
        "cases": len(rows), "H0_members": n,
        "source_ge_minus5": sum(row["source_crossing_minus5"]["source"]["at_or_above_minus5"] for row in rows),
        "final_H0_recovered": final_sum("h0_member_recovered"),
        "source_H0_mean": weighted(lambda row: row["states"]["source"]["logit_h0"]["mean"]),
        "step1_residual_H0_mean": weighted(lambda row: row["step_deltas"]["step1"]["h0"]["mean"]),
        "step2_residual_H0_mean": weighted(lambda row: row["step_deltas"]["step2"]["h0"]["mean"]),
        "final_H0_mean": weighted(lambda row: row["states"]["step2"]["logit_h0"]["mean"]),
        "member_FP_valid": final_sum("fp_all_valid"),
        "member_FN_valid": final_sum("member_fn_all"),
        "matched_negative_count": sum(row["matched_gt_negative_control"]["matched"] for row in rows),
        "matched_negative_FP": sum(row["matched_gt_negative_control"]["states"]["step2"]["control_member_false_positive"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--reference-response", type=Path, required=True)
    parser.add_argument("--variant-response", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    ref_manifest = read_json(args.reference_root / "run_manifest.json")
    var_manifest = read_json(args.variant_root / "run_manifest.json")
    if ref_manifest["status"] != "COMPLETE" or var_manifest["status"] != "COMPLETE":
        raise ValueError("both matched endpoints must complete")
    if ref_manifest["optimizer_steps"] != 384 or var_manifest["optimizer_steps"] != 384:
        raise ValueError("matched successful update count must be 384")
    if var_manifest["variant_id"] != ARM_VARIANT or var_manifest["loss_target"] != "two_step_final_full14":
        raise ValueError("variant label or loss target mismatch")
    if ref_manifest["checkpoint_sha256"] != var_manifest["checkpoint_sha256"]:
        raise ValueError("C0 checkpoint differs")
    if ref_manifest["schedule_sha256"] != var_manifest["schedule_sha256"]:
        raise ValueError("schedule differs")
    if ref_manifest["source_use_o"] or var_manifest["source_use_o"]:
        raise ValueError("explicit O must remain disabled")

    ref_trace = [json.loads(line) for line in (args.reference_root / "TRAINING_TRACE.jsonl").open(encoding="utf-8")]
    var_trace = [json.loads(line) for line in (args.variant_root / "TRAINING_TRACE.jsonl").open(encoding="utf-8")]
    if len(ref_trace) != 384 or len(var_trace) != 384:
        raise ValueError("trace length differs from update count")
    for ref, var in zip(ref_trace, var_trace):
        for key in ("step", "case", "batch_sha256", "fixed_indices", "denominator_c0", "h0_exposure"):
            if ref[key] != var[key]:
                raise ValueError(f"matched {key} differs at step {ref['step']}")
        if ref["AMP"]["skipped"] or var["AMP"]["skipped"]:
            raise ValueError("AMP skipped a matched step")
        if ref["lr"] != var["lr"] or ref["AMP"]["scale_before"] != var["AMP"]["scale_before"]:
            raise ValueError("LR or AMP scale differs")
        if not math.isclose(var["total_loss"], var["native_loss"] + var["extra_final_loss"], abs_tol=1e-6):
            raise ValueError("variant total loss does not equal native + final H0")
    if abs(ref_trace[0]["native_loss"] - var_trace[0]["native_loss"]) > 1e-6:
        raise ValueError("native first-batch loss differs at C0")
    if abs(ref_trace[0]["extra_source_loss"] - var_trace[0]["extra_source_loss_diagnostic"]) > 1e-6:
        raise ValueError("first-batch source diagnostic term differs")

    ref_val = read_json(args.reference_root / "FINAL_VALIDATION.json")
    var_val = read_json(args.variant_root / "FINAL_VALIDATION.json")
    if ref_val["status"] != "COMPLETE" or var_val["status"] != "COMPLETE":
        raise ValueError("original val69 evaluation incomplete")
    ref_data, var_data = ref_val["result"], var_val["result"]
    if ref_data["cases"] != var_data["cases"] or len(ref_data["cases"]) != 69:
        raise ValueError("val case order differs")
    ref_metrics, var_metrics = ref_data["flat"]["step2"], var_data["flat"]["step2"]
    results = [{"metric": metric, "reference": ref_metrics[metric], "final_target": var_metrics[metric],
                "delta": var_metrics[metric] - ref_metrics[metric], "dataset": "RAM_val69",
                "updates_each": 384, "evaluator": "original_R329", "test_used": False}
               for metric in METRICS]

    response = {}
    for subset in ("train-update", "train-heldout", "val"):
        reference = response_aggregate(response_rows(args.reference_response, ARM_REFERENCE, subset))
        variant = response_aggregate(response_rows(args.variant_response, ARM_VARIANT, subset))
        if reference["cases"] != variant["cases"] or reference["H0_members"] != variant["H0_members"]:
            raise ValueError(f"fixed H0/case scope differs for {subset}")
        response[subset] = {
            "reference": reference, "final_target": variant,
            "delta": {key: variant[key] - reference[key] for key in reference if key != "cases"},
        }

    per_case = []
    for case in ref_data["cases"]:
        a = ref_data["per_case_official"][case]["step2"]
        b = var_data["per_case_official"][case]["step2"]
        row = {"case": case}
        for label, key in (("Overall", "overall_instance_metrics"),
                           ("Overlap", "overlap_region_metrics"),
                           ("Pair", "overlap_pair_intersection_metrics")):
            for metric in ("dsc", "nsd_2px", "msd_px"):
                row[f"{label}_{metric}_reference"] = a[key][metric]
                row[f"{label}_{metric}_final_target"] = b[key][metric]
                row[f"{label}_{metric}_delta"] = b[key][metric] - a[key][metric]
        per_case.append(row)
    changes = np.asarray([row["Overlap_dsc_delta"] for row in per_case])
    rng = np.random.default_rng(4421)
    boot = np.mean(changes[rng.integers(0, len(changes), (2000, len(changes)))], axis=1)
    gate = {
        "overlap_dsc_improved": var_metrics["overlap_dsc"] > ref_metrics["overlap_dsc"],
        "overall_dsc_preserved": var_metrics["overall_dsc"] >= ref_metrics["overall_dsc"],
        "overall_iou_preserved": var_metrics["overall_iou"] >= ref_metrics["overall_iou"],
        "overlap_nsd_preserved": var_metrics["overlap_nsd_2px"] >= ref_metrics["overlap_nsd_2px"],
        "pair_dsc_preserved": var_metrics["pair_dsc"] >= ref_metrics["pair_dsc"],
        "overlap_gain_one_pp": var_metrics["overlap_dsc"] - ref_metrics["overlap_dsc"] >= 0.01,
    }
    gate["balanced_gate"] = all(gate[k] for k in (
        "overlap_dsc_improved", "overall_dsc_preserved", "overall_iou_preserved",
        "overlap_nsd_preserved", "pair_dsc_preserved",
    ))
    summary = {
        "status": "COMPLETE", "comparison": "same C0/schedule/384 updates; only fixed-H0 auxiliary target source vs two-step final",
        "reference": ARM_REFERENCE, "variant": ARM_VARIANT, "test_used": False,
        "input_and_optimizer_match": True, "initial_native_loss_match": True,
        "metrics": {row["metric"]: row for row in results}, "gate": gate,
        "H0_response": response,
        "paired_cases": {"improved": int((changes > 0).sum()), "worse": int((changes < 0).sum()),
                         "mean_overlap_dsc_delta": float(changes.mean()),
                         "bootstrap_95pct_descriptive": np.quantile(boot, [0.025, 0.975]).tolist(),
                         "limitation": "single seed and unknown patient clustering"},
    }
    args.out.mkdir(parents=True)
    (args.out / "RESULT_DIFF.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out / "STANDARD_RESULT_DIFF.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.out / "PER_CASE_DIFF.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case[0]))
        writer.writeheader()
        writer.writerows(per_case)
    print(json.dumps({"status": "COMPLETE", "overlap_dsc_delta": summary["metrics"]["overlap_dsc"]["delta"],
                      "gate": gate, "H0_val": response["val"]}, indent=2))


if __name__ == "__main__":
    main()
