#!/usr/bin/env python3
"""Audit matched RAM O-position training and aggregate locked val/test results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


ARMS = ("ZERO", "INPUT_O", "REFINER_O")
METRICS = ("overall_dsc", "overall_nsd_2px", "overlap_dsc", "overlap_nsd_2px",
           "overlap_msd_px", "pair_dsc", "pair_nsd_2px", "pair_msd_px")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--allow-missing-test", action="store_true")
    a = ap.parse_args()
    root = a.root
    lock = read_json(root / "POSITION_RUN_LOCK_v1.json")
    steps = lock["successful_updates_per_arm"]
    training = {arm: read_json(root / "results" / f"TRAIN_{arm}" / "TRAIN_RESULT.json") for arm in ARMS}
    sequences = {arm: read_jsonl(root / "results" / f"TRAIN_{arm}" / "CASE_AND_BATCH_SEQUENCE.jsonl") for arm in ARMS}
    trace = {arm: read_jsonl(root / "results" / f"TRAIN_{arm}" / "TRAINING_TRACE.jsonl") for arm in ARMS}
    checks = {}
    for arm in ARMS:
        checks[f"{arm}_complete"] = training[arm]["status"] == "COMPLETE" and training[arm]["updates"] == steps
        checks[f"{arm}_sequence_length"] = len(sequences[arm]) == steps
        checks[f"{arm}_trace_nonfinite"] = all(np.isfinite(row["loss"]) and row["success"] for row in trace[arm])
    reference = [(r["case"], r["batch_sha256"]) for r in sequences["ZERO"]]
    for arm in ARMS[1:]:
        checks[f"{arm}_same_batch_sequence"] = reference == [
            (r["case"], r["batch_sha256"]) for r in sequences[arm]]
    checks["zero_O_weights_unchanged"] = all(
        training["ZERO"]["final_adapter_l2"][name] == 0 for name in ("input", "refiner"))
    checks["input_active_only"] = (training["INPUT_O"]["final_adapter_l2"]["input"] > 0 and
                                   training["INPUT_O"]["final_adapter_l2"]["refiner"] == 0)
    checks["refiner_active_only"] = (training["REFINER_O"]["final_adapter_l2"]["refiner"] > 0 and
                                     training["REFINER_O"]["final_adapter_l2"]["input"] == 0)
    checks["test_evaluated_after_training"] = (root / "TRAIN_STATUS.txt").read_text().strip() == "ALL_TRAIN_ARMS_COMPLETE"

    results = {}
    for split in ("val", "test"):
        if a.allow_missing_test and split == "test":
            continue
        results[split] = {arm: read_json(root / "results" / f"EVAL_{split.upper()}_{arm}" / "OFFICIAL.json")
                          for arm in ("C0",) + ARMS}
        checks[f"{split}_same_cases"] = len({tuple(row["case"] for row in results[split][arm]["per_case"])
                                              for arm in results[split]}) == 1
        checks[f"{split}_case_count"] = all(results[split][arm]["cases"] == lock[f"{split}_cases"]
                                               for arm in results[split])
        checks[f"{split}_threshold"] = all(results[split][arm]["threshold"] == 0.5
                                              for arm in results[split])
    rows = []
    for split in results:
        for arm in results[split]:
            d = {"split": split, "arm": arm, "cases": results[split][arm]["cases"],
                 "checkpoint_sha256": results[split][arm]["checkpoint_sha256"]}
            for metric in METRICS:
                d[metric] = results[split][arm]["flat"][metric]
                d[f"delta_vs_ZERO_{metric}"] = (d[metric] - results[split]["ZERO"]["flat"][metric])
            rows.append(d)
    if rows:
        with (root / "POSITION_STANDARD_RESULTS.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    per_case = []
    for split in results:
        base = {row["case"]: row for row in results[split]["ZERO"]["per_case"]}
        for arm in ("INPUT_O", "REFINER_O"):
            for item in results[split][arm]["per_case"]:
                case = item["case"]
                old = base[case]
                d = {"split": split, "arm": arm, "case": case}
                for region in ("overall", "overlap", "pair"):
                    for metric in ("dsc", "nsd_2px", "msd_px"):
                        a_value, b_value = item[region].get(metric), old[region].get(metric)
                        d[f"delta_{region}_{metric}"] = a_value - b_value if a_value is not None and b_value is not None else None
                per_case.append(d)
    if per_case:
        with (root / "PER_CASE_POSITION_DIFF.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_case[0]))
            w.writeheader(); w.writerows(per_case)
    metadata = load_workbook("G:/gutou/RAM-W600/Metadata.xlsx", read_only=True, data_only=True).active
    patient = {str(row[0]): str(row[2]) for row in list(metadata.values)[1:]
               if row[0] is not None and row[2] is not None}
    bootstrap = {}
    rng = np.random.default_rng(4421)
    for split in results:
        bootstrap[split] = {}
        for arm in ("INPUT_O", "REFINER_O"):
            bootstrap[split][arm] = {}
            lhs = {x["case"]: x for x in results[split][arm]["per_case"]}
            rhs = {x["case"]: x for x in results[split]["ZERO"]["per_case"]}
            assert set(lhs) == set(rhs)
            by_patient = {}
            for case in lhs:
                key = patient.get(case[:-2])
                if key is None: raise ValueError(f"patient metadata missing {case}")
                by_patient.setdefault(key, []).append(case)
            patient_keys = sorted(by_patient)
            for region in ("overall", "overlap", "pair"):
                rows_patient = []
                for key in patient_keys:
                    cases = [c for c in by_patient[key]
                             if lhs[c][region]["n"] > 0 and
                             lhs[c][region]["dsc"] is not None and
                             rhs[c][region]["dsc"] is not None]
                    count = sum(lhs[c][region]["n"] for c in cases)
                    if not count:
                        continue
                    numerator = sum((lhs[c][region]["dsc"] - rhs[c][region]["dsc"])
                                    * lhs[c][region]["n"] for c in cases)
                    rows_patient.append((numerator, count))
                array = np.asarray(rows_patient, dtype=np.float64)
                observed = float(array[:, 0].sum() / array[:, 1].sum())
                draws = []
                for _ in range(2000):
                    sample = array[rng.integers(0, len(array), size=len(array))].sum(0)
                    draws.append(sample[0] / sample[1])
                bootstrap[split][arm][region + "_dsc"] = {
                    "patients": len(patient_keys), "delta_vs_ZERO": observed,
                    "paired_patient_bootstrap_95pct": np.quantile(draws, [.025, .975]).tolist(),
                    "note": "descriptive single-seed case/patient resampling; not a fresh blind test"}
    (root / "PATIENT_BOOTSTRAP_DIFF.json").write_text(json.dumps(bootstrap, indent=2) + "\n", encoding="utf-8")
    verdict = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "lock_sha256": hashlib.sha256((root / "POSITION_RUN_LOCK_v1.json").read_bytes()).hexdigest(),
               "train_seconds_total": sum(training[a]["seconds"] for a in ARMS),
               "split_results": list(results), "training": training}
    (root / "POSITION_DELIVERY_CHECKS.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": verdict["status"], "checks": len(checks),
                      "train_seconds_total": verdict["train_seconds_total"],
                      "splits": list(results)}))
    if verdict["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
