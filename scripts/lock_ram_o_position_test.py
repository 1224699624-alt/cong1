#!/usr/bin/env python3
"""Freeze completed RAM O-placement training and val endpoints before test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("ZERO", "INPUT_O", "REFINER_O")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sequence(path: Path):
    return [(row["case"], row["batch_sha256"])
            for line in path.read_text(encoding="utf-8").splitlines() if line
            for row in (json.loads(line),)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args(); root = a.root
    destination = root / "TEST_EVAL_LOCK.json"
    if destination.exists(): raise FileExistsError(destination)
    lock_path = root / "POSITION_RUN_LOCK_v1.json"
    lock = load(lock_path)
    if lock["status"] != "LOCKED_BEFORE_TRAIN": raise ValueError("training lock status")
    if (root / "TRAIN_STATUS.txt").read_text().strip() != "ALL_TRAIN_ARMS_COMPLETE":
        raise ValueError("training incomplete")
    if (root / "VAL_STATUS.txt").read_text().strip() != "ALL_VAL_ARMS_COMPLETE":
        raise ValueError("val incomplete")
    files = {
        "runner": root / "code_v1/run_ram_o_position_pair.py",
        "hooks": root / "code_v1/ram_o_position_hooks.py",
        "evaluator": root / "code_v1/evaluate_r329_ram_official_metrics_val.py",
        "dataset_loader": root / "code_v1/train_r325_ram_nnunet_native_resolution_overlap_iem.py",
    }
    for key, path in files.items():
        if sha(path) != lock[key + "_sha256"]:
            raise ValueError(f"{key} code hash mismatch")
    ckpts = {}
    reference = None
    for arm in ARMS:
        train = load(root / "results" / f"TRAIN_{arm}" / "TRAIN_RESULT.json")
        if train["status"] != "COMPLETE" or train["updates"] != lock["successful_updates_per_arm"]:
            raise ValueError(f"{arm} incomplete")
        ckpt = root / "results" / f"TRAIN_{arm}" / "final.pth"
        if sha(ckpt) != train["final_sha256"]:
            raise ValueError(f"{arm} checkpoint SHA mismatch")
        rows = sequence(root / "results" / f"TRAIN_{arm}" / "CASE_AND_BATCH_SEQUENCE.jsonl")
        if len(rows) != lock["successful_updates_per_arm"]:
            raise ValueError(f"{arm} sequence length")
        if reference is None:
            reference = rows
        elif rows != reference:
            raise ValueError(f"{arm} data/augmentation sequence differs")
        val_path = root / "results" / f"EVAL_VAL_{arm}" / "OFFICIAL.json"
        val = load(val_path)
        if val["cases"] != 69 or val["threshold"] != 0.5 or val["checkpoint_sha256"] != sha(ckpt):
            raise ValueError(f"{arm} val protocol/checkpoint mismatch")
        ckpts[arm] = {"path": str(ckpt), "sha256": sha(ckpt),
                      "val_result_sha256": sha(val_path), "updates": train["updates"]}
    c0_val_path = root / "results/EVAL_VAL_C0/OFFICIAL.json"
    c0_val = load(c0_val_path)
    if c0_val["cases"] != 69 or c0_val["threshold"] != 0.5 or c0_val["checkpoint_sha256"] != lock["c0_sha256"]:
        raise ValueError("C0 val mismatch")
    test_manifest = root / "TEST_FILE_SHA256.json"
    test_parity = root / "TEST_REMOTE_PARITY.json"
    parity = load(test_parity)
    if parity["status"] != "PASS" or parity["checked_files"] != 248 or parity["mismatch_count"]:
        raise ValueError("test mirror parity")
    result = {
        "status": "LOCKED_BEFORE_FIRST_TEST_INFERENCE",
        "training_lock_sha256": sha(lock_path),
        "C0_checkpoint_sha256": lock["c0_sha256"],
        "C0_val_result_sha256": sha(c0_val_path),
        "trained_checkpoints": ckpts,
        "same_augmented_case_batch_sequence": True,
        "sequence_rows": len(reference),
        "code_sha256": {key: sha(path) for key, path in files.items()},
        "test_manifest_sha256": sha(test_manifest),
        "test_parity_sha256": sha(test_parity),
        "test_protocol": "once each C0/ZERO/INPUT_O/REFINER_O; 124 cases; same R329 official metrics; step2; threshold .5; train top15 pairs; no selection after test",
        "test_used_when_written": False,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checkpoints": ckpts,
                      "test_manifest_sha256": result["test_manifest_sha256"]}))


if __name__ == "__main__": main()
