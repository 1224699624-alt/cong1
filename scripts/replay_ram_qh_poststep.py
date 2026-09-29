"""Read-only exact-batch Q_H check for the first locked RAM optimizer step."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ram_source_correctability_2x2 as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--response-code", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.response_code.resolve()))
    response = importlib.import_module("audit_ram_source_correctability_response")
    if args.out.exists():
        raise FileExistsError(args.out)
    start = time.monotonic()
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text())
    steps, manifest_path, _ = runner._load_schedule(lock, lock_path)
    if manifest_path != args.schedule.resolve():
        raise ValueError("schedule path differs from the training lock")
    step = steps[0]
    if step["step"] != 1:
        raise ValueError("first schedule step is not one")
    batch, batch_hash = runner._load_step_payload(step)
    image = batch["image"].unsqueeze(0).cuda()
    h0 = batch["h0"].unsqueeze(0).cuda()
    if not torch.cuda.is_available() or not bool(h0.any()):
        raise RuntimeError("CUDA and nonempty H0 are required")
    output = {
        "status": "COMPLETE",
        "purpose": "same augmented batch Q_H before/after actual step 1, using immutable checkpoints",
        "optimizer_updates": 0,
        "test_used": False,
        "batch_sha256": batch_hash,
        "H0_members": int(h0.sum()),
        "C0_sha256": runner.sha256_file(runner._artifact_path(lock, lock_path, "checkpoint")),
        "arms": {},
    }
    locked_controls = None
    control_digest = None
    for arm in runner.ARMS:
        components = runner._prepare_components(lock, lock_path, arm, torch.device("cuda"))
        _, model, refiner, generator, optimizer, scaler, _, _, _, _ = components
        trace_path = args.runs / arm / "TRAINING_TRACE.jsonl"
        recorded = json.loads(trace_path.open().readline())
        if recorded["step"] != 1 or recorded["batch_sha256"] != batch_hash:
            raise ValueError(f"first trace row differs from locked batch for {arm}")
        checkpoint_path = args.runs / arm / "checkpoints" / "step_00001.pth"
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            before = runner._source_logits(model, image)
            q_before = F.softplus(-before.float())[h0].mean()
        controls, metadata = response.make_matched_controls(
            batch["mask"].bool().numpy(), batch["valid"][0].numpy(),
            batch["h0"].numpy(), before[0].float().cpu().numpy(),
        )
        if control_digest is None:
            control_digest = metadata["pair_sha256"]
            locked_controls = controls
        elif metadata["pair_sha256"] != control_digest:
            raise ValueError("C0 matched negative controls changed between arms")
        matched = locked_controls[locked_controls[:, 3] >= 0]
        if len(matched) == 0:
            raise ValueError("first batch has no matched GT-negative controls")
        member = torch.as_tensor(matched[:, 0], device="cuda", dtype=torch.long)
        control_y = torch.as_tensor(matched[:, 3], device="cuda", dtype=torch.long)
        control_x = torch.as_tensor(matched[:, 4], device="cuda", dtype=torch.long)
        negative_before = before[0, member, control_y, control_x].float()
        negative_bce_before = F.softplus(negative_before).mean()
        trace_q = recorded["Q_H"]["mean_softplus_minus_source"]
        if abs(float(q_before) - trace_q) > 1e-5:
            raise ValueError(f"pre-step Q_H mismatch for {arm}: {float(q_before)} != {trace_q}")
        snapshot = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(snapshot["model"], strict=True)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            after = runner._source_logits(model, image)
            q_after = F.softplus(-after.float())[h0].mean()
        negative_after = after[0, member, control_y, control_x].float()
        negative_bce_after = F.softplus(negative_after).mean()
        output["arms"][arm] = {
            "snapshot_sha256": runner.sha256_file(checkpoint_path),
            "Q_H_before": float(q_before),
            "Q_H_after": float(q_after),
            "Q_H_actual_delta": float(q_after - q_before),
            "Q_H_first_order_gradient_dot_delta": sum(recorded["Q_H"]["gradient_dot_actual_delta"].values()),
            "matched_negative_controls": int(len(matched)),
            "matched_negative_pair_sha256": control_digest,
            "matched_negative_exact_bin_count": int(matched[:, 5].sum()),
            "matched_negative_source_BCE_before": float(negative_bce_before),
            "matched_negative_source_BCE_after": float(negative_bce_after),
            "matched_negative_source_BCE_actual_delta": float(negative_bce_after - negative_bce_before),
            "matched_negative_source_logit_mean_before": float(negative_before.mean()),
            "matched_negative_source_logit_mean_after": float(negative_after.mean()),
            "AMP_scale": recorded["AMP"]["scale_after"],
        }
        del model, refiner, generator, optimizer, scaler, components
        torch.cuda.empty_cache()
    output["runtime_seconds"] = time.monotonic() - start
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
