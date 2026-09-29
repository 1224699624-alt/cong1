"""Zero-update CUDA cost and gradient profile for the full14 final H0 loss."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import run_ram_h0_final_target as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    lock_path = args.lock.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    runner._verify_lock_hashes(lock, lock_path)
    steps, _, _ = runner._load_schedule(lock, lock_path)
    batch, batch_sha = runner._load_step_payload(steps[0])
    runner.seed_all(int(lock["seed"]))
    device = torch.device("cuda")
    _, model, refiner, generator, optimizer, scaler, named_groups, _, _, _ = \
        runner._prepare_components(lock, lock_path, "SRC_F0S1", device)
    image = batch["image"].unsqueeze(0).to(device)
    target = batch["mask"].float().unsqueeze(0).to(device)
    valid = batch["valid"].bool().unsqueeze(0).to(device)
    h0 = batch["h0"].bool().unsqueeze(0).to(device)
    indices = torch.tensor(batch["indices"], dtype=torch.long, device=device)
    original_selector = runner.legacy.choose_pair_instances
    runner.legacy.choose_pair_instances = lambda _p, _t, count: indices if count == 4 else None
    rows = []
    try:
        for repeat in range(3):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            with torch.cuda.amp.autocast(enabled=True):
                captured = []
                hook = model.register_forward_hook(lambda _m, _i, output: captured.append(output))
                try:
                    native, _, _, overlap, selected, _ = runner.legacy.loss_for_batch(
                        model, refiner, generator, image, target, valid, False
                    )
                finally:
                    hook.remove()
                if len(captured) != 1 or selected.tolist() != batch["indices"]:
                    raise RuntimeError("native source or fixed selector mismatch")
                source = captured[0]
                states, deltas = runner.legacy.iterative_refine_o(
                    image, source, refiner, 2, overlap, indices=None
                )
                final = states[-1]
                raw = F.softplus(-final.float())[h0].sum()
                extra = runner.EXTRA_LAMBDA * raw / batch["denominator_c0"]
                total = native + extra
            if not all(torch.isfinite(value) for value in (native, extra, total)):
                raise FloatingPointError("nonfinite forward")
            scaler.scale(total).backward()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            scale = float(scaler.get_scale())
            grads = {}
            for group, named in named_groups.items():
                values = [(parameter.grad.float() / scale) for _, parameter in named if parameter.grad is not None]
                grads[group] = {
                    "present": len(values),
                    "l2": math.sqrt(sum(float(value.square().sum()) for value in values)),
                    "finite": all(bool(torch.isfinite(value).all()) for value in values),
                }
            rows.append({
                "repeat": repeat + 1,
                "elapsed_forward_backward_s": elapsed,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device),
                "native_loss": float(native.detach()),
                "extra_final_loss": float(extra.detach()),
                "total_loss": float(total.detach()),
                "selected_four": selected.tolist(),
                "H0_members": int(h0.sum()),
                "final_shape": list(final.shape),
                "source_to_final_abs_max": float((final - source).detach().abs().max()),
                "gradients": grads,
                "scaler_scale": scale,
            })
        if max(abs(row["native_loss"] - 0.023890312761068344) for row in rows) > 1e-5:
            raise ValueError("native first-batch loss differs from matched reference")
        if any(not all(value["finite"] for value in row["gradients"].values()) for row in rows):
            raise FloatingPointError("nonfinite profile gradients")
        result = {
            "status": "PASS",
            "optimizer_updates": 0,
            "test_used": False,
            "batch_sha256": batch_sha,
            "C0_sha256": runner.sha256_file(Path(lock["checkpoint"])),
            "rows": rows,
        }
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        runner.legacy.choose_pair_instances = original_selector


if __name__ == "__main__":
    main()
