#!/usr/bin/env python3
"""Matched RAM overlap-prior placement continuation and locked test evaluation.

All arms start at C0 and train the same mature decoder/refiner parameters.  The
only arm variable is whether frozen image-only O enters the first nnU-Net conv,
the first mature refiner conv, or neither.  The tiny zero-initialized adapters
are identical in trainable size and optimizer settings; no GT enters either O.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_v31_ram_mature as mature
from evaluate_r329_ram_official_metrics_val import _aggregate, _official_one
from ram_o_position_hooks import attach_position_hooks


ARMS = ("ZERO", "INPUT_O", "REFINER_O")
C0_SHA = "ae4c4eb21f6e9424e30c731b2c388d24233d389198a6371b1bd3090d3d788384"
O_SHA = "e7b738e0b413f9138660475062e76e5d67d9df5c94afc0a2c2597ff0cf126ae5"
R332_SHA = "eaf3b6918f7b64d90ae4295be1df4ec4c2c314775e68472e85607667a4334655"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _weights_l2(module: torch.nn.Module) -> float:
    return float(sum(p.detach().float().square().sum() for p in module.parameters()).sqrt())


def load_arm(lock: dict, arm: str, device: torch.device, train: bool):
    assert arm in ARMS or arm == "C0"
    paths = {key: Path(lock[key]) for key in ("c0", "r332", "overlap")}
    for key, expected in (("c0", C0_SHA), ("r332", R332_SHA), ("overlap", O_SHA)):
        if sha(paths[key]) != expected:
            raise ValueError(f"{key} checksum mismatch")
    args = SimpleNamespace(r332=paths["r332"], overlap=paths["overlap"], overlap_sha=O_SHA)
    _, model, refiner, generator, _, _ = mature.load_components(args, device)
    payload = torch.load(paths["c0"], map_location="cpu", weights_only=False)
    if int(payload["updates"]) != 100:
        raise ValueError("C0 is not fresh100")
    model.load_state_dict(payload["model"], strict=True)
    refiner.load_state_dict(payload["refiner"], strict=True)
    generator.eval().requires_grad_(False)
    if train:
        names = mature.set_joint_trainability(model, True)
        decoder = [p for p in model.parameters() if p.requires_grad]
        if len(decoder) != 22 or len(list(refiner.parameters())) != 40:
            raise ValueError("C0 optimizer group structure changed")
        optimizer = torch.optim.AdamW(
            [{"params": list(refiner.parameters()), "lr": lock["refiner_lr"]},
             {"params": decoder, "lr": lock["decoder_lr"]}],
            weight_decay=lock["weight_decay"],
        )
        old = payload["optimizer"]
        if [len(g["params"]) for g in old["param_groups"]] != [40, 22]:
            raise ValueError("C0 optimizer group sizes differ")
        optimizer.load_state_dict(copy.deepcopy(old))
        for group, lr in zip(optimizer.param_groups, (lock["refiner_lr"], lock["decoder_lr"])):
            group["lr"] = lr
        for group in optimizer.param_groups:
            for param in group["params"]:
                state = optimizer.state.get(param, {})
                if set(state) != {"step", "exp_avg", "exp_avg_sq"} or int(state["step"]) != 100:
                    raise ValueError("C0 Adam state incomplete")
                if state["exp_avg"].shape != param.shape or state["exp_avg_sq"].shape != param.shape:
                    raise ValueError("C0 Adam shape mismatch")
    else:
        model.eval().requires_grad_(False)
        refiner.eval().requires_grad_(False)
        optimizer = None
    hooks = attach_position_hooks(model, refiner, arm if arm != "C0" else "ZERO")
    if train:
        params = list(hooks.input_adapter.parameters()) + list(hooks.refiner_adapter.parameters())
        if len(params) != 2 or sum(p.numel() for p in params) != 216:
            raise ValueError("matched O adapter capacity changed")
        if any(int(torch.count_nonzero(p)) for p in params):
            raise ValueError("new O adapters must start at zero")
        optimizer.add_param_group({"params": params, "lr": lock["adapter_lr"],
                                   "weight_decay": 0.0, "name": "o_adapters"})
    return model, refiner, generator, hooks, optimizer, payload


def configure_train_mode(model, refiner) -> None:
    model.eval()
    for stage in (3, 4):
        model.backbone.decoder.stages[stage].train()
        model.backbone.decoder.transpconvs[stage].train()
    model.backbone.decoder.seg_layers[4].train()
    refiner.train()


@torch.no_grad()
def get_o(generator, image):
    # The frozen image-only generator has no GT dependency.  It is always
    # called in all arms, even ZERO, to keep execution and RNG comparable.
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all()
    py = random.getstate()
    np_state = np.random.get_state()
    was_train = generator.training
    generator.eval()
    try:
        value = generator(image).sigmoid().detach()
    finally:
        generator.train(was_train)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        random.setstate(py)
        np.random.set_state(np_state)
    if value.shape != image.shape or not bool(torch.isfinite(value).all()):
        raise ValueError("invalid frozen O map")
    return value


def forward(model, refiner, generator, hooks, image, indices=None):
    o = get_o(generator, image)
    hooks.set_o(o)
    source = model(torch.cat((image, torch.zeros_like(image)), 1))
    states, deltas = mature.iterative_refine_o(
        image, source, refiner, 2, torch.zeros_like(image), indices=indices)
    return states, deltas, o


def train_loss(model, refiner, generator, hooks, image, target, valid):
    o = get_o(generator, image)
    hooks.set_o(o)
    source = model(torch.cat((image, torch.zeros_like(image)), 1))
    probability = source.sigmoid()
    with torch.no_grad():
        indices = mature.choose_pair_instances(probability.detach(), target, 4)
        selected_target = target[0, indices].unsqueeze(1)
        initial_error = torch.abs(probability[0, indices].unsqueeze(1).detach() - selected_target)
        local = F.max_pool2d((initial_error > .15).float(), 11, 1, 5)
    states, deltas = mature.iterative_refine_o(
        image, source, refiner, 2, torch.zeros_like(image), indices=indices)
    losses = []
    for step in (1, 2):
        logits = states[step][0, indices].unsqueeze(1)
        base, _ = mature.completion_loss(logits, selected_target, local, deltas[step - 1], .05)
        prob = logits.float().sigmoid()
        surface = mature.soft_surface_nsd_loss(prob, selected_target)
        instance_volume, pair_volume = mature.volume_losses(prob, selected_target)
        losses.append(base + .03 * surface + .005 * instance_volume + .01 * pair_volume)
    deep = losses[0] / 3 + losses[1] * 2 / 3
    monotonic = F.relu(losses[1] - losses[0].detach())
    contraction = F.relu(deltas[1].abs().mean() - deltas[0].detach().abs().mean())
    global_loss = mature.dice_bce_logits(source, target)
    total = deep + .05 * monotonic + .01 * contraction + .10 * global_loss
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite training loss")
    return total, {"global": float(global_loss.detach()), "step1": float(losses[0].detach()),
                   "step2": float(losses[1].detach()), "O_mean": float(o.mean()),
                   "source_abs": float(source.detach().abs().mean()),
                   "final_abs": float(states[2].detach().abs().mean())}, indices


def make_loaders(lock, split, seed):
    dataset = mature.NativeWristDataset(Path(lock["dataset"]), split, augment=split == "train")
    expected = {"train": 425, "val": 69, "test": 124}[split]
    if len(dataset) != expected:
        raise ValueError(f"{split} case count {len(dataset)} != {expected}")
    return DataLoader(dataset, batch_size=1, shuffle=split == "train", num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def run_smoke(lock, arm, output):
    device = torch.device("cuda")
    seed_all(lock["seed"])
    model, refiner, generator, hooks, optimizer, _ = load_arm(lock, arm, device, True)
    batch = next(iter(make_loaders(lock, "train", lock["seed"])))
    image, target, valid = (batch[k].to(device) for k in ("image", "mask", "valid"))
    model.eval(); refiner.eval()
    with torch.inference_mode():
        hooks.set_o(None)
        base = model(torch.cat((image, torch.zeros_like(image)), 1))
        reference, _ = mature.iterative_refine_o(image, base, refiner, 2, torch.zeros_like(image))
        candidate, _, _ = forward(model, refiner, generator, hooks, image)
        parity = [float((a - b).abs().max()) for a, b in zip(reference, candidate)]
    if parity != [0.0, 0.0, 0.0]:
        raise RuntimeError(f"zero-init placement broke C0 parity: {parity}")
    configure_train_mode(model, refiner)
    optimizer.zero_grad(set_to_none=True)
    loss, pieces, indices = train_loss(model, refiner, generator, hooks, image, target, valid)
    loss.backward()
    gradients = {"input": float(hooks.input_adapter.weight.grad.abs().sum())
                 if hooks.input_adapter.weight.grad is not None else 0.0,
                 "refiner": float(hooks.refiner_adapter.weight.grad.abs().sum())
                 if hooks.refiner_adapter.weight.grad is not None else 0.0}
    expected = {"ZERO": (False, False), "INPUT_O": (True, False),
                "REFINER_O": (False, True)}[arm]
    passed = all((gradients[name] > 0) == flag for name, flag in zip(("input", "refiner"), expected))
    result = {"status": "PASS" if passed else "FAIL", "arm": arm, "loss": float(loss.detach()),
              "gradients": gradients, "indices": indices.tolist(), "pieces": pieces,
              "zero_init_step_logit_max_abs": parity, "hook_stats": hooks.stats,
              "optimizer_steps": 0, "test_used": False}
    save_json(output, result)
    hooks.close()
    if not passed:
        raise RuntimeError(result)


def run_train(lock, arm, output, lock_path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device("cuda")
    seed_all(lock["seed"])
    model, refiner, generator, hooks, optimizer, _ = load_arm(lock, arm, device, True)
    initial_adapter = {"input": _weights_l2(hooks.input_adapter),
                       "refiner": _weights_l2(hooks.refiner_adapter)}
    loader = make_loaders(lock, "train", lock["seed"])
    start = time.monotonic()
    trace = (output / "TRAINING_TRACE.jsonl").open("w", encoding="utf-8")
    sequence = (output / "CASE_AND_BATCH_SEQUENCE.jsonl").open("w", encoding="utf-8")
    steps = 0
    try:
        for epoch in range(1, lock["epochs"] + 1):
            configure_train_mode(model, refiner)
            for batch in loader:
                if time.monotonic() - start > lock["max_seconds_per_arm"]:
                    raise TimeoutError("per-arm runtime budget")
                if steps % 100 == 0 and shutil.disk_usage(output).free < lock["disk_floor_bytes"]:
                    raise RuntimeError("disk free below locked floor")
                image, target, valid = (batch[k].to(device) for k in ("image", "mask", "valid"))
                # Hash the actual augmented tensors so all arms can be checked.
                digest = hashlib.sha256(batch["image"].numpy().tobytes() +
                                        batch["mask"].numpy().tobytes() +
                                        batch["valid"].numpy().tobytes()).hexdigest()
                optimizer.zero_grad(set_to_none=True)
                loss, pieces, indices = train_loss(model, refiner, generator, hooks, image, target, valid)
                loss.backward()
                gradients = {"input": float(hooks.input_adapter.weight.grad.abs().sum())
                             if hooks.input_adapter.weight.grad is not None else 0.0,
                             "refiner": float(hooks.refiner_adapter.weight.grad.abs().sum())
                             if hooks.refiner_adapter.weight.grad is not None else 0.0}
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), 12.0)
                decoder = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(decoder, 5.0)
                torch.nn.utils.clip_grad_norm_(list(hooks.input_adapter.parameters()) +
                                               list(hooks.refiner_adapter.parameters()), 5.0)
                before = [p.detach().clone() for p in hooks.input_adapter.parameters()] + [
                    p.detach().clone() for p in hooks.refiner_adapter.parameters()]
                optimizer.step()
                steps += 1
                seq = {"step": steps, "case": batch["case"][0], "batch_sha256": digest,
                       "indices": indices.tolist()}
                sequence.write(json.dumps(seq) + "\n")
                if steps == 1 or steps % lock["log_every"] == 0 or steps == lock["epochs"] * 425:
                    now = list(hooks.input_adapter.parameters()) + list(hooks.refiner_adapter.parameters())
                    trace.write(json.dumps({"step": steps, "epoch": epoch, "loss": float(loss.detach()),
                                            "parts": pieces, "O_grad_l1": gradients,
                                            "O_delta_theta_l2": [float((p.detach() - old).float().norm())
                                                                 for p, old in zip(now, before)],
                                            "O_adapter_weight_l2": {"input": _weights_l2(hooks.input_adapter),
                                                                     "refiner": _weights_l2(hooks.refiner_adapter)},
                                            "lr": [float(g["lr"]) for g in optimizer.param_groups],
                                            "case": batch["case"][0], "success": True}) + "\n")
                    trace.flush()
                    sequence.flush()
        if steps != lock["epochs"] * 425:
            raise RuntimeError("not all full-data updates completed")
        checkpoint = {"model": model.state_dict(), "refiner": refiner.state_dict(),
                      "input_adapter": hooks.input_adapter.state_dict(),
                      "refiner_adapter": hooks.refiner_adapter.state_dict(),
                      "optimizer": optimizer.state_dict(), "arm": arm, "updates": steps,
                      "lock_sha256": sha(lock_path)}
        torch.save(checkpoint, output / "final.pth")
        summary = {"status": "COMPLETE", "arm": arm, "updates": steps,
                   "seconds": time.monotonic() - start, "C0_sha256": C0_SHA,
                   "O_sha256": O_SHA, "final_sha256": sha(output / "final.pth"),
                   "initial_adapter_l2": initial_adapter,
                   "final_adapter_l2": {"input": _weights_l2(hooks.input_adapter),
                                        "refiner": _weights_l2(hooks.refiner_adapter)},
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                   "test_used": False}
        save_json(output / "TRAIN_RESULT.json", summary)
        print(json.dumps(summary), flush=True)
    finally:
        trace.close()
        sequence.close()
        hooks.close()


@torch.inference_mode()
def run_eval(lock, arm, split, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    device = torch.device("cuda")
    seed_all(lock["seed"])
    model, refiner, generator, hooks, _, _ = load_arm(lock, arm, device, False)
    if arm != "C0":
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["arm"] != arm or saved["lock_sha256"] != lock["self_sha256"] or \
           saved["updates"] != lock["epochs"] * 425:
            raise ValueError("checkpoint arm/lock/step mismatch")
        model.load_state_dict(saved["model"], strict=True)
        refiner.load_state_dict(saved["refiner"], strict=True)
        hooks.input_adapter.load_state_dict(saved["input_adapter"], strict=True)
        hooks.refiner_adapter.load_state_dict(saved["refiner_adapter"], strict=True)
    model.eval(); refiner.eval()
    loader = make_loaders(lock, split, lock["seed"])
    pairs = mature.top_training_pairs(Path(lock["dataset"]), 15)
    overall, overlap, pair_rows = [], [], []
    per_case = []
    counts = {key: np.zeros(14, dtype=np.int64) for key in ("TP", "FP", "FN")}
    start = time.monotonic()
    for batch in loader:
        image = batch["image"].to(device)
        target = (batch["mask"].numpy()[0] > .5)
        states, _, o = forward(model, refiner, generator, hooks, image)
        pred = (states[2].sigmoid() >= .5).cpu().numpy()[0]
        case = batch["case"][0]
        c_overall = [_official_one(pred[k], target[k]) for k in range(14)]
        c_overlap = _official_one(pred.sum(0) >= 2, target.sum(0) >= 2)
        c_pairs = []
        for i, j in pairs:
            gt_pair = target[i] & target[j]
            if gt_pair.any():
                c_pairs.append(_official_one(pred[i] & pred[j], gt_pair))
        overall.extend(c_overall); overlap.append(c_overlap); pair_rows.extend(c_pairs)
        counts["TP"] += (pred & target).sum((1, 2))
        counts["FP"] += (pred & ~target).sum((1, 2))
        counts["FN"] += (~pred & target).sum((1, 2))
        per_case.append({"case": case, "overall": _aggregate(c_overall),
                         "overlap": _aggregate([c_overlap]), "pair": _aggregate(c_pairs),
                         "O_mean": float(o.mean()), "pair_count": len(c_pairs)})
    official = {"overall_instance_metrics": _aggregate(overall),
                "overlap_region_metrics": _aggregate(overlap),
                "overlap_pair_intersection_metrics": _aggregate(pair_rows),
                "overlap_pair_count": len(pair_rows)}
    result = {"status": "COMPLETE", "arm": arm, "split": split,
              "cases": len(per_case), "checkpoint": str(checkpoint) if checkpoint else lock["c0"],
              "checkpoint_sha256": sha(checkpoint) if checkpoint else C0_SHA,
              "official": official, "flat": mature.flatten(official),
              "member_counts": {k: v.tolist() for k, v in counts.items()},
              "per_case": per_case, "threshold": 0.5,
              "pair_definition": "top15 from train only", "test_used": split == "test",
              "seconds": time.monotonic() - start}
    save_json(output / "OFFICIAL.json", result)
    print(json.dumps({"arm": arm, "split": split, "flat": result["flat"],
                      "seconds": result["seconds"]}), flush=True)
    hooks.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("smoke", "train", "eval"), required=True)
    p.add_argument("--arm", choices=ARMS + ("C0",), required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"))
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    torch.set_num_threads(16)
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED_BEFORE_TRAIN" or lock["runner_sha256"] != sha(Path(__file__)):
        raise ValueError("code/run lock mismatch")
    if args.phase == "smoke":
        if args.arm == "C0": raise ValueError("smoke C0 unnecessary")
        run_smoke(lock, args.arm, args.out)
    elif args.phase == "train":
        if args.arm == "C0": raise ValueError("C0 is frozen")
        run_train(lock, args.arm, args.out, args.lock)
    else:
        if args.split is None: raise ValueError("eval split required")
        if args.arm != "C0" and args.checkpoint is None: raise ValueError("checkpoint required")
        lock["self_sha256"] = sha(args.lock)
        run_eval(lock, args.arm, args.split, args.checkpoint, args.out)


if __name__ == "__main__":
    main()
