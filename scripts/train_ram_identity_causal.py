"""Train one isolated RAM identity arm from the common fresh-100 continuation.

The runner is intentionally paired: invoke it once with ``identity_mode`` set
to ``canonical`` and once with ``scrambled`` in otherwise identical locks.  It
keeps the original R332 objective and data path, while adding only the
function-preserving member-relation parameter supplied by
``r332_member_identity``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
import statistics
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_v31_ram_mature as legacy
from r332_member_identity import IdentityContextRefiner, iterative_refine_identity


DEFAULT_UPDATES = 850
DEFAULT_EPOCH_LENGTH = 425
DEFAULT_DETAILED_STEPS = (1, 425, 850)
DEFAULT_IDENTITY_LR = 8e-5
DEFAULT_WEIGHT_DECAY = 1e-4


def write(path: Path, value) -> None:
    def safe(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [safe(x) for x in v]
        return v
    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding="utf-8")


def stats(t: torch.Tensor) -> dict:
    x = t.detach().float().reshape(-1)
    good = x[torch.isfinite(x)]
    if not good.numel():
        return {"shape": list(t.shape), "finite_fraction": 0.0}
    sample = good[:: max(1, good.numel() // 4096)]
    return {
        "shape": list(t.shape),
        "mean": float(good.mean()),
        "std": float(good.std(unbiased=False)),
        "min": float(good.min()),
        "max": float(good.max()),
        "norm": float(good.norm()),
        "zero_fraction": float((good == 0).float().mean()),
        "finite_fraction": float(torch.isfinite(x).float().mean()),
        "q05_q50_q95": torch.quantile(
            sample, torch.tensor([0.05, 0.5, 0.95], device=x.device)
        ).tolist(),
    }


def rng_get():
    return (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )


def rng_set(state) -> None:
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def rng_equal(left, right) -> bool:
    if left[0] != right[0] or not np.array_equal(left[1][1], right[1][1]):
        return False
    if left[1][0] != right[1][0] or left[1][2:] != right[1][2:]:
        return False
    if not torch.equal(left[2], right[2]) or len(left[3]) != len(right[3]):
        return False
    return all(torch.equal(a, b) for a, b in zip(left[3], right[3]))


def torch_rng_unchanged(before, after) -> bool:
    return torch.equal(before[2], after[2]) and len(before[3]) == len(after[3]) and all(
        torch.equal(b, a) for b, a in zip(before[3], after[3])
    )


def grouping(name: str) -> str:
    if name.startswith("refiner."):
        return ".".join(name.split(".")[:2])
    fields = name.split(".")
    return ".".join(fields[:5])


def grad_summary(named, grads) -> dict:
    out = {}
    for (name, parameter), grad in zip(named, grads):
        key = grouping(name)
        row = out.setdefault(
            key,
            {
                "squared_norm": 0.0,
                "elements": 0,
                "none_elements": 0,
                "zero_elements": 0,
                "nonfinite_elements": 0,
            },
        )
        row["elements"] += parameter.numel()
        if grad is None:
            row["none_elements"] += parameter.numel()
            continue
        grad = grad.detach().float()
        row["squared_norm"] += float(grad.square().sum())
        row["zero_elements"] += int((grad == 0).sum())
        row["nonfinite_elements"] += int((~torch.isfinite(grad)).sum())
    for row in out.values():
        row["norm"] = row.pop("squared_norm") ** 0.5
    return out


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def batch_sha(batch: dict) -> str:
    return hashlib.sha256(
        b"".join(batch[key].numpy().tobytes() for key in ("image", "mask", "valid"))
    ).hexdigest()


def parameter_hash_rows(named) -> list[dict]:
    rows = []
    for name, parameter in named:
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "bytes": parameter.numel() * parameter.element_size(),
                "sha256": tensor_sha(parameter),
            }
        )
    return rows


def parameter_hash_rows_with_mature(named, mature_named) -> list[dict]:
    mature_by_name = dict(mature_named)
    rows = []
    for name, parameter in named:
        mature_parameter = mature_by_name.get(name)
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "bytes": parameter.numel() * parameter.element_size(),
                "sha256": tensor_sha(parameter),
                "mature_sha256": None
                if mature_parameter is None
                else tensor_sha(mature_parameter),
                "equal_to_mature": None
                if mature_parameter is None
                else bool(
                    torch.equal(
                        parameter.detach().cpu(), mature_parameter.detach().cpu()
                    )
                ),
            }
        )
    return rows


def state_step(state: dict) -> float:
    value = state.get("step")
    if value is None:
        raise ValueError("Adam state has no step")
    return float(value.detach().cpu() if torch.is_tensor(value) else value)


def state_for_id(state_map: dict, state_id):
    if state_id in state_map:
        return state_map[state_id]
    if torch.is_tensor(state_id):
        state_id = int(state_id.item())
    return state_map.get(int(state_id))


def optimizer_group_options(source_group: dict) -> dict:
    options = {}
    for key in ("lr", "betas", "eps", "weight_decay", "amsgrad"):
        if key in source_group:
            options[key] = source_group[key]
    return options


def restore_old_adam(
    optimizer,
    model,
    mature_refiner,
    identity_refiner,
    continuation_payload: dict,
    expected_step: int,
) -> dict:
    """Map the two historical groups by names, excluding relation_weight.

    The continuation checkpoint does not carry parameter names in its Adam
    state.  The loaded mature module and model reconstruct the historical
    traversal order, while the identity module supplies the destination
    parameters with the relation parameter explicitly removed.
    """
    source_optimizer = continuation_payload.get("optimizer")
    if not source_optimizer:
        raise ValueError("fresh100 continuation has no optimizer state")
    source_groups = source_optimizer.get("param_groups", [])
    if len(source_groups) != 2 or len(optimizer.param_groups) != 2:
        raise ValueError("expected exactly two historical Adam groups before extension")

    source_refiner = list(mature_refiner.named_parameters())
    source_model = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    destination_refiner = [
        (name, parameter)
        for name, parameter in identity_refiner.named_parameters()
        if name != "relation_weight"
    ]
    destination_model = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if [name for name, _ in source_refiner] != [name for name, _ in destination_refiner]:
        raise ValueError("identity refiner changed mature parameter order")
    if [name for name, _ in source_model] != [name for name, _ in destination_model]:
        raise ValueError("decoder parameter order changed during continuation setup")

    destination_groups = [destination_refiner, destination_model]
    source_named_groups = [source_refiner, source_model]
    if any(
        len(source_group["params"]) != len(named)
        for source_group, named in zip(source_groups, source_named_groups)
    ):
        raise ValueError("historical Adam parameter count mismatch")

    mapped = {}
    audit_rows = []
    for group_id, (source_group, source_named, destination_named) in enumerate(
        zip(source_groups, source_named_groups, destination_groups)
    ):
        for (source_name, source_parameter), (destination_name, destination_parameter), state_id in zip(
            source_named, destination_named, source_group["params"]
        ):
            if source_name != destination_name:
                raise ValueError((source_name, destination_name))
            source_state = state_for_id(source_optimizer["state"], state_id)
            if not source_state or set(source_state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("incomplete Adam state for " + source_name)
            step = state_step(source_state)
            if step != float(expected_step):
                raise ValueError(
                    f"historical Adam step for {source_name} is {step}, expected {expected_step}"
                )
            if tuple(source_parameter.shape) != tuple(destination_parameter.shape):
                raise ValueError("parameter shape mismatch for " + source_name)
            state = {"step": source_state["step"].detach().cpu().clone()}
            for key in ("exp_avg", "exp_avg_sq"):
                value = source_state[key]
                if tuple(value.shape) != tuple(source_parameter.shape):
                    raise ValueError(f"{key} shape mismatch for {source_name}")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"nonfinite {key} for {source_name}")
                if key == "exp_avg_sq" and bool((value < 0).any()):
                    raise ValueError(f"negative exp_avg_sq for {source_name}")
                state[key] = value.detach().to(
                    device=destination_parameter.device,
                    dtype=destination_parameter.dtype,
                ).clone()
            mapped[destination_parameter] = state
            audit_rows.append(
                {
                    "name": "refiner." + source_name
                    if group_id == 0
                    else "model." + source_name,
                    "group": group_id,
                    "source_state_id": int(state_id),
                    "shape": list(destination_parameter.shape),
                    "step": step,
                    "exp_avg_norm": float(state["exp_avg"].norm()),
                    "exp_avg_sq_sum": float(state["exp_avg_sq"].sum()),
                }
            )
    if len(mapped) != sum(len(group["params"]) for group in optimizer.param_groups):
        raise ValueError("historical Adam mapping does not cover both groups")
    optimizer.state.clear()
    optimizer.state.update(mapped)
    return {
        "status": "PASS",
        "source_updates": continuation_payload.get("updates"),
        "expected_historical_step": expected_step,
        "restored_groups": 2,
        "relation_parameter_excluded": True,
        "parameters": audit_rows,
    }


def initial_function_parity(
    batch,
    model,
    mature_refiner,
    identity_refiner,
    old_iterative_refine,
    device,
) -> dict:
    """Compare mature and zero-relation identity refinement before step one."""
    saved_rng = rng_get()
    saved_modes = [
        (module, module.training)
        for root in (model, mature_refiner, identity_refiner)
        for module in root.modules()
    ]
    try:
        model.eval()
        mature_refiner.eval()
        identity_refiner.eval()
        image = batch["image"].to(device)
        zero_overlap = torch.zeros_like(image)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
            base_logits = model(torch.cat((image, zero_overlap), 1))
            old_states, old_deltas = old_iterative_refine(
                image, base_logits, mature_refiner, 2, zero_overlap
            )
            new_states, new_deltas = iterative_refine_identity(
                image, base_logits, identity_refiner, 2, zero_overlap
            )
        state_rows = []
        max_logit = 0.0
        mask_mismatch = 0
        for step, (old_state, new_state) in enumerate(zip(old_states, new_states)):
            gap = float((old_state - new_state).abs().max())
            mismatch = int(
                torch.count_nonzero(
                    old_state.sigmoid().ge(0.5) != new_state.sigmoid().ge(0.5)
                )
            )
            max_logit = max(max_logit, gap)
            mask_mismatch += mismatch
            state_rows.append(
                {
                    "step": step,
                    "max_abs_logit": gap,
                    "mask_mismatch_pixels": mismatch,
                }
            )
        delta_rows = []
        max_delta = 0.0
        for step, (old_delta, new_delta) in enumerate(zip(old_deltas, new_deltas), 1):
            gap = float((old_delta - new_delta).abs().max())
            max_delta = max(max_delta, gap)
            delta_rows.append({"step": step, "max_abs_delta": gap})
    finally:
        for module, training in saved_modes:
            module.training = training
        rng_set(saved_rng)
    restored_rng = rng_get()
    modes_restored = all(module.training == training for module, training in saved_modes)
    rng_restored = rng_equal(saved_rng, restored_rng)
    result = {
        "status": "PASS"
        if max_logit <= 1e-6 and mask_mismatch == 0 and rng_restored and modes_restored
        else "FAIL",
        "dtype": "float32",
        "iterations": 2,
        "scope": "all14; common model output; zero relation_weight",
        "max_abs_logit": max_logit,
        "mask_mismatch_pixels": mask_mismatch,
        "max_abs_delta": max_delta,
        "state_rows": state_rows,
        "delta_rows": delta_rows,
        "logit_tolerance": 1e-6,
        "mask_gate": "exact",
        "rng_restored": rng_restored,
        "modes_restored": modes_restored,
        "batch_sha256": batch_sha(batch),
    }
    if result["status"] != "PASS":
        raise RuntimeError({"initial_function_parity": result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("cpu_threads") is not None:
        cpu_threads = int(lock["cpu_threads"])
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        torch.set_num_threads(cpu_threads)

    identity_mode = lock.get("identity_mode")
    amp_enabled = bool(lock.get('amp_enabled',True))
    if identity_mode not in {"canonical", "scrambled"}:
        raise ValueError("identity_mode must be canonical or scrambled")
    updates = int(lock.get("updates", DEFAULT_UPDATES))
    epoch_length = int(lock.get("epoch_length", DEFAULT_EPOCH_LENGTH))
    if updates != DEFAULT_UPDATES or epoch_length != DEFAULT_EPOCH_LENGTH or updates != 2 * epoch_length:
        raise ValueError("identity pair requires exactly 850 updates in two 425-step cycles")
    detailed_steps = tuple(int(step) for step in lock.get("detailed_steps", DEFAULT_DETAILED_STEPS))
    if detailed_steps != DEFAULT_DETAILED_STEPS:
        raise ValueError("detailed steps must be 1, 425, 850")
    if lock.get("intermediate_validation_steps", []):
        raise ValueError("identity pair validation is final-only")
    if lock.get("source_auxiliary_mode") or lock.get("negative_guard_enabled"):
        raise ValueError("source auxiliary and negative guard must be disabled")
    identity_lr = float(lock.get("identity_lr", DEFAULT_IDENTITY_LR))
    identity_weight_decay = float(lock.get("identity_weight_decay", DEFAULT_WEIGHT_DECAY))
    if identity_lr != DEFAULT_IDENTITY_LR or identity_weight_decay != DEFAULT_WEIGHT_DECAY:
        raise ValueError("identity relation group must use lr=8e-5 and weight_decay=1e-4")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)

    for path, expected in lock.get("file_hashes", {}).items():
        if legacy.sha(path) != expected:
            raise ValueError("hash mismatch: " + path)

    start = time.monotonic()
    seed = int(lock["seed"])
    legacy.seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("RAM identity training requires CUDA")
    device = torch.device("cuda")

    r332_path = Path(lock["r332"])
    overlap_path = Path(lock["overlap"])
    load_args = SimpleNamespace(
        r332=r332_path,
        overlap=overlap_path,
        overlap_sha=lock["overlap_sha"],
    )
    base_payload, model, loaded_refiner, generator, backbone_mapping, refiner_mapping = legacy.load_components(
        load_args, device
    )
    continuation_path = lock.get("continuation_checkpoint") or lock.get("checkpoint")
    if not continuation_path:
        raise ValueError("common fresh100 continuation checkpoint is required")
    continuation_path = Path(continuation_path)
    continuation = torch.load(continuation_path, map_location="cpu", weights_only=False)
    if "model" not in continuation or "refiner" not in continuation or "optimizer" not in continuation:
        raise ValueError("continuation must contain model, refiner, and optimizer")
    if continuation.get("updates") not in (None, 100):
        raise ValueError("identity pair must start from the common fresh100 continuation")
    model.load_state_dict(continuation["model"], strict=True)
    loaded_refiner.load_state_dict(continuation["refiner"], strict=True)

    # Keep the mature object for the pre-update parity proof, then replace it
    # before constructing the optimizer or registering any observation hooks.
    mature_refiner = copy.deepcopy(loaded_refiner).to(device)
    mature_refiner.eval().requires_grad_(False)
    refiner = IdentityContextRefiner.from_mature(mature_refiner).to(device)
    refiner.identity_mode = identity_mode
    refiner.train().requires_grad_(True)
    if [name for name, _ in refiner.named_parameters()][0] != "relation_weight":
        raise ValueError("relation_weight must be first in the identity refiner iterator")
    original_iterative_refine = legacy.iterative_refine_o
    legacy.iterative_refine_o = iterative_refine_identity

    legacy.set_joint_trainability(model, True)
    model.eval()
    for module in [
        model.backbone.decoder.stages[3],
        model.backbone.decoder.stages[4],
        model.backbone.decoder.transpconvs[3],
        model.backbone.decoder.transpconvs[4],
        model.backbone.decoder.seg_layers[4],
    ]:
        module.train()

    mature_named = [("refiner." + name, parameter) for name, parameter in mature_refiner.named_parameters()]
    identity_refiner_named = [("refiner." + name, parameter) for name, parameter in refiner.named_parameters()]
    model_named = [("model." + name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    named = model_named + identity_refiner_named
    relation_parameter = refiner.relation_weight
    old_refiner_parameters = [
        parameter for name, parameter in refiner.named_parameters() if name != "relation_weight"
    ]
    decoder_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    source_optimizer = continuation["optimizer"]
    source_groups = source_optimizer["param_groups"]
    if len(source_groups) != 2:
        raise ValueError("fresh100 continuation optimizer must have two historical groups")
    if [g['lr'] for g in source_groups] != [lock['refiner_lr_override'],7e-6]:
        raise ValueError('historical optimizer LR differs from locked values')
    optimizer = torch.optim.AdamW(
        [
            {"params": old_refiner_parameters, **optimizer_group_options(source_groups[0])},
            {"params": decoder_parameters, **optimizer_group_options(source_groups[1])},
        ]
    )
    restored_adam = restore_old_adam(
        optimizer,
        model,
        mature_refiner,
        refiner,
        continuation,
        int(lock.get("old_optimizer_steps", 100)),
    )
    optimizer.add_param_group(
        {
            "params": [relation_parameter],
            "lr": identity_lr,
            "weight_decay": identity_weight_decay,
            "betas": source_groups[0].get("betas", (0.9, 0.999)),
            "eps": source_groups[0].get("eps", 1e-8),
            "amsgrad": source_groups[0].get("amsgrad", False),
        }
    )
    if relation_parameter in optimizer.state:
        raise RuntimeError("relation parameter received Adam state before the first update")
    old_optimizer_steps = int(lock.get("old_optimizer_steps", 100))
    if any(
        state_step(state) != float(old_optimizer_steps)
        for parameter, state in optimizer.state.items()
        for _ in [parameter]
    ):
        raise RuntimeError(f"old Adam states are not at step {old_optimizer_steps}")
    registered = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(registered) != len(set(map(id, registered))) or set(map(id, registered)) != set(
        map(id, [parameter for _, parameter in named])
    ):
        raise RuntimeError("optimizer coverage is not exact")

    # Deterministic DataLoader generator is created once and its iterator is
    # renewed only at the natural 425-sample boundary; no reseeding occurs.
    data = Path(lock["dataset"])
    train = legacy.NativeWristDataset(data, "train", True)
    if len(train) != epoch_length:
        raise ValueError(f"expected {epoch_length} training samples, got {len(train)}")
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler_initial_state = sampler_generator.get_state()
    loader = DataLoader(
        train,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        generator=sampler_generator,
    )
    iterator = iter(loader)
    first_batch = next(iterator)

    parity = initial_function_parity(
        first_batch,
        model,
        mature_refiner,
        refiner,
        original_iterative_refine,
        device,
    )
    write(args.out / "INITIAL_FUNCTION_PARITY.json", parity)

    initial_parameter_hashes = parameter_hash_rows_with_mature(named, model_named + mature_named)
    old_parameter_equal = all(
        row["equal_to_mature"] is True
        for row in initial_parameter_hashes
        if row["name"] != "refiner.relation_weight"
    )
    relation_zero = int(torch.count_nonzero(relation_parameter)) == 0
    if not old_parameter_equal or not relation_zero:
        raise RuntimeError("identity initialization is not an exact mature copy plus zero relation")
    write(args.out / "INITIAL_PARAMETER_SHA.json", {"parameters": initial_parameter_hashes})

    sampler_metadata = {
        "dataset": str(data),
        "dataset_count": len(train),
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 0,
        "seed": seed,
        "generator": "torch.Generator; one object across both cycles",
        "initial_generator_state_sha256": tensor_sha(sampler_initial_state),
        "first_batch_sha256": batch_sha(first_batch),
        "updates": updates,
        "epoch_length": epoch_length,
        "cycles": 2,
        "reseeded": False,
    }
    initial = {name: parameter.detach().cpu().clone() for name, parameter in named}
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if amp_enabled:
        scaler_state=copy.deepcopy(continuation['scaler'])
        if 'amp_initial_scale' in lock:
            scaler_state['scale']=float(lock['amp_initial_scale'])
        scaler.load_state_dict(scaler_state)

    # Audit before hooks and before the first optimizer update.
    initial_state_steps = [state_step(state) for parameter, state in optimizer.state.items()]
    audit = {
        "history_mode": "IDENTITY_CONTEXT_FROM_COMMON_FRESH100_CONTINUATION",
        "identity_mode": identity_mode,
        "updates": updates,
        "cycles": 2,
        "epoch_length": epoch_length,
        "continuation_checkpoint": str(continuation_path),
        "continuation_updates": continuation.get("updates"),
        "historical_checkpoint_keys": list(continuation),
        "restored_adam": restored_adam,
        "old_state_step_range_before_update": [min(initial_state_steps), max(initial_state_steps)],
        "relation_state_empty_before_update": relation_parameter not in optimizer.state,
        "optimizer_lrs_before_update": [group["lr"] for group in optimizer.param_groups],
        "optimizer_weight_decays": [group["weight_decay"] for group in optimizer.param_groups],
        "identity_lr": identity_lr,
        "identity_weight_decay": identity_weight_decay,
        "backbone_mapping": backbone_mapping,
        "refiner_mapping": refiner_mapping,
        "parameters": parameter_hash_rows(named),
        "initial_parameter_hashes_file": "INITIAL_PARAMETER_SHA.json",
        "initial_parameter_exact_mature_copy": old_parameter_equal,
        "relation_weight_zero_initialized": relation_zero,
        "relation_weight_shape": list(relation_parameter.shape),
        "relation_weight_numel": relation_parameter.numel(),
        "sampler": sampler_metadata,
        "frozen_generator_expected": True,
        "scheduler": "constant LR; no scheduler step",
        "routing": "identity canonical/scrambled only",
        "state_decay": "none",
        "deep_supervision": "two refiner steps 1/3, 2/3; original R332 loss only",
        "source_auxiliary": {"enabled": False, "negative_guard": False},
        "train_eval": {
            name: module.training
            for name, module in model.named_modules()
            if name.endswith(("stages.3", "stages.4", "encoder"))
        },
        "initial_function_parity": parity,
        "detailed_steps": list(detailed_steps),
        "validation_schedule": {"intermediate": [], "final": updates},
        "cpu_threads": torch.get_num_threads(),
        "amp_enabled": amp_enabled,
        "amp_initial_scale": float(scaler.get_scale()),
        "amp_initial_scale_override": lock.get('amp_initial_scale'),
        "scaler_state_policy": 'paired_restored' if amp_enabled else 'disabled_FP32_same_for_both_arms',
    }
    write(args.out / "PARAMETER_AUDIT.json", audit)

    # Register only observation hooks after the refiner replacement, optimizer
    # migration, parity proof, and initialization audits have passed.
    activations = {}
    active = [False]
    handles = []
    modules = {
        "encoder0": model.backbone.encoder.stages[0],
        "bottleneck": model.backbone.encoder.stages[-1],
        "decoder3": model.backbone.decoder.stages[3],
        "decoder4": model.backbone.decoder.stages[4],
        "refiner_enc1": refiner.enc1,
        "refiner_mid": refiner.mid,
        "refiner_gate": refiner.gate,
        "refiner_residual": refiner.residual,
    }
    for key, module in modules.items():
        def hook(m, inputs, output, key=key):
            if active[0]:
                row = {"input": stats(inputs[0]), "output": stats(output)}
                if key == "refiner_enc1":
                    row["channels"] = [stats(inputs[0][:, index]) for index in range(inputs[0].shape[1])]
                if key == "refiner_gate":
                    row["sigmoid"] = stats(output.sigmoid())
                if key == "refiner_residual":
                    row["tanh"] = stats(output.tanh())
                activations.setdefault(key, []).append(row)

        handles.append(module.register_forward_hook(hook))

    original_completion = legacy.completion_loss
    original_surface = legacy.soft_surface_nsd_loss
    original_volume = legacy.volume_losses
    terms = {}
    extra = {}
    counts = [0]

    def completion(z, target, local, delta, weight):
        loss, values = original_completion(z, target, local, delta, weight)
        index = counts[0]
        counts[0] += 1
        terms[f"step{index + 1}_completion"] = (loss, (index + 1) / 3)
        if active[0]:
            probability = z.float().sigmoid()
            local_weight = 1 + 3 * local
            pixel_bce = F.binary_cross_entropy_with_logits(
                z.float(), target.float(), reduction="none"
            )
            valid = extra["valid"].expand_as(target)
            padding_bce = (pixel_bce * local_weight * (~valid)).sum() / local_weight.sum()
            extra[f"step{index + 1}"] = {
                "local_fraction": float(local.mean()),
                "bce_denominator": float(local_weight.sum()),
                "padding_bce_share": float(
                    padding_bce
                    / ((pixel_bce * local_weight).sum() / local_weight.sum()).clamp_min(1e-20)
                ),
                "padding_fraction": float((~valid).float().mean()),
                "residual": stats(delta),
                "positive_pixels": int(target.sum()),
                "valid_pixels": int(valid.sum()),
            }
            terms[f"step{index + 1}_padding_bce_diagnostic_only"] = (padding_bce, 0.0)
            multi = target[:, 0].sum(0) >= 2
            if bool(multi.any()):
                y, x = torch.nonzero(multi)[0].tolist()
                extra[f"step{index + 1}"]["true_multimember_pixel"] = {
                    "yx": [y, x],
                    "targets": target[:, 0, y, x].tolist(),
                    "probabilities": probability[:, 0, y, x].tolist(),
                    "unweighted_bce_logit_gradient_sign": torch.sign(
                        probability[:, 0, y, x] - target[:, 0, y, x]
                    ).tolist(),
                }
        return loss, values

    def surface(probability, target):
        loss = original_surface(probability, target)
        terms[f"step{counts[0]}_surface"] = (loss, 0.03 * counts[0] / 3)
        return loss

    def volume(probability, target):
        instance_volume, pair_volume = original_volume(probability, target)
        terms[f"step{counts[0]}_instance_volume"] = (instance_volume, 0.005 * counts[0] / 3)
        terms[f"step{counts[0]}_pair_volume"] = (pair_volume, 0.01 * counts[0] / 3)
        return instance_volume, pair_volume

    legacy.completion_loss = completion
    legacy.soft_surface_nsd_loss = surface
    legacy.volume_losses = volume

    matched_reference = None
    replay_reference = None
    if lock.get('replay_reference_trace'):
        replay_reference=[json.loads(x) for x in Path(lock['replay_reference_trace']).read_text().splitlines()]
    if lock.get("matched_reference_trace"):
        matched_reference = [
            json.loads(line)
            for line in Path(lock["matched_reference_trace"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(matched_reference) != updates:
            raise ValueError("matched reference length does not match 850 updates")

    successful = 0
    cases = set()
    timings = []
    step_rows = []
    iterator_boundary_states = []
    use_o = bool(lock.get("use_o", False))
    projection_step = int(lock.get("projection_check_step", 10))
    max_seconds = float(lock.get("max_seconds", 1800))
    final_validation_reserve = float(lock.get("final_validation_reserve_seconds", 600))
    disk_floor = int(lock.get("disk_floor_bytes", 4 * 1024**3))

    try:
        for step in range(1, updates + 1):
            elapsed = time.monotonic() - start
            if elapsed > max_seconds:
                raise RuntimeError("time budget before update")
            if shutil.disk_usage(args.out).free < disk_floor:
                raise RuntimeError("disk budget")
            if step == 1:
                batch = first_batch
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator_boundary_states.append(
                        {
                            "completed_step": step - 1,
                            "generator_state_sha256": tensor_sha(sampler_generator.get_state()),
                        }
                    )
                    iterator = iter(loader)
                    batch = next(iterator)
            if batch["image"].shape[0] != 1:
                raise ValueError("identity runner requires batch_size=1")
            tick = time.monotonic()
            active[0] = step in detailed_steps
            activations.clear()
            terms.clear()
            counts[0] = 0
            image, target, valid = [batch[key].to(device) for key in ("image", "mask", "valid")]
            extra = {"valid": valid.bool()}
            optimizer.zero_grad(set_to_none=True)
            model.eval()
            for module in [
                model.backbone.decoder.stages[3],
                model.backbone.decoder.stages[4],
                model.backbone.decoder.transpconvs[3],
                model.backbone.decoder.transpconvs[4],
                model.backbone.decoder.seg_layers[4],
            ]:
                module.train()
            refiner.train()
            random_state = rng_get()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss, parts, details, overlap, indices, supervision = legacy.loss_for_batch(
                    model, refiner, generator, image, target, valid, use_o
                )
            after_random = rng_get()
            for key, weight in (("global", 0.1), ("monotonic", 0.05), ("contraction", 0.01)):
                terms[key] = (parts[key], weight)
            row = {
                "step": step,
                "cycle": (step - 1) // epoch_length + 1,
                "cycle_position": (step - 1) % epoch_length + 1,
                "case": str(batch["case"][0]),
                "identity_mode": identity_mode,
                "batch_sha256": batch_sha(batch),
                "loss": float(loss.detach()),
                "loss_terms": {
                    key: {
                        "raw": float(value.detach()),
                        "coefficient": coefficient,
                        "weighted": float(value.detach()) * coefficient,
                    }
                    for key, (value, coefficient) in terms.items()
                },
                "lr": [group["lr"] for group in optimizer.param_groups],
                "supervision": supervision,
                "prior": stats(overlap),
                "activations": dict(activations),
                "relation_context": dict(getattr(refiner, "last_context_stats", {})),
            }
            row["original_loss_before_source_aux"] = float(loss.detach())
            if replay_reference is not None and step<=len(replay_reference):
                oldrow=replay_reference[step-1]
                row['replay_loss_abs_difference']=abs(row['loss']-oldrow['loss'])
                if row['batch_sha256']!=oldrow['batch_sha256'] or (not lock.get('allow_approximate_replay') and row['replay_loss_abs_difference']>lock.get('replay_numeric_tolerance',0.)):
                    raise RuntimeError('replay data/loss tolerance exceeded')
            if step==lock.get('diagnose_before_update'):
                diagnostic={'step':step,'case':row['case'],'loss_amp':row['loss'],'amp_scale':float(scaler.get_scale()),'new_updates_before_diagnosis':successful,'AMP_indices':indices.tolist(),'gradients':{}}
                for scale in [float(scaler.get_scale()),1024.,1.]:
                    gs=torch.autograd.grad(loss*scale,[p for _,p in named],retain_graph=True,allow_unused=True)
                    diagnostic['gradients']['AMP_scale_'+str(scale)]=grad_summary(named,[None if g is None else g.float()/scale for g in gs]);del gs
                rng_set(random_state)
                with torch.cuda.amp.autocast(enabled=False):
                    fp32,_,_,_,idx32,_=legacy.loss_for_batch(model,refiner,generator,image.float(),target.float(),valid,use_o)
                gs=torch.autograd.grad(fp32,[p for _,p in named],allow_unused=True)
                diagnostic['gradients']['FP32']=grad_summary(named,gs);diagnostic['loss_FP32']=float(fp32.detach());diagnostic['FP32_indices']=idx32.tolist();del gs
                rng_set(after_random)
                assert all(p.grad is None for _,p in named)
                write(args.out/'NUMERIC_DIAGNOSIS.json',diagnostic)
                torch.save({'model':model.state_dict(),'refiner':refiner.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'updates':successful,'rng':rng_get()},args.out/'pre_failure_state.pth')
                write(args.out/'RESULT.json',{'status':'DIAGNOSTIC_COMPLETE','updates':successful,'failed_batch_not_updated':step,'seconds':time.monotonic()-start,'replay_scope':'same batch hashes and bounded loss/update-norm differences; not bitwise historical recovery','tolerance':lock.get('replay_numeric_tolerance',0.)})
                print(json.dumps({'numeric_diagnosis_complete':step,'updates':successful}),flush=True)
                return
            if matched_reference is not None:
                reference_row = matched_reference[step - 1]
                if row["batch_sha256"] != reference_row["batch_sha256"]:
                    raise ValueError("matched data sequence failed before optimizer step")
                if step == 1:
                    reference_loss = reference_row.get(
                        "original_loss_before_source_aux", reference_row["loss"]
                    )
                    if row["original_loss_before_source_aux"] != reference_loss:
                        raise ValueError("initial matched-reference loss differs before optimizer step")

            if active[0]:
                row["loss_support"] = {key: value for key, value in extra.items() if key != "valid"}
                samples = []
                seen = set()
                for name, parameter in named:
                    group = grouping(name)
                    if group not in seen:
                        samples.append((name, parameter))
                        seen.add(group)
                components = {}
                global_grad = torch.autograd.grad(
                    parts["global"],
                    [parameter for _, parameter in samples],
                    retain_graph=True,
                    allow_unused=True,
                )
                for key, (value, weight) in terms.items():
                    gradients = torch.autograd.grad(
                        value,
                        [parameter for _, parameter in samples],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    records = {}
                    for (name, parameter), grad, base_grad in zip(samples, gradients, global_grad):
                        raw_norm = float(grad.float().norm()) if grad is not None else None
                        cosine = None
                        if grad is not None and base_grad is not None:
                            denominator = grad.float().norm() * base_grad.float().norm()
                            if float(denominator) > 0:
                                cosine = float((grad.float() * base_grad.float()).sum() / denominator)
                        records[name] = {
                            "raw_norm": raw_norm,
                            "weighted_norm": None if raw_norm is None else raw_norm * weight,
                            "cosine_to_source_seg": cosine,
                        }
                    components[key] = records
                row["per_loss_representative_parameter_gradients"] = components
                assert all(parameter.grad is None for parameter in [p for _, p in named])
                if step == 1:
                    gradients = torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True)
                    active[0] = False
                    rng_set(random_state)
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        check = legacy.loss_for_batch(
                            model, refiner, generator, image, target, valid, use_o
                        )[0]
                    check_gradients = torch.autograd.grad(check, [p for _, p in named], allow_unused=True)
                    gap = max(
                        float((left - right).abs().max())
                        for left, right in zip(gradients, check_gradients)
                        if left is not None and right is not None
                    )
                    row["observation_equivalence"] = {
                        "loss_abs": float((loss - check).abs()),
                        "gradient_max_abs": gap,
                        "none_pattern_equal": all(
                            (left is None) == (right is None)
                            for left, right in zip(gradients, check_gradients)
                        ),
                    }
                    if gap != 0 or float((loss - check).abs()) != 0:
                        raise RuntimeError("observation changed the first loss or gradient")
                    del gradients, check_gradients, check
                    rng_set(after_random)

            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite loss")
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            row["gradient_before_clip"] = grad_summary(named, [parameter.grad for _, parameter in named])
            if any(x['nonfinite_elements'] for x in row['gradient_before_clip'].values()):
                write(args.out/'FAILED_UPDATE.json',{'step':step,'reason':'nonfinite_unscaled_gradient','gradient':row['gradient_before_clip']})
                raise RuntimeError('nonfinite gradient before optimizer step')
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 12.0)
            torch.nn.utils.clip_grad_norm_(decoder_parameters, 5.0)
            row["gradient_after_clip"] = grad_summary(named, [parameter.grad for _, parameter in named])
            row["gradient_optimizer_receives"] = row["gradient_after_clip"]
            before = {name: parameter.detach().clone() for name, parameter in named}
            scaler.step(optimizer)
            scaler.update()
            skipped = float(scaler.get_scale()) < scale_before
            successful += int(not skipped)
            updates_by_module = {}
            for name, parameter in named:
                delta = parameter.detach() - before[name]
                group = grouping(name)
                update = updates_by_module.setdefault(
                    group,
                    {
                        "parameter_squared": 0.0,
                        "update_squared": 0.0,
                        "cumulative_squared": 0.0,
                        "changed": 0,
                        "elements": 0,
                        "exp_avg_squared": 0.0,
                        "exp_avg_sq_sum": 0.0,
                        "optimizer_steps": [],
                    },
                )
                update["parameter_squared"] += float(before[name].float().square().sum())
                update["update_squared"] += float(delta.float().square().sum())
                update["cumulative_squared"] += float(
                    (parameter.detach().cpu() - initial[name]).float().square().sum()
                )
                update["changed"] += int((delta != 0).sum())
                update["elements"] += parameter.numel()
                state = optimizer.state.get(parameter)
                if state:
                    update["optimizer_steps"].append(state_step(state))
                    update["exp_avg_squared"] += float(state["exp_avg"].float().square().sum())
                    update["exp_avg_sq_sum"] += float(state["exp_avg_sq"].float().sum())
            for update in updates_by_module.values():
                update["parameter_norm"] = update.pop("parameter_squared") ** 0.5
                update["update_norm"] = update.pop("update_squared") ** 0.5
                update["cumulative_norm"] = update.pop("cumulative_squared") ** 0.5
                update["changed_fraction"] = update.pop("changed") / update["elements"]
                update["relative_update"] = (
                    update["update_norm"] / update["parameter_norm"]
                    if update["parameter_norm"] > 1e-8
                    else None
                )
                update["exp_avg_norm"] = update.pop("exp_avg_squared") ** 0.5
                update["optimizer_step_range"] = (
                    [min(update["optimizer_steps"]), max(update["optimizer_steps"])]
                    if update["optimizer_steps"]
                    else None
                )
                del update["optimizer_steps"]
            if replay_reference is not None and step<=len(replay_reference):
                differences={n:abs(u['update_norm']-replay_reference[step-1]['actual_parameter_updates'][n]['update_norm']) for n,u in updates_by_module.items()}
                row['replay_update_norm_abs_difference']=differences
                if not lock.get('allow_approximate_replay') and max(differences.values())>lock.get('replay_numeric_tolerance',0.):
                    write(args.out/'REPLAY_MISMATCH.json',{'step':step,'differences':differences})
                    raise RuntimeError('replay actual update tolerance exceeded')
            cases.add(str(batch["case"][0]))
            torch.cuda.synchronize()
            elapsed_step = time.monotonic() - tick
            timings.append(elapsed_step)
            row.update(
                {
                    "actual_parameter_updates": updates_by_module,
                    "per_module_true_updates": updates_by_module,
                    "successful_optimizer_steps": successful,
                    "AMP_skipped": skipped,
                    "amp_scale_before": scale_before,
                    "amp_scale_after": float(scaler.get_scale()),
                    "unique_cases": len(cases),
                    "seconds": elapsed_step,
                    "explicit_o_weight": stats(refiner.enc1.net[0].weight[:, 6]),
                    "relation_optimizer_step": state_step(optimizer.state[relation_parameter])
                    if relation_parameter in optimizer.state
                    else None,
                }
            )
            if step == projection_step:
                median_step = statistics.median(timings[-5:])
                projected = (
                    time.monotonic() - start
                    + median_step * (updates - step)
                    + final_validation_reserve
                )
                row["runtime_projection"] = {
                    "check_step": step,
                    "median_last5_step_seconds": median_step,
                    "final_validation_reserve_seconds": final_validation_reserve,
                    "projected_total_seconds": projected,
                    "max_seconds": max_seconds,
                    "passed": projected <= max_seconds,
                }
                if projected > max_seconds:
                    raise RuntimeError({"runtime_projection": row["runtime_projection"]})
            step_rows.append(row)
            with (args.out / "TRAINING_TRACE.jsonl").open("a", encoding="utf-8") as trace:
                trace.write(json.dumps(row, allow_nan=False) + "\n")
            if skipped:
                raise RuntimeError('AMP skipped update; stop without further training')
            print(
                json.dumps(
                    {
                        "step": step,
                        "mode": identity_mode,
                        "loss": row["loss"],
                        "seconds": elapsed_step,
                        "successful": successful,
                        "relation_step": row["relation_optimizer_step"],
                    }
                ),
                flush=True,
            )
            del before

    finally:
        for handle in handles:
            handle.remove()

    if successful != updates:
        raise RuntimeError(f"AMP skipped optimizer updates: {successful}/{updates}")
    sampler_metadata.update(
        {
            "final_generator_state_sha256": tensor_sha(sampler_generator.get_state()),
            "iterator_boundary_states": iterator_boundary_states,
        }
    )
    write(args.out / "SAMPLER_AUDIT.json", sampler_metadata)

    final_validation = None
    if lock.get("full_validation", True):
        validation_set = legacy.NativeWristDataset(data, "val", False)
        final_validation = legacy.collect(
            model,
            refiner,
            generator,
            DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0),
            device,
            use_o,
            legacy.top_training_pairs(data, 15),
        )
        write(args.out / "FINAL_VALIDATION.json", final_validation)

    torch.save(
        {
            "model": model.state_dict(),
            "refiner": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_get(),
            "updates": successful,
            "identity_mode": identity_mode,
        },
        args.out / "diagnostic_final.pth",
    )
    write(
        args.out / "RESULT.json",
        {
            "status": "COMPLETE",
            "identity_mode": identity_mode,
            "updates": successful,
            "attempted": len(timings),
            "cycles": 2,
            "epoch_length": epoch_length,
            "unique_cases": len(cases),
            "seconds": time.monotonic() - start,
            "step_seconds": timings,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "test_used": False,
            "validation": "final_only",
            "initial_function_parity": parity,
            "initial_parameter_sha": "INITIAL_PARAMETER_SHA.json",
            "sampler_audit": "SAMPLER_AUDIT.json",
            "old_adam_state_step": old_optimizer_steps,
            "relation_state_initially_empty": True,
            "relation_state_final_step": state_step(optimizer.state[relation_parameter]),
            "matched_reference": matched_reference is not None,
            "final_validation_written": final_validation is not None,
            "lock_sha256": legacy.sha(args.lock),
        },
    )


if __name__ == "__main__":
    main()
