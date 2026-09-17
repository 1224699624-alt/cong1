"""nnU-Net v2 trainer for the R254 HA-PDSP + PCR-Loss prototype.

This file is copied into nnU-Net's trainer package by the isolated launcher so
that nnUNetv2_train can discover it. The prior only scales gradients on pixels
whose transformed hard target is already background; it never creates a seam
target and is not used during inference.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


FEATURE_COUNT = 7


def _case_stem(key: str) -> str:
    stem = str(key)
    for prefix in ("train_", "val_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return stem


def _read_metadata(path: Path) -> dict[str, tuple[float, bool]]:
    import csv

    result = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            result[str(row["id"])] = (
                float(row["boneage"]),
                str(row["male"]).strip().lower() in {"1", "true", "yes"},
            )
    return result


class ConditionalRelationPrior:
    def __init__(self, prior_path: Path, metadata_path: Path, age_bandwidth: float = 24.0):
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
        if "train_labels" not in str(payload.get("source_label_dir", "")):
            raise RuntimeError("Prior artifact does not declare train-only labels")
        samples = payload["samples"]
        self.features = np.asarray([x["features"] for x in samples], dtype=np.float64)
        self.ages = np.asarray([x["boneage"] for x in samples], dtype=np.float64)
        self.males = np.asarray([x["male"] for x in samples], dtype=bool)
        self.case_ids = np.asarray([str(x["case"]) for x in samples])
        unique_cases, counts = np.unique(self.case_ids, return_counts=True)
        inverse_count = dict(zip(unique_cases.tolist(), counts.tolist()))
        self.case_balance = np.asarray([1.0 / inverse_count[x] for x in self.case_ids], dtype=np.float64)
        first_index = np.asarray([int(np.flatnonzero(self.case_ids == case)[0]) for case in unique_cases])
        self.case_ages = self.ages[first_index]
        self.case_males = self.males[first_index]
        self.global_mean = np.asarray(payload["global_feature_mean"], dtype=np.float64)
        self.global_std = np.maximum(np.asarray(payload["global_feature_std"], dtype=np.float64), 1e-4)
        self.metadata = _read_metadata(metadata_path)
        self.age_bandwidth = float(age_bandwidth)
        self._condition_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}

    def coefficient(self, case: str, feature: np.ndarray, image_evidence: float) -> tuple[float, dict[str, float]]:
        meta = self.metadata.get(_case_stem(case))
        if meta is None:
            return 0.0, {"support": 0.0, "relation": 0.0, "image": float(image_evidence)}
        cache_key = _case_stem(case)
        if cache_key not in self._condition_cache:
            age, male = meta
            age_w = np.exp(-0.5 * ((self.ages - age) / self.age_bandwidth) ** 2)
            sex_w = np.where(self.males == male, 1.0, 0.25)
            weights = age_w * sex_w * self.case_balance
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-6:
                mean, std = self.global_mean, self.global_std
            else:
                mean = (weights[:, None] * self.features).sum(0) / weight_sum
                variance = (weights[:, None] * (self.features - mean) ** 2).sum(0) / weight_sum
                std = np.maximum(np.sqrt(variance), 0.20 * self.global_std)
            case_age_w = np.exp(-0.5 * ((self.case_ages - age) / self.age_bandwidth) ** 2)
            case_sex_w = np.where(self.case_males == male, 1.0, 0.25)
            case_weights = case_age_w * case_sex_w
            effective_cases = float(case_weights.sum() ** 2 / (np.square(case_weights).sum() + 1e-6))
            support = float(np.clip(effective_cases / 80.0, 0.0, 1.0))
            self._condition_cache[cache_key] = (mean, std, support)
        mean, std, support = self._condition_cache[cache_key]
        z2 = float(np.mean(np.square((feature - mean) / (std + 1e-6))))
        relation = float(np.exp(-0.5 * min(z2, 20.0)))
        image = float(np.clip(image_evidence, 0.0, 1.0))
        k = float(np.clip(support * relation * image, 0.0, 1.0))
        return k, {"support": support, "relation": relation, "image": image}


def _component_records(mask: np.ndarray, min_area: int = 20) -> list[dict[str, Any]]:
    labels, count = ndimage.label(mask)
    records = []
    for value, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        local = labels[component_slice] == value
        area = int(local.sum())
        if area < min_area:
            continue
        eroded = ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
        support_local = eroded if eroded.any() else local
        local_y, local_x = np.nonzero(local)
        support_y, support_x = np.nonzero(support_local)
        y_offset, x_offset = component_slice[0].start, component_slice[1].start
        yy, xx = local_y + y_offset, local_x + x_offset
        support_y, support_x = support_y + y_offset, support_x + x_offset
        boundary = local & ~ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
        by, bx = np.nonzero(boundary)
        by, bx = by + y_offset, bx + x_offset
        if len(bx) > 256:
            take = np.linspace(0, len(bx) - 1, 256).astype(int)
            by, bx = by[take], bx[take]
        records.append(
            {
                "pixels": (yy, xx),
                "support_pixels": (support_y, support_x),
                "area": float(area),
                "x": float(xx.mean()),
                "y": float(yy.mean()),
                "boundary": np.stack([by, bx], axis=1),
            }
        )
    return records


def _feature(a: dict[str, Any], b: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    diag = math.hypot(*shape)
    dx, dy = abs(a["x"] - b["x"]), abs(a["y"] - b["y"])
    distance = math.hypot(dx, dy)
    ra, rb = math.sqrt(a["area"] / math.pi), math.sqrt(b["area"] / math.pi)
    return np.asarray(
        [
            distance / diag,
            dx / diag,
            dy / diag,
            abs(math.log((a["area"] + 1) / (b["area"] + 1))),
            max(0.0, distance - ra - rb) / diag,
            ((a["x"] + b["x"]) * 0.5) / shape[1],
            ((a["y"] + b["y"]) * 0.5) / shape[0],
        ],
        dtype=np.float64,
    )


def _candidate_pairs(records: list[dict[str, Any]], max_pairs: int = 24) -> list[tuple[int, int]]:
    if len(records) < 2:
        return []
    coords = np.asarray([[r["x"], r["y"]] for r in records])
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    pairs = set()
    for i in range(len(records)):
        for j in np.argsort(distances[i])[:2]:
            pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs, key=lambda ij: distances[ij[0], ij[1]])[:max_pairs]


def _corridor(a: dict[str, Any], b: dict[str, Any], foreground: np.ndarray, radius: int = 3) -> np.ndarray:
    pa, pb = a["boundary"], b["boundary"]
    if not len(pa) or not len(pb):
        return np.zeros_like(foreground, dtype=bool)
    distance2 = np.square(pa[:, None, :] - pb[None, :, :]).sum(-1)
    ia, ib = np.unravel_index(int(distance2.argmin()), distance2.shape)
    y0, x0 = pa[ia]
    y1, x1 = pb[ib]
    steps = max(abs(int(y1) - int(y0)), abs(int(x1) - int(x0))) + 1
    yy = np.rint(np.linspace(y0, y1, steps)).astype(int)
    xx = np.rint(np.linspace(x0, x1, steps)).astype(int)
    line = np.zeros_like(foreground, dtype=bool)
    line[np.clip(yy, 0, line.shape[0] - 1), np.clip(xx, 0, line.shape[1] - 1)] = True
    region = ndimage.binary_dilation(line, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))
    return region & ~foreground


def _pair_weight_maps(
    target: np.ndarray,
    image: np.ndarray,
    probability: np.ndarray,
    case: str,
    prior: ConditionalRelationPrior,
    top_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    foreground = target > 0
    records = _component_records(foreground)
    predicted_labels, _ = ndimage.label(probability >= 0.5)
    prediction_backed = []
    for record in records:
        sy, sx = record["support_pixels"]
        ids = predicted_labels[sy, sx]
        ids = ids[ids > 0]
        if len(ids) == 0:
            continue
        values, counts = np.unique(ids, return_counts=True)
        matched_id = int(values[int(np.argmax(counts))])
        overlap = float(np.max(counts) / max(1, len(sy)))
        confidence = float(probability[sy, sx].mean())
        stability = float(np.mean(np.abs(probability[sy, sx] - 0.5) * 2.0))
        if overlap < 0.25 or confidence < 0.35:
            continue
        record["prediction_id"] = matched_id
        record["candidate_confidence"] = confidence
        record["candidate_stability"] = stability
        prediction_backed.append(record)
    records = prediction_backed
    pairs = _candidate_pairs(records)
    weight_map = np.zeros_like(probability, dtype=np.float32)
    bg_distance = ndimage.distance_transform_edt(~foreground)
    ks, supports, relations, evidences = [], [], [], []
    proposed = len(pairs)
    valid = 0
    for i, j in pairs:
        region = _corridor(records[i], records[j], foreground)
        reliability_map = np.clip((bg_distance - 1.0) / 3.0, 0.0, 1.0)
        valid_pixels = np.flatnonzero(region & (reliability_map >= 0.5))
        if len(valid_pixels) < 4:
            continue
        corridor_mean = float(image.ravel()[valid_pixels].mean())
        bone_values = np.concatenate([image[records[i]["pixels"]], image[records[j]["pixels"]]])
        bone_mean = float(np.median(bone_values)) if len(bone_values) else corridor_mean
        image_evidence = float(1.0 / (1.0 + math.exp(-np.clip((bone_mean - corridor_mean) / 0.5, -20, 20))))
        candidate_confidence = math.sqrt(records[i]["candidate_confidence"] * records[j]["candidate_confidence"])
        stability = math.sqrt(records[i]["candidate_stability"] * records[j]["candidate_stability"])
        k, parts = prior.coefficient(case, _feature(records[i], records[j], target.shape), image_evidence)
        k *= candidate_confidence * stability
        if k <= 0:
            continue
        count = int(np.clip(math.ceil(len(valid_pixels) * top_fraction), 4, 128))
        hard_order = np.argpartition(probability.ravel()[valid_pixels], -min(count, len(valid_pixels)))[-min(count, len(valid_pixels)) :]
        selected = valid_pixels[hard_order]
        reliability = reliability_map.ravel()[selected]
        pair_weights = (k * reliability).astype(np.float32)
        flat = weight_map.ravel()
        flat[selected] = np.maximum(flat[selected], pair_weights)
        valid += 1
        ks.append(k)
        supports.append(parts["support"])
        relations.append(parts["relation"])
        evidences.append(parts["image"])
    active = weight_map > 0
    stats = {
        "proposed_pairs": float(proposed),
        "valid_pairs": float(valid),
        "anchors": float(active.sum()),
        "k_mean": float(np.mean(ks)) if ks else 0.0,
        "support_mean": float(np.mean(supports)) if supports else 0.0,
        "relation_mean": float(np.mean(relations)) if relations else 0.0,
        "image_evidence_mean": float(np.mean(evidences)) if evidences else 0.0,
        "anchor_probability": float(probability[active].mean()) if active.any() else 0.0,
        "candidate_components": float(len(records)),
    }
    return weight_map, stats


class nnUNetTrainerHAPDSPPCR(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 15
        self.initial_lr = 1e-3
        self.num_iterations_per_epoch = 30
        self.num_val_iterations_per_epoch = 20
        np.random.seed(254)
        torch.manual_seed(254)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(254)
        root = Path(os.environ.get("R254_PROJECT_ROOT", Path.cwd()))
        self._prior_path = Path(os.environ.get("R254_PRIOR_JSON", root / "outputs/analysis/r254_hapdsp_train_prior.json"))
        self._metadata_path = Path(os.environ.get("R254_METADATA_CSV", root / "filtered_train.csv"))
        self._prior: ConditionalRelationPrior | None = None
        self._rho = float(os.environ.get("R254_PCR_RHO", "0.05"))
        self._alpha_max = float(os.environ.get("R254_ALPHA_MAX", "0.20"))
        self._top_fraction = float(os.environ.get("R254_TOP_FRACTION", "0.25"))
        self._warmup_epochs = int(os.environ.get("R254_WARMUP_EPOCHS", "5"))
        self._snapshot: dict[str, torch.Tensor] = {}

    def _get_prior(self) -> ConditionalRelationPrior:
        if self._prior is None:
            self._prior = ConditionalRelationPrior(self._prior_path, self._metadata_path)
            self.print_to_log_file(f"R254 prior loaded: {self._prior_path}")
        return self._prior

    @staticmethod
    def _robust_loss_one(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = (target[:, 0:1] > 0).float()
        probability = torch.softmax(logits.float(), dim=1)[:, 1:2]
        core = 1.0 - F.max_pool2d(1.0 - y, kernel_size=5, stride=1, padding=2)
        dilated = F.max_pool2d(y, kernel_size=7, stride=1, padding=3)
        far_bg = (dilated < 0.5).float()
        uncertain = torch.clamp(1.0 - core - far_bg, 0.0, 1.0)
        q_label = core + far_bg + 0.35 * uncertain

        intersection = torch.sum(q_label * probability * y)
        denominator = torch.sum(q_label * probability) + torch.sum(q_label * y)
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
        foreground_log_odds = logits[:, 1:2].float() - logits[:, 0:1].float()
        pixel_bce = F.binary_cross_entropy_with_logits(foreground_log_odds, y, reduction="none")
        fg_bce = torch.sum(pixel_bce * core) / (torch.sum(core) + 1e-6)
        bg_bce = torch.sum(pixel_bce * far_bg) / (torch.sum(far_bg) + 1e-6)
        p_target = torch.where(y > 0.5, probability, 1.0 - probability).clamp(1e-6, 1.0)
        gce_q = 0.7
        boundary_gce_map = (1.0 - torch.pow(p_target, gce_q)) / gce_q
        boundary_gce = torch.sum(boundary_gce_map * uncertain) / (torch.sum(uncertain) + 1e-6)
        total = dice_loss + 0.5 * (fg_bce + bg_bce) + 0.25 * boundary_gce
        return total, {
            "robust_dice": dice_loss,
            "core_bce": fg_bce,
            "far_bg_bce": bg_bce,
            "boundary_gce": boundary_gce,
            "boundary_fraction": uncertain.mean(),
        }

    def _robust_deep_supervision_loss(self, output, target):
        outputs = list(output) if isinstance(output, (list, tuple)) else [output]
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        weights = np.asarray([1.0 / (2**i) for i in range(len(outputs))], dtype=np.float64)
        weights /= weights.sum()
        total = outputs[0].sum() * 0.0
        high_stats = None
        for i, (logits, y) in enumerate(zip(outputs, targets)):
            scale_loss, stats = self._robust_loss_one(logits, y)
            total = total + float(weights[i]) * scale_loss
            if i == 0:
                high_stats = stats
        return total, high_stats or {}

    @staticmethod
    def _parameter_group(name: str) -> str:
        lower = name.lower()
        if "seg_layers" in lower or "output" in lower or "head" in lower:
            return "head"
        if "decoder" in lower:
            return "decoder"
        return "encoder"

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._snapshot = {name: p.detach().cpu().clone() for name, p in self.network.named_parameters() if p.requires_grad}

    def _parameter_dynamics(self) -> dict[str, float]:
        accum = {g: {"w2": 0.0, "d2": 0.0} for g in ("encoder", "decoder", "head")}
        for name, p in self.network.named_parameters():
            if name not in self._snapshot:
                continue
            group = self._parameter_group(name)
            current = p.detach().cpu().float()
            previous = self._snapshot[name].float()
            accum[group]["w2"] += float(torch.sum(current * current))
            accum[group]["d2"] += float(torch.sum((current - previous) ** 2))
        result = {}
        for group, values in accum.items():
            w = math.sqrt(values["w2"])
            d = math.sqrt(values["d2"])
            result[f"param_norm_{group}"] = w
            result[f"update_ratio_{group}"] = d / (w + 1e-12)
        return result

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        keys = list(batch.get("keys", ["unknown"] * len(data)))
        if isinstance(target, list):
            target = [x.to(self.device, non_blocking=True) for x in target]
            target_high = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target_high = target

        self.optimizer.zero_grad(set_to_none=True)
        context = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else nullcontext()
        with context:
            output = self.network(data)
            base_loss, robust_stats = self._robust_deep_supervision_loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
            target_np = target_high[:, 0].detach().cpu().numpy()
            image_np = data[:, 0].detach().float().cpu().numpy()
            probability_np = probability.detach().cpu().numpy()
            maps, batch_stats = [], []
            prior = self._get_prior()
            for b in range(len(data)):
                wm, stats = _pair_weight_maps(target_np[b], image_np[b], probability_np[b], keys[b], prior, self._top_fraction)
                maps.append(wm)
                batch_stats.append(stats)
            weight_map = torch.from_numpy(np.stack(maps)).to(logits.device, dtype=torch.float32)
            z_pair = weight_map.float().sum()
            if float(z_pair.detach()) > 1e-6:
                foreground_log_odds = logits[:, 1].float() - logits[:, 0].float()
                pair_loss = (weight_map * F.softplus(foreground_log_odds)).sum() / (z_pair + 1e-6)
                g_base = torch.autograd.grad(base_loss, logits, retain_graph=True, allow_unused=True)[0]
                g_pair = torch.autograd.grad(pair_loss, logits, retain_graph=True, allow_unused=True)[0]
                gb = torch.linalg.vector_norm(g_base.float()) if g_base is not None else torch.zeros((), device=logits.device)
                gp = torch.linalg.vector_norm(g_pair.float()) if g_pair is not None else torch.zeros((), device=logits.device)
                if g_base is not None and g_pair is not None:
                    gradient_cosine = torch.sum(g_base.float() * g_pair.float()) / (gb * gp + 1e-8)
                else:
                    gradient_cosine = torch.zeros((), device=logits.device)
                warmup = float(np.clip((self.current_epoch + 1) / max(1, self._warmup_epochs), 0.0, 1.0))
                alpha = torch.clamp(self._rho * gb.detach() / (gp.detach() + 1e-8), 0.0, self._alpha_max) * warmup
                loss = base_loss + alpha * pair_loss
            else:
                pair_loss = logits.sum() * 0.0
                gb = gp = alpha = gradient_cosine = torch.zeros((), device=logits.device)
                loss = base_loss

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        grad_groups = {g: 0.0 for g in ("encoder", "decoder", "head")}
        for name, p in self.network.named_parameters():
            if p.grad is not None:
                grad_groups[self._parameter_group(name)] += float(torch.sum(p.grad.detach().float() ** 2))
        clip_norm = float(torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12).detach().cpu())
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()

        means = {key: float(np.mean([x[key] for x in batch_stats])) for key in batch_stats[0]} if batch_stats else {}
        core = np.stack(
            [ndimage.binary_erosion(sample > 0, structure=np.ones((3, 3), dtype=bool)) for sample in target_np],
            axis=0,
        )
        result = {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "pair_loss": pair_loss.detach().cpu().numpy(),
            "alpha": alpha.detach().cpu().numpy(),
            "g_base": gb.detach().cpu().numpy(),
            "g_pair": gp.detach().cpu().numpy(),
            "gradient_cosine": gradient_cosine.detach().cpu().numpy(),
            "z_pair": z_pair.detach().cpu().numpy(),
            "core_probability": np.asarray(float(probability_np[core].mean()) if core.any() else 0.0),
            "clip_norm": np.asarray(clip_norm),
            "amp_scale": np.asarray(float(self.grad_scaler.get_scale()) if self.grad_scaler is not None else 1.0),
            "gpu_memory_mb": np.asarray(float(torch.cuda.max_memory_allocated(self.device) / (1024**2)) if self.device.type == "cuda" else 0.0),
        }
        result.update({k: np.asarray(v) for k, v in means.items()})
        result.update({k: v.detach().cpu().numpy() for k, v in robust_stats.items()})
        result.update({f"grad_norm_{g}": np.asarray(math.sqrt(v)) for g, v in grad_groups.items()})
        return result

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        scalar_keys = [k for k in train_outputs[0] if k != "loss"] if train_outputs else []
        summary = {k: float(np.mean([float(np.asarray(x[k])) for x in train_outputs])) for k in scalar_keys}
        summary.update(self._parameter_dynamics())
        summary.update({"epoch": int(self.current_epoch), "lr": float(self.optimizer.param_groups[0]["lr"])})
        path = Path(self.output_folder) / "r254_training_dynamics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        self.print_to_log_file("R254 dynamics", json.dumps(summary, sort_keys=True))
        self._snapshot.clear()


class nnUNetTrainerHAPDSPPCRSanity(nnUNetTrainerHAPDSPPCR):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 25
        self.num_val_iterations_per_epoch = 10


class nnUNetTrainerR254NativeContinuation(nnUNetTrainer):
    """Fair control: identical checkpoint, epochs and LR without HA-PDSP/PCR."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 15
        self.initial_lr = 1e-3
        self.num_iterations_per_epoch = 30
        self.num_val_iterations_per_epoch = 20
        np.random.seed(254)
        torch.manual_seed(254)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(254)
