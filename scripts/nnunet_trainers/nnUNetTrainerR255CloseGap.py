"""R255 nnU-Net trainers for development-conditioned local bridge suppression.

The method is training-only. It derives within-image bone supports from the
augmented binary target, assigns stop-gradient coefficients using train-only
bone-age statistics, local geometry, X-ray separation evidence and a detached
widest-path bridge score, then applies background loss only where target y=0.
Inference remains the unmodified nnU-Net network.
"""

from __future__ import annotations

import csv
import heapq
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
from scipy.spatial import cKDTree
from torch import autocast

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler


def _case_stem(key: str) -> str:
    key = str(key)
    return key[6:] if key.startswith("train_") else key[4:] if key.startswith("val_") else key


class DevelopmentRisk:
    def __init__(self, metadata_csv: Path, gate_csv: Path, bandwidth: float = 24.0):
        metadata = {}
        with metadata_csv.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                metadata[str(row["id"])] = (float(row["boneage"]), str(row["male"]).lower() in {"true", "1", "yes"})
        pair_rows = []
        with gate_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["instance_i"] == row["instance_j"]:
                    continue
                case = str(row["case"])
                if case not in metadata:
                    continue
                age, male = metadata[case]
                pair_rows.append((case, age, male, float(row["midpoint_v"]), float(row["relative_gap"]), float(row["local_width"]), float(1.0 <= float(row["gap_px"]) <= 4.999 and float(row["xray_pass"]) > 0)))
        if not pair_rows:
            raise RuntimeError("R255 development prior has no usable Gate A pair rows")
        self.cases = np.asarray([r[0] for r in pair_rows])
        self.ages = np.asarray([r[1] for r in pair_rows], dtype=np.float64)
        self.males = np.asarray([r[2] for r in pair_rows], dtype=bool)
        self.context = np.asarray([[r[3], r[4], math.log1p(r[5])] for r in pair_rows], dtype=np.float64)
        self.targets = np.asarray([r[6] for r in pair_rows], dtype=np.float64)
        self.context_scale = np.maximum(np.quantile(self.context, 0.75, axis=0) - np.quantile(self.context, 0.25, axis=0), [0.08, 0.05, 0.15])
        self.metadata = metadata
        self.bandwidth = float(bandwidth)
        self.cache: dict[tuple, tuple[float, float]] = {}

    def score(self, key: str, midpoint_v: float, relative_gap: float, width: float) -> tuple[float, float]:
        case = _case_stem(key)
        if case not in self.metadata:
            raise KeyError(f"R255 metadata missing training case {case!r}")
        cache_key = (case, round(midpoint_v, 2), round(relative_gap, 2), round(width, 1))
        if cache_key in self.cache:
            return self.cache[cache_key]
        age, male = self.metadata[case]
        query = np.asarray([midpoint_v, relative_gap, math.log1p(width)])
        context_distance = np.square((self.context - query) / self.context_scale).sum(axis=1)
        weights = np.exp(-0.5 * ((self.ages - age) / self.bandwidth) ** 2 - 0.5 * context_distance)
        weights *= np.where(self.males == male, 1.0, 0.25)
        weights *= self.cases != case  # leave-one-case-out prior
        ess = float(weights.sum() ** 2 / (np.square(weights).sum() + 1e-9))
        close_probability = float(np.sum(weights * self.targets) / (weights.sum() + 1e-9))
        support = float(np.clip(ess / 30.0, 0.0, 1.0))
        self.cache[cache_key] = (float(np.clip(close_probability * support, 0.0, 1.0)), ess)
        return self.cache[cache_key]


def _component_records(mask: np.ndarray, min_area: int = 24) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = ndimage.label(mask)
    records = []
    for value, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        local = labels[component_slice] == value
        ly, lx = np.nonzero(local)
        area = len(lx)
        if area < min_area:
            continue
        y0, x0 = component_slice[0].start, component_slice[1].start
        yy, xx = ly + y0, lx + x0
        boundary = local & ~ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
        by, bx = np.nonzero(boundary)
        by, bx = by + y0, bx + x0
        if len(bx) > 384:
            take = np.linspace(0, len(bx) - 1, 384).astype(int)
            by, bx = by[take], bx[take]
        centered = np.stack([yy, xx], axis=1).astype(np.float64)
        centered -= centered.mean(0, keepdims=True)
        eigvals = np.maximum(np.linalg.eigvalsh(centered.T @ centered / max(1, area - 1)), 1e-6)
        width = float(max(2.0, 4.0 * math.sqrt(float(eigvals.min()))))
        eroded = ndimage.binary_erosion(local, structure=np.ones((3, 3), dtype=bool))
        sy, sx = np.nonzero(eroded if eroded.any() else local)
        records.append(
            {
                "value": value,
                "centroid_y": float(yy.mean()),
                "centroid_x": float(xx.mean()),
                "boundary": np.stack([by, bx], axis=1),
                "support_y": sy + y0,
                "support_x": sx + x0,
                "width": width,
            }
        )
    return labels, records


def _nearest(a: dict[str, Any], b: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    tree = cKDTree(b["boundary"])
    distances, indices = tree.query(a["boundary"], k=1)
    idx = int(np.argmin(distances))
    return a["boundary"][idx], b["boundary"][int(indices[idx])], float(distances[idx])


def _capsule(
    p0: np.ndarray, p1: np.ndarray, radius: int, shape: tuple[int, int]
) -> tuple[tuple[slice, slice], np.ndarray, np.ndarray, np.ndarray]:
    steps = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    yy = np.rint(np.linspace(p0[0], p1[0], steps)).astype(int)
    xx = np.rint(np.linspace(p0[1], p1[1], steps)).astype(int)
    yy, xx = np.clip(yy, 0, shape[0] - 1), np.clip(xx, 0, shape[1] - 1)
    y0, y1 = max(0, int(yy.min()) - radius), min(shape[0], int(yy.max()) + radius + 1)
    x0, x1 = max(0, int(xx.min()) - radius), min(shape[1], int(xx.max()) + radius + 1)
    line = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    line[yy - y0, xx - x0] = True
    capsule = ndimage.binary_dilation(line, structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool))
    return (slice(y0, y1), slice(x0, x1)), line, capsule, ndimage.distance_transform_edt(~line)


def _xray_score(image: np.ndarray, a: dict[str, Any], b: dict[str, Any], gap: np.ndarray) -> float:
    if gap.size == 0:
        return 0.0
    gap = gap.astype(np.float64)
    ia = image[a["support_y"], a["support_x"]].astype(np.float64)
    ib = image[b["support_y"], b["support_x"]].astype(np.float64)
    bone_a, bone_b, gap_i = float(np.median(ia)), float(np.median(ib)), float(np.median(gap))
    values = np.concatenate([ia, ib, gap])
    sigma = float(np.median(np.abs(values - np.median(values))) * 1.4826 + 1e-3)
    da, db = bone_a - gap_i, bone_b - gap_i
    if da * db <= 0:
        return 0.0
    valley = float(np.clip(min(abs(da), abs(db)) / (abs(bone_a - bone_b) + sigma), 0.0, 1.0))
    double = float(np.clip(math.sqrt(abs(da * db)) / sigma, 0.0, 1.0))
    return valley * double


def _widest_path(probability: np.ndarray, allowed: np.ndarray, source: np.ndarray, target: np.ndarray) -> float:
    active = allowed | source | target
    if not source.any() or not target.any() or not active.any():
        return 0.0
    # The capsule occupies only a tiny fraction of a full hand radiograph.
    # Crop before allocating Dijkstra state; allocating a 1536x1024 `best`
    # array for every candidate pair makes the training-time prior needlessly
    # CPU- and memory-heavy without changing the path definition.
    yy, xx = np.nonzero(active)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    probability = probability[y0:y1, x0:x1]
    allowed = allowed[y0:y1, x0:x1]
    source = source[y0:y1, x0:x1]
    target = target[y0:y1, x0:x1]
    best = np.full(probability.shape, -1.0, dtype=np.float32)
    heap: list[tuple[float, int, int]] = []
    for y, x in np.argwhere(source):
        score = float(probability[y, x])
        best[y, x] = score
        heapq.heappush(heap, (-score, int(y), int(x)))
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    h, w = probability.shape
    while heap:
        neg_score, y, x = heapq.heappop(heap)
        score = -neg_score
        if score + 1e-8 < best[y, x]:
            continue
        if target[y, x]:
            return score
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and allowed[ny, nx]:
                candidate = min(score, float(probability[ny, nx]))
                if candidate > best[ny, nx] + 1e-8:
                    best[ny, nx] = candidate
                    heapq.heappush(heap, (-candidate, ny, nx))
    return 0.0


def _weight_map(
    target: np.ndarray,
    image: np.ndarray,
    probability: np.ndarray,
    age_risk: float | Any,
    *,
    roi_v_min: float = 0.45,
    max_gap: float = 12.0,
    max_relative_gap: float = 0.40,
    rho: float = 0.12,
    tau_close: float = 0.15,
    xray_floor: float = 0.15,
    max_pairs: int = 24,
    max_anchors: int = 96,
) -> tuple[np.ndarray, dict[str, float]]:
    empty_stats = {
        "proposed_pairs": 0.0,
        "valid_pairs": 0.0,
        "anchors": 0.0,
        "bridge_mean": 0.0,
        "xray_mean": 0.0,
        "k_mean": 0.0,
        "anchor_probability": 0.0,
        "anchor_gt_foreground_fraction": 0.0,
        "development_risk_mean": 0.0,
        "support_ess_mean": 0.0,
    }
    foreground = target > 0
    labels, records = _component_records(foreground)
    weights = np.zeros_like(probability, dtype=np.float32)
    if len(records) < 2:
        return weights, empty_stats
    coords = np.asarray([[r["centroid_y"], r["centroid_x"]] for r in records])
    distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    pairs = set()
    for i in range(len(records)):
        neighbors = [int(j) for j in np.argsort(distances[i]) if int(j) != i and np.isfinite(distances[i, int(j)])]
        for j in neighbors[:4]:
            pairs.add(tuple(sorted((i, int(j)))))
    pairs = sorted(pairs, key=lambda ij: distances[ij])[:max_pairs]
    fy, fx = np.nonzero(foreground)
    y0, y1 = float(fy.min()), float(fy.max())
    bridges, xrays, ks, development_risks, support_esses = [], [], [], [], []
    valid_pairs = 0
    for i, j in pairs:
        a, b = records[i], records[j]
        p0, p1, boundary_distance = _nearest(a, b)
        gap = max(0.0, boundary_distance - 1.0)
        width = min(a["width"], b["width"])
        relative_gap = gap / max(width, 1e-6)
        midpoint_v = (((a["centroid_y"] + b["centroid_y"]) * 0.5) - y0) / max(1.0, y1 - y0)
        if midpoint_v < roi_v_min or gap > max_gap or relative_gap > max_relative_gap:
            continue
        radius = max(1, int(round(rho * width)))
        region, line, capsule, line_distance = _capsule(p0, p1, radius, target.shape)
        labels_local = labels[region]
        foreground_local = foreground[region]
        probability_local = probability[region]
        values = labels_local[capsule]
        if np.any((values > 0) & (values != a["value"]) & (values != b["value"])):
            continue
        target_local = target[region]
        legal = capsule & (target_local == 0)
        if not legal.any():
            continue
        q_xray = _xray_score(image, a, b, image[region][legal])
        if q_xray < xray_floor:
            continue
        support_a = labels_local == a["value"]
        support_b = labels_local == b["value"]
        source = legal & ndimage.binary_dilation(support_a, structure=np.ones((3, 3), dtype=bool))
        destination = legal & ndimage.binary_dilation(support_b, structure=np.ones((3, 3), dtype=bool))
        bridge = _widest_path(probability_local, legal, source, destination)
        if bridge <= 0:
            continue
        pair_risk, pair_ess = age_risk(midpoint_v, relative_gap, width) if callable(age_risk) else (float(age_risk), 0.0)
        q_close = math.exp(-relative_gap / tau_close)
        k = float(np.clip(pair_risk * q_close * bridge, 0.0, 1.0))
        indices = np.flatnonzero(legal)
        if len(indices) > max_anchors:
            hard = np.argpartition(probability_local.ravel()[indices], -max_anchors)[-max_anchors:]
            indices = indices[hard]
        q_corridor = np.exp(-line_distance.ravel()[indices] / max(1.0, radius)).astype(np.float32)
        pair_weights = (k * q_xray * q_corridor).astype(np.float32)
        local_weights = weights[region]
        anchor_y, anchor_x = np.unravel_index(indices, legal.shape)
        local_weights[anchor_y, anchor_x] = np.maximum(local_weights[anchor_y, anchor_x], pair_weights)
        valid_pairs += 1
        bridges.append(bridge)
        xrays.append(q_xray)
        ks.append(k)
        development_risks.append(pair_risk)
        support_esses.append(pair_ess)
    active = weights > 0
    if active.any() and not np.all(target[active] == 0):
        raise AssertionError("R255 close-gap anchors must be strict target==0 pixels")
    return weights, {
        "proposed_pairs": float(len(pairs)),
        "valid_pairs": float(valid_pairs),
        "anchors": float(active.sum()),
        "bridge_mean": float(np.mean(bridges)) if bridges else 0.0,
        "xray_mean": float(np.mean(xrays)) if xrays else 0.0,
        "k_mean": float(np.mean(ks)) if ks else 0.0,
        "anchor_probability": float(probability[active].mean()) if active.any() else 0.0,
        "anchor_gt_foreground_fraction": float(foreground[active].mean()) if active.any() else 0.0,
        "development_risk_mean": float(np.mean(development_risks)) if development_risks else 0.0,
        "support_ess_mean": float(np.mean(support_esses)) if support_esses else 0.0,
    }


class _R255Base(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.plans_manager.dataset_name != "Dataset202_TSRS_RSNAEpiphysis2D" or "Articular" in self.plans_manager.dataset_name:
            raise RuntimeError(f"R255 permits Dataset202_TSRS_RSNAEpiphysis2D only, got {self.plans_manager.dataset_name}")
        self.initial_lr = 1e-4
        self.num_epochs = 2
        self.num_iterations_per_epoch = 25
        self.num_val_iterations_per_epoch = 10
        np.random.seed(255)
        torch.manual_seed(255)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(255)

    def configure_optimizers(self):
        # Warm-up stops at epoch 2, but all branches use one pre-registered
        # four-epoch PolyLR horizon so a full optimizer checkpoint can fork.
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, 4)


class nnUNetTrainerR255Warmup(_R255Base):
    pass


class nnUNetTrainerR255NativeSanity(_R255Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4


class nnUNetTrainerR255CloseGapSanity(_R255Base):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 4
        root = Path(os.environ.get("R255_PROJECT_ROOT", Path.cwd()))
        self.development_risk = DevelopmentRisk(
            Path(os.environ.get("R255_METADATA_CSV", root / "outputs/metadata/r255/filtered_train.csv")),
            Path(os.environ.get("R255_GATE_CSV", root / "outputs/analysis/r255_gate_a/r255_close_gap_pairs.csv")),
        )
        self._calibration_steps = int(os.environ.get("R255_CALIBRATION_STEPS", "16"))
        self._gradient_ratios: list[float] = []
        self._calibration_cases: list[str] = []
        self._alpha: float | None = None
        self._gmax = float(os.environ.get("R255_GMAX", "0.25"))
        state_path = Path(self.output_folder) / "r255_calibration_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self._alpha = float(state["alpha"])
            self._gradient_ratios = [float(x) for x in state.get("gradient_ratios", [])]
            self._calibration_cases = [str(x) for x in state.get("calibration_cases", [])]

    def _gradient_clipped_bce(self, logits: torch.Tensor) -> torch.Tensor:
        z0 = math.log(self._gmax / (1.0 - self._gmax))
        z = logits.float()
        base = F.softplus(z)
        tail = F.softplus(torch.as_tensor(z0, device=z.device)) + self._gmax * (z - z0)
        return torch.where(torch.sigmoid(z) <= self._gmax, base, tail)

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if "keys" not in batch or len(batch["keys"]) != len(data):
            raise RuntimeError("R255 requires nnU-Net case keys for the training-only development prior")
        keys = list(batch["keys"])
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
            base_loss = self.loss(output, target)
            logits = output[0] if isinstance(output, (list, tuple)) else output
            probability = torch.softmax(logits.float(), dim=1)[:, 1]
            target_np = target_high[:, 0].detach().cpu().numpy()
            image_np = data[:, 0].detach().float().cpu().numpy()
            probability_np = probability.detach().cpu().numpy()
            maps, pair_stats = [], []
            for b in range(len(data)):
                risk_fn = lambda midpoint_v, relative_gap, width, key=keys[b]: self.development_risk.score(key, midpoint_v, relative_gap, width)
                wm, stats = _weight_map(target_np[b], image_np[b], probability_np[b], risk_fn)
                maps.append(wm)
                pair_stats.append(stats)
            weight_map = torch.from_numpy(np.stack(maps)).to(logits.device, dtype=torch.float32)
            z_pair = weight_map.sum()
            if float(z_pair.detach()) > 1e-6:
                log_odds = logits[:, 1].float() - logits[:, 0].float()
                loss_map = weight_map * self._gradient_clipped_bce(log_odds)
                numerator = loss_map.flatten(1).sum(1)
                denominator = weight_map.flatten(1).sum(1)
                valid_images = denominator > 1e-6
                pair_loss = (numerator[valid_images] / (denominator[valid_images] + 1e-6)).mean()
                g_base = torch.autograd.grad(base_loss, logits, retain_graph=True, allow_unused=True)[0]
                g_pair = torch.autograd.grad(pair_loss, logits, retain_graph=True, allow_unused=True)[0]
                gb = torch.linalg.vector_norm(g_base.float()) if g_base is not None else torch.zeros((), device=logits.device)
                gp = torch.linalg.vector_norm(g_pair.float()) if g_pair is not None else torch.zeros((), device=logits.device)
                ratio = float((gb.detach() / (gp.detach() + 1e-8)).cpu())
                if self._alpha is None:
                    self._gradient_ratios.append(ratio)
                    self._calibration_cases.extend(_case_stem(k) for k in keys)
                    if len(self._gradient_ratios) >= self._calibration_steps:
                        self._alpha = float(np.clip(0.05 * np.median(self._gradient_ratios), 1e-4, 0.10))
                        state = {"alpha": self._alpha, "gradient_ratios": self._gradient_ratios, "calibration_cases": self._calibration_cases}
                        state_path = Path(self.output_folder) / "r255_calibration_state.json"
                        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                    alpha = 0.0
                else:
                    alpha = self._alpha
                loss = base_loss + alpha * pair_loss
            else:
                pair_loss = logits.sum() * 0.0
                gb = gp = torch.zeros((), device=logits.device)
                alpha = 0.0 if self._alpha is None else self._alpha
                loss = base_loss
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        clip_norm = float(torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12).detach().cpu())
        if self.grad_scaler is not None:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            self.optimizer.step()
        means = {key: float(np.mean([x[key] for x in pair_stats])) for key in pair_stats[0]} if pair_stats else {}
        result = {
            "loss": loss.detach().cpu().numpy(),
            "base_loss": base_loss.detach().cpu().numpy(),
            "pair_loss": pair_loss.detach().cpu().numpy(),
            "alpha": np.asarray(alpha),
            "g_base": gb.detach().cpu().numpy(),
            "g_pair": gp.detach().cpu().numpy(),
            "z_pair": z_pair.detach().cpu().numpy(),
            "age_risk": np.asarray(float(np.mean([x["development_risk_mean"] for x in pair_stats]))),
            "support_ess": np.asarray(float(np.mean([x["support_ess_mean"] for x in pair_stats]))),
            "calibration_count": np.asarray(len(self._gradient_ratios)),
            "clip_norm": np.asarray(clip_norm),
        }
        result.update({k: np.asarray(v) for k, v in means.items()})
        return result

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if not train_outputs:
            return
        keys = [k for k in train_outputs[0] if k != "loss"]
        summary = {k: float(np.mean([float(np.asarray(x[k])) for x in train_outputs])) for k in keys}
        summary.update({"epoch": int(self.current_epoch), "alpha_frozen": self._alpha})
        path = Path(self.output_folder) / "r255_training_dynamics.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        self.print_to_log_file("R255 dynamics", json.dumps(summary, sort_keys=True))
