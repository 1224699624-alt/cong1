#!/usr/bin/env python3
"""R210 component-preserving instance split gate.

This is a train/val-only anchor-conditioned split experiment. It learns a cut
likelihood inside R110 masks, then accepts cuts only when validation GT proves
that component, recall, and overlap guardrails are preserved. clean-test-v2 is
not touched unless --allow-apply-clean-test is explicitly set after a passed
validation gate.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

from run_r201_unified_eval import build_gt_cache, compute_metrics
from run_r209_component_preserving_feasibility_audit import analyze_component_matches
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like, write_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R210 component-preserving split gate.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--apply-dataset", default="TSRS_RSNA-Epiphysis_clean_test_v2")
    parser.add_argument("--apply-raw-root", type=Path, default=Path("data/raw_variants"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--apply-split", default="test")
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--apply-anchor-exp", default="r110_r100_r108_patch_basic")
    parser.add_argument("--output-exp", default="r210_component_preserving_instance_split")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/r210_component_preserving_instance_split/r210_split_mlp.pt"))
    parser.add_argument("--val-summary-json", type=Path, default=Path("outputs/analysis/r210_component_preserving_instance_split_val_summary.json"))
    parser.add_argument("--val-csv", type=Path, default=Path("outputs/analysis/r210_component_preserving_instance_split_val_per_image.csv"))
    parser.add_argument("--apply-metrics-json", type=Path, default=Path("outputs/analysis/r210_component_preserving_instance_split_clean_test_v2_metrics.json"))
    parser.add_argument("--max-train-images", type=int, default=0)
    parser.add_argument("--max-val-images", type=int, default=0)
    parser.add_argument("--max-apply-images", type=int, default=0)
    parser.add_argument("--samples-per-image", type=int, default=4096)
    parser.add_argument("--samples-per-merge-image", type=int, default=12288)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0)
    parser.add_argument("--split-radii", default="2,3,4")
    parser.add_argument("--prob-thresholds", default="0.55,0.65,0.75,0.85")
    parser.add_argument("--min-cut-area", type=int, default=6)
    parser.add_argument("--max-cut-frac", type=float, default=0.012)
    parser.add_argument("--dice-tol", type=float, default=0.003)
    parser.add_argument("--iou-tol", type=float, default=0.003)
    parser.add_argument("--recall-tol", type=float, default=0.005)
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--surface-tol", type=float, default=2.0)
    parser.add_argument("--surface-tol-extra", type=float, default=5.0)
    parser.add_argument("--min-overlap-frac", type=float, default=0.05)
    parser.add_argument("--allow-apply-clean-test", action="store_true")
    parser.add_argument("--seed", type=int, default=202607210)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def parse_nums(text: str, cast=float) -> list[Any]:
    return [cast(x) for x in text.split(",") if x.strip()]


def read_instance(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def find_anchor(root: Path, exp: str, dataset: str, split: str, name: str) -> Path:
    path = root / exp / dataset / split / "masks" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def separation_band(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((max(1, radius), max(1, radius)), dtype=bool)
    return np.logical_and(ndimage.binary_dilation(mask, structure=structure), ~mask)


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndimage.binary_dilation(mask, structure=structure) ^ ndimage.binary_erosion(mask, structure=structure)


def feature_stack(image: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    anchor_f = anchor.astype(np.float32)
    dist_in = ndimage.distance_transform_edt(anchor)
    dist_out = ndimage.distance_transform_edt(~anchor)
    signed = dist_in - dist_out
    denom = max(1.0, float(np.percentile(np.abs(signed), 95)))
    signed = np.clip(signed / denom, -1.0, 1.0).astype(np.float32)
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0:
        grad = grad / float(grad.max())
    local_anchor_7 = ndimage.uniform_filter(anchor_f, size=7, mode="nearest")
    local_anchor_17 = ndimage.uniform_filter(anchor_f, size=17, mode="nearest")
    local_img_9 = ndimage.uniform_filter(image.astype(np.float32), size=9, mode="nearest")
    local_img_21 = ndimage.uniform_filter(image.astype(np.float32), size=21, mode="nearest")
    inner_edge = boundary_band(anchor, 2).astype(np.float32) * anchor_f
    return np.stack(
        [
            image.astype(np.float32),
            anchor_f,
            signed,
            grad.astype(np.float32),
            local_anchor_7.astype(np.float32),
            local_anchor_17.astype(np.float32),
            local_img_9.astype(np.float32),
            local_img_21.astype(np.float32),
            inner_edge.astype(np.float32),
        ],
        axis=-1,
    )


def load_case(args: argparse.Namespace, split: str, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_path = args.raw_root / args.dataset / f"{split}_labels" / name
    instance = read_instance(label_path)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, split, name))
    anchor = resize_like(read_mask(find_anchor(args.ablations_root, args.anchor_exp, args.dataset, split, name)), gt.shape)
    return image, instance, gt, anchor


def component_merge_count(anchor: np.ndarray, instance: np.ndarray, min_overlap_frac: float) -> int:
    return int(analyze_component_matches(anchor, instance, min_overlap_frac)["merged_pred_components"])


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def sample_training(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for name in tqdm(names(args.raw_root, args.dataset, args.train_split)[: args.max_train_images or None], desc="sample/r210"):
        image, instance, gt, anchor = load_case(args, args.train_split, name)
        feat = feature_stack(image, anchor)
        near_gt_gap = np.logical_and(anchor, ndimage.binary_dilation(separation_band(gt, 5), structure=np.ones((5, 5), dtype=bool)))
        target_cut = np.logical_and.reduce([anchor, ~gt, near_gt_gap])
        merge_count = component_merge_count(anchor, instance, args.min_overlap_frac)
        samples = args.samples_per_merge_image if merge_count > 0 else args.samples_per_image

        idx_pos = np.flatnonzero(target_cut.ravel())
        idx_anchor_edge = np.flatnonzero(np.logical_and(anchor, boundary_band(anchor, 4)).ravel())
        idx_anchor = np.flatnonzero(anchor.ravel())
        picks = []
        n_pos = min(len(idx_pos), samples // 2)
        if n_pos:
            picks.append(rng.choice(idx_pos, size=n_pos, replace=len(idx_pos) < n_pos))
        n_edge = min(len(idx_anchor_edge), samples - sum(len(p) for p in picks))
        if n_edge:
            picks.append(rng.choice(idx_anchor_edge, size=n_edge, replace=len(idx_anchor_edge) < n_edge))
        n_rest = samples - sum(len(p) for p in picks)
        if n_rest:
            picks.append(rng.choice(idx_anchor, size=n_rest, replace=len(idx_anchor) < n_rest))
        idx = np.concatenate(picks)
        xs.append(feat.reshape(-1, feat.shape[-1])[idx])
        ys.append(target_cut.reshape(-1)[idx].astype(np.float32))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def train_model(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> MLP:
    torch.manual_seed(args.seed)
    model = MLP(x.shape[1], args.hidden).to(args.device)
    pos = float(y.mean())
    pos_weight = torch.tensor([args.pos_weight_scale * (1.0 - pos) / max(pos, 1e-4)], device=args.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    rng = np.random.default_rng(args.seed + 11)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y))
        losses = []
        model.train()
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = xt[idx].to(args.device)
            yb = yt[idx].to(args.device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(json.dumps({"epoch": epoch, "loss": float(np.mean(losses)), "pos_rate": pos}), flush=True)
    return model


@torch.no_grad()
def predict_prob(model: MLP, feat: np.ndarray, device: str, batch: int = 262144) -> np.ndarray:
    flat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
    outs = []
    model.eval()
    for start in range(0, len(flat), batch):
        logits = model(torch.from_numpy(flat[start : start + batch]).to(device))
        outs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outs).reshape(feat.shape[:2])


def metric_delta(new: dict[str, float | None], old: dict[str, float | None], key: str) -> float:
    return float(new.get(key) or 0.0) - float(old.get(key) or 0.0)


def accepted_prediction(
    args: argparse.Namespace,
    anchor: np.ndarray,
    gt: np.ndarray,
    instance: np.ndarray,
    prob: np.ndarray,
    threshold: float,
    radius: int,
    gt_cache: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Let the learned split map propose cuts anywhere inside the anchor. The
    # max-cut and per-image acceptance gates below keep this from becoming the
    # free pixel-deletion behavior that failed in R177/R205.
    cut_zone = anchor
    candidate_cut = np.logical_and(cut_zone, prob >= threshold)
    if int(candidate_cut.sum()) < args.min_cut_area:
        candidate_cut[:] = False
    if candidate_cut.any():
        labels, n_labels = ndimage.label(candidate_cut, structure=np.ones((3, 3), dtype=np.uint8))
        kept = np.zeros_like(candidate_cut, dtype=bool)
        for comp_id in range(1, n_labels + 1):
            comp = labels == comp_id
            area = int(comp.sum())
            if area < args.min_cut_area:
                continue
            ys, xs = np.where(comp)
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            slender = max(h, w) / max(1, min(h, w))
            fill = area / max(1, h * w)
            # Prefer thin connected cut strokes over salt-and-pepper deletion.
            if slender >= 2.0 or fill <= 0.45:
                kept |= comp
        candidate_cut = kept
    old = compute_metrics(anchor, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    old_merge = component_merge_count(anchor, instance, args.min_overlap_frac)

    accepted_cut = np.zeros_like(candidate_cut, dtype=bool)
    current = anchor.copy()
    candidate_labels, n_candidate_labels = ndimage.label(candidate_cut, structure=np.ones((3, 3), dtype=np.uint8))
    components = []
    for comp_id in range(1, n_candidate_labels + 1):
        comp = candidate_labels == comp_id
        components.append((float(prob[comp].mean()), int(comp.sum()), comp_id, comp))
    components.sort(reverse=True)
    max_cut = int(max(args.min_cut_area, round(float(anchor.sum()) * args.max_cut_frac)))
    reject_reasons: set[str] = set()
    final = old
    final_merge = old_merge
    for _, area, _, comp in components:
        if int(accepted_cut.sum()) + area > max_cut:
            reject_reasons.add("max_cut_frac")
            continue
        trial = np.logical_and(current, ~comp)
        trial_metrics = compute_metrics(trial, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
        trial_merge = component_merge_count(trial, instance, args.min_overlap_frac)
        checks = {
            "dice_safe": metric_delta(trial_metrics, old, "dice") >= -args.dice_tol,
            "iou_safe": metric_delta(trial_metrics, old, "iou") >= -args.iou_tol,
            "recall_safe": metric_delta(trial_metrics, old, "recall") >= -args.recall_tol,
            "component_count_safe": float(trial_metrics["component_count_mae"] or 0.0) <= float(old["component_count_mae"] or 0.0),
            "has_diagnostic_gain": (
                metric_delta(trial_metrics, old, "boundary_iou") > 0.0
                or metric_delta(trial_metrics, old, "boundary_f1") > 0.0
                or metric_delta(trial_metrics, old, "gap_region_fp_rate") < 0.0
                or trial_merge < old_merge
            ),
        }
        if all(checks.values()):
            accepted_cut |= comp
            current = trial
            final = trial_metrics
            final_merge = trial_merge
        else:
            reject_reasons.update(key for key, ok in checks.items() if not ok)

    accept = bool(accepted_cut.any())
    reject_reason = "accepted" if accept else ";".join(sorted(reject_reasons or {"no_safe_component"}))
    pred = current if accept else anchor
    candidate = np.logical_and(anchor, ~candidate_cut)
    new = compute_metrics(candidate, gt, args.boundary_kernel, args.gap_kernel, args.surface_tol, args.surface_tol_extra, gt_cache)
    new_merge = component_merge_count(candidate, instance, args.min_overlap_frac)
    info = {
        **{f"anchor_{k}": v for k, v in old.items()},
        **{f"candidate_{k}": v for k, v in new.items()},
        **{f"r210_{k}": v for k, v in final.items()},
        "accepted": float(accept),
        "reject_reason": reject_reason,
        "candidate_cut_pixels": float(candidate_cut.sum()),
        "accepted_cut_pixels": float(accepted_cut.sum()),
        "anchor_instance_merge_count": float(old_merge),
        "candidate_instance_merge_count": float(new_merge),
        "r210_instance_merge_count": float(final_merge if accept else old_merge),
    }
    for key in old:
        if isinstance(old[key], (float, int)) or old[key] is None:
            info[f"candidate_delta_{key}"] = None if old[key] is None or new[key] is None else float(new[key]) - float(old[key])
            info[f"delta_{key}"] = None if old[key] is None or final[key] is None else float(final[key]) - float(old[key])
    info["candidate_delta_instance_merge_count"] = float(new_merge - old_merge)
    info["delta_instance_merge_count"] = float(info["r210_instance_merge_count"] - info["anchor_instance_merge_count"])
    return pred, info


def build_val_cache(args: argparse.Namespace, model: MLP, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gt_dir = args.raw_root / args.dataset / f"{split}_labels"
    gt_cache = build_gt_cache(gt_dir, args.boundary_kernel, args.gap_kernel)
    out = []
    split_names = names(args.raw_root, args.dataset, split)[: args.max_val_images or None]
    for name in tqdm(split_names, desc=f"cache/r210/{split}"):
        image, instance, gt, anchor = load_case(args, split, name)
        prob = predict_prob(model, feature_stack(image, anchor), args.device)
        out.append({"name": name, "instance": instance, "gt": gt, "anchor": anchor, "prob": prob, "gt_cache": gt_cache[name]})
    return out, gt_cache


def mean_record(records: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = sorted({k for r in records for k in r if k != "image"})
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [r.get(key) for r in records]
        finite = [float(v) for v in vals if isinstance(v, (float, int))]
        out[key] = float(np.mean(finite)) if finite else None
    return out


def evaluate_grid(args: argparse.Namespace, cache: list[dict[str, Any]], write_dir: Path | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    best: dict[str, Any] | None = None
    grid = []
    best_records: list[dict[str, Any]] = []
    for radius in parse_nums(args.split_radii, int):
        for threshold in parse_nums(args.prob_thresholds, float):
            records = []
            preds: list[tuple[str, np.ndarray]] = []
            for item in tqdm(cache, desc=f"r210/r{radius}/t{threshold:.2f}", leave=False):
                pred, rec = accepted_prediction(
                    args,
                    item["anchor"],
                    item["gt"],
                    item["instance"],
                    item["prob"],
                    threshold,
                    radius,
                    item["gt_cache"],
                )
                rec["image"] = item["name"]
                records.append(rec)
                preds.append((item["name"], pred))
            mean = mean_record(records)
            accepted_frac = float(mean.get("accepted") or 0.0)
            item_summary = {"radius": radius, "threshold": threshold, "mean": mean}
            grid.append(item_summary)
            score = (
                accepted_frac > 0.0,
                -(mean.get("delta_component_count_mae") or 0.0),
                -(mean.get("delta_gap_region_fp_rate") or 0.0),
                -(mean.get("delta_instance_merge_count") or 0.0),
                mean.get("delta_boundary_iou") or 0.0,
                mean.get("delta_dice") or -999.0,
            )
            best_score = best.get("_score") if best is not None else None
            if best is None or score > best_score:
                best = {**item_summary, "_score": score}
                best_records = records
                if write_dir is not None:
                    write_dir.mkdir(parents=True, exist_ok=True)
                    for name, pred in preds:
                        write_mask(write_dir / name, pred)
    assert best is not None
    return best, grid, best_records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gate_pass(best: dict[str, Any]) -> bool:
    mean = best["mean"]
    return bool(
        (mean.get("accepted") or 0.0) > 0.0
        and (mean.get("delta_dice") or 0.0) >= -0.003
        and (mean.get("delta_iou") or 0.0) >= -0.003
        and (mean.get("delta_recall") or 0.0) >= -0.005
        and (mean.get("delta_component_count_mae") or 0.0) <= 0.0
        and (
            (mean.get("delta_gap_region_fp_rate") or 0.0) < 0.0
            or (mean.get("delta_instance_merge_count") or 0.0) < 0.0
            or (mean.get("delta_boundary_iou") or 0.0) > 0.0
            or (mean.get("delta_boundary_f1") or 0.0) > 0.0
        )
    )


def main() -> None:
    args = parse_args()
    x, y = sample_training(args)
    model = train_model(args, x, y)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "in_dim": x.shape[1], "args": vars(args)}, args.checkpoint)

    val_cache, _ = build_val_cache(args, model, args.val_split)
    val_pred_dir = args.ablations_root / args.output_exp / args.dataset / args.val_split / "masks"
    best, grid, records = evaluate_grid(args, val_cache, val_pred_dir)
    best.pop("_score", None)
    passed = gate_pass(best)
    summary = {
        "run_id": "R210-F0",
        "dataset": args.dataset,
        "split": args.val_split,
        "clean_test_v2_used": False,
        "num_evaluated": len(records),
        "val_gate_pass": passed,
        "best": best,
        "grid": grid,
        "checkpoint": str(args.checkpoint),
        "output_exp": args.output_exp,
    }
    args.val_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.val_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.val_csv, records)

    if passed and args.allow_apply_clean_test:
        raise NotImplementedError("Clean-test-v2 application should be launched only after reviewing the validation gate.")

    print(json.dumps({"val_gate_pass": passed, "best": best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
