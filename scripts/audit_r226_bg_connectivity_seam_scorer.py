#!/usr/bin/env python3
"""R226 background-connectivity seam proposal diagnostic.

Train/val-only F0 audit for the "reverse topology" route: instead of
encouraging foreground/bone connectivity, generate candidate cuts that connect
nearby background regions through an R110 foreground component. If the connected
background channel lies in the true epiphyseal gap, it should reduce seam
adhesion without eroding bone interiors.

Candidate generation is GT-free: it uses only the image and the anchor mask.
GT is used only to label/evaluate candidates on original train/val. Do not use
this script on clean-test-v2 for threshold search or model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
except ModuleNotFoundError:
    HistGradientBoostingClassifier = None
    SimpleImputer = None
    KFold = None
    Pipeline = Any
from tqdm import tqdm

from audit_r216_soft_seam_action_candidates import build_one_gt_cache, is_hard_risk, is_quick_useful, quick_metrics
from run_r209_component_preserving_feasibility_audit import read_instance
from train_anchor_pixel_residual import image_path, names, read_gray, read_mask, resize_like


FEATURE_KEYS = [
    "component_area",
    "bg_a_area",
    "bg_b_area",
    "bg_pair_distance",
    "corridor_radius",
    "action_frac",
    "cut_area",
    "cut_anchor_area_frac",
    "cut_bbox_h",
    "cut_bbox_w",
    "cut_slenderness",
    "cut_fill",
    "cut_mean_dist_in",
    "cut_max_dist_in",
    "cut_mean_image",
    "cut_std_image",
    "cut_mean_grad",
    "cut_p90_grad",
    "ring_bg_frac",
    "ring_fg_frac",
    "bridge_score_mean",
    "bridge_score_p90",
    "bg_components_before",
    "bg_components_after",
    "bg_component_count_delta",
    "cut_adjacent_bg_components_before",
    "cut_component_bg_frac_after",
    "cut_component_border_contacts_after",
    "bg_channel_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit R226 background-connectivity seam candidates.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--ablations-root", type=Path, default=Path("outputs/ablations_variants"))
    parser.add_argument("--anchor-exp", default="r110_r100_r108_patch_basic_trainval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--source-candidate-csv", type=Path, default=Path(""))
    parser.add_argument("--min-component-area", type=int, default=96)
    parser.add_argument("--max-component-area", type=int, default=60000)
    parser.add_argument("--min-cut-area", type=int, default=4)
    parser.add_argument("--max-cut-frac", type=float, default=0.006)
    parser.add_argument("--component-pad", type=int, default=18)
    parser.add_argument("--max-bg-components", type=int, default=8)
    parser.add_argument("--max-pairs-per-component", type=int, default=6)
    parser.add_argument("--max-components-per-image", type=int, default=4)
    parser.add_argument("--corridor-radii", default="1,2,3")
    parser.add_argument("--action-fracs", default="0.35,0.50,0.75")
    parser.add_argument("--boundary-kernel", type=int, default=3)
    parser.add_argument("--gap-kernel", type=int, default=9)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--useful-thresholds", default="0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--risk-thresholds", default="0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--max-proposals-per-image", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/analysis/r226_bg_connectivity_seam_val_candidates.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r226_bg_connectivity_seam_val_summary.json"))
    parser.add_argument("--cv-csv", type=Path, default=Path("outputs/analysis/r226_bg_connectivity_seam_val_groupedcv.csv"))
    parser.add_argument("--seed", type=int, default=202607226)
    parser.add_argument("--flush-every", type=int, default=0)
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else default


def anchor_path(args: argparse.Namespace, name: str) -> Path:
    return args.ablations_root / args.anchor_exp / args.dataset / args.split / "masks" / name


def component_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    return ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))


def bbox_slice(mask: np.ndarray, pad: int) -> tuple[slice, slice] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (
        slice(max(0, int(ys.min()) - pad), min(mask.shape[0], int(ys.max()) + pad + 1)),
        slice(max(0, int(xs.min()) - pad), min(mask.shape[1], int(xs.max()) + pad + 1)),
    )


def border_contacts(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    return int(mask[0, :].any()) + int(mask[-1, :].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def load_case(args: argparse.Namespace, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    instance = read_instance(args.raw_root / args.dataset / f"{args.split}_labels" / name)
    gt = instance > 0
    image = read_gray(image_path(args.raw_root, args.dataset, args.split, name))
    anchor = resize_like(read_mask(anchor_path(args, name)), gt.shape)
    return image, instance, gt, anchor


def source_names(args: argparse.Namespace) -> list[str]:
    if str(args.source_candidate_csv) and args.source_candidate_csv != Path(".") and args.source_candidate_csv.exists():
        ordered: list[str] = []
        seen: set[str] = set()
        with args.source_candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                name = str(row.get("image") or "")
                if name and name not in seen:
                    seen.add(name)
                    ordered.append(name)
        return ordered[: args.limit or None]
    return names(args.raw_root, args.dataset, args.split)[: args.limit or None]


def image_grad(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image.astype(np.float32))
    grad = np.sqrt(gx * gx + gy * gy)
    if float(grad.max()) > 0.0:
        grad = grad / float(grad.max())
    return grad


def bg_components_near_component(anchor: np.ndarray, component: np.ndarray, pad: int, max_components: int) -> list[dict[str, Any]]:
    local = bbox_slice(component, pad)
    if local is None:
        return []
    local_component = component[local]
    local_anchor = anchor[local]
    local_bg = ~local_anchor
    labels, n_labels = component_labels(local_bg)
    ring = ndimage.binary_dilation(local_component, structure=np.ones((3, 3), dtype=bool)) & ~local_component
    out: list[dict[str, Any]] = []
    for label_id in range(1, n_labels + 1):
        bg = labels == label_id
        contact = int(np.logical_and(bg, ring).sum())
        if contact <= 0:
            continue
        area = int(bg.sum())
        ys, xs = np.where(bg)
        out.append(
            {
                "label_id": int(label_id),
                "area": area,
                "contact": contact,
                "centroid_y": float(np.mean(ys)) if ys.size else 0.0,
                "centroid_x": float(np.mean(xs)) if xs.size else 0.0,
                "local_slice": local,
                "local_component": local_component,
                "local_anchor": local_anchor,
                "labels": labels,
            }
        )
    out.sort(key=lambda row: (row["contact"], row["area"]), reverse=True)
    return out[:max_components]


def bg_contact_seeds(anchor: np.ndarray, component: np.ndarray, max_seeds: int, min_distance: int = 18) -> list[dict[str, Any]]:
    local = bbox_slice(component, pad=8)
    if local is None:
        return []
    local_component = component[local]
    local_anchor = anchor[local]
    ring = ndimage.binary_dilation(local_component, structure=np.ones((5, 5), dtype=bool)) & ~local_anchor
    if not ring.any():
        return []
    comp_dist = ndimage.distance_transform_edt(~local_component)
    labels, n_labels = component_labels(ring)
    seeds: list[dict[str, Any]] = []
    for label_id in range(1, n_labels + 1):
        region = labels == label_id
        ys, xs = np.where(region)
        if ys.size == 0:
            continue
        scores = comp_dist[ys, xs]
        order = np.argsort(scores)
        y = int(ys[order[0]]) + int(local[0].start)
        x = int(xs[order[0]]) + int(local[1].start)
        if all((y - int(seed["y"])) ** 2 + (x - int(seed["x"])) ** 2 >= min_distance**2 for seed in seeds):
            seeds.append({"y": float(y), "x": float(x), "area": float(region.sum()), "contact": float(region.sum())})
    if len(seeds) < 2:
        ys, xs = np.where(ring)
        order = np.argsort(comp_dist[ys, xs])
        for idx in order:
            y = int(ys[idx]) + int(local[0].start)
            x = int(xs[idx]) + int(local[1].start)
            if all((y - int(seed["y"])) ** 2 + (x - int(seed["x"])) ** 2 >= min_distance**2 for seed in seeds):
                seeds.append({"y": float(y), "x": float(x), "area": 1.0, "contact": 1.0})
            if len(seeds) >= max_seeds:
                break
    return seeds[:max_seeds]


def seed_pair_cut(
    component: np.ndarray,
    seed_a: dict[str, Any],
    seed_b: dict[str, Any],
    image: np.ndarray,
    radius: int,
    min_cut_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(component.shape)
    ay, ax = float(seed_a["y"]), float(seed_a["x"])
    by, bx = float(seed_b["y"]), float(seed_b["x"])
    vx = bx - ax
    vy = by - ay
    denom = max(1e-6, vx * vx + vy * vy)
    t = ((xx.astype(np.float32) - ax) * vx + (yy.astype(np.float32) - ay) * vy) / denom
    t_clip = np.clip(t, 0.0, 1.0)
    proj_x = ax + t_clip * vx
    proj_y = ay + t_clip * vy
    line_dist = np.sqrt((xx.astype(np.float32) - proj_x) ** 2 + (yy.astype(np.float32) - proj_y) ** 2)
    dist_in = ndimage.distance_transform_edt(component)
    grad = image_grad(image)
    raw = component & (line_dist <= float(radius))
    if raw.any():
        raw &= dist_in <= max(1.0, float(np.percentile(dist_in[component], 50)))
    if int(raw.sum()) >= min_cut_area:
        gvals = grad[raw]
        if gvals.size:
            raw &= grad >= float(np.percentile(gvals, 25))
    raw = ndimage.binary_opening(raw, structure=np.ones((2, 2), dtype=bool))
    bridge = line_dist + 0.25 * dist_in
    return raw.astype(bool), bridge


def make_bg_bridge_cut(
    component: np.ndarray,
    bg_a: np.ndarray,
    bg_b: np.ndarray,
    image: np.ndarray,
    radius: int,
    min_cut_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    dist_a = ndimage.distance_transform_edt(~bg_a)
    dist_b = ndimage.distance_transform_edt(~bg_b)
    dist_in = ndimage.distance_transform_edt(component)
    grad = image_grad(image)
    bridge = dist_a + dist_b
    vals = bridge[component]
    if vals.size == 0:
        return np.zeros_like(component, dtype=bool), bridge
    min_bridge = float(np.min(vals))
    # A thin corridor close to the shortest background-to-background route.
    raw = component & (bridge <= min_bridge + float(radius))
    shallow = component & (dist_in <= max(1.0, float(np.percentile(dist_in[component], 45))))
    raw &= shallow
    if int(raw.sum()) < min_cut_area:
        raw = component & (bridge <= min_bridge + float(radius) + 1.5)
    # Remove bulky interior islands, keep narrow high-gradient seam-like pixels.
    if int(raw.sum()) >= min_cut_area:
        gvals = grad[raw]
        if gvals.size:
            raw &= grad >= float(np.percentile(gvals, 35))
    raw = ndimage.binary_opening(raw, structure=np.ones((2, 2), dtype=bool))
    return raw.astype(bool), bridge


def trim_cut_by_score(image: np.ndarray, anchor: np.ndarray, cut: np.ndarray, bridge: np.ndarray, action_frac: float, min_cut_area: int) -> np.ndarray:
    cut = cut.astype(bool)
    n = int(cut.sum())
    if n < min_cut_area:
        return np.zeros_like(cut, dtype=bool)
    dist_in = ndimage.distance_transform_edt(anchor)
    grad = image_grad(image)
    bridge_vals = bridge[cut]
    bridge_norm = np.zeros_like(bridge, dtype=np.float32)
    if bridge_vals.size and float(np.max(bridge_vals)) > float(np.min(bridge_vals)):
        bridge_norm = (bridge - float(np.min(bridge_vals))) / (float(np.max(bridge_vals)) - float(np.min(bridge_vals)))
    score = -0.50 * dist_in + 0.35 * grad - 0.25 * bridge_norm
    ys, xs = np.where(cut)
    keep_n = min(n, max(min_cut_area, int(round(n * action_frac))))
    order = np.argsort(score[ys, xs])[::-1][:keep_n]
    out = np.zeros_like(cut, dtype=bool)
    out[ys[order], xs[order]] = True
    out = ndimage.binary_opening(out, structure=np.ones((2, 2), dtype=bool))
    if int(out.sum()) < min_cut_area:
        out = np.zeros_like(cut, dtype=bool)
        out[ys[order], xs[order]] = True
    return out


def candidate_features(
    image: np.ndarray,
    anchor: np.ndarray,
    component: np.ndarray,
    cut: np.ndarray,
    bridge: np.ndarray,
    bg_a: np.ndarray,
    bg_b: np.ndarray,
    radius: int,
    action_frac: float,
) -> dict[str, float]:
    grad = image_grad(image)
    dist_in = ndimage.distance_transform_edt(anchor)
    local = bbox_slice(cut, pad=18)
    if local is None:
        local_anchor = np.zeros((0, 0), dtype=bool)
        local_cut = np.zeros((0, 0), dtype=bool)
    else:
        local_anchor = anchor[local].astype(bool)
        local_cut = cut[local].astype(bool)
    bg_before = ~local_anchor
    bg_after = ~(local_anchor & ~local_cut)
    bg_labels_before, bg_n_before = component_labels(bg_before)
    bg_labels_after, bg_n_after = component_labels(bg_after)
    cut_ring = ndimage.binary_dilation(local_cut, structure=np.ones((3, 3), dtype=bool)) & ~local_cut
    adjacent_bg = np.unique(bg_labels_before[cut_ring & bg_before])
    adjacent_bg = adjacent_bg[adjacent_bg > 0]
    cut_after_labels = np.unique(bg_labels_after[local_cut])
    cut_after_labels = cut_after_labels[cut_after_labels > 0]
    cut_bg_component = np.isin(bg_labels_after, cut_after_labels) if cut_after_labels.size else np.zeros_like(bg_after, dtype=bool)

    ys, xs = np.where(cut)
    area = int(cut.sum())
    h = int(ys.max() - ys.min() + 1) if ys.size else 0
    w = int(xs.max() - xs.min() + 1) if xs.size else 0
    ring = ndimage.binary_dilation(cut, structure=np.ones((5, 5), dtype=bool)) & ~cut
    bridge_vals = bridge[cut] if area else np.asarray([], dtype=np.float32)
    cut_img = image[cut] if area else np.asarray([], dtype=np.float32)
    cut_grad = grad[cut] if area else np.asarray([], dtype=np.float32)
    cut_dist = dist_in[cut] if area else np.asarray([], dtype=np.float32)
    bg_a_y, bg_a_x = np.where(bg_a)
    bg_b_y, bg_b_x = np.where(bg_b)
    bg_pair_distance = 0.0
    if bg_a_y.size and bg_b_y.size:
        bg_pair_distance = float(np.hypot(np.mean(bg_a_y) - np.mean(bg_b_y), np.mean(bg_a_x) - np.mean(bg_b_x)))
    return {
        "component_area": float(component.sum()),
        "bg_a_area": float(bg_a.sum()),
        "bg_b_area": float(bg_b.sum()),
        "bg_pair_distance": bg_pair_distance,
        "corridor_radius": float(radius),
        "action_frac": float(action_frac),
        "cut_area": float(area),
        "cut_anchor_area_frac": safe_divide(float(area), float(anchor.sum())),
        "cut_bbox_h": float(h),
        "cut_bbox_w": float(w),
        "cut_slenderness": float(max(h, w) / max(1, min(h, w))) if area else 0.0,
        "cut_fill": safe_divide(float(area), float(max(1, h * w))),
        "cut_mean_dist_in": float(np.mean(cut_dist)) if cut_dist.size else 0.0,
        "cut_max_dist_in": float(np.max(cut_dist)) if cut_dist.size else 0.0,
        "cut_mean_image": float(np.mean(cut_img)) if cut_img.size else 0.0,
        "cut_std_image": float(np.std(cut_img)) if cut_img.size else 0.0,
        "cut_mean_grad": float(np.mean(cut_grad)) if cut_grad.size else 0.0,
        "cut_p90_grad": float(np.percentile(cut_grad, 90)) if cut_grad.size else 0.0,
        "ring_bg_frac": safe_divide(float((ring & ~anchor).sum()), float(ring.sum())),
        "ring_fg_frac": safe_divide(float((ring & anchor).sum()), float(ring.sum())),
        "bridge_score_mean": float(np.mean(bridge_vals)) if bridge_vals.size else 0.0,
        "bridge_score_p90": float(np.percentile(bridge_vals, 90)) if bridge_vals.size else 0.0,
        "bg_components_before": float(bg_n_before),
        "bg_components_after": float(bg_n_after),
        "bg_component_count_delta": float(bg_n_after - bg_n_before),
        "cut_adjacent_bg_components_before": float(adjacent_bg.size),
        "cut_component_bg_frac_after": safe_divide(float(cut_bg_component.sum()), float(local_cut.size)),
        "cut_component_border_contacts_after": float(border_contacts(cut_bg_component)),
        "bg_channel_proxy": float(border_contacts(cut_bg_component) >= 2 and safe_divide(float(cut_bg_component.sum()), float(local_cut.size)) >= 0.15),
    }


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, "", "None", "nan"):
        return default
    return float(value)


def metric_delta(new: dict[str, float], old: dict[str, float], key: str) -> float:
    return float(new[key]) - float(old[key])


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    case_names = source_names(args)
    corridor_radii = parse_int_list(args.corridor_radii)
    action_fracs = parse_float_list(args.action_fracs)
    for index, name in enumerate(tqdm(case_names, desc=f"r226/bg-connectivity/{args.split}"), start=1):
        if not anchor_path(args, name).exists():
            continue
        image, instance, gt, anchor = load_case(args, name)
        gt_cache = build_one_gt_cache(name, gt, args.boundary_kernel, args.gap_kernel)
        anchor_metrics = quick_metrics(anchor, gt, gt_cache, args.boundary_kernel)
        comp_labels, n_components = component_labels(anchor)
        max_pixels = max(args.min_cut_area, int(round(float(anchor.sum()) * args.max_cut_frac)))
        component_order: list[tuple[int, int]] = []
        for comp_id in range(1, n_components + 1):
            component_order.append((int((comp_labels == comp_id).sum()), comp_id))
        component_order.sort(reverse=True)
        for _area, comp_id in component_order[: args.max_components_per_image]:
            component = comp_labels == comp_id
            comp_area = int(component.sum())
            if comp_area < args.min_component_area or comp_area > args.max_component_area:
                continue
            bg_infos = bg_components_near_component(anchor, component, args.component_pad, args.max_bg_components)
            pair_specs: list[tuple[float, str, dict[str, Any], dict[str, Any]]] = []
            for i, a in enumerate(bg_infos):
                for b in bg_infos[i + 1 :]:
                    dy = float(a["centroid_y"] - b["centroid_y"])
                    dx = float(a["centroid_x"] - b["centroid_x"])
                    dist = float(np.hypot(dy, dx))
                    score = float(a["contact"] + b["contact"]) - 0.02 * dist
                    pair_specs.append((score, "bg_component", a, b))
            seeds = bg_contact_seeds(anchor, component, args.max_bg_components)
            for i, a in enumerate(seeds):
                for b in seeds[i + 1 :]:
                    dy = float(a["y"] - b["y"])
                    dx = float(a["x"] - b["x"])
                    dist = float(np.hypot(dy, dx))
                    if dist < 12.0:
                        continue
                    score = -abs(dist - 42.0)
                    pair_specs.append((score, "bg_seed", a, b))
            pair_specs.sort(key=lambda item: item[0], reverse=True)
            if not pair_specs:
                continue
            for _, pair_kind, info_a, info_b in pair_specs[: args.max_pairs_per_component]:
                bg_a = np.zeros_like(anchor, dtype=bool)
                bg_b = np.zeros_like(anchor, dtype=bool)
                local_component = component
                if pair_kind == "bg_component":
                    local = info_a["local_slice"]
                    local_component_arr = info_a["local_component"]
                    labels = info_a["labels"]
                    bg_a_local = labels == int(info_a["label_id"])
                    bg_b_local = labels == int(info_b["label_id"])
                    local_comp_full = np.zeros_like(anchor, dtype=bool)
                    bg_a[local] = bg_a_local
                    bg_b[local] = bg_b_local
                    local_comp_full[local] = local_component_arr
                    local_component = local_comp_full & component
                else:
                    bg_a[int(info_a["y"]), int(info_a["x"])] = True
                    bg_b[int(info_b["y"]), int(info_b["x"])] = True
                for radius in corridor_radii:
                    if pair_kind == "bg_component":
                        raw_cut, bridge = make_bg_bridge_cut(local_component, bg_a, bg_b, image, radius, args.min_cut_area)
                    else:
                        raw_cut, bridge = seed_pair_cut(local_component, info_a, info_b, image, radius, args.min_cut_area)
                    raw_cut &= anchor
                    if int(raw_cut.sum()) < args.min_cut_area:
                        continue
                    for action_frac in action_fracs:
                        cut = trim_cut_by_score(image, anchor, raw_cut, bridge, action_frac, args.min_cut_area)
                        cut &= anchor
                        if int(cut.sum()) < args.min_cut_area or int(cut.sum()) > max_pixels:
                            continue
                        trial = anchor & ~cut
                        trial_metrics = quick_metrics(trial, gt, gt_cache, args.boundary_kernel)
                        row: dict[str, Any] = {
                            "run_id": "R226-bg-connectivity-seam-scorer-diagnostic",
                            "split": args.split,
                            "image": name,
                            "pred_component_id": float(comp_id),
                            "pair_kind": pair_kind,
                        }
                        row.update(candidate_features(image, anchor, component, cut, bridge, bg_a, bg_b, radius, action_frac))
                        row["cut_gt_fg_frac"] = safe_divide(float(np.logical_and(cut, instance > 0).sum()), float(cut.sum()))
                        row["cut_gt_gap_frac"] = safe_divide(float(np.logical_and(cut, instance == 0).sum()), float(cut.sum()))
                        for metric, value in anchor_metrics.items():
                            row[f"anchor_{metric}"] = value
                        for metric, value in trial_metrics.items():
                            row[f"candidate_{metric}"] = value
                            row[f"delta_{metric}"] = metric_delta(trial_metrics, anchor_metrics, metric)
                        row["r226_quick_useful"] = float(is_quick_useful(row))
                        row["r226_hard_risk"] = float(is_hard_risk(row))
                        row["r226_safe_gap_positive"] = float(row["cut_gt_fg_frac"] <= 0.25 and row["cut_gt_gap_frac"] >= 0.75)
                        row["r226_overerosion_proxy"] = float(row["cut_gt_fg_frac"] > 0.5)
                        rows.append(row)
        if args.flush_every > 0 and index % args.flush_every == 0:
            write_csv(args.output_csv, rows)
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(summarize(rows, [], args), indent=2), encoding="utf-8")
    return rows


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[as_float(row, key) for key in FEATURE_KEYS] for row in rows], dtype=np.float32)


def make_model(seed: int) -> Pipeline:
    if HistGradientBoostingClassifier is None or SimpleImputer is None:
        raise RuntimeError("scikit-learn is required for R226 model diagnostics.")
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("clf", HistGradientBoostingClassifier(max_iter=220, learning_rate=0.035, random_state=seed, l2_regularization=0.03)),
        ]
    )


def fit_binary_model(rows: list[dict[str, Any]], label_key: str, seed: int) -> Pipeline | None:
    y = np.asarray([int(as_float(row, label_key) > 0.5) for row in rows], dtype=np.int32)
    if len(set(y.tolist())) < 2:
        return None
    model = make_model(seed)
    model.fit(matrix(rows), y)
    return model


def add_cv_probs(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if KFold is None:
        raise RuntimeError("scikit-learn is required for R226 cross-validation diagnostics.")
    images = np.asarray(sorted({str(row["image"]) for row in rows}))
    if len(images) < 2:
        return [], {"skipped": True, "reason": "not_enough_images"}
    folds = min(args.cv_folds, len(images))
    probed: list[dict[str, Any]] = []
    skipped_folds = 0
    kf = KFold(n_splits=folds, shuffle=True, random_state=args.seed)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images), start=1):
        train_images = set(images[train_idx].tolist())
        val_images = set(images[val_idx].tolist())
        train_rows = [row for row in rows if row["image"] in train_images]
        val_rows = [row for row in rows if row["image"] in val_images]
        useful_model = fit_binary_model(train_rows, "r226_quick_useful", args.seed + fold_idx)
        risk_model = fit_binary_model(train_rows, "r226_hard_risk", args.seed + 100 + fold_idx)
        if useful_model is None or risk_model is None:
            skipped_folds += 1
            continue
        x_val = matrix(val_rows)
        useful_prob = useful_model.predict_proba(x_val)[:, 1]
        risk_prob = risk_model.predict_proba(x_val)[:, 1]
        for row, up, rp in zip(val_rows, useful_prob, risk_prob):
            probed.append({**row, "fold": fold_idx, "r226_useful_prob": float(up), "r226_risk_prob": float(rp)})
    return probed, {"skipped": False, "folds": folds, "skipped_folds": skipped_folds, "num_probed": len(probed)}


def selected_rows(rows: list[dict[str, Any]], useful_thr: float, risk_thr: float, max_per_image: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        candidates = [
            row
            for row in image_rows
            if as_float(row, "r226_useful_prob") >= useful_thr and as_float(row, "r226_risk_prob") <= risk_thr
        ]
        candidates.sort(key=lambda row: (-as_float(row, "r226_useful_prob"), as_float(row, "r226_risk_prob"), -as_float(row, "bg_channel_proxy")))
        selected.extend(candidates[:max_per_image])
    return selected


def mean_value(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = []
    for row in rows:
        value = row.get(key)
        if value in (None, "", "None", "nan"):
            continue
        value = float(value)
        if np.isfinite(value):
            vals.append(value)
    return float(np.mean(vals)) if vals else None


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "num_quick_useful": int(sum(as_float(row, "r226_quick_useful") > 0.5 for row in rows)),
        "num_hard_risk": int(sum(as_float(row, "r226_hard_risk") > 0.5 for row in rows)),
        "num_safe_gap_positive": int(sum(as_float(row, "r226_safe_gap_positive") > 0.5 for row in rows)),
        "num_overerosion_proxy": int(sum(as_float(row, "r226_overerosion_proxy") > 0.5 for row in rows)),
        "mean_delta_dice": mean_value(rows, "delta_dice"),
        "mean_delta_iou": mean_value(rows, "delta_iou"),
        "mean_delta_recall": mean_value(rows, "delta_recall"),
        "mean_delta_boundary_iou": mean_value(rows, "delta_boundary_iou"),
        "mean_delta_boundary_f1": mean_value(rows, "delta_boundary_f1"),
        "mean_delta_gap_region_fp_rate": mean_value(rows, "delta_gap_region_fp_rate"),
        "mean_delta_component_count_mae": mean_value(rows, "delta_component_count_mae"),
        "mean_cut_gt_fg_frac": mean_value(rows, "cut_gt_fg_frac"),
        "mean_cut_gt_gap_frac": mean_value(rows, "cut_gt_gap_frac"),
    }


def oracle_best_by_image(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["image"]), []).append(row)
    chosen: list[dict[str, Any]] = []
    for image_rows in grouped.values():
        useful = [row for row in image_rows if as_float(row, "r226_quick_useful") > 0.5 and as_float(row, "r226_hard_risk") <= 0.5]
        if not useful:
            continue
        useful.sort(
            key=lambda row: (
                -as_float(row, "delta_boundary_iou"),
                as_float(row, "delta_gap_region_fp_rate"),
                as_float(row, "delta_component_count_mae"),
                -as_float(row, "delta_dice"),
            )
        )
        chosen.append(useful[0])
    return chosen


def evaluate_cv_grid(probed: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for useful_thr in parse_float_list(args.useful_thresholds):
        for risk_thr in parse_float_list(args.risk_thresholds):
            selected = selected_rows(probed, useful_thr, risk_thr, args.max_proposals_per_image)
            summary = summarize_group(selected)
            grid.append(
                {
                    "useful_threshold": useful_thr,
                    "risk_threshold": risk_thr,
                    "accepted": summary["num_rows"],
                    "accepted_images": summary["num_images"],
                    **summary,
                }
            )
    return grid


def summarize(rows: list[dict[str, Any]], cv_grid: list[dict[str, Any]], args: argparse.Namespace, cv_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    safe = [row for row in rows if as_float(row, "r226_safe_gap_positive") > 0.5]
    oracle = oracle_best_by_image(rows)
    best_cv = None
    viable = [row for row in cv_grid if int(row.get("num_hard_risk") or 0) == 0 and int(row.get("accepted") or 0) > 0]
    if viable:
        best_cv = sorted(
            viable,
            key=lambda row: (
                row["mean_delta_boundary_iou"] or -999.0,
                -(row["mean_delta_gap_region_fp_rate"] or 999.0),
                row["mean_delta_dice"] or -999.0,
            ),
            reverse=True,
        )[0]
    elif cv_grid:
        best_cv = sorted(
            cv_grid,
            key=lambda row: (
                -int(row["num_hard_risk"]),
                int(row["num_quick_useful"]),
                row["mean_delta_boundary_iou"] or -999.0,
            ),
            reverse=True,
        )[0]
    return {
        "run_id": "R226-bg-connectivity-seam-scorer-diagnostic",
        "dataset": args.dataset,
        "split": args.split,
        "clean_test_v2_used": False,
        "writes_masks": False,
        "evidence_level": "train_val_gt_free_candidate_grouped_cv_diagnostic",
        "anchor_exp": args.anchor_exp,
        "idea": "connect local background regions through thin foreground corridors to separate adhered epiphysis seams",
        "feature_keys": FEATURE_KEYS,
        "num_candidate_rows": len(rows),
        "num_images": len({row["image"] for row in rows}),
        "overall": summarize_group(rows),
        "safe_gap_positive": summarize_group(safe),
        "oracle_best_per_image": summarize_group(oracle),
        "grouped_cv": cv_meta or {},
        "best_cv": best_cv,
        "promotion_gate": {
            "candidate_level": "nonzero selected rows, zero hard-risk preferred, positive Boundary IoU/F1 delta, negative gap FP delta, non-worse Dice/IoU/Recall",
            "next_mask_level": "write isolated train/val masks only after candidate-level CV selector is useful; clean-test-v2 remains forbidden for tuning",
        },
        "warning": "GT is used only for train/val diagnostic labels/metrics; this is not clean-test-v2 or paper-facing evidence.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    write_csv(args.output_csv, rows)
    probed, cv_meta = add_cv_probs(rows, args)
    cv_grid = evaluate_cv_grid(probed, args) if probed else []
    write_csv(args.cv_csv, cv_grid)
    report = summarize(rows, cv_grid, args, cv_meta)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(args.output_csv), "output_json": str(args.output_json), "cv_csv": str(args.cv_csv), "best_cv": report["best_cv"]}, indent=2))


if __name__ == "__main__":
    main()
