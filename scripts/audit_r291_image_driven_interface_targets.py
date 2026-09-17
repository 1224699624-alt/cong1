"""Audit deployable training targets for the R291 interface-prior main line.

The raw palette labels are used only to derive *training* supervision.  This
script never creates an inference ROI and never reads clean-test-v2.  Targets
are generated per image (not from a fixed anatomical pair list):

* reliable_gap: background watershed ridges between two nearby instances;
* interface: a local band around a gap ridge or direct instance contact;
* support: high-confidence instance interior inside the interface band;
* ignore: ambiguous interface pixels that are neither gap nor support.

Only compact statistics and a small deterministic visualization set are
written.  The original dataset is never changed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SHIFTS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/analysis/r291_interface_target_audit"))
    p.add_argument("--visual-root", type=Path, default=Path("outputs/visualizations/r291_interface_target_audit"))
    p.add_argument("--audit-size", type=int, default=512, help="long-side size used only for target audit")
    p.add_argument("--max-gap-px", type=float, default=8.0)
    p.add_argument("--gap-halfwidth", type=int, default=1)
    p.add_argument("--interface-radius", type=int, default=7)
    p.add_argument("--support-depth", type=float, default=3.0)
    p.add_argument("--visuals-per-split", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def read_ids(path: Path) -> np.ndarray:
    im = Image.open(path)
    a = np.asarray(im)
    if a.ndim != 2:
        raise ValueError(f"Expected indexed 2-D label, got {im.mode} {a.shape}: {path}")
    return a.astype(np.int32)


def find_image(root: Path, split: str, stem: str) -> Path:
    folder = root / split
    candidates = []
    for ext_index, ext in enumerate(IMAGE_EXTENSIONS):
        for p in folder.glob(f"{stem}{ext}"):
            if p.is_file() and p.stat().st_size > 0:
                candidates.append((ext_index, -p.stat().st_size, p.name.lower(), p))
    if not candidates:
        raise FileNotFoundError(f"No non-empty image for {split}/{stem}")
    return sorted(candidates)[0][-1]


def resize_case(image: np.ndarray, ids: np.ndarray, long_side: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = ids.shape
    scale = min(1.0, float(long_side) / max(h, w))
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    image_r = np.asarray(Image.fromarray(image).resize((nw, nh), Image.Resampling.BILINEAR))
    ids_r = np.asarray(Image.fromarray(ids.astype(np.uint8)).resize((nw, nh), Image.Resampling.NEAREST)).astype(np.int32)
    return image_r, ids_r


def shifted(a: np.ndarray, dy: int, dx: int, fill: int | float = 0) -> np.ndarray:
    out = np.full_like(a, fill)
    y_dst = slice(max(0, dy), min(a.shape[0], a.shape[0] + dy))
    x_dst = slice(max(0, dx), min(a.shape[1], a.shape[1] + dx))
    y_src = slice(max(0, -dy), min(a.shape[0], a.shape[0] - dy))
    x_src = slice(max(0, -dx), min(a.shape[1], a.shape[1] - dx))
    out[y_dst, x_dst] = a[y_src, x_src]
    return out


def label_boundary(ids: np.ndarray) -> np.ndarray:
    b = np.zeros(ids.shape, bool)
    for dy, dx in SHIFTS:
        other = shifted(ids, dy, dx, fill=-1)
        b |= (ids >= 0) & (other >= 0) & (ids != other)
    return b


def build_targets(ids: np.ndarray, max_gap_px: float, gap_halfwidth: int,
                  interface_radius: int, support_depth: float) -> dict[str, object]:
    fg = ids > 0
    background = ~fg
    distance, nearest_index = ndimage.distance_transform_edt(background, return_indices=True)
    nearest_id = ids[nearest_index[0], nearest_index[1]]

    ridge = np.zeros(ids.shape, bool)
    gap_pairs: Counter[tuple[int, int]] = Counter()
    for dy, dx in SHIFTS:
        other_id = shifted(nearest_id, dy, dx)
        candidate = background & (nearest_id > 0) & (other_id > 0) & (nearest_id != other_id)
        candidate &= distance <= float(max_gap_px)
        ridge |= candidate
        aa = nearest_id[candidate]
        bb = other_id[candidate]
        for a, b in zip(aa.tolist(), bb.tolist()):
            gap_pairs[tuple(sorted((int(a), int(b))))] += 1

    reliable_gap = ridge
    if gap_halfwidth > 0:
        reliable_gap = ndimage.binary_dilation(reliable_gap, iterations=gap_halfwidth)
    reliable_gap &= background & (distance <= float(max_gap_px) + gap_halfwidth)

    contact = np.zeros(ids.shape, bool)
    contact_pairs: Counter[tuple[int, int]] = Counter()
    for dy, dx in SHIFTS:
        other = shifted(ids, dy, dx)
        c = (ids > 0) & (other > 0) & (ids != other)
        contact |= c
        aa, bb = ids[c], other[c]
        for a, b in zip(aa.tolist(), bb.tolist()):
            contact_pairs[tuple(sorted((int(a), int(b))))] += 1

    core = reliable_gap | contact
    interface = ndimage.binary_dilation(core, iterations=interface_radius) if core.any() else core.copy()

    any_boundary = label_boundary(ids)
    distance_to_label_boundary = ndimage.distance_transform_edt(~any_boundary)
    support = fg & interface & (distance_to_label_boundary >= float(support_depth))

    # Direct contacts and the transition band are ambiguous for imperfect labels.
    # They guide attention but never receive a hard background target.
    ignore = interface & ~(reliable_gap | support)
    return {
        "foreground": fg,
        "reliable_gap": reliable_gap,
        "contact": contact,
        "interface": interface,
        "support": support,
        "ignore": ignore,
        "gap_pairs": gap_pairs,
        "contact_pairs": contact_pairs,
    }


def normalize_image(a: np.ndarray) -> np.ndarray:
    x = a.astype(np.float32)
    lo, hi = np.percentile(x, (1.0, 99.0))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def deterministic_visual_indices(n: int, count: int) -> set[int]:
    if n <= count:
        return set(range(n))
    return set(np.linspace(0, n - 1, count, dtype=int).tolist())


def color_ids(ids: np.ndarray) -> np.ndarray:
    palette = np.array([
        [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
        [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
        [250, 190, 190], [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200],
    ], dtype=np.uint8)
    return palette[ids % len(palette)]


def save_panel(image: np.ndarray, ids: np.ndarray, targets: dict[str, object], path: Path, title: str) -> None:
    x = normalize_image(image)
    gray = np.uint8(x * 255)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    overlay = rgb.astype(np.float32)
    colors = {
        "interface": np.array([0, 220, 255]),
        "ignore": np.array([255, 210, 0]),
        "support": np.array([0, 255, 80]),
        "reliable_gap": np.array([255, 0, 180]),
        "contact": np.array([255, 70, 0]),
    }
    for key in ("interface", "ignore", "support", "reliable_gap", "contact"):
        mask = np.asarray(targets[key], dtype=bool)
        overlay[mask] = 0.35 * overlay[mask] + 0.65 * colors[key]
    overlay = np.uint8(np.clip(overlay, 0, 255))
    id_rgb = color_ids(ids)
    panels = [rgb, id_rgb, overlay]
    canvas = Image.new("RGB", (rgb.shape[1] * 3, rgb.shape[0] + 34), "black")
    for i, panel in enumerate(panels):
        canvas.paste(Image.fromarray(panel), (i * rgb.shape[1], 34))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{title} | original / instance IDs / targets", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def process_split(args: argparse.Namespace, split: str) -> tuple[list[dict[str, object]], Counter, Counter]:
    label_paths = sorted((args.dataset_root / f"{split}_labels").glob("*.png"))
    if args.limit:
        label_paths = label_paths[:args.limit]
    visual_indices = deterministic_visual_indices(len(label_paths), args.visuals_per_split)
    rows: list[dict[str, object]] = []
    all_gap_pairs: Counter = Counter()
    all_contact_pairs: Counter = Counter()
    for index, label_path in enumerate(label_paths):
        image_path = find_image(args.dataset_root, split, label_path.stem)
        ids_native = read_ids(label_path)
        image_native = np.asarray(Image.open(image_path).convert("L"))
        if image_native.shape != ids_native.shape:
            image_native = np.asarray(Image.fromarray(image_native).resize(
                (ids_native.shape[1], ids_native.shape[0]), Image.Resampling.BILINEAR))
        image, ids = resize_case(image_native, ids_native, args.audit_size)
        targets = build_targets(ids, args.max_gap_px, args.gap_halfwidth,
                                args.interface_radius, args.support_depth)
        all_gap_pairs.update(targets["gap_pairs"])
        all_contact_pairs.update(targets["contact_pairs"])
        n = float(ids.size)
        gap = np.asarray(targets["reliable_gap"], bool)
        support = np.asarray(targets["support"], bool)
        norm = normalize_image(image)
        gap_mean = float(norm[gap].mean()) if gap.any() else None
        support_mean = float(norm[support].mean()) if support.any() else None
        contrast = (support_mean - gap_mean) if gap_mean is not None and support_mean is not None else None
        row = {
            "split": split,
            "case": label_path.stem,
            "height": int(ids.shape[0]),
            "width": int(ids.shape[1]),
            "num_instances": int(np.count_nonzero(np.unique(ids) > 0)),
            "gap_pairs": len(targets["gap_pairs"]),
            "contact_pairs": len(targets["contact_pairs"]),
            "foreground_fraction": float(np.asarray(targets["foreground"]).sum() / n),
            "gap_fraction": float(gap.sum() / n),
            "interface_fraction": float(np.asarray(targets["interface"]).sum() / n),
            "support_fraction": float(support.sum() / n),
            "ignore_fraction": float(np.asarray(targets["ignore"]).sum() / n),
            "gap_mean_intensity": gap_mean,
            "support_mean_intensity": support_mean,
            "support_minus_gap_intensity": contrast,
        }
        rows.append(row)
        if index in visual_indices:
            save_panel(image, ids, targets, args.visual_root / f"{split}_{label_path.stem}.png",
                       f"{split}/{label_path.stem}")
    return rows, all_gap_pairs, all_contact_pairs


def finite_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(float(r[key]))]


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"cases": len(rows)}
    for key in ("num_instances", "gap_pairs", "contact_pairs", "foreground_fraction", "gap_fraction",
                "interface_fraction", "support_fraction", "ignore_fraction", "support_minus_gap_intensity"):
        values = finite_values(rows, key)
        if values:
            result[key] = {
                "mean": float(np.mean(values)), "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            }
    result["cases_without_gap"] = sum(int(r["gap_pairs"]) == 0 for r in rows)
    result["cases_without_support"] = sum(float(r["support_fraction"]) == 0 for r in rows)
    contrasts = finite_values(rows, "support_minus_gap_intensity")
    result["image_evidence_agreement_fraction"] = (
        float(np.mean(np.asarray(contrasts) > 0)) if contrasts else None
    )
    return result


def serializable_pairs(counter: Counter, limit: int = 50) -> list[dict[str, int]]:
    return [{"id_a": int(a), "id_b": int(b), "ridge_pixels": int(count)}
            for (a, b), count in counter.most_common(limit)]


def main() -> None:
    args = parse_args()
    joined = " ".join(str(v) for v in vars(args).values()).lower()
    if "articular" in joined or "clean-test" in joined or "test-v2" in joined:
        raise RuntimeError("R291 is restricted to TSRS_RSNA-Epiphysis train/original-val")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.visual_root.mkdir(parents=True, exist_ok=True)
    train_rows, train_gap_pairs, train_contact_pairs = process_split(args, "train")
    val_rows, val_gap_pairs, val_contact_pairs = process_split(args, "val")
    payload = {
        "run_id": "R291-0_image_driven_interface_target_audit",
        "dataset": "TSRS_RSNA-Epiphysis",
        "source_dataset_unchanged": True,
        "clean_test_v2_used": False,
        "age_or_sex_used": False,
        "fixed_pair_list_used": False,
        "gt_used_at_inference": False,
        "audit_parameters": {
            "audit_size": args.audit_size, "max_gap_px": args.max_gap_px,
            "gap_halfwidth": args.gap_halfwidth, "interface_radius": args.interface_radius,
            "support_depth": args.support_depth,
        },
        "definitions": {
            "reliable_gap": "background watershed ridge between two nearby observed instances",
            "interface": "local band around reliable gap or direct instance contact",
            "support": "instance interior inside interface and away from every label transition",
            "ignore": "ambiguous interface pixels; never a hard background target",
        },
        "train": summarize(train_rows),
        "original_val": summarize(val_rows),
        "train_top_gap_pairs": serializable_pairs(train_gap_pairs),
        "original_val_top_gap_pairs": serializable_pairs(val_gap_pairs),
        "train_top_contact_pairs": serializable_pairs(train_contact_pairs),
        "original_val_top_contact_pairs": serializable_pairs(val_contact_pairs),
        "gate": {
            "train_all_cases_have_support": all(float(r["support_fraction"]) > 0 for r in train_rows),
            "val_all_cases_have_support": all(float(r["support_fraction"]) > 0 for r in val_rows),
            "train_has_gap_supervision": any(int(r["gap_pairs"]) > 0 for r in train_rows),
            "val_has_gap_supervision": any(int(r["gap_pairs"]) > 0 for r in val_rows),
            "image_evidence_majority_agrees": (summarize(train_rows)["image_evidence_agreement_fraction"] or 0) >= 0.5,
        },
        "next_if_gate_pass": "implement R291-A paired nnU-Net loss ablation; no relation head",
    }
    (args.output_root / "case_rows.json").write_text(json.dumps(train_rows + val_rows, indent=2), encoding="utf-8")
    (args.output_root / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
