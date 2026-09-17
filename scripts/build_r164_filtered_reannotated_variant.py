#!/usr/bin/env python3
"""Build a filtered train/val-only reannotated Epiphysis variant for R164."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered reannotated train/val variant.")
    parser.add_argument("--original", type=Path, default=Path("data/raw/TSRS_RSNA-Epiphysis"))
    parser.add_argument("--source", type=Path, default=Path("data/raw_variants/TSRS_RSNA-Epiphysis_reannotated_trainval_only_v1"))
    parser.add_argument("--target-root", type=Path, default=Path("data/raw_variants"))
    parser.add_argument("--target-name", default="TSRS_RSNA-Epiphysis_reannotated_filtered_trainval_v1")
    parser.add_argument("--pair-shift-json", type=Path, default=Path("outputs/analysis/r163_reannotated_pair_shift.json"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r164_filtered_reannotated_variant_build.json"))
    parser.add_argument("--min-paired-dice", type=float, default=0.65)
    parser.add_argument("--min-fg-ratio", type=float, default=0.45)
    parser.add_argument("--max-fg-ratio", type=float, default=2.25)
    parser.add_argument("--max-abs-component-delta", type=float, default=12.0)
    parser.add_argument("--min-fg-frac", type=float, default=0.002)
    parser.add_argument("--max-fg-frac", type=float, default=0.09)
    parser.add_argument("--min-components", type=int, default=8)
    parser.add_argument("--max-components", type=int, default=35)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def component_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return sum(1 for idx in range(1, n) if int(stats[idx, cv2.CC_STAT_AREA]) >= 4)


def basic_reann_gate(label_path: Path, args: argparse.Namespace) -> tuple[bool, list[str], dict[str, float]]:
    mask = read_mask(label_path)
    fg_frac = float(mask.mean())
    components = float(component_count(mask))
    reasons: list[str] = []
    if fg_frac < args.min_fg_frac:
        reasons.append("fg_frac_too_low")
    if fg_frac > args.max_fg_frac:
        reasons.append("fg_frac_too_high")
    if components < args.min_components:
        reasons.append("component_count_too_low")
    if components > args.max_components:
        reasons.append("component_count_too_high")
    return not reasons, reasons, {"fg_frac": fg_frac, "component_count": components}


def image_path(root: Path, split: str, stem: str) -> Path:
    folder = root / split
    matches = sorted(p for p in folder.glob(f"{stem}.*") if p.is_file())
    if not matches:
        raise FileNotFoundError(f"missing image for {split}/{stem} under {folder}")
    return matches[0]


def copy_pair(src_root: Path, dst_root: Path, split: str, name: str, dry_run: bool) -> None:
    stem = Path(name).stem
    src_img = image_path(src_root, split, stem)
    src_lab = src_root / f"{split}_labels" / f"{stem}.png"
    if not src_lab.exists():
        raise FileNotFoundError(src_lab)
    if dry_run:
        return
    (dst_root / split).mkdir(parents=True, exist_ok=True)
    (dst_root / f"{split}_labels").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_img, dst_root / split / src_img.name)
    shutil.copy2(src_lab, dst_root / f"{split}_labels" / src_lab.name)


def select_train(args: argparse.Namespace, pair_shift: dict[str, object]) -> tuple[list[str], list[dict[str, object]]]:
    source_names = {p.name for p in (args.source / "train_labels").glob("*.png")}
    paired_records = pair_shift["paired"]["train"]["per_image"]  # type: ignore[index]
    paired_by_name = {str(r["image"]): r for r in paired_records}
    selected: list[str] = []
    rejected: list[dict[str, object]] = []
    for name in sorted(source_names):
        ok, reasons, features = basic_reann_gate(args.source / "train_labels" / name, args)
        rec = paired_by_name.get(name)
        if rec:
            if float(rec["label_dice"]) < args.min_paired_dice:
                reasons.append("paired_dice_too_low")
            ratio = float(rec["fg_frac_ratio"])
            if ratio < args.min_fg_ratio:
                reasons.append("fg_ratio_too_low")
            if ratio > args.max_fg_ratio:
                reasons.append("fg_ratio_too_high")
            if abs(float(rec["component_count_delta"])) > args.max_abs_component_delta:
                reasons.append("component_delta_too_large")
            features.update({
                "paired": True,
                "label_dice": float(rec["label_dice"]),
                "fg_frac_ratio": ratio,
                "component_count_delta": float(rec["component_count_delta"]),
            })
        else:
            features["paired"] = False
        if ok and not reasons:
            selected.append(name)
        else:
            rejected.append({"image": name, "reasons": sorted(set(reasons)), "features": features})
    return selected, rejected


def select_val(args: argparse.Namespace) -> tuple[list[str], list[dict[str, object]]]:
    selected: list[str] = []
    rejected: list[dict[str, object]] = []
    for path in sorted((args.source / "val_labels").glob("*.png")):
        ok, reasons, features = basic_reann_gate(path, args)
        if ok:
            selected.append(path.name)
        else:
            rejected.append({"image": path.name, "reasons": sorted(set(reasons)), "features": features})
    return selected, rejected


def split_counts(root: Path, split: str) -> dict[str, int]:
    images = {p.stem for p in (root / split).glob("*") if p.is_file()}
    labels = {p.stem for p in (root / f"{split}_labels").glob("*.png") if p.is_file()}
    return {
        "images": len(images),
        "labels": len(labels),
        "paired": len(images & labels),
        "images_without_labels": len(images - labels),
        "labels_without_images": len(labels - images),
    }


def main() -> None:
    args = parse_args()
    target = args.target_root / args.target_name
    pair_shift = json.loads(args.pair_shift_json.read_text(encoding="utf-8"))
    train_selected, train_rejected = select_train(args, pair_shift)
    val_selected, val_rejected = select_val(args)
    test_names = sorted(p.name for p in (args.original / "test_labels").glob("*.png"))

    if target.exists() and not args.dry_run:
        if not args.overwrite:
            raise RuntimeError(f"target exists; use --overwrite to replace: {target}")
        shutil.rmtree(target)
    if not args.dry_run:
        target.mkdir(parents=True, exist_ok=False)
    for name in train_selected:
        copy_pair(args.source, target, "train", name, args.dry_run)
    for name in val_selected:
        copy_pair(args.source, target, "val", name, args.dry_run)
    for name in test_names:
        copy_pair(args.original, target, "test", name, args.dry_run)

    summary = {
        "status": "dry_run_ok" if args.dry_run else "created",
        "source": str(args.source),
        "original": str(args.original),
        "target": str(target),
        "policy": (
            "Filtered train/val are copied from the safe reannotated train/val-only variant after excluding empty, "
            "extreme paired-shift, extreme foreground-area, and extreme component-count labels. Test is copied only "
            "from the original dataset as a control. Reannotated test remains excluded. Success evaluation remains "
            "TSRS_RSNA-Epiphysis_clean_test_v2/test only."
        ),
        "thresholds": {
            "min_paired_dice": args.min_paired_dice,
            "min_fg_ratio": args.min_fg_ratio,
            "max_fg_ratio": args.max_fg_ratio,
            "max_abs_component_delta": args.max_abs_component_delta,
            "min_fg_frac": args.min_fg_frac,
            "max_fg_frac": args.max_fg_frac,
            "min_components": args.min_components,
            "max_components": args.max_components,
        },
        "selected": {
            "train": len(train_selected),
            "val": len(val_selected),
            "test": len(test_names),
        },
        "rejected": {
            "train": len(train_rejected),
            "val": len(val_rejected),
            "train_by_reason": {},
            "val_by_reason": {},
            "train_examples": train_rejected[:50],
            "val_examples": val_rejected[:50],
        },
    }
    for bucket, rows in (("train_by_reason", train_rejected), ("val_by_reason", val_rejected)):
        counts: dict[str, int] = {}
        for row in rows:
            for reason in row["reasons"]:
                counts[str(reason)] = counts.get(str(reason), 0) + 1
        split = "train" if bucket.startswith("train") else "val"
        summary["rejected"][bucket] = dict(sorted(counts.items()))
        summary["rejected"][f"{split}_examples"] = rows[:50]
    if not args.dry_run:
        summary["created_counts"] = {split: split_counts(target, split) for split in ("train", "val", "test")}
        (target / "r164_filtered_reannotated_metadata.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
