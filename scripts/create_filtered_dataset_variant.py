#!/usr/bin/env python3
"""Create a filtered dataset variant and matching candidate-mask subset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a filtered dataset variant without touching the original dataset.")
    parser.add_argument("--manifest", required=True, help="JSON manifest describing the excluded sample stems by split.")
    parser.add_argument("--source-raw-root", default="data/raw")
    parser.add_argument("--target-raw-root", default="data/raw_variants")
    parser.add_argument("--source-ablations-root", default="outputs/ablations")
    parser.add_argument("--target-ablations-root", default="outputs/ablations_variants")
    parser.add_argument("--experiments", nargs="*", default=[], help="Candidate-mask experiments to subset for the filtered dataset variant.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the target dataset variant and variant candidate-mask folders.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "source_dataset" not in payload or "target_dataset" not in payload:
        raise ValueError("Manifest must contain 'source_dataset' and 'target_dataset'.")
    payload.setdefault("excludes", {})
    return payload


def normalize_excludes(raw_excludes: dict) -> dict[str, set[str]]:
    excludes: dict[str, set[str]] = {}
    for split in SPLITS:
        split_payload = raw_excludes.get(split, {})
        if isinstance(split_payload, dict):
            excludes[split] = set(split_payload.keys())
        elif isinstance(split_payload, list):
            excludes[split] = {str(item) for item in split_payload}
        else:
            excludes[split] = set()
    return excludes


def ensure_safe_target(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved not in target_resolved.parents and root_resolved != target_resolved:
        raise ValueError(f"Refusing to touch target outside root: {target_resolved}")


def remove_existing(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    source_dataset = str(manifest["source_dataset"])
    target_dataset = str(manifest["target_dataset"])
    excludes = normalize_excludes(manifest.get("excludes", {}))

    source_raw_root = Path(args.source_raw_root)
    target_raw_root = Path(args.target_raw_root)
    source_ablations_root = Path(args.source_ablations_root)
    target_ablations_root = Path(args.target_ablations_root)

    source_dataset_dir = source_raw_root / source_dataset
    target_dataset_dir = target_raw_root / target_dataset
    if not source_dataset_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset_dir}")

    ensure_safe_target(target_raw_root, target_dataset_dir)
    if target_dataset_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Target dataset variant already exists: {target_dataset_dir}. Use --overwrite to replace it.")
        remove_existing(target_dataset_dir)
    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int | list[str]]] = {}
    for split in SPLITS:
        image_dir = source_dataset_dir / split
        label_dir = source_dataset_dir / f"{split}_labels"
        if not image_dir.exists() or not label_dir.exists():
            raise FileNotFoundError(f"Missing split dirs for {split}: {image_dir} / {label_dir}")

        target_image_dir = target_dataset_dir / split
        target_label_dir = target_dataset_dir / f"{split}_labels"
        target_image_dir.mkdir(parents=True, exist_ok=True)
        target_label_dir.mkdir(parents=True, exist_ok=True)

        kept = 0
        removed = 0
        removed_stems = sorted(excludes[split])
        for label_path in sorted(label_dir.glob("*.png")):
            stem = label_path.stem
            if stem in excludes[split]:
                removed += 1
                continue
            image_path = find_image_for_stem(image_dir, stem)
            if image_path is None:
                raise FileNotFoundError(f"Image for stem '{stem}' not found in {image_dir}")
            shutil.copy2(image_path, target_image_dir / image_path.name)
            shutil.copy2(label_path, target_label_dir / label_path.name)
            kept += 1

        summary[split] = {
            "kept": kept,
            "removed": removed,
            "removed_stems": removed_stems,
        }

    for experiment in args.experiments:
        source_exp_root = source_ablations_root / experiment / source_dataset
        if not source_exp_root.exists():
            raise FileNotFoundError(f"Source experiment masks not found: {source_exp_root}")
        target_exp_root = target_ablations_root / experiment / target_dataset
        ensure_safe_target(target_ablations_root, target_exp_root)
        if target_exp_root.exists() and args.overwrite:
            remove_existing(target_exp_root)

        for split in SPLITS:
            source_mask_dir = source_exp_root / split / "masks"
            if not source_mask_dir.exists():
                raise FileNotFoundError(f"Source mask dir not found: {source_mask_dir}")
            target_mask_dir = target_exp_root / split / "masks"
            target_mask_dir.mkdir(parents=True, exist_ok=True)
            for mask_path in sorted(source_mask_dir.glob("*.png")):
                if mask_path.stem in excludes[split]:
                    continue
                shutil.copy2(mask_path, target_mask_dir / mask_path.name)

    metadata = {
        "manifest": str(manifest_path),
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "source_raw_root": str(source_raw_root),
        "target_raw_root": str(target_raw_root),
        "source_ablations_root": str(source_ablations_root),
        "target_ablations_root": str(target_ablations_root),
        "experiments": list(args.experiments),
        "summary": summary,
        "notes": manifest.get("notes", ""),
    }
    (target_dataset_dir / "filter_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
