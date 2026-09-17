#!/usr/bin/env python3
"""Fuse multiple candidate binary masks into a final mask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


DEFAULT_CANDIDATES = [
    "zero_shot_box_only_conf020_pad005",
    "zero_shot_mask_to_prompt_refine_soft",
    "zero_shot_mask_to_prompt_refine_pad003_neg2_k5",
    "zero_shot_mask_to_prompt_refine_pad003_neg8_k9",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse multiple YOLO-SAM candidate masks.")
    parser.add_argument("--dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ablations-root", default="outputs/ablations")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output-name", default="zero_shot_mask_to_prompt_fusion")
    return parser.parse_args()


def read_binary_mask(path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def main() -> None:
    args = parse_args()
    if args.weights is not None and len(args.weights) not in (0, len(args.candidates)):
        raise ValueError("--weights must be omitted or have the same length as --candidates")

    weights = np.asarray(args.weights if args.weights else [1.0] * len(args.candidates), dtype=np.float32)
    weights = weights / weights.sum()

    ablations_root = Path(args.ablations_root)
    candidate_dirs = [
        ablations_root / candidate / args.dataset / args.split / "masks"
        for candidate in args.candidates
    ]
    for candidate, mask_dir in zip(args.candidates, candidate_dirs, strict=True):
        if not mask_dir.exists():
            raise FileNotFoundError(f"Candidate masks not found for {candidate}: {mask_dir}")

    output_dir = ablations_root / args.output_name / args.dataset / args.split / "masks"
    first_files = sorted(candidate_dirs[0].glob("*.png"))
    records = []
    for mask_path in tqdm(first_files, desc=f"fuse/{args.dataset}/{args.split}"):
        stack = []
        missing = []
        for candidate, mask_dir in zip(args.candidates, candidate_dirs, strict=True):
            current_path = mask_dir / mask_path.name
            if not current_path.exists():
                missing.append(candidate)
                stack.append(np.zeros_like(read_binary_mask(mask_path), dtype=np.float32))
                continue
            stack.append(read_binary_mask(current_path).astype(np.float32))
        score = np.tensordot(weights, np.stack(stack, axis=0), axes=(0, 0))
        fused = score >= args.threshold
        save_binary_mask(fused, output_dir / mask_path.name)
        records.append(
            {
                "image": mask_path.name,
                "mask_area": int(fused.sum()),
                "missing_candidates": missing,
            }
        )

    out_root = output_dir.parent
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "fusion_records.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "candidates": args.candidates,
                "weights": [float(weight) for weight in weights],
                "threshold": float(args.threshold),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved fused masks to: {output_dir}")


if __name__ == "__main__":
    main()
