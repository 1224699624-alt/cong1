#!/usr/bin/env python3
"""Create a hard-case curation manifest from R124 retrieval results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PRIMARY_QUOTAS = {
    "under_separated_bridge": 18,
    "underreach_high_precision": 18,
    "candidate_shared_failure": 10,
    "overmask_low_precision": 8,
    "severe_boundary_shift": 6,
    "mixed_boundary_bridge": 4,
    "all_hard": 4,
    "candidate_fixable": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hard-case curation manifest.")
    parser.add_argument("--retrieval-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dataset", default="TSRS_RSNA-Epiphysis")
    parser.add_argument("--target-name", default="r125_hard_case_curation_manifest")
    parser.add_argument("--max-train", type=int, default=64)
    parser.add_argument("--max-val", type=int, default=16)
    parser.add_argument("--max-score", type=float, default=0.70)
    return parser.parse_args()


def compact(row: dict[str, Any], reason: str) -> dict[str, Any]:
    features = row.get("features", {})
    return {
        "image": str(row["image"]),
        "stem": Path(str(row["image"])).stem,
        "split": str(row["split"]),
        "reason": reason,
        "nearest_prototype": str(row["nearest_prototype"]),
        "hardcase_similarity_score": float(row["hardcase_similarity_score"]),
        "panel": row.get("panel", ""),
        "features": {
            "component_count": float(features.get("component_count", 0.0)),
            "fg_frac": float(features.get("fg_frac", 0.0)),
            "nearest_center_gap": float(features.get("nearest_center_gap", 0.0)),
            "fg_bg_contrast": float(features.get("fg_bg_contrast", 0.0)),
            "image_std": float(features.get("image_std", 0.0)),
            "roi_std": float(features.get("roi_std", 0.0)),
        },
        "top_contributors": row.get("top_contributors", [])[:4],
    }


def quota_select(rows: list[dict[str, Any]], max_items: int, max_score: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    by_proto: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if float(row["hardcase_similarity_score"]) <= max_score:
            by_proto[str(row["nearest_prototype"])].append(row)
    for proto, quota in PRIMARY_QUOTAS.items():
        for row in by_proto.get(proto, [])[:quota]:
            key = f"{row['split']}:{row['image']}"
            if key in used:
                continue
            selected.append(row)
            used.add(key)
            if len(selected) >= max_items:
                return selected
    for row in rows:
        if float(row["hardcase_similarity_score"]) > max_score:
            continue
        key = f"{row['split']}:{row['image']}"
        if key in used:
            continue
        selected.append(row)
        used.add(key)
        if len(selected) >= max_items:
            break
    return selected


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "image",
        "reason",
        "nearest_prototype",
        "hardcase_similarity_score",
        "component_count",
        "fg_frac",
        "nearest_center_gap",
        "fg_bg_contrast",
        "panel",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            features = row["features"]
            writer.writerow({
                "split": row["split"],
                "image": row["image"],
                "reason": row["reason"],
                "nearest_prototype": row["nearest_prototype"],
                "hardcase_similarity_score": f"{row['hardcase_similarity_score']:.6f}",
                "component_count": f"{features['component_count']:.0f}",
                "fg_frac": f"{features['fg_frac']:.6f}",
                "nearest_center_gap": f"{features['nearest_center_gap']:.6f}",
                "fg_bg_contrast": f"{features['fg_bg_contrast']:.6f}",
                "panel": row.get("panel", ""),
            })


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    train_rows = manifest["hard_train"]
    val_rows = manifest["hard_val_audit"]
    train_counts = Counter(row["nearest_prototype"] for row in train_rows)
    val_counts = Counter(row["nearest_prototype"] for row in val_rows)
    md = [
        "# R125 Hard-Case Curation Manifest",
        "",
        "This manifest is an audit/curation artifact. It does not modify the original dataset and does not use the new reannotated test split.",
        "",
        "## Summary",
        "",
        f"- Source dataset: `{manifest['source_dataset']}`",
        f"- Hard train candidates: `{len(train_rows)}`",
        f"- Hard val audit candidates: `{len(val_rows)}`",
        f"- Train prototype counts: `{dict(train_counts)}`",
        f"- Val prototype counts: `{dict(val_counts)}`",
        "",
        "## Decision",
        "",
        "Visual spot checks of R124 top panels confirm train-side analogues of low-contrast wrist/carpal multi-component cases and bridge/boundary-fragment modes.",
        "The pool is not dominated by `candidate_fixable`, so the next GPU run should not be another candidate-fusion readout. A justified next model change would need to use this manifest for hard-case sampling or a label/layout protocol variant, with clean-test-v2 kept only for final evaluation.",
        "",
        "## Recommended Use",
        "",
        "- Review the listed panels and mark any label-protocol ambiguity before training.",
        "- If training is launched, oversample `hard_train` in an isolated R126-style experiment instead of changing the original dataset.",
        "- Keep `hard_val_audit` as a diagnostic slice, not as the final success metric.",
        "",
        "## Top Hard Train",
        "",
        "| Rank | Image | Prototype | Score | Reason | Panel |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ]
    for idx, row in enumerate(train_rows[:40], start=1):
        panel = row.get("panel", "")
        panel_link = f"[panel]({panel})" if panel else ""
        md.append(
            f"| {idx} | {row['image']} | `{row['nearest_prototype']}` | "
            f"{row['hardcase_similarity_score']:.3f} | {row['reason']} | {panel_link} |"
        )
    md.extend([
        "",
        "## Hard Val Audit",
        "",
        "| Rank | Image | Prototype | Score | Reason | Panel |",
        "| ---: | --- | --- | ---: | --- | --- |",
    ])
    for idx, row in enumerate(val_rows, start=1):
        panel = row.get("panel", "")
        panel_link = f"[panel]({panel})" if panel else ""
        md.append(
            f"| {idx} | {row['image']} | `{row['nearest_prototype']}` | "
            f"{row['hardcase_similarity_score']:.3f} | {row['reason']} | {panel_link} |"
        )
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def copy_panels(rows: list[dict[str, Any]], retrieval_json: Path, output_dir: Path) -> None:
    retrieval_dir = retrieval_json.parent
    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        panel = str(row.get("panel") or "")
        if not panel:
            continue
        src = retrieval_dir / panel
        if not src.exists():
            row["panel"] = ""
            continue
        dst = panel_dir / src.name
        shutil.copy2(src, dst)
        row["panel"] = str(dst.relative_to(output_dir)).replace("\\", "/")


def main() -> None:
    args = parse_args()
    retrieval_json = Path(args.retrieval_json)
    payload = json.loads(retrieval_json.read_text(encoding="utf-8"))
    rows = payload["top_candidates"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_pool = [row for row in rows if row["split"] == "train"]
    val_pool = [row for row in rows if row["split"] == "val"]
    hard_train = [compact(row, "r124_hardcase_similarity_train_candidate") for row in quota_select(train_pool, args.max_train, args.max_score)]
    hard_val = [compact(row, "r124_hardcase_similarity_val_audit") for row in quota_select(val_pool, args.max_val, args.max_score)]
    copy_panels(hard_train + hard_val, retrieval_json, output_dir)

    manifest = {
        "run_id": "R125",
        "source_dataset": args.source_dataset,
        "retrieval_json": args.retrieval_json,
        "selection_policy": {
            "max_train": args.max_train,
            "max_val": args.max_val,
            "max_score": args.max_score,
            "prototype_quotas": PRIMARY_QUOTAS,
        },
        "hard_train": hard_train,
        "hard_val_audit": hard_val,
        "excluded_from_training": {
            "clean_test_v2": "All R122/R123 clean-test-v2 hard cases remain final-evaluation evidence only.",
            "new_reannotated_test": "Excluded by project rule.",
        },
        "next_experiment_gate": "Manual review of R124/R125 panels should confirm labels before any R126 hard-case oversampling or isolated data-variant training.",
    }

    json_path = output_dir / f"{args.target_name}.json"
    csv_path = output_dir / f"{args.target_name}.csv"
    md_path = output_dir / "R125_HARD_CASE_CURATION_MANIFEST.md"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(csv_path, hard_train + hard_val)
    write_report(md_path, manifest)
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "hard_train": len(hard_train),
        "hard_val_audit": len(hard_val),
        "train_counts": dict(Counter(row["nearest_prototype"] for row in hard_train)),
        "val_counts": dict(Counter(row["nearest_prototype"] for row in hard_val)),
    }, indent=2))


if __name__ == "__main__":
    main()
