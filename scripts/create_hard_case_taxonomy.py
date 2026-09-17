#!/usr/bin/env python3
"""Create a preliminary taxonomy for visual hard-case audit panels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create hard-case taxonomy from audit rows.")
    parser.add_argument("--audit-pack-dir", required=True)
    parser.add_argument("--output-prefix", default="R123")
    return parser.parse_args()


def auto_tags(row: dict[str, object]) -> list[str]:
    tags: list[str] = []
    precision = float(row["precision"])
    recall = float(row["recall"])
    boundary_iou = float(row["boundary_iou"])
    component_delta = int(row["component_delta"])
    disagree = float(row.get("error_in_disagreement_share") or 0.0)
    oracle = float(row.get("disagreement_pixel_oracle_dice") or 0.0)

    if component_delta <= -5:
        tags.append("under_separated_bridge")
    if precision < 0.88 and recall > 0.94:
        tags.append("overmask_low_precision")
    if recall < 0.88 and precision > 0.91:
        tags.append("underreach_high_precision")
    if boundary_iou < 0.15:
        tags.append("severe_boundary_shift")
    if disagree >= 0.55 and oracle >= 0.95:
        tags.append("candidate_fixable")
    if disagree < 0.40 and oracle < 0.94:
        tags.append("candidate_shared_failure")
    if not tags:
        tags.append("mixed_boundary_bridge")
    return tags


def suggested_action(tags: list[str]) -> str:
    tag_set = set(tags)
    if "candidate_fixable" in tag_set and "candidate_shared_failure" not in tag_set:
        return "candidate-guided training only if matching train/val examples can be curated"
    if "candidate_shared_failure" in tag_set:
        return "visual/label review or collect matching hard train examples; do not use candidate readout alone"
    if "under_separated_bridge" in tag_set or "overmask_low_precision" in tag_set:
        return "inspect bridge/low-contrast pattern; consider hard-case data curation before new model"
    if "underreach_high_precision" in tag_set:
        return "inspect label extent; consider boundary/recall protocol review"
    return "manual review before experiment"


def main() -> None:
    args = parse_args()
    pack_dir = Path(args.audit_pack_dir)
    rows = json.loads((pack_dir / "audit_rows.json").read_text(encoding="utf-8"))
    out_rows = []
    counts: Counter[str] = Counter()
    for row in rows:
        tags = auto_tags(row)
        counts.update(tags)
        out_rows.append({
            **row,
            "auto_tags": tags,
            "primary_tag": tags[0],
            "needs_human_review": True,
            "human_tag": "",
            "human_notes": "",
            "suggested_action": suggested_action(tags),
        })

    json_path = pack_dir / f"{args.output_prefix}_hard_case_taxonomy.json"
    csv_path = pack_dir / f"{args.output_prefix}_hard_case_taxonomy.csv"
    md_path = pack_dir / f"{args.output_prefix}_HARD_CASE_TAXONOMY.md"
    json_path.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")

    fieldnames = [
        "rank", "image", "dice", "precision", "recall", "boundary_iou", "component_delta",
        "error_in_disagreement_share", "disagreement_pixel_oracle_dice", "primary_tag",
        "auto_tags", "human_tag", "human_notes", "suggested_action", "panel",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: (";".join(row[key]) if key == "auto_tags" else row.get(key, "")) for key in fieldnames})

    md = [
        "# R123 Hard-Case Taxonomy",
        "",
        "This is an auto-initialized taxonomy for the R122 visual audit pack. Treat `auto_tags` as a triage aid, not final labels.",
        "",
        "## Tag Counts",
        "",
    ]
    for tag, count in counts.most_common():
        md.append(f"- `{tag}`: {count}")
    md.extend([
        "",
        "## Decision Gate",
        "",
        "- If most manually confirmed cases are `candidate_fixable`, curate matching train/val examples before a new learner.",
        "- If many cases are `candidate_shared_failure`, stop candidate readout experiments and revise data/label protocol.",
        "- If `under_separated_bridge` dominates, prioritize hard-case training data or explicit bridge labels, not threshold sweeps.",
        "",
        "## Cases",
        "",
        "| Rank | Image | Dice | Auto Tags | Suggested Action | Panel |",
        "| ---: | --- | ---: | --- | --- | --- |",
    ])
    for row in out_rows:
        md.append(
            f"| {row['rank']} | {row['image']} | {float(row['dice']):.4f} | "
            f"`{', '.join(row['auto_tags'])}` | {row['suggested_action']} | "
            f"[panel]({row['panel']}) |"
        )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "tag_counts": dict(counts),
    }, indent=2))


if __name__ == "__main__":
    main()
