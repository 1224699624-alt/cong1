#!/usr/bin/env python3
"""Merge R215 background-channel features into an existing candidate CSV.

The merge key is (image, candidate_rank). The left CSV keeps its labels and
metric deltas; only R215 inference-time features are copied from the right CSV.
Missing matches receive zero-filled R215 features so downstream diagnostics can
still evaluate the original candidate universe.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


R215_INFERENCE_FEATURE_KEYS = [
    "r215_crop_h",
    "r215_crop_w",
    "r215_crop_area",
    "r215_cut_area",
    "r215_cut_frac_crop",
    "r215_bg_components_before",
    "r215_bg_components_after",
    "r215_bg_component_count_delta",
    "r215_cut_adjacent_bg_components_before",
    "r215_cut_adjacent_bg_frac_before",
    "r215_bg_components_merge_potential",
    "r215_cut_component_bg_area_after",
    "r215_cut_component_bg_frac_crop_after",
    "r215_cut_component_border_contacts_after",
    "r215_cut_component_touches_crop_border_after",
    "r215_fg_components_before",
    "r215_fg_components_after",
    "r215_fg_component_count_delta",
    "r215_bg_channel_proxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge R215 background-channel inference features into candidate CSV.")
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--r215-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("image") or ""), str(float(row.get("candidate_rank") or 0.0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_rows = read_csv(args.base_csv)
    r215_rows = read_csv(args.r215_csv)
    r215_by_key = {key(row): row for row in r215_rows}

    merged = []
    matched = 0
    for row in base_rows:
        out = dict(row)
        r215 = r215_by_key.get(key(row))
        if r215 is not None:
            matched += 1
        for feature in R215_INFERENCE_FEATURE_KEYS:
            out[feature] = (r215 or {}).get(feature, "0.0")
        out["r215_feature_matched"] = "1.0" if r215 is not None else "0.0"
        merged.append(out)

    report = {
        "run_id": "R215-background-feature-merge",
        "base_csv": str(args.base_csv),
        "r215_csv": str(args.r215_csv),
        "output_csv": str(args.output_csv),
        "num_base_rows": len(base_rows),
        "num_r215_rows": len(r215_rows),
        "num_merged_rows": len(merged),
        "num_matched": matched,
        "num_unmatched": len(base_rows) - matched,
        "match_rate": float(matched / max(1, len(base_rows))),
        "feature_keys": [*R215_INFERENCE_FEATURE_KEYS, "r215_feature_matched"],
        "clean_test_v2_used": False,
        "writes_masks": False,
    }
    write_csv(args.output_csv, merged)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
