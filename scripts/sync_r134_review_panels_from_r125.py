#!/usr/bin/env python3
"""Fill R134 train/val review panel paths from the current R125 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_R134 = Path("outputs/analysis/r134_label_protocol_review_manifest/r134_label_protocol_review_manifest.json")
DEFAULT_R125 = Path("outputs/analysis/r125_hard_case_curation_manifest/r125_hard_case_curation_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync R134 review panel paths from R125.")
    parser.add_argument("--r134-manifest", type=Path, default=DEFAULT_R134)
    parser.add_argument("--r125-manifest", type=Path, default=DEFAULT_R125)
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    r134 = load_json(args.r134_manifest)
    r125 = load_json(args.r125_manifest)

    panel_by_key: dict[tuple[str, str], str] = {}
    for key, split in (("hard_train", "train"), ("hard_val_audit", "val")):
        for row in r125.get(key, []):
            panel = str(row.get("panel") or "")
            if not panel:
                continue
            full_panel = f"outputs/analysis/r125_hard_case_curation_manifest/{panel}".replace("\\", "/")
            panel_by_key[(split, str(row.get("image")))] = full_panel

    updated = 0
    still_missing: list[dict[str, str]] = []
    for section in ("train_review_queue", "val_audit_queue"):
        for row in r134.get(section, []):
            key = (str(row.get("split")), str(row.get("image")))
            panel = panel_by_key.get(key)
            if panel and row.get("panel") != panel:
                row["panel"] = panel
                updated += 1
            if not row.get("panel"):
                still_missing.append({"split": key[0], "image": key[1]})

    if args.backup:
        backup = args.r134_manifest.with_suffix(args.r134_manifest.suffix + ".bak_panels")
        backup.write_text(args.r134_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    args.r134_manifest.write_text(json.dumps(r134, indent=2), encoding="utf-8")
    print(json.dumps({
        "r134_manifest": str(args.r134_manifest),
        "r125_manifest": str(args.r125_manifest),
        "updated_panel_paths": updated,
        "still_missing": still_missing,
    }, indent=2))


if __name__ == "__main__":
    main()
