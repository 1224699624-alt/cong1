#!/usr/bin/env python3
"""Summarize the remote R244/R249/R250/R248 state for the ARIS loop.

This local helper is read-only. It connects to the experiment server, checks the
R244 generator, R249/R250 watchers, and whether full R245/R246/R248 outputs have
appeared. It does not start, stop, or modify remote jobs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize remote R244/R249 experiment state.")
    parser.add_argument("--host", default="10.1.115.157")
    parser.add_argument("--user", default="shenzeyu")
    parser.add_argument("--password", default="")
    parser.add_argument("--remote-root", default="/home/shenzeyu/workspace/YOLO_SAM_generic_src")
    parser.add_argument("--output-json", type=Path, default=Path("outputs/analysis/r244_r249_remote_state_latest.json"))
    return parser.parse_args()


REMOTE_SCRIPT = r'''
cd "__REMOTE_ROOT__" || exit 1
printf "\n===SCREEN===\n"
screen -ls || true
printf "\n===PROC===\n"
ps -eo pid,etime,pcpu,pmem,cmd | grep -E "audit_r244|run_r249|run_r250|search_r245|audit_r246|train_r248" | grep -v grep || true
printf "\n===R244_TAIL===\n"
tail -20 outputs/bridge_logs/r244_gtfree_merge_targeted_fullval.log 2>/dev/null || true
printf "\n===R249_TAIL===\n"
tail -40 outputs/bridge_logs/r249_wait_r244_then_full_audit.log 2>/dev/null || true
printf "\n===R250_TAIL===\n"
tail -40 outputs/bridge_logs/r250_wait_r249_then_maybe_r248.log 2>/dev/null || true
printf "\n===ARTIFACTS===\n"
ls -lh \
  outputs/analysis/r244_gtfree_merge_targeted_fullval_summary.json \
  outputs/analysis/r244_gtfree_merge_targeted_fullval_candidates.csv \
  outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json \
  outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.csv \
  outputs/analysis/r246_gtfree_merge_feature_failure.fullval.json \
  outputs/analysis/r246_gtfree_merge_feature_failure_feature_table.fullval.csv \
  outputs/analysis/r248_r244_context_crop_scorer_fullval.json \
  outputs/analysis/r248_r244_context_crop_scorer_fullval_threshold_grid.csv \
  outputs/analysis/r248_r244_context_crop_scorer_fullval_probed_rows.csv \
  2>/dev/null || true
printf "\n===JSON===\n"
python3 - <<'PY'
import json, os
paths = {
    "r244": "outputs/analysis/r244_gtfree_merge_targeted_fullval_summary.json",
    "r245": "outputs/analysis/r245_gtfree_merge_filter_search.fullval.top1.json",
    "r246": "outputs/analysis/r246_gtfree_merge_feature_failure.fullval.json",
    "r248": "outputs/analysis/r248_r244_context_crop_scorer_fullval.json",
}
out = {}
for key, path in paths.items():
    item = {"path": path, "exists": os.path.exists(path)}
    if os.path.exists(path):
        try:
            data = json.load(open(path))
            if key == "r244":
                item.update({
                    "num_candidate_images": data.get("num_candidate_images"),
                    "num_candidate_rows": data.get("num_candidate_rows"),
                    "decision": data.get("decision"),
                    "oracle_images": (data.get("oracle_safe_best_per_image") or {}).get("num_images"),
                    "oracle_boundary_iou": (data.get("oracle_safe_best_per_image") or {}).get("mean_delta_boundary_iou"),
                    "oracle_gap_fp": (data.get("oracle_safe_best_per_image") or {}).get("mean_delta_gap_region_fp_rate"),
                    "oracle_component_count_mae": (data.get("oracle_safe_best_per_image") or {}).get("mean_delta_component_count_mae"),
                    "clean_safe": (data.get("gtfree_clean_subset") or {}).get("num_safe_useful"),
                    "clean_risk": (data.get("gtfree_clean_subset") or {}).get("num_risk"),
                })
            elif key == "r245":
                best = data.get("best") or {}
                item.update({
                    "decision": data.get("decision"),
                    "num_passing_configs": data.get("num_passing_configs"),
                    "num_input_rows": data.get("num_input_rows"),
                    "num_input_images": data.get("num_input_images"),
                    "best_rows": best.get("num_rows"),
                    "best_images": best.get("num_images"),
                    "best_safe": best.get("num_safe_useful"),
                    "best_risk": best.get("num_risk"),
                    "best_recall": best.get("mean_delta_recall"),
                    "best_boundary_iou": best.get("mean_delta_boundary_iou"),
                    "best_gap_fp": best.get("mean_delta_gap_region_fp_rate"),
                    "best_cut_gt_fg_frac": best.get("mean_cut_gt_fg_frac"),
                })
            elif key == "r246":
                groups = data.get("groups") or {}
                item["groups"] = {name: {
                    "num_rows": group.get("num_rows"),
                    "num_images": group.get("num_images"),
                    "num_safe_useful": group.get("num_safe_useful"),
                    "num_risk": group.get("num_risk"),
                    "mean_delta_recall": group.get("mean_delta_recall"),
                    "mean_delta_boundary_iou": group.get("mean_delta_boundary_iou"),
                    "mean_delta_gap_region_fp_rate": group.get("mean_delta_gap_region_fp_rate"),
                    "mean_cut_gt_fg_frac": group.get("mean_cut_gt_fg_frac"),
                } for name, group in groups.items()}
            elif key == "r248":
                item.update({
                    "num_candidate_rows": data.get("num_candidate_rows"),
                    "num_candidate_images": data.get("num_candidate_images"),
                    "num_passing_configs": (data.get("grouped_cv") or {}).get("num_passing_configs"),
                    "best": (data.get("grouped_cv") or {}).get("best"),
                })
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
    out[key] = item
print(json.dumps(out, indent=2))
PY
'''


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "START"
    for line in text.splitlines():
        if line.startswith("===") and line.endswith("==="):
            current = line.strip("=").strip()
            sections[current] = []
        else:
            sections.setdefault(current, []).append(line)
    return {key: "\n".join(value).strip() for key, value in sections.items()}


def last_progress(text: str) -> dict[str, int | None]:
    matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
    if not matches:
        return {"current": None, "total": None}
    current, total = matches[-1]
    return {"current": int(current), "total": int(total)}


def last_r249_summary(text: str) -> dict[str, int | None]:
    matches = re.findall(r"summary images=\s*(\d+)\s+rows=\s*(\d+)", text)
    if not matches:
        return {"images": None, "rows": None}
    images, rows = matches[-1]
    return {"images": int(images), "rows": int(rows)}


def derive_status(sections: dict[str, str], parsed_json: dict[str, Any]) -> dict[str, Any]:
    proc = sections.get("PROC", "")
    r249_tail = sections.get("R249_TAIL", "")
    r250_tail = sections.get("R250_TAIL", "")
    artifacts = sections.get("ARTIFACTS", "")
    return {
        "r244_running": "audit_r244_gtfree_merge_targeted_candidates.py" in proc,
        "r249_waiting": "run_r249_wait_r244_then_full_audit.sh" in proc and "still waiting for R244" in r249_tail,
        "r250_waiting": "run_r250_wait_r249_then_maybe_r248.sh" in proc and "still waiting for R249" in r250_tail,
        "r248_running": "train_r248_r244_context_crop_scorer.py" in proc,
        "r244_log_progress": last_progress(sections.get("R244_TAIL", "")),
        "r249_seen_summary": last_r249_summary(r249_tail),
        "full_r245_exists": bool((parsed_json.get("r245") or {}).get("exists")),
        "full_r246_exists": bool((parsed_json.get("r246") or {}).get("exists")),
        "full_r248_exists": bool((parsed_json.get("r248") or {}).get("exists")),
        "artifact_lines": [line for line in artifacts.splitlines() if line.strip()],
    }


def main() -> None:
    args = parse_args()
    if not args.password:
        raise SystemExit("--password is required for this read-only SSH helper")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=args.password, timeout=15, look_for_keys=False, allow_agent=False)
    command = REMOTE_SCRIPT.replace("__REMOTE_ROOT__", args.remote_root)
    _stdin, stdout, stderr = client.exec_command(command, timeout=60)
    text = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    client.close()
    sections = split_sections(text)
    parsed: dict[str, Any] = {
        "host": args.host,
        "remote_root": args.remote_root,
        "screen": sections.get("SCREEN", ""),
        "proc": sections.get("PROC", ""),
        "r244_tail": sections.get("R244_TAIL", ""),
        "r249_tail": sections.get("R249_TAIL", ""),
        "r250_tail": sections.get("R250_TAIL", ""),
        "artifacts": sections.get("ARTIFACTS", ""),
        "stderr": err.strip(),
    }
    json_text = sections.get("JSON", "{}")
    try:
        parsed["json"] = json.loads(json_text)
    except json.JSONDecodeError:
        parsed["json_parse_error"] = json_text
        parsed["json"] = {}
    parsed["parsed_status"] = derive_status(sections, parsed.get("json") or {})
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
