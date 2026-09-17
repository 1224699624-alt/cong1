#!/usr/bin/env python3
"""Build R207 baseline coverage audit for R201 protocol readiness."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CANDIDATES = {
    "araa_danet_epoch98": {"family": "ARAA", "priority": "must", "label": "ARAA DANet epoch98"},
    "r110_r100_r108_patch_basic": {"family": "YOLO+SAM", "priority": "must", "label": "R110 current best"},
    "r202_nnunet2d": {"family": "nnU-Net", "priority": "risk_audit", "label": "R202 nnU-Net 2D"},
    "r106_swin_tiny_unet": {"family": "Swin/U-Net", "priority": "baseline_candidate", "label": "R106 Swin tiny U-Net"},
    "r097_timm_unet_convnext_tiny": {"family": "U-Net", "priority": "baseline_candidate", "label": "R097 ConvNeXt tiny U-Net"},
    "r143_highres_medical_recipe_unet": {"family": "U-Net", "priority": "included", "label": "R143 high-res medical U-Net"},
    "r177_boundary_preserving_separation_arbitrator": {"family": "failed_repair", "priority": "diagnostic_only", "label": "R177 boundary arbitrator"},
    "r179_boundary_gap_constrained_dinov3_instance_sep": {"family": "failed_repair", "priority": "diagnostic_only", "label": "R179 boundary/gap constrained"},
    "r184_r110_guarded_arbitrator_smoke": {"family": "failed_repair", "priority": "diagnostic_only", "label": "R184 guarded no-op"},
    "r186_fast_neck_gate_smoke": {"family": "failed_repair", "priority": "diagnostic_only", "label": "R186 fast neck gate"},
}


def read_r201_table(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["experiment_slug"]: row for row in csv.DictReader(f)}


def read_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mask_count(root: Path, slug: str) -> int:
    mask_dir = root / slug / "TSRS_RSNA-Epiphysis_clean_test_v2" / "test" / "masks"
    if not mask_dir.exists():
        return 0
    return len(list(mask_dir.glob("*.png")))


def row_for_candidate(slug: str, meta: dict[str, str], r201: dict[str, dict[str, str]], variants_root: Path, analysis_dir: Path) -> dict[str, Any]:
    r201_row = r201.get(slug)
    metrics = read_metrics(analysis_dir / f"{slug}_clean_test_v2_metrics.json")
    count = mask_count(variants_root, slug)
    legacy_mean = metrics.get("mean", {}) if metrics else {}
    evidence = r201_row.get("evidence_level") if r201_row else "not_in_r201"

    if meta["priority"] == "diagnostic_only":
        recommendation = "diagnostic_only_do_not_backfill"
    elif r201_row and evidence == "full_mask_recomputed":
        recommendation = "ready_for_r201_tables"
    elif count == 81:
        recommendation = "rerun_r201_or_register"
    elif metrics and int(metrics.get("num_evaluated", 0) or 0) == 81:
        recommendation = "sync_or_regenerate_masks_for_full_r201"
    elif r201_row:
        recommendation = "diagnostic_only_legacy_metrics"
    else:
        recommendation = "missing_or_low_priority"

    return {
        "slug": slug,
        "label": meta["label"],
        "family": meta["family"],
        "priority": meta["priority"],
        "r201_evidence_level": evidence,
        "r201_num_evaluated": r201_row.get("num_evaluated") if r201_row else "",
        "mask_count": count,
        "legacy_num_evaluated": metrics.get("num_evaluated") if metrics else "",
        "dice": (r201_row or {}).get("dice") or legacy_mean.get("dice", ""),
        "iou": (r201_row or {}).get("iou") or legacy_mean.get("iou", ""),
        "boundary_iou": (r201_row or {}).get("boundary_iou") or legacy_mean.get("boundary_iou", ""),
        "gap_region_fp_rate": (r201_row or {}).get("gap_region_fp_rate", ""),
        "component_merge_rate": (r201_row or {}).get("component_merge_rate", ""),
        "component_count_mae": (r201_row or {}).get("component_count_mae", ""),
        "recommendation": recommendation,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    analysis_dir = Path("outputs/analysis")
    variants_root = Path("outputs/ablations_variants")
    r201_table = analysis_dir / "r201_unified_eval" / "r201_unified_comparison_table.csv"
    r201 = read_r201_table(r201_table)
    rows = [row_for_candidate(slug, meta, r201, variants_root, analysis_dir) for slug, meta in CANDIDATES.items()]

    summary = {
        "run_id": "R207",
        "purpose": "baseline coverage audit for R201 readiness",
        "r201_table": str(r201_table),
        "counts_by_recommendation": {
            rec: sum(1 for row in rows if row["recommendation"] == rec)
            for rec in sorted({row["recommendation"] for row in rows})
        },
        "rows": rows,
        "next_priority": [
            "Do not train more anatomy-aware pixel-deletion refiners after R205-F0 no-go.",
            "If Swin/U-Net comparison is required, first try to sync/regenerate masks for R106/R097; otherwise keep them as legacy weak baselines.",
            "R143, ARAA, R110, and R202 already have full R201 mask-derived metrics.",
        ],
    }
    out_json = analysis_dir / "r207_baseline_coverage_audit.json"
    out_csv = analysis_dir / "r207_baseline_coverage_audit.csv"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(out_csv, rows)
    print(json.dumps({"output_json": str(out_json), "counts": summary["counts_by_recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
