#!/usr/bin/env python3
"""Build one auditable CSV for existing U-Net/nnU-Net and alpha-screen results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"
OUTPUT = ANALYSIS / "r320_existing_unet_nnunet_alpha_results.csv"

METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "boundary_iou",
    "boundary_f1",
    "gap_region_fp_rate",
    "gap_region_precision",
    "component_merge_rate",
    "component_count_mae",
    "component_delta_mean",
    "surface_dice_2px",
    "surface_dice_5px",
    "hd95_px",
    "assd_px",
]


def load(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_mean(relative: str) -> dict:
    payload = load(relative)
    return payload["mean"]


def add_row(
    rows: list[dict],
    *,
    experiment: str,
    backbone: str,
    arm: str,
    route: str,
    coefficient_name: str,
    coefficient_value: float | None,
    alpha_screening: bool,
    checkpoint_label: str,
    selected: bool | None,
    gate_pass: bool | None,
    source_json: str,
    mean: dict,
    reference_experiment: str = "",
    reference_mean: dict | None = None,
    notes: str = "",
) -> None:
    row = {
        "dataset": "TSRS_RSNA-Epiphysis",
        "split": "original-val",
        "protocol": "R201",
        "num_images": 96,
        "clean_test_used": False,
        "experiment": experiment,
        "backbone": backbone,
        "arm": arm,
        "route": route,
        "coefficient_name": coefficient_name,
        "coefficient_value": "" if coefficient_value is None else coefficient_value,
        "alpha_screening": alpha_screening,
        "checkpoint_label": checkpoint_label,
        "selected": "" if selected is None else selected,
        "gate_pass": "" if gate_pass is None else gate_pass,
        "reference_experiment": reference_experiment,
        "source_json": source_json.replace("\\", "/"),
        "notes": notes,
    }
    for metric in METRICS:
        row[metric] = mean.get(metric, "")
        row[f"delta_{metric}"] = (
            ""
            if reference_mean is None or metric not in mean or metric not in reference_mean
            else mean[metric] - reference_mean[metric]
        )
    rows.append(row)


def main() -> None:
    rows: list[dict] = []

    r260_plain_file = "outputs/analysis/r260_mature_baseline_original_val_r201.json"
    r260_prior_file = "outputs/analysis/r260_mature_prior_original_val_r201.json"
    r260_plain = metric_mean(r260_plain_file)
    r260_prior = metric_mean(r260_prior_file)
    add_row(
        rows,
        experiment="R260_mature_baseline",
        backbone="nnU-Net v2 2D",
        arm="plain",
        route="native",
        coefficient_name="",
        coefficient_value=None,
        alpha_screening=False,
        checkpoint_label="mature_baseline",
        selected=None,
        gate_pass=None,
        source_json=r260_plain_file,
        mean=r260_plain,
        notes="Mature nnU-Net baseline; reference for R260 prior.",
    )
    add_row(
        rows,
        experiment="R260_mature_prior",
        backbone="nnU-Net v2 2D",
        arm="prior",
        route="R259/R260_background_separation",
        coefficient_name="background_separation_weight",
        coefficient_value=0.05,
        alpha_screening=False,
        checkpoint_label="mature_prior",
        selected=True,
        gate_pass=None,
        source_json=r260_prior_file,
        mean=r260_prior,
        reference_experiment="R260_mature_baseline",
        reference_mean=r260_plain,
        notes="The 0.05 coefficient is not the R316c/R317 continuous-prior alpha.",
    )

    r316_summary_file = "outputs/analysis/r316c_dual_route_summary.json"
    r316 = load(r316_summary_file)
    selected_r316 = r316.get("selected") or {}
    for candidate in r316["candidates"]:
        source = candidate["result"]
        route = candidate["route"]
        alpha = float(candidate["alpha"])
        add_row(
            rows,
            experiment=f"R316c_{route}_alpha{alpha:.3f}",
            backbone="nnU-Net v2 2D",
            arm="prior",
            route=route,
            coefficient_name="prior_alpha",
            coefficient_value=alpha,
            alpha_screening=True,
            checkpoint_label="best",
            selected=(
                selected_r316.get("route") == route
                and float(selected_r316.get("alpha", -1)) == alpha
            ),
            gate_pass=bool(candidate["pass"]),
            source_json=source,
            mean=candidate["mean"],
            reference_experiment="R260_mature_prior",
            reference_mean=r260_prior,
            notes="R316c dual-route alpha screening; selection used original-val only.",
        )

    r317_selection_file = "outputs/analysis/r317_image_centers_only_checkpoint_selection.json"
    r317 = load(r317_selection_file)
    selected_r317 = (r317.get("selected") or {}).get("label")
    for candidate in r317["candidates"]:
        add_row(
            rows,
            experiment=f"R317_image_centers_only_{candidate['label']}",
            backbone="nnU-Net v2 2D",
            arm="prior",
            route="continuous_image_centers_only",
            coefficient_name="prior_alpha",
            coefficient_value=0.035,
            alpha_screening=False,
            checkpoint_label=candidate["label"],
            selected=candidate["label"] == selected_r317,
            gate_pass=bool(candidate["pass"]),
            source_json=candidate["metrics"],
            mean=candidate["mean"],
            reference_experiment="R260_mature_prior",
            reference_mean=r260_prior,
            notes="Fixed alpha=0.035 checkpoint-maturity audit; not a new alpha value.",
        )

    unet_plain_file = "outputs/analysis/r320_unet_seed3201_plain_original_val_r201.json"
    unet_prior_file = "outputs/analysis/r320_unet_seed3201_prior_original_val_r201.json"
    unet_plain = metric_mean(unet_plain_file)
    unet_prior = metric_mean(unet_prior_file)
    add_row(
        rows,
        experiment="R320_UNet_plain_seed3201",
        backbone="U-Net",
        arm="plain",
        route="native",
        coefficient_name="prior_alpha",
        coefficient_value=0.0,
        alpha_screening=False,
        checkpoint_label="best_val_dice",
        selected=True,
        gate_pass=None,
        source_json=unet_plain_file,
        mean=unet_plain,
        notes="Exact paired initialization control arm.",
    )
    unet_gate = (
        unet_prior["dice"] > unet_plain["dice"]
        and unet_prior["iou"] > unet_plain["iou"]
        and unet_prior["hd95_px"] < unet_plain["hd95_px"]
        and unet_prior["assd_px"] < unet_plain["assd_px"]
    )
    add_row(
        rows,
        experiment="R320_UNet_prior_seed3201",
        backbone="U-Net",
        arm="prior",
        route="R317_continuous_image_centers_only",
        coefficient_name="prior_alpha",
        coefficient_value=0.035,
        alpha_screening=False,
        checkpoint_label="best_val_dice",
        selected=True,
        gate_pass=unet_gate,
        source_json=unet_prior_file,
        mean=unet_prior,
        reference_experiment="R320_UNet_plain_seed3201",
        reference_mean=unet_plain,
        notes="Exact paired U-Net plugin test with fixed cross-backbone alpha.",
    )

    fields = list(rows[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
