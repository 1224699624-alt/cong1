#!/usr/bin/env python3
"""Export manuscript-facing R320 result and ablation tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "outputs" / "analysis"
LOGS = ROOT / "research-workflow" / "refine-logs"

PROPORTION_METRICS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_iou",
    "boundary_f1",
    "surface_dice_2px",
    "surface_dice_5px",
    "gap_region_fp_rate",
    "component_merge_rate",
]
PIXEL_METRICS = ["hd95_px", "assd_px", "component_count_mae"]
ALL_METRICS = PROPORTION_METRICS + PIXEL_METRICS


def load(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(relative: str) -> dict:
    payload = load(relative)
    result: dict[str, float | int | str] = {
        "source_json": relative,
        "n_cases": int(payload["num_images"]),
    }
    per_image = payload["per_image"]
    for metric in ALL_METRICS:
        values = np.asarray([float(case[metric]) for case in per_image], dtype=np.float64)
        scale = 100.0 if metric in PROPORTION_METRICS else 1.0
        result[f"{metric}_mean"] = float(values.mean() * scale)
        result[f"{metric}_sd"] = float(values.std(ddof=1) * scale)
    return result


def add_deltas(row: dict, reference: dict | None) -> dict:
    for metric in ALL_METRICS:
        row[f"delta_{metric}"] = (
            "" if reference is None else row[f"{metric}_mean"] - reference[f"{metric}_mean"]
        )
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pm(row: dict, metric: str, digits: int = 2) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_sd']:.{digits}f}"


def latex_pm(row: dict, metric: str, digits: int = 2) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} $\\pm$ {row[f'{metric}_sd']:.{digits}f}"


def build_main() -> list[dict]:
    definitions = [
        (
            "U-Net",
            "Baseline",
            "R320_UNet_plain_seed3201",
            0.0,
            "exact paired seed 3201",
            "outputs/analysis/r320_unet_seed3201_plain_original_val_r201.json",
        ),
        (
            "U-Net",
            "+ Ours",
            "R320_UNet_prior_seed3201",
            0.035,
            "exact paired seed 3201",
            "outputs/analysis/r320_unet_seed3201_prior_original_val_r201.json",
        ),
        (
            "nnU-Net v2 2D",
            "Baseline",
            "R260_mature_baseline",
            "",
            "mature historical paired chain",
            "outputs/analysis/r260_mature_baseline_original_val_r201.json",
        ),
        (
            "nnU-Net v2 2D",
            "+ Ours (R317)",
            "R317_image_centers_only_ema_best",
            0.035,
            "mature historical paired chain",
            "outputs/analysis/r317_image_centers_only_ema_best_original_val_r201.json",
        ),
    ]
    rows: list[dict] = []
    references: dict[str, dict] = {}
    for backbone, method, experiment, alpha, pairing, source in definitions:
        row = {
            "dataset": "TSRS_RSNA-Epiphysis",
            "split": "original-val",
            "protocol": "R201",
            "backbone": backbone,
            "method": method,
            "experiment": experiment,
            "prior_alpha": alpha,
            "seed_count": 1,
            "uncertainty_unit": "case-wise SD",
            "pairing_status": pairing,
            **summarize(source),
        }
        reference = references.get(backbone)
        add_deltas(row, reference)
        if method == "Baseline":
            references[backbone] = row
        rows.append(row)
    return rows


def build_alpha() -> list[dict]:
    summary = load("outputs/analysis/r316c_dual_route_summary.json")
    selected = summary["selected"]
    reference = summarize("outputs/analysis/r260_mature_prior_original_val_r201.json")
    rows: list[dict] = []
    for candidate in summary["candidates"]:
        route = candidate["route"]
        alpha = float(candidate["alpha"])
        row = {
            "dataset": "TSRS_RSNA-Epiphysis",
            "split": "original-val",
            "protocol": "R201",
            "backbone": "nnU-Net v2 2D",
            "route": route,
            "prior_alpha": alpha,
            "selected": route == selected["route"] and alpha == float(selected["alpha"]),
            "hard_gate_pass": bool(candidate["pass"]),
            "seed_count": 1,
            "uncertainty_unit": "case-wise SD",
            **summarize(candidate["result"]),
        }
        add_deltas(row, reference)
        rows.append(row)
    return rows


def build_checkpoints() -> list[dict]:
    selection = load("outputs/analysis/r317_image_centers_only_checkpoint_selection.json")
    selected = selection["selected"]["label"]
    reference = summarize("outputs/analysis/r260_mature_prior_original_val_r201.json")
    rows: list[dict] = []
    for candidate in selection["candidates"]:
        row = {
            "dataset": "TSRS_RSNA-Epiphysis",
            "split": "original-val",
            "protocol": "R201",
            "backbone": "nnU-Net v2 2D",
            "checkpoint": candidate["label"],
            "prior_alpha": 0.035,
            "selected": candidate["label"] == selected,
            "hard_gate_pass": bool(candidate["pass"]),
            "seed_count": 1,
            "uncertainty_unit": "case-wise SD",
            **summarize(candidate["metrics"]),
        }
        add_deltas(row, reference)
        rows.append(row)
    return rows


def markdown_main(rows: list[dict]) -> str:
    lines = [
        "| Backbone | Method | Dice (%) | IoU (%) | B-IoU (%) | B-F1 (%) | SDice@2 (%) | SDice@5 (%) | HD95 (px) | ASSD (px) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backbone} | {method} | {dice} | {iou} | {biou} | {bf1} | {sd2} | {sd5} | {hd95} | {assd} |".format(
                backbone=row["backbone"],
                method=row["method"],
                dice=pm(row, "dice"),
                iou=pm(row, "iou"),
                biou=pm(row, "boundary_iou"),
                bf1=pm(row, "boundary_f1"),
                sd2=pm(row, "surface_dice_2px"),
                sd5=pm(row, "surface_dice_5px"),
                hd95=pm(row, "hd95_px"),
                assd=pm(row, "assd_px"),
            )
        )
    return "\n".join(lines)


def markdown_alpha(rows: list[dict]) -> str:
    lines = [
        "| Route | Alpha | Dice (%) | IoU (%) | HD95 (px) | ASSD (px) | Gap FP (%) | Merge rate (%) | Gate | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['route']} | {row['prior_alpha']:.3f} | {pm(row, 'dice')} | {pm(row, 'iou')} | "
            f"{pm(row, 'hd95_px')} | {pm(row, 'assd_px')} | {pm(row, 'gap_region_fp_rate')} | "
            f"{pm(row, 'component_merge_rate')} | {'Yes' if row['hard_gate_pass'] else 'No'} | "
            f"{'Yes' if row['selected'] else 'No'} |"
        )
    return "\n".join(lines)


def markdown_structure(rows: list[dict]) -> str:
    lines = [
        "| Backbone | Method | Precision (%) | Recall (%) | Gap FP (%) | Merge rate (%) | Component-count MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['method']} | {pm(row, 'precision')} | {pm(row, 'recall')} | "
            f"{pm(row, 'gap_region_fp_rate')} | {pm(row, 'component_merge_rate')} | "
            f"{pm(row, 'component_count_mae')} |"
        )
    return "\n".join(lines)


def build_latex(main_rows: list[dict], alpha_rows: list[dict]) -> str:
    lines = [
        "% Auto-generated by scripts/export_r320_paper_tables.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Segmentation results on the 96-case TSRS original validation set under the R201 protocol. Values are case-wise mean $\\pm$ standard deviation. Higher is better except HD95 and ASSD.}",
        "\\label{tab:r320_main}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "Backbone & Method & Dice (\\%) $\\uparrow$ & IoU (\\%) $\\uparrow$ & B-IoU (\\%) $\\uparrow$ & B-F1 (\\%) $\\uparrow$ & SD@2 (\\%) $\\uparrow$ & SD@5 (\\%) $\\uparrow$ & HD95 (px) $\\downarrow$ & ASSD (px) $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in main_rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["backbone"].replace("_", "\\_"),
                row["method"].replace("+", "$+$"),
                latex_pm(row, "dice"),
                latex_pm(row, "iou"),
                latex_pm(row, "boundary_iou"),
                latex_pm(row, "boundary_f1"),
                latex_pm(row, "surface_dice_2px"),
                latex_pm(row, "surface_dice_5px"),
                latex_pm(row, "hd95_px"),
                latex_pm(row, "assd_px"),
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Task-specific separation diagnostics on TSRS. Lower is better for gap-region false positive rate, component merge rate, and component-count MAE.}",
        "\\label{tab:r320_structure}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Backbone & Method & Precision (\\%) $\\uparrow$ & Recall (\\%) $\\uparrow$ & Gap FP (\\%) $\\downarrow$ & Merge (\\%) $\\downarrow$ & Count MAE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in main_rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                row["backbone"].replace("_", "\\_"),
                row["method"].replace("+", "$+$"),
                latex_pm(row, "precision"),
                latex_pm(row, "recall"),
                latex_pm(row, "gap_region_fp_rate"),
                latex_pm(row, "component_merge_rate"),
                latex_pm(row, "component_count_mae"),
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation of the prior-loss coefficient on nnU-Net.}",
        "\\label{tab:r320_alpha}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Route & $\\alpha$ & Dice (\\%) & IoU (\\%) & HD95 (px) & ASSD (px) & Pass \\\\",
        "\\midrule",
    ]
    for row in alpha_rows:
        lines.append(
            f"{row['route']} & {row['prior_alpha']:.3f} & {row['dice_mean']:.2f} & {row['iou_mean']:.2f} & "
            f"{row['hd95_px_mean']:.2f} & {row['assd_px_mean']:.2f} & "
            f"{'Yes' if row['hard_gate_pass'] else 'No'} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    main_rows = build_main()
    alpha_rows = build_alpha()
    checkpoint_rows = build_checkpoints()

    write_csv(ANALYSIS / "r320_paper_main_results.csv", main_rows)
    write_csv(ANALYSIS / "r320_paper_alpha_ablation.csv", alpha_rows)
    write_csv(ANALYSIS / "r320_paper_checkpoint_ablation.csv", checkpoint_rows)

    report = f"""# R320 Paper-Ready Experiment Tables

## Evaluation protocol

- Dataset: TSRS_RSNA-Epiphysis only.
- Split: 96-case original validation set.
- Evaluation: R201; clean-test-v2 was not used.
- Values: case-wise mean +/- sample standard deviation.
- Evidence boundary: current backbone results use one training seed. The SD is across cases, not across seeds.

## Main comparison

{markdown_main(main_rows)}

## Task-specific separation diagnostics

{markdown_structure(main_rows)}

Gap-region FP, merge rate, and component-count MAE are task-specific diagnostic metrics and should be presented separately from standard segmentation metrics.

The U-Net comparison is an exact paired run with shared initialization. The nnU-Net comparison comes from the mature R260/R317 experiment chain under the same R201 original-val protocol; it should not be described as a newly repeated multi-seed R320 pair.

## Alpha ablation

{markdown_alpha(alpha_rows)}

The continuous route with alpha=0.020 has the highest Dice/IoU, whereas alpha=0.035 has slightly lower HD95/ASSD and was selected by the predeclared distance-focused gate. These are single-seed development results.

## Reporting constraints

- Use foreground IoU, not mIoU, in the manuscript unless a separately defined class-mean IoU is computed.
- Do not report the case-wise SD as a seed-wise SD.
- R260's 0.05 coefficient is a different background-separation weight and is not part of the R316c alpha sweep.
- R317 epoch rows are checkpoint-maturity diagnostics and belong in supplementary material, not the primary alpha-ablation table.

## Artifacts

- `outputs/analysis/r320_paper_main_results.csv`
- `outputs/analysis/r320_paper_alpha_ablation.csv`
- `outputs/analysis/r320_paper_checkpoint_ablation.csv`
- `outputs/analysis/r320_paper_tables.tex`
- `scripts/export_r320_paper_tables.py`
"""
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "R320_PAPER_READY_EXPERIMENT_TABLES.md").write_text(report, encoding="utf-8")
    (ANALYSIS / "r320_paper_tables.tex").write_text(build_latex(main_rows, alpha_rows), encoding="utf-8")
    print("paper-ready CSV, Markdown, and LaTeX tables written")


if __name__ == "__main__":
    main()
