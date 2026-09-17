#!/usr/bin/env python3
"""Strict paired analysis for the R254 original-val failure mode."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


DIRECTIONS = {
    "dice": 1,
    "iou": 1,
    "precision": 1,
    "recall": 1,
    "boundary_iou": 1,
    "boundary_f1": 1,
    "surface_dice_2px": 1,
    "surface_dice_5px": 1,
    "hd95_px": -1,
    "assd_px": -1,
    "gap_region_fp_rate": -1,
    "component_merge_rate": -1,
    "component_count_mae": -1,
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, repeats: int = 20000) -> tuple[float, float]:
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(1)
    return tuple(float(x) for x in np.percentile(samples, [2.5, 97.5]))


def rank_biserial(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="outputs/analysis/r254_nnunet_baseline_original_val_r201.json")
    p.add_argument("--ours", default="outputs/analysis/r254_hapdsp_pcr_nnunet_original_val_r201.json")
    p.add_argument("--dynamics", default="outputs/nnunet/r254_hapdsp_pcr/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerHAPDSPPCR__nnUNetPlans__2d/fold_0/r254_training_dynamics.jsonl")
    p.add_argument("--output-dir", default="outputs/analysis/r254_failure_analysis")
    args = p.parse_args()

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    ours = json.loads(Path(args.ours).read_text(encoding="utf-8"))
    b = {r["image"]: r for r in baseline["per_image"]}
    o = {r["image"]: r for r in ours["per_image"]}
    names = sorted(set(b) & set(o))
    if len(names) != 96:
        raise RuntimeError(f"Expected 96 paired images, got {len(names)}")

    rng = np.random.default_rng(254)
    rows = []
    raw_p = []
    for metric, direction in DIRECTIONS.items():
        bv = np.asarray([b[n][metric] for n in names], dtype=float)
        ov = np.asarray([o[n][metric] for n in names], dtype=float)
        delta = ov - bv
        ci = bootstrap_ci(delta, rng)
        try:
            test = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
            statistic, pvalue = float(test.statistic), float(test.pvalue)
        except ValueError:
            statistic, pvalue = 0.0, 1.0
        raw_p.append(pvalue)
        rows.append(
            {
                "metric": metric,
                "direction": "higher" if direction > 0 else "lower",
                "n": len(delta),
                "baseline_mean": float(bv.mean()),
                "ours_mean": float(ov.mean()),
                "mean_delta_ours_minus_baseline": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "delta_ci95_low": ci[0],
                "delta_ci95_high": ci[1],
                "wilcoxon_statistic": statistic,
                "p_raw": pvalue,
                "rank_biserial_ours_minus_baseline": rank_biserial(delta),
                "improved_images": int((direction * delta > 0).sum()),
                "worsened_images": int((direction * delta < 0).sum()),
            }
        )
    for row, p_adj in zip(rows, holm(raw_p)):
        row["p_holm"] = p_adj

    output = Path(args.output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (output / "paired_statistics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = ["dice", "precision", "recall", "boundary_iou", "surface_dice_2px", "hd95_px", "assd_px", "gap_region_fp_rate", "component_merge_rate", "component_count_mae"]
    row_by_metric = {r["metric"]: r for r in rows}
    favorable_pct = []
    for metric in selected:
        r = row_by_metric[metric]
        direction = DIRECTIONS[metric]
        favorable_pct.append(direction * (r["ours_mean"] - r["baseline_mean"]) / (abs(r["baseline_mean"]) + 1e-12) * 100)
    colors = ["#009E73" if x > 0 else "#D55E00" for x in favorable_pct]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(np.arange(len(selected)), favorable_pct, color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(np.arange(len(selected)), [x.replace("_", " ") for x in selected])
    ax.set_xlabel("Directional relative change (%) — positive is better")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "figure-01-directional-metric-change.pdf")
    fig.savefig(figures / "figure-01-directional-metric-change.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, label in zip(axes, ["gap_region_fp_rate", "dice"], ["Gap-region FP rate", "Dice"]):
        bv = np.asarray([b[n][metric] for n in names])
        ov = np.asarray([o[n][metric] for n in names])
        lo, hi = min(bv.min(), ov.min()), max(bv.max(), ov.max())
        ax.scatter(bv, ov, s=20, alpha=0.65, color="#0072B2")
        ax.plot([lo, hi], [lo, hi], "--", color="black", lw=1)
        ax.set_xlabel("Native nnU-Net")
        ax.set_ylabel("HA-PDSP + PCR")
        ax.set_title(label)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "figure-02-paired-gapfp-dice.pdf")
    fig.savefig(figures / "figure-02-paired-gapfp-dice.png", dpi=300)
    plt.close(fig)

    dynamics = [json.loads(line) for line in Path(args.dynamics).read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [r["epoch"] for r in dynamics]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, [r["anchor_probability"] for r in dynamics], marker="o", label="Selected negative p(foreground)", color="#D55E00")
    ax.plot(epochs, [r["core_probability"] for r in dynamics], marker="s", label="Bone-core p(foreground)", color="#0072B2")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "figure-03-training-confidence-dynamics.pdf")
    fig.savefig(figures / "figure-03-training-confidence-dynamics.png", dpi=300)
    plt.close(fig)

    (output / "analysis-summary.json").write_text(json.dumps({"n": len(names), "seed_count": 1, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"n": len(names), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
