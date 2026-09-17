# R211-F2 Full-Val Hard-Case Visual Audit

Date: 2026-07-07

## Purpose

Audit whether R211-F2's GT-free component-guard acceptance gate reduces bone-gap adhesion on original validation cases without obvious over-erosion, missed epiphysis regions, or topology fragmentation.

This audit uses original `TSRS_RSNA-Epiphysis/val` only. It does not use `clean-test-v2` for model selection or threshold tuning.

## Inputs

- Full-val summary: `outputs/analysis/r211_f2_remote_fullval_val_summary.json`
- Full-val per-image metrics: `outputs/analysis/r211_f2_remote_fullval_val_per_image.csv`
- Visual candidate list: `outputs/analysis/r211_f2_remote_fullval_visual_candidates.csv`
- R110 anchor masks: `outputs/ablations_variants/r110_r100_r108_patch_basic_trainval/TSRS_RSNA-Epiphysis/val/masks`
- R211-F2 masks: `outputs/ablations_variants/r211_f2_component_guard_remote_fullval/TSRS_RSNA-Epiphysis/val/masks`

## Generated Artifacts

- Script: `scripts/build_r211_f2_val_hardcase_visualizations.py`
- Panel directory: `outputs/analysis/r211_f2_val_hardcase_visualizations/panels`
- HTML index: `outputs/analysis/r211_f2_val_hardcase_visualizations/index.html`
- Machine index: `outputs/analysis/r211_f2_val_hardcase_visualizations/r211_f2_val_hardcase_visualization_index.json`
- CSV index: `outputs/analysis/r211_f2_val_hardcase_visualizations/r211_f2_val_hardcase_visualization_index.csv`

The package contains `48` ranked panel rows covering `28` unique validation images.

## Full-Val Numeric Context

Best threshold: `0.6`.

Mean deltas vs R110 validation anchor:

- Dice: `-0.000030`
- IoU: `-0.000055`
- Recall: `-0.000178`
- Boundary IoU: `-0.000070`
- Boundary F1: `-0.000099`
- Surface Dice 2px: `-0.000198`
- Surface Dice 5px: `+0.000098`
- HD95 px: `-0.006130`
- ASSD px: `-0.000467`
- gap-region FP rate: `-0.000292`
- component merge rate: `0.000000`
- component count MAE: `-0.010417`
- predicted component count: `0.000000`

## Visual Candidate Group Summary

Mean deltas within the 12-row visual groups:

| Group | Dice | Recall | Boundary IoU | Gap FP | Component MAE |
|---|---:|---:|---:|---:|---:|
| safe_gap_gain | `+0.000056` | `-0.000629` | `+0.000543` | `-0.001923` | `+0.083333` |
| boundary_gain | `+0.000193` | `-0.000327` | `+0.001271` | `-0.001733` | `0.000000` |
| component_gain | `-0.000208` | `-0.000520` | `-0.000591` | `-0.000383` | `-0.166667` |
| risk_check | `-0.000336` | `-0.000761` | `-0.001408` | `-0.000439` | `+0.083333` |

## Visual Spot Check

Positive examples:

- `3840.png`: credible local gap cleanup in the wrist/carpal region. Dice, gap FP, and Boundary IoU all improve; no obvious large missed epiphysis was visible.
- `2248.png`: similar credible local gap cleanup, with one of the strongest Boundary IoU gains in the visual list.
- `6698.png`: small but clean boundary/gap improvement; edit is localized.

Mixed or risky examples:

- `1828.png`: gap FP improves slightly, but component-count MAE increases and Boundary IoU drops. This is not an acceptable template for the next method.
- `15040.png`: component-count MAE improves, but Boundary IoU and Recall drop. This illustrates that component improvement alone is not sufficient.
- `6605.png`: gap FP improves, but Boundary IoU drops visibly enough to classify this as a risk case rather than a clean success.
- `3392.png`: component improvement exists, but boundary cost remains negative.

## Interpretation

R211-F2 is directionally useful because the component guard prevents the severe fragmentation seen in earlier gap-deletion and learned-acceptance attempts. It produces real positive cases where small, localized deletions reduce adhesion without obvious over-erosion.

However, the full-val mean effect is too small, and Boundary IoU/F1 are slightly negative. The visual audit shows why: the gate still accepts some candidates with minor gap/component benefit but boundary or recall cost. The strongest reusable signal is the `boundary_gain` group, not the generic `safe_gap_gain` or `component_gain` group.

## Decision

Verdict: `conditional_pass_for_direction_only`.

- Do not apply R211-F2 to `clean-test-v2` as a locked final method.
- Keep the GT-free component-count guard as a required safety mechanism.
- Use R211-F2 as evidence that deployable topology-safe edits are possible, but not yet strong enough.
- R212 should train or select candidates with explicit boundary-positive / surface-positive acceptance pressure:
  - favor candidates like `2248.png`, `3840.png`, `6698.png`, and `15067.png`;
  - reject candidates like `1828.png`, `6605.png`, `15040.png`, and `3392.png` unless boundary/recall loss is eliminated;
  - preserve the constraints `max_pred_component_increase=0` and `max_step_component_increase=0`;
  - require validation mean Boundary IoU/F1 and Surface Dice 2px to be non-negative before any clean-test action.
