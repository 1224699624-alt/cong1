# R210-F1 Hard-Case Visual Audit

Date: 2026-07-07

## Purpose

Audit whether R210-F1 visually reduces bone-gap adhesion on original validation hard cases without obvious over-erosion, missed epiphyses, or anatomy fragmentation.

This is still train/val-only evidence. `clean-test-v2` was not used for this audit.

## Inputs

- Summary metrics: `outputs/analysis/r210_f1_neck_candidate_gate_val_fastinner_summary.json`
- Per-image metrics: `outputs/analysis/r210_f1_neck_candidate_gate_val_fastinner_per_image.csv`
- Hard-case manifest: `outputs/analysis/r210_f1_hardcase_visualization_manifest.csv`
- Visual panels: `outputs/analysis/r210_f1_val_hardcase_visualizations/index.html`

R210-F1 fixed settings:

- `dist_percentile=10`
- `max_candidates_generated=16`
- `max_components_per_image=5`

## Full-Val Metric Context

R210-F1 passed the original-val quantitative gate but with a mixed metric profile:

| Metric | Delta vs R110 val anchor | Direction |
|---|---:|---|
| Dice | -0.000179 | small loss |
| IoU | -0.000302 | small loss |
| Precision | +0.000661 | improved |
| Recall | -0.000993 | small loss |
| Boundary IoU | -0.000209 | small loss |
| Boundary F1 | -0.000272 | small loss |
| Surface Dice 2px | -0.000233 | small loss |
| Surface Dice 5px | +0.000230 | improved |
| HD95 px | -0.001324 | improved |
| ASSD px | -0.001280 | improved |
| gap-region FP rate | -0.001623 | improved |
| component merge rate | -0.156250 | improved |
| component count MAE | -0.177083 | improved |

Interpretation: R210-F1 is an anatomy-consistency improvement candidate, not yet a clean boundary-quality improvement candidate.

## Positive Cases

### `3840.png`

Strong positive case. The accepted cuts are localized in wrist-bone adhesion regions. Component count MAE improves by `-5`, gap FP improves by `-0.010135`, Dice improves by `+0.001191`, and Boundary IoU improves by `+0.006945`. Visual overlay supports real separation improvement rather than broad erosion.

### `11852.png`

Positive case. The cut removes wrist-region adhesion-like false positives. Dice improves by `+0.000426`, gap FP improves by `-0.008860`, Boundary IoU improves by `+0.004884`, and Surface Dice 2px improves by `+0.006513`. No obvious large missed epiphysis is visible.

### `3354.png`

Positive case. The method reduces a merge error and improves boundary metrics: component merge rate improves by `-1`, component count MAE by `-1`, gap FP by `-0.006400`, Dice by `+0.000579`, and Boundary IoU by `+0.004665`. The visible cuts are local and anatomically plausible.

### `6606.png`

Positive case. Component merge improves by `-1`, gap FP by `-0.007169`, Dice by `+0.000823`, Surface Dice 2px by `+0.005837`, and Surface Dice 5px by `+0.004625`. The overlay shows local adhesion cleanup without obvious whole-bone erosion.

### `3700.png`

Mild positive case. Component count MAE improves by `-2`, gap FP by `-0.004118`, Dice by `+0.000242`, and Boundary IoU by `+0.000396`. Some wrist separation improves, but residual merges remain, so this is evidence of partial correction rather than full repair.

### `15040.png`

Mixed positive case. Component count MAE improves by `-2` and component merge rate by `-1`, while gap FP improves by `-0.003355`. However Dice drops by `-0.000483`, Recall by `-0.002191`, Boundary IoU by `-0.001441`, and Surface Dice 2px by `-0.004613`. Visual effect is localized, but the case should be treated as a trade-off case, not a clean win.

## Risk Cases

### `1560.png`

Highest risk among reviewed cases. Component merge improves by `-1` and component count MAE by `-2`, but Dice drops by `-0.001898`, Recall by `-0.003706`, Boundary IoU by `-0.003148`, Surface Dice 2px by `-0.003024`, and HD95 worsens by `+0.400268 px`. The overlay does not show catastrophic fragmentation, but it shows the current gate is too willing to accept recall-costly cuts.

### `13480.png`

Risk case. gap FP improves only by `-0.000215`, while Dice drops by `-0.001668`, Recall by `-0.003106`, Boundary IoU by `-0.004201`, Surface Dice 2px by `-0.005780`, and HD95 worsens by `+0.385165 px`. The visual change is subtle; the quantitative trade-off is not attractive.

### `3366.png`

Risk case. gap FP improves by only `-0.000323`, but Dice drops by `-0.001563`, Recall by `-0.002816`, Boundary IoU by `-0.001572`, and Surface Dice 2px by `-0.004247`. There is no component-count gain. This case should be rejected by a stricter acceptance policy.

### `3567.png`

Risk case. gap FP improves by `-0.001457`, but Dice drops by `-0.001369`, Recall by `-0.003132`, Boundary IoU by `-0.006761`, and Surface Dice 2px by `-0.003445`. The visual panel does not suggest a large anatomical rescue, so the current gate over-accepts this kind of small-gain edit.

### `3777.png`

Risk case. gap FP improves by only `-0.000209`, while Dice drops by `-0.001347`, Recall by `-0.002547`, Boundary IoU by `-0.002871`, and Surface Dice 2px by `-0.007553`. This should be rejected unless a future gate predicts a clearer anatomy-consistency benefit.

### `1884.png`

Risk case. gap FP improves by `-0.001599` and ASSD/HD95 improve slightly, but Dice drops by `-0.001270`, Recall by `-0.002577`, Boundary IoU by `-0.000695`, and Surface Dice 2px by `-0.001184`. Visual evidence is low-resolution/subtle, so it is not strong enough to justify accepting recall loss.

## Verdict

Visual audit result: `conditional_pass_for_train_val_feasibility`.

R210-F1 shows credible evidence that component-preserving neck candidates can reduce bone-gap adhesion and adjacent-bone merge errors without the catastrophic fragmentation seen in R177/R205-F0. The best positive cases are suitable as internal development examples for the project direction.

However, R210-F1 should not be promoted as a final clean-test model in its current form:

- Mean Boundary IoU, Boundary F1, and Surface Dice 2px are slightly worse on full val.
- Several risk cases accept very small gap improvements while paying recall/boundary cost.
- The current acceptance gate uses GT-derived metric checks on validation. Applying this same accept/reject policy to `clean-test-v2` would be GT leakage if claimed as a deployable model output.

## Decision

Do not apply the current R210-F1 GT-gated output selection to `clean-test-v2` as a final method.

Proceed to R211: a GT-free acceptance gate locked on original train/val. The next version should learn or define an inference-available reject policy that blocks low-benefit recall-costly edits like `13480.png`, `3366.png`, `3567.png`, `3777.png`, and `1884.png`, while preserving strong positive edits like `3840.png`, `11852.png`, `3354.png`, and `6606.png`.

## R211 Requirements

- Train/fit acceptance only on original train/val.
- Use only inference-available features:
  - candidate cut area and shape,
  - neck width / distance-transform statistics,
  - local image-gradient support,
  - anchor component geometry,
  - predicted component split count,
  - candidate proximity to narrow high-risk gap regions.
- Do not use GT or GT-derived metrics at inference.
- Reject edits with weak predicted gap/component benefit and high predicted recall/boundary risk.
- Re-run full original-val R201 metrics before any clean-test-v2 application.
- Keep `clean-test-v2` locked until the GT-free full-val gate passes.
