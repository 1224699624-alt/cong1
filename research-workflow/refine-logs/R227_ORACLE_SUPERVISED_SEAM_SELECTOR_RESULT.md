# R227 Oracle-Supervised Seam Selector Result

## ARIS Status
- Stage: train/val candidate-level diagnostic
- Dataset: `TSRS_RSNA-Epiphysis`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Clean-test-v2 used: no
- Writes masks: no
- Decision: `R227 = weak_go_for_next_localizer; no mask-level promotion yet`

## Purpose
R224 proved that pairwise inter-instance seam candidates are highly pure and
useful, but R224 uses GT instances and is not deployable. R226 generated
deployable background-connectivity candidates, but its self-CV selector accepted
no candidates. R227 tests whether R224 can supervise a selector for R226
GT-free candidates using only deployable morphology/image features.

## Implemented Artifact
- Script: `scripts/audit_r227_oracle_supervised_seam_selector.py`

The script:
- trains useful/risk scorers from R224 oracle candidates;
- optionally adds R226 hard-risk/over-erosion candidates as calibration
  negatives;
- scores R226 GT-free candidates;
- searches useful/risk thresholds;
- writes JSON/CSV only.

It excludes GT-only and evaluation-only fields from features:
- no `gt_a`, `gt_b`, `cut_gt_*` as model features;
- no `delta_*` or `candidate_*` as model features;
- no clean-test-v2 use.

## Outputs
Uncalibrated:
- `outputs/analysis/r227_oracle_supervised_seam_selector_pilot32_summary.json`
- `outputs/analysis/r227_oracle_supervised_seam_selector_pilot32_scored.csv`
- `outputs/analysis/r227_oracle_supervised_seam_selector_pilot32_grid.csv`

Calibrated:
- `outputs/analysis/r227_oracle_supervised_seam_selector_calibrated_pilot32_summary.json`
- `outputs/analysis/r227_oracle_supervised_seam_selector_calibrated_pilot32_scored.csv`
- `outputs/analysis/r227_oracle_supervised_seam_selector_calibrated_pilot32_grid.csv`

Calibrated safe-gate:
- `outputs/analysis/r227_oracle_supervised_seam_selector_calibrated_safe_pilot32_summary.json`
- `outputs/analysis/r227_oracle_supervised_seam_selector_calibrated_safe_pilot32_scored.csv`
- `outputs/analysis/r227_oracle_supervised_seam_selector_calibrated_safe_pilot32_grid.csv`

## Results

### R227-A: R224-only transfer
Best grid:
- accepted: `9`
- quick useful: `4/9`
- hard-risk: `5/9`
- over-erosion proxy: `6/9`
- Boundary IoU delta: `-0.000139`
- gap-region FP delta: `-0.000206`

Interpretation: direct R224-to-R226 transfer is not reliable. It accepts too
many R226 geometry failures.

### R227-B: R224 positives + R226 calibration negatives
Best grid:
- accepted: `2`
- quick useful: `2/2`
- hard-risk: `0/2`
- safe-gap positive: `1/2`
- over-erosion proxy: `1/2`
- Dice delta: `+0.000017`
- IoU delta: `+0.000026`
- Recall delta: `-0.000172`
- Boundary IoU delta: `+0.000458`
- Boundary F1 delta: `+0.000585`
- gap-region FP delta: `-0.000560`
- component count MAE delta: `+0.000000`

Interpretation: calibration fixes the hard-risk problem and recovers the
intended boundary/gap-FP direction, but one accepted candidate still triggers
the over-erosion proxy.

### R227-C: calibrated safe-gate
Best grid:
- accepted: `1`
- quick useful: `1/1`
- hard-risk: `0/1`
- safe-gap positive: `1/1`
- over-erosion proxy: `0/1`
- Dice delta: `+0.000110`
- IoU delta: `+0.000171`
- Recall delta: `+0.000000`
- Boundary IoU delta: `+0.000046`
- Boundary F1 delta: `+0.000069`
- gap-region FP delta: `-0.000511`
- component count MAE delta: `+0.000000`
- cut GT foreground fraction: `0.000000`
- cut GT gap fraction: `1.000000`

Interpretation: strict safety gating can find a clean seam candidate, improving
gap FP without Dice/IoU/Recall regression or over-erosion. However, coverage is
only one accepted image in the pilot, so this is not yet a model-level result.

## Decision
R227 improves the selector story relative to R226:
- R226 grouped-CV accepted `0`.
- R227 calibrated safe-gate accepted `1` clean candidate.

But R227 is still candidate-level evidence only. Do not write masks or run R201
from this result. The main failure is coverage: the R226 candidate generator
does not expose enough clean seam candidates for the selector to matter.

## Next Step
Move to R228 with a localizer instead of a selector-only patch:

1. Build training samples from R224 oracle seams.
2. Use local crops around merged R110 components.
3. Predict a seam probability map from GT-free inputs:
   image crop, anchor crop, distance transform, boundary band, and local
   background/contact maps.
4. Convert the seam probability map into thin background-channel cuts.
5. Evaluate first on original val only.

Promotion gate for R228:
- nonzero mask-level edited images on original val;
- no material Dice/IoU/Recall regression;
- Boundary IoU/F1 improvement;
- gap FP reduction;
- component count MAE non-worsening;
- hard-case visualization confirms reduced seam adhesion without over-erosion.
