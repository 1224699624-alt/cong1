# R223 Structured Seam Background Result

Date: 2026-07-08

## Purpose

Continue the inverted topology route after R222 failed as a pixel-wise MLP.
R223 changes the unit of prediction from independent pixels to coherent
seam/background proposal components, while keeping the same project target:
reduce bone-seam adhesion, false neighboring-bone connections, and boundary
errors without over-erosion or missed epiphysis foreground.

This is original-val development evidence only. It does not use clean-test-v2
and writes no final masks.

## Implemented

Added:

- `scripts/audit_r223_structured_seam_background_proposals.py`

The script:

- reuses R210 neck candidates;
- applies R216 soft seam actions;
- adds R215/R216 background-channel features;
- adds proposal geometry, distance, gradient, local image, and ring features;
- labels candidates with train/val GT diagnostics only;
- evaluates an image-grouped CV useful/risk selector;
- writes JSON/CSV only.

Synced remote artifacts:

- `outputs/analysis/r223_structured_seam_fullval_candidates.csv`
- `outputs/analysis/r223_structured_seam_fullval_summary.json`
- `outputs/analysis/r223_structured_seam_fullval_groupedcv.csv`
- `outputs/logs/r223_structured_seam_fullval.log`

Local follow-up policy-search artifacts:

- `outputs/analysis/r223_structured_seam_fullval_policy_search.csv`
- `outputs/analysis/r223_structured_seam_fullval_policy_search.json`

## Full-Val Candidate Diagnostic

Settings:

- split: original `TSRS_RSNA-Epiphysis/val`
- clean-test-v2 used: `false`
- dist percentile: `10`
- action fracs: `0.25,0.50`
- max candidates generated: `6`
- max components per image: `3`
- classifier: HGB
- grouped CV folds: `5`

Coverage:

- candidate rows: `530`
- images with candidates: `92`

Overall candidate statistics:

- quick-useful: `216/530`
- hard-risk: `284/530`
- safe-channel proxy positive: `194/530`
- over-erosion proxy: `286/530`
- mean Dice delta: `-0.000027`
- mean Recall delta: `-0.000168`
- mean Boundary IoU delta: `+0.000073`
- mean Boundary F1 delta: `+0.000106`
- mean gap-region FP delta: `-0.000331`
- mean component count MAE delta: `+0.056604`

## Oracle / Diagnostic Upper Bound

The safe-channel proxy group remains strong:

- rows: `194`
- images: `68`
- quick-useful: `168`
- hard-risk: `17`
- over-erosion proxy: `0`
- mean Dice delta: `+0.000101`
- mean Recall delta: `-0.000049`
- mean Boundary IoU delta: `+0.000675`
- mean Boundary F1 delta: `+0.000899`
- mean gap-region FP delta: `-0.000672`
- mean component count MAE delta: `+0.041237`
- mean cut GT foreground fraction: `0.126350`
- mean cut GT gap fraction: `0.855330`

The oracle best-per-image group is stronger:

- rows/images: `70`
- quick-useful: `70`
- hard-risk: `0`
- mean Dice delta: `+0.000135`
- mean Recall delta: `-0.000058`
- mean Boundary IoU delta: `+0.001017`
- mean Boundary F1 delta: `+0.001374`
- mean gap-region FP delta: `-0.000908`
- mean component count MAE delta: `-0.057143`

Interpretation: R223 proposal generation contains real positive signal. The
background-channel hypothesis is still valid when GT can identify the right
proposal.

## Deployable Selector Result

The HGB image-grouped CV selector failed:

- probed rows: `530`
- folds: `5`
- best selected policy from CV grid:
  - accepted: `2`
  - quick-useful: `0`
  - hard-risk: `2`
  - mean Boundary IoU delta: `-0.000501`
  - mean gap-region FP delta: `0.000000`
  - mean cut GT foreground fraction: `1.000000`

Local GT-free rule search also failed:

- risk `<= 0`: at most `1` useful accepted candidate;
- risk `<= 2`: still only `1` useful and `1` risk;
- risk `<= 20`: `11` useful but `12` risk, too unsafe;
- broad rule frontiers retain high over-erosion risk.

## Decision

`R223 = no_go_for_mask_level`.

Do not:

- write R223 masks as a model version;
- apply R223 to clean-test-v2;
- continue HGB/LogReg/threshold sweeps on the same candidate table;
- claim R201 anatomical-consistency improvement from R223.

## Interpretation

R223 improved the formulation compared with R222:

- R222 could not generate spatially meaningful edits.
- R223 generates coherent proposal components and has a strong oracle upper
  bound.

But the deployable bottleneck remains candidate selection. Useful and risky
proposal components are still too entangled under current GT-free features.
This repeats the R216-R218 pattern: candidate/action spaces contain the desired
signal, but selection does not generalize safely.

## Next Route

Do not run another shallow selector on the same table.

Recommended R224 direction:

1. Improve candidate generation purity before selection:
   - generate proposals only between two plausible neighboring bone instances;
   - add instance-level context from predicted component geometry;
   - avoid proposals that lie inside thick single-bone interiors.
2. Move from local cut classification to pairwise instance-seam modeling:
   - candidate = seam between two adjacent predicted/anchor components;
   - features include both-side bone shape, gap width, image valley strength,
     and whether suppressing the seam separates a merged component.
3. Keep the first R224 experiment diagnostic-only:
   - JSON/CSV first;
   - no clean-test-v2;
   - no mask writing unless image-grouped validation has a meaningful
     low-risk frontier.

