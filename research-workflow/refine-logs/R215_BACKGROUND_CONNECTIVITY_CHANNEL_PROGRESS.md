# R215 Background-Connectivity Channel Progress

Date: 2026-07-07

## Purpose

Start the inverted SATLoss-inspired route:

- do not encourage epiphysis foreground connectivity;
- instead, detect local background/gap channels whose opening corresponds to
  bone-seam separation.

This stage is candidate-level only. It writes no masks and does not use
clean-test-v2.

## Implemented

Added:

- `scripts/audit_r215_background_connectivity_candidates.py`

The script regenerates R210/R211-style candidate cuts from the R110 anchor on
original train/val and computes local background-connectivity features:

- local background component counts before/after candidate cut;
- cut-adjacent background fraction;
- background channel size after cut;
- border contacts of the cut-connected background channel;
- foreground component count change;
- train/val-only diagnostics such as cut GT foreground/gap fraction.

Outputs:

- candidate CSV;
- JSON summary;
- no masks.

## Local Smoke

Local 5-image validation smoke produced 10 candidate rows.

Early finding:

- some cuts reduce gap FP and improve boundary metrics;
- many cuts also remove GT foreground;
- therefore background-connectivity must be paired with over-erosion/risk
  rejection.

## Remote Runs

Initial full-val run was too slow and wrote only at completion, so it was
replaced with an incremental `--flush-every` run.

Current active remote run:

- screen: `r215_bg_connectivity_channel_val`
- log: `outputs/bridge_logs/r215_background_connectivity_channel_val.log`
- candidate CSV:
  `outputs/analysis/r215_background_connectivity_channel_val_candidates.csv`
- summary JSON:
  `outputs/analysis/r215_background_connectivity_channel_val_summary.json`

Settings:

- split: original `val`
- source image list: `r212_f5_evalonly_fullval_val_candidates.csv`
- max candidates generated: `8`
- max components per image: `3`
- flush every `5` images
- clean-test-v2: not used

## Incremental Evidence

At 40 validation images / 115 candidate rows:

- quick useful candidates: `34`
- hard risk candidates: `75`
- original component-merge background proxy: `0` positives
- safe-channel proxy positives: `35`

Safe-channel proxy diagnostic means:

- quick useful rate: `0.8571428571`
- hard risk rate: `0.0571428571`
- mean cut GT foreground fraction: `0.2075217041`
- mean cut GT gap fraction: `0.7924782959`
- mean Dice delta: `+0.0002046484`
- mean Boundary IoU delta: `+0.0016313265`
- mean gap-region FP delta: `-0.0017119642`

This supports the user's inversion idea: useful bone-seam separation appears
when candidate cuts open a local background channel and mostly lie in GT gap
regions.

## Important Negative Finding

Pure GT-free thresholding on background-channel features is not yet safe.

On the 25-image / 71-row partial table:

- pure background-channel one-candidate-per-image policy accepted `38` images;
- quick useful: `11`;
- hard risk: `24`;
- mean Boundary IoU delta was positive but Dice/IoU/Recall risk was too high.

Interpretation:

- background-channel geometry is a strong candidate signal;
- it cannot by itself predict whether a cut lies in true seam background versus
  eroding bone foreground;
- R215 should feed a learned useful/risk gate rather than a hand-written
  threshold policy.

## Decision

Continue the full channel-val diagnostic to completion.

Next valid step:

1. sync completed R215 candidate CSV;
2. train/evaluate a two-stage candidate gate using R215 inference features;
3. use image-grouped validation, not self-fit, as the promotion evidence;
4. only if the candidate-level frontier improves, build mask-level val
   application and evaluate with R201 metrics.

Do not use clean-test-v2 and do not write final masks from R215 yet.
