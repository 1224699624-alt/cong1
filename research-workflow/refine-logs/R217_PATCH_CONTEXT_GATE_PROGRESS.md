# R217 Patch-Context Gate Progress

Date: 2026-07-07

## Purpose

R216 showed that partial seam actions are safer than full hard deletion, but
the action-level grouped-CV gate could not reliably predict safe actions from
geometry/background-channel features alone.

R217 keeps the R216 partial seam action space and adds GT-free local patch
context to distinguish:

- true inter-bone seam / background gap;
- versus over-erosion cutting into a single bone body.

This is still train/val-only candidate-action diagnostic work. It writes no
masks and does not use `clean-test-v2`.

## Implemented

Added:

- `scripts/enrich_r217_soft_seam_patch_context.py`

Updated:

- `scripts/train_r216_soft_seam_action_gate.py`
  - added optional `--extra-feature-prefix`, so R216 baseline behavior remains
    unchanged unless `--extra-feature-prefix r217_` is passed.

R217 enrich regenerates the same R216 candidate actions and merges local
image/patch features into the R216 CSV by:

- image;
- candidate rank;
- action fraction.

Feature families:

- cut/ring/inner-ring/outer-ring intensity;
- cut/ring/outer gradient;
- cut-to-ring and inner-to-outer contrast;
- anchor distance and background distance;
- local anchor/background fraction;
- boundary contact and neighbor foreground/background fractions;
- local crop size and normalized location.

## Remote Run

Launched:

- screen: `r217_patch_context_enrich`
- log: `outputs/bridge_logs/r217_soft_seam_patch_context_fullval.log`
- output CSV: `outputs/analysis/r217_soft_seam_patch_context_fullval_candidates.csv`
- output JSON: `outputs/analysis/r217_soft_seam_patch_context_fullval_summary.json`

The run is original `val` only and uses the R216 full-val candidate-action CSV
as input.

Completed:

- rows: `1060`
- images: `92`
- matched rows: `1060`
- unmatched rows: `0`
- match rate: `1.0`

Artifacts synced locally:

- `outputs/analysis/r217_soft_seam_patch_context_fullval_candidates.csv`
- `outputs/analysis/r217_soft_seam_patch_context_fullval_summary.json`
- `outputs/bridge_logs/r217_soft_seam_patch_context_fullval.log`

## Next Gate

After enrichment completes:

1. sync the enriched CSV/JSON locally;
2. verify match rate, expected target is near `1.0`;
3. run R216 action gate with `--extra-feature-prefix r217_`:
   - HGB quick/hard;
   - LogReg quick/hard;
   - HGB safe/overerosion;
   - LogReg safe/overerosion.
4. compare the low-risk frontier against R216:
   - R216 LogReg quick/hard: accepts `7`, useful `4`, risk `2`;
   - R216 HGB quick/hard: accepts `91`, useful `41`, risk `48`.

Promotion requires a materially better low-risk frontier. Otherwise R217 should
be treated as another diagnostic feature audit, not a mask-level model.

## Grouped-CV Gate Result

Artifacts:

- `outputs/analysis/r217_patch_context_action_gate_groupedcv5_hgb_quick.json`
- `outputs/analysis/r217_patch_context_action_gate_groupedcv5_logreg_quick.json`
- `outputs/analysis/r217_patch_context_action_gate_groupedcv5_hgb_safe.json`
- `outputs/analysis/r217_patch_context_action_gate_groupedcv5_logreg_safe.json`

Comparison against R216:

- R216 LogReg quick/hard low-risk reference:
  - risk<=2: accepted `7`, useful `4`, risk `2`, Boundary IoU `+0.000330`
  - risk<=5: accepted `10`, useful `6`, risk `3`, Boundary IoU `+0.000229`
- R217 LogReg quick/hard:
  - risk<=5: accepted `10`, useful `4`, risk `4`, Boundary IoU `+0.000110`
  - risk<=10: accepted `18`, useful `7`, risk `9`, Boundary IoU `+0.000065`
- R217 HGB quick/hard:
  - no meaningful low-risk frontier before risk<=20;
  - risk<=20: accepted `30`, useful `12`, risk `18`.
- R217 safe/overerosion targets:
  - both HGB and LogReg remain high-risk;
  - no low-risk frontier improves over R216.

Decision:

`R217 = no_go_for_mask_level`.

The enrichment succeeded technically and produced a complete matched table, but
the added hand-crafted patch-context features did not improve grouped-CV action
selection. Do not write R217 masks and do not use `clean-test-v2`.

Interpretation:

- R216's partial action space remains useful.
- The bottleneck is not just missing scalar context statistics.
- The remaining failure likely requires either a learned local patch scorer or
  a different candidate/action formulation that better preserves instances.

Recommended R218:

1. Keep R216 partial seam actions.
2. Train a small local patch action scorer using grouped train/val only:
   - input crop around candidate action;
   - channels: grayscale image, anchor mask, candidate action mask, distance
     maps/background channel proxy;
   - labels: useful vs hard-risk / overerosion from R216 diagnostics.
3. Evaluate only with image-grouped CV first.
4. If and only if the low-risk frontier improves, build a mask-level original
   val application and run R201 metrics.

