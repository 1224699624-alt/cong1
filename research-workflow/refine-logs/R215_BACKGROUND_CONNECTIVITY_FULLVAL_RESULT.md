# R215 Background Connectivity Full-Val Result

Date: 2026-07-07

## Question

Test the user's inverted topology idea:

- instead of encouraging epiphysis foreground connectivity;
- encourage / detect local background-channel connectivity in the bone seam;
- use it to reduce bone-gap adhesion and boundary errors without over-erosion.

This is original train/val diagnostic work only. It does not use
`clean-test-v2`, writes no final masks, and is not R201 mask-level evidence.

## Candidate Diagnostic

Final synced artifact:

- `outputs/analysis/r215_background_connectivity_channel_val_candidates.csv`
- `outputs/analysis/r215_background_connectivity_channel_val_summary.json`

Coverage:

- candidate rows: `265`
- validation images: `92`
- quick-useful candidates: `73`
- hard-risk candidates: `177`
- GT-diagnostic safe-channel positives: `73`

Safe-channel positives are a train/val diagnostic, not a deployable rule,
because they use GT-derived cut fractions. They show that the signal exists:

- quick-useful rate: `0.821918`
- hard-risk rate: `0.095890`
- mean cut GT foreground fraction: `0.216468`
- mean cut GT gap fraction: `0.767770`
- mean Dice delta: `+0.000210`
- mean Boundary IoU delta: `+0.001323`
- mean gap-region FP delta: `-0.001606`

Interpretation: the inverted idea is valid at candidate level. Useful cuts
often look like opening a local background channel through the seam, and the
best diagnostic group mostly lies in GT gap rather than GT bone.

## Deployable Gate Tests

### R215-only features

Artifacts:

- `outputs/analysis/r215_background_channel_gate_full_groupedcv5_hgb_quick.json`
- `outputs/analysis/r215_background_channel_gate_full_groupedcv5_logreg_quick.json`
- `outputs/analysis/r215_background_channel_gate_full_groupedcv5_hgb_safe.json`
- `outputs/analysis/r215_background_channel_gate_full_groupedcv5_logreg_safe.json`

Best grouped-CV result from R215-only inference features:

- HGB quick/hard: accepts `1`, useful `0`, risk `1`
- LogReg quick/hard: accepts `3`, useful `2`, risk `1`
- HGB safe/overerosion: accepts `1`, useful `0`, risk `1`
- LogReg safe/overerosion: accepts `0`

Decision: R215-only background-channel features are not enough for a safe
deployable gate.

### R215 + R212/R214 fused features

Implemented:

- `scripts/merge_r215_background_features.py`
- optional `--extra-feature-prefix` support in
  `scripts/train_r213_two_stage_candidate_gate.py`

Merged artifact:

- `outputs/analysis/r215_merged_r212_f5_fullval_val_candidates.csv`
- base rows: `433`
- matched R215 rows: `265`
- match rate: `0.612009`

Grouped-CV fused artifacts:

- `outputs/analysis/r215_fused_r213_groupedcv5_hgb.json`
- `outputs/analysis/r215_fused_r213_groupedcv5_logreg.json`
- control: `outputs/analysis/r215_fused_r213_groupedcv5_hgb_noextra_control.json`

Low-risk frontier comparison:

- previous/control HGB risk<=0: `0` strict-useful, `0` risk
- fused HGB risk<=0: `1` strict-useful, `0` risk
- fused HGB risk<=1: still only `1` strict-useful
- reaching `5+` useful candidates still requires many hard-risk accepts

Interpretation: R215 features improve the frontier slightly, but not enough to
justify mask-level validation or clean-test-v2 application.

## Decision

`R215` is a promising diagnostic signal but not yet a promotable model change.

Do not:

- apply R215 to `clean-test-v2`;
- write final masks from the current gate;
- claim R201 improvement from candidate-level diagnostics;
- use GT safe-channel proxy as an actual test-time rule.

Continue only if the next version changes the candidate/action space, not just
thresholds. Current candidate deletion plus feature gate is bottlenecked by
generalization and over-erosion ambiguity.

## Next Direction

Recommended `R216`:

1. Keep the background-connectivity idea.
2. Replace hard deletion candidates with softer seam-aware actions:
   - local probability suppression rather than binary removal;
   - narrow-band seam logits/refiner;
   - component-preserving cut proposal with explicit foreground protection.
3. Add GT-free over-erosion predictors from image context:
   - intensity contrast across the proposed seam;
   - distance-to-boundary / medial-axis position;
   - whether the cut splits a plausible single bone body versus a true inter-bone seam.
4. Validate only on original train/val first.
5. Promote only after R213 validation gate shows meaningful accepted rate,
   positive boundary/gap deltas, and no Dice/Recall/component collapse.

