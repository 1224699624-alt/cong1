# R218 Patch Action Scorer Progress

Date: 2026-07-07

## Purpose

R216 showed partial seam actions are safer than full deletion, but R216/R217
feature gates could not reliably select safe actions. R218 tests whether a
small local CNN patch scorer can learn true-seam versus over-erosion cues from
the crop itself.

This is candidate-action grouped-CV diagnostic work only. It writes no masks
and does not use `clean-test-v2`.

## Implemented

Added:

- `scripts/train_r218_patch_action_scorer.py`

Inputs:

- R216 candidate-action CSV;
- original `TSRS_RSNA-Epiphysis` validation images;
- R110 trainval anchor masks;
- regenerated R216 soft seam action masks.

Patch channels:

- grayscale crop;
- anchor mask crop;
- action mask crop;
- anchor interior distance;
- background distance.

Model:

- small CNN with two-output BCE head:
  - useful probability;
  - risk probability.

Evaluation:

- image-grouped CV;
- frontier over useful/risk thresholds;
- no mask writing.

## Smoke

Remote smoke:

- `limit_images=12`
- folds: `3`
- epochs: `3`
- crops: `132`
- device: CUDA

Smoke completed successfully and produced:

- `outputs/analysis/r218_patch_action_scorer_smoke_groupedcv3.json`
- `outputs/analysis/r218_patch_action_scorer_smoke_groupedcv3.csv`

Smoke was functional only; it was too small to judge promotion.

## Full-Val Run

Launched:

- screen: `r218_patch_action_fullval`
- log: `outputs/bridge_logs/r218_patch_action_scorer_fullval.log`
- JSON: `outputs/analysis/r218_patch_action_scorer_fullval_groupedcv5.json`
- CSV: `outputs/analysis/r218_patch_action_scorer_fullval_groupedcv5.csv`

Settings:

- folds: `5`
- epochs: `8`
- batch size: `64`
- crop size: `48`
- device: CUDA

No clean-test-v2 data is used.

Completed:

- crops: `1060`
- images: `92`
- rows: `1060`
- useful labels: `382`
- risk labels: `619`

Artifacts synced locally:

- `outputs/analysis/r218_patch_action_scorer_fullval_groupedcv5.json`
- `outputs/analysis/r218_patch_action_scorer_fullval_groupedcv5.csv`
- `outputs/bridge_logs/r218_patch_action_scorer_fullval.log`

## Promotion Gate

R218 only moves forward if grouped-CV improves the low-risk frontier over:

- R216 LogReg quick/hard risk<=2:
  - accepted `7`, useful `4`, risk `2`, Boundary IoU `+0.000330`
- R216 LogReg quick/hard risk<=5:
  - accepted `10`, useful `6`, risk `3`, Boundary IoU `+0.000229`

If it cannot improve this frontier, do not write masks and do not run
clean-test-v2.

## Result

R218 did not improve the low-risk frontier.

R216 LogReg quick/hard reference:

- risk<=2: accepted `7`, useful `4`, risk `2`, Boundary IoU `+0.000330`
- risk<=5: accepted `10`, useful `6`, risk `3`, Boundary IoU `+0.000229`
- risk<=10: accepted `18`, useful `10`, risk `7`, Boundary IoU `+0.000380`

R218 patch action scorer:

- risk<=5: accepted `7`, useful `2`, risk `5`, Boundary IoU `-0.000042`
- risk<=10: accepted `13`, useful `4`, risk `9`, Boundary IoU `-0.000147`
- risk<=20: accepted `27`, useful `11`, risk `16`, Boundary IoU `+0.000052`
- best broad-threshold row accepts `92` actions but still has `42` risk
  accepts and is not promotion-safe.

Decision:

`R218 = no_go_for_mask_level`.

Do not write R218 masks and do not use `clean-test-v2`.

Interpretation:

- R216 partial seam actions remain the best discovered action space.
- But neither hand-crafted patch statistics nor a small local CNN can safely
  select actions under image-grouped CV.
- The bottleneck is likely candidate/action ambiguity and label definition:
  many proposed actions are too visually similar at crop scale, and safe
  selection may need instance-aware context or a different target than
  delete/suppress pixels.

Recommended next options:

1. Stop threshold/gate variants around R216-R218 and build hard-case
   visualizations to inspect why useful/risk candidates look ambiguous.
2. Try an instance-preserving formulation rather than deletion:
   - predict a seam/background probability map;
   - constrain edits to not increase component error;
   - apply via conservative post-processing.
3. Revisit baseline path: complete Swin/TransUNet-style baseline comparison
   while the new method direction is being reconsidered.

