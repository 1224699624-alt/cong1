# R210 Component-Preserving Instance Split Smoke Result

Date: 2026-07-06

## Purpose

Implement and smoke-test the first R210 model-changing route after R209-F0:

- anchor-conditioned split likelihood model,
- component-safe per-image acceptance,
- train/val only,
- no clean-test-v2 application.

## Implemented Script

Script:

- `scripts/train_r210_component_preserving_instance_split.py`

Key properties:

- Uses `TSRS_RSNA-Epiphysis` only.
- Uses true R110 train/val anchor masks from `r110_r100_r108_patch_basic_trainval`.
- Trains a small MLP on image/anchor/distance/local-stat features.
- Predicts cut likelihood inside R110 anchor masks.
- Applies cuts only through per-component greedy acceptance.
- Rejects candidates that violate Dice, IoU, Recall, component count MAE, or diagnostic-gain gates.
- Does not apply clean-test-v2 unless explicitly requested after validation review.

## Smoke Runs

Remote workspace:

- `/home/shenzeyu/workspace/YOLO_SAM_generic_src`

Environment:

- `/home/shenzeyu/miniconda3/envs/aris-nnunet-r202-fast/bin/python`

Synced artifacts:

- `outputs/analysis/r210_component_preserving_instance_split_smoke5_val_summary.json`
- `outputs/analysis/r210_component_preserving_instance_split_smoke5_val_per_image.csv`
- `outputs/analysis/r210_component_preserving_instance_split_smoke6_val_summary.json`
- `outputs/analysis/r210_component_preserving_instance_split_smoke6_val_per_image.csv`

## Findings

### Smoke5: broad FP target

The model produced nonzero candidate cuts, but the candidate behaved like a deletion mask instead of a clean split map.

Observed on 8-val smoke:

- `accepted`: `0`
- `accepted_cut_pixels`: `0`
- candidate cuts were often very large.
- candidate component count MAE worsened badly.
- candidate recall and Dice dropped.
- safety gate rejected all edits.

Interpretation:

- The component-safe gate is doing its job.
- Broad `anchor & ~GT` supervision repeats the R177/R205 failure mode: attractive gap-FP reduction but unsafe fragmentation/erosion.

### Smoke6: narrowed gap-adjacent target

The target was narrowed to `anchor & ~GT & near_gt_gap`.

Observed on 8-val smoke:

- `val_gate_pass`: `false`
- `candidate_cut_pixels`: `0`
- `accepted_cut_pixels`: `0`
- all final metrics match anchor.

Interpretation:

- The narrowed target avoids broad deletion but becomes too sparse for the small pixel MLP to produce usable cut candidates.
- This is not enough evidence to reject R210 as a component-preserving idea, but it rejects the cheap pixel-MLP variant.

## Decision

Do not launch a full R210-F0 pixel-MLP run on all val yet.

The current R210 infrastructure is useful:

- feature extraction,
- R201 metric gate,
- candidate/reject diagnostics,
- component-safe acceptance,
- isolated output writing.

But the candidate generator must change before spending more GPU/CPU time.

## Next R210 Variant

Move to `R210-F1`, not more threshold sweeping:

1. Use a small patch/CNN split-map model instead of a per-pixel MLP.
2. Train directly on local hard merge patches from R209/R210 diagnostics.
3. Predict a spatially coherent split stroke map.
4. Keep the same component-safe greedy acceptance gate.
5. Validate only on original val; clean-test-v2 remains locked.

Alternative if R210-F1 is too slow:

- Generate deterministic geometry candidates from merged-component skeleton/neck analysis, then use the existing R210 acceptance gate. This avoids learning a noisy deletion map while still testing component-preserving split feasibility.

Decision: R210 should continue, but not as the current pixel-MLP candidate generator.
