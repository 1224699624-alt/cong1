# R089 Precision-Controlled Disagreement Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R089 tested whether R088's weak local-context gain could be improved by controlling the false-positive additions. R088 increased recall but reduced precision and Boundary IoU, so R089 lowered the positive-class loss weight and used a more conservative threshold grid.

## Setup

- Train anchor: `r025b_anchor_local_pixel_residual_sam_hqsam`
- Apply anchor: `r038_aggr_r1_t045_m000`
- Candidate: `r068_instance_separation_segmenter`
- Zone: `anchor_candidate_disagreement`
- Features: `local_stats`
- Positive loss scale: `0.55`
- Train/tune: original `TSRS_RSNA-Epiphysis` train/val
- Final evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`

## Result

| System | Dice | Delta vs R070 | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| R038 anchor | `0.914357` | `-0.000013` | `0.899054` | `0.931483` | `0.243176` |
| R070 prior best | `0.914370` | - | `0.897204` | `0.933503` | `0.244320` |
| R088 local-stats MLP | `0.914957` | `+0.000587` | `0.891148` | `0.941373` | `0.240111` |
| R089 precision-controlled MLP | `0.915467` | `+0.001097` | `0.904022` | `0.928470` | `0.245345` |

Best validation config:

- radius `0`
- threshold `0.55`
- margin `0.05`
- original-val Dice `0.895551`

Remaining target gap: `0.016299`.

## Interpretation

R089 is a real but small positive result. Lowering positive-class weight reverses the R088 precision collapse and produces the best valid clean-test-v2 score so far. However, the gain is only about `+0.0011` over the previous best, while the target requires another `+0.0163`.

This means the candidate-disagreement region contains useful signal, but small independent-pixel MLPs do not have enough capacity or supervision alignment to recover the R086 oracle.

## Decision

Treat R089 as the new local best, but close open-ended small-MLP tuning.

The next experiment should change the formulation, not just the hyperparameters. The most plausible continuation is a patch-level disagreement arbitrator that classifies/edit pixels using local image crops and candidate-mask crops, with conservative edit gating. If that is too costly, pivot back to literature-driven architecture search.
