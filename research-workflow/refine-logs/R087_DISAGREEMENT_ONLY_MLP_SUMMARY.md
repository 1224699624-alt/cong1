# R087 Disagreement-Only MLP Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R087 tested whether the R086 candidate-disagreement oracle can be recovered by a valid learner. The learner was only allowed to edit pixels where the R038 anchor and R068 candidate disagree.

## Setup

- Train anchor: `r025b_anchor_local_pixel_residual_sam_hqsam`
- Apply anchor: `r038_aggr_r1_t045_m000`
- Candidate: `r068_instance_separation_segmenter`
- Zone: `anchor_candidate_disagreement`
- Model: pointwise local MLP, hidden width 48
- Train/tune: original `TSRS_RSNA-Epiphysis` train/val
- Final evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`

## Result

| System | Dice | IoU | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| R038 anchor | `0.914357` | - | `0.899054` | `0.931483` | `0.243176` |
| R070 best prior local arbitration | `0.914370` | `0.842781` | `0.897204` | `0.933503` | `0.244320` |
| R087 | `0.914349` | `0.842747` | `0.898842` | `0.931698` | `0.243388` |

R087 selected radius `0`, threshold `0.45`, margin `0.15` on original val. The selected val Dice was only `0.894748`, so the weak clean-test-v2 result is consistent with a weak validation signal rather than a target-split accident.

## Interpretation

The R086 oracle remains real but is not learnable with this pointwise feature set. The model sees image intensity, anchor/candidate masks, vote/disagreement indicators, signed anchor distance, and gradient magnitude, but no local texture or neighborhood context. That likely makes the add/remove decision underdetermined in the very narrow disagreement region.

## Decision

Run one final low-cost variant, R088, with multiscale local statistics inside the same disagreement region:

- local image mean/std at 3/7/15 px;
- local anchor/vote/disagreement density at 3/7/15 px;
- narrowed val grid to avoid another broad CPU sweep.

If R088 cannot materially beat R038/R070, close the small-MLP disagreement route and pivot away from local postprocessing.
