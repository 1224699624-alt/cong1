# R091 Stronger Patch Arbitrator Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R091 tested whether R090 was limited by under-sampling the narrow candidate-disagreement region. It kept the online patch/CNN formulation but increased crop supervision and model capacity.

## Setup

- Base method: R090 online patch disagreement arbitrator
- Train anchor: `r025b_anchor_local_pixel_residual_sam_hqsam`
- Apply anchor: `r038_aggr_r1_t045_m000`
- Candidate: `r068_instance_separation_segmenter`
- Online crops per image: `3`
- Crop size: `96`
- Base channels: `24`
- Epochs: `12`
- Final evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`

## Result

| System | Dice | Delta vs R090 | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| R089 precision MLP | `0.915467` | `-0.000973` | `0.904022` | `0.928470` | `0.245345` |
| R090 patch/CNN | `0.916440` | - | `0.906271` | `0.928080` | `0.251366` |
| R091 stronger patch/CNN | `0.915554` | `-0.000885` | `0.906265` | `0.926351` | `0.245437` |

Best validation config:

- radius `0`
- threshold `0.55`
- margin `0.15`
- original-val Dice `0.895338`

Remaining target gap from R091: `0.016212`.
Remaining target gap from current best R090: `0.015326`.

## Interpretation

R091 did not improve over R090. It kept similar precision but lost recall and Boundary IoU. This suggests the R090 gain was not simply limited by crop count or small model capacity.

The patch/CNN direction remains better than pointwise MLP overall, but blind scaling of the same streamed patch arbitrator is not justified.

## Decision

Keep R090 as the current valid best.

Next step should be low-cost diagnosis before any more training:

- measure complementarity among R038/R089/R090/R091;
- compute per-image and pixel oracle gaps;
- test whether simple validation-selected combinations have any headroom.

If the combination oracle is also flat, pivot away from candidate-disagreement postprocessing.
