# R088 Local-Context Disagreement Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R088 was the local-context follow-up to R087. R087 showed that a pointwise MLP cannot recover the R086 disagreement oracle. R088 tested whether multiscale neighborhood statistics make the add/remove decision learnable inside the same narrow disagreement zone.

## Change From R087

Added `--feature-mode local_stats` in `scripts/train_anchor_pixel_residual.py`.

Extra features:

- image mean and standard deviation at 3, 7, and 15 px windows;
- local anchor density at 3, 7, and 15 px;
- local candidate-vote density at 3, 7, and 15 px;
- local disagreement density at 3, 7, and 15 px.

The experiment kept the same R038 apply anchor and R068 candidate, and used a narrowed 18-config val grid.

## Result

| System | Dice | Delta vs R038 | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| R038 anchor | `0.914357` | - | `0.899054` | `0.931483` | `0.243176` |
| R070 prior best | `0.914370` | `+0.000013` | `0.897204` | `0.933503` | `0.244320` |
| R087 pointwise MLP | `0.914349` | `-0.000008` | `0.898842` | `0.931698` | `0.243388` |
| R088 local-stats MLP | `0.914957` | `+0.000600` | `0.891148` | `0.941373` | `0.240111` |

Best validation config:

- radius `1`
- threshold `0.55`
- margin `0.05`
- original-val Dice `0.896103`

## Interpretation

Local context helps, but only weakly. R088 improves Dice by about `+0.0006` over the R038/R070 line, mainly by adding recall. The cost is lower precision and lower Boundary IoU, which means the learner is biased toward adding pixels in the disagreement region rather than making calibrated boundary decisions.

The remaining target gap is `0.016809`, far larger than the observed gain. This is not a credible path to the requested target through small MLP tuning alone.

## Decision

Close blind small-MLP disagreement tuning.

If this branch is revisited, it should be a different formulation with explicit precision control, for example:

- asymmetric loss or sampling that penalizes false-positive additions more strongly;
- calibration/abstention so edits only happen on high-confidence disagreement pixels;
- patch/CNN learner trained on disagreement crops rather than independent pixels.

Given the large remaining gap, the higher-value next ARIS step is to pivot to a stronger patch-level contour/candidate arbitration model or a new literature-driven architecture, not another threshold sweep of the same MLP.
