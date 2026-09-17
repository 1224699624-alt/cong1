# R070 Pixel Arbitration Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Setup

R070 tested whether the GT-leaking R070 oracle signal over R038/R068/R069 could be recovered by a deployable local pixel residual.

- Train anchor: `r025b_anchor_local_pixel_residual_sam_hqsam`
- Apply anchor: `r038_aggr_r1_t045_m000`
- Train candidates: R068, R069, R025b
- Apply candidates: R068, R069, R038
- Evaluation split: `TSRS_RSNA-Epiphysis_clean_test_v2/test` only

Implementation changes made during the run:

- added `--apply-candidate-exps` to keep train/apply candidate mapping aligned;
- added `--resume-checkpoint` so validation-grid fixes did not require retraining;
- added fast `boundary_kernel <= 0` evaluation mode;
- cached edit zones per radius to avoid repeated morphology costs.

## Result

| System | Dice | Precision | Recall | Boundary IoU | Delta vs R038 |
| --- | ---: | ---: | ---: | ---: | ---: |
| R038 anchor | `0.914357` | `0.899054` | `0.931483` | `0.243176` | baseline |
| R070 local arbitration | `0.914370` | `0.897204` | `0.933503` | `0.244320` | `+0.000013` |
| Target | `>0.931766` | - | - | - | `+0.017409` needed from R038 |

Selected validation config:

- radius `2`
- probability threshold `0.45`
- edit margin `0.15`
- validation Dice `0.894818`

## Interpretation

R070 is effectively flat. It recovers a tiny amount of recall and boundary IoU but loses precision, yielding only a negligible Dice gain over R038. This is not a credible path toward the required `+0.017409` Dice improvement.

The earlier pixel oracle Dice `0.938007` remains GT-leaking diagnostic evidence only. R070 shows that the oracle complementarity is not learnably captured by this local residual with current candidates and validation supervision.

## Decision

Do not continue:

- R038/R068/R069 voting;
- per-image selectors;
- wider R069 trimming grids;
- another MLP pixel residual over the same candidate set.

Next ARIS step should be R071 with a genuinely new mechanism: anatomy-layout/data-label intervention or a literature-driven architecture that changes the prediction representation, not another selector/fusion variant.
