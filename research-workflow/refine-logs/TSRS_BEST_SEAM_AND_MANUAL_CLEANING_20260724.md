# TSRS Best Seam Experiment and Manual Cleaning Package

## Best completed seam-separation experiment

Under the original-val R201 protocol, R260 is the strongest completed deployable
trade-off found in the archived TSRS experiments. It improves Dice, Boundary IoU/F1,
gap-region FP, merge rate, component count MAE, Surface Dice, HD95 and ASSD over its
paired mature R202 baseline. Recall decreases from 0.916238 to 0.899632, so the result
is promising rather than proof that every improvement comes from correct separation.

| Metric | R260 paired baseline | R260 prior + loss | Delta |
| --- | ---: | ---: | ---: |
| Dice | 0.897112 | 0.902759 | +0.005647 |
| Recall | 0.916238 | 0.899632 | -0.016606 |
| Boundary IoU | 0.239893 | 0.244642 | +0.004748 |
| Boundary F1 | 0.379017 | 0.385938 | +0.006921 |
| gap-region FP | 0.205547 | 0.175200 | -0.030347 |
| component merge rate | 0.572917 | 0.500000 | -0.072917 |
| component count MAE | 5.666667 | 1.906250 | -3.760417 |
| Surface Dice 2 px | 0.654511 | 0.667281 | +0.012770 |
| Surface Dice 5 px | 0.896460 | 0.912211 | +0.015751 |
| HD95 (px) | 54.341353 | 15.864084 | -38.477269 |
| ASSD (px) | 14.173772 | 4.320395 | -9.853378 |

R275-seam obtains a lower merge rate (0.427083 versus the R275 control 0.510417),
but its HD95 worsens from 16.350589 to 33.386433 px and Dice/Recall both decline.
It is therefore a useful separation-specific comparator, not the best overall model.

## Manual cleaning package

- Source: `data/raw/TSRS_RSNA-Epiphysis` only.
- Output: `outputs/manual_review/tsrs_epiphysis_label_cleaning_v1`.
- Cases: train 875, val 96, test 97; total 1068.
- Each panel: original X-ray, indexed color instance label, X-ray/label overlay.
- Editable labels are copies under `working_labels/<split>/`; raw labels are unchanged.
- Test is explicitly review-only and must not be used for tuning or model selection.
- Worklist diagnostics: 170 non-contiguous-ID cases, 132 cases with at least one
  disconnected same-ID instance, 8 cases with tiny instances, and zero image/label
  size mismatches. These flags prioritize review but are not automatic error labels.

