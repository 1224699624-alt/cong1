# R079/R080 Boundary-Band Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## R079 Diagnostic

R079 asked whether the remaining R038 error is locally correctable near the mask boundary.

| System | Dice | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: |
| R038 anchor | `0.914357` | `0.899054` | `0.931483` | `0.243176` |
| boundary-band oracle r16 | `0.998376` | `0.998224` | `0.998537` | `0.985692` |

This is a GT-leaking upper bound, not a valid model result. It proves that local contour edits could in principle close the target gap.

## R080 Valid Training Run

R080 trained a local boundary-band CNN refiner:

- train/val anchor: `r025b_anchor_local_pixel_residual_sam_hqsam`;
- apply anchor: `r038_aggr_r1_t045_m000`;
- allowed edits only inside the anchor boundary band;
- success evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

Best validation:

- epoch `25`;
- val Dice `0.897662`;
- threshold `0.50`;
- edit radius `4`;
- val Boundary IoU `0.539012`;

Final clean-test-v2:

| Dice | Precision | Recall | Boundary IoU | False Bridge |
| ---: | ---: | ---: | ---: | ---: |
| `0.901690` | `0.888984` | `0.916046` | `0.200874` | `0.567901` |

## Interpretation

The branch is informative but negative.

R079 says the R038 residual error is locally correctable. R080 says the current way of learning the local correction does not transfer from R025b train/val anchors to R038 clean-test-v2 anchors.

The likely bottleneck is anchor-domain mismatch: the learner sees R025b error geometry during training but is asked to edit R038 aggressive masks at test time. It learns a conservative boundary rule that improves some structure metrics, but it removes too much true foreground and lands below R038/R070.

## Decision

Do not continue by tuning R080 thresholds, radii, epochs, or band weights.

Next useful step is R081:

1. diagnose whether R025b-train-anchor errors differ from R038-clean-test-anchor errors;
2. if mismatch is strong, avoid more train-on-R025b/apply-to-R038 editors;
3. pivot toward either self-supervised/test-time consistency, pseudo-anchor generation for train/val that mimics R038, or a different architecture that does not depend on this anchor mapping.

