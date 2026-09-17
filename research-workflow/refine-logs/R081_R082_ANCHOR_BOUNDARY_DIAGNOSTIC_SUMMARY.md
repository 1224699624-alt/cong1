# R081/R082 Anchor-Boundary Diagnostic Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## R081 Anchor-Domain Mismatch

R081 compared local error distributions for three anchors:

| Case | Dice | Precision | Recall | Boundary IoU | FP Share | FN Share | r4 Correction Prevalence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R025b on original val | `0.894701` | `0.888486` | `0.906338` | `0.219753` | `0.535599` | `0.464401` | `0.223058` |
| R025b on clean-test-v2 | `0.914298` | `0.900455` | `0.929894` | `0.242169` | `0.592710` | `0.407290` | `0.204700` |
| R038 on clean-test-v2 | `0.914357` | `0.899054` | `0.931483` | `0.243176` | `0.602643` | `0.397357` | `0.203950` |

R025b and R038 are almost identical on clean-test-v2. The stronger mismatch is not between R025b and R038 on the same target split; it is between original val and clean-test-v2. Original val is weaker, noisier, and has a more balanced FP/FN error profile, while clean-test-v2 errors are more FP-heavy.

## R082 R080 Checkpoint Diagnostic Readout

R082 tested whether the R080 boundary-band checkpoint failed only because of a bad threshold/radius readout. It used a diagnostic clean-test-v2 grid with conservative max-edit limits, so this is not a valid deployable success claim.

Best diagnostic readout:

| Source | Threshold | Radius | Max Edit Frac | Dice | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R080 checkpoint on R038 anchor | `0.65` | `1` | `0.001` | `0.909844` | `0.908297` | `0.912671` | `0.226913` |
| R038 anchor baseline | - | - | - | `0.914357` | `0.899054` | `0.931483` | `0.243176` |

Even with clean-test-v2-assisted readout, the R080 checkpoint remains below R038 by `-0.004512` Dice. Therefore the bottleneck is not simply threshold/radius selection. The learned boundary probability ranking itself is not a useful correction signal for the R038 target anchor.

## Decision

Close the current R080-style learned boundary-band refiner branch:

- do not tune R080 thresholds, radii, max-edit fractions, or epochs;
- do not launch a heavier R038/R025b/R036-derived pseudo-anchor boundary refiner unless a new diagnostic shows substantially better train-target alignment;
- do not claim R082 as a valid result because it uses clean-test-v2 for grid selection.

Next ARIS step should pivot away from local boundary refiner tuning. The next candidate direction should introduce a new source of information or constraint, for example:

1. a test-time consistency / uncertainty agreement method that does not learn corrections from the weak original-val proxy;
2. a data-centric target-split audit to isolate whether the target gap is label-protocol rather than model capacity;
3. a new literature-driven architecture with explicit contour/edge supervision from images rather than anchor-probability editing.

