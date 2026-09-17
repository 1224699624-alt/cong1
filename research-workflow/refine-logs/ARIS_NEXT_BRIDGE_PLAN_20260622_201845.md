# ARIS Next Bridge Plan: R014 Recall Rescue

**Date**: 2026-06-22 20:18
**Workflow position**: experiment-bridge, after R011/R012/R013 failed to beat ARAA.

## Current Frozen Targets

| System | Slice | Dice | IoU | Precision | Recall | Boundary IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ARAA epoch_98 | clean-test-v2 | 0.911766 | 0.838436 | 0.910177 | 0.914691 | 0.235265 |
| keepbone_cutgap_v2 | clean-test-v2 | 0.910029 | 0.835434 | 0.913101 | 0.908201 | 0.228489 |
| R010 precision_tune | clean-test-v2 | 0.910679 | TBD | TBD | TBD | TBD |
| R011 v3 | clean-test-v2 | 0.909178 | TBD | TBD | 0.906651 | TBD |
| R012 thr054 | clean-test-v2 | 0.910030 | TBD | TBD | TBD | TBD |
| R013 prefgate | original test | 0.878543 | 0.786301 | 0.869686 | 0.891269 | 0.197166 |

## Diagnosis

The strongest internal method is only `0.001737` Dice below ARAA on clean-test-v2. Its precision is already slightly higher than ARAA, while recall and boundary IoU are lower. The next candidate should not be another pure precision tightening pass.

## R014 Method

Candidate: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue_sam_hqsam`

Implementation:
- Add target mode `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue`.
- Reuse v2 prediction-aware keepbone/cutgap stage2.
- Add a small recall-rescue term only where:
  - candidate union or boundary foreground supports the missing area,
  - structure prior or instance core supports bone structure,
  - anatomy gradient/band indicates thin boundary structure,
  - gap risk is low according to instance separation, overlap gap, disagreement boundary, and gap band.

## Decision Gate

Promote only if:
- original test does not regress materially relative to `keepbone_cutgap_v2`;
- clean-test-v2 Dice beats `0.910029`;
- preferred success is Dice `>= 0.911766`;
- recall or boundary IoU improves without an obvious precision/specificity collapse.

## Launchers

- Train/evaluate original test:
  `run_epiphysis_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue.sh`
- Evaluate clean-test-v2:
  `run_epiphysis_clean_test_keepbone_cutgap_refiner_v2_recall_rescue.sh`

## ARIS Next Step

Run R014 through experiment-bridge. If R014 beats the current internal anchor or approaches ARAA with coherent hard-case behavior, regenerate hard-case panels for `2982/1475/3040/1433` and then re-enter auto-review-loop.

## R014 Outcome

R014 finished on the original test slice and failed the control gate:

| Metric | R014 |
| --- | ---: |
| Dice | 0.890988 |
| IoU | 0.806119 |
| Precision | 0.893728 |
| Recall | 0.891588 |
| Specificity | 0.996934 |
| Boundary IoU | 0.213154 |

Because this is below the current internal original-test anchor, clean-test-v2 evaluation is skipped. The recall-rescue term did not deliver the intended controlled gain.

## Next Direction: R015 Selector

The next bridge candidate should not add another handcrafted recovery term. Use the validation split to arbitrate among existing strong masks:

- keep `keepbone_cutgap_v2` as the default high-precision base;
- allow local replacement only from candidates that historically improve recall/boundary behavior;
- select replacement regions by validation-calibrated criteria: candidate disagreement, anatomy band/gradient, structure prior, instance-separation low-risk, and boundary foreground/background support;
- reject any rule that improves recall only by lowering precision/specificity on original test.

This turns the remaining ARAA gap into a selector/calibration problem rather than another unbounded morphology problem.

## R015 Outcome

R015 finished successfully and crossed the primary target.

Selected validation rule:

- base: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam`
- add: `allstack_anatomy_roi_aic_refiner_sam_hqsam`
- add radius: `1`
- max add component: `1000000`
- support minimum: `0`
- close kernel: `1`

| Slice | Dice | IoU | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| original test | 0.896905 | 0.815969 | 0.884421 | 0.912982 | 0.225330 |
| clean-test-v2 | 0.913497 | 0.841307 | 0.899351 | 0.929412 | 0.239510 |

Decision:

- clean-test-v2 Dice `0.913497` exceeds ARAA `0.911766`;
- original-test Dice `0.896905` improves over the internal anchor `0.893694`;
- no R016 is needed for the current user target.
