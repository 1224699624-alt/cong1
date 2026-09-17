# R178 Delete-Only Bridge Suppressor Result

Date: 2026-07-03

## Purpose

R178 tested a conservative postprocessor after the R177 free add/delete arbitrator fragmented masks. The rule was delete-only, selected on original validation using the non-leaking train/val anchor proxy, and applied once to locked R110 clean-test-v2 masks.

## Protocol

- Tuning split: `TSRS_RSNA-Epiphysis/val`
- Final evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Apply anchor: `r110_r100_r108_patch_basic`
- Train/val anchor proxy: `r100_like_r025b_r097_patch_basic_trainval`
- Success target: clean-test-v2 Dice `> 0.9317660066557425`
- No new/reannotated test split was used.

## Result

Best validation-selected config:

- `gap_radius=7`
- `min_neck_width=1`
- `min_area=4`
- `max_area=64`

Clean-test-v2 mean metrics:

- Dice: `0.9177231563529792`
- Boundary IoU: `0.25189601044085763`
- component-count error: `2.506172839506173`
- false-bridge flag: `0.7901234567901234`
- separation-band FP: `0.24784517062895237`
- target margin: `-0.01404285030276331`

## Interpretation

R178 did not improve the valid best. The component-count guardrails made the full clean-test-v2 result effectively return to the R110 anchor for most images. This is safer than R177 but does not solve bone-gap adhesion or move toward the target.

Decision: close delete-only geometry postprocessing as a primary route. The next constraint route should put bone-gap and boundary constraints into training, while still selecting on original train/val and keeping clean-test-v2 as the locked final evaluation.

