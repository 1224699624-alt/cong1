# R283 Spatial/Confidence-Gated Support Status

- Completed on `2026-07-22`; early stopped after 28 epochs (best EMA pseudo Dice
  `0.910719`).
- Dataset: `TSRS_RSNA-Epiphysis`, 875 train / 96 original-val.
- `clean-test-v2` and threshold search were not used.
- The spatial invariant held throughout training: mean support/seam-buffer overlap was
  exactly `0.0` in every logged epoch.

## Original-val R201 result versus mature R275

- Dice: `0.897965 -> 0.893652` (`-0.004314`).
- Precision: `0.902028 -> 0.904317` (`+0.002290`).
- Recall: `0.899057 -> 0.891565` (`-0.007493`).
- gap-region FP: `0.187305 -> 0.183797` (`-0.003509`, better).
- component merge rate: `0.510417 -> 0.458333` (`-0.052083`, better).
- component count MAE: `2.333333 -> 2.052083` (`-0.281250`, better).
- Boundary IoU: `0.234335 -> 0.234479` (`+0.000144`).
- Boundary F1: `0.373105 -> 0.373074` (`-0.000030`).
- Surface Dice 2 px: `0.650056 -> 0.649073` (`-0.000982`).
- Surface Dice 5 px: `0.900422 -> 0.901097` (`+0.000676`).
- HD95: `16.350589 -> 18.641295` (`+2.290706`, worse).
- ASSD: `6.165032 -> 4.752355` (`-1.412677`, better).

## Decision

`no_go_after_spatial_confidence_gating`: the gate placement fixed the R282 support
leakage and improved gap/merge behavior, but the high-confidence interior support is
too selective to prevent foreground contraction. Dice and Recall exceed their allowed
drops. Do not use `clean-test-v2` and do not select R283 as the final method.
