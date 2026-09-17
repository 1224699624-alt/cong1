# R281 Low-Support Instance Prior

- Controlled change from R280: `support_alpha 0.05 -> 0.015`.
- Fixed: R279 frozen instance prior, seam alpha `0.05`, four-pixel support radius, mature R275 initialization, 875/96 split and early stopping.
- Dataset: `TSRS_RSNA-Epiphysis` only; clean-test-v2 remains locked.
- Predeclared gate versus R275: gap FP and merge no worse; Recall within `0.004`; Dice within `0.003`; Boundary IoU and Surface Dice 2px within `0.002`.
- If the gate fails, the next experiment is gradient-magnitude normalization rather than another manual coefficient sweep.

## Final result

- R275 -> R281: Dice `0.897965 -> 0.893215`, Recall `0.899057 -> 0.901888`, gap FP `0.187305 -> 0.209843`, merge rate `0.510417 -> 0.520833`, Boundary IoU `0.234335 -> 0.231955`, Surface Dice 2px `0.650056 -> 0.644110`, HD95 `16.350589 -> 19.072712`, ASSD `6.165032 -> 5.021834`.
- The lower support weight recovered Recall but still worsened both separation gates and exceeded the allowed Dice/Boundary/Surface drops.
- Gate: `no_go_use_gradient_normalization`.
- R282 gradient-normalized training was therefore launched without any further manual coefficient sweep.
