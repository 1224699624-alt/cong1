# R282 Gradient-Normalized Instance Prior

- Trigger: R281 failed its predeclared original-val gate.
- Fixed: frozen instance prior, seam/support geometry, mature R275 initialization, 875/96 split, base nnU-Net loss and auxiliary alpha `0.05`.
- Change: support loss is multiplied by the ratio of seam/support gradient RMS measured at the highest-resolution logits. The detached ratio is clipped to `[0.05, 1.0]` and smoothed with EMA `0.95`.
- Goal: equalize the two auxiliary gradient contributions without another manually selected support coefficient.
- `TSRS_RSNA-Epiphysis` only; clean-test-v2 remains locked.

## Final result

- Early stopped after 31 epochs; best EMA pseudo Dice `0.909417`.
- Learned support scale stabilized around `0.097`, giving an effective support coefficient near `0.00485` while retaining seam coefficient `0.05`.
- R275 -> R282:
  - Dice `0.897965 -> 0.893847` (`-0.004118`).
  - Precision `0.902028 -> 0.899049` (`-0.002978`).
  - Recall `0.899057 -> 0.896966` (`-0.002091`).
  - gap FP `0.187305 -> 0.195560` (`+0.008255`, worse).
  - merge rate `0.510417 -> 0.468750` (`-0.041667`, better).
  - component count MAE `2.333333 -> 2.020833` (`-0.312500`, better).
  - Boundary IoU `0.234335 -> 0.234080` (`-0.000255`).
  - Boundary F1 `0.373105 -> 0.372561` (`-0.000544`).
  - Surface Dice 2px `0.650056 -> 0.648298` (`-0.001757`).
  - Surface Dice 5px `0.900422 -> 0.900166` (`-0.000256`).
  - HD95 `16.350589 -> 18.620005` (`+2.269416`, worse).
  - ASSD `6.165032 -> 4.814373` (`-1.350659`, better).
- Gate: no-go under the predeclared criteria because gap FP is worse and Dice exceeds the allowed `0.003` drop. The generic gate JSON retains the older failure label `no_go_use_gradient_normalization`; semantically this means `no_go_after_gradient_normalization` for R282.
- Interpretation: gradient balancing resolves most of the R279/R280 foreground contraction-expansion instability and preserves local boundary/surface metrics, but support placement still leaks foreground into some seam pixels. The next change should be spatial/confidence gating, not another global weight adjustment.
