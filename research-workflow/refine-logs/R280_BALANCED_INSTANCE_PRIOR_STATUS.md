# R280 Balanced Instance-Prior nnU-Net

## Controlled change

R280 keeps the R279 frozen instance-seam input, mature R275 initialization, Dataset279 preprocessing, split, optimizer, learning rate, and early stopping. Only the auxiliary loss changes.

## Loss

- Native nnU-Net Dice/CE.
- Seam-center suppression: foreground odds are penalized on high-prior pixels labeled as background, weight `0.05`.
- Bone-flank support: the prior is dilated by four pixels and foreground is encouraged where this neighborhood overlaps labeled bone, weight `0.05`.
- The support term is restricted to GT bone during training, so it cannot reward filling the labeled seam background.

This tests whether two-sided supervision can retain R279's gap/merge improvements without its foreground-contraction failure.

## Protocol

- `TSRS_RSNA-Epiphysis` only: 875 train / 96 original-val.
- `clean-test-v2` locked and unused.
- Validation input priors remain frozen and GT-free.
- Primary gate: gap-region FP, merge rate, Boundary IoU/F1, Surface Dice, HD95/ASSD, and Recall relative to both R275 and R279.

## Final result (2026-07-21)

- Early stopped after 37 epochs; best EMA pseudo Dice: `0.906313`.
- R275 control -> R280:
  - Dice: `0.897965 -> 0.889279` (`-0.008687`).
  - Precision: `0.902028 -> 0.869291` (`-0.032737`).
  - Recall: `0.899057 -> 0.918552` (`+0.019494`).
  - gap-region FP: `0.187305 -> 0.267155` (`+0.079849`, worse).
  - component merge rate: `0.510417 -> 0.552083` (`+0.041667`, worse).
  - Boundary IoU: `0.234335 -> 0.216154` (`-0.018181`).
  - Boundary F1: `0.373105 -> 0.349243` (`-0.023862`).
  - Surface Dice 2px: `0.650056 -> 0.614750` (`-0.035306`).
  - Surface Dice 5px: `0.900422 -> 0.885949` (`-0.014473`).
  - HD95: `16.350589 -> 18.883937` (`+2.533349`, worse).
  - ASSD: `6.165032 -> 5.226750` (`-0.938282`, better).
- Loss dynamics were finite and both regions had similar pixel mass, but support loss magnitude was roughly three times seam loss (`~0.40` versus `~0.13`). Equal coefficients therefore made bone support dominant.
- Interpretation: R280 successfully reverses R279's foreground contraction and restores/increases Recall, but overshoots and fills seam background. The equal-weight formulation is a no-go.
- Mechanistic bracket: R279 is too suppressive; R280 is too supportive. A follow-up should reduce support strength substantially (roughly `0.015`, not another broad search) or explicitly normalize gradient contributions.
- Twelve comparison panels completed and were synchronized locally.
