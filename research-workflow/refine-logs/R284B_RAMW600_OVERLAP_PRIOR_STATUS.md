# R284b RAM-W600 Overlap-Prior Result

- Dataset: RAM-W600 official split, 425 train / 69 validation / 124 test.
- Baseline: compact U-Net with 14 independent sigmoid instance outputs.
- Improved: paired baseline checkpoint plus a learned image-conditioned overlap-prior
  head, zero-initialized residual segmentation branch, and pairwise overlap consistency.
- Threshold fixed at `0.5`; test set was not used for model selection or threshold search.
- R284's first improved run was invalid because FP16 `log(1-p)` produced NaNs. R284b
  evaluated the predeclared mechanism after moving the sparse pair loss to FP32 and adding
  a non-finite fail-fast check.

## Official test result

- Macro Dice: `0.973944 -> 0.975982` (`+0.002037`).
- Any-overlap Dice: `0.830912 -> 0.854297` (`+0.023385`).
- Any-overlap NSD 2 px: `0.813016 -> 0.855294` (`+0.042278`).
- Validation followed the same direction: macro Dice `0.973456 -> 0.975518`, overlap
  Dice `0.834878 -> 0.858350`.

## Pair-level mechanism evidence

- Metacarpal2--Metacarpal3 overlap Dice: `0.000051 -> 0.753642`; NSD:
  `0.048387 -> 0.804432`.
- Hamate--Tri overlap Dice: `0.000060 -> 0.755256`; NSD:
  `0.137097 -> 0.759907`.
- Trapezium--Metacarpal1 overlap Dice: `0.802656 -> 0.818822`.
- Most frequent pairs improved modestly, while Capitate--Lunate overlap Dice and NSD
  showed a small negative tradeoff and require later pair-specific auditing.

## Interpretation

The result is a positive pilot: explicit overlap supervision recovers instance identity in
relations that a high global-Dice baseline completely misses, while also increasing macro
Dice. This supports replacing a universal background-seam rule with relation-conditioned
`gap` versus `overlap` supervision. It is not yet a multi-seed or multi-backbone claim.
