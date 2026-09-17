# R279 R260-Style Instance-Prior nnU-Net

## Question

Does replacing R260's coarse relation heatmap with a frozen, instance-supervised seam probability map improve separation while retaining the same mature nnU-Net joint-input/loss recipe?

## Protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Development split: 875 train / 96 original-val.
- `clean-test-v2`: locked and unused.
- Input: X-ray plus a frozen seam-probability channel inferred by the R276 three-class model.
- Instance palette labels supervise the frozen seam model only; original-val inputs are generated without reading the corresponding GT.
- Initialization: mature R275 control checkpoint, SHA256 `557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27`.
- Two-channel expansion: all X-ray and decoder/head tensors inherited exactly; the new prior channel is initialized to zero.
- Loss: native nnU-Net Dice/CE plus R260's fixed `0.05` background-only prior suppression.
- No core-retention term or additional loss-weight search in this first controlled probe.
- Model selection: validation-best checkpoint with the R260 early-stopping rule.
- Final development audit: R201 metrics and fixed 12-case original-val visualization against mature R275 control.

## Launch status (2026-07-21)

- Frozen priors exported and verified: 875 train + 96 original-val; no empty maps; `clean_test_used=false`.
- Mean seam probability: `0.002034623`.
- Mean nonzero fraction: `0.004911731`.
- Checkpoint adaptation passed: 444/448 tensors unchanged; four first-layer aliases expanded; prior-channel nonzero count is zero.
- Remote screen: `r279_instance_prior`.
- GPU: RTX 4090D, GPU 0.
- Training began at `2026-07-21 17:28:51 +08:00` with the exact 875/96 split.

## Decision rule

Compare with both R275 mature control and historical R260. The primary evidence is gap-region FP, component merge rate, Boundary IoU/F1, Surface Dice, HD95/ASSD, and Recall. Dice alone is not the gate.

## Final original-val result (2026-07-21)

- Training stopped after 15 epochs; validation-best checkpoint was produced after epoch 2. The best EMA pseudo Dice was reached at the beginning of fine-tuning and later epochs degraded, so this is not evidence of insufficient training.
- R275 control -> R279 instance prior:
  - Dice: `0.897965 -> 0.892006` (`-0.005960`).
  - Recall: `0.899057 -> 0.886296` (`-0.012762`).
  - Precision: `0.902028 -> 0.907066` (`+0.005039`).
  - gap-region FP: `0.187305 -> 0.175365` (`-0.011941`).
  - component merge rate: `0.510417 -> 0.468750` (`-0.041667`).
  - component count MAE: `2.333333 -> 2.000000` (`-0.333333`).
  - Boundary IoU: `0.234335 -> 0.232728` (`-0.001607`).
  - Boundary F1: `0.373105 -> 0.370806` (`-0.002299`).
  - Surface Dice 2px: `0.650056 -> 0.646279` (`-0.003777`).
  - Surface Dice 5px: `0.900422 -> 0.899551` (`-0.000871`).
  - HD95: `16.350589 -> 20.143470` (`+3.792882`, worse).
  - ASSD: `6.165032 -> 5.209741` (`-0.955291`, better).
- Interpretation: the sparse instance-seam prior is genuinely changing the intended separation behavior, but the unchanged R260 suppression loss is too one-sided. It lowers gap false positives and merges by making foreground more conservative, at the cost of Recall and several boundary/surface metrics.
- Gate: mixed result / no-go as the final method. Retain the instance-derived prior, but do not retain this exact one-sided loss unchanged.
- Twelve fixed-case comparison panels were completed and synchronized to `outputs/visualizations/r279_instance_prior_original_val`.
