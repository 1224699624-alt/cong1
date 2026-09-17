# R248 Context-Crop Scorer Result

## Protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Evidence: original-val candidate generation and five-fold image-grouped OOF scoring.
- `clean-test-v2` used: `false`.
- Masks written: `false`.
- Baseline/anchor: R110-like `r110_r100_r108_patch_basic_trainval`.

## Raw Results

| Stage | Rows / Images | Selection | Safe / Risk | Recall Delta | Boundary IoU Delta | Boundary F1 Delta | Gap FP Delta | Cut GT FG | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R244 oracle | 8,568 / 96 | 87 images | 87 / 0 | 0 | +0.001281313 | +0.001695222 | -0.001125485 | 0 | oracle-positive |
| R245 best scalar | 8,568 / 96 | 9 images | 6 / 4 | -0.000053479 | +0.000847426 | +0.001092563 | -0.000833716 | 0.184524 | no-go |
| R248 original grid | 8,568 / 96 | 0 | 0 / 0 | NA | NA | NA | NA | NA | invalid/unreachable grid |
| R248b reachable regrid best | 8,568 / 96 | 4 images | 2 / 2 | -0.000015604 | +0.000255240 | +0.000321331 | -0.000234329 | 0.218750 | no-go |

Probability diagnostics from the OOF rows:

| Head | ROC-AUC | Average Precision | Output Range |
| --- | ---: | ---: | ---: |
| safe-useful | 0.684366 | 0.201947 | 0.000135-0.859895 |
| risk | 0.654707 | 0.933722 | 0.702877-0.999949 |

## Key Findings

1. **The original R248 `best=null` was not sufficient evidence of model failure.** Every OOF risk probability exceeded the configured maximum risk threshold of `0.50`, so all 49 configurations were mechanically empty.
2. **After correcting threshold reachability, the method still fails the safety gate.** R248b evaluated 80 configurations, 68 were nonempty, and none passed. The best four-image selection contained as many risk cuts as safe cuts and reduced Recall.
3. **Local context contains some signal but not enough for deployment.** The useful/risk AUC values are above chance, yet calibration is poor and the safe/risk tails overlap at the precision level required for bone-preserving cuts.
4. **Implication:** do not build an R248 mask editor and do not evaluate it on clean-test-v2. The next experiment should change the supervision target and/or proposal distribution, not add another scalar threshold to the same crop scores.

## Next Experiments

- R251-S0/R251: predict a dense seam/background channel using inverted original train/val bone GT inside candidate corridors; inference receives only image and anchor-derived features.
- R252 if R251 fails: constrain candidate generation by the predicted background channel so unsafe candidates are removed before ranking.
- R253 only after a strict original-val gate passes: isolated mask writing followed by full R201-style audit.

## Artifacts

- `outputs/analysis/r248_r244_context_crop_scorer_fullval.json`
- `outputs/analysis/r248_r244_context_crop_scorer_fullval_threshold_grid.csv`
- `outputs/analysis/r248_r244_context_crop_scorer_fullval_probed_rows.csv`
- `outputs/analysis/r248_r244_context_crop_scorer_fullval_regrid.json`
- `outputs/analysis/r248_r244_context_crop_scorer_fullval_regrid.csv`

