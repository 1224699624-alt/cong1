# R179 Boundary/Gap Constrained Training Result

Date: 2026-07-03

## Purpose

R179 moved the bone-gap adhesion and boundary constraints from postprocessing into the source-model training objective. It used the existing DINOv3 ConvNeXt-tiny instance-separation chain with stronger boundary, separation-band, bridge-gap, and background penalties.

## Protocol

- Training/selection: original `TSRS_RSNA-Epiphysis` train/val
- Final success evaluation: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Control evaluation: original `TSRS_RSNA-Epiphysis/test`
- Server only: GPU0, screen `r179_boundary_gap`
- New/reannotated test: not used

## Training Outcome

R179 early-stopped at epoch 34 with best validation epoch 25.

Best validation epoch:

- val Dice: `0.877670`
- val Precision: `0.853407`
- val Recall: `0.910381`
- val Boundary IoU: `0.402608`
- val false-bridge flag: `0.114583`
- threshold: `0.65`
- separation suppression: `0.25`

## Clean-Test-v2 Result

Clean-test-v2 mean metrics:

- Dice: `0.8929766774679696`
- IoU: `0.8071421575909757`
- Precision: `0.8665400039324684`
- Recall: `0.9230288824537493`
- Boundary IoU: `0.18762334907358627`
- component-count error: `0.9382716049382716`
- false-bridge flag: `0.2962962962962963`

Compared with R110:

- Dice dropped by `-0.024746`
- Boundary IoU dropped by `-0.064273`
- component-count error improved from `2.506173` to `0.938272`
- false-bridge improved from `0.790123` to `0.296296`

## R110/R179 Complementarity

Follow-up audit artifact:

- `outputs/analysis/r180_r110_r179_complementarity_audit.json`

Key audit result:

- R179 beats R110 on only `5/81` clean-test-v2 images.
- R110 mean Dice: `0.917723`
- R179 mean Dice: `0.892977`
- per-image oracle Dice: `0.918213`
- oracle gain over R110: about `+0.000490`
- union Dice: `0.900358`
- intersection Dice: `0.910476`

## Interpretation

R179 proves that stronger training-time bridge/gap constraints can improve topology-style metrics, especially component-count consistency and false-bridge rate. However, it sacrifices too much boundary and region quality, so it is not a valid improvement over R110.

The complementarity is too small to justify an R180 fusion/readout training run: a GT-leaking per-image oracle over R110/R179 still reaches only `0.918213`, far below the target `0.931766`.

Decision: close R179 as a quality-improvement route. Keep it only as evidence that topology constraints are measurable and can be improved, but do not claim SOTA superiority from it. The next ARIS step should not be another loss-weight sweep of the same DINOv3 instance-separation source. Productive next routes are:

1. return to the R168/R170 reviewed train/val label-protocol variant once the human-reviewed CSV exists;
2. use R179's five better cases as diagnostic examples for a future non-leaking case detector, not as a fusion training target;
3. require a genuinely new model family or environment support before another heavy GPU architecture run.

