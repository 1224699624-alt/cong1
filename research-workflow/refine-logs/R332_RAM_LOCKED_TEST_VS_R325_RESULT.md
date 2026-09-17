# R332 RAM Locked Test vs R325 Result

## Protocol

- Dataset: RAM-W600 test, 124 images
- Baseline: R325 `plain_native_best.pth`
- Improved: R332 three-step-trained model, validation-selected epoch 30
- Locked inference depth: two corrections, selected on validation
- Threshold: 0.5, no search
- Resolution: native pixels with right/bottom padding only
- Pair definition: top 15 overlap pairs selected from train only
- Test was used once for this final locked comparison, not for tuning

## Main comparison

| Metric | R325 nnU-Net | R332 step2 | Delta | Favorable |
|---|---:|---:|---:|---|
| Overall DSC ↑ | 0.979362 | 0.979884 | +0.000523 | yes |
| Overall IoU ↑ | 0.960003 | 0.961000 | +0.000998 | yes |
| Overall NSD@2px ↑ | 0.888434 | 0.894604 | +0.006170 | yes |
| Overall MSD px ↓ | 1.307574 | 1.130006 | -0.177568 | yes |
| Overall RAVD ↓ | 0.019481 | 0.019258 | -0.000223 | yes |
| Overlap DSC ↑ | 0.870637 | 0.874531 | +0.003895 | yes |
| Overlap IoU ↑ | 0.772134 | 0.778223 | +0.006089 | yes |
| Overlap NSD@2px ↑ | 0.780356 | 0.789704 | +0.009349 | yes |
| Overlap MSD px ↓ | 1.909904 | 1.804907 | -0.104997 | yes |
| Overlap RAVD ↓ | 0.058705 | 0.059987 | +0.001282 | no |
| Pair DSC ↑ | 0.828841 | 0.831425 | +0.002584 | yes |
| Pair IoU ↑ | 0.733197 | 0.738069 | +0.004872 | yes |
| Pair NSD@2px ↑ | 0.776738 | 0.784847 | +0.008109 | yes |
| Pair MSD px ↓ | 1.948267 | 1.691503 | -0.256764 | yes |
| Pair RAVD ↓ | 0.351319 | 0.330240 | -0.021079 | yes |
| Pair MSD fail rate ↓ | 0.003846 | 0.007692 | +0.003846 | no |

## Iterative diagnostic on test

| Stage | Overall DSC ↑ | Overlap DSC ↑ | Overlap NSD ↑ | Overlap MSD ↓ |
|---|---:|---:|---:|---:|
| R332 step0 | 0.979380 | 0.870062 | 0.779859 | 1.916416 |
| R332 step1 | 0.979739 | 0.873158 | 0.786822 | 1.821786 |
| R332 step2 | 0.979884 | 0.874531 | 0.789704 | 1.804907 |

The locked second correction continues to improve all four primary diagnostic
metrics over the first correction. The jointly fine-tuned step0 alone does not
explain the overlap improvement; most of the gain appears after applying the
iterative prior refiner.

## Interpretation

R332 satisfies the main requirement on this locked test: overall DSC and IoU
both exceed the original nnU-Net, while surface and overlap metrics improve by
larger margins. The main adverse results are a small overlap RAVD increase and
a pair MSD fail-rate increase from 0.38% to 0.77%. The latter is a low absolute
rate but must be retained as a limitation rather than hidden.

These are single-run point estimates. Statistical significance and confidence
intervals require a paired per-case analysis or seed-level test evaluation and
are not claimed here.
