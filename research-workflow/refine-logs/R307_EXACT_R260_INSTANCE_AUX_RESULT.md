# R307 exact R260 binary-vs-instance auxiliary result

Date: 2026-07-24

## Protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Split: 875 train / 96 original-val.
- `clean-test-v2`: not used.
- Common initialization: recovered R260 best checkpoint.
- R260 SHA256: `7d1d8d2916115c2e29696b2a76881da2a7a78f3db91a5f6a9491297369052a6e`.
- Both R307 arms loaded all network keys strictly and used seed 260.
- Binary continue: R260 prior loss 0.05, no instance auxiliary loss.
- Instance auxiliary: R260 prior loss 0.05 plus color-instance-derived seam loss 0.05; output remains binary bone/background.
- Evaluation: R201 definitions on the same 96 original-val images.

## Training outcome

- Binary continue: stopped after 53 epochs; best EMA pseudo Dice `0.9184734667`.
- Instance auxiliary: stopped after 29 epochs; best EMA pseudo Dice `0.9131682973`.
- R306 compatibility attempts did not produce a model result and are not counted as negative experiments.

## R201 results

| Metric | Frozen R260 | R307 binary continue | R307 instance auxiliary | Aux - binary |
|---|---:|---:|---:|---:|
| Dice | 0.902759 | 0.903133 | 0.900397 | -0.002736 |
| IoU | 0.826196 | 0.826765 | 0.822110 | -0.004655 |
| Precision | 0.911294 | 0.911598 | 0.916674 | +0.005076 |
| Recall | 0.899632 | 0.899933 | 0.889593 | -0.010340 |
| Boundary IoU | 0.244642 | 0.244544 | 0.242658 | -0.001886 |
| Boundary F1 | 0.385938 | 0.385791 | 0.383494 | -0.002297 |
| Gap-region FP rate (lower better) | 0.175200 | 0.175323 | 0.157510 | -0.017813 |
| Component merge rate (lower better) | 0.500000 | 0.500000 | 0.333333 | -0.166667 |
| Component count MAE (lower better) | 1.906250 | 1.843750 | 1.864583 | +0.020833 |
| Surface Dice 2 px | 0.667281 | 0.667438 | 0.663747 | -0.003691 |
| Surface Dice 5 px | 0.912211 | 0.912994 | 0.912313 | -0.000680 |
| HD95 px (lower better) | 15.864084 | 13.106486 | 16.371809 | +3.265323 |
| ASSD px (lower better) | 4.320395 | 4.211323 | 4.398018 | +0.186695 |

## Interpretation

The recovered R260 checkpoint was valid: plain binary continuation slightly improved Dice, Recall, Surface Dice 5 px, HD95, and ASSD relative to frozen R260. This rules out an invalid or immature recovered checkpoint as the explanation.

Color-instance supervision contains useful separation information. Relative to binary continuation, it reduced gap-region foreground false positives by `0.017813` absolute and reduced component merge rate from `0.500000` to `0.333333`. Precision also rose by `0.005076`.

However, the current auxiliary seam loss is too aggressive or too broadly applied. Recall fell by `0.010340`, Dice by `0.002736`, and Surface Dice 2 px by `0.003691`. It improved gap FP on 92/96 images and removed a merge flag on 16/96 images, but Recall fell on 91/96 images. This is a systematic foreground under-segmentation tradeoff, not random noise.

HD95 degradation is strongly affected by failure cases. Image `1518.png` changed from `6.403 px` to `260.368 px`, while ASSD changed from `14.792 px` to `31.856 px`. The next revision must inspect this case and gate the instance seam term away from missing/immature or low-confidence structures.

## Decision

R307 instance auxiliary is **no-go as the final configuration**, despite proving that color-instance labels provide a real separation signal. The next design should retain the instance-derived seam target but gate it spatially and by confidence, and protect high-confidence bone support on both sides. A global reduction of all foreground probability is not acceptable.

## Artifacts

- `outputs/analysis/r307_exact_r260_binary_continue_original_val_r201.json`
- `outputs/analysis/r307_exact_r260_instance_aux_original_val_r201.json`
- `outputs/visualizations/r307_exact_r260_original_vs_instance_aux/`
- `outputs/bridge_logs/r307_exact_r260_instance_aux_resume.log`
