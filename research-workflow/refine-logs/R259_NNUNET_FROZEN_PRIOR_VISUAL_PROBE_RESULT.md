# R259 Frozen-Prior nnU-Net Visual Probe Result

## Status

Completed on 96 `TSRS_RSNA-Epiphysis` original-val cases. `clean-test-v2` and Articular-Surface were not used. Twelve frozen baseline-selected four-column full/zoom panels were generated.

## Important limitation

This was the requested short qualitative probe: a shared two-epoch warmup followed by two matched branch epochs, each with only 25 iterations/epoch. Neither branch is a trained-to-convergence nnU-Net baseline. The result therefore diagnoses direct integration behavior but cannot compare the proposed method with a mature nnU-Net.

## R201 original-val metrics

| Metric | Matched zero-prior nnU-Net | Frozen prior + loss | Delta |
|---|---:|---:|---:|
| Dice | 0.229364 | 0.054857 | -0.174507 |
| IoU | 0.132446 | 0.028409 | -0.104038 |
| Recall | 0.787351 | 0.159150 | -0.628201 |
| Boundary IoU | 0.019993 | 0.006622 | -0.013370 |
| Boundary F1 | 0.039057 | 0.013128 | -0.025929 |
| Surface Dice 2px | 0.075312 | 0.024400 | -0.050912 |
| HD95 px | 450.638007 | 502.463699 | +51.825692 |
| ASSD px | 133.555665 | 195.844792 | +62.289127 |

Both outputs are visibly immature, with salt-and-pepper foreground and severe fragmentation. The active branch suppresses foreground much more strongly and is worse on all central overlap/boundary measures. Its fixed auxiliary loss remained finite (`prior_loss 1.1282 -> 0.3179`) and prior mass was small (`0.00262 -> 0.00242`), so the pipeline executed as defined; the qualitative failure is not a NaN/crash artifact.

## Artifacts

- Raw-X-ray four-column panels: `outputs/visualizations/r259_nnunet_frozen_prior_original_val_raw_xray/`
- Matched masks: `outputs/ablations/r259_native/TSRS_RSNA-Epiphysis/val/masks/`
- Frozen-prior masks: `outputs/ablations/r259_frozen_prior/TSRS_RSNA-Epiphysis/val/masks/`
- Metrics: `outputs/analysis/r259_native_original_val_r201.json` and `outputs/analysis/r259_frozen_prior_original_val_r201.json`

No follow-up adjustment was launched, per user instruction.
