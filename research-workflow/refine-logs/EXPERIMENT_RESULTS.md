# R254 HA-PDSP + PCR-Loss nnU-Net Pilot Results

**Date**: 2026-07-13  
**Status**: COMPLETED_MIXED_NEGATIVE  
**Scope**: TSRS_RSNA-Epiphysis original train/val only  
**clean-test-v2**: not used

## Run design

- Backbone: nnU-Net v2 2D, initialized from the same R202 `checkpoint_best.pth`.
- Control: canonical nnU-Net loss continuation.
- Method: robust region-wise base loss + train-only age/sex conditional relation prior + PCR hard-negative loss.
- Fairness: both control and method used seed 254, 15 epochs, 30 train iterations/epoch, 20 validation iterations/epoch, LR 1e-3 and identical original train/val split.
- Prior: 875 original-train cases, 29,266 adjacent instance-pair samples; metadata SHA256 `bfb40b05d22a6b2151af3982b9de2e0dd2534b21c661df79037f33247d599857`.
- Evaluation: frozen R201 metric definitions applied to all 96 original-val masks.
- This is a qualitative/mechanism pilot, not a final multi-seed result.

## Original-val results

| Metric | Native continuation | HA-PDSP + PCR | Delta | Direction |
|---|---:|---:|---:|---|
| Dice | 0.897112 | 0.893513 | -0.003599 | worse |
| IoU | 0.819215 | 0.815056 | -0.004160 | worse |
| Precision | 0.886486 | 0.868809 | -0.017678 | worse |
| Recall | 0.916238 | 0.930436 | +0.014197 | better |
| Boundary IoU | 0.239893 | 0.233664 | -0.006229 | worse |
| Boundary F1 | 0.379017 | 0.370996 | -0.008021 | worse |
| Surface Dice 2 px | 0.654511 | 0.642113 | -0.012398 | worse |
| Surface Dice 5 px | 0.896460 | 0.890069 | -0.006391 | worse |
| HD95 (px) | 54.341353 | 40.565352 | -13.776000 | better |
| ASSD (px) | 14.173772 | 12.483078 | -1.690694 | better |
| gap-region FP rate | 0.205547 | 0.246738 | +0.041191 | worse |
| component merge rate | 0.572917 | 0.583333 | +0.010417 | worse |
| component-count MAE | 5.666667 | 4.395833 | -1.270833 | better |
| component delta mean | 2.333333 | 0.979167 | -1.354167 | closer to zero |

## Training mechanism

| Epoch | Anchor foreground probability | Bone-core probability | Valid pairs/batch | Mean k | Gradient cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.838612 | 0.805016 | 20.63 | 0.201729 | 0.014118 |
| 1 | 0.390179 | 0.938342 | 20.98 | 0.308836 | 0.051343 |
| 2 | 0.087312 | 0.956681 | 21.08 | 0.365700 | 0.052115 |
| 7 | 0.070145 | 0.969028 | 21.10 | 0.371687 | 0.048024 |
| 14 | 0.050752 | 0.961308 | 21.62 | 0.349731 | 0.035753 |

The optimization mechanism activated and remained numerically stable. It strongly reduced model confidence on selected legal background anchors while preserving high foreground-core confidence. However, the selected training anchors did not translate to lower validation gap-region FP.

## Interpretation

This pilot is mixed but does not pass the method gate.

Positive:

- HD95 and ASSD improved materially.
- Component-count MAE and signed component-count error improved.
- Recall increased.
- The prior/pair pipeline had non-zero coverage and stable gradients.

Negative:

- Dice, IoU, precision, Boundary IoU/F1 and Surface Dice decreased.
- Gap-region FP and merge rate worsened, which directly contradicts the intended bone-gap claim.
- Visual inspection shows slightly fuller/thicker masks in several cases, consistent with increased recall and reduced precision.

Most likely cause:

The robust base loss separately normalizes foreground-core and far-background BCE with equal top-level weights. For a small-foreground task this over-rewards foreground recovery and can expand boundaries. The PCR term successfully suppresses its selected anchors, but those anchors are too sparse/location-imprecise to counter the global foreground-expansion pressure. The current method therefore improves outlier surface/component behavior without specifically solving narrow-gap adhesion.

## Decision

**NO-GO for clean-test-v2 and NO-GO for cross-backbone transfer in the current form.**

No threshold search, model selection, or evaluation was performed on clean-test-v2.

Before another run, the next revision should:

1. use canonical nnU-Net Dice/CE as the main base and add only conservative boundary-noise attenuation, or substantially reduce normalized foreground-core BCE;
2. restrict pair activation to same-predicted-component suspected merges or explicit local bridge evidence;
3. use a local boundary-to-boundary corridor rather than generic adjacent support pairing;
4. run robust-base-only and uniform-pair ablations before attributing changes to the age/sex prior.

## Artifacts

- Metrics: `outputs/analysis/r254_nnunet_baseline_original_val_r201.json`
- Metrics: `outputs/analysis/r254_hapdsp_pcr_nnunet_original_val_r201.json`
- Dynamics: `outputs/nnunet/r254_hapdsp_pcr/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerHAPDSPPCR__nnUNetPlans__2d/fold_0/r254_training_dynamics.jsonl`
- Comparisons: `outputs/visualizations/r254_hapdsp_pcr_original_val/`
- Baseline masks: `outputs/ablations/r254_nnunet_baseline/TSRS_RSNA-Epiphysis/val/masks/`
- Method masks: `outputs/ablations/r254_hapdsp_pcr_nnunet/TSRS_RSNA-Epiphysis/val/masks/`


