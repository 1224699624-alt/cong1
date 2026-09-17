# R312 clean paired U-Net prior status

Date: 2026-07-27

## Question

Does the same frozen R259 relation-prior input and fixed background-only prior loss used by the nnU-Net branch improve a compact U-Net on the manually reviewed TSRS keep-list?

## Protocol

- Dataset: `TSRS_RSNA-Epiphysis` only.
- Selection: 814 clean train / 94 clean original-val cases.
- Backbone: identical compact U-Net in both arms, base width 16, input 512 x 512.
- Shared initialization: exact same two-channel state; prior-channel weights start at zero.
- Control: X-ray plus an all-zero prior channel; native Dice/BCE only.
- Improved: X-ray plus frozen R259 prior; native Dice/BCE plus fixed `0.05` background-only prior suppression.
- Optimizer, seed, augmentation, threshold, early stopping and validation cases are matched.
- Model selection: validation Dice only; no gap-metric or threshold search.
- Final masks are restored to original image resolution and evaluated with R201.
- Test and clean-test-v2: unused.

## Prior rebuilding decision

R259 is a learned prior, not a purely deterministic hand-crafted map. A fully clean final pipeline should therefore refit it on the retained 814 training cases and rerun both backbones. R312 deliberately reuses the existing frozen R259 prior because the current purpose is a controlled backbone comparison against R308, which uses that exact prior. Rebuilding the prior for U-Net alone would destroy cross-backbone comparability.

## Runtime

- Environment: local `YOLOSAM`, PyTorch 2.5.1+cu118.
- GPU: RTX 3060 Laptop 6 GiB.
- Launcher: `run_r312_clean_paired_unet_prior_local.ps1`.
- Initial launch found zero-byte placeholder image aliases for `val/10520`; the image resolver was corrected to skip empty/corrupt aliases and select the valid `.jpg`. No training result from the failed launch was retained.
- Full preflight verified all 908 selected X-rays, labels and frozen prior maps.
- Completed on 2026-07-27; no active training process remains.
- Active-run log: `outputs/bridge_logs/r312_clean_paired_unet_prior_local_retry.log`.
- Both arms selected epoch 24 and early-stopped after 34 epochs.
- Training-selection Dice: plain `0.894596154`, prior `0.897989540`.

## Original-resolution clean-val94 R201 result

| Metric | Plain U-Net | Prior U-Net | Delta |
|---|---:|---:|---:|
| Dice | 0.873988 | 0.876396 | +0.002408 |
| IoU | 0.779689 | 0.785116 | +0.005426 |
| Precision | 0.830659 | 0.843272 | +0.012612 |
| Recall | 0.926861 | 0.917264 | -0.009597 |
| Boundary IoU | 0.163191 | 0.175779 | +0.012588 |
| Boundary F1 | 0.274945 | 0.293202 | +0.018257 |
| gap-region FP rate | 0.382389 | 0.340587 | -0.041802 |
| component merge rate | 0.787234 | 0.659574 | -0.127660 |
| component count MAE | 3.606383 | 2.617021 | -0.989362 |
| Surface Dice 2 px | 0.500963 | 0.531449 | +0.030486 |
| Surface Dice 5 px | 0.816254 | 0.842915 | +0.026661 |
| HD95 px | 14.960767 | 12.009965 | -2.950802 |
| ASSD px | 4.146656 | 3.934252 | -0.212404 |

## Interpretation

- The prior improves Dice on 77/94 cases and Boundary IoU/F1 on 85/94 cases.
- Gap-region FP improves on 89/94 cases.
- Component merge state improves on 14 cases, worsens on two, and is unchanged on 78.
- Recall decreases on 77/94 cases, confirming that foreground suppression remains the main trade-off.
- Unlike the clean nnU-Net pair, the U-Net gains are large enough to improve Dice, IoU, all boundary/surface metrics, gap FP and merge metrics simultaneously. This supports a real backbone-strength interaction: the same prior is more beneficial to the weaker U-Net than to mature nnU-Net.

Artifacts:

- `outputs/analysis/r312_clean_plain_unet_val94_r201.json`
- `outputs/analysis/r312_clean_prior_unet_val94_r201.json`
- `outputs/visualizations/r312_clean_plain_vs_prior_unet_val94/`
