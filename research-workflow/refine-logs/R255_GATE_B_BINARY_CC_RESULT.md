# R255 Gate B Binary-CC Probe Result

Date: 2026-07-14  
Scope: TSRS_RSNA-Epiphysis original train/val only  
Decision: **NO-GO**  
Clean-test-v2 used: **No**

The 2+2 epoch matched sanity completed successfully, but it did not improve the target close-gap failure mode. The run is an engineering probe using augmented binary connected components, not final instance-aware R255 evidence.

## Training integrity

- Identical full network/optimizer/AMP checkpoint fork at epoch 2.
- Frozen close-loss alpha: `0.0004174519`.
- Valid pairs: `2.5/batch`; active anchors: `180.7/batch`.
- Anchor GT foreground fraction: `0`.
- No NaN or OOM.

## Original-val result

| Metric | Matched control | R255 | Delta |
|---|---:|---:|---:|
| Dice | 0.844949 | 0.845834 | +0.000885 |
| Recall | 0.785886 | 0.787020 | +0.001133 |
| Boundary IoU | 0.194546 | 0.194814 | +0.000267 |
| Boundary F1 | 0.313150 | 0.313556 | +0.000406 |
| Surface Dice 2px | 0.554807 | 0.555444 | +0.000637 |
| HD95 (px) | 21.093390 | 21.033375 | -0.060015 |
| ASSD (px) | 6.317511 | 6.297420 | -0.020090 |
| Global gap FP | 0.082632 | 0.082795 | +0.000163 |
| 1–4px gap FP | 0.121521 | 0.121550 | +0.000029 |
| 1–4px pair merge | 0.103139 | 0.103139 | 0 |

## Interpretation

The loss slightly improved overlap, recall, boundary and surface-distance metrics, and reduced false splitting, but it did not separate the close metacarpal/carpal pairs. The foreground-area ratio increased from `0.842983` to `0.844196`, consistent with the small gap-FP increase. Therefore this branch must not be promoted to longer training or clean-test evaluation.

Next iteration should change the supervision mechanism rather than merely increase epochs or alpha: synchronized instance sidecars, a true directional two-edge profile, and bottleneck-centered confidence suppression are the relevant unresolved components.
