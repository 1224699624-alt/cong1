# R210-F1 Full Val Result

Date: 2026-07-07

## Purpose

Run R210-F1 on the full original validation split after the val32 subset showed promising component/gap improvements. This remains train/val-only evidence. clean-test-v2 was not used.

## Protocol

- Dataset: `TSRS_RSNA-Epiphysis`
- Split: original `val`
- Anchor: `r110_r100_r108_patch_basic_trainval`
- Output experiment: `r210_f1_neck_candidate_gate_val_fastinner`
- Candidate setting: `dist_percentile=10`
- Candidate caps:
  - `max_candidates_generated=16`
  - `max_components_per_image=5`
- clean-test-v2 used: `false`
- R201 metric implementation used for final per-image metrics.

Artifacts:

- `outputs/analysis/r210_f1_neck_candidate_gate_val_fastinner_summary.json`
- `outputs/analysis/r210_f1_neck_candidate_gate_val_fastinner_per_image.csv`
- `outputs/bridge_logs/r210_f1_neck_candidate_gate_val_fastinner.log`

## Full Val Summary

Validation gate: `true`

| Metric | R110 anchor | R210-F1 | Delta |
|---|---:|---:|---:|
| Dice | 0.895723 | 0.895543 | -0.000179 |
| IoU | 0.814783 | 0.814481 | -0.000302 |
| Precision | 0.904670 | 0.905331 | +0.000661 |
| Recall | 0.892619 | 0.891627 | -0.000993 |
| Boundary IoU | 0.222945 | 0.222736 | -0.000209 |
| Boundary F1 | 0.358058 | 0.357786 | -0.000272 |
| Surface Dice 2px | 0.631388 | 0.631155 | -0.000233 |
| Surface Dice 5px | 0.895343 | 0.895572 | +0.000230 |
| HD95 px | 17.996801 | 17.995477 | -0.001324 |
| ASSD px | 4.957877 | 4.956597 | -0.001280 |
| gap-region FP rate | 0.188199 | 0.186576 | -0.001623 |
| component merge rate | 0.343750 | 0.187500 | -0.156250 |
| component count MAE | 3.020833 | 2.843750 | -0.177083 |

Other diagnostics:

- Accepted images: `77/96` (`0.802083`)
- Mean accepted cut pixels: `95.927083`
- Mean candidate count: `12.625`
- Mean candidates tried: `4.510417`

## Interpretation

R210-F1 satisfies the validation safety gate:

- Dice, IoU, and Recall drops are well within tolerance.
- gap-region FP rate improves.
- component merge rate improves substantially.
- component count MAE improves.
- Precision improves slightly.
- Surface Dice 5px, HD95, and ASSD improve slightly.

But the result is not clean enough to claim a full boundary improvement:

- Boundary IoU is slightly worse.
- Boundary F1 is slightly worse.
- Surface Dice 2px is slightly worse.

This is a meaningful anatomy-consistency improvement, not yet a complete boundary-quality improvement.

## Best Positive Cases For Visualization

These cases show stronger anatomy/gap/component improvements and are candidates for hard-case figures:

| Image | Cut px | Count MAE Δ | Merge Δ | Gap FP Δ | Dice Δ | Boundary IoU Δ |
|---|---:|---:|---:|---:|---:|---:|
| `3840.png` | 326 | -5 | 0 | -0.010135 | +0.001191 | +0.006945 |
| `3700.png` | 229 | -2 | 0 | -0.004118 | +0.000242 | +0.000396 |
| `15040.png` | 243 | -2 | -1 | -0.003355 | -0.000483 | -0.001441 |
| `3354.png` | 200 | -1 | -1 | -0.006400 | +0.000579 | +0.004665 |
| `11852.png` | 322 | 0 | 0 | -0.008860 | +0.000426 | +0.004884 |
| `6606.png` | 334 | 0 | -1 | -0.007169 | +0.000823 | +0.000015 |

## Risk Cases For Visualization

These cases remain within gate tolerance but should be checked for erosion/under-segmentation:

| Image | Cut px | Dice Δ | Recall Δ | Count MAE Δ | Gap FP Δ |
|---|---:|---:|---:|---:|---:|
| `1560.png` | 304 | -0.001898 | -0.003706 | -2 | -0.000462 |
| `13480.png` | 162 | -0.001668 | -0.003106 | 0 | -0.000215 |
| `3366.png` | 72 | -0.001563 | -0.002816 | 0 | -0.000323 |
| `3567.png` | 136 | -0.001369 | -0.003132 | 0 | -0.001457 |
| `3777.png` | 175 | -0.001347 | -0.002547 | 0 | -0.000209 |
| `1884.png` | 62 | -0.001270 | -0.002577 | 0 | -0.001599 |

## Decision

R210-F1 should proceed to hard-case visualization on original val.

Do not apply clean-test-v2 yet.

Clean-test-v2 can only be considered after:

1. Hard-case overlays confirm reduced bone-gap adhesion without visible erosion or missing epiphyses.
2. Risk cases above do not show clinically meaningful under-segmentation.
3. A locked final config is documented.

If hard-case visualization passes, R210-F1 can be applied once to clean-test-v2 and evaluated under R201.
