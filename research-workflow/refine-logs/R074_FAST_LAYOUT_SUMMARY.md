# R074 Fast Layout Constraint Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Setup

R074 tested whether a faster non-learning anatomy-layout repair could improve the R038/R025b bridge-like failure profile found in R072.

The diagnostic used only original validation masks for proxy selection and did not tune on clean-test-v2.

Implementation:

- script: `scripts/apply_fast_layout_bridge_cut.py`
- input proxy: `outputs/ablations/r025b_anchor_local_pixel_residual_sam_hqsam/TSRS_RSNA-Epiphysis/val/masks`
- method: detect large connected components with multiple distance peaks, then cut very low-distance internal pixels only if the result splits into valid components under a strict remove cap.

## Results

Two fixed parameter checks were run after the broader grid proved too slow:

| Setting | Changed Images | Val Dice | Precision | Recall | Boundary IoU | False Bridge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| conservative: peak `0.60`, valley `0.18`, min area `32`, max remove `0.001` | `0` | `0.894701` | `0.888486` | `0.906338` | `0.219753` | `0.104167` |
| aggressive: peak `0.45`, valley `0.35`, min area `18`, max remove `0.004` | `0` | `0.894701` | `0.888486` | `0.906338` | `0.219753` | `0.104167` |

The metric equals the anchor proxy because no images changed.

## Interpretation

R074 did not provide a usable layout repair. The bridge-like errors in R038/R025b are not simple thin-line adhesions that can be fixed by distance-transform geometry alone. This supports the R072 interpretation: layout constraints need either image evidence, learned scoring, or label/layout data intervention, not a purely mask-geometric cut.

## Decision

Do not continue:

- watershed cut grids;
- distance-transform-only bridge cutting;
- broader hand-tuned geometry postprocessing over R038.

Next ARIS step should be R075:

1. a light image-evidence layout scorer around R038 components; or
2. an isolated train/val label-layout audit variant if scorer evidence is weak.

Because R072 found only `5` strong clean-test-v2 label suspects, broad data deletion is not the first choice.
