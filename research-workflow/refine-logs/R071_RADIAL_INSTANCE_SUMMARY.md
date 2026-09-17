# R071 Radial Instance Segmenter Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Setup

R071 tested a StarDist-inspired shape representation after R065-R070 showed that direct binary masks, HoVer-style separation, trimming, and local pixel arbitration were all insufficient.

- Model: U-Net/context encoder with binary mask, center heatmap, radial distance, and boundary heads.
- Decode: select center peaks, generate radial polygons, mix shape mask with mask probability.
- Validation grid: narrowed from an initial 140-config grid to 12 configs after speed diagnosis.
- Evaluation split: `TSRS_RSNA-Epiphysis_clean_test_v2/test` only for success.
- GPU policy: one free GPU, GPU0.

## Result

| System | Dice | Precision | Recall | Boundary IoU | False Bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| R038 anchor | `0.914357` | `0.899054` | `0.931483` | `0.243176` | `0.296296` |
| R070 local arbitration | `0.914370` | `0.897204` | `0.933503` | `0.244320` | - |
| R071 radial instance | `0.893600` | `0.867336` | `0.923358` | `0.181359` | `0.604938` |
| Target | `>0.931766` | - | - | - | - |

Control original-test Dice: `0.878248`.

Best validation epoch:

- epoch `22`
- val Dice `0.886285`
- val Precision `0.869643`
- val Recall `0.909862`
- val Boundary IoU `0.503501`
- selected threshold `0.55`, center threshold `0.35`, shape weight `0.4`

## Interpretation

R071 confirms that a center/radial instance representation is learnable and improves component-count error relative to the early direct segmenter attempts, but it does not beat the strong YOLO+SAM/refiner anchor. The method under-covers boundary/detail quality and still has a high false-bridge flag rate on clean-test-v2.

The key negative signal is that R071's representation change does not recover the `+0.017409` Dice margin needed from R038. Continuing with another direct image-to-mask architecture is therefore low-value unless it introduces a substantially stronger prior or uses a different data signal.

## Decision

Do not continue:

- another direct context segmenter;
- another HoVer/radial standalone segmenter;
- R038/R068/R069/R071 selector or fusion variants.

Next ARIS step should be a non-GPU R072 data/layout audit on clean-test-v2 and train/val mismatch, then a decision between:

1. isolated data/label manifest intervention; or
2. explicit anatomy-layout constraints around ordered epiphysis instances.
