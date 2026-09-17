# R078 Slot-Layout Segmenter Summary

Date: 2026-06-26

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R078 tested whether an explicit anatomical slot representation could solve the bridge problem that defeated R065 context, R068 HoVer-style separation, R071 radial shapes, and R077 tiny label filtering.

The model predicted fixed spatial slots and blended them into the final foreground mask. This was intended to make the anatomical layout explicit instead of relying on generic binary decoding.

## Result

| Split | Dice | Precision | Recall | Boundary IoU | False Bridge |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean-test-v2 | `0.868626` | `0.784847` | `0.974682` | `0.125054` | `0.864198` |
| original test | `0.849879` | `0.762013` | `0.964811` | `0.119867` | `0.814433` |

Best usable anchor remains R038/R070 level:

- R038 clean-test-v2 Dice `0.914357`
- R070 clean-test-v2 Dice `0.914370`
- target Dice `> 0.9317660066557425`

## Interpretation

R078 is a clear negative result. It recovers foreground aggressively, but the precision and boundary metrics collapse. The high recall is not useful because it comes with severe overmasking and adjacent-bone bridging.

This is the same failure mode seen in prior direct segmenters, amplified by slot blending:

- too much foreground recall;
- too little boundary selectivity;
- too many bridges;
- no evidence that another slot threshold/grid will close the `+0.0174` Dice gap.

## Decision

Close the R078 slot-layout route. Do not continue by tuning slot count, slot blend, or full-mask thresholds.

Next ARIS direction should avoid another whole-image direct foreground decoder. A better R079 candidate is a local boundary/contour correction formulation around the strong R038 mask, with an explicit gate that refuses to train if a non-leaking diagnostic cannot show enough clean-test-v2 upside.

