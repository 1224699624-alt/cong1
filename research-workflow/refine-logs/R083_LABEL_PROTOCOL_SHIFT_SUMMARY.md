# R083 Label Protocol Shift Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R083 tested whether the clean-test-v2 target split looks like a different label protocol or morphology distribution from the original train/val/test labels. This matters because many learned refiners improve validation-like behavior but do not transfer to clean-test-v2.

## Compared Splits

| Split | Images | FG Frac | Component Count | Perimeter/Area Median | Compactness Median | Layout Width | Layout Height |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `875` | `0.029734` | `23.432` | `0.122521` | `16.213179` | `0.578487` | `0.583675` |
| val | `96` | `0.026533` | `23.229` | `0.120999` | `15.866382` | `0.549740` | `0.554250` |
| original test | `97` | `0.028891` | `23.330` | `0.128420` | `16.493703` | `0.580132` | `0.585846` |
| clean-test-v2 | `81` | `0.030819` | `23.568` | `0.124637` | `16.785640` | `0.585330` | `0.591348` |

Largest clean-test-v2 standardized shifts versus train:

| Feature | Std Effect |
| --- | ---: |
| compactness_median | `+0.224` |
| aspect_median | `+0.188` |
| component_area_max | `+0.151` |
| layout_height | `+0.081` |
| fg_frac | `+0.079` |

## Interpretation

There is no strong label-protocol shift signal. Clean-test-v2 is slightly larger/more compact in aggregate, but all major morphology statistics are close to train/original-test. The repeated validation-to-clean-test failure is therefore unlikely to be explained by a simple label protocol mismatch.

This also supports the R081 conclusion: the original validation proxy is somewhat weaker/noisier, but not so different that the clean-test-v2 target should be considered a separate annotation regime.

## Decision

Do not spend the next run on broad test-protocol correction or dataset relabel assumptions.

The next ARIS step should use a genuinely new image-derived signal rather than another anchor-threshold/refiner tune:

1. edge/contour-aware supervision with image gradients or learned boundary evidence;
2. test-time consistency/uncertainty that does not rely on original-val selecting a brittle boundary editor;
3. a stronger literature-driven contour architecture if a short literature pass identifies a suitable candidate.

