# R086 Uncertainty/Error Alignment Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R086 tested whether existing candidate-mask disagreement can localize the remaining R038 errors. This was a non-training diagnostic after R084/R085 showed that hand-written image boundary rules are not useful.

## Candidates

- Anchor: `r038_aggr_r1_t045_m000`
- Candidates:
  - `r038_aggr_r1_t045_m000`
  - `r025b_anchor_local_pixel_residual_sam_hqsam`
  - `r068_instance_separation_segmenter`
  - `r085_gradient_boundary_rule` (no-op copy of R038)

## Key Numbers

| Quantity | Value |
| --- | ---: |
| R038 anchor Dice | `0.914357` |
| Candidate disagreement pixel fraction | `0.003684` |
| R038 error captured by disagreement | `0.275269` |
| Disagreement precision for R038 error | `0.396750` |
| Candidate-fixable error share | `0.275269` |
| GT-leaking disagreement pixel oracle Dice | `0.938034` |

Boundary-band view:

| Region | Error Capture | Error Precision |
| --- | ---: | ---: |
| r1 band disagreement | `0.188710` | `0.439501` |
| r2 band disagreement | `0.233288` | `0.410902` |
| r4 band disagreement | `0.254794` | `0.399591` |
| r8 band disagreement | `0.265652` | `0.397606` |

## Interpretation

The signal is real but narrow. Disagreement covers only `0.37%` of pixels, yet captures `27.5%` of R038 error pixels. A GT-leaking edit restricted to this disagreement region reaches Dice `0.938034`, above the target by about `+0.006268`.

This reopens a narrowly scoped learnable route, but only inside candidate-disagreement pixels. It does not justify broad boundary-band editing, hand-written gradient rules, or full-mask architecture reruns.

## Decision

Proceed to R087 as a small valid learner:

- train only on candidate-disagreement/local features;
- apply only inside R038 candidate-disagreement pixels;
- compare against R038 on clean-test-v2;
- stop if it cannot beat R038 materially.

Avoid:

- whole-boundary refiner tuning;
- GrabCut/Canny/gradient rules;
- per-image selectors or simple voting.

