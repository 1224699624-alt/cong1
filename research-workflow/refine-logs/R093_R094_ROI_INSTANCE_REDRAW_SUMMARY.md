# R093/R094 ROI Instance Redraw Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R092 showed that the R038/R089/R090/R091 disagreement-postprocessing family no longer has enough oracle headroom. R093 tested a genuinely different candidate source: an ROI proposal instance segmenter that redraws each proposed epiphysis instance from image evidence inside a box.

## Setup

- Script: `scripts/train_roi_instance_segmenter.py`
- Train/tune dataset: `TSRS_RSNA-Epiphysis` train/val
- Train/tune proposals: `r025b_anchor_local_pixel_residual_sam_hqsam`
- Final success split: `TSRS_RSNA-Epiphysis_clean_test_v2/test`
- Final proposals: `r038_aggr_r1_t045_m000`
- Original-test control proposals: `allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam`
- Output experiment: `r093_roi_instance_segmenter`

R093 trained from GT instance boxes and selected threshold/blend mode on original val. It did not use clean-test-v2 for tuning.

## R093 Result

| Split | Dice | IoU | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean-test-v2 | `0.903906` | `0.826004` | `0.911532` | `0.898393` | `0.248707` |
| original-test control | `0.885693` | `0.797775` | `0.910980` | `0.865380` | `0.219268` |

Best validation checkpoint:

- epoch `6`
- threshold `0.45`
- blend mode `intersect`
- val Dice `0.887288`
- val Precision `0.900633`
- val Recall `0.879332`

Compared with R090 clean-test-v2 Dice `0.916440`, R093 is lower by `0.012534`.

## R094 Complementarity Audit

Candidates:

- R038: `r038_aggr_r1_t045_m000`
- R090: `r090_online_patch_disagreement_arbitrator`
- R093: `r093_roi_instance_segmenter`

| System / Oracle | Dice |
| --- | ---: |
| R038 | `0.914357` |
| R090 | `0.916440` |
| R093 | `0.903906` |
| vote_majority | `0.915563` |
| per-image oracle | `0.917260` |
| GT-leaking pixel oracle | `0.924476` |

Disagreement diagnostics:

- mean candidate disagreement fraction: `0.002158`
- mean R038 error captured by disagreement: `0.128342`

## Interpretation

R093 generated a new mask distribution, but it was too recall-limited. The selected `intersect` readout preserved precision while removing too much true foreground. More importantly, R094 shows that R093 does not provide enough complementary pixels: even a GT-leaking pixel oracle over R038/R090/R093 reaches only `0.924476`, still below the target by about `0.007290`.

## Decision

Close the ROI proposal-redraw branch.

Do not continue:

- threshold/blend sweeps for R093;
- a fusion learner over R038/R090/R093;
- larger ROI U-Net variants using the same proposal/intersect formulation.

Next ARIS direction should use a genuinely different image model or supervision source, for example a contour-aware foundation-model candidate, stronger domain-pretrained encoder, or a protocol that directly improves recall without inheriting R038's proposal omissions.
