# R092 Candidate Combination Headroom Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R092 tested whether the current candidate-disagreement family still has enough headroom after R089/R090/R091. This was a non-training diagnostic over clean-test-v2 candidate masks.

## Candidates

- R038: `r038_aggr_r1_t045_m000`
- R089: `r089_disagreement_precision_local_stats_mlp`
- R090: `r090_online_patch_disagreement_arbitrator`
- R091: `r091_online_patch_disagreement_arbitrator_3crop`

## Result

| System / Oracle | Dice |
| --- | ---: |
| R038 | `0.914357` |
| R089 | `0.915467` |
| R090 | `0.916440` |
| R091 | `0.915554` |
| union_all | `0.914557` |
| intersection_all | `0.916659` |
| vote_majority | `0.915026` |
| vote_ge_3 | `0.915583` |
| per-image oracle | `0.917061` |
| GT-leaking pixel oracle | `0.921790` |

Disagreement diagnostics:

| Quantity | Value |
| --- | ---: |
| Mean candidate disagreement fraction | `0.000679` |
| Mean R038 error fraction | `0.005352` |
| Mean R038 error captured by disagreement | `0.093729` |

## Interpretation

This is a path block. The strongest non-leaking combination, `intersection_all`, reaches only `0.916659`, barely above R090. Even the GT-leaking pixel oracle across these candidates reaches only `0.921790`, which is below the target by about `0.009976`.

The key reason is candidate collapse: R089/R090/R091 are all conservative descendants of R038/R068 disagreement editing. Their remaining disagreement covers only `0.0679%` of pixels and only `9.37%` of R038 errors, much lower than the broader R086 diagnostic signal.

## Decision

Close the current candidate-disagreement postprocessing branch.

Do not continue:

- more MLP threshold sweeps;
- heavier streamed patch arbitrators over the same R038/R068 candidate set;
- voting/intersection/per-image selection among R038/R089/R090/R091.

Next useful ARIS move must generate genuinely new candidate diversity or change the data/model source, for example:

- a new segmentation architecture/candidate that is not just a conservative edit of R038;
- a data/label audit that changes training supervision;
- an image-model route that supplies new recall/contour candidates rather than re-ranking the same narrow disagreement pixels.
