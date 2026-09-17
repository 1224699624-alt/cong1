# R090 Patch Disagreement Arbitrator Summary

Date: 2026-06-27

Target: clean-test-v2 Dice `> 0.9317660066557425`.

## Purpose

R090 changed the candidate-disagreement formulation from independent-pixel MLPs to a lightweight patch/CNN arbitrator. The goal was to recover more of the R086 disagreement oracle by giving the model local image and mask context.

## Implementation Notes

Added `scripts/train_patch_disagreement_arbitrator.py`.

Key design:

- input feature maps: image, anchor mask, candidate masks, vote/union/intersection/disagreement, missing/excess indicators, signed distance, gradient;
- training crops sampled mainly from the R038/R068 disagreement zone;
- weighted loss emphasizes the edit zone;
- inference edits only the R038 anchor inside the R038/R068 disagreement zone;
- final success evaluation uses only `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

The first cache-building implementation was system-killed from memory pressure. The final run used an online `torch.utils.data.Dataset`, which reads images and generates crops per batch.

## Result

| System | Dice | Delta vs R070 | Precision | Recall | Boundary IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| R070 prior local arbitration | `0.914370` | - | `0.897204` | `0.933503` | `0.244320` |
| R089 precision-controlled MLP | `0.915467` | `+0.001097` | `0.904022` | `0.928470` | `0.245345` |
| R090 patch/CNN arbitrator | `0.916440` | `+0.002070` | `0.906271` | `0.928080` | `0.251366` |

Best validation config:

- radius `1`
- threshold `0.55`
- margin `0.15`
- original-val Dice `0.895699`

Remaining target gap: `0.015326`.

## Interpretation

Patch-level context is better than the pointwise MLP route. R090 improves Dice and Boundary IoU while maintaining the precision-controlled behavior found in R089. The Boundary IoU gain is the clearest sign that local CNN context is using contour evidence rather than only shifting foreground bias.

However, the improvement remains small relative to the target. The current R090 uses only one crop per image and a small CNN, so it is likely under-sampling the narrow disagreement region. The result supports continuing this formulation once, but not returning to broad MLP sweeps.

## Decision

Treat R090 as the new valid best.

Next useful experiment should keep the online Dataset design and increase useful disagreement supervision without reintroducing memory pressure:

- more crops per image with streaming DataLoader, not cached arrays;
- slightly larger patch model or more epochs;
- conservative threshold grid centered on the R090/R089 precision regime.

Stop condition: if a stronger streamed patch arbitrator still gives only sub-`+0.001` improvement, pivot away from candidate-disagreement postprocessing.
