# R166 Post-Reannotated Complementarity Decision

## Question

After R165 improved over raw reannotated training but remained far below R110, check whether the filtered reannotated model is useful as a complementary candidate on `TSRS_RSNA-Epiphysis_clean_test_v2/test`.

This is a diagnostic-only comparison. It uses completed clean-test-v2 per-image metrics and does not tune, train, or select a model on clean-test-v2.

## Inputs

- R110 current valid best: `outputs/analysis/r110_r100_r108_patch_basic_clean_test_v2_metrics.json`
- R165 filtered reannotated model: `outputs/analysis/r165_filtered_reannotated_dinov3_instance_sep_clean_test_v2_metrics.json`
- R166 audit output: `outputs/analysis/r166_r110_r165_filtered_reannotated_complementarity.json`

## Result

| Metric | R110 | R165 | Delta |
| --- | ---: | ---: | ---: |
| clean-test-v2 Dice | `0.917723` | `0.890303` | `-0.027420` |
| Boundary IoU | `0.251896` | `0.184990` | `-0.066906` |

Per-image complementarity:

- Common clean-test-v2 images: `81`
- R165 better than R110: `2/81`
- R110 better or tied: `79/81`
- R165 better cases: `8435.png` by `+0.004931`, `3503.png` by `+0.000215`
- Worst R165 regression: `3591.png` by `-0.098347`
- Per-image oracle over R110/R165: `0.917787`
- Oracle gain over R110: `+0.000064`
- Remaining gap to target `0.931766`: `0.013979`

## Decision

Close the reannotated direct-training branch and do not run R165 fusion/readout/threshold variants.

Filtering the reannotated train/val data helped R162 by `+0.017676`, but R165 is still far below R110 and has almost no per-image complementarity. The best possible whole-image selection between R110 and R165 gains only `0.000064` Dice, which is not a material path toward the target.

## Next Direction

Do not continue small automated filtering tweaks on the reannotated labels. The next useful progress requires one of:

- manual label-protocol reconciliation that produces corrected train/val labels, not just filtered reannotations;
- a CUDA-toolkit-capable isolated environment for faithful pretrained MaskDINO/Mask2Former-style implementations;
- a materially new no-custom-CUDA architecture with a strict smoke/overfit gate before any full clean-test-v2 run.

