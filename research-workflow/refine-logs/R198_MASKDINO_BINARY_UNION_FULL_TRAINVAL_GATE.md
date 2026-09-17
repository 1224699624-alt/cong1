# R198 MaskDINO Binary-Union Full Train/Val Gate

R198 was run on the server only with one RTX 4090 and the isolated `aris-maskdino-r189-cu130` environment. No local training was run.

## Purpose

Scale the R197 binary-union MaskDINO signal to the full original train/val split and compute val-only constraint diagnostics aligned with the project's bone-gap/boundary objective.

## Setup

- model: faithful external MaskDINO R50 with official COCO pretrained checkpoint
- target: one binary union foreground mask per image
- train: full original `TSRS_RSNA-Epiphysis/train`
- val: full original `TSRS_RSNA-Epiphysis/val`
- iterations: 12000
- image size: 384
- clean-test-v2: not used
- new/reannotated test: not used

## Result

COCO-style validation:

- bbox AP / AP50 / AP75: `3.919193` / `20.928254` / `0.685453`
- segm AP / AP50 / AP75: `54.394994` / `93.638087` / `70.234824`

Mask-union diagnostic:

- val images: `96`
- best threshold-union Dice: `0.851705` at threshold `0.05`
- best threshold-union IoU: `0.753802`
- best top-k Dice: `0.851705` at top-k `1`
- top-10 best bbox IoU mean: `0.896153`

Selected val readout from constraint evaluator:

- mode/value: `threshold` / `0.05`
- Dice: `0.851705`
- IoU: `0.753802`
- Boundary IoU: `0.545691`
- Boundary F1: `0.699550`
- separation-band FP rate: `0.068332`
- false-bridge flag rate: `0.760417`
- component-count error: `2.947917`

## Decision

R198 confirms the binary-union MaskDINO route is technically viable and gives strong full-val boundary overlap relative to earlier failed MaskDINO attempts. However, the union target still under-segments anatomical components: false-bridge rate remains high and component-count error is not controlled. It should not be evaluated on clean-test-v2 as a locked candidate yet. Next step should be R199: val-only analysis of R198 failure cases and a component-aware readout/decoder constraint that preserves the high union boundary quality while reducing bridge/under-segmentation, selected only on original train/val.
