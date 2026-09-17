# R197 MaskDINO Binary-Union Train/Val Gate

R197 was run on the server only with one RTX 4090 and the isolated `aris-maskdino-r189-cu130` environment. No local training was run.

## Purpose

Scale the R196 binary-union rescue from one-image overfit to a small original train/val generalization gate. This remains a train/val gate only and is not a success evaluation.

## Setup

- model: faithful external MaskDINO R50 with official COCO pretrained checkpoint
- target: one binary union foreground mask per image
- train subset: first 24 original `TSRS_RSNA-Epiphysis/train` images
- val subset: first 12 original `TSRS_RSNA-Epiphysis/val` images
- iterations: 2500
- image size: 384
- clean-test-v2: not used
- new/reannotated test: not used

## Result

COCO-style validation:

- bbox AP / AP50: `10.363282` / `64.970534`
- segm AP / AP50 / AP75: `49.994360` / `96.435644` / `33.624931`

Union-mask diagnostic on val:

- val images: `12`
- best threshold-union Dice: `0.832533` at threshold `0.01`
- best threshold-union IoU: `0.720703`
- best top-k Dice: `0.832315` at top-k `2`
- top-10 best bbox IoU mean: `0.925616`
- predicted area ratio at best threshold: mean `1.123287`

## Decision

R197 passes the small train/val gate: unlike R191-R195 instance-target attempts, binary-union MaskDINO produces real validation segmentation signal. It is still below the final project target and has only been tested on 12 validation images, so it must not be evaluated on clean-test-v2 yet. Proceed to R198: full original train/val binary-union training with validation-selected threshold/top-k readout and boundary/bridge diagnostic metrics on val.
