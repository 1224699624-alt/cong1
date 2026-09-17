# R196 MaskDINO Binary-Union One-Image Gate

R196 was run on the server only with the isolated `aris-maskdino-r189-cu130` environment. No local training was run.

## Purpose

Test whether the R191-R195 failure is caused by faithful MaskDINO itself or by the many-small-instance target formulation. R196 converts all positive bone-epiphysis label IDs in each image into one binary foreground union mask, without changing the original dataset.

## Setup

- model: faithful external MaskDINO R50 with official COCO pretrained checkpoint
- target: one binary union foreground instance per image
- train/eval: first original train image only
- iterations: 1500
- image size: 384
- clean-test-v2: not used
- new/reannotated test: not used

## Result

- status: `train_and_eval_completed`
- segm AP / AP50 / AP75: `90.000000` / `100.000000` / `100.000000`
- same-image best threshold-union Dice: `0.958851` at threshold `0.02`
- same-image best top-k Dice: `0.958851` at top-k `1`
- top-10 best bbox IoU mean: `0.989128`
- predicted area ratio at best threshold: `0.989826`

## Decision

R196 passes the one-image overfit gate. Compared with R195's instance-form target Dice around 0.29, binary-union training reaches high same-image overlap and shows the MaskDINO environment, mapper, and training loop are usable. Proceed to R197: a server-only 24-train / 12-val binary-union train/val gate selected only on original train/val, still without clean-test-v2.
