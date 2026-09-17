# R193/R194 MaskDINO Pretrained Overfit Gate

R193 and R194 were run on the server only, using one RTX 4090 and the isolated `aris-maskdino-r189-cu130` environment.

## R193 Setup

- official MaskDINO R50 COCO pretrained checkpoint downloaded on the server
- train subset: first 24 original train images
- val subset: first 12 original val images
- iterations: 500
- clean-test-v2: not used
- new/reannotated test: not used

## R193 Result

- train/eval status: `train_and_eval_completed`
- final bbox AP / segm AP: `0.0` / `0.0`
- val best threshold-union Dice: `0.000000`
- val top-10 bbox IoU mean: `0.000000`

## R194 Train-Subset Diagnostic

R194 evaluated the R193 checkpoint back on the same 24-image train subset.

- status: `eval_completed`
- train-subset best threshold-union Dice: `0.000137`
- train-subset top-10 bbox IoU mean: `0.024317`

## Decision

The pretrained route still failed the basic overfit check. Since the same train subset has near-zero union Dice, the issue is not simply validation generalization or AP thresholding. Before full training, run an even smaller 1-image overfit / target-format rescue. If that also fails, pause MaskDINO and inspect mapper/label geometry/config instead of launching longer training.
