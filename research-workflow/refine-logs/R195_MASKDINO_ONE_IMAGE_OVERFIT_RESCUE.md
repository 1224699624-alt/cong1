# R195 MaskDINO One-Image Overfit Rescue

R195 was run on the server only, using one RTX 4090 and the isolated `aris-maskdino-r189-cu130` environment. No local training was run.

## Setup

- model: faithful external MaskDINO R50 with official COCO pretrained checkpoint
- data: first image from original `TSRS_RSNA-Epiphysis/train` only
- eval: same single train image
- iterations: 1500
- image size: 384
- clean-test-v2: not used
- new/reannotated test: not used

## Result

- status: `train_and_eval_completed`
- segm AP / AP50: `6.336634` / `7.920792`
- bbox AP: `0.000000`
- best same-image threshold-union Dice: `0.291615`
- best same-image top-k union Dice: `0.291615`
- max prediction score: `0.926548`
- predicted area ratio at best threshold: `0.172965`

## Decision

R195 improves over the zero-overlap R193/R194 diagnostics, so the target path is not completely broken. However, one-image overfit Dice around 0.29 is far below the threshold needed to justify full train/val training or clean-test-v2 evaluation. The next server-only step should inspect target representation and inference scaling, or try a binary-union / semantic-style MaskDINO target before returning to instance-level full training.
