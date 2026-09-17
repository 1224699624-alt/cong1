# R332 RAM Joint Iterative Refinement Protocol

- Status: running
- Server: `connect.westb.seetacloud.com:29459`
- Remote workspace: `/root/autodl-tmp/YOLO_SAM_generic_src`
- Screen: `r332_joint_iterative`
- Dataset: RAM-W600 train/validation only; test locked
- Spatial preprocessing: native pixels with right/bottom padding; no resize
- Baseline: R325 native-resolution nnU-Net
- Refiner initialization: R330 `surface003_volume010_best.pth`

## Matrix

| Arm | Joint fine-tuning | Inference corrections | Epochs |
|---|---|---:|---:|
| R332-A | last two high-resolution decoder stages + output head | 1 | 30 |
| R332-B | same | 2 | 30 |
| R332-C | same | 3 | 30 |

The first four epochs stabilize the refiner with the R325 network frozen. From
epoch 5 onward, only the last two high-resolution decoder stages and final
segmentation head are unfrozen. The encoder remains frozen.

## Optimization and stopping

- Refiner learning rate: `8e-5`
- Decoder learning rate: `7e-6`
- No patience early stopping
- Fixed 30-epoch budget for every arm
- Resumable `last.pth` is overwritten each epoch
- Snapshots: epochs 10, 20 and 30
- Abnormal termination only for exceptions such as NaN/OOM

Each correction step receives segmentation, surface and volume supervision.
Later steps receive larger deep-supervision weights. A monotonic loss penalty
discourages a later correction from having higher supervised loss, and a
contraction penalty discourages later residuals from becoming larger.

## Validation and checkpoint rule

Every epoch reports fast validation trends for step 0 through the configured
final step. Complete RAM paper-aligned validation metrics are computed at fixed
epochs 5, 10, 15, 20, 25 and 30. Test is not loaded.

The official checkpoint gate requires:

1. overall DSC not below R325;
2. overall IoU not below R325;
3. overlap RAVD no more than 0.003 above R325;
4. pair MSD fail rate no more than 0.001 above R325.

Passing checkpoints are ranked by stepwise monotonic checks, overlap NSD,
overlap DSC, overlap MSD and overlap RAVD. Final correction depth is selected
only after comparing the complete 1/2/3-step validation arms.

## Initial verification

CUDA smoke passed:

- finite loss: yes
- decoder gradient sum: `13.2453`
- refiner gradient sum: `9.2766`
- frozen encoder gradient sum: `0`

R332-A epoch 1 completed without NaN. Its fast validation showed step 1 above
the jointly current step 0 for macro DSC/IoU, overlap DSC/IoU, overlap NSD and
overlap MSD. This is preliminary and is not an official checkpoint audit.
