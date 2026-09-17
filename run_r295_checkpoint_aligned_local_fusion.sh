#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
FINAL="$ROOT/outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
export EXP_ID="r295"
export EXP_LABEL="R295"
export RUN_OVERRIDE="outputs/nnunet/r295_checkpoint_aligned_local_fusion"
export SOURCE_OVERRIDE="$FINAL"
export SOURCE_SHA_OVERRIDE="658575e68f94b101fdbc4d861e094bb80882794c5a8762a16d8587d09a5278b9"
export PRED_CHK_OVERRIDE="checkpoint_final.pth"
export PYTHONHASHSEED_OVERRIDE=295
exec bash "$ROOT/run_r294_local_fusion_nnunet.sh"
