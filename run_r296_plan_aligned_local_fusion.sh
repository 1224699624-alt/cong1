#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PRE="$ROOT/outputs/nnunet/r293_dualbranch/data/nnUNet_preprocessed/Dataset293_TSRS_RSNAEpiphysisDualBranch2D"
REF="$ROOT/outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/plans.json"
/root/miniconda3/bin/python "$ROOT/scripts/create_r296_plan_aligned.py" \
  --input "$PRE/nnUNetPlans.json" --reference "$REF" \
  --output "$PRE/R296Plans.json" \
  --manifest "$ROOT/outputs/nnunet/r296_plan_aligned_local_fusion/r296_plan_manifest.json"

FINAL="$ROOT/outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_final.pth"
export EXP_ID="r296"
export EXP_LABEL="R296"
export RUN_OVERRIDE="outputs/nnunet/r296_plan_aligned_local_fusion"
export SOURCE_OVERRIDE="$FINAL"
export SOURCE_SHA_OVERRIDE="658575e68f94b101fdbc4d861e094bb80882794c5a8762a16d8587d09a5278b9"
export PRED_CHK_OVERRIDE="checkpoint_final.pth"
export PLANS_ID_OVERRIDE="R296Plans"
export PYTHONHASHSEED_OVERRIDE=296
exec bash "$ROOT/run_r294_local_fusion_nnunet.sh"
