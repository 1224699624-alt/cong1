#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
WORK=/root/final_prior_v1/tsrs
PY=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
TRAINER=nnUNetTrainerFinalPrior
MODE="${1:-audit}"
VARIANT="${FINAL_PRIOR_VARIANT:-D}"
RUN="$WORK/$MODE"

cd "$ROOT"
mkdir -p "$WORK/logs"
test ! -e "$RUN" || { echo "refusing to overwrite $RUN" >&2; exit 2; }
mkdir -p "$RUN"
exec > >(tee "$WORK/logs/${MODE}.log") 2>&1
export PATH="$PY:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export FINAL_PRIOR_GIT_COMMIT="$(git rev-parse HEAD)"
export nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export FINAL_PRIOR_VARIANT="$VARIANT"
export FINAL_PRIOR_TRAINING_MODE="${FINAL_PRIOR_TRAINING_MODE:-joint_finetune}"
export FINAL_PRIOR_NATIVE_CHECKPOINT="$ROOT/outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth"
export FINAL_PRIOR_NATIVE_SHA256=55d1bd52d3927a6fa9754d9d58adbe93c9abc4190f3b830fee45903d4981f31b
# Core TSRS path is dataset-native: it never loads the RAM overlap checkpoint.
unset FINAL_PRIOR_OVERLAP_CHECKPOINT R351_OVERLAP_CHECKPOINT
export FINAL_PRIOR_OVERLAP_CONTEXT_SCALE=0
export FINAL_PRIOR_OUTPUT_REFINER_CHECKPOINT="$ROOT/outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth"
export FINAL_PRIOR_EPOCHS="${FINAL_PRIOR_EPOCHS:-2}" FINAL_PRIOR_ITERATIONS="${FINAL_PRIOR_ITERATIONS:-100}"
export FINAL_PRIOR_BATCH_SIZE="${FINAL_PRIOR_BATCH_SIZE:-1}" FINAL_PRIOR_AUTOCAST="${FINAL_PRIOR_AUTOCAST:-true}"
export FINAL_PRIOR_AUDIT_ONLY=false

TRAINER_DIR="$($PY/python -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp "$ROOT/scripts/nnunet_trainers/$TRAINER.py" "$TRAINER_DIR/$TRAINER.py"
test "$(find "$DATA/nnUNet_preprocessed/$DATASET/nnUNetPlans_2d" -name 'train_*.pkl' | wc -l)" -eq 875
test "$(find "$DATA/nnUNet_preprocessed/$DATASET/nnUNetPlans_2d" -name 'val_*.pkl' | wc -l)" -eq 96

case "$MODE" in
  audit)
    export FINAL_PRIOR_AUDIT_ONLY=true FINAL_PRIOR_TRAINING_MODE=adapter_only
    nnUNetv2_train 204 2d 0 -tr "$TRAINER"
    ;;
  T0|A)
    export FINAL_PRIOR_VARIANT=A FINAL_PRIOR_TRAINING_MODE=joint_finetune
    nnUNetv2_train 204 2d 0 -tr "$TRAINER"
    ;;
  T1|B)
    export FINAL_PRIOR_VARIANT=B FINAL_PRIOR_TRAINING_MODE=joint_finetune
    nnUNetv2_train 204 2d 0 -tr "$TRAINER"
    ;;
  *) echo "usage: $0 {audit|T0|T1}" >&2; exit 2 ;;
esac
