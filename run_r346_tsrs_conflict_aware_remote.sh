#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
RUN="$ROOT/outputs/bridge_logs/r346_tsrs_conflict_aware"
mkdir -p "$RUN"
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="$ROOT/scripts"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3460 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
export R346_INIT_CHECKPOINT="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
export R346_EPOCHS=60 R346_LR=2e-5 R346_SEAM_WEIGHT=.035 R346_PRESERVE_WEIGHT=.20
TRAINER_DIR="$(python -c 'from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / "training/nnUNetTrainer/variants/loss")')"
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainerR346ConflictAwareSeam.py" "$TRAINER_DIR/nnUNetTrainerR346ConflictAwareSeam.py"
FOLD="$nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR346ConflictAwareSeam__nnUNetPlans__2d/fold_0"
if [[ -s "$FOLD/checkpoint_latest.pth" ]]; then
  echo "[R346-TSRS] resuming from checkpoint_latest"
  nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR346ConflictAwareSeam -c 2>&1 | tee -a "$RUN/train.log"
else
  rm -rf "$nnUNet_results"
  mkdir -p "$nnUNet_results"
  nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR346ConflictAwareSeam 2>&1 | tee "$RUN/train.log"
fi
