#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
RUN="$ROOT/outputs/bridge_logs/r348_tsrs_corrected_seam"
mkdir -p "$RUN"
while screen -list | grep -q '[.]r347_ram'; do sleep 60; done
export PATH=/root/miniconda3/bin:$PATH PYTHONPATH="$ROOT/scripts"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3480 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
export R346_INIT_CHECKPOINT="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/Dataset204_TSRS_RSNAEpiphysisMaturePrior2D/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
export R346_EPOCHS=60 R346_LR=2e-5 R346_SEAM_WEIGHT=.035 R346_PRESERVE_WEIGHT=.20 R346_SEAM_THRESHOLD=.25
TRAINER_DIR="$(python -c 'from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / "training/nnUNetTrainer/variants/loss")')"
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainerR346ConflictAwareSeam.py" "$TRAINER_DIR/nnUNetTrainerR346ConflictAwareSeam.py"
cp "$ROOT/scripts/nnunet_trainers/nnUNetTrainerR348CorrectedSeamPreserve.py" "$TRAINER_DIR/nnUNetTrainerR348CorrectedSeamPreserve.py"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR348CorrectedSeamPreserve 2>&1 | tee "$RUN/train.log"
