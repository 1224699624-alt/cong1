#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
R317="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
RUN="$ROOT/outputs/nnunet/r344_region_seam"
cd "$ROOT"
mkdir -p outputs/bridge_logs "$RUN"
exec > >(tee -a outputs/bridge_logs/r344_region_seam.log) 2>&1
export PATH="$PY:$PATH"
export PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3441 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export R344_INIT_CHECKPOINT="$R317"
export R344_EPOCHS=18 R344_LR=2e-5 R344_REGION_THRESHOLD=.20 R344_MIN_REGION_PIXELS=20
export R344_REGION_BUDGET=.10 R344_REGION_WEIGHT_CAP=.25 R344_LOSS_BUDGET=.02 R344_PRESERVE=.10
TRAINER_DIR="$($PY/python -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR344RegionSeamBudget.py "$TRAINER_DIR/nnUNetTrainerR344RegionSeamBudget.py"
$PY/python -m py_compile "$TRAINER_DIR/nnUNetTrainerR344RegionSeamBudget.py"
test -s "$R317"
test -d "$nnUNet_preprocessed/$DATASET/nnUNetPlans_2d"
rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR344RegionSeamBudget
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerR344RegionSeamBudget__nnUNetPlans__2d/fold_0"
test -s "$FOLD/checkpoint_best.pth"

evaluate_checkpoint () {
  local CHECKPOINT="$1"
  local TAG="$2"
  local PRED="$RUN/predictions_${TAG}"
  local MASKS="$RUN/masks_${TAG}"
  rm -rf "$PRED" "$MASKS"
  nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" -d 204 -c 2d -f 0 -tr nnUNetTrainerR344RegionSeamBudget -chk "$CHECKPOINT"
  $PY/python scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
  $PY/python scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$MASKS" --output "outputs/analysis/r344_tsrs_region_seam_${TAG}_original_val_r201.json" --expected-count 96
  rm -rf "$PRED"
}

evaluate_checkpoint checkpoint_best.pth best
test -s "$FOLD/checkpoint_final.pth"
evaluate_checkpoint checkpoint_final.pth final
rm -rf "$RUN/masks_best" "$RUN/masks_final"
rm -f "$FOLD/checkpoint_latest.pth"
echo '[R344] done'
df -h /root/autodl-tmp
