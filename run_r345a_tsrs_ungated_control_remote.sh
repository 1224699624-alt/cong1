#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
R317="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
RUN="$ROOT/outputs/nnunet/r345a_ungated_control"
cd "$ROOT"
mkdir -p outputs/bridge_logs "$RUN"
exec > >(tee -a outputs/bridge_logs/r345a_ungated_control.log) 2>&1
export PATH="$PY:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3451 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
export R345A_INIT_CHECKPOINT="$R317" R345A_EPOCHS=18 R345A_LR=2e-5 R345A_ALPHA=.035
TRAINER_DIR="$($PY/python -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR345UngatedSeam.py "$TRAINER_DIR/nnUNetTrainerR345UngatedSeam.py"
$PY/python -m py_compile "$TRAINER_DIR/nnUNetTrainerR345UngatedSeam.py"
test -s "$R317"; test -d "$nnUNet_preprocessed/$DATASET/nnUNetPlans_2d"
rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR345UngatedSeam
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerR345UngatedSeam__nnUNetPlans__2d/fold_0"
evaluate_checkpoint () {
  local CHECKPOINT="$1" TAG="$2" PRED="$RUN/predictions_$2" MASKS="$RUN/masks_$2"
  rm -rf "$PRED" "$MASKS"
  nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" -d 204 -c 2d -f 0 -tr nnUNetTrainerR345UngatedSeam -chk "$CHECKPOINT"
  $PY/python scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
  $PY/python scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$MASKS" --output "outputs/analysis/r345a_tsrs_ungated_${TAG}_original_val_r201.json" --expected-count 96
  rm -rf "$PRED" "$MASKS"
}
evaluate_checkpoint checkpoint_best.pth best
evaluate_checkpoint checkpoint_final.pth final
rm -f "$FOLD/checkpoint_latest.pth" "$FOLD/checkpoint_final.pth"
echo '[R345A] done'; df -h /root/autodl-tmp
