#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src;PY=/root/miniconda3/bin/python;BIN=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data";DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
R317="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
RUN="$ROOT/outputs/nnunet/r342_sample_adaptive_seam"
cd "$ROOT";mkdir -p outputs/bridge_logs "$RUN";exec > >(tee -a outputs/bridge_logs/r342_sample_adaptive_seam.log) 2>&1
export PATH="$BIN:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3421 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
export R342_INIT_CHECKPOINT="$R317" R342_EPOCHS=18 R342_LR=2e-5 R342_BUDGET=.10 R342_PRESERVE=.10 R342_CEILING=.10 R342_EDIT=.25
TRAINER_DIR="$($PY -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR342AdaptiveSeamBudget.py "$TRAINER_DIR/nnUNetTrainerR342AdaptiveSeamBudget.py"
$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR342AdaptiveSeamBudget.py"
test -s "$R317";test -d "$nnUNet_preprocessed/$DATASET/nnUNetPlans_2d"
rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR342AdaptiveSeamBudget
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerR342AdaptiveSeamBudget__nnUNetPlans__2d/fold_0";test -s "$FOLD/checkpoint_best.pth"
PRED="$RUN/predictions_val";MASKS="$RUN/masks_val";rm -rf "$PRED" "$MASKS"
nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" -d 204 -c 2d -f 0 -tr nnUNetTrainerR342AdaptiveSeamBudget -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$MASKS" --output outputs/analysis/r342_tsrs_sample_adaptive_seam_original_val_r201.json --expected-count 96
rm -rf "$PRED";rm -f "$FOLD/checkpoint_final.pth" "$FOLD/checkpoint_latest.pth"
echo '[R342] done';df -h /root/autodl-tmp
