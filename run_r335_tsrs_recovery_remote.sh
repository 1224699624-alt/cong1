#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin/python
BIN=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
RUN="$ROOT/outputs/nnunet/r335_conditional_state_loss"
R317="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
cd "$ROOT"; mkdir -p outputs/bridge_logs "$RUN"
exec > >(tee -a outputs/bridge_logs/r335_tsrs_conditional_recovery.log) 2>&1
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3352 nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results"
export R335_TSRS_INIT_CHECKPOINT="$R317" R335_TSRS_EPOCHS=18 R335_TSRS_SEAM_WEIGHT=.020 R335_TSRS_SUPPORT_WEIGHT=.010 R335_TSRS_SEAM_CEILING=.10
TRAINER_DIR="$($PY -c "from pathlib import Path;import nnunetv2;print(Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR335ConditionalStateLoss.py "$TRAINER_DIR/nnUNetTrainerR335ConditionalStateLoss.py"
$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR335ConditionalStateLoss.py"
PLAN_DIR="$nnUNet_preprocessed/$DATASET"
if [[ ! -d "$PLAN_DIR/nnUNetPlans_2d" ]]; then
  AVAIL="$(df -Pk /root/autodl-tmp | awk 'NR==2{print $4}')"; [[ "$AVAIL" -ge 3670016 ]]
  $PY scripts/prepare_r260_mature_prior_dataset.py --dataset-root data/raw/TSRS_RSNA-Epiphysis --prior-root outputs/priors/r317_image_centers_only --nnunet-root "$DATA" --allow-zero-prior --allow-png-only --overwrite
  nnUNetv2_extract_fingerprint -d 204 -np 6 --verify_dataset_integrity --clean
  mkdir -p "$PLAN_DIR"; cp outputs/nnunet/r319_r316b_gated_prior/source/nnUNetPlans.json "$PLAN_DIR/nnUNetPlans.json"; cp "$nnUNet_raw/$DATASET/dataset.json" "$PLAN_DIR/dataset.json"
  nnUNetv2_preprocess -d 204 -plans_name nnUNetPlans -c 2d -np 6
  cp "$DATA/splits_final_r260.json" "$PLAN_DIR/splits_final.json"; rm -rf "$nnUNet_raw"
  AVAIL="$(df -Pk /root/autodl-tmp | awk 'NR==2{print $4}')"; [[ "$AVAIL" -ge 1572864 ]]
fi
rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR335ConditionalStateLoss
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerR335ConditionalStateLoss__nnUNetPlans__2d/fold_0"; test -s "$FOLD/checkpoint_best.pth"
PRED="$RUN/predictions_val"; MASKS="$RUN/masks_val"; rm -rf "$PRED" "$MASKS"
nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" -d 204 -c 2d -f 0 -tr nnUNetTrainerR335ConditionalStateLoss -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$MASKS" --output outputs/analysis/r335_tsrs_conditional_state_loss_original_val_r201.json --expected-count 96
rm -rf "$PRED"; rm -f "$FOLD/checkpoint_final.pth" "$FOLD/checkpoint_latest.pth"
echo R335_TSRS_DONE; df -h /root/autodl-tmp
