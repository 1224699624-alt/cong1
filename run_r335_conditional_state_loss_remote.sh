#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
RAM_OUT="$ROOT/outputs/ram_w600/r335_conditional_state_loss"
TSRS_RUN="$ROOT/outputs/nnunet/r335_conditional_state_loss"
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET="Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
R317_FOLD="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/nnUNetTrainerR317Continuous035__nnUNetPlans__2d/fold_0"
LOG="$ROOT/outputs/bridge_logs/r335_conditional_state_loss.log"

cd "$ROOT"
mkdir -p outputs/bridge_logs "$RAM_OUT" "$TSRS_RUN"
exec > >(tee -a "$LOG") 2>&1
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=3351 nnUNet_compile=false nnUNet_n_proc_DA=0

echo "[R335] start $(date -Is)"
df -h /root/autodl-tmp
test -s outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth
test -s "$R317_FOLD/checkpoint_best.pth"
test "$(find outputs/priors/r323_ram_native_r317_single_seed -type f | wc -l)" -ge 989
test "$(find outputs/priors/r317_image_centers_only -type f | wc -l)" -ge 972

echo "[R335-RAM] train/validation only $(date -Is)"
rm -rf "$RAM_OUT"
$PY scripts/train_r335_ram_conditional_state_loss.py \
  --dataset-root data/remote_variants/RAM-W600 \
  --prior-root outputs/priors/r323_ram_native_r317_single_seed \
  --r332-checkpoint outputs/ram_w600/r332_joint_iterative_refinement/steps_2/best.pth \
  --output "$RAM_OUT" --epochs 18 --steps 2 \
  --seg-weight .10 --seam-weight .025 --overlap-weight .04 --preserve-weight 1.0
test -s "$RAM_OUT/result.json"

echo "[R335-TSRS] install trainer and reuse exact R317 preprocessed data $(date -Is)"
TRAINER_DIR="$($PY -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR335ConditionalStateLoss.py "$TRAINER_DIR/nnUNetTrainerR335ConditionalStateLoss.py"
$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR335ConditionalStateLoss.py"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$TSRS_RUN/nnUNet_results"
export R335_TSRS_INIT_CHECKPOINT="$R317_FOLD/checkpoint_best.pth"
export R335_TSRS_EPOCHS=18 R335_TSRS_SEAM_WEIGHT=.020 R335_TSRS_SUPPORT_WEIGHT=.010 R335_TSRS_SEAM_CEILING=.10

PLAN_DIR="$nnUNet_preprocessed/$DATASET"
if [[ ! -d "$PLAN_DIR/nnUNetPlans_2d" ]]; then
  echo "[R335-TSRS] R317 arrays were historically cleaned; rebuild exact isolated preprocessing $(date -Is)"
  AVAIL_KB="$(df -Pk /root/autodl-tmp | awk 'NR==2{print $4}')"
  [[ "$AVAIL_KB" -ge 3670016 ]] || { echo "Need at least 3.5 GiB free before preprocessing"; exit 31; }
  $PY scripts/prepare_r260_mature_prior_dataset.py \
    --dataset-root data/raw/TSRS_RSNA-Epiphysis \
    --prior-root outputs/priors/r317_image_centers_only \
    --nnunet-root "$DATA" --allow-zero-prior --allow-png-only --overwrite
  nnUNetv2_extract_fingerprint -d 204 -np 6 --verify_dataset_integrity --clean
  mkdir -p "$PLAN_DIR"
  cp outputs/nnunet/r319_r316b_gated_prior/source/nnUNetPlans.json "$PLAN_DIR/nnUNetPlans.json"
  cp "$nnUNet_raw/$DATASET/dataset.json" "$PLAN_DIR/dataset.json"
  nnUNetv2_preprocess -d 204 -plans_name nnUNetPlans -c 2d -np 6
  cp "$DATA/splits_final_r260.json" "$PLAN_DIR/splits_final.json"
  rm -rf "$nnUNet_raw"
  AVAIL_KB="$(df -Pk /root/autodl-tmp | awk 'NR==2{print $4}')"
  [[ "$AVAIL_KB" -ge 1572864 ]] || { echo "Less than 1.5 GiB free after preprocessing; refusing training"; exit 32; }
fi

rm -rf "$nnUNet_results"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR335ConditionalStateLoss
FOLD="$nnUNet_results/$DATASET/nnUNetTrainerR335ConditionalStateLoss__nnUNetPlans__2d/fold_0"
test -s "$FOLD/checkpoint_best.pth"

echo "[R335-TSRS] original-val prediction and R201 audit $(date -Is)"
PRED="$TSRS_RUN/predictions_val"; MASKS="$TSRS_RUN/masks_val"
rm -rf "$PRED" "$MASKS"
nnUNetv2_predict -i outputs/nnunet/r317_image_centers_only/imagesValPrior -o "$PRED" \
  -d 204 -c 2d -f 0 -tr nnUNetTrainerR335ConditionalStateLoss -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$MASKS" \
  --output outputs/analysis/r335_tsrs_conditional_state_loss_original_val_r201.json --expected-count 96

# Storage policy: retain one selected checkpoint and masks; predictions/final/latest are reproducible duplicates.
rm -rf "$PRED"
rm -f "$FOLD/checkpoint_final.pth" "$FOLD/checkpoint_latest.pth"
echo "[R335] done $(date -Is)"
df -h /root/autodl-tmp
