#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin
DATA="$ROOT/outputs/nnunet/r317_image_centers_only/data"
DATASET=Dataset204_TSRS_RSNAEpiphysisMaturePrior2D
TRAINER=nnUNetTrainerR317Continuous035
CONFIG=nnUNetPlans__2d
SOURCE_MODEL="$ROOT/outputs/nnunet/r317_image_centers_only/nnUNet_results/$DATASET/${TRAINER}__${CONFIG}"
NATIVE_CHECKPOINT="$ROOT/outputs/nnunet/r317_image_centers_only/r202_two_channel_zero_prior.pth"
RUN="$ROOT/outputs/nnunet/r350_native_recheck"
RESULTS="$RUN/nnUNet_results"
MODEL="$RESULTS/$DATASET/${TRAINER}__${CONFIG}"
FOLD="$MODEL/fold_0"
PRED="$RUN/predictions"
MASKS="$RUN/masks"
OUT="$ROOT/outputs/analysis/r350_native_nnunet_matched_original_val_r201.json"

cd "$ROOT"
mkdir -p outputs/bridge_logs outputs/analysis "$FOLD"
exec > >(tee -a outputs/bridge_logs/r350_native_nnunet_matched_recheck.log) 2>&1
export PATH="$PY:$PATH" PYTHONPATH="$ROOT/scripts:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0
export nnUNet_compile=false nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw" nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RESULTS"

test -s "$NATIVE_CHECKPOINT"
test -s "$SOURCE_MODEL/dataset.json"
test -s "$SOURCE_MODEL/plans.json"
test "$(find data/raw/TSRS_RSNA-Epiphysis/val_labels -maxdepth 1 -type f | wc -l)" -eq 96
test "$(find outputs/nnunet/r317_image_centers_only/imagesValPrior -maxdepth 1 -type f -name '*_0000.png' | wc -l)" -eq 96
test "$(find outputs/nnunet/r317_image_centers_only/imagesValPrior -maxdepth 1 -type f -name '*_0001.png' | wc -l)" -eq 96
! printf '%s\n' "$RUN" "$OUT" | grep -qi 'clean-test\|articular'

# Reuse only architecture metadata. The evaluated weights are the untouched
# mature native nnU-Net checkpoint recorded as the R317 initialization.
ln -sfn "$SOURCE_MODEL/dataset.json" "$MODEL/dataset.json"
ln -sfn "$SOURCE_MODEL/plans.json" "$MODEL/plans.json"
ln -sfn "$NATIVE_CHECKPOINT" "$FOLD/checkpoint_native.pth"

rm -rf "$PRED" "$MASKS"
echo '[R350 native recheck] matched inference'
nnUNetv2_predict \
  -i outputs/nnunet/r317_image_centers_only/imagesValPrior \
  -o "$PRED" -d 204 -c 2d -f 0 -tr "$TRAINER" -chk checkpoint_native.pth

echo '[R350 native recheck] matched conversion and R201 evaluation'
$PY/python scripts/convert_nnunet_predictions.py \
  --pred-dir "$PRED" --mask-dir "$MASKS" --expected-count 96 --strip-prefix val_
$PY/python scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$MASKS" --output "$OUT" --expected-count 96

test -s "$OUT"
rm -rf "$PRED"
echo '[R350 native recheck] complete'
df -h /root/autodl-tmp
