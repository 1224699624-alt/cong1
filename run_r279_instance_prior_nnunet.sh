#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=279
export nnUNet_compile=false

RUN="outputs/nnunet/r279_instance_prior"
DATA="$RUN/data"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r279_instance_prior_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R279] start $(date -Is)"

PRIOR_ROOT="outputs/priors/r279_frozen_instance_seam"
SOURCE="outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
SOURCE_SHA="557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27"
BASE="outputs/predictions/r275_mature_original_val/control"

test -s "$PRIOR_ROOT/manifest.json"
test "$(find "$PRIOR_ROOT/train" -maxdepth 1 -name '*.png' | wc -l)" = 875
test "$(find "$PRIOR_ROOT/val" -maxdepth 1 -name '*.png' | wc -l)" = 96
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA"
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96

$PY -m py_compile \
  scripts/prepare_r279_instance_prior_dataset.py \
  scripts/prepare_r279_val_inputs.py \
  scripts/adapt_checkpoint_two_channel_r279.py \
  scripts/nnunet_trainers/nnUNetTrainerR279InstancePrior.py

$PY scripts/prepare_r279_instance_prior_dataset.py --overwrite
nnUNetv2_plan_and_preprocess -d 279 -c 2d --verify_dataset_integrity
cp "$DATA/splits_final_r279.json" \
  "$nnUNet_preprocessed/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/splits_final.json"

ADAPTED="$RUN/r275_control_two_channel_zero_prior.pth"
$PY scripts/adapt_checkpoint_two_channel_r279.py \
  --input "$SOURCE" --output "$ADAPTED" --expected-sha256 "$SOURCE_SHA" \
  | tee "$RUN/adaptation_manifest.json"
export R279_ADAPTED_CHECKPOINT="$ADAPTED"
cp scripts/nnunet_trainers/nnUNetTrainerR279InstancePrior.py "$nnUNet_extTrainer/"
$PY -m py_compile "$nnUNet_extTrainer/nnUNetTrainerR279InstancePrior.py"
$PY scripts/prepare_r279_val_inputs.py --output "$RUN/imagesValPrior"

OUT="$nnUNet_results/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/nnUNetTrainerR279InstancePrior__nnUNetPlans__2d/fold_0"
rm -rf "${OUT%/fold_0}"
nnUNetv2_train 279 2d 0 -tr nnUNetTrainerR279InstancePrior
test -s "$OUT/checkpoint_best.pth"
test -s "$OUT/r279_early_stop.json"

rm -rf "$RUN/predictions_prior"
nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/predictions_prior" \
  -d 279 -c 2d -f 0 -tr nnUNetTrainerR279InstancePrior -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r279_instance_prior/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$ACTIVE"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_prior" \
  --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$BASE" \
  --output outputs/analysis/r279_r275_control_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$ACTIVE" \
  --output outputs/analysis/r279_instance_prior_original_val_r201.json --expected-count 96

VIS="outputs/visualizations/r279_instance_prior_original_val"
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$BASE" --improved-dir "$ACTIVE" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" \
  --baseline-label "Mature R275 nnU-Net" \
  --improved-label "R260-style + instance seam prior/loss"
echo "[R279] done $(date -Is)"
