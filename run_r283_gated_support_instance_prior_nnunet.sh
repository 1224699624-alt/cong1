#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=283 nnUNet_compile=false

R279="outputs/nnunet/r279_instance_prior"
RUN="outputs/nnunet/r283_gated_support_instance_prior"
export nnUNet_raw="$R279/data/nnUNet_raw"
export nnUNet_preprocessed="$R279/data/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r283_gated_support_instance_prior_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R283] start $(date -Is)"

ADAPTED="$R279/r275_control_two_channel_zero_prior.pth"
OUT="$nnUNet_results/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/nnUNetTrainerR283GatedSupportInstancePrior__nnUNetPlans__2d/fold_0"
BASE="outputs/predictions/r275_mature_original_val/control"
test "$(sha256sum "$ADAPTED" | cut -d' ' -f1)" = "89eda6b07c0e6c37c3d41abf295d97f4f06c5ea30d4d061ef9ddcb1680128f5a"
test "$("$PY" -c "import json; print(json.load(open('outputs/analysis/r282_gradnorm_gate.json'))['decision'])")" = "no_go_use_gradient_normalization"
test -s "$nnUNet_preprocessed/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/nnUNetPlans.json"
test "$(find "$R279/imagesValPrior" -maxdepth 1 -name '*_0000.png' | wc -l)" = 96
test "$(find "$R279/imagesValPrior" -maxdepth 1 -name '*_0001.png' | wc -l)" = 96
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96

$PY -m py_compile scripts/nnunet_trainers/nnUNetTrainerR280BalancedInstancePrior.py \
  scripts/nnunet_trainers/nnUNetTrainerR283GatedSupportInstancePrior.py \
  scripts/summarize_r283_gate.py
cp scripts/nnunet_trainers/nnUNetTrainerR280BalancedInstancePrior.py "$nnUNet_extTrainer/"
cp scripts/nnunet_trainers/nnUNetTrainerR283GatedSupportInstancePrior.py "$nnUNet_extTrainer/"
export R280_ADAPTED_CHECKPOINT="$ADAPTED"

rm -rf "${OUT%/fold_0}"
nnUNetv2_train 279 2d 0 -tr nnUNetTrainerR283GatedSupportInstancePrior
test -s "$OUT/checkpoint_best.pth"
test -s "$OUT/r283_early_stop.json"
test -s "$OUT/r283_loss_dynamics.jsonl"

rm -rf "$RUN/predictions_prior"
nnUNetv2_predict -i "$R279/imagesValPrior" -o "$RUN/predictions_prior" \
  -d 279 -c 2d -f 0 -tr nnUNetTrainerR283GatedSupportInstancePrior -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r283_gated_support_instance_prior/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$ACTIVE"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_prior" \
  --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_

BASE_JSON="outputs/analysis/r283_r275_control_original_val_r201.json"
RESULT_JSON="outputs/analysis/r283_gated_support_instance_prior_original_val_r201.json"
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$BASE" --output "$BASE_JSON" --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$ACTIVE" --output "$RESULT_JSON" --expected-count 96
$PY scripts/summarize_r283_gate.py --baseline "$BASE_JSON" --result "$RESULT_JSON" \
  --output outputs/analysis/r283_gated_support_gate.json

VIS="outputs/visualizations/r283_gated_support_instance_prior_original_val"
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$BASE" --improved-dir "$ACTIVE" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" \
  --baseline-label "Mature R275 nnU-Net" \
  --improved-label "Gated-support instance-prior nnU-Net"
echo "[R283] done $(date -Is)"
