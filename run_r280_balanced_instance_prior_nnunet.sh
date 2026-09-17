#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=280
export nnUNet_compile=false

R279="outputs/nnunet/r279_instance_prior"
RUN="outputs/nnunet/r280_balanced_instance_prior"
export nnUNet_raw="$R279/data/nnUNet_raw"
export nnUNet_preprocessed="$R279/data/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r280_balanced_instance_prior_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R280] start $(date -Is)"

ADAPTED="$R279/r275_control_two_channel_zero_prior.pth"
ADAPTED_SHA="89eda6b07c0e6c37c3d41abf295d97f4f06c5ea30d4d061ef9ddcb1680128f5a"
BASE="outputs/predictions/r275_mature_original_val/control"
test "$(sha256sum "$ADAPTED" | cut -d' ' -f1)" = "$ADAPTED_SHA"
test -s "$nnUNet_preprocessed/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/nnUNetPlans.json"
test -s "$nnUNet_preprocessed/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/splits_final.json"
test "$(find "$R279/imagesValPrior" -maxdepth 1 -name '*_0000.png' | wc -l)" = 96
test "$(find "$R279/imagesValPrior" -maxdepth 1 -name '*_0001.png' | wc -l)" = 96
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96

$PY -m py_compile scripts/nnunet_trainers/nnUNetTrainerR280BalancedInstancePrior.py
cp scripts/nnunet_trainers/nnUNetTrainerR280BalancedInstancePrior.py "$nnUNet_extTrainer/"
$PY -m py_compile "$nnUNet_extTrainer/nnUNetTrainerR280BalancedInstancePrior.py"
export R280_ADAPTED_CHECKPOINT="$ADAPTED"

OUT="$nnUNet_results/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/nnUNetTrainerR280BalancedInstancePrior__nnUNetPlans__2d/fold_0"
rm -rf "${OUT%/fold_0}"
nnUNetv2_train 279 2d 0 -tr nnUNetTrainerR280BalancedInstancePrior
test -s "$OUT/checkpoint_best.pth"
test -s "$OUT/r280_early_stop.json"

rm -rf "$RUN/predictions_prior"
nnUNetv2_predict -i "$R279/imagesValPrior" -o "$RUN/predictions_prior" \
  -d 279 -c 2d -f 0 -tr nnUNetTrainerR280BalancedInstancePrior -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r280_balanced_instance_prior/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$ACTIVE"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_prior" \
  --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$BASE" \
  --output outputs/analysis/r280_r275_control_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$ACTIVE" \
  --output outputs/analysis/r280_balanced_instance_prior_original_val_r201.json --expected-count 96

VIS="outputs/visualizations/r280_balanced_instance_prior_original_val"
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$BASE" --improved-dir "$ACTIVE" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" \
  --baseline-label "Mature R275 nnU-Net" \
  --improved-label "Balanced instance-prior nnU-Net"
echo "[R280] done $(date -Is)"
