#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=292 nnUNet_compile=false
RUN="outputs/nnunet/r292_dualscale_input"
DATA="$RUN/data"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r292_dualscale_input_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R292] start $(date -Is)"

SOURCE="outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
SOURCE_SHA="557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27"
BASE="outputs/predictions/r275_mature_original_val/control"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA"
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96

$PY -m py_compile scripts/prepare_r292_dualscale_dataset.py \
  scripts/prepare_r292_dualscale_val_inputs.py scripts/adapt_checkpoint_two_channel_r279.py \
  scripts/nnunet_trainers/nnUNetTrainerR292DualScaleInput.py
$PY scripts/prepare_r292_dualscale_dataset.py --overwrite
nnUNetv2_plan_and_preprocess -d 292 -c 2d --verify_dataset_integrity
cp "$DATA/splits_final_r292.json" \
  "$nnUNet_preprocessed/Dataset292_TSRS_RSNAEpiphysisDualScale2D/splits_final.json"

ADAPTED="$RUN/r275_control_two_channel_zero_global.pth"
$PY scripts/adapt_checkpoint_two_channel_r279.py --input "$SOURCE" --output "$ADAPTED" \
  --expected-sha256 "$SOURCE_SHA" | tee "$RUN/adaptation_manifest.json"
export R292_ADAPTED_CHECKPOINT="$ADAPTED"
cp scripts/nnunet_trainers/nnUNetTrainerR292DualScaleInput.py "$nnUNet_extTrainer/"
$PY scripts/prepare_r292_dualscale_val_inputs.py --output "$RUN/imagesValDualScale"

OUT="$nnUNet_results/Dataset292_TSRS_RSNAEpiphysisDualScale2D/nnUNetTrainerR292DualScaleInput__nnUNetPlans__2d/fold_0"
rm -rf "${OUT%/fold_0}"
nnUNetv2_train 292 2d 0 -tr nnUNetTrainerR292DualScaleInput
test -s "$OUT/checkpoint_best.pth"

rm -rf "$RUN/predictions_dualscale"
nnUNetv2_predict -i "$RUN/imagesValDualScale" -o "$RUN/predictions_dualscale" \
  -d 292 -c 2d -f 0 -tr nnUNetTrainerR292DualScaleInput -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r292_dualscale_input/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$ACTIVE"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_dualscale" \
  --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$BASE" --output outputs/analysis/r292_r275_control_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$ACTIVE" --output outputs/analysis/r292_dualscale_input_original_val_r201.json --expected-count 96
VIS="outputs/visualizations/r292_dualscale_input_original_val"
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --baseline-dir "$BASE" \
  --improved-dir "$ACTIVE" --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 --xray-label "Original X-ray" \
  --baseline-label "Mature R275 nnU-Net" --improved-label "R292 dual-scale input nnU-Net"
echo "[R292] done $(date -Is)"
