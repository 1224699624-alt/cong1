#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"; PY="/root/miniconda3/bin/python"; BIN="/root/miniconda3/bin"; cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONHASHSEED=282 nnUNet_compile=false
R279="outputs/nnunet/r279_instance_prior"; RUN="outputs/nnunet/r282_gradnorm_instance_prior"
export nnUNet_raw="$R279/data/nnUNet_raw" nnUNet_preprocessed="$R279/data/nnUNet_preprocessed" nnUNet_results="$RUN/nnUNet_results" nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r282_gradnorm_instance_prior_nnunet.log"; mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"; exec > >(tee -a "$LOG") 2>&1; echo "[R282] start $(date -Is)"
ADAPTED="$R279/r275_control_two_channel_zero_prior.pth"; test "$(sha256sum "$ADAPTED"|cut -d' ' -f1)" = "89eda6b07c0e6c37c3d41abf295d97f4f06c5ea30d4d061ef9ddcb1680128f5a"
test "$(/root/miniconda3/bin/python -c "import json; print(json.load(open('outputs/analysis/r281_low_support_gate.json'))['decision'])")" = "no_go_use_gradient_normalization"
$PY -m py_compile scripts/nnunet_trainers/nnUNetTrainerR280BalancedInstancePrior.py scripts/nnunet_trainers/nnUNetTrainerR282GradNormInstancePrior.py scripts/summarize_r281_gate.py
cp scripts/nnunet_trainers/nnUNetTrainerR280BalancedInstancePrior.py "$nnUNet_extTrainer/"; cp scripts/nnunet_trainers/nnUNetTrainerR282GradNormInstancePrior.py "$nnUNet_extTrainer/"; export R280_ADAPTED_CHECKPOINT="$ADAPTED"
OUT="$nnUNet_results/Dataset279_TSRS_RSNAEpiphysisInstancePrior2D/nnUNetTrainerR282GradNormInstancePrior__nnUNetPlans__2d/fold_0"; rm -rf "${OUT%/fold_0}"
nnUNetv2_train 279 2d 0 -tr nnUNetTrainerR282GradNormInstancePrior; test -s "$OUT/checkpoint_best.pth"; test -s "$OUT/r282_early_stop.json"
rm -rf "$RUN/predictions_prior"; nnUNetv2_predict -i "$R279/imagesValPrior" -o "$RUN/predictions_prior" -d 279 -c 2d -f 0 -tr nnUNetTrainerR282GradNormInstancePrior -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r282_gradnorm_instance_prior/TSRS_RSNA-Epiphysis/val/masks"; rm -rf "$ACTIVE"; $PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_prior" --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_
BASE_JSON="outputs/analysis/r282_r275_control_original_val_r201.json"; RESULT_JSON="outputs/analysis/r282_gradnorm_instance_prior_original_val_r201.json"
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir outputs/predictions/r275_mature_original_val/control --output "$BASE_JSON" --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$ACTIVE" --output "$RESULT_JSON" --expected-count 96
$PY scripts/summarize_r281_gate.py --baseline "$BASE_JSON" --result "$RESULT_JSON" --output outputs/analysis/r282_gradnorm_gate.json
VIS="outputs/visualizations/r282_gradnorm_instance_prior_original_val"; rm -rf "$VIS"; $PY scripts/render_r259_nnunet_prior_comparison.py --image-dir data/raw/TSRS_RSNA-Epiphysis/val --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --baseline-dir outputs/predictions/r275_mature_original_val/control --improved-dir "$ACTIVE" --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv --output-dir "$VIS" --num-cases 12 --xray-label "Original X-ray" --baseline-label "Mature R275 nnU-Net" --improved-label "GradNorm instance-prior nnU-Net"
echo "[R282] done $(date -Is)"
