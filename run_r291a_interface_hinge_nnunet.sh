#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=291 nnUNet_compile=false

RUN="outputs/nnunet/r291a_interface_hinge"
DATA="$RUN/data"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r291a_interface_hinge_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R291-A] start $(date -Is)"

SOURCE="outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
SOURCE_SHA="557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27"
BASE="outputs/predictions/r275_mature_original_val/control"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA"
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96
test -s outputs/analysis/r291_interface_target_audit/audit.json
test "$("$PY" -c "import json; a=json.load(open('outputs/analysis/r291_interface_target_audit/audit.json')); print(all(a['gate'].values()))")" = True

$PY -m py_compile scripts/prepare_r291_interface_aux_dataset.py \
  scripts/prepare_r291_gtfree_val_inputs.py scripts/adapt_checkpoint_four_channel_r291.py \
  scripts/nnunet_trainers/nnUNetTrainerR291AInterfaceHinge.py
$PY scripts/prepare_r291_interface_aux_dataset.py --overwrite
nnUNetv2_plan_and_preprocess -d 291 -c 2d --verify_dataset_integrity
cp "$DATA/splits_final_r291.json" \
  "$nnUNet_preprocessed/Dataset291_TSRS_RSNAEpiphysisInterfaceAux2D/splits_final.json"

ADAPTED="$RUN/r275_control_four_channel_zero_aux.pth"
$PY scripts/adapt_checkpoint_four_channel_r291.py --input "$SOURCE" --output "$ADAPTED" \
  --expected-sha256 "$SOURCE_SHA" | tee "$RUN/adaptation_manifest.json"
export R291_ADAPTED_CHECKPOINT="$ADAPTED"
cp scripts/nnunet_trainers/nnUNetTrainerR291AInterfaceHinge.py "$nnUNet_extTrainer/"
$PY scripts/prepare_r291_gtfree_val_inputs.py --output "$RUN/imagesValXrayOnly"

OUT="$nnUNet_results/Dataset291_TSRS_RSNAEpiphysisInterfaceAux2D/nnUNetTrainerR291AInterfaceHinge__nnUNetPlans__2d/fold_0"
rm -rf "${OUT%/fold_0}"
nnUNetv2_train 291 2d 0 -tr nnUNetTrainerR291AInterfaceHinge
test -s "$OUT/checkpoint_best.pth"
test -s "$OUT/r291a_loss_dynamics.jsonl"

rm -rf "$RUN/predictions_xray_only"
nnUNetv2_predict -i "$RUN/imagesValXrayOnly" -o "$RUN/predictions_xray_only" \
  -d 291 -c 2d -f 0 -tr nnUNetTrainerR291AInterfaceHinge -chk checkpoint_best.pth
ACTIVE="outputs/ablations/r291a_interface_hinge/TSRS_RSNA-Epiphysis/val/masks"
rm -rf "$ACTIVE"
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/predictions_xray_only" \
  --mask-dir "$ACTIVE" --expected-count 96 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$BASE" --output outputs/analysis/r291a_r275_control_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$ACTIVE" --output outputs/analysis/r291a_interface_hinge_original_val_r201.json --expected-count 96

VIS="outputs/visualizations/r291a_interface_hinge_original_val"
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$BASE" --improved-dir "$ACTIVE" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" --baseline-label "Mature R275 nnU-Net" \
  --improved-label "R291-A local-interface hinge nnU-Net"
echo "[R291-A] done $(date -Is)"
