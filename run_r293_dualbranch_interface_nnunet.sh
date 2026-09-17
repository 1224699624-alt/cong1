#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=293 nnUNet_compile=false
RUN="outputs/nnunet/r293_dualbranch"
DATA="$RUN/data"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/r293_dualbranch_interface_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R293] start $(date -Is)"

SOURCE="outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_best.pth"
SOURCE_SHA="557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27"
BASE="outputs/predictions/r275_mature_original_val/control"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA"
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96
export R293_SOURCE_CHECKPOINT="$SOURCE"

$PY -m py_compile scripts/prepare_r293_dualbranch_dataset.py \
  scripts/prepare_r293_val_inputs.py \
  scripts/nnunet_trainers/nnUNetTrainerR293DualBranch.py \
  scripts/nnunet_trainers/nnUNetTrainerR293DualBranchGap.py
$PY scripts/prepare_r293_dualbranch_dataset.py --overwrite
nnUNetv2_plan_and_preprocess -d 293 -c 2d --verify_dataset_integrity
cp "$DATA/splits_final_r293.json" \
  "$nnUNet_preprocessed/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/splits_final.json"
cp scripts/nnunet_trainers/nnUNetTrainerR293DualBranch.py "$nnUNet_extTrainer/"
cp scripts/nnunet_trainers/nnUNetTrainerR293DualBranchGap.py "$nnUNet_extTrainer/"
$PY scripts/prepare_r293_val_inputs.py --output "$RUN/imagesValXrayOnly"

run_variant() {
  local trainer="$1"
  local tag="$2"
  local label="$3"
  local out_base="$nnUNet_results/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/${trainer}__nnUNetPlans__2d"
  local out="$out_base/fold_0"
  rm -rf "$out_base"
  echo "[R293] training $tag $(date -Is)"
  nnUNetv2_train 293 2d 0 -tr "$trainer"
  test -s "$out/checkpoint_best.pth"
  test -s "$out/r293_loss_dynamics.jsonl" -o -s "$out/r293b_loss_dynamics.jsonl"

  local pred="$RUN/predictions_${tag}"
  rm -rf "$pred"
  nnUNetv2_predict -i "$RUN/imagesValXrayOnly" -o "$pred" \
    -d 293 -c 2d -f 0 -tr "$trainer" -chk checkpoint_best.pth
  local active="outputs/ablations/r293_${tag}/TSRS_RSNA-Epiphysis/val/masks"
  rm -rf "$active"
  $PY scripts/convert_nnunet_predictions.py --pred-dir "$pred" \
    --mask-dir "$active" --expected-count 96 --strip-prefix val_
  $PY scripts/evaluate_r201_single_split.py \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$active" \
    --output "outputs/analysis/r293_${tag}_original_val_r201.json" --expected-count 96
  local vis="outputs/visualizations/r293_${tag}_original_val"
  rm -rf "$vis"
  $PY scripts/render_r259_nnunet_prior_comparison.py \
    --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
    --baseline-dir "$BASE" --improved-dir "$active" \
    --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
    --output-dir "$vis" --num-cases 12 --xray-label "Original X-ray" \
    --baseline-label "Mature R275 nnU-Net" --improved-label "$label"
}

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$BASE" \
  --output outputs/analysis/r293_r275_control_original_val_r201.json --expected-count 96

run_variant nnUNetTrainerR293DualBranch dualbranch_control \
  "R293-A explicit dual-branch nnU-Net"
run_variant nnUNetTrainerR293DualBranchGap dualbranch_gap \
  "R293-B dual-branch + interface prior/loss"
echo "[R293] done $(date -Is)"
