#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED_OVERRIDE:-294}" nnUNet_compile=false
EXP_ID="${EXP_ID:-r294}"
EXP_LABEL="${EXP_LABEL:-R294}"
R293_DATA="outputs/nnunet/r293_dualbranch/data"
RUN="${RUN_OVERRIDE:-outputs/nnunet/r294_local_fusion}"
export nnUNet_raw="$R293_DATA/nnUNet_raw"
export nnUNet_preprocessed="$R293_DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
LOG="outputs/bridge_logs/${EXP_ID}_local_fusion_nnunet.log"
mkdir -p outputs/bridge_logs "$RUN" "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[$EXP_LABEL] start $(date -Is)"

SOURCE="${SOURCE_OVERRIDE:-outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_best.pth}"
SOURCE_SHA="${SOURCE_SHA_OVERRIDE:-557f78d7bd5f2987c6e224b87461f939b006df7822ab3170167e0502d9092c27}"
PRED_CHK="${PRED_CHK_OVERRIDE:-checkpoint_best.pth}"
PLANS_ID="${PLANS_ID_OVERRIDE:-nnUNetPlans}"
BASE="outputs/predictions/r275_mature_original_val/control"
test "$(sha256sum "$SOURCE" | cut -d' ' -f1)" = "$SOURCE_SHA"
test "$(find "$BASE" -maxdepth 1 -name '*.png' | wc -l)" = 96
test -s "$nnUNet_preprocessed/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/nnUNetPlans.json"
test -d outputs/nnunet/r293_dualbranch/imagesValXrayOnly
export R293_SOURCE_CHECKPOINT="$SOURCE"

$PY -m py_compile \
  scripts/nnunet_trainers/nnUNetTrainerR293DualBranch.py \
  scripts/nnunet_trainers/nnUNetTrainerR293DualBranchGap.py \
  scripts/nnunet_trainers/nnUNetTrainerR294LocalFusion.py \
  scripts/nnunet_trainers/nnUNetTrainerR294LocalFusionGap.py
cp scripts/nnunet_trainers/nnUNetTrainerR293DualBranch.py "$nnUNet_extTrainer/"
cp scripts/nnunet_trainers/nnUNetTrainerR293DualBranchGap.py "$nnUNet_extTrainer/"
cp scripts/nnunet_trainers/nnUNetTrainerR294LocalFusion.py "$nnUNet_extTrainer/"
cp scripts/nnunet_trainers/nnUNetTrainerR294LocalFusionGap.py "$nnUNet_extTrainer/"

run_variant() {
  local trainer="$1"
  local tag="$2"
  local label="$3"
  local out_base="$nnUNet_results/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/${trainer}__${PLANS_ID}__2d"
  local out="$out_base/fold_0"
  rm -rf "$out_base"
  echo "[$EXP_LABEL] training $tag $(date -Is)"
  nnUNetv2_train 293 2d 0 -tr "$trainer" -p "$PLANS_ID"
  test -s "$out/checkpoint_best.pth"
  test -s "$out/r294_loss_dynamics.jsonl" -o -s "$out/r294b_loss_dynamics.jsonl"

  local pred="$RUN/predictions_${tag}"
  rm -rf "$pred"
  nnUNetv2_predict -i outputs/nnunet/r293_dualbranch/imagesValXrayOnly -o "$pred" \
    -d 293 -c 2d -f 0 -tr "$trainer" -p "$PLANS_ID" -chk "$PRED_CHK"
  local active="outputs/ablations/${EXP_ID}_${tag}/TSRS_RSNA-Epiphysis/val/masks"
  rm -rf "$active"
  $PY scripts/convert_nnunet_predictions.py --pred-dir "$pred" \
    --mask-dir "$active" --expected-count 96 --strip-prefix val_
  $PY scripts/evaluate_r201_single_split.py \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$active" \
    --output "outputs/analysis/${EXP_ID}_${tag}_original_val_r201.json" --expected-count 96
  local vis="outputs/visualizations/${EXP_ID}_${tag}_original_val"
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
  --output "outputs/analysis/${EXP_ID}_r275_control_original_val_r201.json" --expected-count 96
run_variant nnUNetTrainerR294LocalFusion local_fusion_control \
  "R294-A bounded local-fusion nnU-Net"
run_variant nnUNetTrainerR294LocalFusionGap local_fusion_gap \
  "R294-B local-fusion + decoupled interface loss"
echo "[$EXP_LABEL] done $(date -Is)"
