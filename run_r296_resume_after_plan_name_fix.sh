#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
cd "$ROOT"
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=296 nnUNet_compile=false

DATA="outputs/nnunet/r293_dualbranch/data"
RUN="outputs/nnunet/r296_plan_aligned_local_fusion"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export nnUNet_extTrainer="$RUN/ext_trainer"
export R293_SOURCE_CHECKPOINT="outputs/nnunet/r275_mature_control/nnUNet_results/Dataset202_TSRS_RSNAEpiphysis2D/nnUNetTrainerR275ControlMature__nnUNetPlans__2d/fold_0/checkpoint_final.pth"

LOG="outputs/bridge_logs/r296_resume_after_plan_name_fix.log"
mkdir -p outputs/bridge_logs "$nnUNet_extTrainer"
exec > >(tee -a "$LOG") 2>&1
echo "[R296-resume] start $(date -Is)"

# The first R296 plan file was correctly aligned to 512x512, but retained the
# internal name nnUNetPlans. Reuse its completed A checkpoint under that actual
# result identifier. Do not regenerate the plan until R296 is complete.
PLAN="$nnUNet_preprocessed/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/R296Plans.json"
A_DIR="$nnUNet_results/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/nnUNetTrainerR294LocalFusion__nnUNetPlans__2d/fold_0"
B_DIR="$nnUNet_results/Dataset293_TSRS_RSNAEpiphysisDualBranch2D/nnUNetTrainerR294LocalFusionGap__nnUNetPlans__2d/fold_0"
test "$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["configurations"]["2d"]["patch_size"])' "$PLAN")" = "[512, 512]"
test -s "$A_DIR/checkpoint_final.pth"
test -s "$A_DIR/r294_loss_dynamics.jsonl"

predict_and_audit() {
  local trainer="$1" tag="$2" label="$3"
  local pred="$RUN/predictions_${tag}"
  local active="outputs/ablations/r296_${tag}/TSRS_RSNA-Epiphysis/val/masks"
  local vis="outputs/visualizations/r296_${tag}_original_val"
  rm -rf "$pred" "$active" "$vis"
  # -p nnUNetPlans selects the existing result folder; prediction loads the
  # stored 512x512 plans.json from that folder.
  nnUNetv2_predict -i outputs/nnunet/r293_dualbranch/imagesValXrayOnly -o "$pred" \
    -d 293 -c 2d -f 0 -tr "$trainer" -p nnUNetPlans -chk checkpoint_final.pth
  "$PY" scripts/convert_nnunet_predictions.py --pred-dir "$pred" \
    --mask-dir "$active" --expected-count 96 --strip-prefix val_
  "$PY" scripts/evaluate_r201_single_split.py \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$active" \
    --output "outputs/analysis/r296_${tag}_original_val_r201.json" --expected-count 96
  "$PY" scripts/render_r259_nnunet_prior_comparison.py \
    --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
    --baseline-dir outputs/predictions/r275_mature_original_val/control \
    --improved-dir "$active" \
    --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
    --output-dir "$vis" --num-cases 12 --xray-label "Original X-ray" \
    --baseline-label "Mature R275 nnU-Net" --improved-label "$label"
}

echo "[R296-resume] audit completed A $(date -Is)"
predict_and_audit nnUNetTrainerR294LocalFusion local_fusion_control \
  "R296-A plan-aligned local fusion"

echo "[R296-resume] train B $(date -Is)"
rm -rf "${B_DIR%/fold_0}"
nnUNetv2_train 293 2d 0 -tr nnUNetTrainerR294LocalFusionGap -p R296Plans
test -s "$B_DIR/checkpoint_final.pth"
test -s "$B_DIR/r294b_loss_dynamics.jsonl"
echo "[R296-resume] audit B $(date -Is)"
predict_and_audit nnUNetTrainerR294LocalFusionGap local_fusion_gap \
  "R296-B plan-aligned local fusion + interface loss"

echo "[R296-resume] done $(date -Is)"
