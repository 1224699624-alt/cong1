#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
R202="$ROOT/outputs/nnunet/r319_r316b_gated_prior/source/r202_checkpoint_best.pth"
SOURCE_PLAN="$ROOT/outputs/nnunet/r319_r316b_gated_prior/source/nnUNetPlans.json"
DATASET="Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
RUN="$ROOT/outputs/nnunet/r317_continuous"
PRIOR_MODELS="$ROOT/outputs/pair_prior/r317_continuous"
PRIORS="$ROOT/outputs/priors/r317_continuous"
R260_MASKS="$ROOT/outputs/ablations/r260_mature_prior/TSRS_RSNA-Epiphysis/val/masks"
LOG="$ROOT/outputs/bridge_logs/r317_continuous_maturity.log"

cd "$ROOT"
mkdir -p outputs/bridge_logs outputs/analysis
exec > >(tee -a "$LOG") 2>&1
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONHASHSEED=260
export nnUNet_compile=false nnUNet_n_proc_DA=0 R315_EXPECTED_DATASET="$DATASET"

echo "[R317] start $(date -Is)"
test "$(sha256sum "$R202" | cut -d' ' -f1)" = "65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7"
test -s "$SOURCE_PLAN"
test "$(find data/raw/TSRS_RSNA-Epiphysis/train -maxdepth 1 -type f | wc -l)" -ge 875
test "$(find data/raw/TSRS_RSNA-Epiphysis/val -maxdepth 1 -type f | wc -l)" -ge 96
test "$(find data/raw/TSRS_RSNA-Epiphysis/val_labels -maxdepth 1 -type f | wc -l)" -eq 96
! find data/raw/TSRS_RSNA-Epiphysis -type f | grep -qi 'articular'
$PY -c "import importlib.metadata; assert importlib.metadata.version('nnunetv2') == '2.8.1'"
$PY -m py_compile \
  scripts/r316c_pair_inputs.py scripts/train_r258b_prediction_relation_prior.py \
  scripts/build_r259_frozen_prior_maps.py scripts/prepare_r260_mature_prior_dataset.py \
  scripts/nnunet_trainers/nnUNetTrainerR317.py scripts/summarize_r317_checkpoint_selection.py

echo "[R317] train 18-epoch continuous-prior convergence audit $(date -Is)"
rm -rf "$PRIOR_MODELS" "$PRIORS" "$RUN"
$PY scripts/train_r258b_prediction_relation_prior.py \
  --r316c --r317 --crop-size 256 --global-size 128 \
  --center-sigma-rel .15 --center-sigma-min 3 --center-sigma-max 12 --center-jitter-rel .03 \
  --epochs 18 --batch-size 16 --workers 6 --lr 3e-4 \
  --output-dir "$PRIOR_MODELS" \
  --result-json outputs/analysis/r317_continuous_prior_convergence.json \
  --pair-manifest-csv outputs/analysis/r317_continuous_pairs.csv

echo "[R317] build selected mature continuous prior maps $(date -Is)"
$PY scripts/build_r259_frozen_prior_maps.py \
  --r316c --r317 --crop-size 256 --global-size 128 \
  --center-sigma-rel .15 --center-sigma-min 3 --center-sigma-max 12 \
  --r258b-result outputs/analysis/r317_continuous_prior_convergence.json \
  --checkpoint-dir "$PRIOR_MODELS" --output-root "$PRIORS" \
  --batch-size 32 --device cuda --png-only

TRAINER_DIR="$($PY -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR260MaturePrior.py "$TRAINER_DIR/nnUNetTrainerR260MaturePrior.py"
cp scripts/nnunet_trainers/nnUNetTrainerR317.py "$TRAINER_DIR/nnUNetTrainerR317.py"
$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR260MaturePrior.py" "$TRAINER_DIR/nnUNetTrainerR317.py"

DATA="$RUN/data"
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export R260_ADAPTED_CHECKPOINT="$RUN/r202_two_channel_zero_prior.pth"
mkdir -p "$RUN"

echo "[R317] prepare isolated nnU-Net data $(date -Is)"
$PY scripts/prepare_r260_mature_prior_dataset.py \
  --prior-root "$PRIORS" --nnunet-root "$DATA" --allow-zero-prior --allow-png-only --overwrite
nnUNetv2_extract_fingerprint -d 204 -np 8 --verify_dataset_integrity --clean
PLAN_DIR="$nnUNet_preprocessed/$DATASET"
mkdir -p "$PLAN_DIR"
cp "$SOURCE_PLAN" "$PLAN_DIR/nnUNetPlans.json"
cp "$nnUNet_raw/$DATASET/dataset.json" "$PLAN_DIR/dataset.json"
nnUNetv2_preprocess -d 204 -plans_name nnUNetPlans -c 2d -np 8
cp "$DATA/splits_final_r260.json" "$PLAN_DIR/splits_final.json"
$PY scripts/adapt_r202_checkpoint_two_channel.py --input "$R202" --output "$R260_ADAPTED_CHECKPOINT"
$PY scripts/prepare_r260_val_inputs.py --prior-root "$PRIORS/val" --output "$RUN/imagesValPrior"

TRAINER="nnUNetTrainerR317Continuous035"
TRAINER_ROOT="$nnUNet_results/$DATASET/${TRAINER}__nnUNetPlans__2d"
echo "[R317] train alpha=0.035 with periodic checkpoints $(date -Is)"
nnUNetv2_train 204 2d 0 -tr "$TRAINER"
FOLD="$TRAINER_ROOT/fold_0"
test -s "$FOLD/checkpoint_best.pth"
test -s "$FOLD/r317_periodic_checkpoints.json"

evaluate_checkpoint() {
  local label="$1" checkpoint="$2"
  local pred="$RUN/predictions_${label}" masks="$RUN/masks_${label}"
  echo "[R317] evaluate checkpoint=$label $(date -Is)"
  rm -rf "$pred" "$masks"
  nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$pred" -d 204 -c 2d -f 0 -tr "$TRAINER" -chk "$checkpoint"
  $PY scripts/convert_nnunet_predictions.py --pred-dir "$pred" --mask-dir "$masks" --expected-count 96 --strip-prefix val_
  $PY scripts/evaluate_r201_single_split.py \
    --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$masks" \
    --output "outputs/analysis/r317_continuous_${label}_original_val_r201.json" --expected-count 96
  rm -rf "$pred"
}

for checkpoint in "$FOLD"/checkpoint_epoch_*.pth; do
  test -s "$checkpoint"
  epoch="${checkpoint##*checkpoint_epoch_}"; epoch="${epoch%.pth}"
  evaluate_checkpoint "epoch${epoch}" "$(basename "$checkpoint")"
done
evaluate_checkpoint "ema_best" "checkpoint_best.pth"

echo "[R317] R201 hard-gate checkpoint selection $(date -Is)"
$PY scripts/summarize_r317_checkpoint_selection.py --trainer-dir "$FOLD"
SELECTED_LABEL="$($PY -c "import json; print(json.load(open('outputs/analysis/r317_continuous_checkpoint_selection.json'))['selected']['label'])")"
SELECTED_CKPT="$($PY -c "import json; print(json.load(open('outputs/analysis/r317_continuous_checkpoint_selection.json'))['selected']['checkpoint'])")"
cp "$SELECTED_CKPT" "$FOLD/checkpoint_r201_selected.pth"
rm -rf "$RUN/masks_selected"
cp -a "$RUN/masks_${SELECTED_LABEL}" "$RUN/masks_selected"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$R260_MASKS" --improved-dir "$RUN/masks_selected" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir outputs/visualizations/r317_continuous_selected_vs_r260 \
  --num-cases 12 --baseline-label R260 --improved-label "R317 continuous selected"

rm -f "$FOLD/checkpoint_final.pth" "$FOLD/checkpoint_latest.pth"
rm -rf "$DATA/nnUNet_raw" "$DATA/nnUNet_preprocessed/$DATASET/nnUNetPlans_2d"
echo "[R317] done $(date -Is)"
