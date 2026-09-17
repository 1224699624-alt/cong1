#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
R202="$ROOT/outputs/nnunet/r319_r316b_gated_prior/source/r202_checkpoint_best.pth"
SOURCE_PLAN="$ROOT/outputs/nnunet/r319_r316b_gated_prior/source/nnUNetPlans.json"
DATASET="Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
R260_MASKS="$ROOT/outputs/ablations/r260_mature_prior/TSRS_RSNA-Epiphysis/val/masks"
LOG="$ROOT/outputs/bridge_logs/r316c_dual_route.log"

cd "$ROOT"
mkdir -p outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
export PATH="$BIN:$PATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONHASHSEED=260
export nnUNet_compile=false nnUNet_n_proc_DA=0 R315_EXPECTED_DATASET="$DATASET"

echo "[R316C] start $(date -Is)"
test "$(sha256sum "$R202" | cut -d' ' -f1)" = "65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7"
test -s "$SOURCE_PLAN"
$PY -c "import importlib.metadata; assert importlib.metadata.version('nnunetv2') == '2.8.1'"
$PY -m py_compile \
  scripts/r316c_pair_inputs.py \
  scripts/train_r258b_prediction_relation_prior.py \
  scripts/train_r316_safe_relation_selector.py \
  scripts/build_r259_frozen_prior_maps.py \
  scripts/build_r316b_gated_prior_maps.py \
  scripts/nnunet_trainers/nnUNetTrainerR316C.py \
  scripts/summarize_r316c_dual_route.py

if [[ "${R316C_SKIP_PRIORS:-0}" != "1" ]]; then
echo "[R316C] train continuous high-resolution prior $(date -Is)"
$PY scripts/train_r258b_prediction_relation_prior.py \
  --r316c --crop-size 256 --global-size 128 \
  --center-sigma-rel .15 --center-sigma-min 3 --center-sigma-max 12 --center-jitter-rel .03 \
  --epochs 6 --batch-size 16 --workers 6 --lr 3e-4 \
  --output-dir outputs/pair_prior/r316c_continuous \
  --result-json outputs/analysis/r316c_continuous_prior.json \
  --pair-manifest-csv outputs/analysis/r316c_continuous_pairs.csv
$PY scripts/build_r259_frozen_prior_maps.py \
  --r316c --crop-size 256 --global-size 128 \
  --center-sigma-rel .15 --center-sigma-min 3 --center-sigma-max 12 \
  --r258b-result outputs/analysis/r316c_continuous_prior.json \
  --checkpoint-dir outputs/pair_prior/r316c_continuous \
  --output-root outputs/priors/r316c_continuous \
  --batch-size 32 --device cuda --png-only

echo "[R316C] train gated high-resolution prior $(date -Is)"
$PY scripts/train_r316_safe_relation_selector.py \
  --r316c --crop-size 256 --global-size 128 \
  --center-sigma-rel .15 --center-sigma-min 3 --center-sigma-max 12 --center-jitter-rel .03 \
  --epochs 8 --batch-size 16 --workers 6 --lr 3e-4 \
  --hard-negative-quantile .20 --require-full-data \
  --output-dir outputs/pair_prior/r316c_gated \
  --result-json outputs/analysis/r316c_gated_prior.json
$PY scripts/build_r316b_gated_prior_maps.py \
  --r316c --crop-size 256 --global-size 128 \
  --center-sigma-rel .15 --center-sigma-min 3 --center-sigma-max 12 \
  --checkpoint outputs/pair_prior/r316c_gated/safe_relation_seed3161_final.pt \
  --output-root outputs/priors/r316c_gated \
  --threshold .65 --batch-size 32 --device cuda --png-only
else
  echo "[R316C] reuse completed continuous/gated priors $(date -Is)"
  test -s outputs/analysis/r316c_continuous_prior.json
  test -s outputs/analysis/r316c_gated_prior.json
  test "$(find outputs/priors/r316c_continuous/train -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 875
  test "$(find outputs/priors/r316c_continuous/val -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 96
  test "$(find outputs/priors/r316c_gated/train -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 875
  test "$(find outputs/priors/r316c_gated/val -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 96
fi

TRAINER_DIR="$($PY -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR260MaturePrior.py "$TRAINER_DIR/nnUNetTrainerR260MaturePrior.py"
cp scripts/nnunet_trainers/nnUNetTrainerR316C.py "$TRAINER_DIR/nnUNetTrainerR316C.py"
$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR260MaturePrior.py" "$TRAINER_DIR/nnUNetTrainerR316C.py"

run_route() {
  local route="$1" prior="$2"
  local run="$ROOT/outputs/nnunet/r316c_${route}"
  local data="$run/data"
  export nnUNet_raw="$data/nnUNet_raw"
  export nnUNet_preprocessed="$data/nnUNet_preprocessed"
  export nnUNet_results="$run/nnUNet_results"
  export R260_ADAPTED_CHECKPOINT="$run/r202_two_channel_zero_prior.pth"
  mkdir -p "$run"
  if [[ "$route" == "continuous" && "${R316C_REUSE_PREPARED_CONTINUOUS:-0}" == "1" ]]; then
    echo "[R316C] reuse prepared route=$route $(date -Is)"
    test -s "$nnUNet_preprocessed/$DATASET/nnUNetPlans.json"
    test -s "$nnUNet_preprocessed/$DATASET/splits_final.json"
    test -s "$R260_ADAPTED_CHECKPOINT"
    test "$(find "$run/imagesValPrior" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq 192
  else
  echo "[R316C] prepare route=$route $(date -Is)"
  $PY scripts/prepare_r260_mature_prior_dataset.py \
    --prior-root "$prior" --nnunet-root "$data" --allow-zero-prior --allow-png-only --overwrite
  nnUNetv2_extract_fingerprint -d 204 -np 8 --verify_dataset_integrity --clean
  local plan_dir="$nnUNet_preprocessed/$DATASET"
  mkdir -p "$plan_dir"
  cp "$SOURCE_PLAN" "$plan_dir/nnUNetPlans.json"
  cp "$nnUNet_raw/$DATASET/dataset.json" "$plan_dir/dataset.json"
  nnUNetv2_preprocess -d 204 -plans_name nnUNetPlans -c 2d -np 8
  cp "$data/splits_final_r260.json" "$plan_dir/splits_final.json"
  $PY scripts/adapt_r202_checkpoint_two_channel.py --input "$R202" --output "$R260_ADAPTED_CHECKPOINT"
  $PY scripts/prepare_r260_val_inputs.py --prior-root "$prior/val" --output "$run/imagesValPrior"
  fi

  for alpha in 020 035; do
    local trainer="nnUNetTrainerR316CAlpha${alpha}"
    local trainer_root="$nnUNet_results/$DATASET/${trainer}__nnUNetPlans__2d"
    local pred="$run/predictions_alpha${alpha}" masks="$run/masks_alpha${alpha}"
    echo "[R316C] train route=$route alpha=$alpha $(date -Is)"
    rm -rf "$trainer_root" "$pred" "$masks"
    nnUNetv2_train 204 2d 0 -tr "$trainer"
    test -s "$trainer_root/fold_0/checkpoint_best.pth"
    nnUNetv2_predict -i "$run/imagesValPrior" -o "$pred" -d 204 -c 2d -f 0 -tr "$trainer" -chk checkpoint_best.pth
    $PY scripts/convert_nnunet_predictions.py --pred-dir "$pred" --mask-dir "$masks" --expected-count 96 --strip-prefix val_
    $PY scripts/evaluate_r201_single_split.py \
      --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
      --pred-dir "$masks" \
      --output "outputs/analysis/r316c_${route}_alpha${alpha}_original_val_r201.json" \
      --expected-count 96
    rm -f "$trainer_root/fold_0/checkpoint_final.pth"
    rm -rf "$pred"
    $PY scripts/render_r259_nnunet_prior_comparison.py \
      --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
      --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
      --baseline-dir "$R260_MASKS" --improved-dir "$masks" \
      --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
      --output-dir "outputs/visualizations/r316c_${route}_alpha${alpha}_vs_r260" \
      --num-cases 12 --baseline-label "R260" --improved-label "R316C ${route} alpha=${alpha}"
  done
  rm -rf "$data/nnUNet_raw" "$data/nnUNet_preprocessed/$DATASET/nnUNetPlans_2d"
}

run_route continuous "$ROOT/outputs/priors/r316c_continuous"
run_route gated "$ROOT/outputs/priors/r316c_gated"
$PY scripts/summarize_r316c_dual_route.py
echo "[R316C] done $(date -Is)"
