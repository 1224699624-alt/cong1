#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/YOLO_SAM_generic_src"
PY="/root/miniconda3/bin/python"
BIN="/root/miniconda3/bin"
RUN="$ROOT/outputs/nnunet/r319_r316b_gated_prior"
DATA="$RUN/data"
PRIOR_ROOT="$ROOT/outputs/priors/r319_r316b_gated_relation"
DATASET="Dataset204_TSRS_RSNAEpiphysisMaturePrior2D"
R202="$RUN/source/r202_checkpoint_best.pth"
SOURCE_PLAN="$RUN/source/nnUNetPlans.json"
R316B="$ROOT/outputs/pair_prior/r316b_full_quantile_safe_relation/safe_relation_seed3161_final.pt"
ADAPTED="$RUN/r202_two_channel_zero_prior.pth"
R260_MASKS="$ROOT/outputs/ablations/r260_mature_prior/TSRS_RSNA-Epiphysis/val/masks"
LOG="$ROOT/outputs/bridge_logs/r319_r316b_gated_nnunet.log"

cd "$ROOT"
mkdir -p "$RUN" outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
export PATH="$BIN:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=260
export nnUNet_compile=false
export nnUNet_n_proc_DA=0
export nnUNet_raw="$DATA/nnUNet_raw"
export nnUNet_preprocessed="$DATA/nnUNet_preprocessed"
export nnUNet_results="$RUN/nnUNet_results"
export R260_ADAPTED_CHECKPOINT="$ADAPTED"
export R315_EXPECTED_DATASET="$DATASET"

echo "[R319] start $(date -Is)"
test "$(sha256sum "$R202" | cut -d' ' -f1)" = "65b81565ea77d48b1c91017188533e49f1d0d0d4ec557540b73f6779764397a7"
test "$(sha256sum "$R316B" | cut -d' ' -f1)" = "462e538f1f29a5f04ffb9beaddf1583184754043caf478f7fba9af72e7e01f2e"
test -s "$SOURCE_PLAN"
$PY -c "import importlib.metadata; assert importlib.metadata.version('nnunetv2') == '2.8.1'"

$PY scripts/build_r316b_gated_prior_maps.py \
  --checkpoint "$R316B" \
  --output-root "$PRIOR_ROOT" \
  --threshold 0.65 \
  --device cuda

$PY scripts/prepare_r260_mature_prior_dataset.py \
  --prior-root "$PRIOR_ROOT" \
  --nnunet-root "$DATA" \
  --allow-zero-prior \
  --overwrite

nnUNetv2_extract_fingerprint -d 204 -np 8 --verify_dataset_integrity --clean
PLAN_DIR="$nnUNet_preprocessed/$DATASET"
mkdir -p "$PLAN_DIR"
cp "$SOURCE_PLAN" "$PLAN_DIR/nnUNetPlans.json"
cp "$nnUNet_raw/$DATASET/dataset.json" "$PLAN_DIR/dataset.json"
nnUNetv2_preprocess -d 204 -plans_name nnUNetPlans -c 2d -np 8
cp "$DATA/splits_final_r260.json" "$PLAN_DIR/splits_final.json"

$PY scripts/adapt_r202_checkpoint_two_channel.py --input "$R202" --output "$ADAPTED"
TRAINER_DIR="$($PY -c "from pathlib import Path; import nnunetv2; print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss')")"
cp scripts/nnunet_trainers/nnUNetTrainerR260MaturePrior.py "$TRAINER_DIR/nnUNetTrainerR260MaturePrior.py"
$PY -m py_compile "$TRAINER_DIR/nnUNetTrainerR260MaturePrior.py"
$PY scripts/prepare_r260_val_inputs.py --prior-root "$PRIOR_ROOT/val" --output "$RUN/imagesValPrior"

TRAINER_ROOT="$nnUNet_results/$DATASET/nnUNetTrainerR260MaturePrior__nnUNetPlans__2d"
rm -rf "$TRAINER_ROOT"
nnUNetv2_train 204 2d 0 -tr nnUNetTrainerR260MaturePrior
test -s "$TRAINER_ROOT/fold_0/checkpoint_best.pth"

rm -rf "$RUN/predictions" "$RUN/masks"
nnUNetv2_predict \
  -i "$RUN/imagesValPrior" \
  -o "$RUN/predictions" \
  -d 204 -c 2d -f 0 \
  -tr nnUNetTrainerR260MaturePrior \
  -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py \
  --pred-dir "$RUN/predictions" \
  --mask-dir "$RUN/masks" \
  --expected-count 96 \
  --strip-prefix val_
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --pred-dir "$RUN/masks" \
  --output outputs/analysis/r319_r316b_gated_nnunet_original_val_r201.json \
  --expected-count 96
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$R260_MASKS" \
  --improved-dir "$RUN/masks" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir outputs/visualizations/r319_r260_vs_r316b_nnunet_original_val \
  --num-cases 12 \
  --baseline-label "R260 mature prior nnU-Net" \
  --improved-label "R316B gated prior nnU-Net"

echo "[R319] done $(date -Is)"
