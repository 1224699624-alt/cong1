#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/YOLO_SAM_generic_src
PY=/root/miniconda3/bin/python
BIN=/root/miniconda3/bin
cd "$ROOT"
export PATH="$BIN:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=260
export nnUNet_compile=false

DATA=outputs/nnunet/r308_clean_panels_r260/data
RUN=outputs/nnunet/r308_clean_panels_r260
export nnUNet_raw=$DATA/nnUNet_raw
export nnUNet_preprocessed=$DATA/nnUNet_preprocessed
export nnUNet_results=$RUN/nnUNet_results
export R306_SOURCE_CHECKPOINT=outputs/nnunet/r307_exact_r260_instance_aux_pair/source/r260_checkpoint_best.pth
export R306_EXPECTED_SOURCE_SHA256=7d1d8d2916115c2e29696b2a76881da2a7a78f3db91a5f6a9491297369052a6e
LOG=outputs/bridge_logs/r308_clean_panels_r260.log
mkdir -p "$RUN" outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
echo "[R308] start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
test "$(sha256sum "$R306_SOURCE_CHECKPOINT" | cut -d' ' -f1)" = "$R306_EXPECTED_SOURCE_SHA256"

$PY scripts/prepare_r308_clean_panel_r260.py \
  --selection configs/r308_tsrs_clean_panel_selection.json --output "$DATA" --overwrite
rm -rf "$nnUNet_preprocessed/Dataset309_TSRSR260CleanPanels2D"
nnUNetv2_plan_and_preprocess -d 309 --verify_dataset_integrity
cp "$nnUNet_raw/Dataset309_TSRSR260CleanPanels2D/splits_final.json" \
  "$nnUNet_preprocessed/Dataset309_TSRSR260CleanPanels2D/splits_final.json"

TRAINER_DST=$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss/nnUNetTrainerR306R260InstanceAux.py')
PY
)
cp scripts/nnunet_trainers/nnUNetTrainerR306R260InstanceAux.py "$TRAINER_DST"
$PY -m py_compile scripts/prepare_r308_clean_panel_r260.py "$TRAINER_DST"

rm -rf "$nnUNet_results/Dataset309_TSRSR260CleanPanels2D/nnUNetTrainerR308CleanContinue__nnUNetPlans__2d"
nnUNetv2_train 309 2d 0 -tr nnUNetTrainerR308CleanContinue

rm -rf "$RUN/pred_clean" "$RUN/masks_clean"
nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/pred_clean" -d 309 -c 2d -f 0 \
  -tr nnUNetTrainerR308CleanContinue -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_clean" --mask-dir "$RUN/masks_clean" \
  --expected-count 94 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir "$RUN/cleanValGT" --pred-dir "$RUN/frozenR260Masks" \
  --output outputs/analysis/r308_frozen_r260_clean_val94_r201.json --expected-count 94
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir "$RUN/cleanValGT" --pred-dir "$RUN/masks_clean" \
  --output outputs/analysis/r308_clean_finetune_clean_val94_r201.json --expected-count 94

VIS=outputs/visualizations/r308_frozen_vs_clean_finetune_val94
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir "$RUN/cleanValGT" \
  --baseline-dir "$RUN/frozenR260Masks" \
  --improved-dir "$RUN/masks_clean" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" \
  --baseline-label "Frozen R260 best" \
  --improved-label "R260 + clean-panel fine-tune"
echo "[R308] complete $(date -Is)"
