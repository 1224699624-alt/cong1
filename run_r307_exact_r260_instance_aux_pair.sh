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

DATA=outputs/nnunet/r306_r260_instance_aux_pair/data
RUN=outputs/nnunet/r307_exact_r260_instance_aux_pair
export nnUNet_raw=$DATA/nnUNet_raw
export nnUNet_preprocessed=$DATA/nnUNet_preprocessed
export nnUNet_results=$RUN/nnUNet_results
export R306_SOURCE_CHECKPOINT=$RUN/source/r260_checkpoint_best.pth
export R306_EXPECTED_SOURCE_SHA256=7d1d8d2916115c2e29696b2a76881da2a7a78f3db91a5f6a9491297369052a6e
LOG=outputs/bridge_logs/r307_exact_r260_instance_aux_pair.log
mkdir -p "$RUN" outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
echo "[R307] start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
test "$(sha256sum "$R306_SOURCE_CHECKPOINT" | cut -d' ' -f1)" = "$R306_EXPECTED_SOURCE_SHA256"

$PY -m py_compile scripts/nnunet_trainers/nnUNetTrainerR306R260InstanceAux.py
for d in Dataset307_TSRSR260BinaryPrior2D Dataset308_TSRSR260InstanceAuxPrior2D; do
  test -s "$nnUNet_preprocessed/$d/nnUNetPlans.json"
  test -s "$nnUNet_preprocessed/$d/splits_final.json"
done
test "$(find "$DATA/../imagesValPrior" -type f -name '*.png' | wc -l)" -eq 192

TRAINER_DST=$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss/nnUNetTrainerR306R260InstanceAux.py')
PY
)
cp scripts/nnunet_trainers/nnUNetTrainerR306R260InstanceAux.py "$TRAINER_DST"
$PY -m py_compile "$TRAINER_DST"

rm -rf \
  "$nnUNet_results/Dataset307_TSRSR260BinaryPrior2D/nnUNetTrainerR307BinaryContinue__nnUNetPlans__2d" \
  "$nnUNet_results/Dataset308_TSRSR260InstanceAuxPrior2D/nnUNetTrainerR307InstanceAuxContinue__nnUNetPlans__2d"
nnUNetv2_train 307 2d 0 -tr nnUNetTrainerR307BinaryContinue
nnUNetv2_train 308 2d 0 -tr nnUNetTrainerR307InstanceAuxContinue

rm -rf "$RUN/pred_binary" "$RUN/pred_instance_aux" "$RUN/masks_binary" "$RUN/masks_instance_aux"
nnUNetv2_predict -i "$DATA/../imagesValPrior" -o "$RUN/pred_binary" -d 307 -c 2d -f 0 \
  -tr nnUNetTrainerR307BinaryContinue -chk checkpoint_best.pth
nnUNetv2_predict -i "$DATA/../imagesValPrior" -o "$RUN/pred_instance_aux" -d 308 -c 2d -f 0 \
  -tr nnUNetTrainerR307InstanceAuxContinue -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_binary" --mask-dir "$RUN/masks_binary" --expected-count 96 --strip-prefix val_
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_instance_aux" --mask-dir "$RUN/masks_instance_aux" --expected-count 96 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$RUN/masks_binary" \
  --output outputs/analysis/r307_exact_r260_binary_continue_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$RUN/masks_instance_aux" \
  --output outputs/analysis/r307_exact_r260_instance_aux_original_val_r201.json --expected-count 96

VIS=outputs/visualizations/r307_exact_r260_original_vs_instance_aux
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir outputs/ablations/r260_recovered_exact/TSRS_RSNA-Epiphysis/val/masks \
  --improved-dir "$RUN/masks_instance_aux" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" \
  --baseline-label "Original R260 best" \
  --improved-label "R260 best + instance seam auxiliary"
echo "[R307] complete $(date -Is)"
