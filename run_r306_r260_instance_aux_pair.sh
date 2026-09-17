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

RUN=outputs/nnunet/r306_r260_instance_aux_pair
DATA=$RUN/data
export nnUNet_raw=$DATA/nnUNet_raw
export nnUNet_preprocessed=$DATA/nnUNet_preprocessed
export nnUNet_results=$RUN/nnUNet_results
export R306_ADAPTED_CHECKPOINT=outputs/nnunet/r260_adapt_sanity.pth
LOG=outputs/bridge_logs/r306_r260_instance_aux_pair.log
mkdir -p "$RUN" outputs/bridge_logs
exec > >(tee -a "$LOG") 2>&1
echo "[R306] start $(date -Is)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

test "$(sha256sum "$R306_ADAPTED_CHECKPOINT" | cut -d' ' -f1)" = "e8a53da3905151dd7ce355961c0505c3e1cfca82d896bb381f04f9db2924879e"
$PY -m py_compile \
  scripts/build_r259_frozen_prior_maps.py \
  scripts/prepare_r306_r260_instance_aux_pair.py \
  scripts/nnunet_trainers/nnUNetTrainerR306R260InstanceAux.py

if [[ ! -s outputs/priors/r259_frozen_relation/manifest.json ]]; then
  echo "[R306] rebuilding exact frozen R259 prior maps"
  $PY scripts/build_r259_frozen_prior_maps.py \
    --checkpoint-dir outputs/pair_prior/r258b_prediction_relation_prior_full_20260715_133745
fi

$PY scripts/prepare_r306_r260_instance_aux_pair.py --overwrite
nnUNetv2_plan_and_preprocess -d 307 -c 2d --verify_dataset_integrity
# Dataset308 deliberately stores auxiliary seam pixels as value 2 while its
# main task remains binary, so custom validation in the preparation script is
# used instead of nnU-Net's semantic-label integrity check.
nnUNetv2_plan_and_preprocess -d 308 -c 2d
cp "$nnUNet_raw/Dataset307_TSRSR260BinaryPrior2D/splits_final.json" \
   "$nnUNet_preprocessed/Dataset307_TSRSR260BinaryPrior2D/splits_final.json"
cp "$nnUNet_raw/Dataset308_TSRSR260InstanceAuxPrior2D/splits_final.json" \
   "$nnUNet_preprocessed/Dataset308_TSRSR260InstanceAuxPrior2D/splits_final.json"

TRAINER_DST=$($PY - <<'PY'
from pathlib import Path
import nnunetv2
print(Path(nnunetv2.__file__).parent / 'training/nnUNetTrainer/variants/loss/nnUNetTrainerR306R260InstanceAux.py')
PY
)
cp scripts/nnunet_trainers/nnUNetTrainerR306R260InstanceAux.py "$TRAINER_DST"
$PY -m py_compile "$TRAINER_DST"

rm -rf \
  "$nnUNet_results/Dataset307_TSRSR260BinaryPrior2D/nnUNetTrainerR306BinaryPrior__nnUNetPlans__2d" \
  "$nnUNet_results/Dataset308_TSRSR260InstanceAuxPrior2D/nnUNetTrainerR306InstanceAuxPrior__nnUNetPlans__2d"
nnUNetv2_train 307 2d 0 -tr nnUNetTrainerR306BinaryPrior
nnUNetv2_train 308 2d 0 -tr nnUNetTrainerR306InstanceAuxPrior

rm -rf "$RUN/pred_binary" "$RUN/pred_instance_aux" "$RUN/masks_binary" "$RUN/masks_instance_aux"
nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/pred_binary" -d 307 -c 2d -f 0 \
  -tr nnUNetTrainerR306BinaryPrior -chk checkpoint_best.pth
nnUNetv2_predict -i "$RUN/imagesValPrior" -o "$RUN/pred_instance_aux" -d 308 -c 2d -f 0 \
  -tr nnUNetTrainerR306InstanceAuxPrior -chk checkpoint_best.pth
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_binary" --mask-dir "$RUN/masks_binary" --expected-count 96 --strip-prefix val_
$PY scripts/convert_nnunet_predictions.py --pred-dir "$RUN/pred_instance_aux" --mask-dir "$RUN/masks_instance_aux" --expected-count 96 --strip-prefix val_

$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$RUN/masks_binary" \
  --output outputs/analysis/r306_r260_binary_original_val_r201.json --expected-count 96
$PY scripts/evaluate_r201_single_split.py \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels --pred-dir "$RUN/masks_instance_aux" \
  --output outputs/analysis/r306_r260_instance_aux_original_val_r201.json --expected-count 96

VIS=outputs/visualizations/r306_r260_binary_vs_instance_aux_original_val
rm -rf "$VIS"
$PY scripts/render_r259_nnunet_prior_comparison.py \
  --image-dir data/raw/TSRS_RSNA-Epiphysis/val \
  --gt-dir data/raw/TSRS_RSNA-Epiphysis/val_labels \
  --baseline-dir "$RUN/masks_binary" \
  --improved-dir "$RUN/masks_instance_aux" \
  --selection-csv outputs/analysis/r255_close_gap_sanity_original_val_pairs.csv \
  --output-dir "$VIS" --num-cases 12 \
  --xray-label "Original X-ray" \
  --baseline-label "R260 binary" \
  --improved-label "R260 + instance seam auxiliary"
echo "[R306] complete $(date -Is)"
