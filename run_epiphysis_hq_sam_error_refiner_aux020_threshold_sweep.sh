#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
PRED_ROOT="outputs/ablations"
ERROR_ROOT="outputs/error_refiner/${DATASET}"

BASELINE_EXP="hq_sam_box_only_conf020_pad005"
REFINED_EXP="hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9"
CHECKPOINT="${ERROR_ROOT}/best_hq_sam_iter2_aux020.pt"
THRESHOLD_JSON="${ERROR_ROOT}/threshold_sweep_val_hq_sam_iter2_aux020_fine.json"
OUTPUT_NAME="hq_sam_error_refiner_iter2_aux020_fine_threshold"
THRESHOLDS="${THRESHOLDS:-0.46,0.48,0.50,0.52,0.54,0.56}"

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --checkpoint "${CHECKPOINT}" \
  --thresholds "${THRESHOLDS}" \
  --output "${THRESHOLD_JSON}"

SELECTED_THRESHOLD="$(python - <<PY
import json
from pathlib import Path
data = json.loads(Path("${THRESHOLD_JSON}").read_text())
print(data["best"]["threshold"])
PY
)"

python scripts/infer_error_refiner.py \
  --dataset "${DATASET}" \
  --split test \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --checkpoint "${CHECKPOINT}" \
  --output-name "${OUTPUT_NAME}" \
  --threshold "${SELECTED_THRESHOLD}"

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split test \
  --pred-dir "${PRED_ROOT}/${OUTPUT_NAME}/${DATASET}/test/masks"

