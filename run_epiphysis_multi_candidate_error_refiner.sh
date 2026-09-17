#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
PRED_ROOT="outputs/ablations"
ERROR_ROOT="outputs/error_refiner/${DATASET}"

BASELINE_EXP="zero_shot_box_only_conf020_pad005"
REFINED_EXP="zero_shot_mask_to_prompt_refine_pad003_neg2_k5"
REFINED_EXP_2="hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9"
REFINER_EXP="multi_candidate_error_refiner_sam_hqsam"
REFINER_CHECKPOINT="${ERROR_ROOT}/best_multi_candidate_sam_hqsam.pt"
THRESHOLD_JSON="${ERROR_ROOT}/threshold_sweep_val_multi_candidate_sam_hqsam.json"
THRESHOLDS="${THRESHOLDS:-0.40,0.45,0.50,0.55,0.60}"

python scripts/train_error_refiner.py \
  --dataset "${DATASET}" \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --refined-exp-2 "${REFINED_EXP_2}" \
  --img-size 512 \
  --epochs 80 \
  --min-epochs 20 \
  --patience 10 \
  --batch-size 8 \
  --lr 3e-4 \
  --num-workers 8 \
  --base-channels 24 \
  --aux-loss-weight 0.20 \
  --output "${REFINER_CHECKPOINT}"

python scripts/select_error_refiner_threshold.py \
  --dataset "${DATASET}" \
  --split val \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --refined-exp-2 "${REFINED_EXP_2}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
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
  --refined-exp-2 "${REFINED_EXP_2}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
  --output-name "${REFINER_EXP}" \
  --threshold "${SELECTED_THRESHOLD}"

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split test \
  --pred-dir "${PRED_ROOT}/${REFINER_EXP}/${DATASET}/test/masks"

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split test \
  --experiments \
    "${BASELINE_EXP}" \
    "${REFINED_EXP}" \
    "${REFINED_EXP_2}" \
    "${REFINER_EXP}" \
  --output "${PRED_ROOT}/${DATASET}_test_multi_candidate_error_refiner_summary.md"

