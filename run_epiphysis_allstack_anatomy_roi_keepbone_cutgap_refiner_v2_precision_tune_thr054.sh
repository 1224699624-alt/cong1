#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

DATASET="TSRS_RSNA-Epiphysis"
PRED_ROOT="outputs/ablations"
ERROR_ROOT="outputs/error_refiner/${DATASET}"

BASELINE_EXP="${BASELINE_EXP:-zero_shot_box_only_conf020_pad005}"
REFINED_EXP="${REFINED_EXP:-zero_shot_mask_to_prompt_refine_pad003_neg2_k5}"
REFINED_EXP_2="${REFINED_EXP_2:-hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9}"

REFINER_EXP="${REFINER_EXP:-allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_thr054_sam_hqsam}"
REFINER_CHECKPOINT="${REFINER_CHECKPOINT:-${ERROR_ROOT}/best_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam.pt}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-0.54}"

python scripts/infer_error_refiner.py \
  --dataset "${DATASET}" \
  --split test \
  --baseline-exp "${BASELINE_EXP}" \
  --refined-exp "${REFINED_EXP}" \
  --refined-exp-2 "${REFINED_EXP_2}" \
  --checkpoint "${REFINER_CHECKPOINT}" \
  --output-name "${REFINER_EXP}" \
  --threshold "${FIXED_THRESHOLD}" \
  --post-median-ksize 1 \
  --post-open-kernel 1 \
  --post-min-component 0 \
  --post-min-hole 0

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split test \
  --pred-dir "${PRED_ROOT}/${REFINER_EXP}/${DATASET}/test/masks"
