#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

DATASET="TSRS_RSNA-Epiphysis_clean_test_v2"
RAW_ROOT="data/raw_variants"
ABLATIONS_ROOT="outputs/ablations_variants"
PRED_ROOT="${ABLATIONS_ROOT}"
MANIFEST="configs/dataset_filters/tsrs_rsna_epiphysis_clean_test_v2.json"

BASELINE_EXP="${BASELINE_EXP:-zero_shot_box_only_conf020_pad005}"
REFINED_EXP="${REFINED_EXP:-zero_shot_mask_to_prompt_refine_pad003_neg2_k5}"
REFINED_EXP_2="${REFINED_EXP_2:-hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9}"
REFINER_EXP="${REFINER_EXP:-allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue_sam_hqsam}"

python scripts/create_filtered_dataset_variant.py \
  --manifest "${MANIFEST}" \
  --source-raw-root data/raw \
  --target-raw-root "${RAW_ROOT}" \
  --source-ablations-root outputs/ablations \
  --target-ablations-root "${ABLATIONS_ROOT}" \
  --experiments "${BASELINE_EXP}" "${REFINED_EXP}" "${REFINED_EXP_2}" \
  --overwrite

mkdir -p "${PRED_ROOT}/${REFINER_EXP}/${DATASET}/test/masks"
cp outputs/ablations/"${REFINER_EXP}"/TSRS_RSNA-Epiphysis/test/masks/*.png \
  "${PRED_ROOT}/${REFINER_EXP}/${DATASET}/test/masks/"

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --raw-root "${RAW_ROOT}" \
  --split test \
  --pred-dir "${PRED_ROOT}/${REFINER_EXP}/${DATASET}/test/masks"

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split test \
  --ablations-root "${ABLATIONS_ROOT}" \
  --experiments \
    "${BASELINE_EXP}" \
    "${REFINED_EXP}" \
    "${REFINED_EXP_2}" \
    "${REFINER_EXP}" \
  --output "${PRED_ROOT}/${DATASET}_test_clean_test_v2_${REFINER_EXP}_summary.md"
