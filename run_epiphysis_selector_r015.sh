#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
OUTPUT_EXP="${OUTPUT_EXP:-allstack_anatomy_roi_selector_r015_sam_hqsam}"
RULE_JSON="${RULE_JSON:-outputs/error_refiner/${DATASET}/selector_r015_rule.json}"

BASELINE_EXP="${BASELINE_EXP:-zero_shot_box_only_conf020_pad005}"
REFINED_EXP="${REFINED_EXP:-zero_shot_mask_to_prompt_refine_pad003_neg2_k5}"
REFINED_EXP_2="${REFINED_EXP_2:-hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9}"
V2_EXP="${V2_EXP:-allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam}"
PRECISION_EXP="${PRECISION_EXP:-allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam}"
AIC_EXP="${AIC_EXP:-allstack_anatomy_roi_aic_refiner_sam_hqsam}"
R014_EXP="${R014_EXP:-allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue_sam_hqsam}"

infer_if_missing() {
  local exp="$1"
  local ckpt="$2"
  local target_mode="$3"
  local n
  local mask_dir="outputs/ablations/${exp}/${DATASET}/val/masks"
  if [ -d "${mask_dir}" ]; then
    n=$(find "${mask_dir}" -maxdepth 1 -name '*.png' | wc -l)
  else
    n=0
  fi
  if [ "${n}" -lt 96 ]; then
    python scripts/infer_error_refiner.py \
      --dataset "${DATASET}" \
      --split val \
      --baseline-exp "${BASELINE_EXP}" \
      --refined-exp "${REFINED_EXP}" \
      --refined-exp-2 "${REFINED_EXP_2}" \
      --checkpoint "${ckpt}" \
      --target-mode "${target_mode}" \
      --output-name "${exp}" \
      --threshold 0.50 \
      --post-median-ksize 1 \
      --post-open-kernel 1 \
      --post-min-component 0 \
      --post-min-hole 0
  fi
}

infer_if_missing "${V2_EXP}" "outputs/error_refiner/${DATASET}/best_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_sam_hqsam.pt" "allstack_anatomy_roi_keepbone_cutgap_refiner_v2"
infer_if_missing "${PRECISION_EXP}" "outputs/error_refiner/${DATASET}/best_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_precision_tune_sam_hqsam.pt" "allstack_anatomy_roi_keepbone_cutgap_refiner_v2"
infer_if_missing "${AIC_EXP}" "outputs/error_refiner/${DATASET}/best_allstack_anatomy_roi_aic_refiner_sam_hqsam.pt" "allstack_anatomy_roi_aic_refiner"
infer_if_missing "${R014_EXP}" "outputs/error_refiner/${DATASET}/best_allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue_sam_hqsam.pt" "allstack_anatomy_roi_keepbone_cutgap_refiner_v2_recall_rescue"

python scripts/select_mask_ensemble.py \
  --dataset "${DATASET}" \
  --mode select-apply \
  --output-exp "${OUTPUT_EXP}" \
  --rule-json "${RULE_JSON}" \
  --base-exps "${V2_EXP}" "${PRECISION_EXP}" \
  --add-exps "${AIC_EXP}" "${R014_EXP}" "${REFINED_EXP}" "${REFINED_EXP_2}" \
  --support-exps "${BASELINE_EXP}" "${REFINED_EXP}" "${REFINED_EXP_2}" "${AIC_EXP}" \
  --add-radii 0,1,3 \
  --max-add-components 1000000 \
  --support-mins 0,1 \
  --close-kernels 1

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split test \
  --pred-dir "outputs/ablations/${OUTPUT_EXP}/${DATASET}/test/masks"

python scripts/create_filtered_dataset_variant.py \
  --manifest configs/dataset_filters/tsrs_rsna_epiphysis_clean_test_v2.json \
  --source-raw-root data/raw \
  --target-raw-root data/raw_variants \
  --source-ablations-root outputs/ablations \
  --target-ablations-root outputs/ablations_variants \
  --experiments "${BASELINE_EXP}" "${REFINED_EXP}" "${REFINED_EXP_2}" \
  --overwrite

mkdir -p "outputs/ablations_variants/${OUTPUT_EXP}/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks"
cp "outputs/ablations/${OUTPUT_EXP}/${DATASET}/test/masks/"*.png \
  "outputs/ablations_variants/${OUTPUT_EXP}/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks/"

python scripts/evaluate_masks.py \
  --dataset TSRS_RSNA-Epiphysis_clean_test_v2 \
  --raw-root data/raw_variants \
  --split test \
  --pred-dir "outputs/ablations_variants/${OUTPUT_EXP}/TSRS_RSNA-Epiphysis_clean_test_v2/test/masks"
