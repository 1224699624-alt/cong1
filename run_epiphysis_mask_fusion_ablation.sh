#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

DATASET="TSRS_RSNA-Epiphysis"
SPLIT="test"
PRED_ROOT="outputs/ablations"

run_fusion_eval() {
  local exp_name="$1"
  shift

  python scripts/fuse_candidate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --output-name "${exp_name}" \
    "$@"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${PRED_ROOT}/${exp_name}/${DATASET}/${SPLIT}/masks"
}

COMMON_CANDIDATES=(
  zero_shot_box_only_conf020_pad005
  zero_shot_mask_to_prompt_refine_soft
  zero_shot_mask_to_prompt_refine_pad003_neg2_k5
  zero_shot_mask_to_prompt_refine_pad003_neg8_k9
)

EXPERIMENTS=(
  zero_shot_box_only_conf020_pad005
  zero_shot_mask_to_prompt_refine_pad003_neg2_k5
  zero_shot_mask_fusion_majority4_thr050
  zero_shot_mask_fusion_recall_thr040
  zero_shot_mask_fusion_precision_thr060
  zero_shot_mask_fusion_weighted_balanced_thr050
  zero_shot_mask_fusion_weighted_recall_thr050
)

run_fusion_eval \
  zero_shot_mask_fusion_majority4_thr050 \
  --candidates "${COMMON_CANDIDATES[@]}" \
  --threshold 0.50

run_fusion_eval \
  zero_shot_mask_fusion_recall_thr040 \
  --candidates "${COMMON_CANDIDATES[@]}" \
  --threshold 0.40

run_fusion_eval \
  zero_shot_mask_fusion_precision_thr060 \
  --candidates "${COMMON_CANDIDATES[@]}" \
  --threshold 0.60

run_fusion_eval \
  zero_shot_mask_fusion_weighted_balanced_thr050 \
  --candidates "${COMMON_CANDIDATES[@]}" \
  --weights 1.0 1.0 1.3 1.1 \
  --threshold 0.50

run_fusion_eval \
  zero_shot_mask_fusion_weighted_recall_thr050 \
  --candidates "${COMMON_CANDIDATES[@]}" \
  --weights 1.4 1.2 1.2 0.8 \
  --threshold 0.50

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --experiments "${EXPERIMENTS[@]}" \
  --output "${PRED_ROOT}/${DATASET}_${SPLIT}_mask_fusion_summary.md"
