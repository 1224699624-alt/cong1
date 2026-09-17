#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
SPLIT="test"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
PRED_ROOT="outputs/ablations"

run_infer_eval() {
  local exp_name="$1"
  shift
  local pred_out_root="${PRED_ROOT}/${exp_name}"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-checkpoint "${SAM_CHECKPOINT}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --save-overlays \
    --out-root "${pred_out_root}" \
    "$@"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${pred_out_root}/${DATASET}/${SPLIT}/masks"
}

EXPERIMENTS=(
  "zero_shot_box_only_conf020_pad005"
  "zero_shot_mask_to_prompt_refine_conf020_pad005"
  "zero_shot_mask_to_prompt_refine_soft"
  "zero_shot_mask_to_prompt_refine_pad003_neg2_k5"
  "zero_shot_mask_to_prompt_refine_pad003_neg2_k9"
  "zero_shot_mask_to_prompt_refine_pad003_neg4_k5"
  "zero_shot_mask_to_prompt_refine_pad003_neg4_k13"
  "zero_shot_mask_to_prompt_refine_pad003_neg8_k9"
  "zero_shot_mask_to_prompt_refine_pad005_neg4_k9"
  "zero_shot_mask_to_prompt_refine_pad008_neg2_k5"
  "zero_shot_mask_to_prompt_refine_nomaskinput_pad003_neg4_k9"
)

python scripts/download_sam_checkpoint.py

# Keep the original zero-shot baseline in the same comparison table.
run_infer_eval "zero_shot_box_only_conf020_pad005"

# Current best observed setting.
run_infer_eval \
  "zero_shot_mask_to_prompt_refine_conf020_pad005" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 4 \
  --refine-negative-dilate-kernel 9

# Softer refinement: fewer negative points and a smaller near-boundary ring.
run_infer_eval \
  "zero_shot_mask_to_prompt_refine_soft" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.05 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad003_neg2_k5" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad003_neg2_k9" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 9

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad003_neg4_k5" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 4 \
  --refine-negative-dilate-kernel 5

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad003_neg4_k13" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 4 \
  --refine-negative-dilate-kernel 13

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad003_neg8_k9" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 8 \
  --refine-negative-dilate-kernel 9

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad005_neg4_k9" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.05 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 4 \
  --refine-negative-dilate-kernel 9

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_pad008_neg2_k5" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.08 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5

run_infer_eval \
  "zero_shot_mask_to_prompt_refine_nomaskinput_pad003_neg4_k9" \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 4 \
  --refine-negative-dilate-kernel 9 \
  --disable-refine-mask-input

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --experiments "${EXPERIMENTS[@]}" \
  --output "${PRED_ROOT}/${DATASET}_${SPLIT}_mask_to_prompt_refine_summary.md"
