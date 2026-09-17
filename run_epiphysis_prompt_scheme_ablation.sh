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
  local prompt_scheme="$2"
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
    --mask-to-prompt-refine \
    --refine-box-padding-ratio 0.03 \
    --refine-num-positive-points 1 \
    --refine-num-negative-points 2 \
    --refine-negative-dilate-kernel 5 \
    --refine-iters 2 \
    --refine-prompt-scheme "${prompt_scheme}" \
    --out-root "${pred_out_root}"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${pred_out_root}/${DATASET}/${SPLIT}/masks"
}

python scripts/download_sam_checkpoint.py

run_infer_eval "zero_shot_refine_iter2_box_only" "box"
run_infer_eval "zero_shot_refine_iter2_box_point" "box+point"
run_infer_eval "zero_shot_refine_iter2_box_mask" "box+mask"
run_infer_eval "zero_shot_refine_iter2_box_point_mask" "box+point+mask"

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --experiments \
    "zero_shot_refine_iter2_box_only" \
    "zero_shot_refine_iter2_box_point" \
    "zero_shot_refine_iter2_box_mask" \
    "zero_shot_refine_iter2_box_point_mask" \
  --output "${PRED_ROOT}/${DATASET}_${SPLIT}_prompt_scheme_summary.md"
