#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
SPLIT="test"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
PRED_ROOT="outputs/ablations"
EXP_NAME="mugu_gated_sam_to_hqsam"

python scripts/infer_mugu_gated_refine.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --base-sam-backend sam \
  --base-sam-checkpoint checkpoints/sam_vit_b_01ec64.pth \
  --base-sam-model-type vit_b \
  --heavy-sam-backend hq_sam \
  --heavy-sam-checkpoint checkpoints/sam_hq_vit_b.pth \
  --heavy-sam-model-type vit_b \
  --base-refine-box-padding-ratio 0.03 \
  --base-refine-num-positive-points 1 \
  --base-refine-num-negative-points 2 \
  --base-refine-negative-dilate-kernel 5 \
  --base-refine-iters 2 \
  --base-refine-prompt-scheme box+point+mask \
  --heavy-refine-box-padding-ratio 0.03 \
  --heavy-refine-num-positive-points 1 \
  --heavy-refine-num-negative-points 2 \
  --heavy-refine-negative-dilate-kernel 9 \
  --heavy-refine-iters 2 \
  --heavy-refine-prompt-scheme box+point+mask \
  --gating-disagreement-threshold 0.12 \
  --gating-base-score-threshold 0.58 \
  --gating-heavy-score-margin 0.01 \
  --save-overlays \
  --save-gating-overlays \
  --out-root "${PRED_ROOT}/${EXP_NAME}"

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --pred-dir "${PRED_ROOT}/${EXP_NAME}/${DATASET}/${SPLIT}/masks"

