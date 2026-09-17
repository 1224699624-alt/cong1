#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="TSRS_RSNA-Epiphysis"
SPLIT="test"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
HQ_SAM_CHECKPOINT="${HQ_SAM_CHECKPOINT:-checkpoints/sam_hq_vit_b.pth}"
SAM_MODEL_TYPE="${SAM_MODEL_TYPE:-vit_b}"
PRED_ROOT="outputs/ablations"

run_infer_eval() {
  local exp_name="$1"
  shift
  local pred_out_root="${PRED_ROOT}/${exp_name}"

  python scripts/infer_yolo_sam.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    --sam-backend hq_sam \
    --sam-checkpoint "${HQ_SAM_CHECKPOINT}" \
    --sam-model-type "${SAM_MODEL_TYPE}" \
    --imgsz 1024 \
    --conf 0.20 \
    --box-padding-ratio 0.05 \
    --prompt-mode box \
    --mask-to-prompt-refine \
    --refine-prompt-scheme "box+point+mask" \
    --save-overlays \
    --out-root "${pred_out_root}" \
    "$@"

  python scripts/evaluate_masks.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --pred-dir "${pred_out_root}/${DATASET}/${SPLIT}/masks"
}

EXPERIMENTS=(
  "hq_sam_box_only_conf020_pad005"
  "hq_sam_refine_iter2_boxpointmask_pad003_neg2_k5"
  "hq_sam_refine_iter2_boxpointmask_pad003_neg4_k5"
  "hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9"
  "hq_sam_refine_iter2_boxpointmask_pad005_neg2_k5"
  "hq_sam_refine_iter2_boxpointmask_pad003_pos2_neg2_k5"
  "hq_sam_refine_iter3_boxpointmask_pad003_neg2_k5"
)

run_infer_eval "hq_sam_box_only_conf020_pad005" \
  --prompt-mode box

run_infer_eval "hq_sam_refine_iter2_boxpointmask_pad003_neg2_k5" \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5 \
  --refine-iters 2

run_infer_eval "hq_sam_refine_iter2_boxpointmask_pad003_neg4_k5" \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 4 \
  --refine-negative-dilate-kernel 5 \
  --refine-iters 2

run_infer_eval "hq_sam_refine_iter2_boxpointmask_pad003_neg2_k9" \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 9 \
  --refine-iters 2

run_infer_eval "hq_sam_refine_iter2_boxpointmask_pad005_neg2_k5" \
  --refine-box-padding-ratio 0.05 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5 \
  --refine-iters 2

run_infer_eval "hq_sam_refine_iter2_boxpointmask_pad003_pos2_neg2_k5" \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 2 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5 \
  --refine-iters 2

run_infer_eval "hq_sam_refine_iter3_boxpointmask_pad003_neg2_k5" \
  --refine-box-padding-ratio 0.03 \
  --refine-num-positive-points 1 \
  --refine-num-negative-points 2 \
  --refine-negative-dilate-kernel 5 \
  --refine-iters 3

python scripts/summarize_ablation_metrics.py \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --experiments "${EXPERIMENTS[@]}" \
  --output "${PRED_ROOT}/${DATASET}_${SPLIT}_hq_sam_refine_tuning_summary.md"

