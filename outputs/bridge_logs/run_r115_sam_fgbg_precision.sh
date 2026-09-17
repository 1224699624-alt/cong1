#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PY=/home/shenzeyu/.conda/envs/yolo-sam-gpu/bin/python

DATASET="TSRS_RSNA-Epiphysis"
CLEAN_DATASET="TSRS_RSNA-Epiphysis_clean_test_v2"
RUN_ID="r115_sam_fgbg_precision_adapter"
RUN_DIR="outputs/sam_finetune/${DATASET}/${RUN_ID}"
CKPT="${RUN_DIR}/best_mask_decoder.pt"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"

mkdir -p "${RUN_DIR}" outputs/analysis outputs/bridge_logs

"${PY}" scripts/train_sam.py \
  --dataset "${DATASET}" \
  --raw-root data/raw \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --sam-model-type vit_b \
  --output "${CKPT}" \
  --epochs 32 \
  --lr 1e-5 \
  --prompt-lr 5e-6 \
  --weight-decay 1e-4 \
  --num-workers 8 \
  --box-padding-ratio 0.06 \
  --box-jitter-ratio 0.03 \
  --prompt-mode box+fgbg \
  --num-positive-points 2 \
  --num-negative-points 12 \
  --negative-point-offset-ratio 0.07 \
  --max-instances 64 \
  --use-gbc \
  --gbc-reduction 8 \
  --train-prompt-encoder \
  --prompt-loss-weight 0.35 \
  --prompt-heatmap-sigma 3.0 \
  --prompt-heatmap-loss-weight 0.25 \
  --contrastive-loss-weight 0.08 \
  --contrastive-temperature 0.1 \
  --boundary-kernel-size 5 \
  --cache-embeddings \
  --cache-dir "outputs/sam_embedding_cache/${RUN_ID}" \
  --cache-dtype fp16 \
  --val-every 1 \
  --seed 20260715

"${PY}" scripts/infer_yolo_sam.py \
  --dataset "${DATASET}" \
  --split val \
  --raw-root data/raw \
  --out-root "outputs/sam_predictions/${RUN_ID}" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --finetuned-checkpoint "${CKPT}" \
  --sam-model-type vit_b \
  --imgsz 1024 \
  --conf 0.22 \
  --iou 0.50 \
  --max-det 128 \
  --box-padding-ratio 0.04 \
  --prompt-mode box+fgbg \
  --num-positive-points 2 \
  --num-negative-points 12 \
  --negative-point-offset-ratio 0.07 \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.02 \
  --refine-num-positive-points 2 \
  --refine-num-negative-points 8 \
  --refine-negative-dilate-kernel 7 \
  --refine-iters 2 \
  --refine-prompt-scheme box+point+mask

"${PY}" scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --raw-root data/raw \
  --split val \
  --pred-dir "outputs/sam_predictions/${RUN_ID}/${DATASET}/val/masks" \
  --output "outputs/analysis/${RUN_ID}_val_metrics.json"

"${PY}" scripts/infer_yolo_sam.py \
  --dataset "${CLEAN_DATASET}" \
  --split test \
  --raw-root data/raw_variants \
  --out-root "outputs/sam_predictions/${RUN_ID}_variants" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --finetuned-checkpoint "${CKPT}" \
  --sam-model-type vit_b \
  --imgsz 1024 \
  --conf 0.22 \
  --iou 0.50 \
  --max-det 128 \
  --box-padding-ratio 0.04 \
  --prompt-mode box+fgbg \
  --num-positive-points 2 \
  --num-negative-points 12 \
  --negative-point-offset-ratio 0.07 \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.02 \
  --refine-num-positive-points 2 \
  --refine-num-negative-points 8 \
  --refine-negative-dilate-kernel 7 \
  --refine-iters 2 \
  --refine-prompt-scheme box+point+mask

"${PY}" scripts/evaluate_masks.py \
  --dataset "${CLEAN_DATASET}" \
  --raw-root data/raw_variants \
  --split test \
  --pred-dir "outputs/sam_predictions/${RUN_ID}_variants/${CLEAN_DATASET}/test/masks" \
  --output "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json"

"${PY}" scripts/infer_yolo_sam.py \
  --dataset "${DATASET}" \
  --split test \
  --raw-root data/raw \
  --out-root "outputs/sam_predictions/${RUN_ID}" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --finetuned-checkpoint "${CKPT}" \
  --sam-model-type vit_b \
  --imgsz 1024 \
  --conf 0.22 \
  --iou 0.50 \
  --max-det 128 \
  --box-padding-ratio 0.04 \
  --prompt-mode box+fgbg \
  --num-positive-points 2 \
  --num-negative-points 12 \
  --negative-point-offset-ratio 0.07 \
  --mask-to-prompt-refine \
  --refine-box-padding-ratio 0.02 \
  --refine-num-positive-points 2 \
  --refine-num-negative-points 8 \
  --refine-negative-dilate-kernel 7 \
  --refine-iters 2 \
  --refine-prompt-scheme box+point+mask

"${PY}" scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --raw-root data/raw \
  --split test \
  --pred-dir "outputs/sam_predictions/${RUN_ID}/${DATASET}/test/masks" \
  --output "outputs/analysis/${RUN_ID}_original_test_metrics.json"

cat "outputs/analysis/${RUN_ID}_val_metrics.json"
cat "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json"
cat "outputs/analysis/${RUN_ID}_original_test_metrics.json"
