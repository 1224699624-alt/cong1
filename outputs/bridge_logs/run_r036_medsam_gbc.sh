#!/usr/bin/env bash
set -euo pipefail

cd /home/shenzeyu/workspace/YOLO_SAM_generic_src
source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="TSRS_RSNA-Epiphysis"
CLEAN_DATASET="TSRS_RSNA-Epiphysis_clean_test_v2"
RUN_ID="r036_medsam_gbc_adapter"
RUN_DIR="outputs/medsam_finetune/${DATASET}/${RUN_ID}"
CKPT="${RUN_DIR}/best_mask_decoder.pt"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/epiphysis_yolov8l_1024-2/weights/best.pt}"
MEDSAM_CHECKPOINT="${MEDSAM_CHECKPOINT:-checkpoints/medsam_vit_b.pth}"

mkdir -p "${RUN_DIR}" outputs/analysis outputs/bridge_logs

python scripts/train_medsam.py \
  --dataset "${DATASET}" \
  --raw-root data/raw \
  --medsam-checkpoint "${MEDSAM_CHECKPOINT}" \
  --output "${CKPT}" \
  --epochs 36 \
  --lr 1e-5 \
  --prompt-lr 5e-6 \
  --weight-decay 1e-4 \
  --num-workers 8 \
  --box-padding-ratio 0.08 \
  --box-jitter-ratio 0.04 \
  --max-instances 64 \
  --use-gbc \
  --gbc-reduction 8 \
  --train-prompt-encoder \
  --cache-embeddings \
  --cache-dir "outputs/medsam_embedding_cache/${RUN_ID}" \
  --cache-dtype fp16 \
  --val-every 1 \
  --seed 2036

python scripts/infer_yolo_medsam.py \
  --dataset "${DATASET}" \
  --split val \
  --raw-root data/raw \
  --out-root "outputs/medsam_predictions/${RUN_ID}" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --medsam-checkpoint "${MEDSAM_CHECKPOINT}" \
  --finetuned-checkpoint "${CKPT}" \
  --imgsz 1024 \
  --conf 0.20 \
  --iou 0.50 \
  --max-det 128 \
  --box-padding-ratio 0.05

python scripts/evaluate_masks.py \
  --dataset "${DATASET}" \
  --raw-root data/raw \
  --split val \
  --pred-dir "outputs/medsam_predictions/${RUN_ID}/${DATASET}/val/masks" \
  --output "outputs/analysis/${RUN_ID}_val_metrics.json"

python scripts/infer_yolo_medsam.py \
  --dataset "${CLEAN_DATASET}" \
  --split test \
  --raw-root data/raw_variants \
  --out-root "outputs/medsam_predictions/${RUN_ID}_variants" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --medsam-checkpoint "${MEDSAM_CHECKPOINT}" \
  --finetuned-checkpoint "${CKPT}" \
  --imgsz 1024 \
  --conf 0.20 \
  --iou 0.50 \
  --max-det 128 \
  --box-padding-ratio 0.05

python scripts/evaluate_masks.py \
  --dataset "${CLEAN_DATASET}" \
  --raw-root data/raw_variants \
  --split test \
  --pred-dir "outputs/medsam_predictions/${RUN_ID}_variants/${CLEAN_DATASET}/test/masks" \
  --output "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json"

cat "outputs/analysis/${RUN_ID}_val_metrics.json"
cat "outputs/analysis/${RUN_ID}_clean_test_v2_metrics.json"
