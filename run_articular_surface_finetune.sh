#!/usr/bin/env bash
set -euo pipefail

source /home/anaconda3/etc/profile.d/conda.sh
conda activate yolo-sam-gpu

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-runs/detect/outputs/yolo/articular_surface_yolov8l_1024/weights/best.pt}"
SAM_CHECKPOINT="${SAM_CHECKPOINT:-checkpoints/sam_vit_b_01ec64.pth}"
FINETUNED_CHECKPOINT="${FINETUNED_CHECKPOINT:-outputs/sam_finetune/TSRS_RSNA-Articular-Surface/best_mask_decoder.pt}"

python scripts/download_sam_checkpoint.py

python scripts/train_sam.py \
  --dataset TSRS_RSNA-Articular-Surface \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --epochs 50 \
  --lr 1e-5 \
  --num-workers 16 \
  --max-instances 64 \
  --cache-embeddings \
  --output "${FINETUNED_CHECKPOINT}"

python scripts/infer_yolo_sam.py \
  --dataset TSRS_RSNA-Articular-Surface \
  --split test \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --sam-checkpoint "${SAM_CHECKPOINT}" \
  --finetuned-checkpoint "${FINETUNED_CHECKPOINT}" \
  --imgsz 1024 \
  --conf 0.20 \
  --box-padding-ratio 0.05 \
  --save-overlays \
  --out-root outputs/predictions_finetuned

python scripts/evaluate_masks.py \
  --dataset TSRS_RSNA-Articular-Surface \
  --split test \
  --pred-dir outputs/predictions_finetuned/TSRS_RSNA-Articular-Surface/test/masks
